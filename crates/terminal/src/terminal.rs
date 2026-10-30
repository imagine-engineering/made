mod mappings;

mod alacritty;
mod pty_info;
pub mod terminal_settings;

#[cfg(not(windows))]
use anyhow::Context as _;
use anyhow::{Result, bail};
use futures_lite::future::yield_now;
use log::trace;

use futures::{
    FutureExt,
    channel::mpsc::{UnboundedReceiver, unbounded},
};

use alacritty_terminal::grid::Dimensions as _;
use itertools::Itertools as _;
use mappings::mouse::{
    alt_scroll, grid_point, grid_point_and_side, mouse_button_report, mouse_moved_report,
    scroll_report,
};

use async_channel::{Receiver, Sender};
use collections::{HashMap, VecDeque};
use futures::StreamExt;
use pty_info::{ProcessIdGetter, PtyProcessInfo};
use serde::{Deserialize, Serialize};
use settings::Settings;
use task::{HideStrategy, Shell, ShellKind, SpawnInTerminal};
use terminal_settings::{AlternateScroll, CursorShape as SettingsCursorShape, TerminalSettings};
use theme::{ActiveTheme, Theme};
use urlencoding;
use util::{ResultExt as _, paths::PathStyle, truncate_and_trailoff};

#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
use std::{
    borrow::Cow,
    cmp::{self, min},
    fmt::{self, Display, Formatter},
    ops::{BitOr, BitOrAssign, Deref, Range as StdRange},
    path::{Path, PathBuf},
    process::ExitStatus,
    sync::{
        Arc, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};
use thiserror::Error;
use vte::ansi::{Attr, Handler, Processor, StdSyncHandler};
pub use vte::ansi::{Color, NamedColor, Rgb};

use gpui::{
    App, AppContext as _, BackgroundExecutor, Bounds, ClipboardItem, Context, EventEmitter, Hsla,
    Keystroke, Modifiers, MouseButton, MouseDownEvent, MouseMoveEvent, MouseUpEvent, Pixels,
    Point as GpuiPoint, Rgba, ScrollWheelEvent, Size, Task, TouchPhase, Window, actions, black, px,
};

#[cfg(not(windows))]
use crate::alacritty::current_child_signal_mask;
use crate::alacritty::{
    AlacrittyCell, AlacrittyGridIterator, AlacrittyHyperlink, AlacrittySearch, AlacrittyTerm,
    AlacrittyTermConfig, AlacrittyTermLock, HyperlinkMatch, PtySender, RegexSearches,
    append_text_to_term, apply_config, clear_saved_screen, content_text, display_offset,
    display_only_term_config, find_from_terminal_point, full_content_range, last_non_empty_lines,
    make_content, new_term, open_pty, pty_options, pty_term_config, resize, screen_lines,
    scroll_display, scroll_to_point, search_matches, selection_text, set_default_cursor_style,
    set_selection as set_term_selection, shrink_to_used, spawn_event_loop,
    toggle_vi_mode as toggle_term_vi_mode, total_lines, update_selection as update_term_selection,
    update_selection_to_vi_cursor, update_vi_cursor_for_scroll, used_lines, vi_goto_point,
    vi_motion,
};
use crate::mappings::colors::to_vte_rgb;
use crate::mappings::keys::to_esc_str;

/// Process-wide flag set by headless hosts (e.g. the eval CLI) that have no
/// controlling TTY. In such sandboxes PTY allocation and acquiring a
/// controlling terminal fail with `ENOTTY`, so when this is set terminals run
/// their command as a plain subprocess with piped output instead of through a
/// PTY. The normal editor leaves it unset to preserve the interactive PTY
/// experience.
#[derive(Clone, Copy, Default)]
pub struct HeadlessTerminal(pub bool);

impl gpui::Global for HeadlessTerminal {}

impl HeadlessTerminal {
    pub fn is_enabled(cx: &App) -> bool {
        cx.try_global::<Self>().is_some_and(|headless| headless.0)
    }
}

#[derive(Clone, Copy, Debug)]
enum Scroll {
    Delta(i32),
    PageUp,
    PageDown,
    Top,
    Bottom,
}

#[derive(Clone, Copy, Debug)]
enum ViMotion {
    Up,
    Down,
    Left,
    Right,
    First,
    Last,
    FirstOccupied,
    High,
    Middle,
    Low,
    WordLeft,
    WordRight,
    WordRightEnd,
    Bracket,
    ParagraphUp,
    ParagraphDown,
}

#[derive(Clone, Debug)]
pub struct Search {
    search: AlacrittySearch,
}

#[derive(Clone, Debug)]
struct Selection {
    ty: SelectionType,
    start: SelectionAnchor,
    end: SelectionAnchor,
}

#[derive(Clone, Copy, Debug)]
struct SelectionAnchor {
    point: Point,
    side: SelectionSide,
}

#[derive(Clone, Copy, Debug)]
pub enum SelectionMovement {
    Left,
    Right,
    Up,
    Down,
    LineStart,
    LineEnd,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SelectionSide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SelectionType {
    Simple,
    Semantic,
    Lines,
}

impl Selection {
    fn new(selection_type: SelectionType, point: Point, side: SelectionSide) -> Self {
        let anchor = SelectionAnchor { point, side };
        Self {
            ty: selection_type,
            start: anchor,
            end: anchor,
        }
    }

    fn simple_range(range: Range) -> Self {
        let mut selection = Self::new(SelectionType::Simple, range.start(), SelectionSide::Left);
        selection.update(range.end(), SelectionSide::Right);
        selection
    }

    fn update(&mut self, point: Point, side: SelectionSide) {
        self.end = SelectionAnchor { point, side };
    }
}

pub fn is_default_background_color(color: Color) -> bool {
    matches!(color, Color::Named(NamedColor::Background))
}

pub fn is_app_chosen_exact_color(color: Color) -> bool {
    matches!(color, Color::Spec(_) | Color::Indexed(16..=255))
}

pub type AnsiSpans = Vec<(StdRange<usize>, Option<Color>)>;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ParsedAnsiText {
    pub text: String,
    pub foreground_spans: AnsiSpans,
    pub background_spans: AnsiSpans,
}

pub fn parse_ansi_text(input: &[u8]) -> ParsedAnsiText {
    let mut handler = StyledAnsiTextHandler::default();
    let mut processor = Processor::<StdSyncHandler>::default();
    processor.advance(&mut handler, input);
    handler.finish()
}

pub fn strip_ansi_text(input: &[u8]) -> String {
    let mut handler = PlainAnsiTextHandler::default();
    let mut processor = Processor::<StdSyncHandler>::default();
    processor.advance(&mut handler, input);
    handler.text
}

#[derive(Default)]
struct StyledAnsiTextHandler {
    text: String,
    foreground_spans: AnsiSpans,
    background_spans: AnsiSpans,
    current_foreground_range_start: usize,
    current_background_range_start: usize,
    current_foreground_color: Option<Color>,
    current_background_color: Option<Color>,
}

impl StyledAnsiTextHandler {
    fn finish(mut self) -> ParsedAnsiText {
        if self.current_foreground_range_start < self.text.len() {
            self.foreground_spans.push((
                self.current_foreground_range_start..self.text.len(),
                self.current_foreground_color,
            ));
        }

        if self.current_background_range_start < self.text.len() {
            self.background_spans.push((
                self.current_background_range_start..self.text.len(),
                self.current_background_color,
            ));
        }

        ParsedAnsiText {
            text: self.text,
            foreground_spans: self.foreground_spans,
            background_spans: self.background_spans,
        }
    }

    fn break_foreground_span(&mut self, color: Option<Color>) {
        self.foreground_spans.push((
            self.current_foreground_range_start..self.text.len(),
            self.current_foreground_color,
        ));
        self.current_foreground_color = color;
        self.current_foreground_range_start = self.text.len();
    }

    fn break_background_span(&mut self, color: Option<Color>) {
        self.background_spans.push((
            self.current_background_range_start..self.text.len(),
            self.current_background_color,
        ));
        self.current_background_color = color;
        self.current_background_range_start = self.text.len();
    }
}

impl Handler for StyledAnsiTextHandler {
    fn input(&mut self, c: char) {
        self.text.push(c);
    }

    fn linefeed(&mut self) {
        self.text.push('\n');
    }

    fn put_tab(&mut self, count: u16) {
        self.text.extend(std::iter::repeat_n('\t', count as usize));
    }

    fn terminal_attribute(&mut self, attr: Attr) {
        match attr {
            Attr::Foreground(color) => {
                self.break_foreground_span(Some(color));
            }
            Attr::Background(color) => {
                self.break_background_span(Some(color));
            }
            Attr::Reset => {
                self.break_foreground_span(None);
                self.break_background_span(None);
            }
            _ => {}
        }
    }
}

#[derive(Default)]
struct PlainAnsiTextHandler {
    text: String,
    line_start: usize,
}

impl Handler for PlainAnsiTextHandler {
    fn input(&mut self, c: char) {
        self.text.push(c);
    }

    fn linefeed(&mut self) {
        self.text.push('\n');
        self.line_start = self.text.len();
    }

    fn carriage_return(&mut self) {
        self.text.truncate(self.line_start);
    }

    fn put_tab(&mut self, count: u16) {
        self.text.extend(std::iter::repeat_n('\t', count as usize));
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Hyperlink {
    data: HyperlinkData,
}

#[derive(Debug, Clone, Eq, PartialEq)]
enum HyperlinkData {
    Alacritty(AlacrittyHyperlink),
    Owned { id: Option<Arc<str>>, uri: Arc<str> },
}

#[derive(Default, Debug, Clone, Eq, PartialEq)]
pub struct Cell {
    cell: AlacrittyCell,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalSnapshot {
    version: u8,
    columns: usize,
    screen_lines: usize,
    rows: Vec<Vec<AlacrittyCell>>,
    cursor_line: usize,
    cursor_column: usize,
    #[serde(default)]
    cursor_needs_wrap: bool,
    #[serde(default)]
    cursor_template: AlacrittyCell,
    display_offset: usize,
    colors: Vec<Option<Rgb>>,
}

impl TerminalSnapshot {
    pub fn validate(&self) -> Result<()> {
        alacritty::validate_snapshot(self)
    }
}

pub struct RenderableCells<'a> {
    cells: AlacrittyGridIterator<'a>,
}

#[derive(Debug, Clone)]
pub struct IndexedCell {
    pub point: Point,
    pub cell: Cell,
}

impl Deref for IndexedCell {
    type Target = Cell;

    #[inline]
    fn deref(&self) -> &Cell {
        &self.cell
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Modes(u32);

impl Modes {
    pub const NONE: Self = Self(0);
    pub const APP_CURSOR: Self = Self(1 << 0);
    pub const APP_KEYPAD: Self = Self(1 << 1);
    pub const SHOW_CURSOR: Self = Self(1 << 2);
    pub const LINE_WRAP: Self = Self(1 << 3);
    pub const ORIGIN: Self = Self(1 << 4);
    pub const INSERT: Self = Self(1 << 5);
    pub const LINE_FEED_NEW_LINE: Self = Self(1 << 6);
    pub const FOCUS_IN_OUT: Self = Self(1 << 7);
    pub const ALTERNATE_SCROLL: Self = Self(1 << 8);
    pub const BRACKETED_PASTE: Self = Self(1 << 9);
    pub const SGR_MOUSE: Self = Self(1 << 10);
    pub const UTF8_MOUSE: Self = Self(1 << 11);
    pub const ALT_SCREEN: Self = Self(1 << 12);
    pub const MOUSE_REPORT_CLICK: Self = Self(1 << 13);
    pub const MOUSE_DRAG: Self = Self(1 << 14);
    pub const MOUSE_MOTION: Self = Self(1 << 15);
    pub const VI: Self = Self(1 << 16);
    pub const MOUSE_MODE: Self =
        Self(Self::MOUSE_REPORT_CLICK.0 | Self::MOUSE_DRAG.0 | Self::MOUSE_MOTION.0);

    pub const fn empty() -> Self {
        Self::NONE
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    pub fn insert(&mut self, other: Self) {
        self.0 |= other.0;
    }

    pub fn remove(&mut self, other: Self) {
        self.0 &= !other.0;
    }
}

impl BitOr for Modes {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for Modes {
    fn bitor_assign(&mut self, rhs: Self) {
        self.insert(rhs);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Cursor {
    pub shape: CursorShape,
    pub point: Point,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CursorShape {
    Block,
    Underline,
    Bar,
    HollowBlock,
    Hidden,
}

impl From<SettingsCursorShape> for CursorShape {
    fn from(shape: SettingsCursorShape) -> Self {
        match shape {
            SettingsCursorShape::Block => Self::Block,
            SettingsCursorShape::Underline => Self::Underline,
            SettingsCursorShape::Bar => Self::Bar,
            SettingsCursorShape::Hollow => Self::HollowBlock,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Point {
    pub line: i32,
    pub column: usize,
}

impl Point {
    pub fn new(line: i32, column: usize) -> Self {
        Self { line, column }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct Range {
    start: Point,
    end: Point,
}

impl Range {
    pub fn new(start: Point, end: Point) -> Self {
        Self { start, end }
    }

    pub fn start(&self) -> Point {
        self.start
    }

    pub fn end(&self) -> Point {
        self.end
    }

    pub fn contains(&self, point: Point) -> bool {
        self.start <= point && point <= self.end
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SelectionRange {
    pub start: Point,
    pub end: Point,
    pub is_block: bool,
}

impl SelectionRange {
    pub fn point_range(self) -> Range {
        Range::new(self.start, self.end)
    }
}

// TODO: Un-pub
#[derive(Clone)]
pub struct Content {
    pub cells: Vec<IndexedCell>,
    pub mode: Modes,
    pub display_offset: usize,
    pub selection_text: Option<String>,
    pub selection: Option<SelectionRange>,
    pub cursor: Cursor,
    pub cursor_char: char,
    pub terminal_bounds: TerminalBounds,
    pub last_hovered_word: Option<HoveredWord>,
    pub scrolled_to_top: bool,
    pub scrolled_to_bottom: bool,
    pub bottom_row_occupied: bool,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct HoveredWord {
    pub word: String,
    pub word_match: Range,
    pub id: usize,
}

impl Default for Content {
    fn default() -> Self {
        Content {
            cells: Default::default(),
            mode: Default::default(),
            display_offset: Default::default(),
            selection_text: Default::default(),
            selection: Default::default(),
            cursor: Cursor {
                shape: CursorShape::Block,
                point: Point::new(0, 0),
            },
            cursor_char: Default::default(),
            terminal_bounds: Default::default(),
            last_hovered_word: None,
            scrolled_to_top: false,
            scrolled_to_bottom: false,
            bottom_row_occupied: false,
        }
    }
}

#[derive(PartialEq, Eq)]
enum SelectionPhase {
    Selecting,
    Ended,
}

#[cfg(test)]
mod domain_tests {
    use super::*;

    #[test]
    fn strip_ansi_text_removes_ansi_and_handles_carriage_returns() {
        let cases = [
            ("no escape codes here\n", "no escape codes here\n"),
            ("\x1b[31mhello\x1b[0m", "hello"),
            ("\x1b[1;32mfoo\x1b[0m bar", "foo bar"),
            ("progress 10%\rprogress 100%\n", "progress 100%\n"),
        ];

        for (input, expected) in cases {
            assert_eq!(strip_ansi_text(input.as_bytes()), expected);
        }
    }

    #[test]
    fn parse_ansi_text_records_foreground_and_background_spans() {
        let parsed = parse_ansi_text(b"\x1b[31mred\x1b[44mblue-bg\x1b[0mplain");

        assert_eq!(parsed.text, "redblue-bgplain");
        assert_eq!(
            parsed.foreground_spans,
            vec![
                (0..0, None),
                (0..10, Some(Color::Named(NamedColor::Red))),
                (10..15, None),
            ]
        );
        assert_eq!(
            parsed.background_spans,
            vec![
                (0..3, None),
                (3..10, Some(Color::Named(NamedColor::Blue))),
                (10..15, None),
            ]
        );
    }

    #[test]
    fn terminal_cell_clone_shares_extra_storage() {
        let mut cell = Cell::default();
        cell.push_zerowidth('a');

        let clone = cell.clone();

        match (&cell.cell.extra, &clone.cell.extra) {
            (Some(extra), Some(clone_extra)) => assert!(Arc::ptr_eq(extra, clone_extra)),
            _ => panic!("expected extra storage on both cells"),
        }
    }
}

actions!(
    terminal,
    [
        /// Clears the terminal screen.
        Clear,
        /// Copies selected text to the clipboard.
        Copy,
        /// Pastes from the clipboard.
        Paste,
        /// Pastes the text from the clipboard.
        PasteText,
        /// Shows the character palette for special characters.
        ShowCharacterPalette,
        /// Searches for text in the terminal.
        SearchTest,
        /// Scrolls up by one line.
        ScrollLineUp,
        /// Scrolls down by one line.
        ScrollLineDown,
        /// Scrolls up by one page.
        ScrollPageUp,
        /// Scrolls down by one page.
        ScrollPageDown,
        /// Scrolls up by half a page.
        ScrollHalfPageUp,
        /// Scrolls down by half a page.
        ScrollHalfPageDown,
        /// Scrolls to the top of the terminal buffer.
        ScrollToTop,
        /// Scrolls to the bottom of the terminal buffer.
        ScrollToBottom,
        /// Toggles vi mode in the terminal.
        ToggleViMode,
        /// Selects all text in the terminal.
        SelectAll,
    ]
);

const DEBUG_TERMINAL_WIDTH: Pixels = px(500.);
const DEBUG_TERMINAL_HEIGHT: Pixels = px(30.);
const DEBUG_CELL_WIDTH: Pixels = px(5.);
const DEBUG_LINE_HEIGHT: Pixels = px(5.);

/// Inserts Zed-specific environment variables for terminal sessions.
/// Used by both local terminals and remote terminals (via SSH).
pub fn insert_zed_terminal_env(
    env: &mut HashMap<String, String>,
    version: &impl std::fmt::Display,
) {
    env.insert("ZED_TERM".to_string(), "true".to_string());
    env.insert("TERM_PROGRAM".to_string(), "zed".to_string());
    env.insert("TERM".to_string(), "xterm-256color".to_string());
    env.insert("COLORTERM".to_string(), "truecolor".to_string());
    env.insert("TERM_PROGRAM_VERSION".to_string(), version.to_string());
}

pub fn validate_native_terminal_environment(env: &HashMap<String, String>) -> Result<()> {
    validate_native_terminal_environment_entries(
        env.iter()
            .map(|(name, value)| (name.as_str(), value.as_str())),
    )
}

fn validate_native_terminal_environment_entries<'a>(
    entries: impl IntoIterator<Item = (&'a str, &'a str)>,
) -> Result<()> {
    let mut total_bytes = 0_usize;
    let mut entry_count = 0_usize;
    for (name, value) in entries {
        entry_count += 1;
        anyhow::ensure!(
            entry_count <= 512,
            "Native Task environment has too many entries"
        );
        anyhow::ensure!(
            !name.is_empty()
                && name.len() <= 4096
                && !name.contains(['=', '\0'])
                && value.len() <= 4096
                && !value.contains('\0'),
            "Native Task environment contains an unsupported entry"
        );
        total_bytes = total_bytes
            .checked_add(name.len())
            .and_then(|total| total.checked_add(value.len()))
            .ok_or_else(|| anyhow::anyhow!("Native Task environment exceeds its byte budget"))?;
        anyhow::ensure!(
            total_bytes <= 64 * 1024,
            "Native Task environment exceeds its byte budget"
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub struct NativeTerminalEnvironment {
    environment: HashMap<String, String>,
    window_id: u64,
    uses_pty: bool,
}

#[cfg(target_os = "macos")]
impl NativeTerminalEnvironment {
    pub fn for_spawn(
        environment: HashMap<String, String>,
        window_id: u64,
        uses_pty: bool,
    ) -> Result<Self> {
        use util::macos_task_completion::{
            MAX_TASK_BUILD_INVOCATION_BYTES, TASK_BUILD_INVOCATION_ENV, TASK_COMPLETION_TOKEN_ENV,
            TASK_NATIVE_INVOCATION_ENV,
        };
        let is_transport = |name: &str| {
            matches!(
                name,
                TASK_BUILD_INVOCATION_ENV | TASK_COMPLETION_TOKEN_ENV | TASK_NATIVE_INVOCATION_ENV
            )
        };
        validate_native_terminal_environment_entries(
            environment
                .iter()
                .filter(|(name, _)| !is_transport(name))
                .map(|(name, value)| (name.as_str(), value.as_str())),
        )?;
        anyhow::ensure!(
            environment.len() <= 514,
            "Native Task environment has too many transport entries"
        );
        anyhow::ensure!(
            !(environment.contains_key(TASK_BUILD_INVOCATION_ENV)
                && environment.contains_key(TASK_NATIVE_INVOCATION_ENV)),
            "Native Task has conflicting invocation envelopes"
        );
        for name in [TASK_BUILD_INVOCATION_ENV, TASK_NATIVE_INVOCATION_ENV] {
            if let Some(value) = environment.get(name) {
                anyhow::ensure!(
                    !value.is_empty()
                        && value.len() <= MAX_TASK_BUILD_INVOCATION_BYTES
                        && !value.contains('\0'),
                    "Native Task invocation envelope exceeds its bound"
                );
            }
        }
        if let Some(token) = environment.get(TASK_COMPLETION_TOKEN_ENV) {
            anyhow::ensure!(
                token.len() == 64 && token.bytes().all(|byte| byte.is_ascii_hexdigit()),
                "Native Task has an invalid completion token"
            );
        }
        Ok(Self {
            environment,
            window_id,
            uses_pty,
        })
    }

    fn validate_backend(
        &self,
        window_id: u64,
        uses_pty: bool,
        version: &impl std::fmt::Display,
    ) -> Result<()> {
        anyhow::ensure!(
            self.window_id == window_id && self.uses_pty == uses_pty,
            "Native Task terminal environment backend changed"
        );
        let expected = [
            ("ZED_TERM", "true"),
            ("TERM_PROGRAM", "zed"),
            ("TERM", "xterm-256color"),
            ("COLORTERM", "truecolor"),
        ];
        for (name, value) in expected {
            anyhow::ensure!(
                self.environment.get(name).map(String::as_str) == Some(value),
                "Native Task terminal environment is incomplete"
            );
        }
        anyhow::ensure!(
            self.environment.get("TERM_PROGRAM_VERSION") == Some(&version.to_string())
                && !self.environment.contains_key("SHLVL"),
            "Native Task terminal environment stamp changed"
        );
        if uses_pty {
            let window = window_id.to_string();
            anyhow::ensure!(
                self.environment
                    .get("USER")
                    .is_some_and(|value| !value.is_empty())
                    && self
                        .environment
                        .get("HOME")
                        .is_some_and(|value| !value.is_empty())
                    && self.environment.get("ALACRITTY_WINDOW_ID") == Some(&window)
                    && self.environment.get("WINDOWID") == Some(&window)
                    && !self.environment.contains_key("XDG_ACTIVATION_TOKEN")
                    && !self.environment.contains_key("DESKTOP_STARTUP_ID"),
                "Native Task PTY environment is incomplete"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum TerminalEnvironmentMode {
    Inherited,
    #[cfg(target_os = "macos")]
    ExactNative,
}

impl TerminalEnvironmentMode {
    fn clears_environment(self) -> bool {
        match self {
            Self::Inherited => false,
            #[cfg(target_os = "macos")]
            Self::ExactNative => true,
        }
    }
}

pub fn complete_native_terminal_environment(
    overrides: &HashMap<String, String>,
    window_id: u64,
    cx: &App,
) -> Result<HashMap<String, String>> {
    anyhow::ensure!(
        cfg!(target_os = "macos"),
        "Checked native Task environments currently require macOS"
    );
    validate_native_terminal_environment(overrides)?;
    let mut inherited = HashMap::default();
    let mut total_bytes = 0_usize;
    for (name, value) in std::env::vars_os() {
        let name = name
            .into_string()
            .map_err(|_| anyhow::anyhow!("Native Task host environment contains non-UTF8 names"))?;
        let value = value.into_string().map_err(|_| {
            anyhow::anyhow!("Native Task host environment contains non-UTF8 values")
        })?;
        anyhow::ensure!(
            inherited.len() < 512 && name.len() <= 4096 && value.len() <= 4096,
            "Native Task host environment exceeds its entry budget"
        );
        total_bytes = total_bytes
            .checked_add(name.len())
            .and_then(|total| total.checked_add(value.len()))
            .ok_or_else(|| {
                anyhow::anyhow!("Native Task host environment exceeds its byte budget")
            })?;
        anyhow::ensure!(
            total_bytes <= 64 * 1024,
            "Native Task host environment exceeds its byte budget"
        );
        inherited.insert(name, value);
    }
    let parent_has_locale = std::env::var("LANG").is_ok();
    complete_native_terminal_environment_with_host(
        inherited,
        overrides,
        &release_channel::AppVersion::global(cx),
        window_id,
        !HeadlessTerminal::is_enabled(cx),
        parent_has_locale,
    )
}

fn complete_native_terminal_environment_with_host(
    mut inherited: HashMap<String, String>,
    overrides: &HashMap<String, String>,
    version: &impl std::fmt::Display,
    window_id: u64,
    uses_pty: bool,
    parent_has_locale: bool,
) -> Result<HashMap<String, String>> {
    validate_native_terminal_environment(&inherited)?;
    validate_native_terminal_environment(overrides)?;
    if uses_pty {
        // The pinned PTY owner applies passwd fallbacks before overrides. A checked
        // recipe cannot claim those values without a shared fallback snapshot.
        anyhow::ensure!(
            inherited.get("USER").is_some_and(|value| !value.is_empty())
                && inherited.get("HOME").is_some_and(|value| !value.is_empty()),
            "Native Task PTY environment requires explicit host USER and HOME"
        );
        inherited.insert("ALACRITTY_WINDOW_ID".into(), window_id.to_string());
        inherited.insert("WINDOWID".into(), window_id.to_string());
    }
    inherited.extend(
        overrides
            .iter()
            .map(|(name, value)| (name.clone(), value.clone())),
    );
    apply_terminal_runtime_environment(&mut inherited, version, parent_has_locale);
    if uses_pty {
        inherited.remove("XDG_ACTIVATION_TOKEN");
        inherited.remove("DESKTOP_STARTUP_ID");
    }
    inherited.remove("ZED_PROFILE_TASK_TOKEN");
    inherited.remove("ZED_PROFILE_BUILD_INVOCATION_ENV");
    inherited.remove("ZED_PROFILE_NATIVE_INVOCATION_ENV");
    validate_native_terminal_environment(&inherited)?;
    Ok(inherited)
}

fn apply_terminal_runtime_environment(
    env: &mut HashMap<String, String>,
    version: &impl std::fmt::Display,
    parent_has_locale: bool,
) {
    env.remove("SHLVL");
    if !parent_has_locale {
        env.entry("LANG".into())
            .or_insert_with(|| "en_US.UTF-8".into());
    }
    insert_zed_terminal_env(env, version);
}

#[cfg(windows)]
pub fn apply_task_runtime_environment(env: &mut HashMap<String, String>, cx: &App) -> Result<()> {
    let mut overrides = HashMap::default();
    apply_terminal_runtime_environment(
        &mut overrides,
        &release_channel::AppVersion::global(cx),
        std::env::var("LANG").is_ok() || env.keys().any(|name| name.eq_ignore_ascii_case("LANG")),
    );
    env.retain(|name, _| !name.eq_ignore_ascii_case("SHLVL"));
    *env = util::shell_env::merge_task_environment_layers(&[env, &overrides])?;
    Ok(())
}

///Upward flowing events, for changing the title and such
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    TitleChanged,
    BreadcrumbsChanged,
    CloseTerminal,
    Bell,
    Wakeup,
    BlinkChanged(bool),
    SelectionsChanged,
    NewNavigationTarget(Option<MaybeNavigationTarget>),
    Open(MaybeNavigationTarget),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathLikeTarget {
    /// File system path, absolute or relative, existing or not.
    /// Might have line and column number(s) attached as `file.rs:1:23`
    pub maybe_path: String,
    /// Current working directory of the terminal
    pub working_directory: Option<PathBuf>,
}

/// A string inside terminal, potentially useful as a URI that can be opened.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MaybeNavigationTarget {
    /// HTTP, git, etc. string determined by the `URL_REGEX` regex.
    Url(String),
    /// File system path, absolute or relative, existing or not.
    /// Might have line and column number(s) attached as `file.rs:1:23`
    PathLike(PathLikeTarget),
}

#[derive(Clone)]
enum InternalEvent {
    Resize(TerminalBounds),
    Clear,
    // FocusNextMatch,
    Scroll(Scroll),
    ScrollToPoint(Point),
    SetSelection(Option<Selection>),
    UpdateSelection(GpuiPoint<Pixels>),
    MoveSelection(SelectionMovement, bool),
    FindHyperlink(GpuiPoint<Pixels>, bool),
    ProcessHyperlink(HyperlinkMatch, bool),
    // Whether keep selection when copy
    Copy(Option<bool>),
    // Vi mode events
    ToggleViMode,
    ViMotion(ViMotion),
    MoveViCursorToPoint(Point),
}

type ClipboardFormatter = Arc<dyn Fn(&str) -> String + Sync + Send + 'static>;
type ColorFormatter = Arc<dyn Fn(Rgb) -> String + Sync + Send + 'static>;
type TextAreaSizeFormatter = Arc<dyn Fn(TerminalBounds) -> String + Sync + Send + 'static>;

#[derive(Clone)]
pub(crate) enum TerminalBackendEvent {
    MouseCursorDirty,
    Title(String),
    ResetTitle,
    ClipboardStore(String),
    ClipboardLoad(ClipboardFormatter),
    ColorRequest(usize, ColorFormatter),
    PtyWrite(String),
    TextAreaSizeRequest(TextAreaSizeFormatter),
    CursorBlinkingChange,
    Wakeup,
    Bell,
    Exit,
    ChildExit(ExitStatus),
}

impl fmt::Debug for TerminalBackendEvent {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::MouseCursorDirty => f.write_str("MouseCursorDirty"),
            Self::Title(title) => write!(f, "Title({title})"),
            Self::ResetTitle => f.write_str("ResetTitle"),
            Self::ClipboardStore(data) => write!(f, "ClipboardStore({data})"),
            Self::ClipboardLoad(_) => f.write_str("ClipboardLoad"),
            Self::ColorRequest(index, _) => write!(f, "ColorRequest({index})"),
            Self::PtyWrite(output) => write!(f, "PtyWrite({output})"),
            Self::TextAreaSizeRequest(_) => f.write_str("TextAreaSizeRequest"),
            Self::CursorBlinkingChange => f.write_str("CursorBlinkingChange"),
            Self::Wakeup => f.write_str("Wakeup"),
            Self::Bell => f.write_str("Bell"),
            Self::Exit => f.write_str("Exit"),
            Self::ChildExit(status) => write!(f, "ChildExit({status})"),
        }
    }
}

enum PtyEvent {
    Event(TerminalBackendEvent),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalBounds {
    pub cell_width: Pixels,
    pub line_height: Pixels,
    pub bounds: Bounds<Pixels>,
}

impl TerminalBounds {
    pub fn new(line_height: Pixels, cell_width: Pixels, bounds: Bounds<Pixels>) -> Self {
        TerminalBounds {
            cell_width,
            line_height,
            bounds,
        }
    }

    pub fn num_lines(&self) -> usize {
        // Tolerance to prevent f32 precision from losing a row:
        // `N * line_height / line_height` can be N-epsilon, which floor()
        // would round down, pushing the first line into invisible scrollback.
        let raw = self.bounds.size.height / self.line_height;
        raw.next_up().floor() as usize
    }

    pub fn num_columns(&self) -> usize {
        let raw = self.bounds.size.width / self.cell_width;
        raw.next_up().floor() as usize
    }

    pub fn height(&self) -> Pixels {
        self.bounds.size.height
    }

    pub fn width(&self) -> Pixels {
        self.bounds.size.width
    }

    pub fn cell_width(&self) -> Pixels {
        self.cell_width
    }

    pub fn line_height(&self) -> Pixels {
        self.line_height
    }
}

impl Default for TerminalBounds {
    fn default() -> Self {
        TerminalBounds::new(
            DEBUG_LINE_HEIGHT,
            DEBUG_CELL_WIDTH,
            Bounds {
                origin: GpuiPoint::default(),
                size: Size {
                    width: DEBUG_TERMINAL_WIDTH,
                    height: DEBUG_TERMINAL_HEIGHT,
                },
            },
        )
    }
}

fn normalize_terminal_bounds(mut bounds: TerminalBounds) -> TerminalBounds {
    bounds.bounds.size.height = cmp::max(bounds.line_height, bounds.height());
    bounds.bounds.size.width = cmp::max(bounds.cell_width, bounds.width());
    bounds
}

#[derive(Error, Debug)]
pub struct TerminalError {
    pub directory: Option<PathBuf>,
    pub program: Option<String>,
    pub args: Option<Vec<String>>,
    pub title_override: Option<String>,
    pub source: std::io::Error,
}

impl TerminalError {
    fn fmt_directory(&self) -> String {
        self.directory
            .clone()
            .map(|path| {
                match path
                    .into_os_string()
                    .into_string()
                    .map_err(|os_str| format!("<non-utf8 path> {}", os_str.to_string_lossy()))
                {
                    Ok(s) => s,
                    Err(s) => s,
                }
            })
            .unwrap_or_else(|| "<none specified>".to_string())
    }

    fn fmt_shell(&self) -> String {
        if let Some(title_override) = &self.title_override {
            format!(
                "{} {} ({})",
                self.program.as_deref().unwrap_or("<system defined shell>"),
                self.args.as_ref().into_iter().flatten().format(" "),
                title_override
            )
        } else {
            format!(
                "{} {}",
                self.program.as_deref().unwrap_or("<system defined shell>"),
                self.args.as_ref().into_iter().flatten().format(" ")
            )
        }
    }
}

impl Display for TerminalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let dir_string: String = self.fmt_directory();
        let shell = self.fmt_shell();

        write!(
            f,
            "Working directory: {} Shell command: `{}`, IOError: {}",
            dir_string, shell, self.source
        )
    }
}

// https://github.com/alacritty/alacritty/blob/cb3a79dbf6472740daca8440d5166c1d4af5029e/extra/man/alacritty.5.scd?plain=1#L207-L213
const DEFAULT_SCROLL_HISTORY_LINES: usize = 10_000;
pub const MAX_SCROLL_HISTORY_LINES: usize = 100_000;
static NEXT_INIT_COMMAND_STARTUP_MARKER_ID: AtomicU64 = AtomicU64::new(1);

const INIT_COMMAND_STARTUP_MARKER_PREFIX: &str = "__zed_init_command_ready_";
const INIT_COMMAND_STARTUP_MARKER_SUFFIX: &str = "__";
const INIT_COMMAND_STARTUP_MARKER_SEARCH_LINES: usize = 64;

fn init_command_startup_marker(marker_id: u64) -> String {
    format!("{INIT_COMMAND_STARTUP_MARKER_PREFIX}{marker_id}{INIT_COMMAND_STARTUP_MARKER_SUFFIX}")
}

fn init_command_startup_marker_command(shell_kind: ShellKind, marker_id: u64) -> String {
    // Split the marker across the command so its echo can't satisfy the
    // handshake; only the command's output contains the contiguous marker.
    match shell_kind {
        ShellKind::PowerShell | ShellKind::Pwsh => format!(
            "Write-Output ('{INIT_COMMAND_STARTUP_MARKER_PREFIX}' + '{marker_id}' + '{INIT_COMMAND_STARTUP_MARKER_SUFFIX}')"
        ),
        ShellKind::Cmd => {
            format!(
                "<nul set /p zed_init_ready={INIT_COMMAND_STARTUP_MARKER_PREFIX}&echo {marker_id}{INIT_COMMAND_STARTUP_MARKER_SUFFIX}"
            )
        }
        ShellKind::Nushell => {
            format!(
                "print $\"{INIT_COMMAND_STARTUP_MARKER_PREFIX}({marker_id}){INIT_COMMAND_STARTUP_MARKER_SUFFIX}\""
            )
        }
        ShellKind::Posix
        | ShellKind::Csh
        | ShellKind::Tcsh
        | ShellKind::Rc
        | ShellKind::Fish
        | ShellKind::Xonsh
        | ShellKind::Elvish => format!(
            "printf '%s%s%s\\n' {INIT_COMMAND_STARTUP_MARKER_PREFIX} {marker_id} {INIT_COMMAND_STARTUP_MARKER_SUFFIX}"
        ),
    }
}

pub struct TerminalBuilder {
    terminal: Terminal,
    events_rx: UnboundedReceiver<PtyEvent>,
}

impl TerminalBuilder {
    pub fn new_display_only(
        cursor_shape: SettingsCursorShape,
        alternate_scroll: AlternateScroll,
        max_scroll_history_lines: Option<usize>,
        window_id: u64,
        background_executor: &BackgroundExecutor,
        path_style: PathStyle,
    ) -> TerminalBuilder {
        Self::new_display_only_with_bounds(
            cursor_shape,
            alternate_scroll,
            max_scroll_history_lines,
            window_id,
            background_executor,
            path_style,
            TerminalBounds::default(),
        )
    }

    pub fn new_display_only_with_bounds(
        cursor_shape: SettingsCursorShape,
        alternate_scroll: AlternateScroll,
        max_scroll_history_lines: Option<usize>,
        window_id: u64,
        background_executor: &BackgroundExecutor,
        path_style: PathStyle,
        terminal_bounds: TerminalBounds,
    ) -> TerminalBuilder {
        let terminal_bounds = normalize_terminal_bounds(terminal_bounds);

        let scrolling_history = max_scroll_history_lines
            .unwrap_or(DEFAULT_SCROLL_HISTORY_LINES)
            .min(MAX_SCROLL_HISTORY_LINES);
        let config = display_only_term_config(scrolling_history, cursor_shape);

        let (events_tx, events_rx) = unbounded();
        let term = new_term(&config, terminal_bounds, events_tx, alternate_scroll);

        let terminal = Terminal {
            task: None,
            terminal_type: TerminalType::DisplayOnly,
            subprocess: None,
            completion_tx: None,
            task_completion: OnceLock::new(),
            term,
            term_config: config,
            output_processor: Processor::<StdSyncHandler>::new(),
            title_override: None,
            events: VecDeque::with_capacity(10),
            last_content: Content {
                terminal_bounds,
                ..Default::default()
            },
            last_mouse: None,
            mouse_down_position: None,
            matches: Vec::new(),

            selection_head: None,
            breadcrumb_text: String::new(),
            scroll_px: px(0.),
            next_link_id: 0,
            selection_phase: SelectionPhase::Ended,
            hyperlink_regex_searches: RegexSearches::default(),
            vi_mode_enabled: false,
            is_remote_terminal: false,
            last_mouse_move_time: Instant::now(),
            last_hyperlink_search_position: None,
            mouse_down_hyperlink: None,
            #[cfg(windows)]
            shell_program: None,
            activation_script: Vec::new(),
            template: CopyTemplate {
                shell: Shell::System,
                env: HashMap::default(),
                cursor_shape,
                alternate_scroll,
                max_scroll_history_lines,
                path_hyperlink_regexes: Vec::default(),
                path_hyperlink_timeout_ms: 0,
                window_id,
            },
            child_exited: None,
            process_finished: false,
            keyboard_input_sent: false,
            user_stop_requested: false,
            init_command_startup_marker: None,
            init_command_startup_tx: None,
            event_loop_task: Task::ready(Ok(())),
            background_executor: background_executor.clone(),
            path_style,
            cwd_history: Vec::new(),
            pending_cwd_boundary: None,
            #[cfg(any(test, feature = "test-support"))]
            input_log: Vec::new(),
            #[cfg(any(test, feature = "test-support"))]
            pty_write_log: Default::default(),
        };

        TerminalBuilder {
            terminal,
            events_rx,
        }
    }

    pub fn new(
        working_directory: Option<PathBuf>,
        task: Option<TaskState>,
        shell: Shell,
        env: HashMap<String, String>,
        cursor_shape: SettingsCursorShape,
        alternate_scroll: AlternateScroll,
        max_scroll_history_lines: Option<usize>,
        path_hyperlink_regexes: Vec<String>,
        path_hyperlink_timeout_ms: u64,
        is_remote_terminal: bool,
        window_id: u64,
        completion_tx: Option<Sender<Option<ExitStatus>>>,
        cx: &App,
        activation_script: Vec<String>,
        path_style: PathStyle,
    ) -> Task<Result<TerminalBuilder>> {
        Self::new_with_environment_mode(
            working_directory,
            task,
            shell,
            env,
            cursor_shape,
            alternate_scroll,
            max_scroll_history_lines,
            path_hyperlink_regexes,
            path_hyperlink_timeout_ms,
            is_remote_terminal,
            window_id,
            completion_tx,
            cx,
            activation_script,
            path_style,
            TerminalEnvironmentMode::Inherited,
        )
    }

    #[cfg(target_os = "macos")]
    pub fn new_with_exact_native_environment(
        working_directory: PathBuf,
        mut task: TaskState,
        program: String,
        arguments: Vec<String>,
        environment: NativeTerminalEnvironment,
        cursor_shape: SettingsCursorShape,
        alternate_scroll: AlternateScroll,
        max_scroll_history_lines: Option<usize>,
        path_hyperlink_regexes: Vec<String>,
        path_hyperlink_timeout_ms: u64,
        window_id: u64,
        completion_tx: Option<Sender<Option<ExitStatus>>>,
        cx: &App,
        path_style: PathStyle,
    ) -> Task<Result<TerminalBuilder>> {
        let validate = || -> Result<()> {
            for path in [Path::new(&program), working_directory.as_path()] {
                anyhow::ensure!(
                    path.is_absolute()
                        && path.to_str().is_some_and(
                            |value| value.len() <= 4096 && !value.chars().any(char::is_control)
                        ),
                    "Native Task requires bounded absolute program and working directory"
                );
            }
            anyhow::ensure!(arguments.len() <= 512, "Native Task has too many arguments");
            let mut bytes = 0_usize;
            for argument in &arguments {
                anyhow::ensure!(
                    argument.len() <= 4096 && !argument.contains('\0'),
                    "Native Task has an unsupported argument"
                );
                bytes = bytes
                    .checked_add(argument.len())
                    .context("Native Task arguments exceed their byte budget")?;
                anyhow::ensure!(
                    bytes <= 32 * 1024,
                    "Native Task arguments exceed their byte budget"
                );
            }
            environment.validate_backend(
                window_id,
                !HeadlessTerminal::is_enabled(cx),
                &release_channel::AppVersion::global(cx),
            )
        };
        if let Err(error) = validate() {
            return Task::ready(Err(error));
        }
        task.spawned_task.env.clear();
        Self::new_with_environment_mode(
            Some(working_directory),
            Some(task),
            Shell::WithArguments {
                program,
                args: arguments,
                title_override: None,
            },
            environment.environment,
            cursor_shape,
            alternate_scroll,
            max_scroll_history_lines,
            path_hyperlink_regexes,
            path_hyperlink_timeout_ms,
            false,
            window_id,
            completion_tx,
            cx,
            Vec::new(),
            path_style,
            TerminalEnvironmentMode::ExactNative,
        )
    }

    fn new_with_environment_mode(
        working_directory: Option<PathBuf>,
        task: Option<TaskState>,
        shell: Shell,
        mut env: HashMap<String, String>,
        cursor_shape: SettingsCursorShape,
        alternate_scroll: AlternateScroll,
        max_scroll_history_lines: Option<usize>,
        path_hyperlink_regexes: Vec<String>,
        path_hyperlink_timeout_ms: u64,
        is_remote_terminal: bool,
        window_id: u64,
        completion_tx: Option<Sender<Option<ExitStatus>>>,
        cx: &App,
        activation_script: Vec<String>,
        path_style: PathStyle,
        environment_mode: TerminalEnvironmentMode,
    ) -> Task<Result<TerminalBuilder>> {
        let direct_windows_program = task
            .as_ref()
            .is_some_and(|task| task.spawned_task.windows_direct_program);
        if direct_windows_program && (!cfg!(windows) || is_remote_terminal) {
            return Task::ready(Err(anyhow::anyhow!(
                "Literal Windows executable Tasks require a local Windows host"
            )));
        }
        let clear_environment = environment_mode.clears_environment() || direct_windows_program;
        let version = release_channel::AppVersion::global(cx);
        let background_executor = cx.background_executor().clone();
        // Headless hosts (e.g. the eval CLI) have no controlling TTY, so PTY
        // allocation / acquiring a controlling terminal fails with `ENOTTY`.
        // When set, run the command as a plain subprocess instead.
        let no_pty = HeadlessTerminal::is_enabled(cx);
        #[cfg(not(windows))]
        let child_signal_mask = match current_child_signal_mask()
            .context("failed to capture terminal child signal mask")
        {
            Ok(signal_mask) => Some(signal_mask),
            Err(error) => return Task::ready(Err(error)),
        };
        let fut = async move {
            if !clear_environment {
                apply_terminal_runtime_environment(
                    &mut env,
                    &version,
                    std::env::var("LANG").is_ok(),
                );
            }

            let contains_private_invocation = {
                #[cfg(target_os = "macos")]
                {
                    use util::macos_task_completion::{
                        TASK_BUILD_INVOCATION_ENV, TASK_COMPLETION_TOKEN_ENV,
                        TASK_NATIVE_INVOCATION_ENV,
                    };
                    env.contains_key(TASK_BUILD_INVOCATION_ENV)
                        || env.contains_key(TASK_NATIVE_INVOCATION_ENV)
                        || env.contains_key(TASK_COMPLETION_TOKEN_ENV)
                }
                #[cfg(not(target_os = "macos"))]
                {
                    false
                }
            };

            #[derive(Default)]
            struct ShellParams {
                program: String,
                args: Option<Vec<String>>,
                title_override: Option<String>,
            }

            impl ShellParams {
                fn new(
                    program: String,
                    args: Option<Vec<String>>,
                    title_override: Option<String>,
                ) -> Self {
                    log::debug!("Using {program} as shell");
                    Self {
                        program,
                        args,
                        title_override,
                    }
                }
            }

            let shell_params = match shell.clone() {
                Shell::System => {
                    if cfg!(windows) {
                        Some(ShellParams::new(
                            util::shell::get_windows_system_shell(),
                            None,
                            None,
                        ))
                    } else {
                        None
                    }
                }
                Shell::Program(program) => Some(ShellParams::new(program, None, None)),
                Shell::WithArguments {
                    program,
                    args,
                    title_override,
                } => Some(ShellParams::new(program, Some(args), title_override)),
            };
            #[cfg(windows)]
            let mut shell_params = shell_params;
            #[cfg(windows)]
            let application_name = if direct_windows_program {
                let params = shell_params
                    .as_mut()
                    .ok_or_else(|| anyhow::anyhow!("Literal Windows Task has no executable"))?;
                let application = resolve_windows_task_program(
                    &params.program,
                    &env,
                    working_directory.as_deref(),
                )?;
                params.program = application
                    .to_str()
                    .ok_or_else(|| {
                        anyhow::anyhow!("Literal Windows executable path is not valid Unicode")
                    })?
                    .to_owned();
                Some(application)
            } else {
                None
            };
            let terminal_title_override =
                shell_params.as_ref().and_then(|e| e.title_override.clone());

            #[cfg(windows)]
            let shell_program = shell_params.as_ref().map(|params| {
                use util::ResultExt;

                Self::resolve_path(&params.program)
                    .log_err()
                    .unwrap_or(params.program.clone())
            });

            // Note: when remoting, this shell_kind will scrutinize `ssh` or
            // `wsl.exe` as a shell and fall back to posix or powershell based on
            // the compilation target. This is fine right now due to the restricted
            // way we use the return value, but would become incorrect if we
            // supported remoting into windows.
            let shell_kind = shell.shell_kind(cfg!(windows));

            let scrolling_history = if task.is_some() {
                // Tasks like `cargo build --all` may produce a lot of output, ergo allow maximum scrolling.
                // After the task finishes, we do not allow appending to that terminal, so small tasks output should not
                // cause excessive memory usage over time.
                MAX_SCROLL_HISTORY_LINES
            } else {
                max_scroll_history_lines
                    .unwrap_or(DEFAULT_SCROLL_HISTORY_LINES)
                    .min(MAX_SCROLL_HISTORY_LINES)
            };
            let config = pty_term_config(scrolling_history, cursor_shape);

            //Spawn a task so the Alacritty EventLoop (or the subprocess reader) can communicate with us
            //TODO: Remove with a bounded sender which can be dispatched on &self
            let (events_tx, events_rx) = unbounded();
            //Set up the terminal...
            let term = new_term(
                &config,
                TerminalBounds::default(),
                events_tx.clone(),
                alternate_scroll,
            );

            // When `no_pty` is set (headless hosts), run the task as a plain
            // subprocess and pump its piped output into the same emulator the
            // PTY path would feed.
            let (terminal_type, subprocess) = if no_pty {
                let (program, args) = match &shell_params {
                    Some(params) => (
                        params.program.clone(),
                        params.args.clone().unwrap_or_default(),
                    ),
                    None => (util::shell::get_system_shell(), Vec::new()),
                };
                let subprocess = match spawn_task_subprocess(
                    program,
                    args,
                    env.clone(),
                    clear_environment,
                    working_directory.clone(),
                    term.clone(),
                    events_tx,
                    &background_executor,
                ) {
                    Ok(subprocess) => subprocess,
                    Err(error) => {
                        bail!(TerminalError {
                            directory: working_directory,
                            program: shell_params.as_ref().map(|params| params.program.clone()),
                            args: shell_params.as_ref().and_then(|params| params.args.clone()),
                            title_override: terminal_title_override,
                            source: if clear_environment || contains_private_invocation {
                                // The generic spawn context includes Command's private
                                // environment. Retain only the OS cause for these paths.
                                match error.downcast_ref::<std::io::Error>() {
                                    Some(cause) => match cause.raw_os_error() {
                                        Some(code) => std::io::Error::from_raw_os_error(code),
                                        None => std::io::Error::new(
                                            cause.kind(),
                                            "Native Task child could not be spawned",
                                        ),
                                    },
                                    None => std::io::Error::other(
                                        "Native Task child could not be spawned",
                                    ),
                                }
                            } else {
                                std::io::Error::other(format!("{error:#}"))
                            },
                        });
                    }
                };
                (TerminalType::DisplayOnly, Some(subprocess))
            } else {
                let alacritty_shell = shell_params.as_ref().map(|params| {
                    (
                        params.program.clone(),
                        params.args.clone().unwrap_or_default(),
                    )
                });
                let pty_options = pty_options(
                    alacritty_shell,
                    working_directory.clone(),
                    env.clone(),
                    // We pass in the foreground thread's signal mask to the child process via pty_options,
                    // so terminal construction can run on a background thread without breaking Ctrl-C and other signals
                    // otherwise the terminal would inherit the background executor's signal mask which blocks
                    // some terminal signals
                    #[cfg(not(windows))]
                    child_signal_mask,
                    #[cfg(windows)]
                    shell_kind.tty_escape_args(),
                    #[cfg(windows)]
                    application_name,
                );

                //Setup the pty...
                let pty = match open_pty(
                    &pty_options,
                    TerminalBounds::default(),
                    window_id,
                    clear_environment,
                ) {
                    Ok(pty) => pty,
                    Err(error) => {
                        bail!(TerminalError {
                            directory: working_directory,
                            program: shell_params.as_ref().map(|params| params.program.clone()),
                            args: shell_params.as_ref().and_then(|params| params.args.clone()),
                            title_override: terminal_title_override,
                            source: error,
                        });
                    }
                };

                let pty_info = PtyProcessInfo::new(ProcessIdGetter::from(&pty));

                //And connect them together
                let pty_tx =
                    spawn_event_loop(term.clone(), events_tx, pty, pty_options.drain_on_exit)?;

                (
                    TerminalType::Pty {
                        pty_tx,
                        info: Arc::new(pty_info),
                    },
                    None,
                )
            };

            let no_task = task.is_none();
            let terminal = Terminal {
                task,
                terminal_type,
                subprocess,
                completion_tx,
                task_completion: OnceLock::new(),
                term,
                term_config: config,
                output_processor: Processor::<StdSyncHandler>::new(),
                title_override: terminal_title_override,
                events: VecDeque::with_capacity(10), //Should never get this high.
                last_content: Default::default(),
                last_mouse: None,
                mouse_down_position: None,
                matches: Vec::new(),

                selection_head: None,
                breadcrumb_text: String::new(),
                scroll_px: px(0.),
                next_link_id: 0,
                selection_phase: SelectionPhase::Ended,
                hyperlink_regex_searches: RegexSearches::new(
                    &path_hyperlink_regexes,
                    path_hyperlink_timeout_ms,
                ),
                vi_mode_enabled: false,
                is_remote_terminal,
                last_mouse_move_time: Instant::now(),
                last_hyperlink_search_position: None,
                mouse_down_hyperlink: None,
                #[cfg(windows)]
                shell_program,
                activation_script: activation_script.clone(),
                template: CopyTemplate {
                    shell: if clear_environment {
                        Shell::System
                    } else {
                        shell
                    },
                    env: if clear_environment {
                        HashMap::default()
                    } else {
                        env
                    },
                    cursor_shape,
                    alternate_scroll,
                    max_scroll_history_lines,
                    path_hyperlink_regexes,
                    path_hyperlink_timeout_ms,
                    window_id,
                },
                child_exited: None,
                process_finished: false,
                keyboard_input_sent: false,
                user_stop_requested: false,
                init_command_startup_marker: None,
                init_command_startup_tx: None,
                event_loop_task: Task::ready(Ok(())),
                background_executor,
                path_style,
                cwd_history: if is_remote_terminal {
                    Vec::new()
                } else {
                    working_directory
                        .as_ref()
                        .map(|working_directory| {
                            vec![CwdHistoryEntry {
                                scrollback_position: i32::MIN,
                                working_directory: working_directory.clone(),
                            }]
                        })
                        .unwrap_or_default()
                },
                pending_cwd_boundary: None,
                #[cfg(any(test, feature = "test-support"))]
                input_log: Vec::new(),
                #[cfg(any(test, feature = "test-support"))]
                pty_write_log: Default::default(),
            };

            if !activation_script.is_empty() && no_task {
                for activation_script in activation_script {
                    terminal.write_to_pty(activation_script.into_bytes());
                    // Simulate enter key press
                    // NOTE(PowerShell): using `\r\n` will put PowerShell in a continuation mode (infamous >> character)
                    // and generally mess up the rendering.
                    terminal.write_to_pty(b"\x0d");
                }
                // In order to clear the screen at this point, we have two options:
                // 1. We can send a shell-specific command such as "clear" or "cls"
                // 2. We can "echo" a marker message that we will then catch when handling a Wakeup event
                //    and clear the screen using `terminal.clear()` method
                // We cannot issue a `terminal.clear()` command at this point as alacritty is evented
                // and while we have sent the activation script to the pty, it will be executed asynchronously.
                // Therefore, we somehow need to wait for the activation script to finish executing before we
                // can proceed with clearing the screen.
                terminal.write_to_pty(shell_kind.clear_screen_command().as_bytes());
                // Simulate enter key press
                terminal.write_to_pty(b"\x0d");
            }

            Ok(TerminalBuilder {
                terminal,
                events_rx,
            })
        };
        cx.background_spawn(fut)
    }

    pub fn subscribe(mut self, cx: &Context<Terminal>) -> Terminal {
        //Event loop
        self.terminal.event_loop_task = cx.spawn(async move |terminal, cx| {
            while let Some(event) = self.events_rx.next().await {
                terminal.update(cx, |terminal, cx| {
                    //Process the first event immediately for lowered latency
                    terminal.process_pty_event(event, cx);
                })?;

                'outer: loop {
                    let mut events = Vec::new();

                    #[cfg(any(test, feature = "test-support"))]
                    let mut timer = cx.background_executor().simulate_random_delay().fuse();
                    #[cfg(not(any(test, feature = "test-support")))]
                    let mut timer = cx
                        .background_executor()
                        .timer(std::time::Duration::from_millis(4))
                        .fuse();

                    let mut wakeup = false;
                    loop {
                        futures::select_biased! {
                            _ = timer => break,
                            event = self.events_rx.next() => {
                                if let Some(event) = event {
                                    if matches!(event, PtyEvent::Event(TerminalBackendEvent::Wakeup))
                                    {
                                        wakeup = true;
                                    } else {
                                        events.push(event);
                                    }

                                    if events.len() > 100 {
                                        break;
                                    }
                                } else {
                                    break;
                                }
                            },
                        }
                    }

                    if events.is_empty() && !wakeup {
                        yield_now().await;
                        break 'outer;
                    }

                    terminal.update(cx, |this, cx| {
                        if wakeup {
                            this.process_event(TerminalBackendEvent::Wakeup, cx);
                        }

                        for event in events {
                            this.process_pty_event(event, cx);
                        }
                    })?;
                    yield_now().await;
                }
            }
            anyhow::Ok(())
        });
        self.terminal
    }

    #[cfg(windows)]
    fn resolve_path(path: &str) -> Result<String> {
        use windows::Win32::Storage::FileSystem::SearchPathW;
        use windows::core::HSTRING;

        let path = if path.starts_with(r"\\?\") || !path.contains(&['/', '\\']) {
            path.to_string()
        } else {
            r"\\?\".to_string() + path
        };

        let required_length = unsafe { SearchPathW(None, &HSTRING::from(&path), None, None, None) };
        let mut buf = vec![0u16; required_length as usize];
        let size = unsafe { SearchPathW(None, &HSTRING::from(&path), None, Some(&mut buf), None) };

        Ok(String::from_utf16(&buf[..size as usize])?)
    }
}

#[cfg(windows)]
fn resolve_windows_task_program(
    program: &str,
    environment: &HashMap<String, String>,
    working_directory: Option<&Path>,
) -> Result<PathBuf> {
    anyhow::ensure!(
        !program.is_empty() && !program.contains('\0'),
        "Literal Windows Task has an invalid executable"
    );
    let mut search_paths = environment
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("PATH"));
    let search_path = search_paths.next().map(|(_, value)| value);
    anyhow::ensure!(
        search_paths.next().is_none(),
        "Literal Windows Task environment has ambiguous PATH names"
    );
    let mut executable = PathBuf::from(program);
    if executable.extension().is_none() {
        executable.set_extension("exe");
    }
    anyhow::ensure!(
        executable
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("exe")),
        "Literal Windows Tasks require a native .exe; shell scripts need an authored shell Task"
    );
    let directory = match working_directory {
        Some(directory) => directory.to_path_buf(),
        None => std::env::current_dir()?,
    };
    let directory = std::path::absolute(directory)?;
    let candidates = if executable.components().count() > 1 {
        vec![directory.join(executable)]
    } else {
        search_path
            .into_iter()
            .flat_map(std::env::split_paths)
            .map(|path| directory.join(path).join(&executable))
            .collect()
    };
    // PATH extension helpers may consult the editor's PATHEXT and append another
    // suffix to a missing .exe; this lookup must select the exact native file.
    for candidate in candidates {
        anyhow::ensure!(
            candidate.is_absolute(),
            "Literal Windows Task executable requires a fully qualified path"
        );
        match std::fs::metadata(&candidate) {
            Ok(metadata) if metadata.is_file() => return Ok(candidate),
            Ok(_) => continue,
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                ) =>
            {
                continue;
            }
            Err(error) => return Err(error.into()),
        }
    }
    anyhow::bail!("Literal Windows Task executable was not found in its environment")
}

enum TerminalType {
    Pty {
        pty_tx: PtySender,
        info: Arc<PtyProcessInfo>,
    },
    DisplayOnly,
}

pub struct Terminal {
    terminal_type: TerminalType,
    /// Set for non-PTY terminals (see [`HeadlessTerminal`]); owns the spawned
    /// subprocess and the task pumping its output into the grid.
    subprocess: Option<SubprocessHandle>,
    completion_tx: Option<Sender<Option<ExitStatus>>>,
    task_completion:
        OnceLock<futures::future::Shared<futures::future::BoxFuture<'static, Option<ExitStatus>>>>,
    term: Arc<AlacrittyTermLock>,
    term_config: AlacrittyTermConfig,
    output_processor: Processor<StdSyncHandler>,
    events: VecDeque<InternalEvent>,
    /// This is only used for mouse mode cell change detection
    last_mouse: Option<(Point, SelectionSide)>,
    /// Window-relative position of the most recent left mouse-down. Used to
    /// apply a drag threshold before starting a selection (see #58970).
    mouse_down_position: Option<GpuiPoint<Pixels>>,
    pub matches: Vec<Range>,
    pub last_content: Content,
    selection_head: Option<SelectionAnchor>,

    pub breadcrumb_text: String,
    title_override: Option<String>,
    scroll_px: Pixels,
    next_link_id: usize,
    selection_phase: SelectionPhase,
    hyperlink_regex_searches: RegexSearches,
    task: Option<TaskState>,
    vi_mode_enabled: bool,
    is_remote_terminal: bool,
    last_mouse_move_time: Instant,
    last_hyperlink_search_position: Option<GpuiPoint<Pixels>>,
    mouse_down_hyperlink: Option<HyperlinkMatch>,
    #[cfg(windows)]
    shell_program: Option<String>,
    template: CopyTemplate,
    activation_script: Vec<String>,
    child_exited: Option<ExitStatus>,
    process_finished: bool,
    keyboard_input_sent: bool,
    user_stop_requested: bool,
    init_command_startup_marker: Option<String>,
    init_command_startup_tx: Option<Sender<()>>,
    event_loop_task: Task<Result<(), anyhow::Error>>,
    background_executor: BackgroundExecutor,
    path_style: PathStyle,
    cwd_history: Vec<CwdHistoryEntry>,
    pending_cwd_boundary: Option<i32>,
    #[cfg(any(test, feature = "test-support"))]
    input_log: Vec<Vec<u8>>,
    #[cfg(any(test, feature = "test-support"))]
    pty_write_log: std::cell::RefCell<Vec<Vec<u8>>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CwdHistoryEntry {
    /// Line offset in the retained scrollback buffer.
    scrollback_position: i32,
    working_directory: PathBuf,
}

struct CopyTemplate {
    shell: Shell,
    env: HashMap<String, String>,
    cursor_shape: SettingsCursorShape,
    alternate_scroll: AlternateScroll,
    max_scroll_history_lines: Option<usize>,
    path_hyperlink_regexes: Vec<String>,
    path_hyperlink_timeout_ms: u64,
    window_id: u64,
}

#[derive(Debug)]
pub struct TaskState {
    pub status: TaskStatus,
    pub completion_rx: Receiver<Option<ExitStatus>>,
    pub spawned_task: SpawnInTerminal,
    pub structured_task: Option<task::StructuredTaskHandle>,
}

/// A status of the current terminal tab's task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    /// The task had been started, but got cancelled or somehow otherwise it did not
    /// report its exit code before the terminal event loop was shut down.
    Unknown,
    /// The task is started and running currently.
    Running,
    /// After the start, the task stopped running and reported its error code back.
    Completed { success: bool },
}

impl TaskStatus {
    fn register_terminal_exit(&mut self) {
        if self == &Self::Running {
            *self = Self::Unknown;
        }
    }

    fn register_task_exit(&mut self, error_code: i32) {
        *self = TaskStatus::Completed {
            success: error_code == 0,
        };
    }
}

const FIND_HYPERLINK_THROTTLE_PX: Pixels = px(5.0);

/// Minimum pointer movement before a left click begins a selection. This keeps
/// a click that jitters by a pixel or two (such as the window-focusing click)
/// from starting a selection and, with `copy_on_select` enabled, clobbering the
/// clipboard. Mirrors the drag threshold used by gpui's `div` element.
const SELECTION_DRAG_THRESHOLD: f64 = 2.0;

impl Terminal {
    fn process_pty_event(&mut self, event: PtyEvent, cx: &mut Context<Self>) {
        match event {
            PtyEvent::Event(event) => self.process_event(event, cx),
        }
    }

    fn process_event(&mut self, event: TerminalBackendEvent, cx: &mut Context<Self>) {
        match event {
            TerminalBackendEvent::Title(title) => {
                // ignore default shell program title change as windows always sends those events
                // and it would end up showing the shell executable path in breadcrumbs
                #[cfg(windows)]
                if self
                    .shell_program
                    .as_ref()
                    .map(|e| *e == title)
                    .unwrap_or(false)
                {
                    return;
                }

                self.breadcrumb_text = title;
                cx.emit(Event::BreadcrumbsChanged);
            }
            TerminalBackendEvent::ResetTitle => {
                self.breadcrumb_text = String::new();
                cx.emit(Event::BreadcrumbsChanged);
            }
            TerminalBackendEvent::ClipboardStore(data) => {
                cx.write_to_clipboard(ClipboardItem::new_string(data))
            }
            TerminalBackendEvent::ClipboardLoad(format) => {
                self.write_to_pty(
                    match &cx.read_from_clipboard().and_then(|item| item.text()) {
                        // The terminal only supports pasting strings, not images.
                        Some(text) => format(text),
                        _ => format(""),
                    }
                    .into_bytes(),
                )
            }
            TerminalBackendEvent::PtyWrite(out) => self.write_to_pty(out.into_bytes()),
            TerminalBackendEvent::TextAreaSizeRequest(format) => {
                self.write_to_pty(format(self.last_content.terminal_bounds).into_bytes())
            }
            TerminalBackendEvent::CursorBlinkingChange => {
                let terminal = self.term.lock();
                let blinking = terminal.cursor_style().blinking;
                cx.emit(Event::BlinkChanged(blinking));
            }
            TerminalBackendEvent::Bell => {
                cx.emit(Event::Bell);
            }
            TerminalBackendEvent::Exit => self.register_task_finished(None, cx),
            TerminalBackendEvent::MouseCursorDirty => {
                //NOOP, Handled in render
            }
            TerminalBackendEvent::Wakeup => {
                self.detect_init_command_startup_marker();
                cx.emit(Event::Wakeup);

                if let TerminalType::Pty { info, .. } = &self.terminal_type {
                    info.emit_title_changed_if_changed(cx);
                }
            }
            TerminalBackendEvent::ColorRequest(index, format) => {
                // It's important that the color request is processed here to retain relative order
                // with other PTY writes. Otherwise applications might witness out-of-order
                // responses to requests. For example: An application sending `OSC 11 ; ? ST`
                // (color request) followed by `CSI c` (request device attributes) would receive
                // the response to `CSI c` first.
                // Instead of locking, we could store the colors in `self.last_content`. But then
                // we might respond with out of date value if a "set color" sequence is immediately
                // followed by a color request sequence.

                let color = self.term.lock().colors()[index]
                    .unwrap_or_else(|| to_vte_rgb(get_color_at_index(index, cx.theme().as_ref())));
                self.write_to_pty(format(color).into_bytes());
            }
            TerminalBackendEvent::ChildExit(exit_status) => {
                self.register_task_finished(Some(exit_status), cx);
            }
        }
    }

    pub fn selection_started(&self) -> bool {
        self.selection_phase == SelectionPhase::Selecting
    }

    fn process_terminal_event(
        &mut self,
        event: &InternalEvent,
        term: &mut AlacrittyTerm,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        match event {
            &InternalEvent::Resize(new_bounds) => {
                let new_bounds = normalize_terminal_bounds(new_bounds);
                trace!("Resizing: new_bounds={new_bounds:?}");

                let columns_changed =
                    self.last_content.terminal_bounds.num_columns() != new_bounds.num_columns();
                self.last_content.terminal_bounds = new_bounds;

                if let TerminalType::Pty { pty_tx, .. } = &self.terminal_type {
                    pty_tx.resize(new_bounds);
                }

                resize(term, new_bounds);
                if columns_changed {
                    self.reset_cwd_history();
                }
                // If there are matches we need to emit a wake up event to
                // invalidate the matches and recalculate their locations
                // in the new terminal layout
                if !self.matches.is_empty() {
                    cx.emit(Event::Wakeup);
                }
            }
            InternalEvent::Clear => {
                trace!("Clearing");
                clear_saved_screen(term);
                self.reset_cwd_history();
                cx.emit(Event::Wakeup);
            }
            InternalEvent::Scroll(scroll) => {
                trace!("Scrolling: scroll={scroll:?}");
                scroll_display(term, *scroll);
                self.refresh_hovered_word(window);

                if self.vi_mode_enabled {
                    update_vi_cursor_for_scroll(term, *scroll);
                    if let Some(selection_head) = update_selection_to_vi_cursor(term) {
                        #[cfg(any(target_os = "linux", target_os = "freebsd"))]
                        if let Some(selection_text) = selection_text(term) {
                            cx.write_to_primary(ClipboardItem::new_string(selection_text));
                        }

                        self.selection_head = Some(SelectionAnchor {
                            point: selection_head,
                            side: SelectionSide::Right,
                        });
                        cx.emit(Event::SelectionsChanged)
                    }
                }
            }
            InternalEvent::SetSelection(selection) => {
                trace!("Setting selection: selection={selection:?}");
                set_term_selection(term, selection.as_ref());

                #[cfg(any(target_os = "linux", target_os = "freebsd"))]
                if let Some(selection_text) = selection_text(term) {
                    cx.write_to_primary(ClipboardItem::new_string(selection_text));
                }

                self.selection_head = selection.as_ref().map(|selection| selection.end);
                cx.emit(Event::SelectionsChanged)
            }
            InternalEvent::UpdateSelection(position) => {
                trace!("Updating selection: position={position:?}");
                let (point, side) = grid_point_and_side(
                    *position,
                    self.last_content.terminal_bounds,
                    display_offset(term),
                );

                if update_term_selection(term, point, side) {
                    #[cfg(any(target_os = "linux", target_os = "freebsd"))]
                    if let Some(selection_text) = selection_text(term) {
                        cx.write_to_primary(ClipboardItem::new_string(selection_text));
                    }

                    self.selection_head = Some(SelectionAnchor { point, side });
                    cx.emit(Event::SelectionsChanged)
                }
            }

            InternalEvent::MoveSelection(movement, extend) => {
                self.selection_head =
                    alacritty::move_selection(term, self.selection_head, *movement, *extend);
                cx.emit(Event::SelectionsChanged);
            }

            InternalEvent::Copy(keep_selection) => {
                trace!("Copying selection: keep_selection={keep_selection:?}");
                if let Some(txt) = selection_text(term) {
                    cx.write_to_clipboard(ClipboardItem::new_string(txt));
                    if !keep_selection.unwrap_or_else(|| {
                        let settings = TerminalSettings::get_global(cx);
                        settings.keep_selection_on_copy
                    }) {
                        self.events.push_back(InternalEvent::SetSelection(None));
                    }
                }
            }
            InternalEvent::ScrollToPoint(point) => {
                trace!("Scrolling to point: point={point:?}");
                scroll_to_point(term, *point);
                self.refresh_hovered_word(window);
            }
            InternalEvent::MoveViCursorToPoint(point) => {
                trace!("Move vi cursor to point: point={point:?}");
                vi_goto_point(term, *point);
                self.refresh_hovered_word(window);
            }
            InternalEvent::ToggleViMode => {
                trace!("Toggling vi mode");
                self.vi_mode_enabled = !self.vi_mode_enabled;
                toggle_term_vi_mode(term);
            }
            InternalEvent::ViMotion(motion) => {
                trace!("Performing vi motion: motion={motion:?}");
                vi_motion(term, *motion);
            }
            InternalEvent::FindHyperlink(position, open) => {
                trace!("Finding hyperlink at position: position={position:?}, open={open:?}");

                let point = grid_point(
                    *position,
                    self.last_content.terminal_bounds,
                    display_offset(term),
                );

                match find_from_terminal_point(
                    term,
                    point,
                    &mut self.hyperlink_regex_searches,
                    self.path_style,
                ) {
                    Some(hyperlink) => {
                        let history_size = term.history_size();
                        self.process_hyperlink(hyperlink, *open, history_size, cx);
                    }
                    None => {
                        self.last_content.last_hovered_word = None;
                        cx.emit(Event::NewNavigationTarget(None));
                    }
                }
            }
            InternalEvent::ProcessHyperlink(hyperlink, open) => {
                // history_size must be read here since process_hyperlink cannot lock term
                // (sync() already holds the lock when dispatching events)
                let history_size = term.history_size();
                self.process_hyperlink(hyperlink.clone(), *open, history_size, cx);
            }
        }
    }

    fn process_hyperlink(
        &mut self,
        hyperlink: HyperlinkMatch,
        open: bool,
        history_size: usize,
        cx: &mut Context<Self>,
    ) {
        let HyperlinkMatch {
            text: maybe_url_or_path,
            is_url,
            range,
        } = hyperlink;
        let prev_hovered_word = self.last_content.last_hovered_word.take();
        let match_line = range.start().line;
        let working_directory = self.cwd_at_line(match_line, history_size);

        let target = if is_url {
            if let Some(path) = maybe_url_or_path.strip_prefix("file://") {
                let decoded_path = urlencoding::decode(path)
                    .map(|decoded| decoded.into_owned())
                    .unwrap_or(path.to_owned());

                MaybeNavigationTarget::PathLike(PathLikeTarget {
                    maybe_path: decoded_path,
                    working_directory,
                })
            } else {
                MaybeNavigationTarget::Url(maybe_url_or_path.clone())
            }
        } else {
            MaybeNavigationTarget::PathLike(PathLikeTarget {
                maybe_path: maybe_url_or_path.clone(),
                working_directory,
            })
        };

        if open {
            cx.emit(Event::Open(target));
        } else {
            self.update_selected_word(prev_hovered_word, range, maybe_url_or_path, target, cx);
        }
    }

    fn find_hyperlink_at_point(&mut self, point: Point) -> Option<HyperlinkMatch> {
        let term_lock = self.term.lock();
        find_from_terminal_point(
            &term_lock,
            point,
            &mut self.hyperlink_regex_searches,
            self.path_style,
        )
    }

    fn update_selected_word(
        &mut self,
        prev_word: Option<HoveredWord>,
        word_match: Range,
        word: String,
        navigation_target: MaybeNavigationTarget,
        cx: &mut Context<Self>,
    ) {
        if let Some(prev_word) = prev_word
            && prev_word.word == word
            && prev_word.word_match == word_match
        {
            self.last_content.last_hovered_word = Some(HoveredWord {
                word,
                word_match,
                id: prev_word.id,
            });
            return;
        }

        self.last_content.last_hovered_word = Some(HoveredWord {
            word,
            word_match,
            id: self.next_link_id(),
        });
        cx.emit(Event::NewNavigationTarget(Some(navigation_target)));
        cx.notify()
    }

    fn next_link_id(&mut self) -> usize {
        let res = self.next_link_id;
        self.next_link_id = self.next_link_id.wrapping_add(1);
        res
    }

    pub fn last_content(&self) -> &Content {
        &self.last_content
    }

    pub fn set_cursor_shape(&mut self, cursor_shape: SettingsCursorShape) {
        set_default_cursor_style(&mut self.term_config, cursor_shape);
        apply_config(&self.term, &self.term_config);
    }

    pub fn write_output(&mut self, bytes: &[u8], cx: &mut Context<Self>) {
        // Inject bytes directly into the terminal emulator and refresh the UI.
        // This bypasses the PTY/event loop for display-only terminals.
        let mut previous_byte_was_cr = false;
        let converted = convert_lf_to_crlf(bytes, &mut previous_byte_was_cr);

        let mut term = self.term.lock();
        self.output_processor.advance(&mut *term, &converted);
        drop(term);
        self.detect_init_command_startup_marker();
        cx.emit(Event::Wakeup);
    }

    pub fn total_lines(&self) -> usize {
        total_lines(&self.term.lock_unfair())
    }

    pub fn viewport_lines(&self) -> usize {
        screen_lines(&self.term.lock_unfair())
    }

    pub fn used_lines(&self) -> usize {
        used_lines(&self.term.lock_unfair())
    }

    //To test:
    //- Activate match on terminal (scrolling and selection)
    //- Editor search snapping behavior

    pub fn activate_match(&mut self, index: usize) {
        if let Some(search_match) = self.matches.get(index).cloned() {
            self.set_selection(Some(Selection::simple_range(search_match)));
            if self.vi_mode_enabled {
                self.events
                    .push_back(InternalEvent::MoveViCursorToPoint(search_match.end()));
            } else {
                self.events
                    .push_back(InternalEvent::ScrollToPoint(search_match.start()));
            }
        }
    }

    pub fn select_matches(&mut self, matches: &[Range]) {
        let matches_to_select = self
            .matches
            .iter()
            .filter(|self_match| matches.contains(self_match))
            .cloned()
            .collect::<Vec<_>>();
        for match_to_select in matches_to_select {
            self.set_selection(Some(Selection::simple_range(match_to_select)));
        }
    }

    pub fn select_all(&mut self) {
        let term = self.term.lock();
        let range = full_content_range(&term);
        drop(term);
        self.set_selection(Some(Selection::simple_range(range)));
    }

    fn set_selection(&mut self, selection: Option<Selection>) {
        self.events
            .push_back(InternalEvent::SetSelection(selection));
    }

    pub fn copy(&mut self, keep_selection: Option<bool>) {
        self.events.push_back(InternalEvent::Copy(keep_selection));
    }

    pub fn selection_head(&self) -> Option<Point> {
        self.selection_head.map(|anchor| anchor.point)
    }

    pub fn selection_cursor(&self) -> Option<Point> {
        self.selection_head.map(|anchor| Point {
            line: anchor.point.line,
            column: anchor.point.column + usize::from(anchor.side == SelectionSide::Right),
        })
    }

    pub fn move_output_selection(&mut self, movement: SelectionMovement, extend: bool) {
        self.events
            .push_back(InternalEvent::MoveSelection(movement, extend));
    }

    pub fn clear(&mut self) {
        self.events.push_back(InternalEvent::Clear)
    }

    pub fn shrink_to_used(&mut self) {
        shrink_to_used(&mut self.term.lock());
    }

    pub fn scroll_line_up(&mut self) {
        self.events
            .push_back(InternalEvent::Scroll(Scroll::Delta(1)));
    }

    pub fn scroll_up_by(&mut self, lines: usize) {
        self.events
            .push_back(InternalEvent::Scroll(Scroll::Delta(lines as i32)));
    }

    pub fn scroll_line_down(&mut self) {
        self.events
            .push_back(InternalEvent::Scroll(Scroll::Delta(-1)));
    }

    pub fn scroll_down_by(&mut self, lines: usize) {
        self.events
            .push_back(InternalEvent::Scroll(Scroll::Delta(-(lines as i32))));
    }

    pub fn scroll_page_up(&mut self) {
        self.events.push_back(InternalEvent::Scroll(Scroll::PageUp));
    }

    pub fn scroll_page_down(&mut self) {
        self.events
            .push_back(InternalEvent::Scroll(Scroll::PageDown));
    }

    pub fn scroll_to_top(&mut self) {
        self.events.push_back(InternalEvent::Scroll(Scroll::Top));
    }

    pub fn scroll_to_bottom(&mut self) {
        self.events.push_back(InternalEvent::Scroll(Scroll::Bottom));
    }

    pub fn scrolled_to_top(&self) -> bool {
        self.last_content.scrolled_to_top
    }

    pub fn scrolled_to_bottom(&self) -> bool {
        self.last_content.scrolled_to_bottom
    }

    ///Resize the terminal and the PTY.
    pub fn set_size(&mut self, new_bounds: TerminalBounds) {
        let new_bounds = normalize_terminal_bounds(new_bounds);

        let old_bounds = self.last_content.terminal_bounds;
        self.last_content.terminal_bounds = new_bounds;

        // Avoid spamming PTY resizes on pixel-level size changes (e.g. while dragging edges),
        // since those can generate excessive SIGWINCH/reflows and cause visible flicker.
        let requires_resize = old_bounds.num_lines() != new_bounds.num_lines()
            || old_bounds.num_columns() != new_bounds.num_columns()
            || old_bounds.cell_width != new_bounds.cell_width
            || old_bounds.line_height != new_bounds.line_height;

        if !requires_resize {
            return;
        }

        match self.events.back_mut() {
            Some(InternalEvent::Resize(pending_bounds)) => *pending_bounds = new_bounds,
            _ => self.events.push_back(InternalEvent::Resize(new_bounds)),
        }
    }

    /// Write the Input payload to the PTY, if applicable.
    /// (This is a no-op for display-only terminals.)
    fn write_to_pty(&self, input: impl Into<Cow<'static, [u8]>>) {
        let input = input.into();
        #[cfg(any(test, feature = "test-support"))]
        self.pty_write_log.borrow_mut().push(input.to_vec());
        if let TerminalType::Pty { pty_tx, .. } = &self.terminal_type {
            if log::log_enabled!(log::Level::Debug) {
                if let Ok(str) = str::from_utf8(&input) {
                    log::debug!("Writing to PTY: {:?}", str);
                } else {
                    log::debug!("Writing to PTY: {:?}", input);
                }
            }
            pty_tx.notify(input);
        }
    }

    pub fn input(&mut self, input: impl Into<Cow<'static, [u8]>>) {
        self.keyboard_input_sent = true;
        self.complete_init_command_startup_handshake();
        self.write_input(input);
    }

    /// Sends a shell-level marker command and returns a task that completes when
    /// the marker appears in terminal output. Already complete for non-PTY
    /// terminals or those whose child has exited.
    ///
    /// Call at most once per terminal: a second handshake drops the previous
    /// `Sender`, which would write the init command twice.
    pub fn start_init_command_startup_handshake(&mut self) -> Task<()> {
        if !self.is_pty() || self.child_exited.is_some() {
            return Task::ready(());
        }

        debug_assert!(
            self.init_command_startup_tx.is_none(),
            "start_init_command_startup_handshake called while a handshake is already in flight"
        );

        let (startup_tx, startup_rx) = async_channel::bounded(1);
        let startup_task = self.background_executor.spawn(async move {
            match startup_rx.recv().await {
                Ok(()) | Err(_) => {}
            }
        });

        let marker_id = NEXT_INIT_COMMAND_STARTUP_MARKER_ID.fetch_add(1, Ordering::Relaxed);
        self.init_command_startup_marker = Some(init_command_startup_marker(marker_id));
        self.init_command_startup_tx = Some(startup_tx);

        let shell_kind = self.template.shell.shell_kind(self.path_style.is_windows());
        let mut input = init_command_startup_marker_command(shell_kind, marker_id).into_bytes();
        input.push(b'\x0d');
        self.write_to_pty(input);

        startup_task
    }

    fn detect_init_command_startup_marker(&mut self) {
        let Some(marker) = self.init_command_startup_marker.as_deref() else {
            return;
        };

        let has_marker = {
            let term = self.term.lock_unfair();
            last_non_empty_lines(&term, INIT_COMMAND_STARTUP_MARKER_SEARCH_LINES)
                .iter()
                .any(|line| line.contains(marker))
        };

        if has_marker {
            self.complete_init_command_startup_handshake();
        }
    }

    fn complete_init_command_startup_handshake(&mut self) {
        self.init_command_startup_marker = None;
        if let Some(startup_tx) = self.init_command_startup_tx.take() {
            match startup_tx.try_send(()) {
                Ok(()) | Err(async_channel::TrySendError::Full(())) => {}
                Err(async_channel::TrySendError::Closed(())) => {}
            }
        }
    }

    /// Write a programmatically-generated command to the PTY as if it had been
    /// typed, without marking the terminal as having received user keyboard
    /// input.
    pub fn write_init_command(&mut self, input: impl Into<Cow<'static, [u8]>>) {
        self.write_input(input);
    }

    pub fn is_pty(&self) -> bool {
        matches!(self.terminal_type, TerminalType::Pty { .. })
    }

    pub fn write_init_command_after_startup(
        &mut self,
        input: impl Into<Cow<'static, [u8]>>,
        cx: &mut Context<Self>,
    ) -> bool {
        // Ends the handshake even if the marker was never seen (timeout
        // fallback), so detection stops scanning on every wakeup.
        self.complete_init_command_startup_handshake();

        if self.keyboard_input_sent || self.child_exited.is_some() {
            return false;
        }

        self.clear_for_init_command(cx);
        self.write_init_command(input);
        true
    }

    fn clear_for_init_command(&mut self, cx: &mut Context<Self>) {
        let mut term = self.term.lock_unfair();
        clear_saved_screen(&mut term);
        self.last_content = make_content(&term, &self.last_content);
        drop(term);
        self.reset_cwd_history();
        cx.emit(Event::Wakeup);
    }

    fn write_input(&mut self, input: impl Into<Cow<'static, [u8]>>) {
        let input = input.into();
        if !self.is_remote_terminal && input.contains(&b'\r') {
            let term = self.term.lock_unfair();
            self.pending_cwd_boundary = Some(Self::scrollback_position(
                term.grid().cursor.point.line.0,
                term.history_size(),
            ));
        }

        self.events.push_back(InternalEvent::Scroll(Scroll::Bottom));
        self.events.push_back(InternalEvent::SetSelection(None));
        #[cfg(any(test, feature = "test-support"))]
        self.input_log.push(input.to_vec());

        self.write_to_pty(input);
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn take_input_log(&mut self) -> Vec<Vec<u8>> {
        std::mem::take(&mut self.input_log)
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn take_pty_write_log(&mut self) -> Vec<Vec<u8>> {
        std::mem::take(self.pty_write_log.get_mut())
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn keyboard_input_sent(&self) -> bool {
        self.keyboard_input_sent
    }

    pub fn toggle_vi_mode(&mut self) {
        self.events.push_back(InternalEvent::ToggleViMode);
    }

    pub fn vi_motion(&mut self, keystroke: &Keystroke) {
        if !self.vi_mode_enabled {
            return;
        }

        let key: Cow<'_, str> = if keystroke.modifiers.shift {
            Cow::Owned(keystroke.key.to_uppercase())
        } else {
            Cow::Borrowed(keystroke.key.as_str())
        };

        let motion: Option<ViMotion> = match key.as_ref() {
            "h" | "left" => Some(ViMotion::Left),
            "j" | "down" => Some(ViMotion::Down),
            "k" | "up" => Some(ViMotion::Up),
            "l" | "right" => Some(ViMotion::Right),
            "w" => Some(ViMotion::WordRight),
            "b" if !keystroke.modifiers.control => Some(ViMotion::WordLeft),
            "e" => Some(ViMotion::WordRightEnd),
            "%" => Some(ViMotion::Bracket),
            "$" => Some(ViMotion::Last),
            "0" => Some(ViMotion::First),
            "^" => Some(ViMotion::FirstOccupied),
            "H" => Some(ViMotion::High),
            "M" => Some(ViMotion::Middle),
            "L" => Some(ViMotion::Low),
            "{" => Some(ViMotion::ParagraphUp),
            "}" => Some(ViMotion::ParagraphDown),
            _ => None,
        };

        if let Some(motion) = motion {
            let cursor = self.last_content.cursor.point;
            let cursor_pos = GpuiPoint {
                x: cursor.column as f32 * self.last_content.terminal_bounds.cell_width,
                y: cursor.line as f32 * self.last_content.terminal_bounds.line_height,
            };
            self.events
                .push_back(InternalEvent::UpdateSelection(cursor_pos));
            self.events.push_back(InternalEvent::ViMotion(motion));
            return;
        }

        let scroll_motion = match key.as_ref() {
            "g" => Some(Scroll::Top),
            "G" => Some(Scroll::Bottom),
            "b" if keystroke.modifiers.control => Some(Scroll::PageUp),
            "f" if keystroke.modifiers.control => Some(Scroll::PageDown),
            "d" if keystroke.modifiers.control => {
                let amount = self.last_content.terminal_bounds.line_height().to_f64() as i32 / 2;
                Some(Scroll::Delta(-amount))
            }
            "u" if keystroke.modifiers.control => {
                let amount = self.last_content.terminal_bounds.line_height().to_f64() as i32 / 2;
                Some(Scroll::Delta(amount))
            }
            _ => None,
        };

        if let Some(scroll_motion) = scroll_motion {
            self.events.push_back(InternalEvent::Scroll(scroll_motion));
            return;
        }

        match key.as_ref() {
            "v" => {
                let point = self.last_content.cursor.point;
                let selection_type = SelectionType::Simple;
                let side = SelectionSide::Right;
                let selection = Selection::new(selection_type, point, side);
                self.events
                    .push_back(InternalEvent::SetSelection(Some(selection)));
            }

            "escape" => {
                self.events.push_back(InternalEvent::SetSelection(None));
            }

            "y" => {
                self.copy(Some(false));
            }

            "i" => {
                self.scroll_to_bottom();
                self.toggle_vi_mode();
            }
            _ => {}
        }
    }

    pub fn try_keystroke(&mut self, keystroke: &Keystroke, option_as_meta: bool) -> bool {
        if self.vi_mode_enabled {
            self.vi_motion(keystroke);
            return true;
        }

        // Keep default terminal behavior
        let esc = to_esc_str(keystroke, self.last_content.mode, option_as_meta);
        if let Some(esc) = esc {
            match esc {
                Cow::Borrowed(string) => self.input(string.as_bytes()),
                Cow::Owned(string) => self.input(string.into_bytes()),
            };
            true
        } else {
            false
        }
    }

    pub fn try_modifiers_change(
        &mut self,
        modifiers: &Modifiers,
        window: &Window,
        cx: &mut Context<Self>,
    ) {
        if self
            .last_content
            .terminal_bounds
            .bounds
            .contains(&window.mouse_position())
            && modifiers.secondary()
        {
            self.refresh_hovered_word(window);
        }
        cx.notify();
    }

    ///Paste text into the terminal
    pub fn paste(&mut self, text: &str) {
        let paste_text = if self.last_content.mode.contains(Modes::BRACKETED_PASTE) {
            format!("{}{}{}", "\x1b[200~", text.replace('\x1b', ""), "\x1b[201~")
        } else {
            text.replace("\r\n", "\r").replace('\n', "\r")
        };

        self.input(paste_text.into_bytes());
    }

    pub fn sync(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let term = self.term.clone();
        let mut terminal = term.lock_unfair();
        //Note that the ordering of events matters for event processing
        while let Some(e) = self.events.pop_front() {
            self.process_terminal_event(&e, &mut terminal, window, cx)
        }

        self.last_content = make_content(&terminal, &self.last_content);
    }

    pub fn with_renderable_cells<R>(&self, f: impl for<'a> FnOnce(RenderableCells<'a>) -> R) -> R {
        let term = self.term.lock_unfair();
        let content = term.renderable_content();
        f(RenderableCells::new(content.display_iter))
    }

    pub fn snapshot(&self) -> Result<TerminalSnapshot> {
        alacritty::capture_snapshot(&self.term.lock_unfair())
    }

    pub fn restore_snapshot(
        &mut self,
        snapshot: &TerminalSnapshot,
        cx: &mut Context<Self>,
    ) -> Result<()> {
        if !matches!(self.terminal_type, TerminalType::DisplayOnly)
            || self.subprocess.is_some()
            || self.task.is_some()
        {
            bail!("Terminal history can only be restored into a display-only terminal");
        }
        let mut term = self.term.lock_unfair();
        alacritty::restore_snapshot(&mut term, snapshot)?;
        resize(&mut term, self.last_content.terminal_bounds);
        self.last_content = make_content(&term, &self.last_content);
        self.matches.clear();
        self.selection_head = None;
        drop(term);
        // Retained cells are installed directly, without interpreting historical OSC/BEL bytes.
        cx.emit(Event::Wakeup);
        Ok(())
    }

    pub fn get_content(&self) -> String {
        let term = self.term.lock_unfair();
        content_text(&term)
    }

    pub fn last_n_non_empty_lines(&self, n: usize) -> Vec<String> {
        let terminal = self.term.lock_unfair();
        last_non_empty_lines(&terminal, n)
    }

    pub fn focus_in(&self) {
        if self.last_content.mode.contains(Modes::FOCUS_IN_OUT) {
            self.write_to_pty("\x1b[I".as_bytes());
        }
    }

    pub fn focus_out(&mut self) {
        if self.last_content.mode.contains(Modes::FOCUS_IN_OUT) {
            self.write_to_pty("\x1b[O".as_bytes());
        }
    }

    fn mouse_changed(&mut self, point: Point, side: SelectionSide) -> bool {
        match self.last_mouse {
            Some((old_point, old_side)) => {
                if old_point == point && old_side == side {
                    false
                } else {
                    self.last_mouse = Some((point, side));
                    true
                }
            }
            None => {
                self.last_mouse = Some((point, side));
                true
            }
        }
    }

    pub fn mouse_mode(&self, shift: bool) -> bool {
        self.last_content.mode.intersects(Modes::MOUSE_MODE) && !shift
    }

    pub fn mouse_move(&mut self, e: &MouseMoveEvent, cx: &mut Context<Self>) {
        let position = e.position - self.last_content.terminal_bounds.bounds.origin;
        if self.mouse_mode(e.modifiers.shift) {
            // A ctrl/cmd press on a link suppressed its button-press report in
            // `mouse_down`. Since the app never saw the press, we must swallow
            // the whole gesture rather than forward later motion/release
            // reports, which would be a press-less (malformed) sequence.
            // `mouse_up` resolves it: release on the same link opens it,
            // otherwise the gesture is dropped.
            if self.mouse_down_hyperlink.is_none() {
                let (point, side) = grid_point_and_side(
                    position,
                    self.last_content.terminal_bounds,
                    self.last_content.display_offset,
                );

                if self.mouse_changed(point, side) {
                    let bytes = mouse_moved_report(
                        point,
                        e.pressed_button,
                        e.modifiers,
                        self.last_content.mode,
                    );

                    if let Some(bytes) = bytes {
                        self.write_to_pty(bytes);
                    }
                }
            }
        } else {
            self.schedule_find_hyperlink(e.modifiers, e.position);
        }
        cx.notify();
    }

    fn schedule_find_hyperlink(&mut self, modifiers: Modifiers, position: GpuiPoint<Pixels>) {
        if self.selection_phase == SelectionPhase::Selecting
            || !modifiers.secondary()
            || !self.last_content.terminal_bounds.bounds.contains(&position)
        {
            self.last_content.last_hovered_word = None;
            return;
        }

        // Throttle hyperlink searches to avoid excessive processing
        let now = Instant::now();
        if self
            .last_hyperlink_search_position
            .map_or(true, |last_pos| {
                // Only search if mouse moved significantly or enough time passed
                let distance_moved = ((position.x - last_pos.x).abs()
                    + (position.y - last_pos.y).abs())
                    > FIND_HYPERLINK_THROTTLE_PX;
                let time_elapsed = now.duration_since(self.last_mouse_move_time).as_millis() > 100;
                distance_moved || time_elapsed
            })
        {
            self.last_mouse_move_time = now;
            self.last_hyperlink_search_position = Some(position);
            self.events.push_back(InternalEvent::FindHyperlink(
                position - self.last_content.terminal_bounds.bounds.origin,
                false,
            ));
        }
    }

    pub fn select_word_at_event_position(&mut self, e: &MouseDownEvent) {
        let position = e.position - self.last_content.terminal_bounds.bounds.origin;
        let (point, side) = grid_point_and_side(
            position,
            self.last_content.terminal_bounds,
            self.last_content.display_offset,
        );
        let selection = Selection::new(SelectionType::Semantic, point, side);
        self.events
            .push_back(InternalEvent::SetSelection(Some(selection)));
    }

    pub fn mouse_drag(
        &mut self,
        e: &MouseMoveEvent,
        region: Bounds<Pixels>,
        cx: &mut Context<Self>,
    ) {
        let position = e.position - self.last_content.terminal_bounds.bounds.origin;
        if !self.mouse_mode(e.modifiers.shift) {
            if let Some(hyperlink) = &self.mouse_down_hyperlink {
                let point = grid_point(
                    position,
                    self.last_content.terminal_bounds,
                    self.last_content.display_offset,
                );

                if !hyperlink.range.contains(point) {
                    self.mouse_down_hyperlink = None;
                } else {
                    return;
                }
            }

            // Ignore tiny pointer movements so that a click that jitters by a
            // pixel or two (e.g. the window-focusing click) does not begin a
            // selection. Mirrors the drag threshold used by gpui's `div`.
            if self.selection_phase != SelectionPhase::Selecting
                && let Some(mouse_down_position) = self.mouse_down_position
                && (e.position - mouse_down_position).magnitude() <= SELECTION_DRAG_THRESHOLD
            {
                return;
            }

            self.selection_phase = SelectionPhase::Selecting;
            // Alacritty has the same ordering, of first updating the selection
            // then scrolling 15ms later
            self.events
                .push_back(InternalEvent::UpdateSelection(position));

            // Doesn't make sense to scroll the alt screen
            if !self.last_content.mode.contains(Modes::ALT_SCREEN) {
                let scroll_lines = match self.drag_line_delta(e, region) {
                    Some(value) => value,
                    None => return,
                };

                self.events
                    .push_back(InternalEvent::Scroll(Scroll::Delta(scroll_lines)));
            }

            cx.notify();
        }
    }

    fn drag_line_delta(&self, e: &MouseMoveEvent, region: Bounds<Pixels>) -> Option<i32> {
        let top = region.origin.y;
        let bottom = region.bottom_left().y;

        let scroll_lines = if e.position.y < top {
            let scroll_delta = (top - e.position.y).pow(1.1);
            (scroll_delta / self.last_content.terminal_bounds.line_height).ceil() as i32
        } else if e.position.y > bottom {
            let scroll_delta = -((e.position.y - bottom).pow(1.1));
            (scroll_delta / self.last_content.terminal_bounds.line_height).floor() as i32
        } else {
            return None;
        };

        Some(scroll_lines.clamp(-3, 3))
    }

    pub fn output_mouse_down(&mut self, e: &MouseDownEvent) {
        if e.button != MouseButton::Left {
            return;
        }
        let position = e.position - self.last_content.terminal_bounds.bounds.origin;
        self.mouse_down_position = Some(e.position);
        let (point, side) = grid_point_and_side(
            position,
            self.last_content.terminal_bounds,
            self.last_content.display_offset,
        );

        let selection_type = match e.click_count {
            0 => return, //This is a release
            1 => Some(SelectionType::Simple),
            2 => Some(SelectionType::Semantic),
            3 => Some(SelectionType::Lines),
            _ => None,
        };

        if selection_type == Some(SelectionType::Simple) && e.modifiers.shift {
            if self.last_content.selection.is_some() {
                // Shift+click extends the existing selection to this point.
                self.events
                    .push_back(InternalEvent::UpdateSelection(position));
            } else {
                // With no selection yet, Shift is the escape hatch for
                // selecting text while an app has mouse tracking enabled,
                // so anchor a selection here for the drag to extend.
                self.events
                    .push_back(InternalEvent::SetSelection(Some(Selection::new(
                        SelectionType::Simple,
                        point,
                        side,
                    ))));
            }
            return;
        }

        let selection =
            selection_type.map(|selection_type| Selection::new(selection_type, point, side));

        if let Some(selection) = selection {
            self.events
                .push_back(InternalEvent::SetSelection(Some(selection)));
        }
    }

    pub fn output_mouse_drag(&mut self, e: &MouseMoveEvent, region: Bounds<Pixels>) {
        if e.pressed_button != Some(MouseButton::Left) {
            return;
        }
        if self.selection_phase != SelectionPhase::Selecting
            && let Some(position) = self.mouse_down_position
            && (e.position - position).magnitude() <= SELECTION_DRAG_THRESHOLD
        {
            return;
        }
        self.selection_phase = SelectionPhase::Selecting;
        self.events.push_back(InternalEvent::UpdateSelection(
            e.position - self.last_content.terminal_bounds.bounds.origin,
        ));
        if let Some(lines) = self.drag_line_delta(e, region) {
            self.events
                .push_back(InternalEvent::Scroll(Scroll::Delta(lines)));
        }
    }

    pub fn output_mouse_up(&mut self, cx: &Context<Self>) {
        if TerminalSettings::get_global(cx).copy_on_select {
            self.copy(Some(true));
        }
        self.selection_phase = SelectionPhase::Ended;
        self.last_mouse = None;
        self.mouse_down_position = None;
    }

    pub fn mouse_down(&mut self, e: &MouseDownEvent, cx: &mut Context<Self>) {
        let position = e.position - self.last_content.terminal_bounds.bounds.origin;
        let point = grid_point(
            position,
            self.last_content.terminal_bounds,
            self.last_content.display_offset,
        );

        if e.button == MouseButton::Left
            && e.modifiers.secondary()
            && (TerminalSettings::get_global(cx).open_links_in_mouse_mode
                || !self.mouse_mode(e.modifiers.shift))
        {
            self.mouse_down_hyperlink = self.find_hyperlink_at_point(point);

            if self.mouse_down_hyperlink.is_some() {
                return;
            }
        }

        if self.mouse_mode(e.modifiers.shift) {
            let bytes =
                mouse_button_report(point, e.button, e.modifiers, true, self.last_content.mode);

            if let Some(bytes) = bytes {
                self.write_to_pty(bytes);
            }
        } else {
            match e.button {
                MouseButton::Left => self.output_mouse_down(e),
                #[cfg(any(target_os = "linux", target_os = "freebsd"))]
                MouseButton::Middle => {
                    if let Some(item) = cx.read_from_primary() {
                        let text = item.text().unwrap_or_default();
                        self.paste(&text);
                    }
                }
                _ => {}
            }
        }
    }

    pub fn mouse_up(&mut self, e: &MouseUpEvent, cx: &Context<Self>) {
        let setting = TerminalSettings::get_global(cx);

        let position = e.position - self.last_content.terminal_bounds.bounds.origin;
        if let Some(mouse_down_hyperlink) = self.mouse_down_hyperlink.take() {
            let point = grid_point(
                position,
                self.last_content.terminal_bounds,
                self.last_content.display_offset,
            );

            if self
                .find_hyperlink_at_point(point)
                .is_some_and(|mouse_up_hyperlink| mouse_up_hyperlink == mouse_down_hyperlink)
            {
                self.events
                    .push_back(InternalEvent::ProcessHyperlink(mouse_down_hyperlink, true));
                self.selection_phase = SelectionPhase::Ended;
                self.last_mouse = None;
                self.mouse_down_position = None;
                return;
            }

            if self.mouse_mode(e.modifiers.shift) {
                self.selection_phase = SelectionPhase::Ended;
                self.last_mouse = None;
                self.mouse_down_position = None;
                return;
            }
        }

        if self.mouse_mode(e.modifiers.shift) {
            let point = grid_point(
                position,
                self.last_content.terminal_bounds,
                self.last_content.display_offset,
            );

            let bytes =
                mouse_button_report(point, e.button, e.modifiers, false, self.last_content.mode);

            if let Some(bytes) = bytes {
                self.write_to_pty(bytes);
            }
        } else {
            if e.button == MouseButton::Left && setting.copy_on_select {
                self.copy(Some(true));
            }

            //Hyperlinks
            if self.selection_phase == SelectionPhase::Ended {
                let mouse_cell_index =
                    content_index_for_mouse(position, &self.last_content.terminal_bounds);
                if let Some(link) = self
                    .last_content
                    .cells
                    .get(mouse_cell_index)
                    .and_then(|cell| cell.hyperlink())
                {
                    cx.open_url(link.uri());
                } else if e.modifiers.secondary() {
                    self.events
                        .push_back(InternalEvent::FindHyperlink(position, true));
                }
            }
        }

        self.selection_phase = SelectionPhase::Ended;
        self.last_mouse = None;
        self.mouse_down_position = None;
    }

    ///Scroll the terminal
    pub fn scroll_wheel(&mut self, e: &ScrollWheelEvent, scroll_multiplier: f32) {
        let mouse_mode = self.mouse_mode(e.shift);
        let scroll_multiplier = if mouse_mode { 1. } else { scroll_multiplier };

        if let Some(scroll_lines) = self.determine_scroll_lines(e, scroll_multiplier)
            && scroll_lines != 0
        {
            if mouse_mode {
                let point = grid_point(
                    e.position - self.last_content.terminal_bounds.bounds.origin,
                    self.last_content.terminal_bounds,
                    self.last_content.display_offset,
                );

                if let Some(scrolls) = scroll_report(point, scroll_lines, e, self.last_content.mode)
                {
                    for scroll in scrolls {
                        self.write_to_pty(scroll);
                    }
                };
            } else if self
                .last_content
                .mode
                .contains(Modes::ALT_SCREEN | Modes::ALTERNATE_SCROLL)
                && !e.shift
            {
                self.write_to_pty(alt_scroll(scroll_lines));
            } else {
                self.events
                    .push_back(InternalEvent::Scroll(Scroll::Delta(scroll_lines)));
            }
        }
    }

    fn refresh_hovered_word(&mut self, window: &Window) {
        self.schedule_find_hyperlink(window.modifiers(), window.mouse_position());
    }

    fn determine_scroll_lines(
        &mut self,
        e: &ScrollWheelEvent,
        scroll_multiplier: f32,
    ) -> Option<i32> {
        let line_height = self.last_content.terminal_bounds.line_height;
        match e.touch_phase {
            /* Reset scroll state on started */
            TouchPhase::Started => {
                self.scroll_px = px(0.);
                None
            }
            /* Calculate the appropriate scroll lines */
            TouchPhase::Moved => {
                let old_offset = (self.scroll_px / line_height) as i32;

                self.scroll_px += e.delta.pixel_delta(line_height).y * scroll_multiplier;

                let new_offset = (self.scroll_px / line_height) as i32;

                // Whenever we hit the edges, reset our stored scroll to 0
                // so we can respond to changes in direction quickly
                self.scroll_px %= self.last_content.terminal_bounds.height();

                Some(new_offset - old_offset)
            }
            // Cancellation does not commit a scroll, same as a plain end.
            TouchPhase::Ended | TouchPhase::Cancelled => None,
        }
    }

    pub fn find_matches(&self, searcher: Search, cx: &Context<Self>) -> Task<Vec<Range>> {
        let term = self.term.clone();
        cx.background_spawn(async move {
            let term = term.lock();
            search_matches(&term, searcher)
        })
    }

    pub fn working_directory(&self) -> Option<PathBuf> {
        if self.is_remote_terminal {
            // We can't yet reliably detect the working directory of a shell on the
            // SSH host. Until we can do that, it doesn't make sense to display
            // the working directory on the client and persist that.
            None
        } else {
            self.client_side_working_directory()
        }
    }

    /// Normalizes the command name of the foreground process, if one is known.
    pub fn foreground_process_command_name(&self) -> Option<String> {
        match &self.terminal_type {
            TerminalType::Pty { info, .. } => info
                .current
                .read()
                .as_ref()
                .and_then(|process| foreground_process_command_from_argv(&process.argv)),
            TerminalType::DisplayOnly => None,
        }
    }

    /// Returns the working directory of the process that's connected to the PTY.
    /// That means it returns the working directory of the local shell or program
    /// that's running inside the terminal.
    ///
    /// This does *not* return the working directory of the shell that runs on the
    /// remote host, in case Zed is connected to a remote host.
    fn client_side_working_directory(&self) -> Option<PathBuf> {
        match &self.terminal_type {
            TerminalType::Pty { info, .. } => info
                .current
                .read()
                .as_ref()
                .map(|process| process.cwd.clone()),
            TerminalType::DisplayOnly => None,
        }
    }

    pub(crate) fn record_cwd_change(&mut self, new_working_directory: PathBuf) {
        if self.is_remote_terminal {
            return;
        }

        let scrollback_position = self.pending_cwd_boundary.take().unwrap_or_else(|| {
            let term = self.term.lock_unfair();
            Self::scrollback_position(term.grid().cursor.point.line.0, term.history_size())
        });
        self.cwd_history.push(CwdHistoryEntry {
            scrollback_position,
            working_directory: new_working_directory,
        });
    }

    fn reset_cwd_history(&mut self) {
        self.pending_cwd_boundary = None;
        self.cwd_history = self
            .working_directory()
            .map(|working_directory| {
                vec![CwdHistoryEntry {
                    scrollback_position: i32::MIN,
                    working_directory,
                }]
            })
            .unwrap_or_default();
    }

    fn cwd_at_line(&self, line: i32, history_size: usize) -> Option<PathBuf> {
        // Once the scrollback cap is reached, evictions move retained lines without changing
        // `history_size`, so stored row offsets no longer identify their original lines.
        if self.is_remote_terminal
            || self.cwd_history.is_empty()
            || history_size >= self.term_config.scrolling_history
        {
            return self.working_directory();
        }
        let scrollback_position = Self::scrollback_position(line, history_size);
        self.cwd_history
            .iter()
            .rev()
            .find(|entry| entry.scrollback_position <= scrollback_position)
            .map(|entry| entry.working_directory.clone())
            .or_else(|| self.working_directory())
    }

    fn scrollback_position(line: i32, history_size: usize) -> i32 {
        let history_size = i32::try_from(history_size).unwrap_or(i32::MAX);
        history_size.saturating_add(line)
    }

    pub fn title(&self, truncate: bool) -> String {
        const MAX_CHARS: usize = 25;
        match &self.task {
            Some(task_state) => {
                if truncate {
                    truncate_and_trailoff(&task_state.spawned_task.label, MAX_CHARS)
                } else {
                    task_state.spawned_task.full_label.clone()
                }
            }
            None => self
                .title_override
                .as_ref()
                .map(|title_override| title_override.to_string())
                .unwrap_or_else(|| match &self.terminal_type {
                    TerminalType::Pty { info, .. } => info
                        .current
                        .read()
                        .as_ref()
                        .map(|fpi| {
                            let process_file = fpi
                                .cwd
                                .file_name()
                                .map(|name| name.to_string_lossy().into_owned())
                                .unwrap_or_default();

                            let argv = fpi.argv.as_slice();
                            let process_name = format!(
                                "{}{}",
                                fpi.name,
                                if !argv.is_empty() {
                                    format!(" {}", (argv[1..]).join(" "))
                                } else {
                                    "".to_string()
                                }
                            );
                            let (process_file, process_name) = if truncate {
                                (
                                    truncate_and_trailoff(&process_file, MAX_CHARS),
                                    truncate_and_trailoff(&process_name, MAX_CHARS),
                                )
                            } else {
                                (process_file, process_name)
                            };
                            format!("{process_file} — {process_name}")
                        })
                        .unwrap_or_else(|| "Terminal".to_string()),
                    TerminalType::DisplayOnly => "Terminal".to_string(),
                }),
        }
    }

    pub fn kill_active_task(&mut self) {
        if let Some(task) = self.task()
            && task.status == TaskStatus::Running
        {
            match &self.terminal_type {
                TerminalType::Pty { info, .. } => {
                    // First kill the foreground process group (the command running in the shell)
                    info.kill_current_process();
                    // Then kill the shell itself so that the terminal exits properly
                    // and wait_for_completed_task can complete
                    info.kill_child_process();
                }
                TerminalType::DisplayOnly => {
                    // Non-PTY task terminals own their subprocess directly.
                    if let Some(subprocess) = &self.subprocess {
                        subprocess.kill();
                    }
                }
            }
        }
    }

    pub fn can_reuse_interactive_process(&self) -> bool {
        self.task.is_none()
            && self.is_pty()
            && !self.process_finished
            && !self.user_stop_requested
            && self.child_exited.is_none()
    }

    pub fn request_stop_interactive_process(&mut self) -> Result<()> {
        if self.task.is_some() {
            bail!("Task terminals must use their task cancellation owner");
        }
        let TerminalType::Pty { info, .. } = &self.terminal_type else {
            bail!("Terminal history has no interactive process to stop");
        };
        // A retained exited terminal must never signal a reused process identifier.
        if self.process_finished {
            return Ok(());
        }
        let foreground_signal_sent = info.kill_current_process();
        let shell_signal_sent = info.kill_child_process();
        if !foreground_signal_sent && !shell_signal_sent {
            bail!("Could not deliver a stop signal to the interactive terminal");
        }
        self.user_stop_requested = true;
        Ok(())
    }

    pub fn pid(&self) -> Option<sysinfo::Pid> {
        match &self.terminal_type {
            TerminalType::Pty { info, .. } => info.pid(),
            TerminalType::DisplayOnly => None,
        }
    }

    // A direct local DAP Task needs its original child ID even without a PTY.
    // Wrappers cannot identify the requested process from their own child ID.
    // This retained identifier says nothing about liveness or successful completion.
    pub fn task_process_id(&self) -> Option<sysinfo::Pid> {
        if let Some(process_id) = self.pid() {
            return Some(process_id);
        }
        let task = self.task.as_ref()?;
        if self.is_remote_terminal || !self.activation_script.is_empty() {
            return None;
        }
        let Shell::WithArguments { program, args, .. } = &self.template.shell else {
            return None;
        };
        if task.spawned_task.command.as_deref() != Some(program.as_str())
            || task.spawned_task.args != *args
        {
            return None;
        }
        self.subprocess
            .as_ref()
            .map(|subprocess| sysinfo::Pid::from_u32(subprocess.process_id))
    }

    pub fn pid_getter(&self) -> Option<&ProcessIdGetter> {
        match &self.terminal_type {
            TerminalType::Pty { info, .. } => Some(info.pid_getter()),
            TerminalType::DisplayOnly => None,
        }
    }

    pub fn blocks_application_restart(&self) -> bool {
        self.task
            .as_ref()
            .is_some_and(|task| task.status == TaskStatus::Running)
            || (self.is_pty() && self.child_exited.is_none())
    }

    pub fn task(&self) -> Option<&TaskState> {
        self.task.as_ref()
    }

    pub fn wait_for_completed_task(&self, cx: &App) -> Task<Option<ExitStatus>> {
        let Some(task) = self.task() else {
            return Task::ready(None);
        };
        if task.status != TaskStatus::Running {
            return Task::ready(self.child_exited);
        }
        // Channel clones compete for one exit message; the Task observer and queue need the same result.
        let completion = self
            .task_completion
            .get_or_init(|| {
                let receiver = task.completion_rx.clone();
                async move {
                    match receiver.recv().await {
                        Ok(status) => status,
                        Err(error) => {
                            log::warn!(
                                "task completion channel closed before an exit status: {error}"
                            );
                            None
                        }
                    }
                }
                .boxed()
                .shared()
            })
            .clone();
        cx.spawn(async move |_| completion.await)
    }

    fn register_task_finished(
        &mut self,
        exit_status: Option<ExitStatus>,
        cx: &mut Context<Terminal>,
    ) {
        self.process_finished = true;
        if let Some(tx) = &self.completion_tx {
            tx.try_send(exit_status).ok();
        }
        if let Some(e) = exit_status {
            self.child_exited = Some(e);
        }
        self.complete_init_command_startup_handshake();
        let task = match &mut self.task {
            Some(task) => task,
            None => {
                // For interactive shells (no task), we need to differentiate:
                // 1. User-initiated exits (typed "exit", Ctrl+D, etc.) - always close,
                //    even if the shell exits with a non-zero code (e.g. after `false`).
                // 2. Shell spawn failures (bad $SHELL) - don't close, so the user sees
                //    the error. Spawn failures never receive keyboard input.
                let should_close = if self.keyboard_input_sent || self.user_stop_requested {
                    true
                } else {
                    self.child_exited.is_none_or(|e| e.code() == Some(0))
                };
                if should_close {
                    cx.emit(Event::CloseTerminal);
                }
                return;
            }
        };
        if task.status != TaskStatus::Running {
            return;
        }
        match exit_status.and_then(|e| e.code()) {
            Some(error_code) => {
                task.status.register_task_exit(error_code);
            }
            None => {
                task.status.register_terminal_exit();
            }
        };
        if let Some(handle) = task.structured_task.as_ref() {
            match handle.state() {
                task::StructuredTaskState::Cancelled { terminal_id, .. } => {
                    handle.mark_cancelled(terminal_id, true, cx);
                }
                _ => {
                    if let Some(exit_status) = exit_status {
                        handle.mark_completed(exit_status, cx);
                    } else {
                        handle.mark_cancelled(handle.state().terminal_id(), true, cx);
                    }
                }
            }
        }

        let (finished_successfully, task_line, command_line) = task_summary(task, exit_status);
        let mut lines_to_show = Vec::new();
        if task.spawned_task.show_summary {
            lines_to_show.push(task_line.as_str());
        }
        if task.spawned_task.show_command {
            lines_to_show.push(command_line.as_str());
        }
        let hide = task.spawned_task.hide;

        if !lines_to_show.is_empty() {
            // SAFETY: the invocation happens on non `TaskStatus::Running` tasks, once,
            // after either `AlacTermEvent::Exit` or `AlacTermEvent::ChildExit` events that are spawned
            // when Zed task finishes and no more output is made.
            // After the task summary is output once, no more text is appended to the terminal.
            unsafe { append_text_to_term(&mut self.term.lock(), &lines_to_show) };
        }

        match hide {
            HideStrategy::Never => {}
            HideStrategy::Always => {
                cx.emit(Event::CloseTerminal);
            }
            HideStrategy::OnSuccess => {
                if finished_successfully {
                    cx.emit(Event::CloseTerminal);
                }
            }
        }
    }

    pub fn vi_mode_enabled(&self) -> bool {
        self.vi_mode_enabled
    }

    pub fn clone_builder(&self, cx: &App, cwd: Option<PathBuf>) -> Task<Result<TerminalBuilder>> {
        let working_directory = self.working_directory().or_else(|| cwd);
        TerminalBuilder::new(
            working_directory,
            None,
            self.template.shell.clone(),
            self.template.env.clone(),
            self.template.cursor_shape,
            self.template.alternate_scroll,
            self.template.max_scroll_history_lines,
            self.template.path_hyperlink_regexes.clone(),
            self.template.path_hyperlink_timeout_ms,
            self.is_remote_terminal,
            self.template.window_id,
            None,
            cx,
            self.activation_script.clone(),
            self.path_style,
        )
    }
}

const TASK_DELIMITER: &str = "⏵ ";
fn task_summary(task: &TaskState, exit_status: Option<ExitStatus>) -> (bool, String, String) {
    let escaped_full_label = task
        .spawned_task
        .full_label
        .replace("\r\n", "\r")
        .replace('\n', "\r");
    let task_label = |suffix: &str| format!("{TASK_DELIMITER}Task `{escaped_full_label}` {suffix}");
    let cancelled = task.structured_task.as_ref().is_some_and(|handle| {
        matches!(handle.state(), task::StructuredTaskState::Cancelled { .. })
    });
    let (success, task_line) = if cancelled {
        (false, task_label("cancelled"))
    } else {
        match exit_status {
            Some(status) => {
                let code = status.code();
                #[cfg(unix)]
                let signal = status.signal();
                #[cfg(not(unix))]
                let signal: Option<i32> = None;

                match (code, signal) {
                    (Some(0), _) => (true, task_label("finished successfully")),
                    (Some(code), _) => (
                        false,
                        task_label(&format!("finished with exit code: {code}")),
                    ),
                    (None, Some(signal)) => (
                        false,
                        task_label(&format!("terminated by signal: {signal}")),
                    ),
                    (None, None) => (false, task_label("finished")),
                }
            }
            None => (false, task_label("finished")),
        }
    };
    let escaped_command_label = task
        .spawned_task
        .command_label
        .replace("\r\n", "\r")
        .replace('\n', "\r");
    let command_line = format!("{TASK_DELIMITER}Command: {escaped_command_label}");
    (success, task_line, command_line)
}

/// Converts bare LFs into CRLFs so output captured from a pipe (rather than a
/// PTY) wraps correctly in Alacritty. A PTY's line discipline performs this
/// `ONLCR` translation for us; piped output (e.g. `ls` run outside a PTY) only
/// emits `\n`, which moves Alacritty's cursor down without returning it to
/// column zero and makes the rendered output look misaligned. Alacritty has no
/// setting for this, so we insert a `\r` before each `\n` that lacks one.
fn convert_lf_to_crlf(bytes: &[u8], previous_byte_was_cr: &mut bool) -> Vec<u8> {
    let mut converted = Vec::with_capacity(bytes.len());
    for &byte in bytes {
        if byte == b'\n' && !*previous_byte_was_cr {
            converted.push(b'\r');
        }
        converted.push(byte);
        *previous_byte_was_cr = byte == b'\r';
    }
    converted
}

/// Owns a non-PTY task subprocess and the background task pumping its output
/// into the terminal emulator. Used by headless hosts (e.g. the eval CLI) where
/// PTY allocation fails with `ENOTTY`. Dropping this kills the child.
struct SubprocessHandle {
    process_id: u32,
    child: Arc<parking_lot::Mutex<Option<util::process::Child>>>,
    _reader: Task<()>,
}

impl SubprocessHandle {
    fn kill(&self) {
        if let Some(child) = self.child.lock().as_mut() {
            child.kill().log_err();
        }
    }
}

/// Spawns `program`/`args` as a plain subprocess with piped stdout/stderr and
/// drives its output into `term`, mirroring what the Alacritty event loop does
/// for a PTY but without one. Used when [`HeadlessTerminal`] is enabled.
fn spawn_task_subprocess(
    program: String,
    args: Vec<String>,
    env: HashMap<String, String>,
    clear_environment: bool,
    working_directory: Option<PathBuf>,
    term: Arc<AlacrittyTermLock>,
    events_tx: futures::channel::mpsc::UnboundedSender<PtyEvent>,
    executor: &BackgroundExecutor,
) -> Result<SubprocessHandle> {
    use futures::io::AsyncReadExt as _;
    use std::process::Stdio;

    let mut command = util::command::new_std_command(&program);
    command.args(&args);
    if clear_environment {
        command.env_clear();
    }
    command.envs(&env);
    if let Some(directory) = &working_directory {
        command.current_dir(directory);
    }

    let mut child =
        util::process::Child::spawn(command, Stdio::null(), Stdio::piped(), Stdio::piped())?;
    let process_id = child.id();
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let child = Arc::new(parking_lot::Mutex::new(Some(child)));

    let reader = executor.spawn({
        let child = child.clone();
        let executor = executor.clone();
        async move {
            // stdout and stderr are pumped concurrently, each through its own
            // parser; the shared term mutex serializes grid mutation.
            type BoxedReader = Box<dyn futures::io::AsyncRead + Unpin + Send>;
            let pump = |reader: Option<BoxedReader>| {
                let term = term.clone();
                let events_tx = events_tx.clone();
                async move {
                    let Some(mut reader) = reader else { return };
                    let mut processor = Processor::<StdSyncHandler>::new();
                    let mut buffer = [0u8; 8192];
                    let mut previous_byte_was_cr = false;
                    loop {
                        match reader.read(&mut buffer).await {
                            Ok(0) => return,
                            Err(error) => {
                                log::warn!("failed to read subprocess output: {error}");
                                return;
                            }
                            Ok(count) => {
                                let converted =
                                    convert_lf_to_crlf(&buffer[..count], &mut previous_byte_was_cr);
                                {
                                    let mut term = term.lock();
                                    processor.advance(&mut *term, &converted);
                                }
                                events_tx
                                    .unbounded_send(PtyEvent::Event(TerminalBackendEvent::Wakeup))
                                    .ok();
                            }
                        }
                    }
                }
            };
            let stdout = stdout.map(|reader| Box::new(reader) as BoxedReader);
            let stderr = stderr.map(|reader| Box::new(reader) as BoxedReader);
            futures::future::join(pump(stdout), pump(stderr)).await;

            // Both pipes are closed, so the child has exited or is about to.
            // Poll for its status without holding the lock across an await.
            let status = loop {
                let status = match child.lock().as_mut() {
                    Some(child) => match child.try_status() {
                        Ok(status) => status,
                        Err(error) => {
                            log::warn!("failed to get subprocess exit status: {error}");
                            break None;
                        }
                    },
                    None => Some(ExitStatus::default()),
                };
                match status {
                    Some(status) => break Some(status),
                    None => executor.timer(Duration::from_millis(20)).await,
                }
            };
            child.lock().take();
            let event = match status {
                Some(status) => TerminalBackendEvent::ChildExit(status),
                None => TerminalBackendEvent::Exit,
            };
            events_tx.unbounded_send(PtyEvent::Event(event)).ok();
        }
    });

    Ok(SubprocessHandle {
        process_id,
        child,
        _reader: reader,
    })
}

impl Drop for Terminal {
    fn drop(&mut self) {
        if let Some(subprocess) = self.subprocess.take() {
            subprocess.kill();
        }
        if let TerminalType::Pty { pty_tx, info } =
            std::mem::replace(&mut self.terminal_type, TerminalType::DisplayOnly)
        {
            pty_tx.shutdown();
            info.terminate_child_process();

            let timer = self.background_executor.timer(Duration::from_millis(100));
            self.background_executor
                .spawn(async move {
                    timer.await;
                    info.kill_child_process();
                })
                .detach();
        }
    }
}

impl EventEmitter<Event> for Terminal {}

fn normalize_path_command_name(command: &str) -> Option<String> {
    const MAX_COMMAND_NAME_LENGTH: usize = 64;

    let command = command.trim();
    if command.is_empty()
        || command.len() > MAX_COMMAND_NAME_LENGTH
        || command.starts_with('.')
        || command.starts_with('-')
        || command.contains('/')
        || command.contains('\\')
    {
        return None;
    }

    let mut command = command.to_ascii_lowercase();
    for suffix in [".exe", ".cmd", ".bat", ".ps1"] {
        if command.ends_with(suffix) {
            command.truncate(command.len() - suffix.len());
            break;
        }
    }

    if command.is_empty()
        || !command.chars().all(|character| {
            character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.')
        })
    {
        return None;
    }

    Some(command)
}

fn foreground_process_command_from_argv(argv: &[String]) -> Option<String> {
    let command = argv
        .first()
        .and_then(|command| normalize_path_command_name(command));

    if !matches!(
        command.as_deref(),
        Some("node" | "python" | "python3" | "bun" | "deno")
    ) {
        return command;
    }

    argv.iter()
        .skip(1)
        .filter_map(|argument| normalize_script_command_name(argument))
        .next()
        .or(command)
}

fn normalize_script_command_name(argument: &str) -> Option<String> {
    let path = Path::new(argument);
    let file_stem = path
        .file_stem()
        .and_then(|file_stem| file_stem.to_str())
        .and_then(normalize_path_command_name)?;

    if file_stem != "index" {
        return Some(file_stem);
    }

    path.parent()
        .and_then(|parent| parent.parent())
        .and_then(|package_path| package_path.file_name())
        .and_then(|package_name| package_name.to_str())
        .and_then(|package_name| package_name.strip_suffix("-cli").or(Some(package_name)))
        .and_then(normalize_path_command_name)
}

fn content_index_for_mouse(pos: GpuiPoint<Pixels>, terminal_bounds: &TerminalBounds) -> usize {
    let col = (pos.x / terminal_bounds.cell_width()).round() as usize;
    let clamped_col = min(col, terminal_bounds.num_columns().saturating_sub(1));
    let row = (pos.y / terminal_bounds.line_height()).round() as usize;
    let clamped_row = min(row, terminal_bounds.num_lines().saturating_sub(1));
    clamped_row * terminal_bounds.num_columns() + clamped_col
}

/// Converts an 8 bit ANSI color to its GPUI equivalent.
/// Accepts `usize` for compatibility with the `alacritty::Colors` interface,
/// Other than that use case, should only be called with values in the `[0,255]` range
pub fn get_color_at_index(index: usize, theme: &Theme) -> Hsla {
    let colors = theme.colors();

    match index {
        // 0-15 are the same as the named colors above
        0 => colors.terminal_ansi_black,
        1 => colors.terminal_ansi_red,
        2 => colors.terminal_ansi_green,
        3 => colors.terminal_ansi_yellow,
        4 => colors.terminal_ansi_blue,
        5 => colors.terminal_ansi_magenta,
        6 => colors.terminal_ansi_cyan,
        7 => colors.terminal_ansi_white,
        8 => colors.terminal_ansi_bright_black,
        9 => colors.terminal_ansi_bright_red,
        10 => colors.terminal_ansi_bright_green,
        11 => colors.terminal_ansi_bright_yellow,
        12 => colors.terminal_ansi_bright_blue,
        13 => colors.terminal_ansi_bright_magenta,
        14 => colors.terminal_ansi_bright_cyan,
        15 => colors.terminal_ansi_bright_white,
        // 16-231 are a 6x6x6 RGB color cube, mapped to 0-255 using steps defined by XTerm.
        // See: https://github.com/xterm-x11/xterm-snapshots/blob/master/256colres.pl
        16..=231 => {
            let (r, g, b) = rgb_for_index(index as u8);
            rgba_color(
                if r == 0 { 0 } else { r * 40 + 55 },
                if g == 0 { 0 } else { g * 40 + 55 },
                if b == 0 { 0 } else { b * 40 + 55 },
            )
        }
        // 232-255 are a 24-step grayscale ramp from (8, 8, 8) to (238, 238, 238).
        232..=255 => {
            let i = index as u8 - 232; // Align index to 0..24
            let value = i * 10 + 8;
            rgba_color(value, value, value)
        }
        // For compatibility with the alacritty::Colors interface
        // See: https://github.com/alacritty/alacritty/blob/master/alacritty_terminal/src/term/color.rs
        256 => colors.terminal_foreground,
        257 => colors.terminal_background,
        258 => theme.players().local().cursor,
        259 => colors.terminal_ansi_dim_black,
        260 => colors.terminal_ansi_dim_red,
        261 => colors.terminal_ansi_dim_green,
        262 => colors.terminal_ansi_dim_yellow,
        263 => colors.terminal_ansi_dim_blue,
        264 => colors.terminal_ansi_dim_magenta,
        265 => colors.terminal_ansi_dim_cyan,
        266 => colors.terminal_ansi_dim_white,
        267 => colors.terminal_bright_foreground,
        268 => colors.terminal_ansi_black, // 'Dim Background', non-standard color

        _ => black(),
    }
}

/// Generates the RGB channels in [0, 5] for a given index into the 6x6x6 ANSI color cube.
///
/// See: [8 bit ANSI color](https://en.wikipedia.org/wiki/ANSI_escape_code#8-bit).
///
/// Wikipedia gives a formula for calculating the index for a given color:
///
/// ```text
/// index = 16 + 36 × r + 6 × g + b (0 ≤ r, g, b ≤ 5)
/// ```
///
/// This function does the reverse, calculating the `r`, `g`, and `b` components from a given index.
fn rgb_for_index(i: u8) -> (u8, u8, u8) {
    debug_assert!((16..=231).contains(&i));
    let i = i - 16;
    let r = (i - (i % 36)) / 36;
    let g = ((i % 36) - (i % 6)) / 6;
    let b = (i % 36) % 6;
    (r, g, b)
}

pub fn rgba_color(r: u8, g: u8, b: u8) -> Hsla {
    Rgba {
        r: (r as f32 / 255.),
        g: (g as f32 / 255.),
        b: (b as f32 / 255.),
        a: 1.,
    }
    .into()
}

#[cfg(test)]
mod tests {
    #[cfg(windows)]
    #[test]
    fn windows_literal_task_uses_captured_search_path_and_directory() {
        let directory = tempfile::tempdir().expect("Task directory");
        let bin = directory.path().join("tool chain");
        std::fs::create_dir(&bin).expect("toolchain directory");
        let executable = bin.join("rust-workspace-test-receiver.exe");
        std::fs::write(&executable, []).expect("path-resolution fixture, not an executable test");
        let mut environment = HashMap::default();
        environment.insert("Path".into(), "tool chain".into());
        assert_eq!(
            resolve_windows_task_program(
                "rust-workspace-test-receiver",
                &environment,
                Some(directory.path()),
            )
            .expect("resolve against captured relative PATH"),
            executable
        );
        environment.clear();
        assert!(
            resolve_windows_task_program(
                "rust-workspace-test-receiver",
                &environment,
                Some(directory.path()),
            )
            .is_err()
        );
        assert_eq!(
            resolve_windows_task_program(
                executable.to_str().expect("fixture Unicode path"),
                &environment,
                Some(directory.path()),
            )
            .expect("explicit executable does not need PATH"),
            executable
        );
        environment.insert("Path".into(), bin.to_string_lossy().into_owned());
        std::fs::write(bin.join("missing-native.exe.CMD"), []).expect("suffix fallback control");
        assert!(
            resolve_windows_task_program(
                "missing-native.exe",
                &environment,
                Some(directory.path()),
            )
            .is_err()
        );
        environment.insert("PATH".into(), bin.to_string_lossy().into_owned());
        assert!(
            resolve_windows_task_program("cargo", &environment, Some(directory.path())).is_err()
        );
        assert!(
            resolve_windows_task_program("cargo.cmd", &HashMap::default(), Some(directory.path()))
                .is_err()
        );
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn native_terminal_exact_environment_spawn_errors_are_private_and_recover(
        cx: &mut TestAppContext,
    ) {
        use util::macos_task_completion::{
            TASK_BUILD_INVOCATION_ENV, TASK_COMPLETION_TOKEN_ENV, TASK_NATIVE_INVOCATION_ENV,
        };
        init_test(cx);
        cx.executor().allow_parking();
        cx.update(|cx| cx.set_global(HeadlessTerminal(true)));
        let directory = tempfile::tempdir().expect("owned actual spawn controls");
        let missing_program = directory.path().join("missing-program");
        let missing_directory = directory.path().join("missing-directory");
        let nonce = "a4".repeat(32);
        let envelope = format!("private-native-envelope-{nonce}");
        let private_value = "private-complete-map-value";
        let mut scenarios = 0;
        for (exact, invocation_environment_name) in [
            (false, Some(TASK_BUILD_INVOCATION_ENV)),
            (true, Some(TASK_NATIVE_INVOCATION_ENV)),
            (false, None),
        ] {
            let mut controlled = HashMap::from_iter([
                (
                    "NATIVE_TERMINAL_PRIVATE_SPAWN_VALUE".into(),
                    private_value.into(),
                ),
                (TASK_COMPLETION_TOKEN_ENV.into(), nonce.clone()),
            ]);
            if let Some(invocation_environment_name) = invocation_environment_name {
                controlled.insert(invocation_environment_name.into(), envelope.clone());
            }
            for (program, working_directory, succeeds) in [
                (
                    missing_program.clone(),
                    directory.path().to_path_buf(),
                    false,
                ),
                (
                    PathBuf::from("/usr/bin/true"),
                    missing_directory.clone(),
                    false,
                ),
                (
                    PathBuf::from("/usr/bin/true"),
                    directory.path().to_path_buf(),
                    true,
                ),
            ] {
                let (completion_tx, completion_rx) = async_channel::unbounded();
                let builder = cx
                    .update(|cx| {
                        let program = program.to_str().expect("owned literal program").to_owned();
                        let task = TaskState {
                            status: TaskStatus::Running,
                            completion_rx: completion_rx.clone(),
                            spawned_task: SpawnInTerminal {
                                command: Some(program.clone()),
                                env: controlled.clone(),
                                ..Default::default()
                            },
                            structured_task: None,
                        };
                        if exact {
                            let mut semantic = controlled.clone();
                            semantic.remove(TASK_COMPLETION_TOKEN_ENV);
                            semantic.remove(TASK_NATIVE_INVOCATION_ENV);
                            let mut complete = complete_native_terminal_environment_with_host(
                                HashMap::default(),
                                &semantic,
                                &release_channel::AppVersion::global(cx),
                                42,
                                false,
                                false,
                            )
                            .expect("complete controlled lower-level data");
                            complete.insert(TASK_COMPLETION_TOKEN_ENV.into(), nonce.clone());
                            complete.insert(TASK_NATIVE_INVOCATION_ENV.into(), envelope.clone());
                            let environment =
                                NativeTerminalEnvironment::for_spawn(complete, 42, false)
                                    .expect("bounded exact spawn data, not native authority");
                            TerminalBuilder::new_with_exact_native_environment(
                                working_directory,
                                task,
                                program,
                                Vec::new(),
                                environment,
                                SettingsCursorShape::default(),
                                AlternateScroll::On,
                                None,
                                Vec::new(),
                                0,
                                42,
                                Some(completion_tx),
                                cx,
                                PathStyle::local(),
                            )
                        } else {
                            TerminalBuilder::new(
                                Some(working_directory),
                                Some(task),
                                Shell::WithArguments {
                                    program,
                                    args: Vec::new(),
                                    title_override: None,
                                },
                                controlled.clone(),
                                SettingsCursorShape::default(),
                                AlternateScroll::On,
                                None,
                                Vec::new(),
                                0,
                                false,
                                42,
                                Some(completion_tx),
                                cx,
                                Vec::new(),
                                PathStyle::local(),
                            )
                        }
                    })
                    .await;
                scenarios += 1;
                if succeeds {
                    let builder = builder.expect("fresh actual child recovery");
                    let terminal = cx.new(|cx| builder.subscribe(cx));
                    let wait = completion_rx.recv();
                    let timeout = cx.background_executor.timer(Duration::from_secs(30));
                    futures::pin_mut!(wait, timeout);
                    let status = match futures::future::select(wait, timeout).await {
                        futures::future::Either::Left((status, _)) => status
                            .expect("actual completion channel")
                            .expect("actual kernel child status"),
                        futures::future::Either::Right(_) => {
                            terminal.update(cx, |terminal, _| terminal.kill_active_task());
                            panic!("fresh exact child exceeded original completion deadline");
                        }
                    };
                    assert_eq!(status.code(), Some(0));
                    terminal.read_with(cx, |terminal, _| {
                        if exact {
                            assert!(
                                terminal
                                    .task()
                                    .expect("actual recovered Task")
                                    .spawned_task
                                    .env
                                    .is_empty()
                            );
                            assert!(terminal.template.env.is_empty());
                        } else if invocation_environment_name.is_some() {
                            assert!(
                                terminal
                                    .task()
                                    .expect("actual recovered inherited Task")
                                    .spawned_task
                                    .env
                                    .get(TASK_BUILD_INVOCATION_ENV)
                                    == Some(&envelope),
                                "existing inherited Task data must not be silently rewritten"
                            );
                        } else {
                            let environment = &terminal
                                .task()
                                .expect("actual recovered token-only inherited Task")
                                .spawned_task
                                .env;
                            assert_eq!(environment.get(TASK_COMPLETION_TOKEN_ENV), Some(&nonce));
                            assert_eq!(
                                environment
                                    .get("NATIVE_TERMINAL_PRIVATE_SPAWN_VALUE")
                                    .map(String::as_str),
                                Some(private_value),
                            );
                            assert!(!environment.contains_key(TASK_BUILD_INVOCATION_ENV));
                            assert!(!environment.contains_key(TASK_NATIVE_INVOCATION_ENV));
                        }
                    });
                } else {
                    let error = match builder {
                        Ok(_) => panic!("missing executable or cwd unexpectedly spawned"),
                        Err(error) => error,
                    };
                    let terminal_error = error
                        .downcast_ref::<TerminalError>()
                        .expect("actual Terminal spawn refusal");
                    for diagnostic in [format!("{error:#}"), format!("{terminal_error:?}")] {
                        assert!(
                            diagnostic.len() <= 2048,
                            "bounded actual OS spawn diagnostic"
                        );
                        for private in [
                            "NATIVE_TERMINAL_PRIVATE_SPAWN_VALUE",
                            private_value,
                            TASK_COMPLETION_TOKEN_ENV,
                            TASK_BUILD_INVOCATION_ENV,
                            TASK_NATIVE_INVOCATION_ENV,
                            nonce.as_str(),
                            envelope.as_str(),
                        ] {
                            assert!(
                                !diagnostic.contains(private),
                                "private spawn data escaped in error"
                            );
                        }
                    }
                    assert_eq!(terminal_error.source.kind(), std::io::ErrorKind::NotFound);
                    assert_eq!(terminal_error.source.raw_os_error(), Some(libc::ENOENT));
                    assert!(
                        completion_rx.try_recv().is_err(),
                        "refusal must not fabricate completion"
                    );
                }
            }
        }
        assert_eq!(scenarios, 9);
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn native_terminal_exact_environment_actual_pty_isolated(
        cx: &mut TestAppContext,
        rng: rand::rngs::StdRng,
    ) {
        assert_native_terminal_isolated_children(false, rng, cx).await;
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn native_terminal_exact_environment_actual_pipe_isolated(
        cx: &mut TestAppContext,
        rng: rand::rngs::StdRng,
    ) {
        assert_native_terminal_isolated_children(true, rng, cx).await;
    }

    #[cfg(target_os = "macos")]
    async fn assert_native_terminal_isolated_children(
        no_pty: bool,
        mut rng: rand::rngs::StdRng,
        cx: &mut TestAppContext,
    ) {
        use rand::RngCore as _;
        use std::process::Stdio;
        cx.executor().allow_parking();
        for exact in [false, true] {
            for exit_code in [0, 7] {
                let directory = tempfile::tempdir().expect("owned isolated child directory");
                let stdout = std::fs::File::create(directory.path().join("controller.stdout"))
                    .expect("private child stdout");
                let stderr = std::fs::File::create(directory.path().join("controller.stderr"))
                    .expect("private child stderr");
                let mut command = util::command::new_std_command(
                    std::env::current_exe().expect("actual current test image"),
                );
                command.env_clear();
                command.env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin");
                command.env("USER", "native-terminal-fixture");
                command.env("HOME", directory.path());
                command.env("TMPDIR", directory.path());
                command.env("SHELL", "/bin/sh");
                command.env("LANG", "en_US.UTF-8");
                command.env("RUST_BACKTRACE", "0");
                if let Some(value) = std::env::var_os("DYLD_FALLBACK_LIBRARY_PATH") {
                    let value = value
                        .into_string()
                        .expect("bounded UTF-8 actual test-loader path");
                    assert!(value.len() <= 4096 && !value.contains('\0'));
                    for path in value.split(':') {
                        assert!(
                            !path.is_empty() && Path::new(path).is_absolute(),
                            "test-loader paths must be explicit absolute directories"
                        );
                        match std::fs::metadata(path) {
                            Ok(metadata) => assert!(metadata.is_dir()),
                            // Cargo includes default fallback paths that dyld skips when absent.
                            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                            Err(_) => panic!("actual test-loader directory could not be inspected"),
                        }
                    }
                    command.env("DYLD_FALLBACK_LIBRARY_PATH", value);
                }
                let controller_seed = rng.next_u64();
                command.env("SEED", controller_seed.to_string());
                command.args([
                    "--exact",
                    "tests::native_terminal_exact_environment_isolated_controller",
                    "--nocapture",
                ]);
                command.env(
                    "NATIVE_TERMINAL_CONTROLLER",
                    if exact { "exact" } else { "legacy" },
                );
                command.env(
                    "NATIVE_TERMINAL_PIPE",
                    if no_pty { "true" } else { "false" },
                );
                command.env("NATIVE_TERMINAL_EXIT", exit_code.to_string());
                command.env("NATIVE_TERMINAL_OWNED_ROOT", directory.path());
                command.env(
                    "NATIVE_TERMINAL_INHERITED_SENTINEL",
                    "parent-only-private-control",
                );
                command.env("ITERATIONS", "1");
                let mut child = util::process::Child::spawn(
                    command,
                    Stdio::null(),
                    Stdio::from(stdout),
                    Stdio::from(stderr),
                )
                .expect("isolated actual child parent");
                let status = {
                    let wait = child.status();
                    let timeout = cx.background_executor.timer(Duration::from_secs(60));
                    futures::pin_mut!(wait, timeout);
                    match futures::future::select(wait, timeout).await {
                        futures::future::Either::Left((status, _)) => {
                            Some(status.expect("actual child parent exit"))
                        }
                        futures::future::Either::Right(_) => None,
                    }
                };
                if status.is_none() {
                    child.kill().expect("stop only owned isolated child group");
                    child.status().await.expect("reap isolated child group");
                }
                if !status.is_some_and(|status| status.success()) {
                    let refusal = std::fs::read(directory.path().join("receiver.refused"));
                    let classification = match refusal.as_deref() {
                        Ok(b"inherited-sentinel") => "inherited-sentinel",
                        Ok(b"complete-map-mismatch") => "complete-map-mismatch",
                        _ => "unclassified-child-failure",
                    };
                    let diagnostic_directory = directory.keep();
                    panic!(
                        "isolated lower Terminal control failed ({classification}); private diagnostics retained at {}",
                        diagnostic_directory.display()
                    );
                }
                eprintln!(
                    "native-terminal-exact-child backend={} mode={} controller_seed={} controller_iterations=1 exit={}",
                    if no_pty { "pipe" } else { "pty" },
                    if exact { "exact" } else { "legacy" },
                    controller_seed,
                    exit_code
                );
                assert_eq!(
                    std::fs::read(directory.path().join("receiver.verified"))
                        .expect("actual receiver marker"),
                    b"verified"
                );
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn native_terminal_exact_environment_isolated_controller(cx: &mut TestAppContext) {
        let Ok(mode) = std::env::var("NATIVE_TERMINAL_CONTROLLER") else {
            return;
        };
        assert!(matches!(mode.as_str(), "exact" | "legacy"));
        let exact = mode == "exact";
        let no_pty = std::env::var("NATIVE_TERMINAL_PIPE").expect("isolated backend") == "true";
        let exit_code: i32 = std::env::var("NATIVE_TERMINAL_EXIT")
            .expect("isolated status")
            .parse()
            .expect("bounded status");
        assert!(matches!(exit_code, 0 | 7));
        let directory = std::fs::canonicalize(PathBuf::from(
            std::env::var_os("NATIVE_TERMINAL_OWNED_ROOT").expect("owned parent directory"),
        ))
        .expect("actual owned canonical parent directory");
        assert!(directory.is_absolute());
        assert_eq!(
            std::env::var("NATIVE_TERMINAL_INHERITED_SENTINEL").expect("actual inherited sentinel"),
            "parent-only-private-control"
        );
        init_test(cx);
        cx.executor().allow_parking();
        cx.update(|cx| cx.set_global(HeadlessTerminal(no_pty)));
        let executable = std::env::current_exe()
            .expect("actual receiver image")
            .to_str()
            .expect("literal UTF-8 receiver")
            .to_owned();
        let receiver_args = vec![
            "--exact".into(),
            "tests::native_terminal_exact_environment_actual_receiver".into(),
            "--nocapture".into(),
        ];
        let mut overlay = HashMap::from_iter([
            ("NATIVE_TERMINAL_RECEIVER".into(), mode),
            (
                "NATIVE_TERMINAL_RECEIVER_ROOT".into(),
                directory.to_str().expect("owned UTF-8 root").into(),
            ),
            (
                "NATIVE_TERMINAL_RECEIVER_EXIT".into(),
                exit_code.to_string(),
            ),
            (
                "NATIVE_TERMINAL_OWNED_VALUE".into(),
                "literal owned control".into(),
            ),
        ]);
        let (completion_tx, completion_rx) = async_channel::unbounded();
        let builder = cx
            .update(|cx| {
                let (program, arguments) = if exact {
                    (executable.clone(), receiver_args.clone())
                } else {
                    let (program, mut arguments) =
                        ShellBuilder::new(&Shell::Program("/bin/sh".into()), false)
                            .build_with_literal_prefix(
                                Some(executable.clone()),
                                &receiver_args,
                                receiver_args.len(),
                            );
                    let script = arguments.last_mut().expect("actual authored shell command");
                    script.insert_str(
                        0,
                        "printf legacy >\"$NATIVE_TERMINAL_RECEIVER_ROOT/shell.started\"; ",
                    );
                    (program, arguments)
                };
                let task = TaskState {
                    status: TaskStatus::Running,
                    completion_rx: completion_rx.clone(),
                    spawned_task: SpawnInTerminal {
                        command: Some(program.clone()),
                        args: arguments.clone(),
                        env: overlay.clone(),
                        ..Default::default()
                    },
                    structured_task: None,
                };
                assert!(!task.spawned_task.env.is_empty());
                if exact {
                    let mut complete = complete_native_terminal_environment(&overlay, 42, cx)
                        .expect("complete lower-level spawn data");
                    complete.remove("NATIVE_TERMINAL_INHERITED_SENTINEL");
                    overlay.clear();
                    std::fs::write(
                        directory.join("expected.private.json"),
                        serde_json::to_vec(&complete).expect("private expected map"),
                    )
                    .expect("owned private map file");
                    let environment = NativeTerminalEnvironment::for_spawn(complete, 42, !no_pty)
                        .expect("bounded exact spawn data; not Project authority");
                    TerminalBuilder::new_with_exact_native_environment(
                        directory.clone(),
                        task,
                        program,
                        arguments,
                        environment,
                        SettingsCursorShape::default(),
                        AlternateScroll::On,
                        None,
                        Vec::new(),
                        0,
                        42,
                        Some(completion_tx),
                        cx,
                        PathStyle::local(),
                    )
                } else {
                    TerminalBuilder::new(
                        Some(directory.clone()),
                        Some(task),
                        Shell::WithArguments {
                            program,
                            args: arguments,
                            title_override: None,
                        },
                        overlay,
                        SettingsCursorShape::default(),
                        AlternateScroll::On,
                        None,
                        Vec::new(),
                        0,
                        false,
                        42,
                        Some(completion_tx),
                        cx,
                        Vec::new(),
                        PathStyle::local(),
                    )
                }
            })
            .await
            .expect("actual existing Terminal child owner");
        let terminal = cx.new(|cx| builder.subscribe(cx));
        let status = {
            let wait = completion_rx.recv();
            let timeout = cx.background_executor.timer(Duration::from_secs(30));
            futures::pin_mut!(wait, timeout);
            match futures::future::select(wait, timeout).await {
                futures::future::Either::Left((result, _)) => result
                    .expect("real Terminal completion")
                    .expect("real kernel child status"),
                futures::future::Either::Right(_) => {
                    terminal.update(cx, |terminal, _| terminal.kill_active_task());
                    panic!(
                        "owned lower Terminal child failed to complete within original fixture deadline"
                    );
                }
            }
        };
        assert_eq!(status.code(), Some(exit_code));
        terminal.update(cx, |terminal, _| {
            assert_eq!(terminal.is_pty(), !no_pty);
            if exact {
                assert!(
                    terminal
                        .task()
                        .expect("real Task terminal")
                        .spawned_task
                        .env
                        .is_empty()
                );
                assert!(terminal.template.env.is_empty());
                assert!(matches!(terminal.template.shell, Shell::System));
            }
        });
        assert_eq!(directory.join("shell.started").exists(), !exact);
        assert_eq!(
            std::fs::read(directory.join("receiver.verified"))
                .expect("actual receiver completed assertions"),
            b"verified"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_terminal_exact_environment_actual_receiver() {
        let Ok(mode) = std::env::var("NATIVE_TERMINAL_RECEIVER") else {
            return;
        };
        let directory = PathBuf::from(
            std::env::var_os("NATIVE_TERMINAL_RECEIVER_ROOT").expect("owned receiver directory"),
        );
        assert_eq!(
            std::env::current_dir().expect("actual child cwd"),
            directory
        );
        assert_eq!(
            std::env::var("NATIVE_TERMINAL_OWNED_VALUE").expect("positive exact-map entry"),
            "literal owned control"
        );
        match mode.as_str() {
            "exact" => {
                if std::env::var_os("NATIVE_TERMINAL_INHERITED_SENTINEL").is_some() {
                    std::fs::write(directory.join("receiver.refused"), b"inherited-sentinel")
                        .expect("owned bounded refusal marker");
                }
                assert!(std::env::var_os("NATIVE_TERMINAL_INHERITED_SENTINEL").is_none());
                let expected: HashMap<String, String> = serde_json::from_slice(
                    &std::fs::read(directory.join("expected.private.json"))
                        .expect("owned expected data"),
                )
                .expect("bounded private expected map");
                let actual: HashMap<String, String> = std::env::vars().collect();
                if actual != expected {
                    std::fs::write(directory.join("receiver.refused"), b"complete-map-mismatch")
                        .expect("owned bounded refusal marker");
                }
                assert!(
                    actual == expected,
                    "actual exact child map differs; never print private map values"
                );
                assert!(!directory.join("shell.started").exists());
            }
            "legacy" => {
                assert_eq!(
                    std::env::var("NATIVE_TERMINAL_INHERITED_SENTINEL")
                        .expect("legacy inherited positive control"),
                    "parent-only-private-control"
                );
                assert!(directory.join("shell.started").exists());
            }
            _ => panic!("unsupported isolated receiver mode"),
        }
        std::fs::write(directory.join("receiver.verified"), b"verified")
            .expect("owned verified marker");
        let exit_code: i32 = std::env::var("NATIVE_TERMINAL_RECEIVER_EXIT")
            .expect("actual receiver status")
            .parse()
            .expect("bounded exit status");
        assert!(matches!(exit_code, 0 | 7));
        std::process::exit(exit_code);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_terminal_exact_environment_bounds_transport_separately() {
        use util::macos_task_completion::{
            MAX_TASK_BUILD_INVOCATION_BYTES, TASK_BUILD_INVOCATION_ENV, TASK_COMPLETION_TOKEN_ENV,
            TASK_NATIVE_INVOCATION_ENV,
        };
        let mut environment = HashMap::from_iter([
            (TASK_COMPLETION_TOKEN_ENV.into(), "42".repeat(32)),
            (
                TASK_BUILD_INVOCATION_ENV.into(),
                "x".repeat(MAX_TASK_BUILD_INVOCATION_BYTES),
            ),
        ]);
        let prepared = NativeTerminalEnvironment::for_spawn(environment.clone(), 42, false)
            .expect("bounded transport envelope");
        assert_eq!(prepared.environment, environment);
        environment.insert("SEMANTIC".into(), "x".repeat(4097));
        assert!(NativeTerminalEnvironment::for_spawn(environment.clone(), 42, false).is_err());
        environment.remove("SEMANTIC");
        environment.insert(
            TASK_BUILD_INVOCATION_ENV.into(),
            "x".repeat(MAX_TASK_BUILD_INVOCATION_BYTES + 1),
        );
        assert!(NativeTerminalEnvironment::for_spawn(environment.clone(), 42, false).is_err());
        environment.insert(TASK_BUILD_INVOCATION_ENV.into(), "bounded".into());
        environment.insert(TASK_NATIVE_INVOCATION_ENV.into(), "conflicting".into());
        assert!(NativeTerminalEnvironment::for_spawn(environment.clone(), 42, false).is_err());
        environment.remove(TASK_NATIVE_INVOCATION_ENV);
        environment.insert(TASK_COMPLETION_TOKEN_ENV.into(), "not-a-token".into());
        assert!(NativeTerminalEnvironment::for_spawn(environment, 42, false).is_err());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_terminal_exact_environment_rejects_backend_drift_without_repair() {
        let complete = complete_native_terminal_environment_with_host(
            HashMap::from_iter([
                ("USER".into(), "host-user".into()),
                ("HOME".into(), "/host/home".into()),
            ]),
            &HashMap::default(),
            &"1.2.3",
            42,
            true,
            false,
        )
        .expect("complete PTY recipe");
        let prepared =
            NativeTerminalEnvironment::for_spawn(complete.clone(), 42, true).expect("bounded map");
        assert!(prepared.validate_backend(42, true, &"1.2.3").is_ok());
        assert!(prepared.validate_backend(43, true, &"1.2.3").is_err());
        assert!(prepared.validate_backend(42, false, &"1.2.3").is_err());
        assert!(prepared.validate_backend(42, true, &"1.2.4").is_err());
        assert_eq!(prepared.environment, complete);
        let mut missing_host = complete;
        missing_host.remove("HOME");
        let prepared = NativeTerminalEnvironment::for_spawn(missing_host, 42, true)
            .expect("bounded but incomplete map");
        assert!(prepared.validate_backend(42, true, &"1.2.3").is_err());
        assert!(!prepared.environment.contains_key("HOME"));
    }

    #[test]
    fn native_terminal_environment_retains_inheritance_and_runtime() {
        let inherited = HashMap::from_iter([
            ("PATH".into(), "/host/bin".into()),
            ("HOME".into(), "/host/home".into()),
            ("USER".into(), "host-user".into()),
            ("INHERITED_ONLY".into(), "retained".into()),
            ("SHLVL".into(), "7".into()),
            ("XDG_ACTIVATION_TOKEN".into(), "private-startup".into()),
            ("DESKTOP_STARTUP_ID".into(), "private-startup".into()),
            ("ZED_PROFILE_TASK_TOKEN".into(), "private-token".into()),
            (
                "ZED_PROFILE_BUILD_INVOCATION_ENV".into(),
                "private-envelope".into(),
            ),
        ]);
        let overrides = HashMap::from_iter([
            ("PATH".into(), "/settings/bin".into()),
            ("HOME".into(), "/settings/home".into()),
            ("TERM".into(), "ignored-custom-term".into()),
        ]);
        let actual = complete_native_terminal_environment_with_host(
            inherited, &overrides, &"1.2.3", 42, true, false,
        )
        .expect("complete eligible map");
        let expected = HashMap::from_iter([
            ("PATH".into(), "/settings/bin".into()),
            ("HOME".into(), "/settings/home".into()),
            ("USER".into(), "host-user".into()),
            ("INHERITED_ONLY".into(), "retained".into()),
            ("LANG".into(), "en_US.UTF-8".into()),
            ("ALACRITTY_WINDOW_ID".into(), "42".into()),
            ("WINDOWID".into(), "42".into()),
            ("ZED_TERM".into(), "true".into()),
            ("TERM_PROGRAM".into(), "zed".into()),
            ("TERM_PROGRAM_VERSION".into(), "1.2.3".into()),
            ("TERM".into(), "xterm-256color".into()),
            ("COLORTERM".into(), "truecolor".into()),
        ]);
        assert_eq!(actual, expected, "all issued keys, not a required subset");
    }

    #[test]
    fn native_terminal_environment_preserves_headless_and_locale_semantics() {
        let inherited = HashMap::from_iter([
            ("LANG".into(), "host.UTF-8".into()),
            ("DESKTOP_STARTUP_ID".into(), "headless-inherited".into()),
        ]);
        let overrides = HashMap::from_iter([("LANG".into(), "task.UTF-8".into())]);
        let actual = complete_native_terminal_environment_with_host(
            inherited, &overrides, &"1.2.3", 42, false, true,
        )
        .expect("headless map needs no passwd fallback");
        assert_eq!(actual.get("LANG").map(String::as_str), Some("task.UTF-8"));
        assert_eq!(
            actual.get("DESKTOP_STARTUP_ID").map(String::as_str),
            Some("headless-inherited")
        );
        assert!(!actual.contains_key("ALACRITTY_WINDOW_ID"));
        assert!(!actual.contains_key("WINDOWID"));
        let mut ordinary = overrides;
        ordinary.insert("SHLVL".into(), "5".into());
        apply_terminal_runtime_environment(&mut ordinary, &"1.2.3", true);
        for (key, value) in ordinary {
            assert_eq!(actual.get(&key), Some(&value), "shared runtime owner");
        }
    }

    #[test]
    fn native_terminal_environment_rejects_unresolved_and_oversized_maps() {
        let missing_home = HashMap::from_iter([("USER".into(), "host-user".into())]);
        assert!(
            complete_native_terminal_environment_with_host(
                missing_home,
                &HashMap::default(),
                &"1.2.3",
                42,
                true,
                false,
            )
            .is_err()
        );
        let invalid = HashMap::from_iter([("BAD=KEY".into(), "value".into())]);
        assert!(validate_native_terminal_environment(&invalid).is_err());
        let too_large = HashMap::from_iter([("KEY".into(), "x".repeat(4097))]);
        assert!(validate_native_terminal_environment(&too_large).is_err());
        let too_many = (0..513)
            .map(|index| (format!("KEY{index}"), "value".into()))
            .collect();
        assert!(validate_native_terminal_environment(&too_many).is_err());
        let too_many_bytes = (0..17)
            .map(|index| (format!("KEY{index}"), "x".repeat(4096)))
            .collect();
        assert!(validate_native_terminal_environment(&too_many_bytes).is_err());
    }

    use std::time::Duration;

    use super::*;
    use crate::{
        Cell, Content, IndexedCell, TerminalBounds, TerminalBuilder, content_index_for_mouse,
        rgb_for_index,
    };
    use async_channel::Receiver;
    use collections::HashMap;
    use gpui::MouseMoveEvent;
    use gpui::{
        ClipboardItem, Entity, Modifiers, MouseButton, MouseDownEvent, MouseUpEvent, Pixels,
        TestAppContext, bounds, point, size,
    };
    use parking_lot::Mutex;
    use rand::{Rng, distr, rngs::StdRng};
    use task::{
        Shell, ShellBuilder, StructuredTaskHandle, StructuredTaskState, StructuredTerminalId,
        TaskId,
    };

    #[test]
    fn test_init_command_startup_marker_commands_do_not_contain_marker() {
        let marker_id = 42;
        let marker = init_command_startup_marker(marker_id);

        for shell_kind in [
            ShellKind::Posix,
            ShellKind::Csh,
            ShellKind::Tcsh,
            ShellKind::Rc,
            ShellKind::Fish,
            ShellKind::PowerShell,
            ShellKind::Pwsh,
            ShellKind::Nushell,
            ShellKind::Cmd,
            ShellKind::Xonsh,
            ShellKind::Elvish,
        ] {
            let command = init_command_startup_marker_command(shell_kind, marker_id);
            assert!(
                !command.contains(&marker),
                "startup marker command for {shell_kind:?} should not contain the full marker, got {command:?}"
            );
        }
    }

    #[gpui::test]
    async fn test_init_command_startup_marker_ignores_echoed_command(cx: &mut TestAppContext) {
        let terminal = cx.new(|cx| {
            TerminalBuilder::new_display_only(
                SettingsCursorShape::default(),
                AlternateScroll::On,
                None,
                0,
                cx.background_executor(),
                PathStyle::local(),
            )
            .subscribe(cx)
        });
        let marker_id = 4242;
        let marker = init_command_startup_marker(marker_id);
        let command = init_command_startup_marker_command(ShellKind::Posix, marker_id);
        let (startup_tx, startup_rx) = async_channel::bounded(1);

        terminal.update(cx, |terminal, cx| {
            terminal.init_command_startup_marker = Some(marker.clone());
            terminal.init_command_startup_tx = Some(startup_tx);
            terminal.write_output(command.as_bytes(), cx);
        });
        assert!(matches!(
            startup_rx.try_recv(),
            Err(async_channel::TryRecvError::Empty)
        ));

        terminal.update(cx, |terminal, cx| {
            terminal.write_output(marker.as_bytes(), cx);
        });
        assert!(startup_rx.try_recv().is_ok());
    }

    #[test]
    fn test_normalize_path_command_name() {
        assert_eq!(normalize_path_command_name("claude"), Some("claude".into()));
        assert_eq!(normalize_path_command_name("Cargo"), Some("cargo".into()));
        assert_eq!(normalize_path_command_name("node.exe"), Some("node".into()));
        assert_eq!(
            normalize_path_command_name("my-agent_cli.1"),
            Some("my-agent_cli.1".into())
        );
        assert_eq!(normalize_path_command_name("./local-agent"), None);
        assert_eq!(normalize_path_command_name("../local-agent"), None);
        assert_eq!(normalize_path_command_name("/usr/local/bin/cargo"), None);
        assert_eq!(
            normalize_path_command_name("target\\debug\\agent.exe"),
            None
        );
        assert_eq!(normalize_path_command_name(".hidden-agent"), None);
        assert_eq!(normalize_path_command_name("agent with spaces"), None);
        assert_eq!(normalize_path_command_name("zsh"), Some("zsh".into()));
        assert_eq!(normalize_path_command_name("-zsh"), None);
        assert_eq!(normalize_path_command_name("pwsh.exe"), Some("pwsh".into()));
    }

    #[test]
    fn test_foreground_process_command_from_interpreter_wrapper() {
        assert_eq!(
            foreground_process_command_from_argv(&[
                "node".to_string(),
                "/opt/homebrew/lib/node_modules/@google/gemini-cli/dist/index.js".to_string(),
            ]),
            Some("gemini".to_string())
        );
        assert_eq!(
            foreground_process_command_from_argv(&[
                "python3".to_string(),
                "/Users/me/.local/bin/codex.py".to_string(),
            ]),
            Some("codex".to_string())
        );
        assert_eq!(
            foreground_process_command_from_argv(&[
                "node".to_string(),
                "/Users/me/private-project/scripts/customer-data-export.js".to_string(),
            ]),
            Some("customer-data-export".to_string())
        );
    }

    #[cfg(not(target_os = "windows"))]
    fn init_test(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
    }

    /// Helper to build a test terminal running a shell command.
    /// Returns the terminal entity and a receiver for the completion signal.
    async fn build_test_terminal(
        cx: &mut TestAppContext,
        command: &str,
        args: &[&str],
    ) -> (Entity<Terminal>, Receiver<Option<ExitStatus>>) {
        let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
        let (program, args) =
            ShellBuilder::new(&Shell::System, false).build(Some(command.to_owned()), &args);
        build_test_terminal_with_arguments(cx, program, args).await
    }

    async fn build_test_terminal_with_arguments(
        cx: &mut TestAppContext,
        program: String,
        args: Vec<String>,
    ) -> (Entity<Terminal>, Receiver<Option<ExitStatus>>) {
        let (completion_tx, completion_rx) = async_channel::unbounded();
        let builder = cx
            .update(|cx| {
                TerminalBuilder::new(
                    None,
                    None,
                    task::Shell::WithArguments {
                        program,
                        args,
                        title_override: None,
                    },
                    HashMap::default(),
                    SettingsCursorShape::default(),
                    AlternateScroll::On,
                    None,
                    vec![],
                    0,
                    false,
                    0,
                    Some(completion_tx),
                    cx,
                    vec![],
                    PathStyle::local(),
                )
            })
            .await
            .unwrap();
        let terminal = cx.new(|cx| builder.subscribe(cx));
        (terminal, completion_rx)
    }

    #[cfg(not(target_os = "windows"))]
    async fn build_test_task_terminal(
        cx: &mut TestAppContext,
        program: String,
        args: Vec<String>,
        no_pty: bool,
    ) -> (Entity<Terminal>, Receiver<Option<ExitStatus>>) {
        let (completion_tx, completion_rx) = async_channel::unbounded();
        let task_state = TaskState {
            status: TaskStatus::Running,
            completion_rx: completion_rx.clone(),
            spawned_task: SpawnInTerminal {
                command: Some(program.clone()),
                args: args.clone(),
                ..Default::default()
            },
            structured_task: None,
        };
        let builder = cx
            .update(|cx| {
                cx.set_global(HeadlessTerminal(no_pty));
                TerminalBuilder::new(
                    None,
                    Some(task_state),
                    task::Shell::WithArguments {
                        program,
                        args,
                        title_override: None,
                    },
                    HashMap::default(),
                    SettingsCursorShape::default(),
                    AlternateScroll::On,
                    None,
                    vec![],
                    0,
                    false,
                    0,
                    Some(completion_tx),
                    cx,
                    vec![],
                    PathStyle::local(),
                )
            })
            .await
            .unwrap();
        let terminal = cx.new(|cx| builder.subscribe(cx));
        (terminal, completion_rx)
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn task_process_id_refuses_actual_headless_wrappers(cx: &mut TestAppContext) {
        use std::os::unix::fs::PermissionsExt;

        init_test(cx);
        cx.executor().allow_parking();
        cx.update(|cx| cx.set_global(HeadlessTerminal(true)));
        let directory = tempfile::tempdir().expect("Owned wrapper controls");
        let receiver = directory.path().join("actual task receiver");
        std::fs::write(
            &receiver,
            "#!/bin/sh\nprintf '%s\\000' \"$DAP_ACTIVATED\" \"$#\" \"$$\" \"${2-}\" > \"$1\"\nexit 7\n",
        )
        .expect("Write actual wrapper receiver");
        std::fs::set_permissions(&receiver, std::fs::Permissions::from_mode(0o700))
            .expect("Owned receiver executable");
        let mut cases = 0;
        for role in ["direct", "activation", "remote-marked", "changed-argv"] {
            let receipt = directory.path().join(format!("{role}.receipt"));
            let command = receiver.to_string_lossy().into_owned();
            let requested_arguments = vec![receipt.to_string_lossy().into_owned()];
            let mut actual_arguments = requested_arguments.clone();
            let activation = if role == "activation" {
                vec!["export DAP_ACTIVATED=wrapped".to_owned()]
            } else {
                Vec::new()
            };
            let shell = if matches!(role, "activation" | "remote-marked") {
                let prefix = if role == "activation" {
                    "export DAP_ACTIVATED=wrapped; "
                } else {
                    ""
                };
                Shell::WithArguments {
                    program: "/bin/sh".into(),
                    args: vec![
                        "-c".into(),
                        format!("{prefix}\"$1\" \"$2\"; status=$?; exit \"$status\""),
                        "owned-wrapper".into(),
                        command.clone(),
                        receipt.to_string_lossy().into_owned(),
                    ],
                    title_override: None,
                }
            } else {
                if role == "changed-argv" {
                    actual_arguments.push("unexpected operand".into());
                }
                Shell::WithArguments {
                    program: command.clone(),
                    args: actual_arguments,
                    title_override: None,
                }
            };
            let (completion_tx, completion_rx) = async_channel::unbounded();
            let task = TaskState {
                spawned_task: SpawnInTerminal {
                    command: Some(command),
                    args: requested_arguments,
                    ..Default::default()
                },
                status: TaskStatus::Running,
                completion_rx,
                structured_task: None,
            };
            let builder = cx
                .update(|cx| {
                    TerminalBuilder::new(
                        Some(directory.path().to_path_buf()),
                        Some(task),
                        shell,
                        HashMap::from_iter([("DAP_ACTIVATED".into(), "off".into())]),
                        SettingsCursorShape::default(),
                        AlternateScroll::On,
                        None,
                        vec![],
                        0,
                        role == "remote-marked",
                        0,
                        Some(completion_tx),
                        cx,
                        activation,
                        PathStyle::local(),
                    )
                })
                .await
                .expect("Actual existing TerminalBuilder child");
            let terminal = cx.new(|cx| builder.subscribe(cx));
            let completion =
                terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
            let deadline = cx.background_executor.timer(Duration::from_secs(10));
            let status = match futures::future::select(Box::pin(completion), Box::pin(deadline))
                .await
            {
                futures::future::Either::Left((status, _)) => {
                    status.expect("Actual wrapper Task exit")
                }
                futures::future::Either::Right(_) => {
                    terminal.update(cx, |terminal, _| terminal.kill_active_task());
                    let cleanup =
                        terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
                    let cleanup_deadline = cx.background_executor.timer(Duration::from_secs(10));
                    let cleanup_result = match futures::future::select(
                        Box::pin(cleanup),
                        Box::pin(cleanup_deadline),
                    )
                    .await
                    {
                        futures::future::Either::Left((status, _)) => {
                            format!("owned Task completion: {status:?}")
                        }
                        futures::future::Either::Right(_) => "cleanup deadline exceeded".into(),
                    };
                    panic!(
                        "Actual wrapper Task timed out ({role}); canonical cleanup: {cleanup_result}"
                    );
                }
            };
            assert_eq!(status.code(), Some(7));
            assert!(
                std::fs::metadata(&receipt)
                    .expect("Actual receiver receipt")
                    .len()
                    < 1024
            );
            let bytes = std::fs::read(&receipt).expect("Read actual receiver data");
            let fields = bytes.split(|byte| *byte == 0).collect::<Vec<_>>();
            assert_eq!(fields.len(), 5);
            assert_eq!(
                fields.first().copied(),
                Some(if role == "activation" {
                    b"wrapped".as_slice()
                } else {
                    b"off".as_slice()
                })
            );
            assert_eq!(
                fields.get(1).copied(),
                Some(if role == "changed-argv" {
                    b"2".as_slice()
                } else {
                    b"1".as_slice()
                })
            );
            assert_eq!(
                fields.get(3).copied(),
                Some(if role == "changed-argv" {
                    b"unexpected operand".as_slice()
                } else {
                    b"".as_slice()
                })
            );
            let receiver_process_id =
                std::str::from_utf8(fields.get(2).expect("Actual receiver PID"))
                    .expect("PID encoding")
                    .parse::<u32>()
                    .expect("Actual receiver numeric PID");
            terminal.read_with(cx, |terminal, _| {
                assert!(terminal.pid().is_none());
                assert!(terminal.pid_getter().is_none());
                let spawned_process_id = terminal
                    .subprocess
                    .as_ref()
                    .expect("Existing owned headless child")
                    .process_id;
                if role == "direct" {
                    assert_eq!(
                        terminal.task_process_id().map(|id| id.as_u32()),
                        Some(receiver_process_id)
                    );
                    assert_eq!(spawned_process_id, receiver_process_id);
                } else {
                    assert!(terminal.task_process_id().is_none());
                    if matches!(role, "activation" | "remote-marked") {
                        assert_ne!(spawned_process_id, receiver_process_id);
                    }
                }
            });
            drop(terminal);
            cx.run_until_parked();
            cases += 1;
        }
        assert_eq!(cases, 4);
        let empty = cx.new(|cx| {
            TerminalBuilder::new_display_only(
                SettingsCursorShape::default(),
                AlternateScroll::On,
                None,
                0,
                cx.background_executor(),
                PathStyle::local(),
            )
            .subscribe(cx)
        });
        assert!(empty.read_with(cx, |terminal, _| terminal.task_process_id().is_none()));
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn task_completion_multiple_waiters_and_late_waiter_share_actual_exit(
        cx: &mut TestAppContext,
    ) {
        cx.executor().allow_parking();
        for exit_code in [0, 7] {
            let controls = tempfile::tempdir().expect("Owned completion controls");
            let release = controls.path().join("release");
            let (terminal, _completion_receiver) = build_test_task_terminal(
                cx,
                "/bin/sh".into(),
                vec![
                    "-c".into(),
                    "printf completion-ready; while [ ! -f \"$1\" ]; do sleep 0.01; done; exit \"$2\"".into(),
                    "completion-waiters".into(),
                    release.to_string_lossy().into_owned(),
                    exit_code.to_string(),
                ],
                false,
            )
            .await;
            assert_content_eventually(&terminal, "completion-ready", cx).await;
            terminal.read_with(cx, |terminal, _| {
                assert_eq!(
                    terminal.task().expect("Actual task").status,
                    TaskStatus::Running
                );
                assert!(!terminal.process_finished);
            });
            let first = terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
            let second =
                terminal.read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx));
            assert!(
                terminal
                    .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
                    .now_or_never()
                    .is_none()
            );
            std::fs::write(&release, b"release").expect("Release actual child");
            let completed = futures::future::join(first, second);
            let deadline = cx.background_executor.timer(Duration::from_secs(10));
            let (first, second) =
                match futures::future::select(Box::pin(completed), Box::pin(deadline)).await {
                    futures::future::Either::Left((statuses, _)) => statuses,
                    futures::future::Either::Right(_) => {
                        panic!("Both actual completion observers must finish")
                    }
                };
            let first = first.expect("Actual child exit");
            assert_eq!(first.code(), Some(exit_code));
            assert_eq!(second, Some(first));
            let late = terminal
                .read_with(cx, |terminal, cx| terminal.wait_for_completed_task(cx))
                .await;
            assert_eq!(late, Some(first));
            assert_eq!(
                terminal.read_with(cx, |terminal, _| terminal.child_exited),
                Some(first)
            );
        }
    }

    #[test]
    fn test_convert_lf_to_crlf_preserves_split_crlf() {
        let mut previous_byte_was_cr = false;
        assert_eq!(
            convert_lf_to_crlf(b"one\n", &mut previous_byte_was_cr),
            b"one\r\n"
        );
        assert!(!previous_byte_was_cr);

        let mut previous_byte_was_cr = false;
        assert_eq!(
            convert_lf_to_crlf(b"two\r", &mut previous_byte_was_cr),
            b"two\r"
        );
        assert!(previous_byte_was_cr);
        assert_eq!(
            convert_lf_to_crlf(b"\nthree", &mut previous_byte_was_cr),
            b"\nthree"
        );
        assert!(!previous_byte_was_cr);
    }

    /// Regression test for the agent terminal failing with `Not a tty (os error
    /// 25)` in headless/eval sandboxes: a `no_pty` task terminal must run
    /// without a PTY, capture stdout, and report its exit status.
    #[cfg(not(target_os = "windows"))]
    #[gpui::test]
    async fn test_no_pty_task_terminal_captures_output(cx: &mut TestAppContext) {
        cx.executor().allow_parking();

        let (program, args) = ShellBuilder::new(&Shell::System, false)
            .non_interactive()
            .build(Some("echo hello-from-subprocess".to_owned()), &[]);
        let (terminal, completion_rx) = build_test_task_terminal(cx, program, args, true).await;

        assert!(
            !terminal.update(cx, |term, _| term.is_pty()),
            "no_pty terminal should not be PTY-backed"
        );
        assert_eq!(
            completion_rx.recv().await.unwrap(),
            Some(ExitStatus::default())
        );
        assert_content_eventually(&terminal, "hello-from-subprocess", cx).await;
    }

    fn init_ctrl_click_hyperlink_test(cx: &mut TestAppContext, output: &[u8]) -> Entity<Terminal> {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
        });

        let terminal = cx.new(|cx| {
            TerminalBuilder::new_display_only(
                SettingsCursorShape::default(),
                AlternateScroll::On,
                None,
                0,
                cx.background_executor(),
                PathStyle::local(),
            )
            .subscribe(cx)
        });

        terminal.update(cx, |terminal, cx| {
            terminal.write_output(output, cx);
        });

        cx.run_until_parked();

        terminal.update(cx, |terminal, _cx| {
            let term_lock = terminal.term.lock();
            terminal.last_content = make_content(&term_lock, &terminal.last_content);
            drop(term_lock);

            let terminal_bounds = TerminalBounds::new(
                px(20.0),
                px(10.0),
                bounds(point(px(0.0), px(0.0)), size(px(400.0), px(400.0))),
            );
            terminal.last_content.terminal_bounds = terminal_bounds;
            terminal.events.clear();
            terminal.take_pty_write_log();
        });

        terminal
    }

    fn ctrl_mouse_down_at(
        terminal: &mut Terminal,
        position: GpuiPoint<Pixels>,
        cx: &mut Context<Terminal>,
    ) {
        let mouse_down = MouseDownEvent {
            button: MouseButton::Left,
            position,
            modifiers: Modifiers::secondary_key(),
            click_count: 1,
            first_mouse: true,
        };
        terminal.mouse_down(&mouse_down, cx);
    }

    fn ctrl_mouse_move_to(
        terminal: &mut Terminal,
        position: GpuiPoint<Pixels>,
        cx: &mut Context<Terminal>,
    ) {
        let terminal_bounds = terminal.last_content.terminal_bounds.bounds;
        let drag_event = MouseMoveEvent {
            position,
            pressed_button: Some(MouseButton::Left),
            modifiers: Modifiers::secondary_key(),
        };
        terminal.mouse_drag(&drag_event, terminal_bounds, cx);
    }

    fn ctrl_mouse_up_at(
        terminal: &mut Terminal,
        position: GpuiPoint<Pixels>,
        cx: &mut Context<Terminal>,
    ) {
        let mouse_up = MouseUpEvent {
            button: MouseButton::Left,
            position,
            modifiers: Modifiers::secondary_key(),
            click_count: 1,
        };
        terminal.mouse_up(&mouse_up, cx);
    }

    fn left_mouse_down_at(
        terminal: &mut Terminal,
        position: GpuiPoint<Pixels>,
        cx: &mut Context<Terminal>,
    ) {
        let mouse_down = MouseDownEvent {
            button: MouseButton::Left,
            position,
            modifiers: Modifiers::none(),
            click_count: 1,
            first_mouse: true,
        };
        terminal.mouse_down(&mouse_down, cx);
    }

    fn left_mouse_up_at(
        terminal: &mut Terminal,
        position: GpuiPoint<Pixels>,
        cx: &mut Context<Terminal>,
    ) {
        let mouse_up = MouseUpEvent {
            button: MouseButton::Left,
            position,
            modifiers: Modifiers::none(),
            click_count: 1,
        };
        terminal.mouse_up(&mouse_up, cx);
    }

    fn left_mouse_drag_to(
        terminal: &mut Terminal,
        position: GpuiPoint<Pixels>,
        cx: &mut Context<Terminal>,
    ) {
        let region = terminal.last_content.terminal_bounds.bounds;
        let drag_event = MouseMoveEvent {
            position,
            pressed_button: Some(MouseButton::Left),
            modifiers: Modifiers::none(),
        };
        terminal.mouse_drag(&drag_event, region, cx);
    }

    /// A left click that jitters by a pixel or two (e.g. the window-focusing
    /// click) must not begin a selection, otherwise `copy_on_select` would
    /// overwrite the clipboard. Regression test for #58970.
    #[gpui::test]
    async fn test_terminal_click_jitter_does_not_start_selection(cx: &mut TestAppContext) {
        let terminal = init_ctrl_click_hyperlink_test(cx, b"hello world\r\n");

        terminal.update(cx, |terminal, cx| {
            left_mouse_down_at(terminal, point(px(50.0), px(10.0)), cx);
            terminal.events.clear();

            // One pixel of movement is below the drag threshold.
            left_mouse_drag_to(terminal, point(px(51.0), px(10.0)), cx);

            assert!(
                !terminal
                    .events
                    .iter()
                    .any(|event| matches!(event, InternalEvent::UpdateSelection(_))),
                "a sub-threshold click jitter should not start a selection"
            );
            assert!(terminal.selection_phase == SelectionPhase::Ended);
        });
    }

    /// A deliberate drag past the threshold must still start a selection.
    #[gpui::test]
    async fn test_terminal_deliberate_drag_starts_selection(cx: &mut TestAppContext) {
        let terminal = init_ctrl_click_hyperlink_test(cx, b"hello world\r\n");

        terminal.update(cx, |terminal, cx| {
            left_mouse_down_at(terminal, point(px(50.0), px(10.0)), cx);
            terminal.events.clear();

            // Well beyond the drag threshold.
            left_mouse_drag_to(terminal, point(px(90.0), px(10.0)), cx);

            assert!(
                terminal
                    .events
                    .iter()
                    .any(|event| matches!(event, InternalEvent::UpdateSelection(_))),
                "a deliberate drag should start a selection"
            );
            assert!(terminal.selection_phase == SelectionPhase::Selecting);
        });
    }

    /// With mouse tracking active (e.g. htop), Shift is the escape hatch to
    /// select terminal text. Shift+drag must start a selection rather than being
    /// swallowed as a "extend existing selection" no-op. Regression test for #60254.
    #[gpui::test]
    async fn test_terminal_shift_drag_selects_while_mouse_tracking(cx: &mut TestAppContext) {
        // `?1002h` enables button-event mouse tracking, `?1006h` selects SGR encoding.
        let terminal = init_ctrl_click_hyperlink_test(cx, b"\x1b[?1002h\x1b[?1006hhello world\r\n");

        terminal.update(cx, |terminal, cx| {
            assert!(
                terminal.last_content.mode.intersects(Modes::MOUSE_MODE),
                "mouse tracking should be active"
            );

            let shift = Modifiers {
                shift: true,
                ..Modifiers::none()
            };
            terminal.mouse_down(
                &MouseDownEvent {
                    button: MouseButton::Left,
                    position: point(px(50.0), px(10.0)),
                    modifiers: shift,
                    click_count: 1,
                    first_mouse: true,
                },
                cx,
            );

            // With no selection yet, the shift press must anchor a new selection
            // so the following drag has something to extend.
            assert!(
                terminal
                    .events
                    .iter()
                    .any(|event| matches!(event, InternalEvent::SetSelection(Some(_)))),
                "shift+click with no existing selection should anchor a selection"
            );
            terminal.events.clear();

            let region = terminal.last_content.terminal_bounds.bounds;
            terminal.mouse_drag(
                &MouseMoveEvent {
                    position: point(px(90.0), px(10.0)),
                    pressed_button: Some(MouseButton::Left),
                    modifiers: shift,
                },
                region,
                cx,
            );

            assert!(
                terminal
                    .events
                    .iter()
                    .any(|event| matches!(event, InternalEvent::UpdateSelection(_))),
                "shift+drag should extend the selection while mouse tracking is active"
            );
            assert!(terminal.selection_phase == SelectionPhase::Selecting);
        });
    }

    /// Shift+click with a selection already on screen must keep extending it
    /// (the behavior added in #25143), not re-anchor a fresh one.
    #[gpui::test]
    async fn test_terminal_shift_click_extends_existing_selection(cx: &mut TestAppContext) {
        let terminal = init_ctrl_click_hyperlink_test(cx, b"hello world\r\n");

        terminal.update(cx, |terminal, cx| {
            // A visible selection, as a sync would have populated in production.
            terminal.last_content.selection = Some(SelectionRange {
                start: Point::new(0, 0),
                end: Point::new(0, 5),
                is_block: false,
            });
            terminal.events.clear();

            terminal.mouse_down(
                &MouseDownEvent {
                    button: MouseButton::Left,
                    position: point(px(90.0), px(10.0)),
                    modifiers: Modifiers {
                        shift: true,
                        ..Modifiers::none()
                    },
                    click_count: 1,
                    first_mouse: true,
                },
                cx,
            );

            assert!(
                terminal
                    .events
                    .iter()
                    .any(|event| matches!(event, InternalEvent::UpdateSelection(_))),
                "shift+click with an existing selection should extend it"
            );
            assert!(
                !terminal
                    .events
                    .iter()
                    .any(|event| matches!(event, InternalEvent::SetSelection(Some(_)))),
                "shift+click should extend, not re-anchor, an existing selection"
            );
        });
    }

    #[gpui::test]
    async fn test_basic_terminal(cx: &mut TestAppContext) {
        cx.executor().allow_parking();

        let (terminal, completion_rx) = build_test_terminal(cx, "echo", &["hello"]).await;
        assert_eq!(
            completion_rx.recv().await.unwrap(),
            Some(ExitStatus::default())
        );
        assert_content_eventually(&terminal, "hello", cx).await;

        // Inject additional output directly into the emulator (display-only path)
        terminal.update(cx, |term, cx| {
            term.write_output(b"\nfrom_injection", cx);
        });

        let content_after = terminal.update(cx, |term, _| term.get_content());
        assert!(
            content_after.contains("from_injection"),
            "expected injected output to appear, got: {content_after}"
        );
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn test_foreground_process_command_tracks_path_command(cx: &mut TestAppContext) {
        cx.executor().allow_parking();

        let (terminal, completion_rx) =
            build_test_terminal_with_arguments(cx, "sleep".to_string(), vec!["1".to_string()])
                .await;

        assert_foreground_process_command_eventually(&terminal, "sleep", cx).await;

        assert!(
            completion_rx.recv().await.is_ok(),
            "expected terminal completion after sleep exits"
        );
    }

    // TODO should be tested on Linux too, but does not work there well
    #[cfg(target_os = "macos")]
    #[gpui::test(iterations = 10)]
    async fn test_terminal_eof(cx: &mut TestAppContext) {
        init_test(cx);

        cx.executor().allow_parking();

        let (completion_tx, completion_rx) = async_channel::unbounded();
        let builder = cx
            .update(|cx| {
                TerminalBuilder::new(
                    None,
                    None,
                    task::Shell::System,
                    HashMap::default(),
                    SettingsCursorShape::default(),
                    AlternateScroll::On,
                    None,
                    vec![],
                    0,
                    false,
                    0,
                    Some(completion_tx),
                    cx,
                    Vec::new(),
                    PathStyle::local(),
                )
            })
            .await
            .unwrap();
        // Build an empty command, which will result in a tty shell spawned.
        let terminal = cx.new(|cx| builder.subscribe(cx));

        let (event_tx, event_rx) = async_channel::unbounded::<Event>();
        cx.update(|cx| {
            cx.subscribe(&terminal, move |_, e, _| {
                event_tx.send_blocking(e.clone()).unwrap();
            })
        })
        .detach();
        cx.background_spawn(async move {
            assert_eq!(
                completion_rx.recv().await.unwrap(),
                Some(ExitStatus::default()),
                "EOF should result in the tty shell exiting successfully",
            );
        })
        .detach();

        let first_event = event_rx.recv().await.expect("No wakeup event received");

        terminal.update(cx, |terminal, _| {
            let success = terminal.try_keystroke(&Keystroke::parse("ctrl-d").unwrap(), false);
            assert!(success, "Should have registered ctrl-d sequence");
        });

        let mut all_events = vec![first_event];
        while let Ok(new_event) = event_rx.recv().await {
            all_events.push(new_event.clone());
            if new_event == Event::CloseTerminal {
                break;
            }
        }
        assert!(
            all_events.contains(&Event::CloseTerminal),
            "EOF command sequence should have triggered a TTY terminal exit, but got events: {all_events:?}",
        );
    }

    #[cfg(not(target_os = "windows"))]
    #[gpui::test(iterations = 10)]
    async fn test_terminal_closes_after_nonzero_exit(cx: &mut TestAppContext) {
        init_test(cx);

        cx.executor().allow_parking();

        let builder = cx
            .update(|cx| {
                TerminalBuilder::new(
                    None,
                    None,
                    task::Shell::System,
                    HashMap::default(),
                    SettingsCursorShape::default(),
                    AlternateScroll::On,
                    None,
                    vec![],
                    0,
                    false,
                    0,
                    None,
                    cx,
                    Vec::new(),
                    PathStyle::local(),
                )
            })
            .await
            .unwrap();
        let terminal = cx.new(|cx| builder.subscribe(cx));

        let (event_tx, event_rx) = async_channel::unbounded::<Event>();
        cx.update(|cx| {
            cx.subscribe(&terminal, move |_, e, _| {
                event_tx.send_blocking(e.clone()).unwrap();
            })
        })
        .detach();

        let first_event = event_rx.recv().await.expect("No wakeup event received");

        terminal.update(cx, |terminal, _| {
            terminal.input(b"false\r".to_vec());
        });
        cx.executor().timer(Duration::from_millis(500)).await;
        terminal.update(cx, |terminal, _| {
            terminal.input(b"exit\r".to_vec());
        });

        let mut all_events = vec![first_event];
        while let Ok(new_event) = event_rx.recv().await {
            all_events.push(new_event.clone());
            if new_event == Event::CloseTerminal {
                break;
            }
        }
        assert!(
            all_events.contains(&Event::CloseTerminal),
            "Shell exiting after `false && exit` should close terminal, but got events: {all_events:?}",
        );
    }

    #[gpui::test(iterations = 10)]
    async fn test_terminal_no_exit_on_spawn_failure(cx: &mut TestAppContext) {
        cx.executor().allow_parking();

        let (completion_tx, completion_rx) = async_channel::unbounded();
        let (program, args) = ShellBuilder::new(&Shell::System, false)
            .build(Some("asdasdasdasd".to_owned()), &["@@@@@".to_owned()]);
        let builder = cx
            .update(|cx| {
                TerminalBuilder::new(
                    None,
                    None,
                    task::Shell::WithArguments {
                        program,
                        args,
                        title_override: None,
                    },
                    HashMap::default(),
                    SettingsCursorShape::default(),
                    AlternateScroll::On,
                    None,
                    Vec::new(),
                    0,
                    false,
                    0,
                    Some(completion_tx),
                    cx,
                    Vec::new(),
                    PathStyle::local(),
                )
            })
            .await
            .unwrap();
        let terminal = cx.new(|cx| builder.subscribe(cx));

        let all_events: Arc<Mutex<Vec<Event>>> = Arc::new(Mutex::new(Vec::new()));
        cx.update({
            let all_events = all_events.clone();
            |cx| {
                cx.subscribe(&terminal, move |_, e, _| {
                    all_events.lock().push(e.clone());
                })
            }
        })
        .detach();
        let completion_check_task = cx.background_spawn(async move {
            // The channel may be closed if the terminal is dropped before sending
            // the completion signal, which can happen with certain task scheduling orders.
            let exit_status = completion_rx.recv().await.ok().flatten();
            if let Some(exit_status) = exit_status {
                assert!(
                    !exit_status.success(),
                    "Wrong shell command should result in a failure"
                );
                #[cfg(target_os = "windows")]
                assert_eq!(exit_status.code(), Some(1));
                #[cfg(not(target_os = "windows"))]
                assert_eq!(exit_status.code(), Some(127)); // code 127 means "command not found" on Unix
            }
        });

        completion_check_task.await;
        cx.executor().timer(Duration::from_millis(500)).await;

        assert!(
            !all_events
                .lock()
                .iter()
                .any(|event| event == &Event::CloseTerminal),
            "Wrong shell command should update the title but not should not close the terminal to show the error message, but got events: {all_events:?}",
        );
    }

    #[test]
    fn test_rgb_for_index() {
        // Test every possible value in the color cube.
        for i in 16..=231 {
            let (r, g, b) = rgb_for_index(i);
            assert_eq!(i, 16 + 36 * r + 6 * g + b);
        }
    }

    #[gpui::test]
    fn test_mouse_to_cell_test(mut rng: StdRng) {
        const ITERATIONS: usize = 10;
        const PRECISION: usize = 1000;

        for _ in 0..ITERATIONS {
            let viewport_cells = rng.random_range(15..20);
            let cell_size =
                rng.random_range(5 * PRECISION..20 * PRECISION) as f32 / PRECISION as f32;

            let size = crate::TerminalBounds {
                cell_width: Pixels::from(cell_size),
                line_height: Pixels::from(cell_size),
                bounds: bounds(
                    GpuiPoint::default(),
                    size(
                        Pixels::from(cell_size * (viewport_cells as f32)),
                        Pixels::from(cell_size * (viewport_cells as f32)),
                    ),
                ),
            };

            let cells = get_cells(size, &mut rng);
            let content = convert_cells_to_content(size, &cells);

            for row in 0..(viewport_cells - 1) {
                let row = row as usize;
                for col in 0..(viewport_cells - 1) {
                    let col = col as usize;

                    let row_offset = rng.random_range(0..PRECISION) as f32 / PRECISION as f32;
                    let col_offset = rng.random_range(0..PRECISION) as f32 / PRECISION as f32;

                    let mouse_pos = point(
                        Pixels::from(col as f32 * cell_size + col_offset),
                        Pixels::from(row as f32 * cell_size + row_offset),
                    );

                    let content_index =
                        content_index_for_mouse(mouse_pos, &content.terminal_bounds);
                    let mouse_cell = content.cells[content_index].character();
                    let real_cell = cells[row][col];

                    assert_eq!(mouse_cell, real_cell);
                }
            }
        }
    }

    #[gpui::test]
    fn test_mouse_to_cell_clamp(mut rng: StdRng) {
        let size = crate::TerminalBounds {
            cell_width: Pixels::from(10.),
            line_height: Pixels::from(10.),
            bounds: bounds(
                GpuiPoint::default(),
                size(Pixels::from(100.), Pixels::from(100.)),
            ),
        };

        let cells = get_cells(size, &mut rng);
        let content = convert_cells_to_content(size, &cells);

        assert_eq!(
            content.cells[content_index_for_mouse(
                point(Pixels::from(-10.), Pixels::from(-10.)),
                &content.terminal_bounds,
            )]
            .character(),
            cells[0][0]
        );
        assert_eq!(
            content.cells[content_index_for_mouse(
                point(Pixels::from(1000.), Pixels::from(1000.)),
                &content.terminal_bounds,
            )]
            .character(),
            cells[9][9]
        );
    }

    #[gpui::test]
    async fn test_set_size_coalesces_pixel_only_changes(cx: &mut TestAppContext) {
        let builder = cx.update(|cx| {
            TerminalBuilder::new_display_only(
                SettingsCursorShape::Block,
                AlternateScroll::On,
                None,
                0,
                cx.background_executor(),
                PathStyle::local(),
            )
        });
        let mut terminal = builder.terminal;

        let base_bounds = TerminalBounds {
            cell_width: Pixels::from(10.),
            line_height: Pixels::from(10.),
            bounds: bounds(
                GpuiPoint::default(),
                size(Pixels::from(100.), Pixels::from(100.)),
            ),
        };

        terminal.set_size(base_bounds);
        terminal.events.clear();
        assert_eq!(terminal.last_content.terminal_bounds, base_bounds);

        // Pixel-only change: height grows by 1px but still the same number of rows/cols.
        let mut pixel_changed = base_bounds;
        pixel_changed.bounds.size.height = Pixels::from(101.);
        terminal.set_size(pixel_changed);
        assert!(terminal.events.is_empty());
        assert_eq!(terminal.last_content.terminal_bounds, pixel_changed);

        // Grid change: height increases enough to add a row.
        let mut grid_changed = base_bounds;
        grid_changed.bounds.size.height = Pixels::from(110.);
        terminal.set_size(grid_changed);
        assert!(matches!(
            terminal.events.back(),
            Some(InternalEvent::Resize(_))
        ));
    }

    fn get_cells(size: TerminalBounds, rng: &mut StdRng) -> Vec<Vec<char>> {
        let mut cells = Vec::new();

        for _ in 0..size.num_lines() {
            let mut row_vec = Vec::new();
            for _ in 0..size.num_columns() {
                let cell_char = rng.sample(distr::Alphanumeric) as char;
                row_vec.push(cell_char)
            }
            cells.push(row_vec)
        }

        cells
    }

    fn convert_cells_to_content(terminal_bounds: TerminalBounds, cells: &[Vec<char>]) -> Content {
        let mut ic = Vec::new();

        for (index, row) in cells.iter().enumerate() {
            for (cell_index, cell_char) in row.iter().enumerate() {
                let mut cell = Cell::default();
                cell.set_character(*cell_char);
                ic.push(IndexedCell {
                    point: Point::new(index as i32, cell_index),
                    cell,
                });
            }
        }

        Content {
            cells: ic,
            terminal_bounds,
            ..Default::default()
        }
    }

    #[gpui::test]
    async fn test_write_init_command_after_startup_clears_without_shell_command(
        cx: &mut TestAppContext,
    ) {
        let terminal = cx.new(|cx| {
            TerminalBuilder::new_display_only(
                SettingsCursorShape::default(),
                AlternateScroll::On,
                None,
                0,
                cx.background_executor(),
                PathStyle::local(),
            )
            .subscribe(cx)
        });

        terminal.update(cx, |terminal, cx| {
            terminal.write_output(b"startup output\nprompt", cx);
        });

        let wrote = terminal.update(cx, |terminal, cx| {
            terminal.cwd_history.push(CwdHistoryEntry {
                scrollback_position: 42,
                working_directory: PathBuf::from("/stale/cwd"),
            });
            terminal.write_init_command_after_startup(b"agent\r".to_vec(), cx)
        });
        assert!(wrote);
        let (content, cwd_history) = terminal.update(cx, |terminal, _| {
            (terminal.get_content(), terminal.cwd_history.clone())
        });
        assert!(cwd_history.is_empty());
        assert!(
            !content.contains("startup output"),
            "startup output should be cleared internally before writing the init command"
        );
        let input_log = terminal.update(cx, |terminal, _| terminal.take_input_log());
        assert_eq!(input_log, vec![b"agent\r".to_vec()]);
    }

    #[gpui::test]
    async fn test_write_init_command_after_startup_skips_after_keyboard_input(
        cx: &mut TestAppContext,
    ) {
        let terminal = cx.new(|cx| {
            TerminalBuilder::new_display_only(
                SettingsCursorShape::default(),
                AlternateScroll::On,
                None,
                0,
                cx.background_executor(),
                PathStyle::local(),
            )
            .subscribe(cx)
        });

        let wrote = terminal.update(cx, |terminal, cx| {
            terminal.write_output(b"startup output\nprompt", cx);
            terminal.input(b"user input".to_vec());
            terminal.write_init_command_after_startup(b"agent\r".to_vec(), cx)
        });
        assert!(!wrote);
        let content = terminal.update(cx, |terminal, _| terminal.get_content());
        assert!(
            content.contains("startup output"),
            "startup output should be left alone when the init command is skipped"
        );
        let input_log = terminal.update(cx, |terminal, _| terminal.take_input_log());
        assert_eq!(input_log, vec![b"user input".to_vec()]);
    }

    #[gpui::test]
    async fn test_write_init_command_after_startup_skips_after_child_exit(cx: &mut TestAppContext) {
        let terminal = cx.new(|cx| {
            TerminalBuilder::new_display_only(
                SettingsCursorShape::default(),
                AlternateScroll::On,
                None,
                0,
                cx.background_executor(),
                PathStyle::local(),
            )
            .subscribe(cx)
        });

        terminal.update(cx, |terminal, cx| {
            terminal.write_output(b"shell failed to start\nprompt", cx);
            #[cfg(unix)]
            let exit_status =
                <ExitStatus as std::os::unix::process::ExitStatusExt>::from_raw(1 << 8);
            #[cfg(windows)]
            let exit_status = <ExitStatus as std::os::windows::process::ExitStatusExt>::from_raw(1);
            terminal.register_task_finished(Some(exit_status), cx);
        });

        let wrote = terminal.update(cx, |terminal, cx| {
            terminal.write_init_command_after_startup(b"agent\r".to_vec(), cx)
        });
        assert!(!wrote);
        let content = terminal.update(cx, |terminal, _| terminal.get_content());
        assert!(
            content.contains("shell failed to start"),
            "startup failure output should be preserved when the init command is skipped"
        );
        let input_log = terminal.update(cx, |terminal, _| terminal.take_input_log());
        assert!(
            input_log.is_empty(),
            "init command should not be written after the child has exited, got {input_log:?}"
        );
    }

    #[gpui::test]
    async fn structured_task_terminal_reports_completion_and_close(cx: &mut TestAppContext) {
        let terminal = cx.new(|cx| {
            TerminalBuilder::new_display_only(
                SettingsCursorShape::default(),
                AlternateScroll::On,
                None,
                0,
                cx.background_executor(),
                PathStyle::local(),
            )
            .subscribe(cx)
        });
        let completed = StructuredTaskHandle::new(TaskId("completed".to_string()));
        let (_, completion_rx) = async_channel::unbounded();
        terminal.update(cx, |terminal, cx| {
            completed.mark_running(Some(StructuredTerminalId(cx.entity_id().as_u64())), cx);
            terminal.task = Some(TaskState {
                status: TaskStatus::Running,
                completion_rx,
                spawned_task: SpawnInTerminal {
                    full_label: "Copper task".to_string(),
                    show_summary: true,
                    ..SpawnInTerminal::default()
                },
                structured_task: Some(completed.clone()),
            });
            #[cfg(unix)]
            let exit_status =
                <ExitStatus as std::os::unix::process::ExitStatusExt>::from_raw(3 << 8);
            #[cfg(windows)]
            let exit_status = <ExitStatus as std::os::windows::process::ExitStatusExt>::from_raw(3);
            terminal.register_task_finished(Some(exit_status), cx);
        });
        assert!(matches!(
            completed.state(),
            StructuredTaskState::Completed {
                exit_code: Some(3),
                success: false,
                ..
            }
        ));
        let content = terminal.update(cx, |terminal, _| terminal.get_content());
        assert!(content.contains("Task `Copper task` finished with exit code: 3"));

        let closed = StructuredTaskHandle::new(TaskId("closed".to_string()));
        let (_, completion_rx) = async_channel::unbounded();
        terminal.update(cx, |terminal, cx| {
            closed.mark_running(Some(StructuredTerminalId(cx.entity_id().as_u64())), cx);
            terminal.task = Some(TaskState {
                status: TaskStatus::Running,
                completion_rx,
                spawned_task: SpawnInTerminal::default(),
                structured_task: Some(closed.clone()),
            });
            terminal.register_task_finished(None, cx);
        });
        assert!(matches!(
            closed.state(),
            StructuredTaskState::Cancelled {
                termination_confirmed: true,
                ..
            }
        ));

        let cancelled = StructuredTaskHandle::new(TaskId("cancelled".to_string()));
        let (_, completion_rx) = async_channel::unbounded();
        terminal.update(cx, |terminal, cx| {
            let terminal_id = Some(StructuredTerminalId(cx.entity_id().as_u64()));
            cancelled.mark_running(terminal_id, cx);
            cancelled.mark_cancelled(terminal_id, false, cx);
            terminal.task = Some(TaskState {
                status: TaskStatus::Running,
                completion_rx,
                spawned_task: SpawnInTerminal {
                    full_label: "Java test".to_string(),
                    show_summary: true,
                    ..SpawnInTerminal::default()
                },
                structured_task: Some(cancelled.clone()),
            });
            terminal.register_task_finished(None, cx);
        });
        assert!(matches!(
            cancelled.state(),
            StructuredTaskState::Cancelled {
                termination_confirmed: true,
                ..
            }
        ));
        let content = terminal.update(cx, |terminal, _| terminal.get_content());
        assert!(content.contains("Task `Java test` cancelled"));
    }

    #[gpui::test]
    async fn test_write_output_converts_lf_to_crlf(cx: &mut TestAppContext) {
        let terminal = cx.new(|cx| {
            TerminalBuilder::new_display_only(
                SettingsCursorShape::default(),
                AlternateScroll::On,
                None,
                0,
                cx.background_executor(),
                PathStyle::local(),
            )
            .subscribe(cx)
        });

        // Test simple LF conversion
        terminal.update(cx, |terminal, cx| {
            terminal.write_output(b"line1\nline2\n", cx);
        });

        // Get the content by directly accessing the term
        let content = terminal.update(cx, |terminal, _cx| {
            let term = terminal.term.lock_unfair();
            make_content(&term, &terminal.last_content)
        });

        // If LF is properly converted to CRLF, each line should start at column 0
        // The diagonal staircase bug would cause increasing column positions

        // Get the cells and check that lines start at column 0
        let cells = &content.cells;
        let mut line1_col0 = false;
        let mut line2_col0 = false;

        for cell in cells {
            if cell.character() == 'l' && cell.point.column == 0 {
                if cell.point.line == 0 && !line1_col0 {
                    line1_col0 = true;
                } else if cell.point.line == 1 && !line2_col0 {
                    line2_col0 = true;
                }
            }
        }

        assert!(line1_col0, "First line should start at column 0");
        assert!(line2_col0, "Second line should start at column 0");
    }

    #[gpui::test]
    async fn test_write_output_preserves_existing_crlf(cx: &mut TestAppContext) {
        let terminal = cx.new(|cx| {
            TerminalBuilder::new_display_only(
                SettingsCursorShape::default(),
                AlternateScroll::On,
                None,
                0,
                cx.background_executor(),
                PathStyle::local(),
            )
            .subscribe(cx)
        });

        // Test that existing CRLF doesn't get doubled
        terminal.update(cx, |terminal, cx| {
            terminal.write_output(b"line1\r\nline2\r\n", cx);
        });

        // Get the content by directly accessing the term
        let content = terminal.update(cx, |terminal, _cx| {
            let term = terminal.term.lock_unfair();
            make_content(&term, &terminal.last_content)
        });

        let cells = &content.cells;

        // Check that both lines start at column 0
        let mut found_lines_at_column_0 = 0;
        for cell in cells {
            if cell.character() == 'l' && cell.point.column == 0 {
                found_lines_at_column_0 += 1;
            }
        }

        assert!(
            found_lines_at_column_0 >= 2,
            "Both lines should start at column 0"
        );
    }

    #[gpui::test]
    async fn test_write_output_preserves_bare_cr(cx: &mut TestAppContext) {
        let terminal = cx.new(|cx| {
            TerminalBuilder::new_display_only(
                SettingsCursorShape::default(),
                AlternateScroll::On,
                None,
                0,
                cx.background_executor(),
                PathStyle::local(),
            )
            .subscribe(cx)
        });

        // Test that bare CR (without LF) is preserved
        terminal.update(cx, |terminal, cx| {
            terminal.write_output(b"hello\rworld", cx);
        });

        // Get the content by directly accessing the term
        let content = terminal.update(cx, |terminal, _cx| {
            let term = terminal.term.lock_unfair();
            make_content(&term, &terminal.last_content)
        });

        let cells = &content.cells;

        // Check that we have "world" at the beginning of the line
        let mut text = String::new();
        for cell in cells.iter().take(5) {
            if cell.point.line == 0 {
                text.push(cell.character());
            }
        }

        assert!(
            text.starts_with("world"),
            "Bare CR should allow overwriting: got '{}'",
            text
        );
    }

    fn snapshot_test_terminal(cx: &mut TestAppContext) -> Entity<Terminal> {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
        });
        cx.new(|cx| {
            TerminalBuilder::new_display_only_with_bounds(
                SettingsCursorShape::default(),
                AlternateScroll::On,
                None,
                0,
                cx.background_executor(),
                PathStyle::local(),
                TerminalBounds::new(
                    px(20.),
                    px(10.),
                    bounds(point(px(0.), px(0.)), size(px(100.), px(40.))),
                ),
            )
            .subscribe(cx)
        })
    }

    #[gpui::test]
    async fn test_terminal_snapshot_preserves_styled_unicode_scrollback(cx: &mut TestAppContext) {
        let source = snapshot_test_terminal(cx);
        source.update(cx, |terminal, cx| {
            terminal.write_output(
                "\u{1b}[1;4;38;2;12;34;56;48;5;42m界e\u{301} wrapped content\u{1b}[0m\nsecond\nthird\nfourth\n\u{1b}]4;1;rgb:aa/bb/cc\u{7}\u{1b}]8;;https://example.test\u{1b}\\linked\u{1b}]8;;\u{1b}\\".as_bytes(),
                cx,
            );
        });
        cx.run_until_parked();
        source.update(cx, |terminal, _| {
            scroll_display(&mut terminal.term.lock_unfair(), Scroll::Delta(1));
        });
        let snapshot = source.read_with(cx, |terminal, _| {
            terminal.snapshot().expect("capture canonical cells")
        });
        assert_eq!(snapshot.display_offset, 1);
        assert!(snapshot.rows.len() > snapshot.screen_lines);
        assert!(
            snapshot
                .rows
                .iter()
                .flatten()
                .any(|cell| cell.flags.contains(
                    alacritty_terminal::term::cell::Flags::BOLD
                        | alacritty_terminal::term::cell::Flags::UNDERLINE
                ))
        );
        assert!(snapshot.rows.iter().flatten().any(|cell| {
            cell.flags
                .contains(alacritty_terminal::term::cell::Flags::WIDE_CHAR)
        }));
        assert!(snapshot.rows.iter().flatten().any(|cell| {
            cell.flags
                .contains(alacritty_terminal::term::cell::Flags::WRAPLINE)
        }));
        assert!(
            snapshot
                .rows
                .iter()
                .flatten()
                .any(|cell| cell.bg == Color::Indexed(42))
        );
        assert!(
            snapshot
                .rows
                .iter()
                .flatten()
                .any(|cell| cell.hyperlink().is_some())
        );
        assert_eq!(
            snapshot.colors.get(1),
            Some(&Some(Rgb {
                r: 170,
                g: 187,
                b: 204
            }))
        );
        assert!(
            snapshot
                .rows
                .iter()
                .flatten()
                .any(|cell| cell.zerowidth().is_some())
        );
        assert!(snapshot.rows.iter().flatten().any(|cell| cell.fg
            == Color::Spec(Rgb {
                r: 12,
                g: 34,
                b: 56
            })));
        let encoded = serde_json::to_vec(&snapshot).expect("serialize terminal snapshot");
        let decoded: TerminalSnapshot =
            serde_json::from_slice(&encoded).expect("deserialize terminal snapshot");
        assert_eq!(decoded, snapshot);
        let restored = snapshot_test_terminal(cx);
        restored.update(cx, |terminal, cx| {
            terminal
                .restore_snapshot(&decoded, cx)
                .expect("restore display-only cells")
        });
        cx.run_until_parked();
        assert_eq!(
            restored.read_with(cx, |terminal, _| terminal
                .snapshot()
                .expect("capture restored cells")),
            snapshot
        );
        assert_eq!(
            restored.read_with(cx, |terminal, _| terminal.get_content()),
            source.read_with(cx, |terminal, _| terminal.get_content())
        );
        restored.read_with(cx, |terminal, _| {
            assert!(!terminal.is_pty() && terminal.subprocess.is_none() && terminal.task.is_none())
        });
    }

    #[gpui::test]
    async fn test_terminal_snapshot_reflow_preserves_retained_content(cx: &mut TestAppContext) {
        let source = snapshot_test_terminal(cx);
        source.update(cx, |terminal, cx| {
            terminal.write_output(b"one two three four five six", cx)
        });
        let snapshot = source.read_with(cx, |terminal, _| {
            terminal.snapshot().expect("snapshot wrapped output")
        });
        let restored = snapshot_test_terminal(cx);
        restored.update(cx, |terminal, cx| {
            terminal.last_content.terminal_bounds.bounds.size.width = px(200.);
            terminal
                .restore_snapshot(&snapshot, cx)
                .expect("restore at wider bounds");
        });
        assert_eq!(
            restored.read_with(cx, |terminal, _| terminal.get_content()),
            source.read_with(cx, |terminal, _| terminal.get_content())
        );
        assert_eq!(
            restored.read_with(cx, |terminal, _| terminal
                .snapshot()
                .expect("resized snapshot")
                .columns),
            20
        );
        let wider = restored.read_with(cx, |terminal, _| {
            terminal.snapshot().expect("wider retained snapshot")
        });
        restored.update(cx, |terminal, cx| {
            terminal.last_content.terminal_bounds.bounds.size.width = px(50.);
            terminal
                .restore_snapshot(&wider, cx)
                .expect("narrower retained reflow");
        });
        assert_eq!(
            restored.read_with(cx, |terminal, _| terminal.get_content()),
            source.read_with(cx, |terminal, _| terminal.get_content())
        );
    }

    #[gpui::test]
    async fn test_terminal_snapshot_preserves_deferred_wrap_during_reflow(cx: &mut TestAppContext) {
        let source = snapshot_test_terminal(cx);
        source.update(cx, |terminal, cx| {
            terminal.write_output(b"\x1b[41m0123456789", cx);
        });
        let snapshot = source.read_with(cx, |terminal, _| {
            terminal
                .snapshot()
                .expect("snapshot at filled final column")
        });
        assert!(snapshot.cursor_needs_wrap);
        let restored = snapshot_test_terminal(cx);
        let mut wider_bounds =
            source.read_with(cx, |terminal, _| terminal.last_content.terminal_bounds);
        wider_bounds.bounds.size.width = px(200.);
        source.update(cx, |terminal, _| {
            let mut term = terminal.term.lock_unfair();
            resize(&mut term, wider_bounds);
            terminal.last_content.terminal_bounds = wider_bounds;
            terminal.last_content = make_content(&term, &terminal.last_content);
        });
        restored.update(cx, |terminal, cx| {
            terminal.last_content.terminal_bounds = wider_bounds;
            terminal
                .restore_snapshot(&snapshot, cx)
                .expect("restore and reflow final column");
        });
        let expected = source.read_with(cx, |terminal, _| {
            terminal.snapshot().expect("live grid after widening")
        });
        assert_eq!(expected.cursor_column, 10);
        assert!(!expected.cursor_needs_wrap);
        assert_eq!(
            restored.read_with(cx, |terminal, _| terminal
                .snapshot()
                .expect("restored resized grid")),
            expected
        );
    }

    #[cfg(not(target_os = "windows"))]
    #[gpui::test]
    async fn test_terminal_snapshot_refuses_a_real_pty(cx: &mut TestAppContext) {
        cx.executor().allow_parking();
        init_test(cx);
        let source = snapshot_test_terminal(cx);
        source.update(cx, |terminal, cx| terminal.write_output(b"historical", cx));
        let snapshot = source.read_with(cx, |terminal, _| {
            terminal.snapshot().expect("history snapshot")
        });
        let (live, completion) = build_test_terminal(cx, "true", &[]).await;
        assert_eq!(
            completion.recv().await.expect("live completion"),
            Some(ExitStatus::default())
        );
        cx.run_until_parked();
        live.update(cx, |terminal, cx| {
            assert!(terminal.is_pty());
            assert!(terminal.restore_snapshot(&snapshot, cx).is_err());
            assert!(terminal.is_pty());
        });
    }

    #[gpui::test]
    async fn test_terminal_snapshot_rejects_corruption_without_changing_output(
        cx: &mut TestAppContext,
    ) {
        let terminal = snapshot_test_terminal(cx);
        terminal.update(cx, |terminal, cx| terminal.write_output(b"retained", cx));
        let original = terminal.read_with(cx, |terminal, _| {
            terminal.snapshot().expect("valid original")
        });
        for index in 0..7 {
            let mut corrupt = original.clone();
            match index {
                0 => corrupt.version = 2,
                1 => corrupt.columns = 0,
                2 => corrupt.screen_lines = 0,
                3 => {
                    corrupt.rows.first_mut().expect("first row").pop();
                }
                4 => corrupt.cursor_column = corrupt.columns,
                5 => corrupt.display_offset = usize::MAX,
                _ => {
                    corrupt.colors.pop();
                }
            }
            terminal.update(cx, |terminal, cx| {
                assert!(terminal.restore_snapshot(&corrupt, cx).is_err())
            });
            assert_eq!(
                terminal.read_with(cx, |terminal, _| terminal
                    .snapshot()
                    .expect("preserved original")),
                original
            );
        }
    }

    #[gpui::test]
    async fn test_terminal_snapshot_restore_does_not_replay_clipboard_or_bell(
        cx: &mut TestAppContext,
    ) {
        let source = snapshot_test_terminal(cx);
        source.update(cx, |terminal, cx| {
            terminal.write_output(b"retained\x07\x1b]52;c;b3ZlcndyaXR0ZW4=\x07", cx)
        });
        cx.run_until_parked();
        let snapshot = source.read_with(cx, |terminal, _| {
            terminal
                .snapshot()
                .expect("snapshot after side-effect sequences")
        });
        let restored = snapshot_test_terminal(cx);
        let bells = std::rc::Rc::new(std::cell::Cell::new(0));
        let subscription = cx.update(|cx| {
            cx.write_to_clipboard(ClipboardItem::new_string("original".into()));
            cx.subscribe(&restored, {
                let bells = bells.clone();
                move |_, event, _| {
                    if matches!(event, Event::Bell) {
                        bells.set(bells.get() + 1);
                    }
                }
            })
        });
        restored.update(cx, |terminal, cx| {
            terminal
                .restore_snapshot(&snapshot, cx)
                .expect("restore historical cells")
        });
        cx.run_until_parked();
        assert_eq!(bells.get(), 0);
        assert_eq!(
            cx.update(|cx| cx.read_from_clipboard().and_then(|item| item.text()))
                .as_deref(),
            Some("original")
        );
        drop(subscription);
    }

    #[gpui::test]
    async fn test_display_only_write_output_ignores_osc52(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
            cx.write_to_clipboard(ClipboardItem::new_string("original".to_string()));
        });

        let terminal = cx.new(|cx| {
            TerminalBuilder::new_display_only(
                SettingsCursorShape::default(),
                AlternateScroll::On,
                None,
                0,
                cx.background_executor(),
                PathStyle::local(),
            )
            .subscribe(cx)
        });

        terminal.update(cx, |terminal, cx| {
            terminal.write_output(b"\x1b]52;c;b3ZlcndyaXR0ZW4=\x07", cx);
        });
        cx.run_until_parked();

        let clipboard_text = cx.update(|cx| cx.read_from_clipboard().and_then(|item| item.text()));
        assert_eq!(clipboard_text.as_deref(), Some("original"));
    }

    #[gpui::test]
    async fn test_hyperlink_ctrl_click_same_position(cx: &mut TestAppContext) {
        let terminal = init_ctrl_click_hyperlink_test(cx, b"Visit https://zed.dev/ for more\r\n");

        terminal.update(cx, |terminal, cx| {
            let click_position = point(px(80.0), px(10.0));
            ctrl_mouse_down_at(terminal, click_position, cx);
            ctrl_mouse_up_at(terminal, click_position, cx);

            assert!(
                terminal
                    .events
                    .iter()
                    .any(|event| matches!(event, InternalEvent::ProcessHyperlink(_, true))),
                "Should have ProcessHyperlink event when ctrl+clicking on same hyperlink position"
            );
        });
    }

    #[gpui::test]
    async fn test_hyperlink_ctrl_click_same_position_in_mouse_mode(cx: &mut TestAppContext) {
        let terminal = init_ctrl_click_hyperlink_test(cx, b"Visit https://zed.dev/ for more\r\n");

        terminal.update(cx, |terminal, cx| {
            terminal.last_content.mode = Modes::MOUSE_MODE;

            let click_position = point(px(80.0), px(10.0));
            ctrl_mouse_down_at(terminal, click_position, cx);
            ctrl_mouse_up_at(terminal, click_position, cx);

            assert!(
                terminal
                    .events
                    .iter()
                    .any(|event| matches!(event, InternalEvent::ProcessHyperlink(_, true))),
                "Should have ProcessHyperlink event when ctrl+clicking on same hyperlink position in mouse mode"
            );
            assert!(
                terminal.take_pty_write_log().is_empty(),
                "a consumed link click must not be reported to the PTY"
            );
        });
    }

    #[gpui::test]
    async fn test_hyperlink_ctrl_click_mismatch_in_mouse_mode_consumes_gesture(
        cx: &mut TestAppContext,
    ) {
        let terminal = init_ctrl_click_hyperlink_test(
            cx,
            b"Visit https://zed.dev/ for more\r\nThis is another line\r\n",
        );

        terminal.update(cx, |terminal, cx| {
            terminal.last_content.mode = Modes::MOUSE_MODE;
            terminal.take_pty_write_log();

            let down_position = point(px(80.0), px(10.0));
            let up_position = point(px(10.0), px(30.0));

            ctrl_mouse_down_at(terminal, down_position, cx);
            terminal.mouse_move(
                &MouseMoveEvent {
                    position: up_position,
                    pressed_button: Some(MouseButton::Left),
                    modifiers: Modifiers::secondary_key(),
                },
                cx,
            );
            ctrl_mouse_up_at(terminal, up_position, cx);

            assert!(
                !terminal
                    .events
                    .iter()
                    .any(|event| matches!(event, InternalEvent::ProcessHyperlink(_, _))),
                "Should NOT open a link when press and release land on different hyperlinks"
            );
            let pty_writes = terminal.take_pty_write_log();
            assert!(
                pty_writes.is_empty(),
                "a captured press must consume the whole gesture, but reports leaked to the PTY: {pty_writes:?}"
            );
        });
    }

    #[gpui::test]
    async fn test_plain_click_on_hyperlink_in_mouse_mode_is_reported(cx: &mut TestAppContext) {
        let terminal = init_ctrl_click_hyperlink_test(cx, b"Visit https://zed.dev/ for more\r\n");

        terminal.update(cx, |terminal, cx| {
            terminal.last_content.mode = Modes::MOUSE_MODE;
            terminal.take_pty_write_log();

            let click_position = point(px(80.0), px(10.0));
            left_mouse_down_at(terminal, click_position, cx);
            left_mouse_up_at(terminal, click_position, cx);

            assert!(
                !terminal
                    .events
                    .iter()
                    .any(|event| matches!(event, InternalEvent::ProcessHyperlink(_, _))),
                "a plain click must not open a link"
            );
            let pty_writes = terminal.take_pty_write_log();
            assert_eq!(
                pty_writes.len(),
                2,
                "expected press and release reports, got {pty_writes:?}"
            );
        });
    }

    #[gpui::test]
    async fn test_ctrl_click_on_non_hyperlink_in_mouse_mode_is_reported(cx: &mut TestAppContext) {
        let terminal = init_ctrl_click_hyperlink_test(cx, b"Visit https://zed.dev/ for more\r\n");

        terminal.update(cx, |terminal, cx| {
            terminal.last_content.mode = Modes::MOUSE_MODE;
            terminal.take_pty_write_log();

            // Past the end of the line: nothing link-like under the cursor.
            let click_position = point(px(370.0), px(10.0));
            ctrl_mouse_down_at(terminal, click_position, cx);
            ctrl_mouse_up_at(terminal, click_position, cx);

            assert!(
                !terminal
                    .events
                    .iter()
                    .any(|event| matches!(event, InternalEvent::ProcessHyperlink(_, _))),
                "a secondary click off a link must not open anything"
            );
            let pty_writes = terminal.take_pty_write_log();
            assert_eq!(
                pty_writes.len(),
                2,
                "expected press and release reports, got {pty_writes:?}"
            );
        });
    }

    #[gpui::test]
    async fn test_ctrl_click_in_mouse_mode_forwards_when_setting_disabled(cx: &mut TestAppContext) {
        let terminal = init_ctrl_click_hyperlink_test(cx, b"Visit https://zed.dev/ for more\r\n");

        cx.update_global(|store: &mut settings::SettingsStore, cx| {
            store.update_user_settings(cx, |settings| {
                settings
                    .terminal
                    .get_or_insert_default()
                    .open_links_in_mouse_mode = Some(false);
            });
        });

        terminal.update(cx, |terminal, cx| {
            terminal.last_content.mode = Modes::MOUSE_MODE;

            let click_position = point(px(80.0), px(10.0));
            ctrl_mouse_down_at(terminal, click_position, cx);
            ctrl_mouse_up_at(terminal, click_position, cx);

            assert!(
                !terminal
                    .events
                    .iter()
                    .any(|event| matches!(event, InternalEvent::ProcessHyperlink(_, _))),
                "with the setting disabled, ctrl+click must not open links in mouse mode"
            );
            let pty_writes = terminal.take_pty_write_log();
            assert_eq!(
                pty_writes.len(),
                2,
                "expected press and release reports, got {pty_writes:?}"
            );
        });
    }

    #[gpui::test]
    async fn test_hyperlink_ctrl_click_drag_outside_bounds(cx: &mut TestAppContext) {
        let terminal = init_ctrl_click_hyperlink_test(
            cx,
            b"Visit https://zed.dev/ for more\r\nThis is another line\r\n",
        );

        terminal.update(cx, |terminal, cx| {
            let down_position = point(px(80.0), px(10.0));
            let up_position = point(px(10.0), px(50.0));

            ctrl_mouse_down_at(terminal, down_position, cx);
            ctrl_mouse_move_to(terminal, up_position, cx);
            ctrl_mouse_up_at(terminal, up_position, cx);

            assert!(
                !terminal
                    .events
                    .iter()
                    .any(|event| matches!(event, InternalEvent::ProcessHyperlink(_, _))),
                "Should NOT have ProcessHyperlink event when dragging outside the hyperlink"
            );
        });
    }

    #[gpui::test]
    async fn test_hyperlink_ctrl_click_drag_within_bounds(cx: &mut TestAppContext) {
        let terminal = init_ctrl_click_hyperlink_test(cx, b"Visit https://zed.dev/ for more\r\n");

        terminal.update(cx, |terminal, cx| {
            let down_position = point(px(70.0), px(10.0));
            let up_position = point(px(130.0), px(10.0));

            ctrl_mouse_down_at(terminal, down_position, cx);
            ctrl_mouse_move_to(terminal, up_position, cx);
            ctrl_mouse_up_at(terminal, up_position, cx);

            assert!(
                terminal
                    .events
                    .iter()
                    .any(|event| matches!(event, InternalEvent::ProcessHyperlink(_, true))),
                "Should have ProcessHyperlink event when dragging within hyperlink bounds"
            );
        });
    }

    /// Polls the terminal content until `expected` appears, or panics after ~1s.
    /// The PTY IO thread writes into the terminal grid independently of the
    /// GPUI executor, so we need a real-time polling loop to synchronize.
    async fn assert_content_eventually(
        terminal: &Entity<Terminal>,
        expected: &str,
        cx: &mut TestAppContext,
    ) {
        let mut content = String::new();
        for _ in 0..100 {
            content = terminal.update(cx, |term, _| term.get_content());
            if content.contains(expected) {
                return;
            }
            cx.background_executor
                .timer(Duration::from_millis(10))
                .await;
        }
        panic!("Expected terminal content to contain {expected:?}, got: {content}");
    }

    #[cfg(unix)]
    async fn assert_foreground_process_command_eventually(
        terminal: &Entity<Terminal>,
        expected: &str,
        cx: &mut TestAppContext,
    ) {
        let mut command_name = None;
        for _ in 0..100 {
            terminal.update(cx, |terminal, _| {
                if let TerminalType::Pty { info, .. } = &terminal.terminal_type {
                    info.load_for_test();
                }
            });
            command_name =
                terminal.update(cx, |terminal, _| terminal.foreground_process_command_name());
            if command_name.as_deref() == Some(expected) {
                return;
            }
            cx.background_executor
                .timer(Duration::from_millis(10))
                .await;
        }
        let process_info = terminal.update(cx, |terminal, _| match &terminal.terminal_type {
            TerminalType::Pty { info, .. } => format!(
                "pid={:?}, fallback_pid={:?}, has_current_info={}",
                info.pid(),
                info.pid_getter().fallback_pid(),
                info.current.read().is_some()
            ),
            TerminalType::DisplayOnly => "display-only".to_string(),
        });
        panic!(
            "Expected foreground process command name to be {expected:?}, got {command_name:?}; process info: {process_info:?}"
        );
    }

    /// Test that kill_active_task properly terminates both the foreground process
    /// and the shell, allowing wait_for_completed_task to complete and output to be captured.
    #[cfg(unix)]
    #[gpui::test]
    async fn test_kill_active_task_completes_and_captures_output(cx: &mut TestAppContext) {
        cx.executor().allow_parking();

        // Run a command that prints output then sleeps for a long time
        // The echo ensures we have output to capture before killing
        let (terminal, completion_rx) =
            build_test_terminal(cx, "echo", &["test_output_before_kill; sleep 60"]).await;

        assert_content_eventually(&terminal, "test_output_before_kill", cx).await;

        // Kill the active task
        terminal.update(cx, |term, _cx| {
            term.kill_active_task();
        });

        // wait_for_completed_task should complete within a reasonable time (not hang)
        let completion_result = completion_rx.recv().await;
        assert!(
            completion_result.is_ok(),
            "wait_for_completed_task should complete after kill_active_task, but it timed out"
        );

        // The exit status should indicate the process was killed (not a clean exit)
        let exit_status = completion_result.unwrap();
        assert!(
            exit_status.is_some(),
            "Should have received an exit status after killing"
        );

        // Verify that output captured before killing is still available
        let content = terminal.update(cx, |term, _| term.get_content());
        assert!(
            content.contains("test_output_before_kill"),
            "Output from before kill should be captured, got: {content}"
        );
    }

    /// Test that kill_active_task on a task that's not running is a no-op
    #[gpui::test]
    async fn test_kill_active_task_on_completed_task_is_noop(cx: &mut TestAppContext) {
        cx.executor().allow_parking();

        // Run a command that exits immediately
        let (terminal, completion_rx) = build_test_terminal(cx, "echo", &["done"]).await;

        // Wait for the command to complete naturally
        let exit_status = completion_rx
            .recv()
            .await
            .expect("Should receive exit status");
        assert_eq!(exit_status, Some(ExitStatus::default()));

        assert_content_eventually(&terminal, "done", cx).await;

        // Now try to kill - should be a no-op since task already completed
        terminal.update(cx, |term, _cx| {
            term.kill_active_task();
        });

        // Content should still be there
        let content = terminal.update(cx, |term, _| term.get_content());
        assert!(
            content.contains("done"),
            "Output should still be present after no-op kill, got: {content}"
        );
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn test_stop_idle_interactive_process_preserves_companion(cx: &mut TestAppContext) {
        assert_stop_interactive_process_preserves_companion(false, cx).await;
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn test_stop_foreground_interactive_process_preserves_companion(cx: &mut TestAppContext) {
        assert_stop_interactive_process_preserves_companion(true, cx).await;
    }

    #[cfg(unix)]
    async fn assert_stop_interactive_process_preserves_companion(
        run_foreground_command: bool,
        cx: &mut TestAppContext,
    ) {
        init_test(cx);
        cx.executor().allow_parking();
        let mut terminals = Vec::new();
        for _ in 0..2 {
            let builder = cx
                .update(|cx| {
                    TerminalBuilder::new(
                        None,
                        None,
                        Shell::Program("/bin/sh".into()),
                        HashMap::default(),
                        SettingsCursorShape::default(),
                        AlternateScroll::On,
                        None,
                        Vec::new(),
                        0,
                        false,
                        0,
                        None,
                        cx,
                        Vec::new(),
                        PathStyle::local(),
                    )
                })
                .await
                .expect("actual interactive shell");
            terminals.push(cx.new(|cx| builder.subscribe(cx)));
        }
        let stopped = terminals.first().expect("target shell").clone();
        let companion = terminals.get(1).expect("companion shell").clone();
        if run_foreground_command {
            stopped.update(cx, |terminal, _| {
                terminal.input(b"printf 'FOREGROUND_STOP_%s\\n' 20261005; sleep 120\n".as_slice());
            });
            assert_content_eventually(&stopped, "FOREGROUND_STOP_20261005", cx).await;
            assert_foreground_process_command_eventually(&stopped, "sleep", cx).await;
        }
        let foreground_process_id = stopped
            .read_with(cx, |terminal, _| terminal.pid())
            .expect("actual owned foreground process");
        let close_seen = Arc::new(Mutex::new(false));
        let _subscription = cx.update(|cx| {
            let close_seen = close_seen.clone();
            cx.subscribe(&stopped, move |_, event: &Event, _| {
                if matches!(event, Event::CloseTerminal) {
                    *close_seen.lock() = true;
                }
            })
        });
        stopped.update(cx, |terminal, _| {
            assert!(terminal.is_pty());
            assert!(terminal.task().is_none());
            assert!(terminal.can_reuse_interactive_process());
            assert_eq!(terminal.keyboard_input_sent(), run_foreground_command);
            terminal
                .request_stop_interactive_process()
                .expect("deliver an idle-shell stop request");
            assert!(
                !terminal.can_reuse_interactive_process(),
                "stop-pending executions are ineligible before the exit event"
            );
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        while !*close_seen.lock() {
            cx.run_until_parked();
            assert!(Instant::now() < deadline, "actual idle shell must exit");
            cx.background_executor
                .timer(Duration::from_millis(50))
                .await;
        }
        stopped.update(cx, |terminal, _| {
            assert!(terminal.child_exited.is_some());
            assert!(!terminal.can_reuse_interactive_process());
            assert_eq!(terminal.keyboard_input_sent(), run_foreground_command);
            terminal
                .request_stop_interactive_process()
                .expect("retained stopped shell must not be signalled again");
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut system = sysinfo::System::new();
        loop {
            system.refresh_processes_specifics(
                sysinfo::ProcessesToUpdate::Some(&[foreground_process_id]),
                true,
                sysinfo::ProcessRefreshKind::nothing(),
            );
            if system.process(foreground_process_id).is_none() {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "owned foreground process must stop"
            );
            cx.background_executor
                .timer(Duration::from_millis(50))
                .await;
        }
        if run_foreground_command {
            assert!(stopped.read_with(cx, |terminal, _| {
                terminal.get_content().contains("FOREGROUND_STOP_20261005")
            }));
        }
        companion.update(cx, |terminal, _| {
            assert!(terminal.child_exited.is_none());
            terminal.input(b"printf 'COMPANION_AFTER_TARGETED_STOP_%s\\n' 20261005\n".as_slice());
        });
        assert_content_eventually(&companion, "COMPANION_AFTER_TARGETED_STOP_20261005", cx).await;
        companion.update(cx, |terminal, _| terminal.input(b"exit\n".as_slice()));
        let deadline = Instant::now() + Duration::from_secs(10);
        while companion.read_with(cx, |terminal, _| terminal.child_exited.is_none()) {
            cx.run_until_parked();
            assert!(
                Instant::now() < deadline,
                "owned companion must cleanly exit"
            );
            cx.background_executor
                .timer(Duration::from_millis(50))
                .await;
        }
    }

    #[cfg(unix)]
    #[gpui::test]
    async fn test_stop_interactive_process_ignores_exit_without_status(cx: &mut TestAppContext) {
        init_test(cx);
        cx.executor().allow_parking();
        let builder = cx
            .update(|cx| {
                TerminalBuilder::new(
                    None,
                    None,
                    Shell::Program("/bin/sh".into()),
                    HashMap::default(),
                    SettingsCursorShape::default(),
                    AlternateScroll::On,
                    None,
                    Vec::new(),
                    0,
                    false,
                    0,
                    None,
                    cx,
                    Vec::new(),
                    PathStyle::local(),
                )
            })
            .await
            .expect("actual interactive shell");
        let terminal = cx.new(|cx| builder.subscribe(cx));
        terminal.update(cx, |terminal, cx| {
            // An EOF notification can arrive without a numeric child exit status.
            terminal.process_event(TerminalBackendEvent::Exit, cx);
            assert!(terminal.process_finished);
            assert!(terminal.child_exited.is_none());
            terminal
                .request_stop_interactive_process()
                .expect("completed backend must not signal its retained process identifier");
            assert!(!terminal.user_stop_requested);
            terminal.input(b"printf 'STATUSLESS_EXIT_%s\\n' 20261005\n".as_slice());
        });
        assert_content_eventually(&terminal, "STATUSLESS_EXIT_20261005", cx).await;
        terminal.update(cx, |terminal, _| terminal.input(b"exit\n".as_slice()));
        let deadline = Instant::now() + Duration::from_secs(10);
        while terminal.read_with(cx, |terminal, _| terminal.child_exited.is_none()) {
            cx.run_until_parked();
            assert!(Instant::now() < deadline, "owned shell must cleanly exit");
            cx.background_executor
                .timer(Duration::from_millis(50))
                .await;
        }
    }

    #[gpui::test]
    async fn test_stop_interactive_process_rejects_readonly_history(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        let builder = cx.update(|cx| {
            TerminalBuilder::new_display_only(
                SettingsCursorShape::default(),
                AlternateScroll::On,
                None,
                0,
                cx.background_executor(),
                PathStyle::local(),
            )
        });
        let terminal = cx.new(|cx| builder.subscribe(cx));
        terminal.update(cx, |terminal, _| {
            assert!(terminal.request_stop_interactive_process().is_err());
            assert!(!terminal.user_stop_requested);
            assert!(!terminal.is_pty());
            assert!(terminal.pid().is_none());
        });
    }

    mod perf {
        use super::super::*;
        use gpui::{
            Entity, ScrollDelta, ScrollWheelEvent, TestAppContext, VisualContext,
            VisualTestContext, point,
        };
        use util::default;
        use util_macros::perf;

        async fn init_scroll_perf_test(
            cx: &mut TestAppContext,
        ) -> (Entity<Terminal>, &mut VisualTestContext) {
            cx.update(|cx| {
                let settings_store = settings::SettingsStore::test(cx);
                cx.set_global(settings_store);
            });

            cx.executor().allow_parking();

            let window = cx.add_empty_window();
            let builder = window
                .update(|window, cx| {
                    let settings = TerminalSettings::get_global(cx);
                    let test_path_hyperlink_timeout_ms = 100;
                    TerminalBuilder::new(
                        None,
                        None,
                        task::Shell::System,
                        HashMap::default(),
                        SettingsCursorShape::default(),
                        AlternateScroll::On,
                        None,
                        settings.path_hyperlink_regexes.clone(),
                        test_path_hyperlink_timeout_ms,
                        false,
                        window.window_handle().window_id().as_u64(),
                        None,
                        cx,
                        vec![],
                        PathStyle::local(),
                    )
                })
                .await
                .unwrap();
            let terminal = window.new(|cx| builder.subscribe(cx));

            terminal.update(window, |term, cx| {
                term.write_output("long line ".repeat(1000).as_bytes(), cx);
            });

            (terminal, window)
        }

        #[perf]
        #[gpui::test]
        async fn scroll_long_line_benchmark(cx: &mut TestAppContext) {
            let (terminal, window) = init_scroll_perf_test(cx).await;
            let wobble = point(FIND_HYPERLINK_THROTTLE_PX, px(0.0));
            let mut scroll_by = |lines: i32| {
                window.update_window_entity(&terminal, |terminal, window, cx| {
                    let bounds = terminal.last_content.terminal_bounds.bounds;
                    let center = bounds.origin + bounds.center();
                    let position = center + wobble * lines as f32;

                    terminal.mouse_move(
                        &MouseMoveEvent {
                            position,
                            ..default()
                        },
                        cx,
                    );

                    terminal.scroll_wheel(
                        &ScrollWheelEvent {
                            position,
                            delta: ScrollDelta::Lines(GpuiPoint::new(0.0, lines as f32)),
                            ..default()
                        },
                        1.0,
                    );

                    assert!(
                        terminal
                            .events
                            .iter()
                            .any(|event| matches!(event, InternalEvent::Scroll(_))),
                        "Should have Scroll event when scrolling within terminal bounds"
                    );
                    terminal.sync(window, cx);
                });
            };

            for _ in 0..20000 {
                scroll_by(1);
                scroll_by(-1);
            }
        }

        #[test]
        fn test_num_lines_float_precision() {
            let line_heights = [
                20.1f32, 16.7, 18.3, 22.9, 14.1, 15.6, 17.8, 19.4, 21.3, 23.7,
            ];
            for &line_height in &line_heights {
                for n in 1..=100 {
                    let height = n as f32 * line_height;
                    let bounds = TerminalBounds::new(
                        px(line_height),
                        px(8.0),
                        Bounds {
                            origin: GpuiPoint::default(),
                            size: Size {
                                width: px(800.0),
                                height: px(height),
                            },
                        },
                    );
                    assert_eq!(
                        bounds.num_lines(),
                        n,
                        "num_lines() should be {n} for height={height}, line_height={line_height}"
                    );
                }
            }
        }

        #[test]
        fn test_num_columns_float_precision() {
            let cell_widths = [8.1f32, 7.3, 9.7, 6.9, 10.1];
            for &cell_width in &cell_widths {
                for n in 1..=200 {
                    let width = n as f32 * cell_width;
                    let bounds = TerminalBounds::new(
                        px(20.0),
                        px(cell_width),
                        Bounds {
                            origin: GpuiPoint::default(),
                            size: Size {
                                width: px(width),
                                height: px(400.0),
                            },
                        },
                    );
                    assert_eq!(
                        bounds.num_columns(),
                        n,
                        "num_columns() should be {n} for width={width}, cell_width={cell_width}"
                    );
                }
            }
        }
    }

    fn make_display_only_terminal() -> Terminal {
        let dispatcher = gpui::TestDispatcher::new(rand::random());
        let executor = gpui::BackgroundExecutor::new(std::sync::Arc::new(dispatcher));
        TerminalBuilder::new_display_only(
            SettingsCursorShape::default(),
            AlternateScroll::On,
            None,
            0,
            &executor,
            PathStyle::local(),
        )
        .terminal
    }

    #[test]
    fn test_cwd_at_line_empty_history_returns_none() {
        let terminal = make_display_only_terminal();
        assert_eq!(terminal.cwd_at_line(0, 0), None);
    }

    #[test]
    fn test_cwd_at_line_returns_cwd_for_line_at_or_after_recorded_position() {
        let mut terminal = make_display_only_terminal();
        let working_directory_a = PathBuf::from("/home/user/project_a");
        terminal.cwd_history.push(CwdHistoryEntry {
            scrollback_position: 5,
            working_directory: working_directory_a.clone(),
        });

        // click_pos = history_size(5) + line(3) = 8 >= 5
        assert_eq!(
            terminal.cwd_at_line(3, 5),
            Some(working_directory_a.clone())
        );
        // click_pos = history_size(5) + line(0) = 5 == 5 (exact match)
        assert_eq!(terminal.cwd_at_line(0, 5), Some(working_directory_a));
    }

    #[test]
    fn test_cwd_at_line_ignores_history_at_scrollback_cap() {
        let mut terminal = make_display_only_terminal();
        terminal.term_config.scrolling_history = 10;
        terminal.cwd_history.push(CwdHistoryEntry {
            scrollback_position: 0,
            working_directory: PathBuf::from("/stale/cwd"),
        });

        assert_eq!(terminal.cwd_at_line(-5, 10), None);
    }

    #[test]
    fn test_cwd_at_line_returns_none_when_line_is_before_any_recorded_cwd() {
        let mut terminal = make_display_only_terminal();
        terminal.cwd_history.push(CwdHistoryEntry {
            scrollback_position: 10,
            working_directory: PathBuf::from("/home/user/project_a"),
        });

        // click_pos = 0 + 3 = 3 < 10, no match, falls back to working_directory (None)
        assert_eq!(terminal.cwd_at_line(3, 0), None);
    }

    #[test]
    fn test_cwd_at_line_selects_most_recent_cwd_before_click() {
        let mut terminal = make_display_only_terminal();
        let working_directory_a = PathBuf::from("/home/user/project_a");
        let working_directory_b = PathBuf::from("/home/user/project_b");
        let working_directory_c = PathBuf::from("/home/user/project_c");
        terminal.cwd_history.push(CwdHistoryEntry {
            scrollback_position: 0,
            working_directory: working_directory_a.clone(),
        });
        terminal.cwd_history.push(CwdHistoryEntry {
            scrollback_position: 10,
            working_directory: working_directory_b.clone(),
        });
        terminal.cwd_history.push(CwdHistoryEntry {
            scrollback_position: 20,
            working_directory: working_directory_c.clone(),
        });

        // click_pos=5: between 0 and 10, working_directory_a
        assert_eq!(terminal.cwd_at_line(5, 0), Some(working_directory_a));
        // click_pos=15: between 10 and 20, working_directory_b
        assert_eq!(terminal.cwd_at_line(15, 0), Some(working_directory_b));
        // click_pos=25: after 20, working_directory_c
        assert_eq!(terminal.cwd_at_line(25, 0), Some(working_directory_c));
    }

    #[test]
    fn test_record_cwd_change_stores_entry_at_current_cursor_position() {
        let mut terminal = make_display_only_terminal();
        let working_directory = PathBuf::from("/tmp/test");
        terminal.record_cwd_change(working_directory.clone());

        assert_eq!(terminal.cwd_history.len(), 1);
        let entry = &terminal.cwd_history[0];
        assert_eq!(entry.scrollback_position, 0);
        assert_eq!(entry.working_directory, working_directory);
    }

    #[test]
    fn test_record_cwd_change_uses_command_boundary() {
        let mut terminal = make_display_only_terminal();
        terminal.write_input(b"\r".to_vec());
        assert_eq!(terminal.pending_cwd_boundary, Some(0));

        let working_directory = PathBuf::from("/tmp/test");
        terminal.record_cwd_change(working_directory.clone());

        assert_eq!(terminal.pending_cwd_boundary, None);
        assert_eq!(
            terminal.cwd_history,
            vec![CwdHistoryEntry {
                scrollback_position: 0,
                working_directory,
            }]
        );
    }

    #[test]
    fn test_remote_terminal_does_not_record_local_cwd() {
        let mut terminal = make_display_only_terminal();
        terminal.is_remote_terminal = true;
        terminal.write_input(b"\r".to_vec());
        terminal.record_cwd_change(PathBuf::from("/local/ssh/cwd"));

        assert_eq!(terminal.pending_cwd_boundary, None);
        assert!(terminal.cwd_history.is_empty());
        assert_eq!(terminal.cwd_at_line(0, 0), None);
    }

    #[test]
    fn test_reset_cwd_history_discards_stale_coordinates() {
        let mut terminal = make_display_only_terminal();
        terminal.cwd_history.push(CwdHistoryEntry {
            scrollback_position: 42,
            working_directory: PathBuf::from("/tmp/test"),
        });
        terminal.pending_cwd_boundary = Some(43);

        terminal.reset_cwd_history();

        assert!(terminal.cwd_history.is_empty());
        assert_eq!(terminal.pending_cwd_boundary, None);
    }
}
