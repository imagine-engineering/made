use crate::{
    Edit, EditFileTool, EditFileToolInput, GrepTool, GrepToolInput, ListDirectoryTool,
    ListDirectoryToolInput, ReadFileTool, ReadFileToolInput, TerminalTool, TerminalToolInput,
};
use agent_client_protocol::schema::v1 as acp;
use serde::Deserialize;
use serde_json::Value;
use std::{collections::HashSet, error::Error, fmt, path::Component, path::Path, path::PathBuf};

const MAX_PATH_BYTES: usize = 4_096;
const MAX_COMMAND_BYTES: usize = 1_000_000;
const MAX_EDIT_BYTES: usize = 1_000_000;
const MAX_READ_LINES: u32 = 2_000;
const MAX_TODOS: usize = 50;
const MAX_TODO_CHARACTERS: usize = 200;
const MAX_TIMEOUT_MS: u64 = 600_000;
const MAX_OUTPUT_BYTES: usize = 8 * 1_024;
const MAX_OUTPUT_LINES: usize = 2_000;

pub struct BuzzToolRequest {
    pub name: String,
    pub arguments: Value,
}

impl fmt::Debug for BuzzToolRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BuzzToolRequest")
            .field("name", &self.name)
            .field("arguments", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, PartialEq)]
pub struct NativeBuzzToolCall {
    pub tool_name: &'static str,
    pub arguments: Value,
}

impl fmt::Debug for NativeBuzzToolCall {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeBuzzToolCall")
            .field("tool_name", &self.tool_name)
            .field("arguments", &"<redacted>")
            .finish()
    }
}

#[derive(Clone, PartialEq)]
pub enum NativeBuzzToolRequest {
    Tool(NativeBuzzToolCall),
    CurrentPlan,
    ReplacePlan(acp::Plan),
}

impl fmt::Debug for NativeBuzzToolRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tool(call) => call.fmt(formatter),
            Self::CurrentPlan => formatter.write_str("CurrentPlan"),
            Self::ReplacePlan(plan) => formatter
                .debug_struct("ReplacePlan")
                .field("entry_count", &plan.entries.len())
                .finish(),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum BuzzToolCompatibilityError {
    UnsupportedTool,
    InvalidArguments,
    InvalidPath,
    InputTooLarge,
    ReplaceAllUnsupported,
    Denied,
    Cancelled,
}

impl fmt::Display for BuzzToolCompatibilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedTool => formatter.write_str("unsupported Buzz tool"),
            Self::InvalidArguments => formatter.write_str("invalid Buzz tool arguments"),
            Self::InvalidPath => formatter.write_str("invalid project-relative path"),
            Self::InputTooLarge => formatter.write_str("Buzz tool input exceeds its limit"),
            Self::ReplaceAllUnsupported => {
                formatter.write_str("replace_all cannot preserve native edit permission semantics")
            }
            Self::Denied => formatter.write_str("native tool is unavailable or denied"),
            Self::Cancelled => formatter.write_str("Buzz tool request was cancelled"),
        }
    }
}

impl Error for BuzzToolCompatibilityError {}

pub struct BuzzToolCompatibilityMapper {
    default_worktree: String,
}

impl BuzzToolCompatibilityMapper {
    pub fn new(default_worktree: impl Into<String>) -> Result<Self, BuzzToolCompatibilityError> {
        let default_worktree = normalize_relative_path(&default_worktree.into(), false)?;
        Ok(Self { default_worktree })
    }

    pub fn map(
        &self,
        request: BuzzToolRequest,
        cancelled: bool,
        mut native_tool_available: impl FnMut(&str) -> bool,
    ) -> Result<NativeBuzzToolRequest, BuzzToolCompatibilityError> {
        if cancelled {
            return Err(BuzzToolCompatibilityError::Cancelled);
        }

        let mapped = match request.name.as_str() {
            "shell" => self.map_shell(request.arguments)?,
            "read" | "read_file" => self.map_read(request.arguments)?,
            "edit" | "str_replace" => self.map_edit(request.arguments)?,
            "search" | "rg" => self.map_search(request.arguments)?,
            "tree" => self.map_tree(request.arguments)?,
            "image" | "view_image" => self.map_image(request.arguments)?,
            "todo" => return self.map_todo(request.arguments),
            _ => return Err(BuzzToolCompatibilityError::UnsupportedTool),
        };
        if !native_tool_available(mapped.tool_name) {
            return Err(BuzzToolCompatibilityError::Denied);
        }
        Ok(NativeBuzzToolRequest::Tool(mapped))
    }

    fn map_shell(
        &self,
        arguments: Value,
    ) -> Result<NativeBuzzToolCall, BuzzToolCompatibilityError> {
        let input: BuzzShellInput = decode(arguments)?;
        if input.command.is_empty() {
            return Err(BuzzToolCompatibilityError::InvalidArguments);
        }
        if input.command.len() > MAX_COMMAND_BYTES {
            return Err(BuzzToolCompatibilityError::InputTooLarge);
        }
        let cd = self.resolve_directory(input.workdir.as_deref())?;
        native_call::<TerminalTool, _>(TerminalToolInput {
            command: input.command,
            cd,
            timeout_ms: input.timeout_ms.map(|timeout| timeout.min(MAX_TIMEOUT_MS)),
            head_lines: None,
            tail_lines: Some(MAX_OUTPUT_LINES),
        })
    }

    fn map_read(&self, arguments: Value) -> Result<NativeBuzzToolCall, BuzzToolCompatibilityError> {
        let input: BuzzReadInput = decode(arguments)?;
        let path = self.resolve_path(&input.path, input.workdir.as_deref(), false)?;
        let start_line = input
            .offset
            .unwrap_or(0)
            .checked_add(1)
            .and_then(|line| u32::try_from(line).ok())
            .ok_or(BuzzToolCompatibilityError::InvalidArguments)?;
        let limit = input.limit.unwrap_or(MAX_READ_LINES as usize);
        if limit == 0 {
            return Err(BuzzToolCompatibilityError::InvalidArguments);
        }
        let limit = u32::try_from(limit.min(MAX_READ_LINES as usize))
            .map_err(|_| BuzzToolCompatibilityError::InvalidArguments)?;
        let end_line = Some(start_line.saturating_add(limit - 1));
        native_call::<ReadFileTool, _>(ReadFileToolInput {
            path,
            start_line: Some(start_line),
            end_line,
        })
    }

    fn map_edit(&self, arguments: Value) -> Result<NativeBuzzToolCall, BuzzToolCompatibilityError> {
        let input: BuzzEditInput = decode(arguments)?;
        if input.replace_all {
            return Err(BuzzToolCompatibilityError::ReplaceAllUnsupported);
        }
        if input.old_str.is_empty()
            || input.old_str.len() > MAX_EDIT_BYTES
            || input.new_str.len() > MAX_EDIT_BYTES
        {
            return Err(BuzzToolCompatibilityError::InputTooLarge);
        }
        let path =
            PathBuf::from(self.resolve_path(&input.path, input.workdir.as_deref(), false)?);
        native_call::<EditFileTool, _>(EditFileToolInput {
            path,
            edits: vec![Edit {
                old_text: input.old_str,
                new_text: input.new_str,
            }],
        })
    }

    fn map_search(
        &self,
        arguments: Value,
    ) -> Result<NativeBuzzToolCall, BuzzToolCompatibilityError> {
        let input: BuzzSearchInput = decode(arguments)?;
        if input.regex.is_empty() || input.regex.len() > MAX_EDIT_BYTES {
            return Err(BuzzToolCompatibilityError::InputTooLarge);
        }
        let glob = input
            .glob
            .map(|glob| normalize_relative_path(&glob, false))
            .transpose()?;
        let search_root = match input.path {
            Some(path) => self.resolve_path(&path, input.workdir.as_deref(), true)?,
            None => self.resolve_directory(input.workdir.as_deref())?,
        };
        let include_pattern = Some(match glob {
            Some(glob) => format!("{search_root}/{glob}"),
            None => format!("{search_root}/**/*"),
        });
        native_call::<GrepTool, _>(GrepToolInput {
            regex: input.regex,
            include_pattern,
            offset: input.offset.unwrap_or(0),
            case_sensitive: input.case_sensitive,
        })
    }

    fn map_tree(&self, arguments: Value) -> Result<NativeBuzzToolCall, BuzzToolCompatibilityError> {
        let input: BuzzTreeInput = decode(arguments)?;
        if input.depth.is_some_and(|depth| depth != 1) {
            return Err(BuzzToolCompatibilityError::InvalidArguments);
        }
        let path = self.resolve_path(
            input.path.as_deref().unwrap_or("."),
            input.workdir.as_deref(),
            true,
        )?;
        native_call::<ListDirectoryTool, _>(ListDirectoryToolInput { path })
    }

    fn map_image(
        &self,
        arguments: Value,
    ) -> Result<NativeBuzzToolCall, BuzzToolCompatibilityError> {
        let input: BuzzImageInput = decode(arguments)?;
        if input
            .max_dim
            .is_some_and(|maximum_dimension| !(64..=2_048).contains(&maximum_dimension))
        {
            return Err(BuzzToolCompatibilityError::InvalidArguments);
        }
        if input.source.starts_with("http://")
            || input.source.starts_with("https://")
            || input.source.starts_with("data:")
        {
            return Err(BuzzToolCompatibilityError::InvalidPath);
        }
        let path = self.resolve_path(&input.source, input.workdir.as_deref(), false)?;
        native_call::<ReadFileTool, _>(ReadFileToolInput {
            path,
            start_line: None,
            end_line: None,
        })
    }

    fn map_todo(
        &self,
        arguments: Value,
    ) -> Result<NativeBuzzToolRequest, BuzzToolCompatibilityError> {
        let input: BuzzTodoInput = decode(arguments)?;
        let Some(todos) = input.todos else {
            return Ok(NativeBuzzToolRequest::CurrentPlan);
        };
        if todos.len() > MAX_TODOS {
            return Err(BuzzToolCompatibilityError::InputTooLarge);
        }
        let mut entries = Vec::with_capacity(todos.len());
        let mut seen_todos = HashSet::with_capacity(todos.len());
        for todo in todos {
            let text = todo.text.trim();
            if text.is_empty()
                || text.chars().count() > MAX_TODO_CHARACTERS
                || text.chars().any(invalid_todo_character)
                || !seen_todos.insert(text.to_owned())
            {
                return Err(BuzzToolCompatibilityError::InvalidArguments);
            }
            entries.push(acp::PlanEntry::new(
                text,
                acp::PlanEntryPriority::Medium,
                if todo.done {
                    acp::PlanEntryStatus::Completed
                } else {
                    acp::PlanEntryStatus::Pending
                },
            ));
        }
        Ok(NativeBuzzToolRequest::ReplacePlan(acp::Plan::new(entries)))
    }

    fn resolve_directory(
        &self,
        workdir: Option<&str>,
    ) -> Result<String, BuzzToolCompatibilityError> {
        match workdir {
            Some(workdir) => normalize_relative_path(workdir, true),
            None => Ok(self.default_worktree.clone()),
        }
    }

    fn resolve_path(
        &self,
        path: &str,
        workdir: Option<&str>,
        directory: bool,
    ) -> Result<String, BuzzToolCompatibilityError> {
        let base = self.resolve_directory(workdir)?;
        let path = normalize_relative_path(path, directory)?;
        if path == "." {
            return Ok(base);
        }
        if path == base || path.starts_with(&format!("{base}/")) {
            Ok(path)
        } else {
            normalize_relative_path(&format!("{base}/{path}"), directory)
        }
    }
}

fn invalid_todo_character(character: char) -> bool {
    character.is_control()
        || (character.is_whitespace() && character != ' ')
        || matches!(
            character,
            '\u{200B}'..='\u{200F}'
                | '\u{202A}'..='\u{202E}'
                | '\u{2060}'..='\u{206F}'
                | '\u{FEFF}'
        )
}

pub fn bound_buzz_tool_output(output: &str) -> String {
    let mut lines = output
        .lines()
        .rev()
        .take(MAX_OUTPUT_LINES)
        .collect::<Vec<_>>();
    lines.reverse();
    let output = lines.join("\n");
    if output.len() <= MAX_OUTPUT_BYTES {
        return output;
    }
    let mut start = output.len() - MAX_OUTPUT_BYTES;
    while !output.is_char_boundary(start) {
        start += 1;
    }
    format!("[output truncated]\n{}", &output[start..])
}

fn native_call<Tool, Input>(input: Input) -> Result<NativeBuzzToolCall, BuzzToolCompatibilityError>
where
    Tool: crate::AgentTool<Input = Input>,
    Input: serde::Serialize,
{
    Ok(NativeBuzzToolCall {
        tool_name: Tool::NAME,
        arguments: serde_json::to_value(input)
            .map_err(|_| BuzzToolCompatibilityError::InvalidArguments)?,
    })
}

fn decode<T: for<'de> Deserialize<'de>>(value: Value) -> Result<T, BuzzToolCompatibilityError> {
    serde_json::from_value(value).map_err(|_| BuzzToolCompatibilityError::InvalidArguments)
}

fn normalize_relative_path(
    path: &str,
    allow_current: bool,
) -> Result<String, BuzzToolCompatibilityError> {
    if path.is_empty()
        || path.len() > MAX_PATH_BYTES
        || path.chars().any(char::is_control)
        || Path::new(path).is_absolute()
    {
        return Err(BuzzToolCompatibilityError::InvalidPath);
    }
    let mut parts = Vec::new();
    for component in Path::new(path).components() {
        match component {
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(BuzzToolCompatibilityError::InvalidPath);
            }
        }
    }
    if parts.is_empty() {
        if allow_current {
            Ok(".".to_owned())
        } else {
            Err(BuzzToolCompatibilityError::InvalidPath)
        }
    } else {
        Ok(parts.join("/"))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuzzShellInput {
    command: String,
    #[serde(default)]
    workdir: Option<String>,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuzzReadInput {
    path: String,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    workdir: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuzzEditInput {
    path: String,
    old_str: String,
    new_str: String,
    #[serde(default)]
    replace_all: bool,
    #[serde(default)]
    workdir: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuzzSearchInput {
    #[serde(alias = "pattern")]
    regex: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    glob: Option<String>,
    #[serde(default)]
    workdir: Option<String>,
    #[serde(default)]
    offset: Option<u32>,
    #[serde(default)]
    case_sensitive: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuzzTreeInput {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    workdir: Option<String>,
    #[serde(default)]
    depth: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuzzImageInput {
    source: String,
    #[serde(default)]
    max_dim: Option<u32>,
    #[serde(default)]
    workdir: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuzzTodoInput {
    #[serde(default)]
    todos: Option<Vec<BuzzTodoItem>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BuzzTodoItem {
    text: String,
    #[serde(default)]
    done: bool,
}
