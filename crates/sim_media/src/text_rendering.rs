use freetype_sys as ft;
use sha2::{Digest, Sha256};
use sim_tensor::{CpuBackend, ExecutionContext, ImageTensor, TensorError};
use sim_types::{CancellationError, CancellationToken};
use std::{ptr, slice};
use thiserror::Error;

const PILLOW_DEFAULT_FONT: &[u8] = include_bytes!("../assets/pillow-default-font.ttf");
pub const PILLOW_DEFAULT_FONT_SHA256: &str =
    "69853909b940023570964e29cffe30da95aea8de3627736b5cd15ab30143169f";
pub const MAX_TEXT_CHARACTERS: usize = 1_000_000;
pub const MAX_TEXT_RUNS: usize = 4_096;
pub const MAX_FONT_SIZE: u32 = 4_096;
pub const MAX_TEXT_RENDER_PIXELS: u64 = 16_777_216;

#[derive(Clone, Debug, PartialEq)]
pub struct TextMetrics {
    pub advance: f32,
    pub ascent: f32,
    pub descent: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextRun {
    pub text: String,
    pub x: i32,
    pub y: i32,
    pub font_size: u32,
    pub color: [u8; 4],
}

#[derive(Clone, Debug, Default)]
pub struct NativeMediaTextRenderer;

#[derive(Debug, Error)]
pub enum TextRenderingError {
    #[error("the pinned Pillow default font is invalid")]
    InvalidPinnedFont,
    #[error("font size must be in 1..={MAX_FONT_SIZE}, got {0}")]
    InvalidFontSize(u32),
    #[error("text exceeds Pillow's {MAX_TEXT_CHARACTERS}-character limit")]
    TextTooLong,
    #[error("text run count exceeds {MAX_TEXT_RUNS}")]
    TooManyRuns,
    #[error(
        "render dimensions must be nonzero and contain at most {MAX_TEXT_RENDER_PIXELS} pixels"
    )]
    InvalidDimensions,
    #[error("RGBA background storage does not match the render dimensions")]
    InvalidBackground,
    #[error("text layout or raster allocation failed: {0}")]
    Allocation(String),
    #[error(transparent)]
    Cancelled(#[from] CancellationError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
}

#[derive(Clone, Copy, Debug)]
struct PositionedGlyph {
    id: ft::FT_UInt,
    x: i32,
}

impl NativeMediaTextRenderer {
    pub fn new() -> Result<Self, TextRenderingError> {
        let digest = hex_sha256(PILLOW_DEFAULT_FONT);
        let face = FreeTypeFace::new(10)?;
        if digest != PILLOW_DEFAULT_FONT_SHA256 || face.units_per_em() == 0 {
            return Err(TextRenderingError::InvalidPinnedFont);
        }
        Ok(Self)
    }

    pub fn font_sha256(&self) -> &'static str {
        PILLOW_DEFAULT_FONT_SHA256
    }

    pub fn measure(
        &self,
        text: &str,
        font_size: u32,
        cancellation: &CancellationToken,
    ) -> Result<TextMetrics, TextRenderingError> {
        cancellation.check()?;
        validate_text(text)?;
        validate_font_size(font_size)?;
        let mut face = FreeTypeFace::new(font_size)?;
        let (glyphs, advance) = face.shape(text, cancellation)?;
        drop(glyphs);
        Ok(TextMetrics {
            advance: advance as f32,
            ascent: face.ascent() as f32,
            descent: face.descent() as f32,
        })
    }

    pub fn wrap(
        &self,
        text: &str,
        font_size: u32,
        maximum_width: f32,
        cancellation: &CancellationToken,
    ) -> Result<Vec<String>, TextRenderingError> {
        cancellation.check()?;
        validate_text(text)?;
        validate_font_size(font_size)?;
        if !maximum_width.is_finite() || maximum_width < 0.0 {
            return Err(TextRenderingError::InvalidDimensions);
        }
        let mut lines = Vec::new();
        lines
            .try_reserve(text.split('\n').count())
            .map_err(|error| TextRenderingError::Allocation(error.to_string()))?;
        for paragraph in text.split('\n') {
            cancellation.check()?;
            let mut line = String::new();
            for (word_index, word) in paragraph.split_whitespace().enumerate() {
                if word_index & 0xff == 0 {
                    cancellation.check()?;
                }
                let test = if line.is_empty() {
                    try_owned(word)?
                } else {
                    let mut test = String::new();
                    test.try_reserve(line.len().saturating_add(word.len()).saturating_add(1))
                        .map_err(|error| TextRenderingError::Allocation(error.to_string()))?;
                    test.push_str(&line);
                    test.push(' ');
                    test.push_str(word);
                    test
                };
                if !line.is_empty()
                    && self.measure(&test, font_size, cancellation)?.advance > maximum_width
                {
                    lines
                        .try_reserve(1)
                        .map_err(|error| TextRenderingError::Allocation(error.to_string()))?;
                    lines.push(line);
                    line = try_owned(word)?;
                } else {
                    line = test;
                }
            }
            lines
                .try_reserve(1)
                .map_err(|error| TextRenderingError::Allocation(error.to_string()))?;
            lines.push(line);
        }
        Ok(lines)
    }

    pub fn render_over_rgba8(
        &self,
        width: u64,
        height: u64,
        background: &[u8],
        runs: &[TextRun],
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<ImageTensor, TextRenderingError> {
        context.check()?;
        if runs.len() > MAX_TEXT_RUNS {
            return Err(TextRenderingError::TooManyRuns);
        }
        let pixel_count = checked_pixel_count(width, height)?;
        let rgba_bytes = pixel_count
            .checked_mul(4)
            .ok_or(TextRenderingError::InvalidDimensions)?;
        if background.len() != rgba_bytes {
            return Err(TextRenderingError::InvalidBackground);
        }
        for run in runs {
            validate_text(&run.text)?;
            validate_font_size(run.font_size)?;
        }

        let mut overlay = backend.workspace_vec(context, rgba_bytes)?;
        for _ in 0..rgba_bytes {
            overlay.try_push(0_u8)?;
        }
        for run in runs {
            context.check()?;
            self.rasterize_run(run, width, height, &mut overlay, backend, context)?;
        }
        context.check()?;
        Ok(ImageTensor::alpha_composite_rgba8(
            background, &overlay, width, height, backend, context,
        )?)
    }

    fn rasterize_run(
        &self,
        run: &TextRun,
        width: u64,
        height: u64,
        overlay: &mut [u8],
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
    ) -> Result<(), TextRenderingError> {
        let mut face = FreeTypeFace::new(run.font_size)?;
        let (glyphs, _) = face.shape(&run.text, context.cancellation)?;
        let glyph_scratch_bytes = u64::from(run.font_size)
            .checked_mul(u64::from(run.font_size))
            .ok_or(TextRenderingError::InvalidDimensions)?;
        let _glyph_scratch = backend.reserve_workspace(context, glyph_scratch_bytes)?;
        let baseline = run.y.saturating_add(face.ascent());
        for (glyph_index, glyph) in glyphs.iter().enumerate() {
            if glyph_index & 0xff == 0 {
                context.check()?;
            }
            let bitmap = face.render(glyph.id)?;
            let image_pixels = u64::from(bitmap.width)
                .checked_mul(u64::from(bitmap.height))
                .ok_or(TextRenderingError::InvalidDimensions)?;
            if image_pixels > glyph_scratch_bytes || bitmap.data.len() != image_pixels as usize {
                return Err(TextRenderingError::InvalidDimensions);
            }
            let left = run.x.saturating_add(glyph.x).saturating_add(bitmap.left);
            let top = baseline.saturating_sub(bitmap.top);
            composite_mask(
                overlay,
                width,
                height,
                left,
                top,
                bitmap.width,
                bitmap.height,
                bitmap.data,
                run.color,
                context.cancellation,
            )?;
        }
        Ok(())
    }
}

pub fn zero_padded_numeric_tag(value: u64) -> String {
    format!("{value:02}")
}

struct FreeTypeFace {
    library: ft::FT_Library,
    face: ft::FT_Face,
}

struct RenderedGlyph<'a> {
    left: i32,
    top: i32,
    width: u32,
    height: u32,
    data: &'a [u8],
}

impl FreeTypeFace {
    fn new(font_size: u32) -> Result<Self, TextRenderingError> {
        let mut library = ptr::null_mut();
        // The wrapper owns the returned library and face and never exposes their raw handles.
        if unsafe { ft::FT_Init_FreeType(&mut library) } != 0 || library.is_null() {
            return Err(TextRenderingError::InvalidPinnedFont);
        }
        let mut face = ptr::null_mut();
        let font_length = ft::FT_Long::try_from(PILLOW_DEFAULT_FONT.len())
            .map_err(|_| TextRenderingError::InvalidPinnedFont)?;
        // The embedded bytes are static and therefore outlive every face created from them.
        let face_error = unsafe {
            ft::FT_New_Memory_Face(
                library,
                PILLOW_DEFAULT_FONT.as_ptr(),
                font_length,
                0,
                &mut face,
            )
        };
        if face_error != 0 || face.is_null() {
            // No face exists yet, so only the initialized library needs releasing.
            unsafe {
                ft::FT_Done_FreeType(library);
            }
            return Err(TextRenderingError::InvalidPinnedFont);
        }
        // Pillow's load_default(size) uses FreeType's integer pixel-size path.
        if unsafe { ft::FT_Set_Pixel_Sizes(face, 0, font_size) } != 0 {
            unsafe {
                ft::FT_Done_Face(face);
                ft::FT_Done_FreeType(library);
            }
            return Err(TextRenderingError::InvalidFontSize(font_size));
        }
        Ok(Self { library, face })
    }

    fn units_per_em(&self) -> u16 {
        // The face remains valid for the wrapper lifetime.
        unsafe { (*self.face).units_per_EM }
    }

    fn ascent(&self) -> i32 {
        // FreeType exposes configured size metrics in 26.6 fixed-point units.
        unsafe { ((*(*self.face).size).metrics.ascender / 64) as i32 }
    }

    fn descent(&self) -> i32 {
        unsafe { (-(*(*self.face).size).metrics.descender / 64) as i32 }
    }

    fn shape(
        &mut self,
        text: &str,
        cancellation: &CancellationToken,
    ) -> Result<(Vec<PositionedGlyph>, i32), TextRenderingError> {
        let mut glyphs = Vec::new();
        glyphs
            .try_reserve_exact(text.chars().count())
            .map_err(|error| TextRenderingError::Allocation(error.to_string()))?;
        let mut pen_x = 0_i32;
        let mut previous_glyph_id = 0;
        for (character_index, character) in text.chars().enumerate() {
            if character_index & 0xff == 0 {
                cancellation.check()?;
            }
            let glyph_id = unsafe { ft::FT_Get_Char_Index(self.face, character as ft::FT_ULong) };
            if previous_glyph_id != 0 && glyph_id != 0 && ft::FT_HAS_KERNING(self.face) {
                let mut kerning = ft::FT_Vector { x: 0, y: 0 };
                if unsafe {
                    ft::FT_Get_Kerning(
                        self.face,
                        previous_glyph_id,
                        glyph_id,
                        ft::FT_KERNING_DEFAULT,
                        &mut kerning,
                    )
                } != 0
                {
                    return Err(TextRenderingError::InvalidPinnedFont);
                }
                pen_x = pen_x
                    .checked_add(
                        i32::try_from(kerning.x / 64)
                            .map_err(|_| TextRenderingError::InvalidDimensions)?,
                    )
                    .ok_or(TextRenderingError::InvalidDimensions)?;
            }
            if unsafe { ft::FT_Load_Glyph(self.face, glyph_id, ft::FT_LOAD_DEFAULT) } != 0 {
                return Err(TextRenderingError::InvalidPinnedFont);
            }
            glyphs.push(PositionedGlyph {
                id: glyph_id,
                x: pen_x,
            });
            let slot = unsafe { (*self.face).glyph };
            if slot.is_null() {
                return Err(TextRenderingError::InvalidPinnedFont);
            }
            let advance = i32::try_from(unsafe { (*slot).advance.x / 64 })
                .map_err(|_| TextRenderingError::InvalidDimensions)?;
            pen_x = pen_x
                .checked_add(advance)
                .ok_or(TextRenderingError::InvalidDimensions)?;
            previous_glyph_id = glyph_id;
        }
        Ok((glyphs, pen_x))
    }

    fn render(&mut self, glyph_id: ft::FT_UInt) -> Result<RenderedGlyph<'_>, TextRenderingError> {
        if unsafe { ft::FT_Load_Glyph(self.face, glyph_id, ft::FT_LOAD_DEFAULT) } != 0 {
            return Err(TextRenderingError::InvalidPinnedFont);
        }
        let slot = unsafe { (*self.face).glyph };
        if slot.is_null() || unsafe { ft::FT_Render_Glyph(slot, ft::FT_RENDER_MODE_NORMAL) } != 0 {
            return Err(TextRenderingError::InvalidPinnedFont);
        }
        let bitmap = unsafe { &(*slot).bitmap };
        if bitmap.pixel_mode as u32 != ft::FT_PIXEL_MODE_GRAY
            || bitmap.pitch < 0
            || bitmap.width < 0
            || bitmap.rows < 0
        {
            return Err(TextRenderingError::InvalidPinnedFont);
        }
        let width =
            u32::try_from(bitmap.width).map_err(|_| TextRenderingError::InvalidDimensions)?;
        let height =
            u32::try_from(bitmap.rows).map_err(|_| TextRenderingError::InvalidDimensions)?;
        let pitch =
            usize::try_from(bitmap.pitch).map_err(|_| TextRenderingError::InvalidDimensions)?;
        if pitch != width as usize {
            return Err(TextRenderingError::InvalidPinnedFont);
        }
        let length = pitch
            .checked_mul(height as usize)
            .ok_or(TextRenderingError::InvalidDimensions)?;
        let data = if length == 0 {
            &[]
        } else if bitmap.buffer.is_null() {
            return Err(TextRenderingError::InvalidPinnedFont);
        } else {
            // The bitmap belongs to the glyph slot and is consumed before the next face call.
            unsafe { slice::from_raw_parts(bitmap.buffer, length) }
        };
        Ok(RenderedGlyph {
            left: unsafe { (*slot).bitmap_left },
            top: unsafe { (*slot).bitmap_top },
            width,
            height,
            data,
        })
    }
}

impl Drop for FreeTypeFace {
    fn drop(&mut self) {
        // Both handles were created together and are released in dependency order.
        unsafe {
            ft::FT_Done_Face(self.face);
            ft::FT_Done_FreeType(self.library);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn composite_mask(
    overlay: &mut [u8],
    width: u64,
    height: u64,
    left: i32,
    top: i32,
    mask_width: u32,
    mask_height: u32,
    mask: &[u8],
    color: [u8; 4],
    cancellation: &CancellationToken,
) -> Result<(), TextRenderingError> {
    let mut processed_pixels = 0_usize;
    for mask_y in 0..mask_height {
        cancellation.check()?;
        let destination_y = top.saturating_add(mask_y as i32);
        if destination_y < 0 || destination_y >= height as i32 {
            continue;
        }
        for mask_x in 0..mask_width {
            if processed_pixels & 0xff == 0 {
                cancellation.check()?;
            }
            processed_pixels = processed_pixels.saturating_add(1);
            let destination_x = left.saturating_add(mask_x as i32);
            if destination_x < 0 || destination_x >= width as i32 {
                continue;
            }
            let mask_index = (mask_y as usize)
                .checked_mul(mask_width as usize)
                .and_then(|value| value.checked_add(mask_x as usize))
                .ok_or(TextRenderingError::InvalidDimensions)?;
            let coverage = u32::from(
                *mask
                    .get(mask_index)
                    .ok_or(TextRenderingError::InvalidDimensions)?,
            );
            let source_alpha = (coverage * u32::from(color[3]) + 127) / 255;
            if source_alpha == 0 {
                continue;
            }
            let destination_pixel = (destination_y as usize)
                .checked_mul(width as usize)
                .and_then(|value| value.checked_add(destination_x as usize))
                .and_then(|value| value.checked_mul(4))
                .ok_or(TextRenderingError::InvalidDimensions)?;
            alpha_composite_pixel(overlay, destination_pixel, color, source_alpha)?;
        }
    }
    Ok(())
}

fn alpha_composite_pixel(
    destination: &mut [u8],
    offset: usize,
    color: [u8; 4],
    source_alpha: u32,
) -> Result<(), TextRenderingError> {
    let destination_alpha = u32::from(
        *destination
            .get(offset + 3)
            .ok_or(TextRenderingError::InvalidDimensions)?,
    );
    let inverse_source_alpha = 255_u32.saturating_sub(source_alpha);
    let output_alpha = source_alpha
        .checked_add((destination_alpha * inverse_source_alpha + 127) / 255)
        .ok_or(TextRenderingError::InvalidDimensions)?;
    for channel in 0..3 {
        let destination_channel = u32::from(
            *destination
                .get(offset + channel)
                .ok_or(TextRenderingError::InvalidDimensions)?,
        );
        let numerator = u32::from(color[channel])
            .checked_mul(source_alpha)
            .and_then(|value| {
                destination_channel
                    .checked_mul(destination_alpha)
                    .and_then(|value| value.checked_mul(inverse_source_alpha))
                    .and_then(|value| value.checked_add(127))
                    .map(|value| value / 255)
                    .and_then(|destination| value.checked_add(destination))
            })
            .ok_or(TextRenderingError::InvalidDimensions)?;
        let value = (numerator + output_alpha / 2)
            .checked_div(output_alpha)
            .unwrap_or(0);
        *destination
            .get_mut(offset + channel)
            .ok_or(TextRenderingError::InvalidDimensions)? = value as u8;
    }
    *destination
        .get_mut(offset + 3)
        .ok_or(TextRenderingError::InvalidDimensions)? = output_alpha as u8;
    Ok(())
}

fn checked_pixel_count(width: u64, height: u64) -> Result<usize, TextRenderingError> {
    let pixels = width
        .checked_mul(height)
        .filter(|pixels| width != 0 && height != 0 && *pixels <= MAX_TEXT_RENDER_PIXELS)
        .ok_or(TextRenderingError::InvalidDimensions)?;
    usize::try_from(pixels).map_err(|_| TextRenderingError::InvalidDimensions)
}

fn validate_text(text: &str) -> Result<(), TextRenderingError> {
    if text.chars().count() > MAX_TEXT_CHARACTERS {
        Err(TextRenderingError::TextTooLong)
    } else {
        Ok(())
    }
}

fn validate_font_size(font_size: u32) -> Result<(), TextRenderingError> {
    if (1..=MAX_FONT_SIZE).contains(&font_size) {
        Ok(())
    } else {
        Err(TextRenderingError::InvalidFontSize(font_size))
    }
}

fn try_owned(value: &str) -> Result<String, TextRenderingError> {
    let mut output = String::new();
    output
        .try_reserve_exact(value.len())
        .map_err(|error| TextRenderingError::Allocation(error.to_string()))?;
    output.push_str(value);
    Ok(output)
}

fn hex_sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use sim_tensor::{CpuWorkspaceAuthority, StreamId};
    use sim_types::CancellationToken;

    fn context<'a>(
        backend: &CpuBackend,
        authority: &'a CpuWorkspaceAuthority,
        cancellation: &'a CancellationToken,
        bytes: u64,
    ) -> Result<ExecutionContext<'a>, TensorError> {
        Ok(backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(bytes)?,
            cancellation,
        ))
    }

    #[test]
    fn source_text_wrapping_font_identity_and_metrics_are_stable()
    -> Result<(), Box<dyn std::error::Error>> {
        let renderer = NativeMediaTextRenderer::new()?;
        let cancellation = CancellationToken::default();
        assert_eq!(renderer.font_sha256(), PILLOW_DEFAULT_FONT_SHA256);
        assert_eq!(zero_padded_numeric_tag(1), "01");
        assert_eq!(zero_padded_numeric_tag(103), "103");
        let small = renderer.measure("Aileron café — 01", 10, &cancellation)?;
        let large = renderer.measure("Aileron café — 01", 20, &cancellation)?;
        assert_eq!(small.advance, 75.0);
        assert_eq!(large.advance, 147.0);
        let pair = renderer.measure("AV", 40, &cancellation)?.advance;
        let separated = renderer.measure("A", 40, &cancellation)?.advance
            + renderer.measure("V", 40, &cancellation)?.advance;
        assert_eq!(pair, 51.0);
        assert_eq!(separated, 51.0);
        let boundary = renderer.measure("alpha beta", 12, &cancellation)?.advance;
        assert_eq!(
            renderer.wrap(
                "alpha beta gamma\n\nUnicode café",
                12,
                boundary,
                &cancellation,
            )?,
            ["alpha beta", "gamma", "", "Unicode", "café"]
        );
        Ok(())
    }

    #[test]
    fn measurement_cancels_cooperatively_after_work_begins()
    -> Result<(), Box<dyn std::error::Error>> {
        let renderer = NativeMediaTextRenderer::new()?;
        let cancellation = CancellationToken::default();
        let worker_cancellation = cancellation.clone();
        let text = "W".repeat(MAX_TEXT_CHARACTERS);
        let (started_sender, started_receiver) = std::sync::mpsc::sync_channel(0);
        let worker = std::thread::spawn(move || {
            started_sender.send(()).map_err(|error| error.to_string())?;
            renderer
                .measure(&text, 12, &worker_cancellation)
                .map_err(|error| error.to_string())
        });
        started_receiver.recv()?;
        std::thread::sleep(std::time::Duration::from_millis(1));
        assert!(cancellation.cancel());
        let result = worker
            .join()
            .map_err(|_| "text measurement worker panicked")?;
        assert_eq!(result, Err(CancellationError.to_string()));
        Ok(())
    }

    #[test]
    fn source_rgba_pixels_cancel_and_release_scratch() -> Result<(), Box<dyn std::error::Error>> {
        let renderer = NativeMediaTextRenderer::new()?;
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(4 * 1024 * 1024)?;
        let cancellation = CancellationToken::default();
        let execution_context = context(&backend, &authority, &cancellation, 2 * 1024 * 1024)?;
        let background = [32_u8, 32, 32, 255].repeat(64 * 24);
        let runs = [TextRun {
            text: "01 café".to_owned(),
            x: 3,
            y: 2,
            font_size: 12,
            color: [240, 80, 40, 255],
        }];
        let first =
            renderer.render_over_rgba8(64, 24, &background, &runs, &backend, &execution_context)?;
        let first_bytes = first
            .as_f32_slice()?
            .iter()
            .map(|value| (value * 255.0).round() as u8)
            .collect::<Vec<_>>();
        let expected = include_bytes!(
            "../../sim_test_support/fixtures/media/text-rendering/pillow-12.3.0-64x24.rgb"
        );
        let differences = first_bytes
            .iter()
            .zip(expected.iter())
            .enumerate()
            .filter(|(_, (actual, expected))| actual != expected)
            .take(12)
            .map(|(index, (actual, expected))| (index, *actual, *expected))
            .collect::<Vec<_>>();
        assert_eq!(
            first_bytes.as_slice(),
            expected,
            "first channel differences: {differences:?}"
        );
        drop(first);
        let second =
            renderer.render_over_rgba8(64, 24, &background, &runs, &backend, &execution_context)?;
        assert_eq!(
            second
                .as_f32_slice()?
                .iter()
                .map(|value| (value * 255.0).round() as u8)
                .collect::<Vec<_>>(),
            first_bytes
        );
        assert!(
            second
                .as_f32_slice()?
                .iter()
                .any(|value| *value > 32.0 / 255.0)
        );
        drop(second);
        assert_eq!(execution_context.scratch.in_use_bytes(), 0);

        let cancelled = CancellationToken::default();
        cancelled.cancel();
        let cancelled_context = context(&backend, &authority, &cancelled, 2 * 1024 * 1024)?;
        assert!(matches!(
            renderer.render_over_rgba8(64, 24, &background, &runs, &backend, &cancelled_context),
            Err(TextRenderingError::Tensor(TensorError::Cancelled))
        ));
        assert_eq!(cancelled_context.scratch.in_use_bytes(), 0);

        let (tiny_backend, tiny_authority) = CpuWorkspaceAuthority::create_backend(64)?;
        let tiny_context = context(&tiny_backend, &tiny_authority, &cancellation, 64)?;
        assert!(
            renderer
                .render_over_rgba8(64, 24, &background, &runs, &tiny_backend, &tiny_context,)
                .is_err()
        );
        assert_eq!(tiny_context.scratch.in_use_bytes(), 0);
        Ok(())
    }
}
