use image::{DynamicImage, ImageFormat};
use std::io::Cursor;

pub mod gaussian_splat;
pub mod image_quantization;
pub mod metadata;
pub mod native_node_payload;
pub mod png;
pub mod text_rendering;
pub mod video;

pub use gaussian_splat::*;
pub use image_quantization::*;
pub use metadata::*;
pub use native_node_payload::*;
pub use png::*;
pub use text_rendering::*;
pub use video::*;

pub fn encode_png(image: &DynamicImage) -> Result<Vec<u8>, image::ImageError> {
    let mut bytes = Vec::new();
    image.write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)?;
    Ok(bytes)
}
