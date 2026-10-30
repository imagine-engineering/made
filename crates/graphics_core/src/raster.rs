use std::{fmt, panic::AssertUnwindSafe};

use futures::FutureExt as _;
use graphite_core_types::{
    Color, Node,
    list::{Item, List},
    registry::FutureWrapperNode,
    value::ClonedNode,
};
use graphite_raster_types::{CPU, Raster};
use graphite_vector_types::Gradient;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RasterEvaluationError;

impl fmt::Display for RasterEvaluationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Node graph execution panicked")
    }
}

impl std::error::Error for RasterEvaluationError {}

pub fn evaluate_invert_cpu(
    input: Item<Raster<CPU>>,
) -> Result<Item<Raster<CPU>>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::InvertNode::new(FutureWrapperNode::new(
        ClonedNode::new(input),
    ));
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_invert_cpu_list(
    input: List<Raster<CPU>>,
) -> Result<List<Raster<CPU>>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::InvertNodeMapped::new(FutureWrapperNode::new(
        ClonedNode::new(input),
    ));
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_invert_color(input: Item<Color>) -> Result<Item<Color>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::InvertNode::new(FutureWrapperNode::new(
        ClonedNode::new(input),
    ));
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_invert_color_channels(
    [red, green, blue, alpha]: [f32; 4],
) -> Result<[f32; 4], RasterEvaluationError> {
    evaluate_invert_color(Item::new_from_element(Color::from_gamma_srgb_channels(
        red, green, blue, alpha,
    )))
    .map(|output| output.element().to_gamma_srgb_channels())
}

pub fn evaluate_invert_color_list(
    input: List<Color>,
) -> Result<List<Color>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::InvertNodeMapped::new(FutureWrapperNode::new(
        ClonedNode::new(input),
    ));
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_invert_gradient(
    input: Item<Gradient>,
) -> Result<Item<Gradient>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::InvertNode::new(FutureWrapperNode::new(
        ClonedNode::new(input),
    ));
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_invert_gradient_list(
    input: List<Gradient>,
) -> Result<List<Gradient>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::InvertNodeMapped::new(FutureWrapperNode::new(
        ClonedNode::new(input),
    ));
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_make_opaque_cpu(
    input: Item<Raster<CPU>>,
) -> Result<Item<Raster<CPU>>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::MakeOpaqueNode::new(FutureWrapperNode::new(
        ClonedNode::new(input),
    ));
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_make_opaque_cpu_list(
    input: List<Raster<CPU>>,
) -> Result<List<Raster<CPU>>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::MakeOpaqueNodeMapped::new(
        FutureWrapperNode::new(ClonedNode::new(input)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_make_opaque_color(
    input: Item<Color>,
) -> Result<Item<Color>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::MakeOpaqueNode::new(FutureWrapperNode::new(
        ClonedNode::new(input),
    ));
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_make_opaque_color_list(
    input: List<Color>,
) -> Result<List<Color>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::MakeOpaqueNodeMapped::new(
        FutureWrapperNode::new(ClonedNode::new(input)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_make_opaque_gradient(
    input: Item<Gradient>,
) -> Result<Item<Gradient>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::MakeOpaqueNode::new(FutureWrapperNode::new(
        ClonedNode::new(input),
    ));
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_make_opaque_gradient_list(
    input: List<Gradient>,
) -> Result<List<Gradient>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::MakeOpaqueNodeMapped::new(
        FutureWrapperNode::new(ClonedNode::new(input)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_mix_cpu(
    over: Item<Raster<CPU>>,
    under: Item<Raster<CPU>>,
    blend_mode: Item<graphite_core_types::blending::BlendMode>,
    opacity: Item<f64>,
) -> Result<Item<Raster<CPU>>, RasterEvaluationError> {
    let node = graphite_raster_nodes::blending_nodes::MixNode::new(
        FutureWrapperNode::new(ClonedNode::new(over)),
        FutureWrapperNode::new(ClonedNode::new(under)),
        FutureWrapperNode::new(ClonedNode::new(blend_mode)),
        FutureWrapperNode::new(ClonedNode::new(opacity)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_mix_cpu_list(
    over: List<Raster<CPU>>,
    under: List<Raster<CPU>>,
    blend_mode: List<graphite_core_types::blending::BlendMode>,
    opacity: List<f64>,
) -> Result<List<Raster<CPU>>, RasterEvaluationError> {
    let node = graphite_raster_nodes::blending_nodes::MixNodeMapped::new(
        FutureWrapperNode::new(ClonedNode::new(over)),
        FutureWrapperNode::new(ClonedNode::new(under)),
        FutureWrapperNode::new(ClonedNode::new(blend_mode)),
        FutureWrapperNode::new(ClonedNode::new(opacity)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_mix_color(
    over: Item<Color>,
    under: Item<Color>,
    blend_mode: Item<graphite_core_types::blending::BlendMode>,
    opacity: Item<f64>,
) -> Result<Item<Color>, RasterEvaluationError> {
    let node = graphite_raster_nodes::blending_nodes::MixNode::new(
        FutureWrapperNode::new(ClonedNode::new(over)),
        FutureWrapperNode::new(ClonedNode::new(under)),
        FutureWrapperNode::new(ClonedNode::new(blend_mode)),
        FutureWrapperNode::new(ClonedNode::new(opacity)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_mix_color_gamma_srgb_channels(
    over: [f32; 4],
    under: [f32; 4],
    blend_mode: graphite_core_types::blending::BlendMode,
    opacity: f64,
) -> Result<[f32; 4], RasterEvaluationError> {
    let [red, green, blue, alpha] = over;
    let over = Color::from_gamma_srgb_channels(red, green, blue, alpha);
    let [red, green, blue, alpha] = under;
    let under = Color::from_gamma_srgb_channels(red, green, blue, alpha);
    evaluate_mix_color(
        Item::new_from_element(over),
        Item::new_from_element(under),
        Item::new_from_element(blend_mode),
        Item::new_from_element(opacity),
    )
    .map(|output| output.element().to_gamma_srgb_channels())
}

pub fn evaluate_mix_color_list(
    over: List<Color>,
    under: List<Color>,
    blend_mode: List<graphite_core_types::blending::BlendMode>,
    opacity: List<f64>,
) -> Result<List<Color>, RasterEvaluationError> {
    let node = graphite_raster_nodes::blending_nodes::MixNodeMapped::new(
        FutureWrapperNode::new(ClonedNode::new(over)),
        FutureWrapperNode::new(ClonedNode::new(under)),
        FutureWrapperNode::new(ClonedNode::new(blend_mode)),
        FutureWrapperNode::new(ClonedNode::new(opacity)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_mix_gradient(
    over: Item<Gradient>,
    under: Item<Gradient>,
    blend_mode: Item<graphite_core_types::blending::BlendMode>,
    opacity: Item<f64>,
) -> Result<Item<Gradient>, RasterEvaluationError> {
    let node = graphite_raster_nodes::blending_nodes::MixNode::new(
        FutureWrapperNode::new(ClonedNode::new(over)),
        FutureWrapperNode::new(ClonedNode::new(under)),
        FutureWrapperNode::new(ClonedNode::new(blend_mode)),
        FutureWrapperNode::new(ClonedNode::new(opacity)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_mix_gradient_list(
    over: List<Gradient>,
    under: List<Gradient>,
    blend_mode: List<graphite_core_types::blending::BlendMode>,
    opacity: List<f64>,
) -> Result<List<Gradient>, RasterEvaluationError> {
    let node = graphite_raster_nodes::blending_nodes::MixNodeMapped::new(
        FutureWrapperNode::new(ClonedNode::new(over)),
        FutureWrapperNode::new(ClonedNode::new(under)),
        FutureWrapperNode::new(ClonedNode::new(blend_mode)),
        FutureWrapperNode::new(ClonedNode::new(opacity)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlackAndWhiteParameters {
    pub use_tint: Item<bool>,
    pub tint: Item<Color>,
    pub reds: Item<f64>,
    pub yellows: Item<f64>,
    pub greens: Item<f64>,
    pub cyans: Item<f64>,
    pub blues: Item<f64>,
    pub magentas: Item<f64>,
}

pub fn evaluate_black_and_white_cpu(
    image: Item<Raster<CPU>>,
    parameters: BlackAndWhiteParameters,
) -> Result<Item<Raster<CPU>>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::BlackAndWhiteNode::new(
        FutureWrapperNode::new(ClonedNode::new(image)),
        FutureWrapperNode::new(ClonedNode::new(parameters.use_tint)),
        FutureWrapperNode::new(ClonedNode::new(parameters.tint)),
        FutureWrapperNode::new(ClonedNode::new(parameters.reds)),
        FutureWrapperNode::new(ClonedNode::new(parameters.yellows)),
        FutureWrapperNode::new(ClonedNode::new(parameters.greens)),
        FutureWrapperNode::new(ClonedNode::new(parameters.cyans)),
        FutureWrapperNode::new(ClonedNode::new(parameters.blues)),
        FutureWrapperNode::new(ClonedNode::new(parameters.magentas)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

#[derive(Clone, Debug, PartialEq)]
pub struct BlackAndWhiteListParameters {
    pub use_tint: List<bool>,
    pub tint: List<Color>,
    pub reds: List<f64>,
    pub yellows: List<f64>,
    pub greens: List<f64>,
    pub cyans: List<f64>,
    pub blues: List<f64>,
    pub magentas: List<f64>,
}

pub fn evaluate_black_and_white_color(
    image: Item<Color>,
    parameters: BlackAndWhiteParameters,
) -> Result<Item<Color>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::BlackAndWhiteNode::new(
        FutureWrapperNode::new(ClonedNode::new(image)),
        FutureWrapperNode::new(ClonedNode::new(parameters.use_tint)),
        FutureWrapperNode::new(ClonedNode::new(parameters.tint)),
        FutureWrapperNode::new(ClonedNode::new(parameters.reds)),
        FutureWrapperNode::new(ClonedNode::new(parameters.yellows)),
        FutureWrapperNode::new(ClonedNode::new(parameters.greens)),
        FutureWrapperNode::new(ClonedNode::new(parameters.cyans)),
        FutureWrapperNode::new(ClonedNode::new(parameters.blues)),
        FutureWrapperNode::new(ClonedNode::new(parameters.magentas)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_black_and_white_gradient(
    image: Item<Gradient>,
    parameters: BlackAndWhiteParameters,
) -> Result<Item<Gradient>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::BlackAndWhiteNode::new(
        FutureWrapperNode::new(ClonedNode::new(image)),
        FutureWrapperNode::new(ClonedNode::new(parameters.use_tint)),
        FutureWrapperNode::new(ClonedNode::new(parameters.tint)),
        FutureWrapperNode::new(ClonedNode::new(parameters.reds)),
        FutureWrapperNode::new(ClonedNode::new(parameters.yellows)),
        FutureWrapperNode::new(ClonedNode::new(parameters.greens)),
        FutureWrapperNode::new(ClonedNode::new(parameters.cyans)),
        FutureWrapperNode::new(ClonedNode::new(parameters.blues)),
        FutureWrapperNode::new(ClonedNode::new(parameters.magentas)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_black_and_white_cpu_list(
    image: List<Raster<CPU>>,
    parameters: BlackAndWhiteListParameters,
) -> Result<List<Raster<CPU>>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::BlackAndWhiteNodeMapped::new(
        FutureWrapperNode::new(ClonedNode::new(image)),
        FutureWrapperNode::new(ClonedNode::new(parameters.use_tint)),
        FutureWrapperNode::new(ClonedNode::new(parameters.tint)),
        FutureWrapperNode::new(ClonedNode::new(parameters.reds)),
        FutureWrapperNode::new(ClonedNode::new(parameters.yellows)),
        FutureWrapperNode::new(ClonedNode::new(parameters.greens)),
        FutureWrapperNode::new(ClonedNode::new(parameters.cyans)),
        FutureWrapperNode::new(ClonedNode::new(parameters.blues)),
        FutureWrapperNode::new(ClonedNode::new(parameters.magentas)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_black_and_white_color_list(
    image: List<Color>,
    parameters: BlackAndWhiteListParameters,
) -> Result<List<Color>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::BlackAndWhiteNodeMapped::new(
        FutureWrapperNode::new(ClonedNode::new(image)),
        FutureWrapperNode::new(ClonedNode::new(parameters.use_tint)),
        FutureWrapperNode::new(ClonedNode::new(parameters.tint)),
        FutureWrapperNode::new(ClonedNode::new(parameters.reds)),
        FutureWrapperNode::new(ClonedNode::new(parameters.yellows)),
        FutureWrapperNode::new(ClonedNode::new(parameters.greens)),
        FutureWrapperNode::new(ClonedNode::new(parameters.cyans)),
        FutureWrapperNode::new(ClonedNode::new(parameters.blues)),
        FutureWrapperNode::new(ClonedNode::new(parameters.magentas)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_black_and_white_gradient_list(
    image: List<Gradient>,
    parameters: BlackAndWhiteListParameters,
) -> Result<List<Gradient>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::BlackAndWhiteNodeMapped::new(
        FutureWrapperNode::new(ClonedNode::new(image)),
        FutureWrapperNode::new(ClonedNode::new(parameters.use_tint)),
        FutureWrapperNode::new(ClonedNode::new(parameters.tint)),
        FutureWrapperNode::new(ClonedNode::new(parameters.reds)),
        FutureWrapperNode::new(ClonedNode::new(parameters.yellows)),
        FutureWrapperNode::new(ClonedNode::new(parameters.greens)),
        FutureWrapperNode::new(ClonedNode::new(parameters.cyans)),
        FutureWrapperNode::new(ClonedNode::new(parameters.blues)),
        FutureWrapperNode::new(ClonedNode::new(parameters.magentas)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrightnessContrastParameters {
    pub brightness: Item<f64>,
    pub contrast: Item<f64>,
    pub use_classic: Item<bool>,
    pub classic_pivot: Item<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BrightnessContrastListParameters {
    pub brightness: List<f64>,
    pub contrast: List<f64>,
    pub use_classic: List<bool>,
    pub classic_pivot: List<f64>,
}

pub fn evaluate_brightness_contrast_cpu(
    input: Item<Raster<CPU>>,
    parameters: BrightnessContrastParameters,
) -> Result<Item<Raster<CPU>>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::BrightnessContrastNode::new(
        FutureWrapperNode::new(ClonedNode::new(input)),
        FutureWrapperNode::new(ClonedNode::new(parameters.brightness)),
        FutureWrapperNode::new(ClonedNode::new(parameters.contrast)),
        FutureWrapperNode::new(ClonedNode::new(parameters.use_classic)),
        FutureWrapperNode::new(ClonedNode::new(parameters.classic_pivot)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_brightness_contrast_cpu_list(
    input: List<Raster<CPU>>,
    parameters: BrightnessContrastListParameters,
) -> Result<List<Raster<CPU>>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::BrightnessContrastNodeMapped::new(
        FutureWrapperNode::new(ClonedNode::new(input)),
        FutureWrapperNode::new(ClonedNode::new(parameters.brightness)),
        FutureWrapperNode::new(ClonedNode::new(parameters.contrast)),
        FutureWrapperNode::new(ClonedNode::new(parameters.use_classic)),
        FutureWrapperNode::new(ClonedNode::new(parameters.classic_pivot)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_brightness_contrast_color(
    input: Item<Color>,
    parameters: BrightnessContrastParameters,
) -> Result<Item<Color>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::BrightnessContrastNode::new(
        FutureWrapperNode::new(ClonedNode::new(input)),
        FutureWrapperNode::new(ClonedNode::new(parameters.brightness)),
        FutureWrapperNode::new(ClonedNode::new(parameters.contrast)),
        FutureWrapperNode::new(ClonedNode::new(parameters.use_classic)),
        FutureWrapperNode::new(ClonedNode::new(parameters.classic_pivot)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_brightness_contrast_color_list(
    input: List<Color>,
    parameters: BrightnessContrastListParameters,
) -> Result<List<Color>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::BrightnessContrastNodeMapped::new(
        FutureWrapperNode::new(ClonedNode::new(input)),
        FutureWrapperNode::new(ClonedNode::new(parameters.brightness)),
        FutureWrapperNode::new(ClonedNode::new(parameters.contrast)),
        FutureWrapperNode::new(ClonedNode::new(parameters.use_classic)),
        FutureWrapperNode::new(ClonedNode::new(parameters.classic_pivot)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_brightness_contrast_gradient(
    input: Item<Gradient>,
    parameters: BrightnessContrastParameters,
) -> Result<Item<Gradient>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::BrightnessContrastNode::new(
        FutureWrapperNode::new(ClonedNode::new(input)),
        FutureWrapperNode::new(ClonedNode::new(parameters.brightness)),
        FutureWrapperNode::new(ClonedNode::new(parameters.contrast)),
        FutureWrapperNode::new(ClonedNode::new(parameters.use_classic)),
        FutureWrapperNode::new(ClonedNode::new(parameters.classic_pivot)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

pub fn evaluate_brightness_contrast_gradient_list(
    input: List<Gradient>,
    parameters: BrightnessContrastListParameters,
) -> Result<List<Gradient>, RasterEvaluationError> {
    let node = graphite_raster_nodes::adjustments::BrightnessContrastNodeMapped::new(
        FutureWrapperNode::new(ClonedNode::new(input)),
        FutureWrapperNode::new(ClonedNode::new(parameters.brightness)),
        FutureWrapperNode::new(ClonedNode::new(parameters.contrast)),
        FutureWrapperNode::new(ClonedNode::new(parameters.use_classic)),
        FutureWrapperNode::new(ClonedNode::new(parameters.classic_pivot)),
    );
    futures::executor::block_on(AssertUnwindSafe(node.eval(())).catch_unwind())
        .map_err(|_| RasterEvaluationError)
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphite_raster_types::{Color, Image};

    pub(super) fn image_item(width: u32, height: u32, colors: &[[f32; 4]]) -> Item<Raster<CPU>> {
        Item::new_from_element(Raster::new_cpu(Image {
            width,
            height,
            data: colors
                .iter()
                .map(|&[red, green, blue, alpha]| {
                    Color::from_gamma_srgb_channels(red, green, blue, alpha)
                })
                .collect(),
            base64_string: None,
        }))
    }

    pub(super) fn assert_gamma_channels(actual: &Color, expected: [f32; 4]) {
        for (actual, expected) in actual.to_gamma_srgb_channels().into_iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "expected gamma channel {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn at_gra_node_152_2_invert_uses_original_gamma_space_channels() {
        let input = image_item(
            3,
            2,
            &[
                [1., 0., 0., 1.],
                [0., 0., 1., 1.],
                [0., 0., 0., 1.],
                [1., 1., 1., 1.],
                [0.5, 0.5, 0.5, 1.],
                [1., 0.25, 0., 1.],
            ],
        );
        let output = evaluate_invert_cpu(input).expect("original CPU Invert succeeds");
        let expected = [
            [0., 1., 1., 1.],
            [1., 1., 0., 1.],
            [1., 1., 1., 1.],
            [0., 0., 0., 1.],
            [0.5, 0.5, 0.5, 1.],
            [0., 0.75, 1., 1.],
        ];
        assert_eq!(output.element().data().data.len(), expected.len());
        for (actual, expected) in output.element().data().data.iter().zip(expected) {
            assert_gamma_channels(actual, expected);
        }
    }

    #[test]
    fn at_gra_node_152_2_invert_preserves_translucent_and_zero_alpha_bits() {
        let input = image_item(
            2,
            2,
            &[
                [1., 0.25, 0., 0.5],
                [0.2, 0.4, 0.6, 0.],
                [0.1, 0.3, 0.7, -0.],
                [0.8, 0.6, 0.4, 0.125],
            ],
        );
        let original = input.clone();
        let output = evaluate_invert_cpu(input).expect("alpha does not prevent Invert");
        let expected = [
            [0., 0.75, 1., 0.5],
            [0.8, 0.6, 0.4, 0.],
            [0.9, 0.7, 0.3, -0.],
            [0.2, 0.4, 0.6, 0.125],
        ];
        assert_eq!(output.element().data().data.len(), expected.len());
        for ((actual, original), expected) in output
            .element()
            .data()
            .data
            .iter()
            .zip(&original.element().data().data)
            .zip(expected)
        {
            assert_gamma_channels(actual, expected);
            assert_eq!(
                actual.to_gamma_srgb_channels()[3].to_bits(),
                original.to_gamma_srgb_channels()[3].to_bits()
            );
        }
    }

    #[test]
    fn at_gra_node_152_2_invert_retains_whole_attributes_dimensions_and_input() {
        let input = image_item(2, 1, &[[1., 0., 0., 1.], [0., 0., 1., 0.5]])
            .with_attribute("asset", String::from("original-image"))
            .with_attribute("revision", 17_u64)
            .with_attribute("visible", true);
        let original = input.clone();
        let first = evaluate_invert_cpu(input.clone()).expect("first original evaluation");
        let second = evaluate_invert_cpu(input.clone()).expect("repeat original evaluation");

        assert_eq!(input, original);
        assert_eq!(first, second);
        assert_eq!(first.attributes(), original.attributes());
        assert_eq!(
            first.element().data().width,
            original.element().data().width
        );
        assert_eq!(
            first.element().data().height,
            original.element().data().height
        );
        assert_ne!(first.element().data().data, original.element().data().data);
        assert_gamma_channels(&first.element().data().data[0], [0., 1., 1., 1.]);
        assert_gamma_channels(&original.element().data().data[0], [1., 0., 0., 1.]);
    }

    #[test]
    fn at_gra_node_152_2_invert_leaves_empty_rasters_and_attributes_unchanged() {
        for (width, height) in [(0, 0), (0, 3), (4, 0)] {
            let input = image_item(width, height, &[])
                .with_attribute("empty-image", String::from("retained"));
            let output = evaluate_invert_cpu(input.clone()).expect("empty pixel iteration");
            assert_eq!(output, input);
            assert!(output.element().data().data.is_empty());
        }
    }

    #[test]
    fn at_gra_node_152_2_color_invert_keeps_gamma_and_alpha_bits() {
        for alpha in [0., -0., 0.5, 1.] {
            let input =
                Item::new_from_element(Color::from_gamma_srgb_channels(1., 0.25, 0., alpha));
            let original = input.clone();
            let output = evaluate_invert_color(input.clone()).expect("original Color Invert");
            assert_gamma_channels(output.element(), [0., 0.75, 1., alpha]);
            assert_eq!(
                output.element().to_gamma_srgb_channels()[3].to_bits(),
                alpha.to_bits()
            );
            assert_eq!(input, original);
            assert_eq!(
                input.element().to_gamma_srgb_channels()[3].to_bits(),
                alpha.to_bits()
            );
        }
    }

    #[test]
    fn at_gra_node_152_2_color_invert_retains_whole_attributes_and_repeat() {
        let input = Item::new_from_element(Color::from_gamma_srgb_channels(0.2, 0.4, 0.6, 0.5))
            .with_attribute("asset", String::from("original-color"))
            .with_attribute("revision", 21_u64)
            .with_attribute("visible", false);
        let original = input.clone();
        let first = evaluate_invert_color(input.clone()).expect("first Color Invert");
        let second = evaluate_invert_color(input.clone()).expect("repeat Color Invert");
        assert_eq!(first, second);
        assert_eq!(first.attributes(), original.attributes());
        assert_gamma_channels(first.element(), [0.8, 0.6, 0.4, 0.5]);
        assert_eq!(input, original);
        assert_gamma_channels(input.element(), [0.2, 0.4, 0.6, 0.5]);
    }

    #[test]
    fn at_gra_node_152_2_color_list_invert_preserves_order_attributes_alpha_and_input() {
        let channels = [
            [1., 0.25, 0., -0.],
            [0., 0., 1., 0.5],
            [0.2, 0.4, 0.6, 1.],
            [0.5, 0.5, 0.5, 0.],
        ];
        let input: List<Color> = channels
            .into_iter()
            .enumerate()
            .map(|(index, [red, green, blue, alpha])| {
                Item::new_from_element(Color::from_gamma_srgb_channels(red, green, blue, alpha))
                    .with_attribute("original-index", index)
                    .with_attribute("name", format!("color-{index}"))
                    .with_attribute("visible", index % 2 == 0)
            })
            .collect();
        let original = input.clone();
        let first =
            evaluate_invert_color_list(input.clone()).expect("original mapped Color Invert");
        let second = evaluate_invert_color_list(input.clone()).expect("repeat mapped Color Invert");
        assert_eq!(first, second);
        assert_eq!(first.len(), channels.len());
        let expected = [
            [0., 0.75, 1., -0.],
            [1., 1., 0., 0.5],
            [0.8, 0.6, 0.4, 1.],
            [0.5, 0.5, 0.5, 0.],
        ];
        for ((output, original), expected) in first.into_iter().zip(original.clone()).zip(expected)
        {
            assert_gamma_channels(output.element(), expected);
            assert_eq!(output.attributes(), original.attributes());
            assert_eq!(
                output.element().to_gamma_srgb_channels()[3].to_bits(),
                original.element().to_gamma_srgb_channels()[3].to_bits()
            );
        }
        assert_eq!(input, original);
        for (original, [red, green, blue, alpha]) in input.into_iter().zip(channels) {
            assert_gamma_channels(original.element(), [red, green, blue, alpha]);
            assert_eq!(
                original.element().to_gamma_srgb_channels()[3].to_bits(),
                alpha.to_bits()
            );
        }
    }

    #[test]
    fn at_gra_node_152_2_color_list_invert_retains_empty_list() {
        let input = List::<Color>::new();
        let first = evaluate_invert_color_list(input.clone()).expect("empty mapped Color Invert");
        let second =
            evaluate_invert_color_list(input.clone()).expect("repeat empty mapped Color Invert");
        assert!(first.is_empty());
        assert_eq!(first, input);
        assert_eq!(first, second);
    }

    #[test]
    fn at_gra_node_152_2_cpu_list_preserves_slot_order_dimensions_attributes_and_alpha() {
        let input: List<Raster<CPU>> = [
            image_item(1, 2, &[[1., 0.25, 0., -0.], [0., 0., 1., 0.5]])
                .with_attribute("name", String::from("first")),
            image_item(0, 3, &[]).with_attribute("name", String::from("empty")),
            image_item(2, 1, &[[0., 0., 0., 1.], [0.5, 0.5, 0.5, 0.]])
                .with_attribute("name", String::from("last")),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, item)| item.with_attribute("original-index", index))
        .collect();
        let original = input.clone();
        let first = evaluate_invert_cpu_list(input.clone()).expect("mapped CPU Invert");
        let second = evaluate_invert_cpu_list(input.clone()).expect("repeat mapped CPU Invert");
        assert_eq!(first, second);
        assert_eq!(first.len(), 3);
        let expected = [
            vec![[0., 0.75, 1., -0.], [1., 1., 0., 0.5]],
            vec![],
            vec![[1., 1., 1., 1.], [0.5, 0.5, 0.5, 0.]],
        ];
        for ((output, original), expected) in first.into_iter().zip(original.clone()).zip(expected)
        {
            assert_eq!(output.attributes(), original.attributes());
            let output = output.element().data();
            let original = original.element().data();
            assert_eq!(
                (output.width, output.height),
                (original.width, original.height)
            );
            assert_eq!(output.data.len(), expected.len());
            for ((output, original), expected) in
                output.data.iter().zip(&original.data).zip(expected)
            {
                assert_gamma_channels(output, expected);
                assert_eq!(
                    output.to_gamma_srgb_channels()[3].to_bits(),
                    original.to_gamma_srgb_channels()[3].to_bits()
                );
            }
        }
        assert_eq!(input, original);
        let first_source = input
            .into_iter()
            .next()
            .expect("the source first slot is retained");
        assert_eq!(
            first_source
                .element()
                .data()
                .data
                .first()
                .expect("source pixel")
                .to_gamma_srgb_channels()[3]
                .to_bits(),
            (-0_f32).to_bits()
        );
    }

    #[test]
    fn at_gra_node_152_2_cpu_list_retains_empty_list() {
        let input = List::<Raster<CPU>>::new();
        let first = evaluate_invert_cpu_list(input.clone()).expect("empty mapped CPU Invert");
        let second =
            evaluate_invert_cpu_list(input.clone()).expect("repeat empty mapped CPU Invert");
        assert!(first.is_empty());
        assert_eq!(first, input);
        assert_eq!(first, second);
    }

    #[test]
    fn at_gra_node_152_2_color_channels_match_original_typed_gamma_route() {
        for alpha in [0., -0., 0.5, 1.] {
            let channels = [1., 0.25, 0., alpha];
            let typed = evaluate_invert_color(Item::new_from_element(
                Color::from_gamma_srgb_channels(channels[0], channels[1], channels[2], channels[3]),
            ))
            .expect("typed original Color Invert");
            let output =
                evaluate_invert_color_channels(channels).expect("gamma channel convenience");
            assert_eq!(
                output.map(f32::to_bits),
                typed.element().to_gamma_srgb_channels().map(f32::to_bits)
            );
            for (actual, expected) in output.into_iter().zip([0., 0.75, 1., alpha]) {
                assert!((actual - expected).abs() < 1e-5);
            }
            assert_eq!(output[3].to_bits(), alpha.to_bits());
            assert_eq!(
                evaluate_invert_color_channels(channels)
                    .expect("repeat gamma channels")
                    .map(f32::to_bits),
                output.map(f32::to_bits)
            );
        }
    }

    pub(super) fn gradient_item() -> Item<Gradient> {
        use graphite_core_types::list::{
            ATTR_GRADIENT_CYCLIC, ATTR_GRADIENT_HUE_DIRECTION, ATTR_GRADIENT_INTERPOLATION,
            ATTR_GRADIENT_SPACE, ATTR_GRADIENT_SPREAD,
        };
        use graphite_vector_types::{
            GradientHueDirection, GradientInterpolation, GradientSpace, GradientSpread,
            GradientStop,
        };

        let mut gradient = Gradient::new([
            GradientStop {
                position: 0.75,
                midpoint: 0.2,
                color: Color::from_gamma_srgb_channels(1., 0.25, 0., -0.),
            },
            GradientStop {
                position: 0.,
                midpoint: 0.8,
                color: Color::from_gamma_srgb_channels(0., 0., 1., 0.5),
            },
            GradientStop {
                position: 1.,
                midpoint: 0.5,
                color: Color::from_gamma_srgb_channels(0.2, 0.4, 0.6, 1.),
            },
            GradientStop {
                position: 0.25,
                midpoint: 0.3,
                color: Color::from_gamma_srgb_channels(0.5, 0.5, 0.5, 0.),
            },
        ]);
        for index in 0..gradient.len() {
            gradient
                .0
                .set_attribute("stop-name", index, format!("stop-{index}"));
            gradient
                .0
                .set_attribute("stop-visible", index, index % 2 == 0);
        }
        Item::new_from_element(gradient)
            .with_attribute(ATTR_GRADIENT_CYCLIC, true)
            .with_attribute(ATTR_GRADIENT_SPREAD, GradientSpread::Reflect)
            .with_attribute(ATTR_GRADIENT_SPACE, GradientSpace::RgbLinear)
            .with_attribute(
                ATTR_GRADIENT_HUE_DIRECTION,
                GradientHueDirection::Increasing,
            )
            .with_attribute(ATTR_GRADIENT_INTERPOLATION, GradientInterpolation::Smooth)
            .with_attribute("ramp-name", String::from("original-ramp"))
    }

    #[test]
    fn at_gra_node_152_2_gradient_retains_stops_metadata_ramp_attributes_and_input() {
        use graphite_core_types::list::{ATTR_MIDPOINT, ATTR_POSITION};

        let input = gradient_item();
        let original = input.clone();
        let first = evaluate_invert_gradient(input.clone()).expect("original Gradient Invert");
        let second = evaluate_invert_gradient(input.clone()).expect("repeat Gradient Invert");
        assert_eq!(first, second);
        assert_eq!(first.attributes(), original.attributes());
        assert_eq!(first.element().len(), original.element().len());
        let expected = [
            [0., 0.75, 1., -0.],
            [1., 1., 0., 0.5],
            [0.8, 0.6, 0.4, 1.],
            [0.5, 0.5, 0.5, 0.],
        ];
        for ((output, original), expected) in first
            .into_element()
            .into_color_list()
            .into_iter()
            .zip(original.element().clone().into_color_list())
            .zip(expected)
        {
            assert_gamma_channels(output.element(), expected);
            assert_eq!(
                output.element().to_gamma_srgb_channels()[3].to_bits(),
                original.element().to_gamma_srgb_channels()[3].to_bits()
            );
            assert_eq!(output.attributes(), original.attributes());
            assert_eq!(
                output.attribute::<f64>(ATTR_POSITION),
                original.attribute::<f64>(ATTR_POSITION)
            );
            assert_eq!(
                output.attribute::<f64>(ATTR_MIDPOINT),
                original.attribute::<f64>(ATTR_MIDPOINT)
            );
            assert!(output.attribute::<String>("stop-name").is_some());
        }
        assert_eq!(input, original);
        assert_eq!(
            input
                .element()
                .as_color_list()
                .iter_element_values()
                .next()
                .expect("source stop")
                .to_gamma_srgb_channels()[3]
                .to_bits(),
            (-0_f32).to_bits()
        );
    }

    #[test]
    fn at_gra_node_152_2_gradient_retains_empty_gradient_and_ramp_attributes() {
        let input = Item::new_from_element(Gradient::default())
            .with_attribute("ramp-name", String::from("empty-ramp"));
        let output = evaluate_invert_gradient(input.clone()).expect("empty Gradient Invert");
        assert!(output.element().is_empty());
        assert_eq!(output, input);
    }

    #[test]
    fn at_gra_node_152_2_gradient_list_preserves_stop_and_ramp_order_and_metadata() {
        let input: List<Gradient> = [
            gradient_item(),
            Item::new_from_element(Gradient::default())
                .with_attribute("ramp-name", String::from("empty-ramp")),
            Item::new_from_element(Gradient::new([graphite_vector_types::GradientStop {
                position: 0.9,
                midpoint: 0.4,
                color: Color::from_gamma_srgb_channels(0., 0., 0., 0.5),
            }]))
            .with_attribute("ramp-name", String::from("last-ramp")),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, item)| item.with_attribute("ramp-index", index))
        .collect();
        let original = input.clone();
        let first = evaluate_invert_gradient_list(input.clone()).expect("mapped Gradient Invert");
        let second =
            evaluate_invert_gradient_list(input.clone()).expect("repeat mapped Gradient Invert");
        assert_eq!(first, second);
        assert_eq!(first.len(), 3);
        let expected = [
            vec![
                [0., 0.75, 1., -0.],
                [1., 1., 0., 0.5],
                [0.8, 0.6, 0.4, 1.],
                [0.5, 0.5, 0.5, 0.],
            ],
            vec![],
            vec![[1., 1., 1., 0.5]],
        ];
        for ((output, original), expected) in first.into_iter().zip(original.clone()).zip(expected)
        {
            assert_eq!(output.attributes(), original.attributes());
            assert_eq!(output.element().len(), expected.len());
            for ((output, original), expected) in output
                .into_element()
                .into_color_list()
                .into_iter()
                .zip(original.into_element().into_color_list())
                .zip(expected)
            {
                assert_gamma_channels(output.element(), expected);
                assert_eq!(output.attributes(), original.attributes());
                assert_eq!(
                    output.element().to_gamma_srgb_channels()[3].to_bits(),
                    original.element().to_gamma_srgb_channels()[3].to_bits()
                );
            }
        }
        assert_eq!(input, original);
    }

    #[test]
    fn at_gra_node_152_2_gradient_list_retains_empty_list() {
        let input = List::<Gradient>::new();
        let first =
            evaluate_invert_gradient_list(input.clone()).expect("empty mapped Gradient Invert");
        let second = evaluate_invert_gradient_list(input.clone())
            .expect("repeat empty mapped Gradient Invert");
        assert!(first.is_empty());
        assert_eq!(first, input);
        assert_eq!(first, second);
    }

    fn assert_opaque_color(output: &Color, original: &Color) {
        assert_eq!(output.r().to_bits(), original.r().to_bits());
        assert_eq!(output.g().to_bits(), original.g().to_bits());
        assert_eq!(output.b().to_bits(), original.b().to_bits());
        assert_eq!(output.a().to_bits(), 1_f32.to_bits());
    }

    fn assert_opaque_gradient(output: &Item<Gradient>, original: &Item<Gradient>) {
        assert_eq!(output.attributes(), original.attributes());
        assert_eq!(output.element().len(), original.element().len());
        for (output, original) in output
            .element()
            .clone()
            .into_color_list()
            .into_iter()
            .zip(original.element().clone().into_color_list())
        {
            assert_eq!(output.attributes(), original.attributes());
            assert_opaque_color(output.element(), original.element());
        }
    }

    #[test]
    fn at_gra_node_164_2_cpu_preserves_rgb_attributes_dimensions_and_source() {
        let input = image_item(
            3,
            2,
            &[
                [1., 0., 0., 0.],
                [0., 0., 1., -0.],
                [0.25, 0.5, 0.75, 0.25],
                [0.1, 0.3, 0.7, 0.5],
                [0., 0., 0., 1.],
                [1., 1., 1., 1.],
            ],
        )
        .with_attribute("asset", String::from("original-image"))
        .with_attribute("revision", 19_u64)
        .with_attribute("visible", false);
        let original = input.clone();
        let first = evaluate_make_opaque_cpu(input.clone()).expect("original CPU Make Opaque");
        let second = evaluate_make_opaque_cpu(input.clone()).expect("repeat CPU Make Opaque");
        assert_eq!(first, second);
        assert_eq!(first.attributes(), original.attributes());
        let image = first.element().data();
        let source = original.element().data();
        assert_eq!((image.width, image.height), (source.width, source.height));
        assert_eq!(image.data.len(), source.data.len());
        for (output, original) in image.data.iter().zip(&source.data) {
            assert_opaque_color(output, original);
        }
        assert_eq!(input, original);
    }

    #[test]
    fn at_gra_node_164_2_cpu_retains_empty_raster() {
        let input = image_item(0, 3, &[]).with_attribute("empty", true);
        let first = evaluate_make_opaque_cpu(input.clone()).expect("empty CPU Make Opaque");
        let second = evaluate_make_opaque_cpu(input.clone()).expect("repeat empty CPU");
        assert_eq!(first, input);
        assert_eq!(first, second);
    }

    #[test]
    fn at_gra_node_164_2_cpu_list_retains_slot_order_dimensions_and_whole_attributes() {
        let input: List<Raster<CPU>> = [
            image_item(1, 2, &[[0.25, 0.5, 0.75, 0.], [0., 0., 1., 0.5]])
                .with_attribute("name", String::from("first")),
            image_item(0, 3, &[]).with_attribute("name", String::from("empty")),
            image_item(2, 1, &[[1., 0., 0., -0.], [0.1, 0.3, 0.7, 1.]])
                .with_attribute("name", String::from("last")),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, item)| item.with_attribute("slot", index))
        .collect();
        let original = input.clone();
        let first = evaluate_make_opaque_cpu_list(input.clone()).expect("mapped CPU Make Opaque");
        let second = evaluate_make_opaque_cpu_list(input.clone()).expect("repeat mapped CPU");
        assert_eq!(first, second);
        assert_eq!(first.len(), original.len());
        for (output, original) in first.into_iter().zip(original.clone()) {
            assert_eq!(output.attributes(), original.attributes());
            let output = output.element().data();
            let original = original.element().data();
            assert_eq!(
                (output.width, output.height),
                (original.width, original.height)
            );
            assert_eq!(output.data.len(), original.data.len());
            for (output, original) in output.data.iter().zip(&original.data) {
                assert_opaque_color(output, original);
            }
        }
        assert_eq!(input, original);
    }

    #[test]
    fn at_gra_node_164_2_color_preserves_exact_rgb_at_zero_partial_and_full_alpha() {
        for alpha in [0., -0., 0.125, 0.5, 1.] {
            let input =
                Item::new_from_element(Color::from_gamma_srgb_channels(0.25, 0.5, 0.75, alpha))
                    .with_attribute("name", String::from("source-color"))
                    .with_attribute("alpha", alpha);
            let original = input.clone();
            let first =
                evaluate_make_opaque_color(input.clone()).expect("original Color Make Opaque");
            let second =
                evaluate_make_opaque_color(input.clone()).expect("repeat Color Make Opaque");
            assert_eq!(first, second);
            assert_eq!(first.attributes(), original.attributes());
            assert_opaque_color(first.element(), original.element());
            assert_eq!(input, original);
            assert_eq!(input.element().a().to_bits(), alpha.to_bits());
        }
    }

    #[test]
    fn at_gra_node_164_2_color_list_retains_order_attributes_and_source() {
        let input: List<Color> = [
            [1., 0., 0., 0.],
            [0., 0., 1., -0.],
            [0.25, 0.5, 0.75, 0.5],
            [0., 0., 0., 1.],
        ]
        .into_iter()
        .enumerate()
        .map(|(index, [red, green, blue, alpha])| {
            Item::new_from_element(Color::from_gamma_srgb_channels(red, green, blue, alpha))
                .with_attribute("slot", index)
                .with_attribute("name", format!("source-{index}"))
        })
        .collect();
        let original = input.clone();
        let first =
            evaluate_make_opaque_color_list(input.clone()).expect("mapped Color Make Opaque");
        let second = evaluate_make_opaque_color_list(input.clone()).expect("repeat mapped Color");
        assert_eq!(first, second);
        assert_eq!(first.len(), original.len());
        for (output, original) in first.into_iter().zip(original.clone()) {
            assert_eq!(output.attributes(), original.attributes());
            assert_opaque_color(output.element(), original.element());
        }
        assert_eq!(input, original);
    }

    #[test]
    fn at_gra_node_164_2_gradient_retains_stop_order_metadata_and_ramp_attributes() {
        let input = gradient_item();
        let original = input.clone();
        let first =
            evaluate_make_opaque_gradient(input.clone()).expect("original Gradient Make Opaque");
        let second =
            evaluate_make_opaque_gradient(input.clone()).expect("repeat Gradient Make Opaque");
        assert_eq!(first, second);
        assert_opaque_gradient(&first, &original);
        assert_eq!(input, original);
    }

    #[test]
    fn at_gra_node_164_2_gradient_retains_empty_gradient_and_ramp_attributes() {
        let input = Item::new_from_element(Gradient::default())
            .with_attribute("ramp", String::from("empty-ramp"));
        let first =
            evaluate_make_opaque_gradient(input.clone()).expect("empty Gradient Make Opaque");
        let second = evaluate_make_opaque_gradient(input.clone()).expect("repeat empty Gradient");
        assert_eq!(first, input);
        assert_eq!(first, second);
    }

    #[test]
    fn at_gra_node_164_2_gradient_list_retains_stop_ramp_order_and_attributes() {
        let input: List<Gradient> = [
            gradient_item(),
            Item::new_from_element(Gradient::default())
                .with_attribute("ramp-name", String::from("empty-ramp")),
            Item::new_from_element(Gradient::new([graphite_vector_types::GradientStop {
                position: 0.9,
                midpoint: 0.4,
                color: Color::from_gamma_srgb_channels(0.25, 0.5, 0.75, 0.5),
            }]))
            .with_attribute("ramp-name", String::from("last-ramp")),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, item)| item.with_attribute("ramp-index", index))
        .collect();
        let original = input.clone();
        let first =
            evaluate_make_opaque_gradient_list(input.clone()).expect("mapped Gradient Make Opaque");
        let second =
            evaluate_make_opaque_gradient_list(input.clone()).expect("repeat mapped Gradient");
        assert_eq!(first, second);
        assert_eq!(first.len(), original.len());
        for (output, original) in first.into_iter().zip(original.clone()) {
            assert_opaque_gradient(&output, &original);
        }
        assert_eq!(input, original);
    }

    #[test]
    fn at_gra_node_164_2_all_mapped_types_retain_empty_lists() {
        let cpu = List::<Raster<CPU>>::new();
        let color = List::<Color>::new();
        let gradient = List::<Gradient>::new();
        for _ in 0..2 {
            assert_eq!(
                evaluate_make_opaque_cpu_list(cpu.clone()).expect("empty CPU list"),
                cpu
            );
            assert_eq!(
                evaluate_make_opaque_color_list(color.clone()).expect("empty Color list"),
                color
            );
            assert_eq!(
                evaluate_make_opaque_gradient_list(gradient.clone()).expect("empty Gradient list"),
                gradient
            );
        }
    }

    fn assert_mix_linear(actual: &Color, expected: [f32; 4]) {
        for (actual, expected) in [actual.r(), actual.g(), actual.b(), actual.a()]
            .into_iter()
            .zip(expected)
        {
            assert!(
                (actual - expected).abs() < 1e-6,
                "expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn at_gra_node_143_2_mix_cpu_retains_over_ports_alpha_attributes_and_inputs() {
        use graphite_core_types::blending::BlendMode;
        let over = image_item(
            4,
            1,
            &[
                [1., 0., 0., 0.],
                [1., 0., 0., 0.5],
                [1., 0., 0., 1.],
                [1., 0., 0., 0.5],
            ],
        )
        .with_attribute("over-name", String::from("foreground"));
        let under = image_item(
            4,
            1,
            &[
                [0., 0., 1., 1.],
                [0., 0., 1., 1.],
                [0., 0., 1., 1.],
                [0., 0., 1., 0.25],
            ],
        )
        .with_attribute("under-only", true);
        let mode = Item::new_from_element(BlendMode::Normal).with_attribute("mode-only", true);
        let opacity = Item::new_from_element(100.).with_attribute("opacity-only", true);
        let original = (over.clone(), under.clone(), mode.clone(), opacity.clone());
        let output = evaluate_mix_cpu(over.clone(), under.clone(), mode.clone(), opacity.clone())
            .expect("original CPU Mix");
        assert_eq!(output.attributes(), over.attributes());
        assert!(output.attribute::<bool>("under-only").is_none());
        assert!(output.attribute::<bool>("mode-only").is_none());
        assert!(output.attribute::<bool>("opacity-only").is_none());
        assert_eq!((output.element().width, output.element().height), (4, 1));
        for (pixel, expected) in output.element().data.iter().zip([
            [0., 0., 1., 1.],
            [0.5, 0., 0.5, 1.],
            [1., 0., 0., 1.],
            [0.8, 0., 0.2, 0.625],
        ]) {
            assert_mix_linear(pixel, expected);
        }
        let swapped = evaluate_mix_cpu(under.clone(), over.clone(), mode.clone(), opacity.clone())
            .expect("reverse actual ports");
        assert_eq!(swapped.attributes(), under.attributes());
        assert_mix_linear(
            swapped.element().data.get(1).expect("second pixel"),
            [0., 0., 1., 1.],
        );
        assert_ne!(swapped.element().data, output.element().data);
        assert_eq!(
            evaluate_mix_cpu(over.clone(), under.clone(), mode.clone(), opacity.clone())
                .expect("repeat"),
            output
        );
        assert_gamma_channels(
            over.element().data.get(1).expect("unchanged over"),
            [1., 0., 0., 0.5],
        );
        assert_gamma_channels(
            under.element().data.get(3).expect("unchanged under"),
            [0., 0., 1., 0.25],
        );
        assert_eq!(*mode.element(), BlendMode::Normal);
        assert_eq!(*opacity.element(), 100.);
        assert_eq!((over, under, mode, opacity), original);
    }

    #[test]
    fn at_gra_node_143_2_mix_color_uses_original_percent_clamp_and_blend_mode() {
        use graphite_core_types::blending::BlendMode;
        let over = Item::new_from_element(Color::from_rgbaf32_unchecked(1., 0., 0., 1.))
            .with_attribute("over-name", String::from("red"));
        let under = Item::new_from_element(Color::from_rgbaf32_unchecked(0., 0., 1., 1.))
            .with_attribute("under-only", true);
        let original = (over.clone(), under.clone());
        for (opacity, expected) in [
            (0., [0., 0., 1., 1.]),
            (1., [0.01, 0., 0.99, 1.]),
            (50., [0.5, 0., 0.5, 1.]),
            (100., [1., 0., 0., 1.]),
            (-1., [0., 0., 1., 1.]),
            (200., [1., 0., 0., 1.]),
            (f64::NEG_INFINITY, [0., 0., 1., 1.]),
            (f64::INFINITY, [1., 0., 0., 1.]),
            (f64::NAN, [0., 0., 1., 1.]),
        ] {
            let output = evaluate_mix_color(
                over.clone(),
                under.clone(),
                BlendMode::Normal.into(),
                opacity.into(),
            )
            .expect("original constructor clamp");
            assert_mix_linear(output.element(), expected);
            assert_eq!(output.attributes(), over.attributes());
            assert!(output.attribute::<bool>("under-only").is_none());
        }
        let output = evaluate_mix_color(
            over.clone(),
            under.clone(),
            BlendMode::Multiply.into(),
            100_f64.into(),
        )
        .expect("original multiply blend mode");
        assert_mix_linear(output.element(), [0., 0., 0., 1.]);
        assert_eq!((over, under), original);
    }

    #[test]
    fn at_gra_node_143_2_mix_raw_cpu_zip_is_not_checked_host_shape_admission() {
        use graphite_core_types::blending::BlendMode;
        let over = image_item(2, 1, &[[1., 0., 0., 1.], [0., 1., 0., 1.]]);
        let under = image_item(1, 1, &[[0., 0., 1., 1.]]);
        let output = evaluate_mix_cpu(
            over.clone(),
            under.clone(),
            BlendMode::Normal.into(),
            100_f64.into(),
        )
        .expect("raw source zips pixels");
        assert_eq!((output.element().width, output.element().height), (2, 1));
        assert_eq!(output.element().data.len(), 1);
        assert_mix_linear(
            output.element().data.first().expect("one zipped pixel"),
            [1., 0., 0., 1.],
        );
        let empty = image_item(0, 0, &[]).with_attribute("empty", true);
        let output = evaluate_mix_cpu(
            empty.clone(),
            under.clone(),
            BlendMode::Normal.into(),
            100_f64.into(),
        )
        .expect("empty raw source");
        assert_eq!(output, empty);
        assert_eq!(over.element().data.len(), 2);
        assert_eq!(under.element().data.len(), 1);
    }

    #[test]
    fn at_gra_node_143_2_mix_cpu_list_repeats_all_ranked_ports_and_over_attributes() {
        use graphite_core_types::blending::BlendMode;
        let over: List<_> = [
            image_item(1, 1, &[[1., 0., 0., 1.]]).with_attribute("slot", 0usize),
            image_item(1, 1, &[[0., 1., 0., 0.5]]).with_attribute("slot", 1usize),
        ]
        .into_iter()
        .collect();
        let under: List<_> =
            [image_item(1, 1, &[[0., 0., 1., 1.]]).with_attribute("under-only", true)]
                .into_iter()
                .collect();
        let modes: List<_> = [BlendMode::Normal, BlendMode::Multiply]
            .into_iter()
            .map(Item::new_from_element)
            .collect();
        let opacity: List<_> = [0., 100., 50.]
            .into_iter()
            .map(Item::new_from_element)
            .collect();
        let original = (over.clone(), under.clone(), modes.clone(), opacity.clone());
        let output =
            evaluate_mix_cpu_list(over.clone(), under.clone(), modes.clone(), opacity.clone())
                .expect("original mapped CPU Mix");
        assert_eq!(output.len(), 3);
        for index in 0..output.len() {
            let expected = evaluate_mix_cpu(
                over.clone_item(index.min(1)).expect("original over slot"),
                under.clone_item(0).expect("broadcast under"),
                modes.clone_item(index.min(1)).expect("source mode slot"),
                opacity.clone_item(index).expect("ordered opacity slot"),
            )
            .expect("original Item oracle");
            assert_eq!(output.clone_item(index), Some(expected));
        }
        assert_eq!(output.attribute::<usize>("slot", 2), Some(&1));
        assert!(output.attribute::<bool>("under-only", 2).is_none());
        assert_eq!(
            evaluate_mix_cpu_list(over.clone(), under.clone(), modes.clone(), opacity.clone())
                .expect("repeat"),
            output
        );
        assert_eq!((over, under, modes, opacity), original);
    }

    #[test]
    fn at_gra_node_143_2_mix_color_list_keeps_order_and_original_nonfinite_clamp() {
        use graphite_core_types::blending::BlendMode;
        let over: List<_> = [
            Item::new_from_element(Color::RED).with_attribute("slot", 0usize),
            Item::new_from_element(Color::GREEN).with_attribute("slot", 1usize),
        ]
        .into_iter()
        .collect();
        let under: List<_> = [Item::new_from_element(Color::BLUE)].into_iter().collect();
        let modes: List<_> = [Item::new_from_element(BlendMode::Normal)]
            .into_iter()
            .collect();
        let opacities = [-1., 200., f64::NEG_INFINITY, f64::INFINITY, f64::NAN, 1.];
        let opacity: List<_> = opacities.into_iter().map(Item::new_from_element).collect();
        let original = (over.clone(), under.clone(), modes.clone(), opacity.clone());
        let output =
            evaluate_mix_color_list(over.clone(), under.clone(), modes.clone(), opacity.clone())
                .expect("original mapped Color Mix");
        assert_eq!(output.len(), opacities.len());
        for (index, opacity) in opacities.into_iter().enumerate() {
            let expected = evaluate_mix_color(
                over.clone_item(index.min(1)).expect("over slot"),
                under.clone_item(0).expect("under slot"),
                BlendMode::Normal.into(),
                opacity.into(),
            )
            .expect("source Item clamp oracle");
            assert_eq!(output.clone_item(index), Some(expected));
        }
        assert_eq!(output.element(4), Some(&Color::BLUE));
        assert_eq!(output.element(3), Some(&Color::GREEN));
        assert_eq!(output.attribute::<usize>("slot", 5), Some(&1));
        assert_eq!(
            (&over, &under, &modes),
            (&original.0, &original.1, &original.2)
        );
        for (actual, original) in opacity.into_iter().zip(original.3) {
            assert_eq!(actual.element().to_bits(), original.element().to_bits());
            assert_eq!(actual.attributes(), original.attributes());
        }
    }

    #[test]
    fn at_gra_node_143_2_mix_each_empty_ranked_connector_yields_empty() {
        use graphite_core_types::blending::BlendMode;
        for empty in 0..4 {
            let over: List<_> = [Item::new_from_element(Color::RED)].into_iter().collect();
            let under: List<_> = [Item::new_from_element(Color::BLUE)].into_iter().collect();
            let mode: List<_> = [Item::new_from_element(BlendMode::Normal)]
                .into_iter()
                .collect();
            let opacity: List<_> = [Item::new_from_element(100_f64)].into_iter().collect();
            let output = evaluate_mix_color_list(
                if empty == 0 { List::new() } else { over },
                if empty == 1 { List::new() } else { under },
                if empty == 2 { List::new() } else { mode },
                if empty == 3 { List::new() } else { opacity },
            )
            .expect("empty original ranked connector");
            assert!(output.is_empty());
        }
        assert!(
            evaluate_mix_cpu_list(List::new(), List::new(), List::new(), List::new())
                .expect("empty CPU lists")
                .is_empty()
        );
        assert!(
            evaluate_mix_gradient_list(List::new(), List::new(), List::new(), List::new())
                .expect("empty Gradient lists")
                .is_empty()
        );
    }

    #[test]
    fn at_gra_node_143_2_mix_gradient_keeps_original_union_reconstruction_and_outer_attributes() {
        use graphite_core_types::blending::BlendMode;
        use graphite_core_types::list::{ATTR_MIDPOINT, ATTR_POSITION};
        use graphite_vector_types::GradientStop;
        let mut over_gradient = Gradient::new([
            GradientStop {
                position: 0.,
                midpoint: 0.2,
                color: Color::RED,
            },
            GradientStop {
                position: 1.,
                midpoint: 0.8,
                color: Color::RED,
            },
        ]);
        over_gradient.0.set_attribute(
            "stop-only",
            0,
            String::from("not copied by source reconstruction"),
        );
        let over = Item::new_from_element(over_gradient)
            .with_attribute("ramp-owner", String::from("over"));
        let under = Item::new_from_element(Gradient::new([
            GradientStop {
                position: 0.,
                midpoint: 0.1,
                color: Color::BLUE,
            },
            GradientStop {
                position: 0.5,
                midpoint: 0.9,
                color: Color::BLUE,
            },
            GradientStop {
                position: 1.,
                midpoint: 0.1,
                color: Color::BLUE,
            },
        ]))
        .with_attribute("under-only", true);
        let original = (over.clone(), under.clone());
        let output = evaluate_mix_gradient(
            over.clone(),
            under.clone(),
            BlendMode::Normal.into(),
            50_f64.into(),
        )
        .expect("original Gradient Mix");
        assert_eq!(output.attributes(), over.attributes());
        assert!(output.attribute::<bool>("under-only").is_none());
        assert_eq!(output.element().positions(false), vec![0., 0.5, 1.]);
        for (index, color) in output.element().0.iter_element_values().enumerate() {
            assert_mix_linear(color, [0.5, 0., 0.5, 1.]);
            assert_eq!(
                output.element().0.attribute::<f64>(ATTR_MIDPOINT, index),
                Some(&0.5)
            );
            assert!(
                output
                    .element()
                    .0
                    .attribute::<String>("stop-only", index)
                    .is_none()
            );
            assert_eq!(
                output.element().0.attribute::<f64>(ATTR_POSITION, index),
                Some(&[0., 0.5, 1.][index])
            );
        }
        assert_eq!(
            evaluate_mix_gradient(
                over.clone(),
                under.clone(),
                BlendMode::Normal.into(),
                50_f64.into()
            )
            .expect("repeat original Gradient Mix"),
            output
        );
        assert_eq!((over, under), original);
        let empty = Item::new_from_element(Gradient::new([])).with_attribute("empty-ramp", true);
        let output = evaluate_mix_gradient(
            empty.clone(),
            empty.clone(),
            BlendMode::Normal.into(),
            100_f64.into(),
        )
        .expect("original empty ramps");
        assert_eq!(output, empty);
    }

    #[test]
    fn at_gra_node_143_2_mix_gradient_list_repeats_last_ramp_without_input_mutation() {
        use graphite_core_types::blending::BlendMode;
        let over: List<_> = [
            gradient_item().with_attribute("original-slot", 0usize),
            gradient_item().with_attribute("original-slot", 1usize),
        ]
        .into_iter()
        .collect();
        let under: List<_> = [gradient_item().with_attribute("under-only", true)]
            .into_iter()
            .collect();
        let mode: List<_> = [Item::new_from_element(BlendMode::Normal)]
            .into_iter()
            .collect();
        let opacity: List<_> = [0., 100., 25.]
            .into_iter()
            .map(Item::new_from_element)
            .collect();
        let original = (over.clone(), under.clone(), mode.clone(), opacity.clone());
        let output =
            evaluate_mix_gradient_list(over.clone(), under.clone(), mode.clone(), opacity.clone())
                .expect("original mapped Gradient Mix");
        assert_eq!(output.len(), 3);
        for index in 0..output.len() {
            let expected = evaluate_mix_gradient(
                over.clone_item(index.min(1)).expect("over ramp"),
                under.clone_item(0).expect("under ramp"),
                mode.clone_item(0).expect("mode"),
                opacity.clone_item(index).expect("opacity"),
            )
            .expect("original Item ramp oracle");
            assert_eq!(output.clone_item(index), Some(expected));
        }
        assert_eq!(output.attribute::<usize>("original-slot", 2), Some(&1));
        assert_eq!((over, under, mode, opacity), original);
    }

    #[test]
    fn at_gra_node_143_2_mix_all_original_modes_match_source_kernel_and_ranked_constructor_binding()
    {
        use graphite_core_types::blending::BlendMode;
        use graphite_core_types::misc::Clampable as _;
        use graphite_raster_nodes::blending_nodes::blend_colors;

        let modes: Vec<_> = BlendMode::list()
            .into_iter()
            .flat_map(|group| group.iter().copied())
            .chain([
                BlendMode::Erase,
                BlendMode::Restore,
                BlendMode::MultiplyAlpha,
            ])
            .collect();
        assert_eq!(modes.len(), 29);
        assert_eq!(
            modes
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            29
        );
        let corpus = [
            ([1., 0., 0., 1.], [0., 0., 1., 1.]),
            ([0.25, 0.6, 0.9, 0.5], [0.7, 0.1, 0.3, 0.25]),
            ([0.8, 0.2, 0.4, 0.], [0.1, 0.9, 0.3, 0.]),
            ([0.9, 0.3, 0.6, -0.], [0.2, 0.4, 0.8, 1.]),
            ([1., 1., 1., 1.], [0., 0., 0., 1.]),
            ([0., 0., 0., 1.], [1., 1., 1., 1.]),
            ([0.5, 0.5, 0.5, 1e-8], [0.5, 0.5, 0.5, 0.5]),
        ];
        let opacities = [
            0.,
            1.,
            50.,
            100.,
            -10.,
            110.,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NAN,
        ];
        let color = |[red, green, blue, alpha]: [f32; 4]| {
            Color::from_rgbaf32_unchecked(red, green, blue, alpha)
        };
        let bits = |color: &Color| [color.r(), color.g(), color.b(), color.a()].map(f32::to_bits);
        let oracle = |over: Color, under: Color, mode: BlendMode, opacity: f64| {
            let opacity = opacity.clamp_hard_min(0.).clamp_hard_max(100.);
            blend_colors(over, under, mode, (opacity / 100.) as f32)
        };
        for (corpus_index, (over_channels, under_channels)) in corpus.into_iter().enumerate() {
            let over_color = color(over_channels);
            let under_color = color(under_channels);
            let over =
                Item::new_from_element(over_color).with_attribute("source-slot", corpus_index);
            let under = Item::new_from_element(under_color).with_attribute("under-only", true);
            let over_cpu = Item::new_from_element(Raster::new_cpu(Image::new(1, 1, over_color)))
                .with_attribute("source-slot", corpus_index);
            let under_cpu = Item::new_from_element(Raster::new_cpu(Image::new(1, 1, under_color)))
                .with_attribute("under-only", true);
            let original = (
                over.clone(),
                under.clone(),
                over_cpu.clone(),
                under_cpu.clone(),
            );
            for &mode in &modes {
                for opacity in opacities {
                    let expected = oracle(over_color, under_color, mode, opacity);
                    let mode_item = Item::new_from_element(mode).with_attribute("mode-only", true);
                    let opacity_item =
                        Item::new_from_element(opacity).with_attribute("opacity-only", true);
                    let direct = evaluate_mix_color(
                        over.clone(),
                        under.clone(),
                        mode_item.clone(),
                        opacity_item.clone(),
                    )
                    .expect("original direct Color Mix");
                    assert_eq!(
                        bits(direct.element()),
                        bits(&expected),
                        "Color {mode:?}, opacity {opacity:?}, corpus {corpus_index}"
                    );
                    assert_eq!(direct.attributes(), over.attributes());
                    let cpu = evaluate_mix_cpu(
                        over_cpu.clone(),
                        under_cpu.clone(),
                        mode_item.clone(),
                        opacity_item.clone(),
                    )
                    .expect("original direct CPU Mix");
                    assert_eq!(cpu.attributes(), over_cpu.attributes());
                    assert_eq!(
                        (
                            cpu.element().width,
                            cpu.element().height,
                            cpu.element().data.len()
                        ),
                        (1, 1, 1)
                    );
                    assert_eq!(
                        bits(cpu.element().data.first().expect("CPU pixel")),
                        bits(&expected),
                        "CPU {mode:?}, opacity {opacity:?}, corpus {corpus_index}"
                    );
                    let mapped_color = evaluate_mix_color_list(
                        [over.clone()].into_iter().collect(),
                        [under.clone()].into_iter().collect(),
                        [mode_item.clone()].into_iter().collect(),
                        [opacity_item.clone()].into_iter().collect(),
                    )
                    .expect("original singleton mapped Color Mix");
                    let mapped_cpu = evaluate_mix_cpu_list(
                        [over_cpu.clone()].into_iter().collect(),
                        [under_cpu.clone()].into_iter().collect(),
                        [mode_item.clone()].into_iter().collect(),
                        [opacity_item.clone()].into_iter().collect(),
                    )
                    .expect("original singleton mapped CPU Mix");
                    assert_eq!(mapped_color.len(), 1);
                    assert_eq!(mapped_cpu.len(), 1);
                    let mapped_color = mapped_color.clone_item(0).expect("one mapped Color");
                    let mapped_cpu = mapped_cpu.clone_item(0).expect("one mapped CPU raster");
                    assert_eq!(bits(mapped_color.element()), bits(&expected));
                    assert_eq!(
                        bits(mapped_cpu.element().data.first().expect("mapped CPU pixel")),
                        bits(&expected)
                    );
                    assert_eq!(mapped_color.attributes(), over.attributes());
                    assert_eq!(mapped_cpu.attributes(), over_cpu.attributes());
                    assert_eq!(mode_item.element(), &mode);
                    assert_eq!(mode_item.attribute::<bool>("mode-only"), Some(&true));
                    assert_eq!(opacity_item.element().to_bits(), opacity.to_bits());
                    assert_eq!(opacity_item.attribute::<bool>("opacity-only"), Some(&true));
                }
            }
            assert_eq!(
                (&over, &under, &over_cpu, &under_cpu),
                (&original.0, &original.1, &original.2, &original.3)
            );

            assert_eq!(bits(over.element()), bits(&over_color));
            assert_eq!(bits(under.element()), bits(&under_color));
            assert_eq!(
                bits(over_cpu.element().data.first().expect("unchanged CPU over")),
                bits(&over_color)
            );
            assert_eq!(
                bits(
                    under_cpu
                        .element()
                        .data
                        .first()
                        .expect("unchanged CPU under")
                ),
                bits(&under_color)
            );

            let over_frame: List<_> = [
                over.clone(),
                Item::new_from_element(under_color)
                    .with_attribute("source-slot", corpus_index + 100),
            ]
            .into_iter()
            .collect();
            let over_cpu_frame: List<_> = [
                over_cpu.clone(),
                Item::new_from_element(Raster::new_cpu(Image::new(1, 1, under_color)))
                    .with_attribute("source-slot", corpus_index + 100),
            ]
            .into_iter()
            .collect();
            let under_frame: List<_> = [under.clone()].into_iter().collect();
            let under_cpu_frame: List<_> = [under_cpu.clone()].into_iter().collect();
            let mode_frame: List<_> = modes
                .iter()
                .copied()
                .map(|mode| Item::new_from_element(mode).with_attribute("mode-only", true))
                .collect();
            let opacity_frame: List<_> = [25., 75.]
                .into_iter()
                .map(|opacity| Item::new_from_element(opacity).with_attribute("opacity-only", true))
                .collect();
            let original_frames = (
                over_frame.clone(),
                over_cpu_frame.clone(),
                under_frame.clone(),
                under_cpu_frame.clone(),
                mode_frame.clone(),
                opacity_frame.clone(),
            );
            let mapped_color = evaluate_mix_color_list(
                over_frame.clone(),
                under_frame.clone(),
                mode_frame.clone(),
                opacity_frame.clone(),
            )
            .expect("original 29-mode Color frame");
            let mapped_cpu = evaluate_mix_cpu_list(
                over_cpu_frame.clone(),
                under_cpu_frame.clone(),
                mode_frame.clone(),
                opacity_frame.clone(),
            )
            .expect("original 29-mode CPU frame");
            assert_eq!(mapped_color.len(), modes.len());
            assert_eq!(mapped_cpu.len(), modes.len());
            for (index, &mode) in modes.iter().enumerate() {
                let over_slot = over_frame
                    .clone_item(index.min(1))
                    .expect("original repeated over slot");
                let opacity_slot = opacity_frame
                    .clone_item(index.min(1))
                    .expect("original repeated opacity slot");
                let expected = oracle(
                    *over_slot.element(),
                    under_color,
                    mode,
                    *opacity_slot.element(),
                );
                let actual_color = mapped_color.clone_item(index).expect("ordered mode output");
                let actual_cpu = mapped_cpu
                    .clone_item(index)
                    .expect("ordered CPU mode output");
                assert_eq!(
                    bits(actual_color.element()),
                    bits(&expected),
                    "mode-vector Color {mode:?}"
                );
                assert_eq!(
                    bits(
                        actual_cpu
                            .element()
                            .data
                            .first()
                            .expect("mode-vector pixel")
                    ),
                    bits(&expected),
                    "mode-vector CPU {mode:?}"
                );
                assert_eq!(actual_color.attributes(), over_slot.attributes());
                assert_eq!(actual_cpu.attributes(), over_slot.attributes());
            }
            assert_eq!(
                (
                    over_frame,
                    over_cpu_frame,
                    under_frame,
                    under_cpu_frame,
                    mode_frame,
                    opacity_frame
                ),
                original_frames
            );
        }
    }
}

#[cfg(test)]
mod checked_mix_resource_tests {
    use super::tests::{assert_gamma_channels, gradient_item, image_item};
    use super::*;
    use crate::export::{PngExportError, prepare_cpu_mix_png};
    use crate::resources::{
        AdmittedCpuImage, AlphaAssociation, CpuBlendMode, CpuImageBudget, CpuImageDescriptor,
        CpuImageOrigin, CpuImageParent, CpuImageResultError, CpuMixBudget, CpuMixError,
        CpuMixParameters, CpuSampleFormat, EncodedPngBudget, GRAPHITE_IMAGE_SOURCE_REVISION,
        ImageColorSpace, ImageResourceIdentity, admit_cpu_image, admit_encoded_png,
        evaluate_cpu_image_mix,
    };
    use graphite_core_types::{Color, list::ATTR_TRANSFORM, misc::Clampable as _};
    use graphite_glam::DAffine2;
    use graphite_raster_types::Image;
    use sha2::{Digest, Sha256};
    use std::sync::Arc;

    const IDENTITY: [f64; 6] = [1., 0., 0., 1., 0., 0.];
    const OVER: [u8; 16] = [255, 0, 0, 128, 0, 255, 0, 255, 64, 128, 192, 64, 0, 0, 0, 0];
    const UNDER: [u8; 16] = [
        0, 0, 255, 255, 255, 255, 255, 128, 0, 0, 0, 255, 255, 0, 0, 64,
    ];

    fn raw_image(
        width: u32,
        height: u32,
        asset: &str,
        version: &str,
        bytes: &[u8],
        transform: [f64; 6],
    ) -> AdmittedCpuImage {
        admit_cpu_image(
            ImageResourceIdentity {
                asset_id: asset.into(),
                version_id: version.into(),
                content_sha256: Sha256::digest(bytes).into(),
            },
            CpuImageDescriptor {
                width,
                height,
                row_stride: usize::try_from(width).expect("fixture width") * 4,
                sample_format: CpuSampleFormat::Rgba8,
                color_space: ImageColorSpace::Srgb,
                alpha: AlphaAssociation::Straight,
                transform,
            },
            Arc::from(bytes),
            CpuImageBudget {
                max_input_bytes: bytes.len(),
                max_decoded_bytes: usize::MAX,
            },
        )
        .expect("real original raw pixel admission")
    }

    fn encoded_image(source: &AdmittedCpuImage, version: &str) -> AdmittedCpuImage {
        let bytes: Arc<[u8]> = source.image().to_png().into();
        admit_encoded_png(
            ImageResourceIdentity {
                asset_id: source.identity().asset_id.clone(),
                version_id: version.into(),
                content_sha256: Sha256::digest(bytes.as_ref()).into(),
            },
            source.descriptor().transform,
            bytes.clone(),
            EncodedPngBudget {
                max_encoded_bytes: bytes.len(),
                max_pixel_bytes: usize::MAX,
                max_metadata_bytes: bytes.len(),
            },
        )
        .expect("real original PNG encoder and decoder admission")
    }

    fn budget(pixels: usize) -> CpuMixBudget {
        CpuMixBudget {
            max_parent_bytes: pixels * size_of::<Color>() * 2,
            max_result_bytes: pixels * size_of::<Color>(),
        }
    }

    fn bits(color: &Color) -> [u32; 4] {
        [color.r(), color.g(), color.b(), color.a()].map(f32::to_bits)
    }

    fn assert_pixels(actual: &Image<Color>, expected: &Image<Color>) {
        assert_eq!(
            (actual.width, actual.height),
            (expected.width, expected.height)
        );
        assert_eq!(actual.data.len(), expected.data.len());
        for (actual, expected) in actual.data.iter().zip(&expected.data) {
            assert_eq!(bits(actual), bits(expected));
        }
    }

    fn assert_parent(parent: &CpuImageParent, input: &AdmittedCpuImage) {
        assert_eq!(parent.identity(), input.identity());
        assert_eq!(parent.origin(), input.origin());
        assert_eq!(parent.descriptor(), input.descriptor());
        assert_eq!(
            parent.descriptor().transform.map(f64::to_bits),
            input.descriptor().transform.map(f64::to_bits)
        );
    }

    #[test]
    fn at_gra_node_143_2_checked_mix_original_tiles_preserve_order_sources_and_float_result() {
        for (width, height) in [(2, 2), (16, 12)] {
            let count = usize::try_from(width * height).expect("finite tile");
            let over_bytes = OVER.repeat(count / 4);
            let under_bytes = UNDER.repeat(count / 4);
            let over = raw_image(
                width,
                height,
                "test/over",
                "supplied-over-v1",
                &over_bytes,
                IDENTITY,
            );
            let raw_under = raw_image(
                width,
                height,
                "test/under",
                "supplied-under-v1",
                &under_bytes,
                IDENTITY,
            );
            let under = encoded_image(&raw_under, "supplied-under-png-v1");
            let before = (
                over.image().clone(),
                under.image().clone(),
                over.identity().clone(),
                under.identity().clone(),
            );
            let parameters = CpuMixParameters {
                blend_mode: CpuBlendMode::Normal,
                opacity: 37.5,
            };
            let result = evaluate_cpu_image_mix(&over, &under, parameters, budget(count))
                .expect("checked equal-shape Mix");
            let original = evaluate_mix_cpu(
                Item::new_from_element(Raster::new_cpu(over.image().clone())).with_attribute(
                    ATTR_TRANSFORM,
                    DAffine2::from_cols_array(&over.descriptor().transform),
                ),
                Item::new_from_element(Raster::new_cpu(under.image().clone())).with_attribute(
                    ATTR_TRANSFORM,
                    DAffine2::from_cols_array(&under.descriptor().transform),
                ),
                Item::new_from_element(parameters.blend_mode),
                Item::new_from_element(parameters.opacity),
            )
            .expect("pinned original Mix constructor");
            assert_pixels(result.image(), original.element().data());
            assert_eq!(result.attributes(), original.attributes());
            assert_eq!(result.parents().len(), 2);
            assert_parent(&result.parents()[0], &over);
            assert_parent(&result.parents()[1], &under);
            assert_ne!(
                result.parents()[0].pixels_sha256(),
                result.parents()[1].pixels_sha256()
            );
            assert_eq!(result.parameters(), parameters);
            assert_eq!(result.pixels().image(), result.image());
            assert_eq!(result.pixels().pixels_sha256(), result.pixels_sha256());
            assert_eq!(
                result.descriptor().sample_format,
                CpuSampleFormat::Rgba32Float
            );
            assert_eq!(result.descriptor().color_space, ImageColorSpace::LinearSrgb);
            assert_eq!(result.descriptor().alpha, AlphaAssociation::Straight);
            assert_eq!(
                result.descriptor().row_stride,
                usize::try_from(width).expect("width") * size_of::<Color>()
            );
            assert!(!result.requires_scene_render());
            assert_eq!(
                result.source_operation(),
                "raster_nodes::blending_nodes::MixNode"
            );
            assert_eq!(result.source_revision(), GRAPHITE_IMAGE_SOURCE_REVISION);
            let repeat = evaluate_cpu_image_mix(&over, &under, parameters, budget(count))
                .expect("repeat original evaluation");
            assert_pixels(repeat.image(), result.image());
            assert_eq!(repeat.pixels_sha256(), result.pixels_sha256());
            assert_pixels(over.image(), &before.0);
            assert_pixels(under.image(), &before.1);
            assert_eq!((over.identity(), under.identity()), (&before.2, &before.3));
        }
    }

    #[test]
    fn at_gra_node_143_2_checked_mix_keeps_duplicate_roles_and_distinct_parent_provenance() {
        let opaque = [255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 0, 0, 0, 255];
        let first = raw_image(2, 2, "test/shared", "supplied-v1", &opaque, IDENTITY);
        let changed = raw_image(2, 2, "test/shared", "supplied-v2", &opaque, IDENTITY);
        let encoded = encoded_image(&first, "supplied-png-v3");
        assert_pixels(encoded.image(), first.image());
        let parameters = CpuMixParameters {
            blend_mode: CpuBlendMode::Normal,
            opacity: 100.,
        };
        let duplicate = evaluate_cpu_image_mix(&first, &first, parameters, budget(4))
            .expect("same immutable input has two ordered roles");
        assert_eq!(duplicate.parents().len(), 2);
        assert_parent(&duplicate.parents()[0], &first);
        assert_parent(&duplicate.parents()[1], &first);
        assert_eq!(
            duplicate.parents()[0].pixels_sha256(),
            duplicate.parents()[1].pixels_sha256()
        );
        let versions = evaluate_cpu_image_mix(&first, &changed, parameters, budget(4))
            .expect("same pixels distinct supplied versions");
        let origins = evaluate_cpu_image_mix(&changed, &encoded, parameters, budget(4))
            .expect("same pixels distinct original admission origins");
        assert_parent(&versions.parents()[0], &first);
        assert_parent(&versions.parents()[1], &changed);
        assert_parent(&origins.parents()[0], &changed);
        assert_parent(&origins.parents()[1], &encoded);
        assert_ne!(
            versions.parents()[0].identity(),
            versions.parents()[1].identity()
        );
        assert_ne!(origins.parents()[0].origin(), origins.parents()[1].origin());
        assert!(matches!(
            origins.parents()[0].origin(),
            CpuImageOrigin::Rgba8Rows { bytes: 16 }
        ));
        assert!(matches!(
            origins.parents()[1].origin(),
            CpuImageOrigin::EncodedPng {
                bit_depth: 8,
                channels: 4,
                ..
            }
        ));
        for parent in versions.parents().iter().chain(origins.parents()) {
            assert_eq!(
                parent.pixels_sha256(),
                duplicate.parents()[0].pixels_sha256()
            );
        }
        assert_eq!(versions.pixels_sha256(), duplicate.pixels_sha256());
        assert_eq!(origins.pixels_sha256(), duplicate.pixels_sha256());
        assert_pixels(versions.image(), duplicate.image());
        assert_pixels(origins.image(), duplicate.image());
        let reversed = evaluate_cpu_image_mix(&encoded, &changed, parameters, budget(4))
            .expect("preserved reverse parent order");
        assert_parent(&reversed.parents()[0], &encoded);
        assert_parent(&reversed.parents()[1], &changed);
    }

    #[test]
    fn at_gra_node_143_2_checked_mix_rejects_equal_count_different_shape_before_original_zip() {
        let over = raw_image(2, 2, "test/shape-over", "v1", &OVER, IDENTITY);
        let under = raw_image(4, 1, "test/shape-under", "v1", &UNDER, IDENTITY);
        let before = (over.image().clone(), under.image().clone());
        let parameters = CpuMixParameters {
            blend_mode: CpuBlendMode::Normal,
            opacity: 100.,
        };
        assert_eq!(
            evaluate_cpu_image_mix(&over, &under, parameters, budget(4))
                .expect_err("equal count is not equal dimensions"),
            CpuMixError::ParentShapeMismatch
        );
        assert_pixels(over.image(), &before.0);
        assert_pixels(under.image(), &before.1);
    }

    #[test]
    fn at_gra_node_143_2_checked_mix_caps_and_scene_requirements_are_atomic_and_recoverable() {
        let over = raw_image(2, 2, "test/cap-over", "v1", &OVER, IDENTITY);
        let under = raw_image(2, 2, "test/cap-under", "v1", &UNDER, IDENTITY);
        let before = (over.image().clone(), under.image().clone());
        let parameters = CpuMixParameters {
            blend_mode: CpuBlendMode::Normal,
            opacity: 100.,
        };
        let exact = budget(4);
        for max_parent_bytes in [0, exact.max_parent_bytes - 1] {
            assert_eq!(
                evaluate_cpu_image_mix(
                    &over,
                    &under,
                    parameters,
                    CpuMixBudget {
                        max_parent_bytes,
                        ..exact
                    }
                )
                .expect_err("parent cap"),
                CpuMixError::ParentBudgetExceeded
            );
        }
        for max_result_bytes in [0, exact.max_result_bytes - 1] {
            assert_eq!(
                evaluate_cpu_image_mix(
                    &over,
                    &under,
                    parameters,
                    CpuMixBudget {
                        max_result_bytes,
                        ..exact
                    }
                )
                .expect_err("result cap"),
                CpuMixError::Image(CpuImageResultError::ResultBudgetExceeded)
            );
        }
        let transformed = raw_image(2, 2, "test/scene", "v1", &OVER, [1., 0., 0., 1., 12., 0.]);
        for (over, under) in [(&transformed, &under), (&over, &transformed)] {
            assert_eq!(
                evaluate_cpu_image_mix(over, under, parameters, exact)
                    .expect_err("scene transform must not be discarded"),
                CpuMixError::SceneRenderRequired
            );
        }
        let recovered = evaluate_cpu_image_mix(&over, &under, parameters, exact)
            .expect("exact retained-byte caps");
        assert_eq!((recovered.image().width, recovered.image().height), (2, 2));
        assert_pixels(over.image(), &before.0);
        assert_pixels(under.image(), &before.1);
        assert_eq!(
            over.descriptor().transform.map(f64::to_bits),
            IDENTITY.map(f64::to_bits)
        );
        assert_eq!(
            under.descriptor().transform.map(f64::to_bits),
            IDENTITY.map(f64::to_bits)
        );
    }

    #[test]
    fn at_gra_node_143_2_checked_mix_selected_utilities_and_original_hard_clamp_are_preserved() {
        let over = raw_image(2, 2, "test/utility-over", "v1", &OVER, IDENTITY);
        let under = raw_image(2, 2, "test/utility-under", "v1", &UNDER, IDENTITY);
        let before = (over.image().clone(), under.image().clone());
        for mode in [
            CpuBlendMode::Normal,
            CpuBlendMode::Erase,
            CpuBlendMode::Restore,
            CpuBlendMode::MultiplyAlpha,
        ] {
            for opacity in [
                -1.,
                -0.,
                0.,
                37.5,
                100.,
                101.,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NAN,
            ] {
                let parameters = CpuMixParameters {
                    blend_mode: mode,
                    opacity,
                };
                let effective = opacity.clamp_hard_min(0.).clamp_hard_max(100.);
                assert_eq!(
                    parameters.effective_opacity_percent().to_bits(),
                    effective.to_bits()
                );
                let result = evaluate_cpu_image_mix(&over, &under, parameters, budget(4))
                    .expect("original clamp and utility constructor");
                assert_eq!(result.parameters().opacity.to_bits(), opacity.to_bits());
                assert_eq!(result.parameters(), parameters);
                for ((actual, over), under) in result
                    .image()
                    .data
                    .iter()
                    .zip(&over.image().data)
                    .zip(&under.image().data)
                {
                    let original = graphite_raster_nodes::blending_nodes::blend_colors(
                        *over,
                        *under,
                        mode,
                        (effective / 100.) as f32,
                    );
                    assert_eq!(bits(actual), bits(&original));
                }
            }
        }
        assert_ne!(
            CpuMixParameters {
                blend_mode: CpuBlendMode::Normal,
                opacity: 0.
            },
            CpuMixParameters {
                blend_mode: CpuBlendMode::Normal,
                opacity: -0.
            }
        );
        assert_pixels(over.image(), &before.0);
        assert_pixels(under.image(), &before.1);
    }

    #[test]
    fn at_gra_node_143_2_checked_mix_signed_zero_transform_bits_follow_only_over_attributes() {
        let over_transform = [1., -0., 0., 1., -0., 0.];
        let under_transform = [1., 0., -0., 1., 0., -0.];
        let over = raw_image(2, 2, "test/signed-over", "v1", &OVER, over_transform);
        let under = raw_image(2, 2, "test/signed-under", "v1", &UNDER, under_transform);
        let result = evaluate_cpu_image_mix(
            &over,
            &under,
            CpuMixParameters {
                blend_mode: CpuBlendMode::Normal,
                opacity: 75.,
            },
            budget(4),
        )
        .expect("numerically identity transforms retain exact source metadata");
        assert_parent(&result.parents()[0], &over);
        assert_parent(&result.parents()[1], &under);
        assert_eq!(
            result.descriptor().transform.map(f64::to_bits),
            over_transform.map(f64::to_bits)
        );
        assert_eq!(
            result
                .attributes()
                .get::<DAffine2>(ATTR_TRANSFORM)
                .expect("original Over attribute")
                .to_cols_array()
                .map(f64::to_bits),
            over_transform.map(f64::to_bits)
        );
        assert_ne!(
            result.descriptor().transform.map(f64::to_bits),
            under_transform.map(f64::to_bits)
        );
        assert!(!result.requires_scene_render());
    }

    #[test]
    fn at_r22_318_1_mix_png_reuses_original_encoder_and_retains_both_parent_provenance() {
        let over = raw_image(2, 2, "test/png-over", "supplied-over-v1", &OVER, IDENTITY);
        let raw_under = raw_image(
            2,
            2,
            "test/png-under",
            "supplied-under-v1",
            &UNDER,
            IDENTITY,
        );
        let under = encoded_image(&raw_under, "supplied-encoded-v1");
        let before = (over.image().clone(), under.image().clone());
        let parameters = CpuMixParameters {
            blend_mode: CpuBlendMode::Normal,
            opacity: 37.5,
        };
        let result = evaluate_cpu_image_mix(&over, &under, parameters, budget(4))
            .expect("original composed pixels");
        let original_pixels = result.image().clone();
        let original_attributes = result.attributes().clone();
        let original_png = result.image().to_png();
        let prepared =
            prepare_cpu_mix_png(&result, original_png.len()).expect("exact PNG result-size cap");
        assert_eq!(prepared.png().encoded_bytes(), original_png);
        assert_eq!(
            prepared.png().encoded_bytes().get(..8),
            Some(&b"\x89PNG\r\n\x1a\n"[..])
        );
        assert_eq!(
            prepared.png().encoded_bytes().get(12..16),
            Some(&b"IHDR"[..])
        );
        assert_eq!(
            prepared.png().encoded_bytes().get(16..24),
            Some(&[0, 0, 0, 2, 0, 0, 0, 2][..])
        );
        assert_eq!(
            prepared.png().encoded_bytes().get(24..26),
            Some(&[8, 6][..])
        );
        let decoded = Image::<Color>::from_encoded(prepared.png().encoded_bytes())
            .expect("original PNG decoder");
        let original_decoded =
            Image::<Color>::from_encoded(&original_png).expect("original source PNG");
        assert_pixels(&decoded, &original_decoded);
        assert_eq!((prepared.png().width(), prepared.png().height()), (2, 2));
        assert_eq!(
            prepared.png().content_sha256(),
            <[u8; 32]>::from(Sha256::digest(prepared.png().encoded_bytes()))
        );
        assert_eq!(prepared.png().sample_format(), CpuSampleFormat::Rgba8);
        assert_eq!(prepared.png().color_space(), ImageColorSpace::Srgb);
        assert_eq!(prepared.png().alpha(), AlphaAssociation::Straight);
        assert_eq!(prepared.png().content_type(), "image/png");
        assert_parent(&prepared.parents()[0], &over);
        assert_parent(&prepared.parents()[1], &under);
        for (parent, original) in prepared.parents().iter().zip(result.parents()) {
            assert_eq!(parent.pixels_sha256(), original.pixels_sha256());
        }
        assert_eq!(prepared.parameters(), parameters);
        assert_eq!(prepared.pixels_sha256(), result.pixels_sha256());
        assert_eq!(prepared.source_operation(), result.source_operation());
        assert_eq!(
            prepared.source_revision(),
            "28ed22824fae8828756f6f4163db62f3585f21e0"
        );
        for limit in [0, original_png.len() - 1] {
            assert_eq!(
                prepare_cpu_mix_png(&result, limit).expect_err("encoded size cap"),
                PngExportError::EncodedSizeExceeded {
                    actual: original_png.len(),
                    limit
                }
            );
        }
        let recovered =
            prepare_cpu_mix_png(&result, original_png.len()).expect("recovered preparation");
        assert_eq!(
            recovered.png().encoded_bytes(),
            prepared.png().encoded_bytes()
        );
        assert_eq!(
            recovered.png().content_sha256(),
            prepared.png().content_sha256()
        );
        assert_pixels(result.image(), &original_pixels);
        assert_eq!(result.attributes(), &original_attributes);
        assert_pixels(over.image(), &before.0);
        assert_pixels(under.image(), &before.1);
    }

    fn black_and_white_parameters(
        use_tint: bool,
        tint: [f32; 4],
        percentages: [f64; 6],
    ) -> BlackAndWhiteParameters {
        let [red, green, blue, alpha] = tint;
        let [reds, yellows, greens, cyans, blues, magentas] = percentages;
        BlackAndWhiteParameters {
            use_tint: Item::new_from_element(use_tint),
            tint: Item::new_from_element(Color::from_gamma_srgb_channels(red, green, blue, alpha)),
            reds: Item::new_from_element(reds),
            yellows: Item::new_from_element(yellows),
            greens: Item::new_from_element(greens),
            cyans: Item::new_from_element(cyans),
            blues: Item::new_from_element(blues),
            magentas: Item::new_from_element(magentas),
        }
    }

    fn original_black_and_white_color(color: Color, parameters: BlackAndWhiteParameters) -> Color {
        let node = graphite_raster_nodes::adjustments::BlackAndWhiteNode::new(
            FutureWrapperNode::new(ClonedNode::new(Item::new_from_element(color))),
            FutureWrapperNode::new(ClonedNode::new(parameters.use_tint)),
            FutureWrapperNode::new(ClonedNode::new(parameters.tint)),
            FutureWrapperNode::new(ClonedNode::new(parameters.reds)),
            FutureWrapperNode::new(ClonedNode::new(parameters.yellows)),
            FutureWrapperNode::new(ClonedNode::new(parameters.greens)),
            FutureWrapperNode::new(ClonedNode::new(parameters.cyans)),
            FutureWrapperNode::new(ClonedNode::new(parameters.blues)),
            FutureWrapperNode::new(ClonedNode::new(parameters.magentas)),
        );
        futures::executor::block_on(node.eval(())).into_element()
    }

    #[test]
    fn at_gra_node_144_2_black_and_white_preserves_connector_order_and_gamma_space() {
        let input = image_item(
            7,
            1,
            &[
                [1., 0., 0., 1.],
                [1., 1., 0., 1.],
                [0., 1., 0., 1.],
                [0., 1., 1., 1.],
                [0., 0., 1., 1.],
                [1., 0., 1., 1.],
                [0.25, 0.25, 0.25, 0.5],
            ],
        );
        let parameters =
            black_and_white_parameters(false, [0.9, 0.2, 0.1, 0.], [11., 22., 33., 44., 55., 66.]);
        let output = evaluate_black_and_white_cpu(input, parameters).expect("original CPU node");
        let expected = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.25];
        assert_eq!(output.element().data().data.len(), expected.len());
        for (index, (pixel, luminance)) in output
            .element()
            .data()
            .data
            .iter()
            .zip(expected)
            .enumerate()
        {
            let alpha = if index == 6 { 0.5 } else { 1. };
            assert_gamma_channels(pixel, [luminance, luminance, luminance, alpha]);
        }
    }

    #[test]
    fn at_gra_node_144_2_black_and_white_does_not_clamp_soft_range_inputs() {
        for (color, reds, expected) in [
            ([0.1, 0., 0., 1.], 400., 0.4),
            ([0.4, 0.3, 0.3, 0.5], -300., 0.),
            ([1., 0., 0., 1.], 400., 1.),
        ] {
            let input = image_item(1, 1, &[color]);
            let parameters =
                black_and_white_parameters(false, [1., 1., 1., 1.], [reds, 0., 0., 0., 0., 0.]);
            let output =
                evaluate_black_and_white_cpu(input, parameters).expect("soft range is not a clamp");
            assert_gamma_channels(
                &output.element().data().data[0],
                [expected, expected, expected, color[3]],
            );
        }
    }

    #[test]
    fn at_gra_node_144_2_black_and_white_matches_pinned_fixed_tint_oracles() {
        for (input, tint, expected, tolerance) in [
            (
                [200., 200., 200.],
                [225., 211., 179.],
                [213., 199., 167.],
                1.,
            ),
            ([50., 50., 50.], [225., 211., 179.], [63., 49., 17.], 1.),
            ([200., 100., 50.], [225., 211., 179.], [133., 119., 87.], 1.),
            ([200., 200., 200.], [30., 60., 120.], [176., 202., 255.], 1.),
            ([50., 50., 50.], [30., 60., 120.], [22., 52., 112.], 1.),
            ([200., 100., 50.], [30., 60., 120.], [92., 122., 182.], 1.),
            ([1., 1., 1.], [255., 0., 0.], [3.33, 0., 0.], 0.51),
            ([78., 78., 78.], [255., 0., 0.], [255., 2.15, 2.15], 0.51),
            (
                [200., 200., 200.],
                [255., 0., 0.],
                [255., 176.43, 176.43],
                0.51,
            ),
        ] {
            let image = image_item(
                1,
                1,
                &[[input[0] / 255., input[1] / 255., input[2] / 255., 0.5]],
            );
            let parameters = black_and_white_parameters(
                true,
                [tint[0] / 255., tint[1] / 255., tint[2] / 255., 1.],
                [40., 60., 40., 60., 20., 80.],
            );
            let output =
                evaluate_black_and_white_cpu(image, parameters).expect("original tint path");
            let [red, green, blue, alpha] =
                output.element().data().data[0].to_gamma_srgb_channels();
            for (actual, expected) in [red * 255., green * 255., blue * 255.]
                .into_iter()
                .zip(expected)
            {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "expected {expected}, got {actual}"
                );
            }
            assert_eq!(alpha.to_bits(), 0.5_f32.to_bits());
        }
    }

    #[test]
    fn at_gra_node_144_2_black_and_white_ignores_tint_alpha_and_retains_input_alpha() {
        let image = image_item(
            2,
            2,
            &[
                [0.8, 0.4, 0.2, 0.],
                [0.1, 0.3, 0.7, -0.],
                [0.5, 0.5, 0.5, 0.5],
                [1., 0., 0., 1.],
            ],
        );
        for use_tint in [false, true] {
            let mut prior = None;
            for tint_alpha in [0., -0., 0.5, 1.] {
                let parameters = black_and_white_parameters(
                    use_tint,
                    [0.8, 0.6, 0.3, tint_alpha],
                    [40., 60., 40., 60., 20., 80.],
                );
                let output = evaluate_black_and_white_cpu(image.clone(), parameters)
                    .expect("alpha-preserving source path");
                for (pixel, original) in output
                    .element()
                    .data()
                    .data
                    .iter()
                    .zip(&image.element().data().data)
                {
                    assert_eq!(
                        pixel.to_gamma_srgb_channels()[3].to_bits(),
                        original.to_gamma_srgb_channels()[3].to_bits()
                    );
                }
                if let Some(prior) = prior.as_ref() {
                    assert_eq!(&output, prior);
                }
                prior = Some(output);
            }
        }
    }

    #[test]
    fn at_gra_node_144_2_black_and_white_cpu_matches_original_color_constructor_and_retains_items()
    {
        let input = image_item(
            3,
            2,
            &[
                [0.25, 0.1, 0.8, 0.],
                [0.2, 0.9, 0.4, -0.],
                [0.7, 0.5, 0.1, 0.125],
                [1., 0., 1., 0.5],
                [0., 1., 1., 1.],
                [0.3, 0.3, 0.3, 1.],
            ],
        )
        .with_attribute("asset", String::from("black-and-white-input"))
        .with_attribute("revision", 19_u64)
        .with_attribute("visible", true);
        let original_input = input.clone();
        for use_tint in [false, true] {
            let mut parameters = black_and_white_parameters(
                use_tint,
                [0.7, 0.5, 0.2, 0.5],
                [-310., 430., 13., 57., 89., 121.],
            );
            parameters.use_tint = parameters
                .use_tint
                .with_attribute("parameter", String::from("use-tint"));
            parameters.tint = parameters
                .tint
                .with_attribute("parameter", String::from("tint"));
            parameters.reds = parameters
                .reds
                .with_attribute("parameter", String::from("reds"));
            parameters.yellows = parameters
                .yellows
                .with_attribute("parameter", String::from("yellows"));
            parameters.greens = parameters
                .greens
                .with_attribute("parameter", String::from("greens"));
            parameters.cyans = parameters
                .cyans
                .with_attribute("parameter", String::from("cyans"));
            parameters.blues = parameters
                .blues
                .with_attribute("parameter", String::from("blues"));
            parameters.magentas = parameters
                .magentas
                .with_attribute("parameter", String::from("magentas"));
            let original_parameters = parameters.clone();
            let output = evaluate_black_and_white_cpu(input.clone(), parameters.clone())
                .expect("CPU original constructor");
            let repeat = evaluate_black_and_white_cpu(input.clone(), parameters.clone())
                .expect("repeat original constructor");
            assert_eq!(output, repeat);
            assert_eq!(input, original_input);
            assert_eq!(parameters, original_parameters);
            assert_eq!(output.attributes(), input.attributes());
            assert_eq!(
                (
                    output.element().data().width,
                    output.element().data().height
                ),
                (3, 2)
            );
            assert_eq!(
                output.element().data().data.len(),
                input.element().data().data.len()
            );
            for (actual, input_color) in output
                .element()
                .data()
                .data
                .iter()
                .zip(&input.element().data().data)
            {
                let expected = original_black_and_white_color(*input_color, parameters.clone());
                assert_eq!(actual, &expected);
                assert_eq!(
                    actual.to_gamma_srgb_channels()[3].to_bits(),
                    input_color.to_gamma_srgb_channels()[3].to_bits()
                );
            }
        }
    }

    #[test]
    fn at_gra_node_144_2_black_and_white_preserves_empty_rasters_and_attributes() {
        for (width, height) in [(0, 0), (0, 3), (4, 0)] {
            let input = image_item(width, height, &[])
                .with_attribute("empty", String::from("source-retained"));
            let original = input.clone();
            let parameters = black_and_white_parameters(
                true,
                [0.9, 0.8, 0.7, 0.],
                [400., -300., 40., 60., 20., 80.],
            );
            let first = evaluate_black_and_white_cpu(input.clone(), parameters.clone())
                .expect("empty source raster");
            let second = evaluate_black_and_white_cpu(input.clone(), parameters)
                .expect("repeat empty source raster");
            assert_eq!(input, original);
            assert_eq!(first, original);
            assert_eq!(second, first);
        }
    }

    fn black_and_white_list_parameters() -> BlackAndWhiteListParameters {
        BlackAndWhiteListParameters {
            use_tint: [false, true]
                .into_iter()
                .map(Item::new_from_element)
                .collect(),
            tint: [Color::from_gamma_srgb_channels(0.8, 0.6, 0.3, 0.)]
                .into_iter()
                .map(Item::new_from_element)
                .collect(),
            reds: [11., 73., 430.]
                .into_iter()
                .map(Item::new_from_element)
                .collect(),
            yellows: [22.].into_iter().map(Item::new_from_element).collect(),
            greens: [33., -310.]
                .into_iter()
                .map(Item::new_from_element)
                .collect(),
            cyans: [44.].into_iter().map(Item::new_from_element).collect(),
            blues: [55., 80., 110., -270.]
                .into_iter()
                .map(Item::new_from_element)
                .collect(),
            magentas: [66.].into_iter().map(Item::new_from_element).collect(),
        }
    }

    fn black_and_white_list_slot(
        parameters: &BlackAndWhiteListParameters,
        index: usize,
    ) -> BlackAndWhiteParameters {
        let use_tint = parameters
            .use_tint
            .clone_item(index.min(parameters.use_tint.len() - 1))
            .expect("nonempty original use_tint slot");
        let tint = parameters
            .tint
            .clone_item(index.min(parameters.tint.len() - 1))
            .expect("nonempty original tint slot");
        let reds = parameters
            .reds
            .clone_item(index.min(parameters.reds.len() - 1))
            .expect("nonempty original reds slot");
        let yellows = parameters
            .yellows
            .clone_item(index.min(parameters.yellows.len() - 1))
            .expect("nonempty original yellows slot");
        let greens = parameters
            .greens
            .clone_item(index.min(parameters.greens.len() - 1))
            .expect("nonempty original greens slot");
        let cyans = parameters
            .cyans
            .clone_item(index.min(parameters.cyans.len() - 1))
            .expect("nonempty original cyans slot");
        let blues = parameters
            .blues
            .clone_item(index.min(parameters.blues.len() - 1))
            .expect("nonempty original blues slot");
        let magentas = parameters
            .magentas
            .clone_item(index.min(parameters.magentas.len() - 1))
            .expect("nonempty original magentas slot");
        BlackAndWhiteParameters {
            use_tint,
            tint,
            reds,
            yellows,
            greens,
            cyans,
            blues,
            magentas,
        }
    }

    fn assert_black_and_white_gradient(
        output: &Item<Gradient>,
        input: &Item<Gradient>,
        parameters: &BlackAndWhiteParameters,
    ) {
        use graphite_core_types::list::{ATTR_MIDPOINT, ATTR_POSITION};
        assert_eq!(output.attributes(), input.attributes());
        assert_eq!(output.element().len(), input.element().len());
        for (actual, original) in output
            .element()
            .clone()
            .into_color_list()
            .into_iter()
            .zip(input.element().clone().into_color_list())
        {
            let expected = original_black_and_white_color(*original.element(), parameters.clone());
            assert_eq!(actual.element(), &expected);
            assert_eq!(actual.attributes(), original.attributes());
            assert_eq!(
                actual.attribute::<f64>(ATTR_POSITION),
                original.attribute::<f64>(ATTR_POSITION)
            );
            assert_eq!(
                actual.attribute::<f64>(ATTR_MIDPOINT),
                original.attribute::<f64>(ATTR_MIDPOINT)
            );
            assert_eq!(
                actual.element().to_gamma_srgb_channels()[3].to_bits(),
                original.element().to_gamma_srgb_channels()[3].to_bits()
            );
        }
    }

    #[test]
    fn at_gra_node_144_2_black_and_white_color_has_fixed_gamma_results_and_attributes() {
        for (channels, expected) in [
            ([1., 0., 0., 0.], 0.4),
            ([1., 1., 0., -0.], 0.6),
            ([0., 1., 0., 0.5], 0.4),
            ([0., 1., 1., 1.], 0.6),
            ([0., 0., 1., 1.], 0.2),
            ([1., 0., 1., 0.125], 0.8),
            ([0.25, 0.25, 0.25, 1.], 0.25),
        ] {
            let [red, green, blue, alpha] = channels;
            let input =
                Item::new_from_element(Color::from_gamma_srgb_channels(red, green, blue, alpha))
                    .with_attribute("color-source", String::from("retained"));
            let original = input.clone();
            let parameters = black_and_white_parameters(
                false,
                [0.1, 0.8, 0.6, 0.],
                [40., 60., 40., 60., 20., 80.],
            );
            let output = evaluate_black_and_white_color(input.clone(), parameters.clone())
                .expect("original Color signature");
            assert_gamma_channels(output.element(), [expected, expected, expected, alpha]);
            assert_eq!(
                output.element().to_gamma_srgb_channels()[3].to_bits(),
                alpha.to_bits()
            );
            assert_eq!(output.attributes(), input.attributes());
            assert_eq!(
                output,
                evaluate_black_and_white_color(input.clone(), parameters).expect("repeat")
            );
            assert_eq!(input, original);
        }
    }

    #[test]
    fn at_gra_node_144_2_black_and_white_gradient_retains_stops_ramp_and_empty() {
        for input in [
            gradient_item(),
            Item::new_from_element(Gradient::default()).with_attribute("empty-ramp", true),
        ] {
            let original = input.clone();
            for use_tint in [false, true] {
                let parameters = black_and_white_parameters(
                    use_tint,
                    [0.8, 0.6, 0.3, -0.],
                    [-310., 430., 13., 57., 89., 121.],
                );
                let original_parameters = parameters.clone();
                let output = evaluate_black_and_white_gradient(input.clone(), parameters.clone())
                    .expect("original Gradient signature");
                assert_black_and_white_gradient(&output, &input, &parameters);
                assert_eq!(
                    output,
                    evaluate_black_and_white_gradient(input.clone(), parameters.clone())
                        .expect("repeat")
                );
                assert_eq!(parameters, original_parameters);
            }
            assert_eq!(input, original);
        }
    }

    #[test]
    fn at_gra_node_144_2_black_and_white_color_list_uses_every_ranked_connector_frame() {
        for [red, green, blue] in [
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.],
            [0., 1., 1.],
            [0., 0., 1.],
            [1., 0., 1.],
            [0.25, 0.8, 0.1],
        ] {
            let color =
                Item::new_from_element(Color::from_gamma_srgb_channels(red, green, blue, -0.))
                    .with_attribute("color-slot", 7usize);
            for longest_connector in 0..9 {
                let mut input: List<Color> = [color.clone()].into_iter().collect();
                let mut parameters = black_and_white_list_parameters();
                parameters.use_tint = [parameters.use_tint.clone_item(0).expect("first parameter")]
                    .into_iter()
                    .collect();
                parameters.tint = [parameters.tint.clone_item(0).expect("first parameter")]
                    .into_iter()
                    .collect();
                parameters.reds = [parameters.reds.clone_item(0).expect("first parameter")]
                    .into_iter()
                    .collect();
                parameters.yellows = [parameters.yellows.clone_item(0).expect("first parameter")]
                    .into_iter()
                    .collect();
                parameters.greens = [parameters.greens.clone_item(0).expect("first parameter")]
                    .into_iter()
                    .collect();
                parameters.cyans = [parameters.cyans.clone_item(0).expect("first parameter")]
                    .into_iter()
                    .collect();
                parameters.blues = [parameters.blues.clone_item(0).expect("first parameter")]
                    .into_iter()
                    .collect();
                parameters.magentas = [parameters.magentas.clone_item(0).expect("first parameter")]
                    .into_iter()
                    .collect();
                match longest_connector {
                    0 => {
                        input = [
                            color.clone(),
                            color.clone().with_attribute("color-slot", 8usize),
                            color.clone().with_attribute("color-slot", 9usize),
                        ]
                        .into_iter()
                        .collect()
                    }
                    1 => {
                        parameters.use_tint = [false, true, false]
                            .into_iter()
                            .map(Item::new_from_element)
                            .collect()
                    }
                    2 => {
                        parameters.use_tint = [Item::new_from_element(true)].into_iter().collect();
                        parameters.tint = [
                            Color::from_gamma_srgb_channels(1., 0., 0., 0.),
                            Color::from_gamma_srgb_channels(0., 1., 0., 0.5),
                            Color::from_gamma_srgb_channels(0., 0., 1., 1.),
                        ]
                        .into_iter()
                        .map(Item::new_from_element)
                        .collect();
                    }
                    3 => {
                        parameters.reds = [11., 70., 430.]
                            .into_iter()
                            .map(Item::new_from_element)
                            .collect()
                    }
                    4 => {
                        parameters.yellows = [22., 80., -310.]
                            .into_iter()
                            .map(Item::new_from_element)
                            .collect()
                    }
                    5 => {
                        parameters.greens = [33., 90., 430.]
                            .into_iter()
                            .map(Item::new_from_element)
                            .collect()
                    }
                    6 => {
                        parameters.cyans = [44., 100., -310.]
                            .into_iter()
                            .map(Item::new_from_element)
                            .collect()
                    }
                    7 => {
                        parameters.blues = [55., 110., 430.]
                            .into_iter()
                            .map(Item::new_from_element)
                            .collect()
                    }
                    8 => {
                        parameters.magentas = [66., 120., -310.]
                            .into_iter()
                            .map(Item::new_from_element)
                            .collect()
                    }
                    _ => unreachable!("finite connector cases"),
                }
                let original = (input.clone(), parameters.clone());
                let output = evaluate_black_and_white_color_list(input.clone(), parameters.clone())
                    .expect("all nine mapped connectors");
                assert_eq!(
                    output.len(),
                    3,
                    "connector {longest_connector} frames output"
                );
                for index in 0..3 {
                    let original_color = input
                        .clone_item(index.min(input.len() - 1))
                        .expect("ranked input");
                    let expected = original_black_and_white_color(
                        *original_color.element(),
                        black_and_white_list_slot(&parameters, index),
                    );
                    let actual = output.clone_item(index).expect("ordered output");
                    assert_eq!(actual.element(), &expected);
                    assert_eq!(actual.attributes(), original_color.attributes());
                    assert_eq!(
                        actual.element().to_gamma_srgb_channels()[3].to_bits(),
                        (-0_f32).to_bits()
                    );
                }
                assert_eq!(
                    output,
                    evaluate_black_and_white_color_list(input.clone(), parameters.clone())
                        .expect("repeat mapped Color")
                );
                assert_eq!((input, parameters), original);
            }
        }
    }

    #[test]
    fn at_gra_node_144_2_black_and_white_cpu_list_preserves_broadcast_dimensions_and_pixels() {
        let input: List<_> = [
            image_item(2, 1, &[[0.2, 0.8, 0.4, -0.], [0.1, 0.5, 0.9, 0.5]])
                .with_attribute("image-slot", 0usize),
            image_item(1, 2, &[[1., 0., 0., 0.], [0., 0., 1., 1.]])
                .with_attribute("image-slot", 1usize),
        ]
        .into_iter()
        .collect();
        let parameters = black_and_white_list_parameters();
        let original = (input.clone(), parameters.clone());
        let output = evaluate_black_and_white_cpu_list(input.clone(), parameters.clone())
            .expect("original mapped CPU signature");
        assert_eq!(output.len(), 4);
        for index in 0..4 {
            let original_image = input.clone_item(index.min(1)).expect("source raster slot");
            let actual = output.clone_item(index).expect("output raster slot");
            let expected = evaluate_black_and_white_cpu(
                original_image.clone(),
                black_and_white_list_slot(&parameters, index),
            )
            .expect("original Item differential");
            assert_eq!(actual, expected);
            assert_eq!(actual.attributes(), original_image.attributes());
            assert_eq!(
                (
                    actual.element().data().width,
                    actual.element().data().height
                ),
                (
                    original_image.element().data().width,
                    original_image.element().data().height
                )
            );
            for (pixel, source) in actual
                .element()
                .data()
                .data
                .iter()
                .zip(&original_image.element().data().data)
            {
                assert_eq!(
                    pixel.to_gamma_srgb_channels()[3].to_bits(),
                    source.to_gamma_srgb_channels()[3].to_bits()
                );
            }
        }
        assert_eq!(output.attribute::<usize>("image-slot", 3), Some(&1));
        assert_eq!(
            output,
            evaluate_black_and_white_cpu_list(input.clone(), parameters.clone()).expect("repeat")
        );
        assert_eq!((input, parameters), original);
    }

    #[test]
    fn at_gra_node_144_2_black_and_white_gradient_list_retains_ranked_stops_and_empty_ramps() {
        let input: List<_> = [
            gradient_item().with_attribute("ramp-slot", 0usize),
            Item::new_from_element(Gradient::default()).with_attribute("ramp-slot", 1usize),
        ]
        .into_iter()
        .collect();
        let parameters = black_and_white_list_parameters();
        let original = (input.clone(), parameters.clone());
        let output = evaluate_black_and_white_gradient_list(input.clone(), parameters.clone())
            .expect("original mapped Gradient signature");
        assert_eq!(output.len(), 4);
        for index in 0..4 {
            let original_gradient = input.clone_item(index.min(1)).expect("ranked gradient");
            let actual = output.clone_item(index).expect("output gradient");
            assert_black_and_white_gradient(
                &actual,
                &original_gradient,
                &black_and_white_list_slot(&parameters, index),
            );
        }
        assert_eq!(output.attribute::<usize>("ramp-slot", 3), Some(&1));
        assert_eq!(
            output,
            evaluate_black_and_white_gradient_list(input.clone(), parameters.clone())
                .expect("repeat")
        );
        assert_eq!((input, parameters), original);
    }

    #[test]
    fn at_gra_node_144_2_black_and_white_each_empty_ranked_connector_empties_all_mapped_families() {
        for empty_connector in 0..9 {
            let mut parameters = black_and_white_list_parameters();
            match empty_connector {
                1 => parameters.use_tint = List::new(),
                2 => parameters.tint = List::new(),
                3 => parameters.reds = List::new(),
                4 => parameters.yellows = List::new(),
                5 => parameters.greens = List::new(),
                6 => parameters.cyans = List::new(),
                7 => parameters.blues = List::new(),
                8 => parameters.magentas = List::new(),
                0 => {}
                _ => unreachable!("finite connector cases"),
            }
            let colors: List<_> = if empty_connector == 0 {
                List::new()
            } else {
                [Item::new_from_element(Color::from_gamma_srgb_channels(
                    0.2, 0.4, 0.8, 0.5,
                ))]
                .into_iter()
                .collect()
            };
            let rasters: List<_> = if empty_connector == 0 {
                List::new()
            } else {
                [image_item(1, 1, &[[0.2, 0.4, 0.8, 0.5]])]
                    .into_iter()
                    .collect()
            };
            let gradients: List<_> = if empty_connector == 0 {
                List::new()
            } else {
                [gradient_item()].into_iter().collect()
            };
            let original = (
                colors.clone(),
                rasters.clone(),
                gradients.clone(),
                parameters.clone(),
            );
            assert!(
                evaluate_black_and_white_color_list(colors.clone(), parameters.clone())
                    .expect("empty Color connector")
                    .is_empty()
            );
            assert!(
                evaluate_black_and_white_cpu_list(rasters.clone(), parameters.clone())
                    .expect("empty CPU connector")
                    .is_empty()
            );
            assert!(
                evaluate_black_and_white_gradient_list(gradients.clone(), parameters.clone())
                    .expect("empty Gradient connector")
                    .is_empty()
            );
            assert_eq!((colors, rasters, gradients, parameters), original);
        }
    }

    fn brightness_contrast_parameters(
        brightness: f64,
        contrast: f64,
        use_classic: bool,
        classic_pivot: f64,
    ) -> BrightnessContrastParameters {
        BrightnessContrastParameters {
            brightness: Item::new_from_element(brightness)
                .with_attribute("port", String::from("brightness")),
            contrast: Item::new_from_element(contrast)
                .with_attribute("port", String::from("contrast")),
            use_classic: Item::new_from_element(use_classic)
                .with_attribute("port", String::from("use-classic")),
            classic_pivot: Item::new_from_element(classic_pivot)
                .with_attribute("port", String::from("classic-pivot")),
        }
    }

    fn original_brightness_contrast_color(
        input: Item<Color>,
        parameters: BrightnessContrastParameters,
    ) -> Item<Color> {
        let node = graphite_raster_nodes::adjustments::BrightnessContrastNode::new(
            FutureWrapperNode::new(ClonedNode::new(input)),
            FutureWrapperNode::new(ClonedNode::new(parameters.brightness)),
            FutureWrapperNode::new(ClonedNode::new(parameters.contrast)),
            FutureWrapperNode::new(ClonedNode::new(parameters.use_classic)),
            FutureWrapperNode::new(ClonedNode::new(parameters.classic_pivot)),
        );
        futures::executor::block_on(node.eval(()))
    }

    fn assert_brightness_contrast_fixed_oracles(use_classic: bool, cases: &[(f32, f64, f64, f32)]) {
        for &(value, brightness, contrast, expected) in cases {
            let gamma = value / 255.;
            let input =
                Item::new_from_element(Color::from_gamma_srgb_channels(gamma, gamma, gamma, 0.5))
                    .with_attribute("source-value", value);
            let parameters =
                brightness_contrast_parameters(brightness, contrast, use_classic, 127.);
            let color = evaluate_brightness_contrast_color(input.clone(), parameters.clone())
                .expect("original Color constructor");
            let raster = image_item(1, 1, &[[gamma, gamma, gamma, 0.5]])
                .with_attribute("source-value", value);
            let output = evaluate_brightness_contrast_cpu(raster.clone(), parameters)
                .expect("original CPU constructor");
            for actual in [
                color.element(),
                output
                    .element()
                    .data()
                    .data
                    .first()
                    .expect("one original pixel"),
            ] {
                for channel in actual.to_gamma_srgb_channels().into_iter().take(3) {
                    assert!(
                        (channel * 255. - expected).abs() <= 1.,
                        "source oracle {value}/{brightness}/{contrast}, classic={use_classic}: expected {expected}, got {}",
                        channel * 255.
                    );
                }
                assert_eq!(
                    actual.to_gamma_srgb_channels()[3].to_bits(),
                    0.5_f32.to_bits()
                );
            }
            assert_eq!(color.attributes(), input.attributes());
            assert_eq!(output.attributes(), raster.attributes());
        }
    }

    #[test]
    fn at_gra_node_145_2_brightness_contrast_matches_pinned_current_curve_oracles() {
        assert_brightness_contrast_fixed_oracles(
            false,
            &[
                (16., 100., 0., 30.),
                (64., 100., 0., 120.),
                (128., 100., 0., 209.),
                (192., 100., 0., 245.),
                (128., 20., 0., 145.),
                (64., -100., 0., 34.),
                (128., -100., 0., 68.),
                (192., -100., 0., 111.),
                (240., -100., 0., 177.),
                (64., 150., 0., 162.),
                (128., 150., 0., 239.),
                (128., -150., 0., 50.),
                (240., -150., 0., 131.),
                (32., 0., 100., 14.),
                (64., 0., 100., 40.),
                (192., 0., 100., 216.),
                (64., 0., -50., 76.),
                (64., 0., 25., 58.),
                (64., 50., 30., 81.),
                (128., 50., 30., 178.),
                (192., 50., 30., 233.),
                (64., -60., -20., 48.),
                (128., -60., -20., 92.),
                (192., -60., -20., 139.),
            ],
        );
    }

    #[test]
    fn at_gra_node_145_2_brightness_contrast_matches_pinned_classic_oracles() {
        assert_brightness_contrast_fixed_oracles(
            true,
            &[
                (0., 0., -50., 64.),
                (100., 0., -50., 114.),
                (255., 0., -50., 191.),
                (64., 0., 50., 1.),
                (100., 0., 50., 73.),
                (200., 0., 50., 255.),
                (50., 40., 40., 65.),
                (100., 40., 40., 148.),
                (50., -40., -40., 41.),
                (200., -40., -40., 131.),
                (126., 0., 100., 0.),
                (128., 0., 100., 255.),
            ],
        );
    }

    #[test]
    fn at_gra_node_145_2_brightness_contrast_cpu_preserves_alpha_dimensions_attributes_and_inputs()
    {
        let input = image_item(
            2,
            2,
            &[
                [0.25, 0.5, 0.75, 0.],
                [0.1, 0.4, 0.9, -0.],
                [0.8, 0.6, 0.2, 0.5],
                [0., 1., 0.5, 1.],
            ],
        )
        .with_attribute("asset", String::from("brightness-source"))
        .with_attribute("revision", 41_u64)
        .with_attribute("visible", false);
        let original = input.clone();
        for use_classic in [false, true] {
            let parameters = brightness_contrast_parameters(63., -37., use_classic, 81.);
            let original_parameters = parameters.clone();
            let output = evaluate_brightness_contrast_cpu(input.clone(), parameters.clone())
                .expect("original CPU brightness/contrast");
            assert_eq!(output.attributes(), input.attributes());
            assert_eq!(
                (
                    output.element().data().width,
                    output.element().data().height
                ),
                (2, 2)
            );
            assert_eq!(output.element().data().data.len(), 4);
            for (actual, original) in output
                .element()
                .data()
                .data
                .iter()
                .zip(&input.element().data().data)
            {
                let expected = original_brightness_contrast_color(
                    Item::new_from_element(*original),
                    parameters.clone(),
                );
                assert_eq!(actual, expected.element());
                assert_eq!(
                    actual.to_gamma_srgb_channels()[3].to_bits(),
                    original.to_gamma_srgb_channels()[3].to_bits()
                );
            }
            assert_ne!(output.element().data().data, input.element().data().data);
            assert_eq!(
                output,
                evaluate_brightness_contrast_cpu(input.clone(), parameters.clone())
                    .expect("repeat")
            );
            assert_eq!(parameters, original_parameters);
            assert_eq!(input, original);
            for (width, height) in [(0, 0), (0, 3), (4, 0)] {
                let empty = image_item(width, height, &[]).with_attribute("empty", true);
                assert_eq!(
                    evaluate_brightness_contrast_cpu(empty.clone(), parameters.clone())
                        .expect("empty CPU"),
                    empty
                );
            }
        }
    }

    #[test]
    fn at_gra_node_145_2_brightness_contrast_keeps_generated_hard_bounds_and_algorithm_pivot_rules()
    {
        let input = Item::new_from_element(Color::from_gamma_srgb_channels(0.25, 0.5, 0.75, -0.))
            .with_attribute("bounded-source", String::from("unchanged"));
        for (use_classic, brightness, contrast, expected_brightness, expected_contrast) in [
            (false, 400., 20., 150., 20.),
            (false, -400., 20., -150., 20.),
            (true, 140., 20., 100., 20.),
            (true, -140., 20., -100., 20.),
            (false, 20., -90., 20., -50.),
            (true, 20., -400., 20., -100.),
            (false, 20., 400., 20., 100.),
            (true, 20., 400., 20., 100.),
            (false, f64::INFINITY, 20., 150., 20.),
            (true, f64::NEG_INFINITY, 20., -100., 20.),
            (false, f64::NAN, 20., -150., 20.),
            (false, 20., f64::NAN, 20., -50.),
            (true, 20., f64::INFINITY, 20., 100.),
            (true, 20., f64::NEG_INFINITY, 20., -100.),
        ] {
            let actual = evaluate_brightness_contrast_color(
                input.clone(),
                brightness_contrast_parameters(brightness, contrast, use_classic, 127.),
            )
            .expect("original constructor clamping");
            let expected = evaluate_brightness_contrast_color(
                input.clone(),
                brightness_contrast_parameters(
                    expected_brightness,
                    expected_contrast,
                    use_classic,
                    127.,
                ),
            )
            .expect("original bounded values");
            assert_eq!(actual, expected);
            assert_eq!(actual.attributes(), input.attributes());
            assert_eq!(
                actual.element().to_gamma_srgb_channels()[3].to_bits(),
                (-0_f32).to_bits()
            );
        }
        let half = Item::new_from_element(Color::from_gamma_srgb_channels(0.5, 0.5, 0.5, 0.5));
        for (pivot, contrast, expected) in [
            (0., 100., 1.),
            (255., 100., 0.),
            (-255., -50., 0.),
            (510., -50., 1.),
        ] {
            let output = evaluate_brightness_contrast_color(
                half.clone(),
                brightness_contrast_parameters(0., contrast, true, pivot),
            )
            .expect("soft-only Classic pivot");
            assert_gamma_channels(output.element(), [expected, expected, expected, 0.5]);
        }
        let current = evaluate_brightness_contrast_color(
            input.clone(),
            brightness_contrast_parameters(35., 24., false, 127.),
        )
        .expect("current algorithm");
        for pivot in [-255., 0., 255., 510.] {
            assert_eq!(
                evaluate_brightness_contrast_color(
                    input.clone(),
                    brightness_contrast_parameters(35., 24., false, pivot)
                )
                .expect("unused current pivot"),
                current
            );
        }
        assert_gamma_channels(input.element(), [0.25, 0.5, 0.75, -0.]);
    }

    fn assert_brightness_contrast_gradient(
        output: &Item<Gradient>,
        input: &Item<Gradient>,
        parameters: &BrightnessContrastParameters,
    ) {
        use graphite_core_types::list::{ATTR_MIDPOINT, ATTR_POSITION};
        assert_eq!(output.attributes(), input.attributes());
        assert_eq!(output.element().len(), input.element().len());
        for (actual, original) in output
            .element()
            .clone()
            .into_color_list()
            .into_iter()
            .zip(input.element().clone().into_color_list())
        {
            let expected = original_brightness_contrast_color(original.clone(), parameters.clone());
            assert_eq!(actual.element(), expected.element());
            assert_eq!(actual.attributes(), original.attributes());
            assert_eq!(
                actual.attribute::<f64>(ATTR_POSITION),
                original.attribute::<f64>(ATTR_POSITION)
            );
            assert_eq!(
                actual.attribute::<f64>(ATTR_MIDPOINT),
                original.attribute::<f64>(ATTR_MIDPOINT)
            );
            assert_eq!(
                actual.element().to_gamma_srgb_channels()[3].to_bits(),
                original.element().to_gamma_srgb_channels()[3].to_bits()
            );
        }
    }

    #[test]
    fn at_gra_node_145_2_brightness_contrast_gradient_retains_stop_and_ramp_metadata() {
        for input in [
            gradient_item(),
            Item::new_from_element(Gradient::default()).with_attribute("empty-ramp", true),
        ] {
            let original = input.clone();
            for use_classic in [false, true] {
                let parameters = brightness_contrast_parameters(-63., 41., use_classic, 83.);
                let original_parameters = parameters.clone();
                let output =
                    evaluate_brightness_contrast_gradient(input.clone(), parameters.clone())
                        .expect("original Gradient constructor");
                assert_brightness_contrast_gradient(&output, &input, &parameters);
                assert_eq!(
                    output,
                    evaluate_brightness_contrast_gradient(input.clone(), parameters.clone())
                        .expect("repeat Gradient")
                );
                assert_eq!(parameters, original_parameters);
                assert_eq!(input, original);
            }
        }
    }

    fn brightness_contrast_list_parameters() -> BrightnessContrastListParameters {
        BrightnessContrastListParameters {
            brightness: [Item::new_from_element(38.), Item::new_from_element(-61.)]
                .into_iter()
                .collect(),
            contrast: [
                Item::new_from_element(-73.),
                Item::new_from_element(29.),
                Item::new_from_element(84.),
            ]
            .into_iter()
            .collect(),
            use_classic: [Item::new_from_element(false)].into_iter().collect(),
            classic_pivot: [
                Item::new_from_element(127.),
                Item::new_from_element(43.),
                Item::new_from_element(214.),
                Item::new_from_element(19.),
            ]
            .into_iter()
            .collect(),
        }
    }

    fn brightness_contrast_list_slot(
        parameters: &BrightnessContrastListParameters,
        index: usize,
    ) -> BrightnessContrastParameters {
        BrightnessContrastParameters {
            brightness: parameters
                .brightness
                .clone_item(index.min(parameters.brightness.len() - 1))
                .expect("nonempty brightness"),
            contrast: parameters
                .contrast
                .clone_item(index.min(parameters.contrast.len() - 1))
                .expect("nonempty contrast"),
            use_classic: parameters
                .use_classic
                .clone_item(index.min(parameters.use_classic.len() - 1))
                .expect("nonempty use_classic"),
            classic_pivot: parameters
                .classic_pivot
                .clone_item(index.min(parameters.classic_pivot.len() - 1))
                .expect("nonempty classic_pivot"),
        }
    }

    #[test]
    fn at_gra_node_145_2_brightness_contrast_color_list_frames_every_connector_and_repeats_last() {
        for longest in 0..5 {
            let primary =
                Item::new_from_element(Color::from_gamma_srgb_channels(0.25, 0.5, 0.75, -0.))
                    .with_attribute("source-slot", 0_usize);
            let mut input: List<Color> = [primary.clone()].into_iter().collect();
            let single = brightness_contrast_parameters(27., -32., false, 97.);
            let mut parameters = BrightnessContrastListParameters {
                brightness: [single.brightness].into_iter().collect(),
                contrast: [single.contrast].into_iter().collect(),
                use_classic: [single.use_classic].into_iter().collect(),
                classic_pivot: [single.classic_pivot].into_iter().collect(),
            };
            match longest {
                0 => {
                    input = [
                        primary,
                        Item::new_from_element(Color::from_gamma_srgb_channels(0.8, 0.1, 0.3, 0.5))
                            .with_attribute("source-slot", 1_usize),
                        Item::new_from_element(Color::from_gamma_srgb_channels(0.4, 0.9, 0.2, 1.))
                            .with_attribute("source-slot", 2_usize),
                    ]
                    .into_iter()
                    .collect()
                }
                1 => {
                    parameters.brightness = [
                        Item::new_from_element(-210.),
                        Item::new_from_element(37.),
                        Item::new_from_element(230.),
                    ]
                    .into_iter()
                    .collect()
                }
                2 => {
                    parameters.contrast = [
                        Item::new_from_element(-130.),
                        Item::new_from_element(33.),
                        Item::new_from_element(140.),
                    ]
                    .into_iter()
                    .collect()
                }
                3 => {
                    parameters.use_classic = [
                        Item::new_from_element(false),
                        Item::new_from_element(true),
                        Item::new_from_element(false),
                    ]
                    .into_iter()
                    .collect()
                }
                4 => {
                    parameters.use_classic = [Item::new_from_element(true)].into_iter().collect();
                    parameters.classic_pivot = [
                        Item::new_from_element(-255.),
                        Item::new_from_element(127.),
                        Item::new_from_element(510.),
                    ]
                    .into_iter()
                    .collect();
                }
                _ => unreachable!("five original ranked connectors"),
            }
            let original_input = input.clone();
            let original_parameters = parameters.clone();
            let output = evaluate_brightness_contrast_color_list(input.clone(), parameters.clone())
                .expect("original mapped Color");
            assert_eq!(output.len(), 3);
            for (index, actual) in output.clone().into_iter().enumerate() {
                let original = input
                    .clone_item(index.min(input.len() - 1))
                    .expect("nonempty primary");
                let expected = original_brightness_contrast_color(
                    original.clone(),
                    brightness_contrast_list_slot(&parameters, index),
                );
                assert_eq!(
                    actual, expected,
                    "longest connector {longest}, slot {index}"
                );
                assert_eq!(actual.attributes(), original.attributes());
                assert_eq!(
                    actual.element().to_gamma_srgb_channels()[3].to_bits(),
                    original.element().to_gamma_srgb_channels()[3].to_bits()
                );
            }
            assert_eq!(
                output,
                evaluate_brightness_contrast_color_list(input.clone(), parameters.clone())
                    .expect("repeat mapped Color")
            );
            assert_eq!(input, original_input);
            assert_eq!(parameters, original_parameters);
        }
    }

    #[test]
    fn at_gra_node_145_2_brightness_contrast_mapped_cpu_and_gradient_keep_slot_attributes_and_shapes()
     {
        let input: List<Raster<CPU>> = [
            image_item(2, 1, &[[0.25, 0.5, 0.75, -0.], [0.8, 0.4, 0.1, 0.5]])
                .with_attribute("source-slot", 0_usize),
            image_item(1, 2, &[[0.1, 0.2, 0.3, 1.], [0.9, 0.7, 0.5, 0.]])
                .with_attribute("source-slot", 1_usize),
        ]
        .into_iter()
        .collect();
        let gradients: List<Gradient> = [
            gradient_item().with_attribute("source-slot", 0_usize),
            Item::new_from_element(Gradient::default()).with_attribute("source-slot", 1_usize),
        ]
        .into_iter()
        .collect();
        for use_classic in [false, true] {
            let mut parameters = brightness_contrast_list_parameters();
            parameters.use_classic = [Item::new_from_element(use_classic)].into_iter().collect();
            let original = (input.clone(), gradients.clone(), parameters.clone());
            let output = evaluate_brightness_contrast_cpu_list(input.clone(), parameters.clone())
                .expect("original mapped CPU");
            let ramps =
                evaluate_brightness_contrast_gradient_list(gradients.clone(), parameters.clone())
                    .expect("original mapped Gradient");
            assert_eq!(output.len(), 4);
            assert_eq!(ramps.len(), 4);
            for (index, actual) in output.clone().into_iter().enumerate() {
                let original = input
                    .clone_item(index.min(input.len() - 1))
                    .expect("nonempty original raster");
                let expected = evaluate_brightness_contrast_cpu(
                    original.clone(),
                    brightness_contrast_list_slot(&parameters, index),
                )
                .expect("original direct CPU binding");
                assert_eq!(actual, expected);
                assert_eq!(actual.attributes(), original.attributes());
                assert_eq!(
                    (
                        actual.element().data().width,
                        actual.element().data().height
                    ),
                    (
                        original.element().data().width,
                        original.element().data().height
                    )
                );
                for (pixel, source) in actual
                    .element()
                    .data()
                    .data
                    .iter()
                    .zip(&original.element().data().data)
                {
                    assert_eq!(
                        pixel.to_gamma_srgb_channels()[3].to_bits(),
                        source.to_gamma_srgb_channels()[3].to_bits()
                    );
                }
            }
            for (index, actual) in ramps.clone().into_iter().enumerate() {
                let original = gradients
                    .clone_item(index.min(gradients.len() - 1))
                    .expect("nonempty original ramp slot");
                assert_brightness_contrast_gradient(
                    &actual,
                    &original,
                    &brightness_contrast_list_slot(&parameters, index),
                );
            }
            assert_eq!(
                output,
                evaluate_brightness_contrast_cpu_list(input.clone(), parameters.clone())
                    .expect("repeat CPU frame")
            );
            assert_eq!(
                ramps,
                evaluate_brightness_contrast_gradient_list(gradients.clone(), parameters.clone())
                    .expect("repeat Gradient frame")
            );
            assert_eq!((input.clone(), gradients.clone(), parameters), original);
        }
    }

    #[test]
    fn at_gra_node_145_2_brightness_contrast_each_empty_connector_empties_every_mapped_family() {
        for empty_connector in 0..5 {
            let mut colors: List<Color> = [Item::new_from_element(
                Color::from_gamma_srgb_channels(0.25, 0.5, 0.75, -0.),
            )]
            .into_iter()
            .collect();
            let mut rasters: List<Raster<CPU>> = [image_item(1, 1, &[[0.25, 0.5, 0.75, -0.]])]
                .into_iter()
                .collect();
            let mut gradients: List<Gradient> = [gradient_item()].into_iter().collect();
            let mut parameters = brightness_contrast_list_parameters();
            match empty_connector {
                0 => {
                    colors = List::new();
                    rasters = List::new();
                    gradients = List::new();
                }
                1 => parameters.brightness = List::new(),
                2 => parameters.contrast = List::new(),
                3 => parameters.use_classic = List::new(),
                4 => parameters.classic_pivot = List::new(),
                _ => unreachable!("five original ranked connectors"),
            }
            let original = (
                colors.clone(),
                rasters.clone(),
                gradients.clone(),
                parameters.clone(),
            );
            assert!(
                evaluate_brightness_contrast_color_list(colors.clone(), parameters.clone())
                    .expect("empty Color connector")
                    .is_empty()
            );
            assert!(
                evaluate_brightness_contrast_cpu_list(rasters.clone(), parameters.clone())
                    .expect("empty CPU connector")
                    .is_empty()
            );
            assert!(
                evaluate_brightness_contrast_gradient_list(gradients.clone(), parameters.clone())
                    .expect("empty Gradient connector")
                    .is_empty()
            );
            assert_eq!((colors, rasters, gradients, parameters), original);
        }
    }
}
