#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AddMode {
    Scalar,
    Lists,
    Integer,
    Vector,
    ScalarVector,
    VectorScalar,
    IntegerLists,
    VectorLists,
    ScalarVectorLists,
    VectorScalarLists,
    Sort,
    MultiplyScalar,
    MultiplyLists,
    MultiplyInteger,
    MultiplyVector,
    MultiplyScalarVector,
    MultiplyVectorScalar,
    MultiplyIntegerLists,
    MultiplyVectorLists,
    MultiplyScalarVectorLists,
    MultiplyVectorScalarLists,
    MultiplyAffine,
    MultiplyAffineLists,
    InvertColor,
    MixColor,
}

impl AddMode {
    pub(crate) fn is_multiply(self) -> bool {
        matches!(
            self,
            Self::MultiplyScalar
                | Self::MultiplyLists
                | Self::MultiplyInteger
                | Self::MultiplyVector
                | Self::MultiplyScalarVector
                | Self::MultiplyVectorScalar
                | Self::MultiplyIntegerLists
                | Self::MultiplyVectorLists
                | Self::MultiplyScalarVectorLists
                | Self::MultiplyVectorScalarLists
                | Self::MultiplyAffine
                | Self::MultiplyAffineLists
        )
    }

    pub(crate) fn is_affine(self) -> bool {
        matches!(self, Self::MultiplyAffine | Self::MultiplyAffineLists)
    }

    pub(crate) fn is_list(self) -> bool {
        matches!(
            self,
            Self::Lists
                | Self::IntegerLists
                | Self::VectorLists
                | Self::ScalarVectorLists
                | Self::VectorScalarLists
                | Self::MultiplyLists
                | Self::MultiplyIntegerLists
                | Self::MultiplyVectorLists
                | Self::MultiplyScalarVectorLists
                | Self::MultiplyVectorScalarLists
                | Self::MultiplyAffineLists
        )
    }

    pub(crate) fn augend_is_vector(self) -> bool {
        matches!(
            self,
            Self::Vector
                | Self::VectorScalar
                | Self::VectorLists
                | Self::VectorScalarLists
                | Self::MultiplyVector
                | Self::MultiplyVectorScalar
                | Self::MultiplyVectorLists
                | Self::MultiplyVectorScalarLists
        )
    }

    pub(crate) fn addend_is_vector(self) -> bool {
        matches!(
            self,
            Self::Vector
                | Self::ScalarVector
                | Self::VectorLists
                | Self::ScalarVectorLists
                | Self::MultiplyVector
                | Self::MultiplyScalarVector
                | Self::MultiplyVectorLists
                | Self::MultiplyScalarVectorLists
        )
    }
}

enum Inputs {
    Scalar(f64, f64),
    Lists(Vec<f64>, Vec<f64>),
    Integer(i64, i64),
    Vector([f64; 2], [f64; 2]),
    ScalarVector(f64, [f64; 2]),
    VectorScalar([f64; 2], f64),
    IntegerLists(Vec<i64>, Vec<i64>),
    VectorLists(Vec<[f64; 2]>, Vec<[f64; 2]>),
    ScalarVectorLists(Vec<f64>, Vec<[f64; 2]>),
    VectorScalarLists(Vec<[f64; 2]>, Vec<f64>),
    Sort(Vec<f64>, Vec<f64>, bool),
    Multiply(MultiplyInputs),
    InvertColor([f32; 4]),
    MixColor {
        over: [f32; 4],
        under: [f32; 4],
        parameters: graphics_core::resources::CpuMixParameters,
    },
}

enum Output {
    Pending,
    Scalar(f64),
    Lists(Vec<f64>),
    Integer(i64),
    Vector([f64; 2]),
    IntegerLists(Vec<i64>),
    VectorLists(Vec<[f64; 2]>),
    Affine([f64; 6]),
    AffineLists(Vec<[f64; 6]>),
    Color([f32; 4]),
}

impl Output {
    fn text(&self) -> String {
        match self {
            Self::Pending => "Awaiting evaluation".into(),
            Self::Scalar(value) => value.to_string(),
            Self::Lists(value) => format!("{value:?}"),
            Self::Integer(value) => value.to_string(),
            Self::Vector(value) => format!("{value:?}"),
            Self::IntegerLists(value) => format!("{value:?}"),
            Self::VectorLists(value) => format!("{value:?}"),
            Self::Affine(value) => format!("{value:?}"),
            Self::AffineLists(value) => format!("{value:?}"),
            Self::Color(value) => format!("{value:?}"),
        }
    }
}

enum MultiplyInputs {
    Scalar(f64, f64),
    Lists(Vec<f64>, Vec<f64>),
    Integer(i64, i64),
    Vector([f64; 2], [f64; 2]),
    ScalarVector(f64, [f64; 2]),
    VectorScalar([f64; 2], f64),
    IntegerLists(Vec<i64>, Vec<i64>),
    VectorLists(Vec<[f64; 2]>, Vec<[f64; 2]>),
    ScalarVectorLists(Vec<f64>, Vec<[f64; 2]>),
    VectorScalarLists(Vec<[f64; 2]>, Vec<f64>),
    Affine([f64; 6], [f64; 6]),
    AffineLists(Vec<[f64; 6]>, Vec<[f64; 6]>),
}

impl MultiplyInputs {
    fn same_values(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Scalar(multiplier, multiplicand),
                Self::Scalar(other_multiplier, other_multiplicand),
            ) => {
                multiplier.to_bits() == other_multiplier.to_bits()
                    && multiplicand.to_bits() == other_multiplicand.to_bits()
            }
            (
                Self::Lists(multiplier, multiplicand),
                Self::Lists(other_multiplier, other_multiplicand),
            ) => {
                same_values(multiplier, other_multiplier)
                    && same_values(multiplicand, other_multiplicand)
            }
            (
                Self::Integer(multiplier, multiplicand),
                Self::Integer(other_multiplier, other_multiplicand),
            ) => multiplier == other_multiplier && multiplicand == other_multiplicand,
            (
                Self::Vector(multiplier, multiplicand),
                Self::Vector(other_multiplier, other_multiplicand),
            ) => {
                same_values(multiplier, other_multiplier)
                    && same_values(multiplicand, other_multiplicand)
            }
            (
                Self::ScalarVector(multiplier, multiplicand),
                Self::ScalarVector(other_multiplier, other_multiplicand),
            ) => {
                multiplier.to_bits() == other_multiplier.to_bits()
                    && same_values(multiplicand, other_multiplicand)
            }
            (
                Self::VectorScalar(multiplier, multiplicand),
                Self::VectorScalar(other_multiplier, other_multiplicand),
            ) => {
                same_values(multiplier, other_multiplier)
                    && multiplicand.to_bits() == other_multiplicand.to_bits()
            }
            (
                Self::IntegerLists(multiplier, multiplicand),
                Self::IntegerLists(other_multiplier, other_multiplicand),
            ) => multiplier == other_multiplier && multiplicand == other_multiplicand,
            (
                Self::VectorLists(multiplier, multiplicand),
                Self::VectorLists(other_multiplier, other_multiplicand),
            ) => {
                same_vectors(multiplier, other_multiplier)
                    && same_vectors(multiplicand, other_multiplicand)
            }
            (
                Self::ScalarVectorLists(multiplier, multiplicand),
                Self::ScalarVectorLists(other_multiplier, other_multiplicand),
            ) => {
                same_values(multiplier, other_multiplier)
                    && same_vectors(multiplicand, other_multiplicand)
            }
            (
                Self::VectorScalarLists(multiplier, multiplicand),
                Self::VectorScalarLists(other_multiplier, other_multiplicand),
            ) => {
                same_vectors(multiplier, other_multiplier)
                    && same_values(multiplicand, other_multiplicand)
            }
            (
                Self::Affine(multiplier, multiplicand),
                Self::Affine(other_multiplier, other_multiplicand),
            ) => {
                same_values(multiplier, other_multiplier)
                    && same_values(multiplicand, other_multiplicand)
            }
            (
                Self::AffineLists(multiplier, multiplicand),
                Self::AffineLists(other_multiplier, other_multiplicand),
            ) => {
                same_vectors(multiplier, other_multiplier)
                    && same_vectors(multiplicand, other_multiplicand)
            }
            _ => false,
        }
    }

    fn evaluate(&self) -> Result<Output, graphics_core::AddEvaluationError> {
        Ok(match self {
            Self::Scalar(multiplier, multiplicand) => Output::Scalar(
                graphics_core::evaluate_multiply(*multiplier, *multiplicand)?,
            ),
            Self::Lists(multiplier, multiplicand) => Output::Lists(
                graphics_core::evaluate_multiply_lists(multiplier, multiplicand)?,
            ),
            Self::Integer(multiplier, multiplicand) => Output::Integer(
                graphics_core::evaluate_multiply_i64(*multiplier, *multiplicand)?,
            ),
            Self::Vector(multiplier, multiplicand) => Output::Vector(
                graphics_core::evaluate_multiply_vector(*multiplier, *multiplicand)?,
            ),
            Self::ScalarVector(multiplier, multiplicand) => Output::Vector(
                graphics_core::evaluate_multiply_scalar_vector(*multiplier, *multiplicand)?,
            ),
            Self::VectorScalar(multiplier, multiplicand) => Output::Vector(
                graphics_core::evaluate_multiply_vector_scalar(*multiplier, *multiplicand)?,
            ),
            Self::IntegerLists(multiplier, multiplicand) => Output::IntegerLists(
                graphics_core::evaluate_multiply_i64_lists(multiplier, multiplicand)?,
            ),
            Self::VectorLists(multiplier, multiplicand) => Output::VectorLists(
                graphics_core::evaluate_multiply_vector_lists(multiplier, multiplicand)?,
            ),
            Self::ScalarVectorLists(multiplier, multiplicand) => Output::VectorLists(
                graphics_core::evaluate_multiply_scalar_vector_lists(multiplier, multiplicand)?,
            ),
            Self::VectorScalarLists(multiplier, multiplicand) => Output::VectorLists(
                graphics_core::evaluate_multiply_vector_scalar_lists(multiplier, multiplicand)?,
            ),
            Self::Affine(multiplier, multiplicand) => Output::Affine(
                graphics_core::evaluate_multiply_affine(*multiplier, *multiplicand)?,
            ),
            Self::AffineLists(multiplier, multiplicand) => Output::AffineLists(
                graphics_core::evaluate_multiply_affine_lists(multiplier, multiplicand)?,
            ),
        })
    }

    fn expression(&self) -> String {
        match self {
            Self::Scalar(multiplier, multiplicand) => {
                format!("Multiply {multiplier} by {multiplicand}")
            }
            Self::Lists(multiplier, multiplicand) => {
                format!("Multiply {multiplier:?} by {multiplicand:?}")
            }
            Self::Integer(multiplier, multiplicand) => {
                format!("Multiply {multiplier} by {multiplicand}")
            }
            Self::Vector(multiplier, multiplicand) => {
                format!("Multiply {multiplier:?} by {multiplicand:?}")
            }
            Self::ScalarVector(multiplier, multiplicand) => {
                format!("Multiply {multiplier} by {multiplicand:?}")
            }
            Self::VectorScalar(multiplier, multiplicand) => {
                format!("Multiply {multiplier:?} by {multiplicand}")
            }
            Self::IntegerLists(multiplier, multiplicand) => {
                format!("Multiply {multiplier:?} by {multiplicand:?}")
            }
            Self::VectorLists(multiplier, multiplicand) => {
                format!("Multiply {multiplier:?} by {multiplicand:?}")
            }
            Self::ScalarVectorLists(multiplier, multiplicand) => {
                format!("Multiply {multiplier:?} by {multiplicand:?}")
            }
            Self::VectorScalarLists(multiplier, multiplicand) => {
                format!("Multiply {multiplier:?} by {multiplicand:?}")
            }
            Self::Affine(multiplier, multiplicand) => {
                format!("Multiply {multiplier:?} by {multiplicand:?}")
            }
            Self::AffineLists(multiplier, multiplicand) => {
                format!("Multiply {multiplier:?} by {multiplicand:?}")
            }
        }
    }
}

enum EvaluationError {
    Arithmetic(graphics_core::AddEvaluationError),
    Raster(graphics_core::raster::RasterEvaluationError),
}

impl std::fmt::Display for EvaluationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Arithmetic(error) => error.fmt(formatter),
            Self::Raster(error) => error.fmt(formatter),
        }
    }
}

impl From<graphics_core::AddEvaluationError> for EvaluationError {
    fn from(error: graphics_core::AddEvaluationError) -> Self {
        Self::Arithmetic(error)
    }
}

impl From<graphics_core::raster::RasterEvaluationError> for EvaluationError {
    fn from(error: graphics_core::raster::RasterEvaluationError) -> Self {
        Self::Raster(error)
    }
}

struct Evaluation {
    inputs: Inputs,
    output: Output,
}

impl Inputs {
    fn same_values(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::MixColor {
                    over,
                    under,
                    parameters,
                },
                Self::MixColor {
                    over: other_over,
                    under: other_under,
                    parameters: other_parameters,
                },
            ) => {
                over.iter()
                    .zip(other_over)
                    .all(|(first, second)| first.to_bits() == second.to_bits())
                    && under
                        .iter()
                        .zip(other_under)
                        .all(|(first, second)| first.to_bits() == second.to_bits())
                    && parameters == other_parameters
            }
            (Self::InvertColor(first), Self::InvertColor(second)) => first
                .iter()
                .zip(second)
                .all(|(first, second)| first.to_bits() == second.to_bits()),
            (Self::Scalar(augend, addend), Self::Scalar(other_augend, other_addend)) => {
                augend.to_bits() == other_augend.to_bits()
                    && addend.to_bits() == other_addend.to_bits()
            }
            (Self::Lists(augend, addend), Self::Lists(other_augend, other_addend)) => {
                same_values(augend, other_augend) && same_values(addend, other_addend)
            }
            (Self::Integer(augend, addend), Self::Integer(other_augend, other_addend)) => {
                augend == other_augend && addend == other_addend
            }
            (Self::Vector(augend, addend), Self::Vector(other_augend, other_addend)) => {
                same_values(augend, other_augend) && same_values(addend, other_addend)
            }
            (
                Self::ScalarVector(augend, addend),
                Self::ScalarVector(other_augend, other_addend),
            ) => augend.to_bits() == other_augend.to_bits() && same_values(addend, other_addend),
            (
                Self::VectorScalar(augend, addend),
                Self::VectorScalar(other_augend, other_addend),
            ) => same_values(augend, other_augend) && addend.to_bits() == other_addend.to_bits(),
            (
                Self::IntegerLists(augend, addend),
                Self::IntegerLists(other_augend, other_addend),
            ) => augend == other_augend && addend == other_addend,
            (Self::VectorLists(augend, addend), Self::VectorLists(other_augend, other_addend)) => {
                same_vectors(augend, other_augend) && same_vectors(addend, other_addend)
            }
            (
                Self::ScalarVectorLists(augend, addend),
                Self::ScalarVectorLists(other_augend, other_addend),
            ) => same_values(augend, other_augend) && same_vectors(addend, other_addend),
            (
                Self::VectorScalarLists(augend, addend),
                Self::VectorScalarLists(other_augend, other_addend),
            ) => same_vectors(augend, other_augend) && same_values(addend, other_addend),
            (
                Self::Sort(values, keys, reverse),
                Self::Sort(other_values, other_keys, other_reverse),
            ) => {
                same_values(values, other_values)
                    && same_values(keys, other_keys)
                    && reverse == other_reverse
            }
            (Self::Multiply(inputs), Self::Multiply(other_inputs)) => {
                inputs.same_values(other_inputs)
            }
            _ => false,
        }
    }

    fn evaluate(self) -> Result<Evaluation, EvaluationError> {
        let output = match &self {
            Self::MixColor {
                over,
                under,
                parameters,
            } => Output::Color(
                graphics_core::raster::evaluate_mix_color_gamma_srgb_channels(
                    *over,
                    *under,
                    parameters.blend_mode,
                    parameters.opacity,
                )?,
            ),
            Self::InvertColor(channels) => Output::Color(
                graphics_core::raster::evaluate_invert_color_channels(*channels)?,
            ),
            Self::Multiply(inputs) => inputs.evaluate()?,
            Self::Scalar(augend, addend) => {
                Output::Scalar(graphics_core::evaluate_add(*augend, *addend))
            }
            Self::Lists(augend, addend) => {
                Output::Lists(graphics_core::evaluate_add_lists(augend, addend))
            }
            Self::Integer(augend, addend) => {
                Output::Integer(graphics_core::evaluate_add_i64(*augend, *addend)?)
            }
            Self::Vector(augend, addend) => {
                Output::Vector(graphics_core::evaluate_add_vector(*augend, *addend)?)
            }
            Self::ScalarVector(augend, addend) => {
                Output::Vector(graphics_core::evaluate_add_scalar_vector(*augend, *addend)?)
            }
            Self::VectorScalar(augend, addend) => {
                Output::Vector(graphics_core::evaluate_add_vector_scalar(*augend, *addend)?)
            }
            Self::IntegerLists(augend, addend) => {
                Output::IntegerLists(graphics_core::evaluate_add_i64_lists(augend, addend)?)
            }
            Self::VectorLists(augend, addend) => {
                Output::VectorLists(graphics_core::evaluate_add_vector_lists(augend, addend)?)
            }
            Self::ScalarVectorLists(augend, addend) => Output::VectorLists(
                graphics_core::evaluate_add_scalar_vector_lists(augend, addend)?,
            ),
            Self::VectorScalarLists(augend, addend) => Output::VectorLists(
                graphics_core::evaluate_add_vector_scalar_lists(augend, addend)?,
            ),
            Self::Sort(values, keys, reverse) => Output::Lists(
                graphics_core::evaluate_sort_numbers(values, keys, *reverse)?,
            ),
        };
        Ok(Evaluation {
            inputs: self,
            output,
        })
    }

    fn expression(&self) -> String {
        match self {
            Self::MixColor {
                over,
                under,
                parameters,
            } => format!(
                "Mix Over RGBA {over:?} with Under RGBA {under:?} ({}, opacity: {}%)",
                parameters.blend_mode, parameters.opacity,
            ),
            Self::InvertColor(channels) => format!("Invert RGBA {channels:?}"),
            Self::Multiply(inputs) => inputs.expression(),
            Self::Scalar(augend, addend) => format!("{augend} + {addend}"),
            Self::Lists(augend, addend) => format!("{augend:?} + {addend:?}"),
            Self::Integer(augend, addend) => format!("{augend} + {addend}"),
            Self::Vector(augend, addend) => format!("{augend:?} + {addend:?}"),
            Self::ScalarVector(augend, addend) => format!("{augend} + {addend:?}"),
            Self::VectorScalar(augend, addend) => format!("{augend:?} + {addend}"),
            Self::IntegerLists(augend, addend) => format!("{augend:?} + {addend:?}"),
            Self::VectorLists(augend, addend) => format!("{augend:?} + {addend:?}"),
            Self::ScalarVectorLists(augend, addend) => format!("{augend:?} + {addend:?}"),
            Self::VectorScalarLists(augend, addend) => format!("{augend:?} + {addend:?}"),
            Self::Sort(values, keys, reverse) => {
                format!("Sort {values:?} by {keys:?} (reverse: {reverse})")
            }
        }
    }
}

#[cfg(test)]
type ListValues<'session> = (&'session [f64], &'session [f64], &'session [f64]);

#[cfg(test)]
type SortValues<'session> = (&'session [f64], &'session [f64], bool, &'session [f64]);

#[cfg(test)]
type MixColorValues = (
    [f32; 4],
    [f32; 4],
    graphics_core::resources::CpuMixParameters,
    [f32; 4],
);

pub struct NumericEvaluationRequest {
    identity: std::sync::Arc<()>,
    snapshot: graphics_core::document::NumericAddSnapshot,
}

pub struct NumericEvaluationCompletion {
    identity: std::sync::Arc<()>,
    snapshot: graphics_core::document::NumericAddSnapshot,
    result: Result<f64, graphics_core::document::NumericNetworkEvaluationError>,
}

impl NumericEvaluationRequest {
    pub async fn evaluate(self) -> NumericEvaluationCompletion {
        let result = self.snapshot.evaluate_network().await;
        NumericEvaluationCompletion {
            identity: self.identity,
            snapshot: self.snapshot,
            result,
        }
    }
}

struct NativeNumericOwner {
    document: graphics_core::document::NumericAddDocument,
    pending: Option<std::sync::Arc<()>>,
    accepted_revision: Option<u64>,
    task: Option<gpui::Task<()>>,
}

pub struct GraphicsSession {
    native: Option<NativeNumericOwner>,
    mode: AddMode,
    evaluation: Evaluation,
    revision: u64,
    diagnostic: Option<String>,
    reverse: bool,
    mix_color_parameters: graphics_core::resources::CpuMixParameters,
}

fn parse_list<T: std::str::FromStr>(text: &str) -> Result<Vec<T>, ()> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    text.split(',')
        .map(|token| token.trim().parse().map_err(|_| ()))
        .collect()
}

pub(crate) fn parse_color(text: &str, name: &str) -> Result<[f32; 4], String> {
    let message = || format!("Enter exactly four comma-separated 32-bit RGBA numbers for {name}.");
    let values = parse_list::<f32>(text).map_err(|_| message())?;
    values.try_into().map_err(|_| message())
}

fn parse_number(text: &str, name: &str) -> Result<f64, String> {
    text.trim()
        .parse()
        .map_err(|_| format!("Enter a number for {name}."))
}

fn parse_integer(text: &str, name: &str) -> Result<i64, String> {
    text.trim()
        .parse()
        .map_err(|_| format!("Enter a whole number in the 64-bit integer range for {name}."))
}

fn parse_vector(x: &str, y: &str, name: &str) -> Result<[f64; 2], String> {
    Ok([
        parse_number(x, &format!("{name}.x"))?,
        parse_number(y, &format!("{name}.y"))?,
    ])
}

fn parse_number_list(text: &str, name: &str) -> Result<Vec<f64>, String> {
    parse_list(text)
        .map_err(|_| format!("Enter comma-separated numbers for {name}, or leave it blank."))
}

fn parse_integer_list(text: &str, name: &str) -> Result<Vec<i64>, String> {
    parse_list(text).map_err(|_| format!("Enter comma-separated whole numbers in the 64-bit integer range for {name}, or leave it blank."))
}

fn parse_vector_list(x: &str, y: &str, name: &str) -> Result<Vec<[f64; 2]>, String> {
    let x = parse_number_list(x, &format!("{name}.x"))?;
    let y = parse_number_list(y, &format!("{name}.y"))?;
    if x.len() != y.len() {
        return Err(format!(
            "Enter the same number of x and y components for {name}."
        ));
    }
    Ok(x.into_iter().zip(y).map(|(x, y)| [x, y]).collect())
}

fn parse_affine(text: &str, name: &str) -> Result<[f64; 6], String> {
    let values = parse_number_list(text, name)?;
    values.try_into().map_err(|_| {
        format!(
            "Enter exactly six comma-separated components for {name}: m11, m12, m21, m22, tx, ty."
        )
    })
}

fn parse_affine_list(text: &str, name: &str) -> Result<Vec<[f64; 6]>, String> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    text.split(';')
        .map(|tuple| parse_affine(tuple, name))
        .collect()
}

fn same_vectors<const COMPONENTS: usize>(
    first: &[[f64; COMPONENTS]],
    second: &[[f64; COMPONENTS]],
) -> bool {
    first.len() == second.len()
        && first
            .iter()
            .zip(second)
            .all(|(first, second)| same_values(first, second))
}

fn same_values(first: &[f64], second: &[f64]) -> bool {
    first.len() == second.len()
        && first
            .iter()
            .zip(second)
            .all(|(first, second)| first.to_bits() == second.to_bits())
}

impl GraphicsSession {
    pub fn new(augend: f64, addend: f64) -> Self {
        Self {
            native: None,
            mode: AddMode::Scalar,
            evaluation: Evaluation {
                inputs: Inputs::Scalar(augend, addend),
                output: Output::Scalar(graphics_core::evaluate_add(augend, addend)),
            },
            revision: 0,
            diagnostic: None,
            reverse: false,
            mix_color_parameters: graphics_core::resources::CpuMixParameters {
                blend_mode: graphics_core::resources::CpuBlendMode::Normal,
                opacity: 100.,
            },
        }
    }

    pub fn from_numeric_document(
        document: graphics_core::document::NumericAddDocument,
    ) -> Result<Self, graphics_core::document::NumericDocumentError> {
        let (augend, addend) = document.inputs()?;
        let revision = document.revision();
        Ok(Self {
            native: Some(NativeNumericOwner {
                document,
                pending: None,
                accepted_revision: None,
                task: None,
            }),
            mode: AddMode::Scalar,
            evaluation: Evaluation {
                inputs: Inputs::Scalar(augend, addend),
                output: Output::Pending,
            },
            revision,
            diagnostic: None,
            reverse: false,
            mix_color_parameters: graphics_core::resources::CpuMixParameters {
                blend_mode: graphics_core::resources::CpuBlendMode::Normal,
                opacity: 100.,
            },
        })
    }

    pub fn numeric_document(&self) -> Option<&graphics_core::document::NumericAddDocument> {
        self.native.as_ref().map(|owner| &owner.document)
    }

    pub fn numeric_pending(&self) -> bool {
        self.native
            .as_ref()
            .is_some_and(|owner| owner.pending.is_some())
    }

    pub fn current_numeric_result(&self) -> Option<f64> {
        let owner = self.native.as_ref()?;
        if owner.accepted_revision != Some(owner.document.revision())
            || owner.pending.is_some()
            || self.diagnostic.is_some()
        {
            return None;
        }
        match self.evaluation.output {
            Output::Scalar(value) => Some(value),
            _ => None,
        }
    }

    pub fn begin_numeric_evaluation(
        &mut self,
        cx: &mut gpui::Context<Self>,
    ) -> Result<NumericEvaluationRequest, graphics_core::document::NumericDocumentError> {
        let owner = self
            .native
            .as_mut()
            .ok_or(graphics_core::document::NumericDocumentError::InvalidNodeContract)?;
        owner.document.inputs()?;
        owner.task = None;
        let identity = std::sync::Arc::new(());
        owner.pending = Some(identity.clone());
        self.diagnostic = None;
        cx.notify();
        Ok(NumericEvaluationRequest {
            identity,
            snapshot: owner.document.snapshot(),
        })
    }

    pub fn accept_numeric_evaluation(
        &mut self,
        completion: NumericEvaluationCompletion,
        cx: &mut gpui::Context<Self>,
    ) -> bool {
        let Some(owner) = self.native.as_mut() else {
            return false;
        };
        if !owner
            .pending
            .as_ref()
            .is_some_and(|pending| std::sync::Arc::ptr_eq(pending, &completion.identity))
            || completion.snapshot.revision() != owner.document.revision()
        {
            return false;
        }
        owner.pending = None;
        owner.task = None;
        match completion.result {
            Ok(value) => match completion.snapshot.inputs() {
                Ok((augend, addend)) => {
                    self.evaluation = Evaluation {
                        inputs: Inputs::Scalar(augend, addend),
                        output: Output::Scalar(value),
                    };
                    owner.accepted_revision = Some(completion.snapshot.revision());
                    self.diagnostic = None;
                }
                Err(error) => self.diagnostic = Some(error.to_string()),
            },
            Err(error) => self.diagnostic = Some(error.to_string()),
        }
        cx.notify();
        true
    }

    pub fn request_numeric_evaluation(
        &mut self,
        cx: &mut gpui::Context<Self>,
    ) -> Result<(), graphics_core::document::NumericDocumentError> {
        let request = self.begin_numeric_evaluation(cx)?;
        // Original execution errors and futures are local; the task retains only a snapshot and weak owner.
        let task = cx.spawn(async move |session, cx| {
            let completion = request.evaluate().await;
            if let Err(error) = session.update(cx, |session, cx| {
                session.accept_numeric_evaluation(completion, cx);
            }) {
                log::debug!("Numeric Graphite owner released before completion: {error}");
            }
        });
        if let Some(owner) = self.native.as_mut() {
            owner.task = Some(task);
        }
        Ok(())
    }

    pub fn cancel_numeric_evaluation(&mut self, cx: &mut gpui::Context<Self>) {
        self.invalidate_numeric_result("Evaluation cancelled.".into());
        cx.notify();
    }

    fn invalidate_numeric_result(&mut self, diagnostic: String) {
        if let Some(owner) = self.native.as_mut() {
            owner.pending = None;
            owner.task = None;
        }
        self.diagnostic = Some(diagnostic);
    }

    pub fn apply_numeric_command(
        &mut self,
        command: graphics_core::document::NumericAddCommand,
        cx: &mut gpui::Context<Self>,
    ) -> Result<bool, graphics_core::document::NumericDocumentError> {
        let owner = self
            .native
            .as_mut()
            .ok_or(graphics_core::document::NumericDocumentError::InvalidNodeContract)?;
        owner.document.validate_command(&command)?;
        let (augend, addend) = owner.document.inputs()?;
        let old_value = match command.input {
            graphics_core::document::NumericAddInput::Augend => augend,
            graphics_core::document::NumericAddInput::Addend => addend,
        };
        let changed = old_value.to_bits() != command.value.to_bits();
        if changed {
            self.revision = owner.document.apply(command)?;
        }
        if changed
            || self.diagnostic.is_some()
            || owner.accepted_revision != Some(owner.document.revision())
        {
            self.request_numeric_evaluation(cx)?;
        }
        Ok(changed)
    }

    pub fn submit_numeric_input(
        &mut self,
        expected_revision: u64,
        input: graphics_core::document::NumericAddInput,
        expression: &str,
        cx: &mut gpui::Context<Self>,
    ) -> Result<bool, String> {
        let result = (|| {
            let value = graphics_core::numeric_input::evaluate_number_input(
                expression,
                graphics_core::numeric_input::NumberInputConstraints::default(),
            )
            .map_err(|error| error.to_string())?;
            let document = self
                .numeric_document()
                .ok_or_else(|| "No native numeric document is attached.".to_owned())?;
            self.apply_numeric_command(
                graphics_core::document::NumericAddCommand {
                    document_id_encoding: document.document_id_encoding(),
                    network_path: Vec::new(),
                    target_node_id: document.node_id(),
                    expected_revision,
                    input,
                    value,
                },
                cx,
            )
            .map_err(|error| error.to_string())
        })();
        if let Err(error) = &result {
            self.invalidate_numeric_result(error.clone());
            cx.notify();
        }
        result
    }

    pub(crate) fn mode(&self) -> AddMode {
        self.mode
    }

    pub(crate) fn mix_color_parameters(&self) -> graphics_core::resources::CpuMixParameters {
        self.mix_color_parameters
    }

    pub(crate) fn set_mix_color_parameters(
        &mut self,
        parameters: graphics_core::resources::CpuMixParameters,
        over: &str,
        under: &str,
    ) {
        self.mix_color_parameters = parameters;
        if self.mode == AddMode::MixColor {
            self.update_components(over, "", under, "");
        }
    }

    pub(crate) fn reverse(&self) -> bool {
        self.reverse
    }

    pub(crate) fn set_reverse(&mut self, reverse: bool, values: &str, keys: &str) {
        self.reverse = reverse;
        if self.mode == AddMode::Sort {
            self.update_components(values, "", keys, "");
        }
    }

    pub(crate) fn set_mode_components(
        &mut self,
        mode: AddMode,
        augend: &str,
        augend_y: &str,
        addend: &str,
        addend_y: &str,
    ) {
        if self.native.is_some() {
            if mode != AddMode::Scalar {
                self.invalidate_numeric_result("This document admits only scalar Add.".into());
            }
            return;
        }
        self.mode = mode;
        self.update_components(augend, augend_y, addend, addend_y);
    }

    #[cfg(test)]
    pub(crate) fn set_mode(&mut self, mode: AddMode, augend: &str, addend: &str) {
        self.set_mode_components(mode, augend, "", addend, "");
    }

    #[cfg(test)]
    pub fn update_inputs(&mut self, augend: &str, addend: &str) {
        self.update_components(augend, "", addend, "");
    }

    fn parse_inputs(
        &self,
        augend: &str,
        augend_y: &str,
        addend: &str,
        addend_y: &str,
    ) -> Result<Inputs, String> {
        Ok(match self.mode {
            AddMode::MixColor => Inputs::MixColor {
                over: parse_color(augend, "Over")?,
                under: parse_color(addend, "Under")?,
                parameters: self.mix_color_parameters,
            },
            AddMode::InvertColor => {
                let mut channels = [0.0; 4];
                for (channel, (text, name)) in channels.iter_mut().zip([
                    (augend, "red"),
                    (augend_y, "green"),
                    (addend, "blue"),
                    (addend_y, "alpha"),
                ]) {
                    *channel = text
                        .trim()
                        .parse::<f32>()
                        .map_err(|_| format!("Enter a 32-bit number for {name}."))?;
                }
                Inputs::InvertColor(channels)
            }
            AddMode::MultiplyScalar => Inputs::Multiply(MultiplyInputs::Scalar(
                parse_number(augend, "multiplier")?,
                parse_number(addend, "multiplicand")?,
            )),
            AddMode::MultiplyLists => Inputs::Multiply(MultiplyInputs::Lists(
                parse_number_list(augend, "multiplier")?,
                parse_number_list(addend, "multiplicand")?,
            )),
            AddMode::MultiplyInteger => Inputs::Multiply(MultiplyInputs::Integer(
                parse_integer(augend, "multiplier")?,
                parse_integer(addend, "multiplicand")?,
            )),
            AddMode::MultiplyVector => Inputs::Multiply(MultiplyInputs::Vector(
                parse_vector(augend, augend_y, "multiplier")?,
                parse_vector(addend, addend_y, "multiplicand")?,
            )),
            AddMode::MultiplyScalarVector => Inputs::Multiply(MultiplyInputs::ScalarVector(
                parse_number(augend, "multiplier")?,
                parse_vector(addend, addend_y, "multiplicand")?,
            )),
            AddMode::MultiplyVectorScalar => Inputs::Multiply(MultiplyInputs::VectorScalar(
                parse_vector(augend, augend_y, "multiplier")?,
                parse_number(addend, "multiplicand")?,
            )),
            AddMode::MultiplyIntegerLists => Inputs::Multiply(MultiplyInputs::IntegerLists(
                parse_integer_list(augend, "multiplier")?,
                parse_integer_list(addend, "multiplicand")?,
            )),
            AddMode::MultiplyVectorLists => Inputs::Multiply(MultiplyInputs::VectorLists(
                parse_vector_list(augend, augend_y, "multiplier")?,
                parse_vector_list(addend, addend_y, "multiplicand")?,
            )),
            AddMode::MultiplyScalarVectorLists => {
                Inputs::Multiply(MultiplyInputs::ScalarVectorLists(
                    parse_number_list(augend, "multiplier")?,
                    parse_vector_list(addend, addend_y, "multiplicand")?,
                ))
            }
            AddMode::MultiplyVectorScalarLists => {
                Inputs::Multiply(MultiplyInputs::VectorScalarLists(
                    parse_vector_list(augend, augend_y, "multiplier")?,
                    parse_number_list(addend, "multiplicand")?,
                ))
            }
            AddMode::MultiplyAffine => Inputs::Multiply(MultiplyInputs::Affine(
                parse_affine(augend, "multiplier")?,
                parse_affine(addend, "multiplicand")?,
            )),
            AddMode::MultiplyAffineLists => Inputs::Multiply(MultiplyInputs::AffineLists(
                parse_affine_list(augend, "multiplier")?,
                parse_affine_list(addend, "multiplicand")?,
            )),

            AddMode::Scalar => Inputs::Scalar(
                parse_number(augend, "augend")?,
                parse_number(addend, "addend")?,
            ),
            AddMode::Lists => Inputs::Lists(
                parse_number_list(augend, "augend")?,
                parse_number_list(addend, "addend")?,
            ),
            AddMode::Integer => Inputs::Integer(
                parse_integer(augend, "augend")?,
                parse_integer(addend, "addend")?,
            ),
            AddMode::Vector => Inputs::Vector(
                parse_vector(augend, augend_y, "augend")?,
                parse_vector(addend, addend_y, "addend")?,
            ),
            AddMode::ScalarVector => Inputs::ScalarVector(
                parse_number(augend, "augend")?,
                parse_vector(addend, addend_y, "addend")?,
            ),
            AddMode::VectorScalar => Inputs::VectorScalar(
                parse_vector(augend, augend_y, "augend")?,
                parse_number(addend, "addend")?,
            ),
            AddMode::IntegerLists => Inputs::IntegerLists(
                parse_integer_list(augend, "augend")?,
                parse_integer_list(addend, "addend")?,
            ),
            AddMode::VectorLists => Inputs::VectorLists(
                parse_vector_list(augend, augend_y, "augend")?,
                parse_vector_list(addend, addend_y, "addend")?,
            ),
            AddMode::ScalarVectorLists => Inputs::ScalarVectorLists(
                parse_number_list(augend, "augend")?,
                parse_vector_list(addend, addend_y, "addend")?,
            ),
            AddMode::VectorScalarLists => Inputs::VectorScalarLists(
                parse_vector_list(augend, augend_y, "augend")?,
                parse_number_list(addend, "addend")?,
            ),
            AddMode::Sort => Inputs::Sort(
                parse_number_list(augend, "values")?,
                parse_number_list(addend, "sort keys")?,
                self.reverse,
            ),
        })
    }

    pub(crate) fn update_components(
        &mut self,
        augend: &str,
        augend_y: &str,
        addend: &str,
        addend_y: &str,
    ) {
        if self.native.is_some() {
            self.invalidate_numeric_result("Edit one addressed native input at a time.".into());
            return;
        }
        let inputs = match self.parse_inputs(augend, augend_y, addend, addend_y) {
            Ok(inputs) => inputs,
            Err(message) => {
                self.diagnostic = Some(message);
                return;
            }
        };
        if inputs.same_values(&self.evaluation.inputs) {
            self.diagnostic = None;
            return;
        }
        let Some(revision) = self.revision.checked_add(1) else {
            self.diagnostic = Some("The evaluation revision limit was reached.".into());
            return;
        };
        match inputs.evaluate() {
            Ok(evaluation) => {
                self.evaluation = evaluation;
                self.revision = revision;
                self.diagnostic = None;
            }
            Err(error) => self.diagnostic = Some(error.to_string()),
        }
    }

    pub(crate) fn result_text(&self) -> String {
        self.evaluation.output.text()
    }

    pub(crate) fn expression_text(&self) -> String {
        if self.native.is_some() && self.current_numeric_result().is_none() {
            return match self.evaluation.output {
                Output::Pending => {
                    if self.numeric_pending() {
                        "Evaluating…".into()
                    } else {
                        "Awaiting evaluation".into()
                    }
                }
                _ => format!("Last valid result: {}", self.result_text()),
            };
        }
        if (self.mode == AddMode::MixColor)
            != matches!(self.evaluation.inputs, Inputs::MixColor { .. })
            || (self.mode == AddMode::InvertColor)
                != matches!(self.evaluation.inputs, Inputs::InvertColor(..))
            || (self.mode == AddMode::Sort) != matches!(self.evaluation.inputs, Inputs::Sort(..))
            || self.mode.is_multiply() != matches!(self.evaluation.inputs, Inputs::Multiply(..))
        {
            return format!("Last valid result: {}", self.result_text());
        }
        format!(
            "{} = {}",
            self.evaluation.inputs.expression(),
            self.result_text()
        )
    }

    #[cfg(test)]
    pub(crate) fn mix_color_values(&self) -> Option<MixColorValues> {
        match (&self.evaluation.inputs, &self.evaluation.output) {
            (
                Inputs::MixColor {
                    over,
                    under,
                    parameters,
                },
                Output::Color(output),
            ) => Some((*over, *under, *parameters, *output)),
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn color_values(&self) -> Option<([f32; 4], [f32; 4])> {
        match (&self.evaluation.inputs, &self.evaluation.output) {
            (Inputs::InvertColor(input), Output::Color(output)) => Some((*input, *output)),
            _ => None,
        }
    }

    #[cfg(test)]
    pub fn inputs(&self) -> (f64, f64) {
        let Inputs::Scalar(augend, addend) = &self.evaluation.inputs else {
            panic!("scalar test inputs requested for another mode")
        };
        (*augend, *addend)
    }

    #[cfg(test)]
    pub fn result(&self) -> f64 {
        let Output::Scalar(result) = &self.evaluation.output else {
            panic!("scalar test result requested for another mode")
        };
        *result
    }

    #[cfg(test)]
    pub(crate) fn list_values(&self) -> Option<ListValues<'_>> {
        match (&self.evaluation.inputs, &self.evaluation.output) {
            (Inputs::Lists(augend, addend), Output::Lists(result)) => {
                Some((augend, addend, result))
            }
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn integer_values(&self) -> Option<(i64, i64, i64)> {
        match (&self.evaluation.inputs, &self.evaluation.output) {
            (Inputs::Integer(augend, addend), Output::Integer(result)) => {
                Some((*augend, *addend, *result))
            }
            _ => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn sort_values(&self) -> Option<SortValues<'_>> {
        match (&self.evaluation.inputs, &self.evaluation.output) {
            (Inputs::Sort(values, keys, reverse), Output::Lists(result)) => {
                Some((values, keys, *reverse, result))
            }
            _ => None,
        }
    }

    #[cfg(test)]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn diagnostic(&self) -> Option<&str> {
        self.diagnostic.as_deref()
    }
}

use gpui::AppContext as _;

pub struct GraphitePngLoad {
    pub assets: sim_runtime::assets::SharedAssetService,
    pub asset_identity: sim_runtime::assets::AssetIdentity,
    pub authorization: sim_runtime::AuthorizedCapabilities,
    pub identity: graphics_core::resources::ImageResourceIdentity,
    pub descriptor: graphics_core::resources::CpuImageDescriptor,
    pub encoded_budget: graphics_core::resources::EncodedPngBudget,
}

pub struct GraphiteEncodedPngLoad {
    pub assets: sim_runtime::assets::SharedAssetService,
    pub asset_identity: sim_runtime::assets::AssetIdentity,
    pub authorization: sim_runtime::AuthorizedCapabilities,
    pub identity: graphics_core::resources::ImageResourceIdentity,
    pub transform: [f64; 6],
    pub encoded_budget: graphics_core::resources::EncodedPngBudget,
}

enum ImageLoadOutcome {
    Known(graphics_core::resources::AdmittedCpuImage),
    Encoded(crate::resources::ResolvedEncodedPngImage),
}

#[derive(Clone)]
enum PendingImageRequest {
    Known(crate::resources::ImageResourceRequest),
    Encoded(crate::resources::EncodedPngResourceRequest),
}

impl PendingImageRequest {
    fn generation(&self) -> u64 {
        match self {
            Self::Known(request) => request.generation(),
            Self::Encoded(request) => request.generation(),
        }
    }

    fn identity(&self) -> &graphics_core::resources::ImageResourceIdentity {
        match self {
            Self::Known(request) => request.identity(),
            Self::Encoded(request) => request.identity(),
        }
    }

    fn accept(
        &self,
        resources: &mut crate::resources::ImageResourceState,
        image: ImageLoadOutcome,
    ) -> Result<(), crate::resources::ImageResourceError> {
        match (self, image) {
            (Self::Known(request), ImageLoadOutcome::Known(image)) => {
                resources.accept(request, image)
            }
            (Self::Encoded(request), ImageLoadOutcome::Encoded(image)) => {
                resources.accept_encoded_png(request, image)
            }
            _ => Err(crate::resources::ImageResourceError::StaleRequest),
        }
    }

    fn reject(
        &self,
        resources: &mut crate::resources::ImageResourceState,
        failure: crate::resources::ImageResourceFailure,
    ) -> Result<(), crate::resources::ImageResourceError> {
        match self {
            Self::Known(request) => resources.reject(request, failure),
            Self::Encoded(request) => resources.reject_encoded_png(request, failure),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ImageSessionDiagnostic {
    Resource(crate::resources::ImageResourceError),
    Read(String),
    Evaluation(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ImageSessionEvent {
    Changed,
}

struct PendingImageLoad {
    request: PendingImageRequest,
    cancellation: sim_tensor::CancellationToken,
    _task: gpui::Task<()>,
}

struct PendingImageEdit {
    cancellation: sim_tensor::CancellationToken,
    _task: gpui::Task<()>,
}

struct CommittedImageEdit {
    source: std::sync::Weak<graphics_core::resources::AdmittedCpuImage>,
    source_revision: u64,
    result: std::sync::Arc<graphics_core::resources::CpuImageResult>,
}

pub struct PreparedImageScenePreview {
    pub(crate) image: std::sync::Arc<gpui::RenderImage>,
    pub(crate) scene: std::sync::Arc<graphics_core::render::PreparedCpuSvgComposition>,
    pub(crate) source: std::sync::Arc<graphics_core::resources::AdmittedCpuImage>,
    pub(crate) result: Option<std::sync::Arc<graphics_core::resources::CpuImageResult>>,
    pub(crate) source_revision: u64,
    edit_generation: u64,
    pub(crate) scene_generation: u64,
    cancellation: std::sync::Arc<sim_tensor::CancellationToken>,
    exact_target: bool,
}

impl PreparedImageScenePreview {
    pub(crate) fn is_exact_target(&self) -> bool {
        self.exact_target
    }
}

pub struct PreparedScenePngAsset {
    preview: PreparedImageScenePreview,
    encoded: graphics_core::export::PreparedRgba8Png,
    cancellation: std::sync::Arc<sim_tensor::CancellationToken>,
}

impl PreparedScenePngAsset {
    pub fn renderer_profile(&self) -> &'static str {
        "gpui::SvgRenderer::render_parsed_in_canvas"
    }

    pub fn png(&self) -> &graphics_core::export::PreparedPngImage {
        self.encoded.png()
    }
    pub fn frame_pixels_sha256(&self) -> [u8; 32] {
        self.encoded.pixels_sha256()
    }
    pub fn scene(&self) -> &graphics_core::render::PreparedCpuSvgComposition {
        &self.preview.scene
    }
    pub fn source(&self) -> &graphics_core::resources::AdmittedCpuImage {
        &self.preview.source
    }
    pub fn result(&self) -> Option<&graphics_core::resources::CpuImageResult> {
        self.preview.result.as_deref()
    }
    pub fn source_revision(&self) -> u64 {
        self.preview.source_revision
    }
}

enum ScenePreviewRasterization {
    Legacy { render_scale: f32 },
    ExactCanvas,
}

pub struct GraphiteImageSession {
    resources: crate::resources::ImageResourceState,
    operation: Option<graphics_core::resources::CpuImageEdit>,
    result: Option<CommittedImageEdit>,
    diagnostic: Option<ImageSessionDiagnostic>,
    max_result_bytes: usize,
    edit_generation: u64,
    pending_load: Option<PendingImageLoad>,
    pending_edit: Option<PendingImageEdit>,
    pending_png: Option<std::sync::Arc<sim_tensor::CancellationToken>>,
    pending_scene: Option<std::sync::Arc<sim_tensor::CancellationToken>>,
    scene_generation: u64,
}

impl gpui::EventEmitter<ImageSessionEvent> for GraphiteImageSession {}

impl Drop for GraphiteImageSession {
    fn drop(&mut self) {
        self.cancel_work();
    }
}

impl GraphiteImageSession {
    pub fn new(max_result_bytes: usize) -> Self {
        Self {
            resources: crate::resources::ImageResourceState::default(),
            operation: None,
            result: None,
            diagnostic: None,
            max_result_bytes,
            edit_generation: 0,
            pending_load: None,
            pending_edit: None,
            pending_png: None,
            pending_scene: None,
            scene_generation: 0,
        }
    }

    pub fn resource_state(&self) -> &crate::resources::ImageResourceState {
        &self.resources
    }

    pub fn operation(&self) -> Option<graphics_core::resources::CpuImageEdit> {
        self.operation
    }

    pub fn result(&self) -> Option<&std::sync::Arc<graphics_core::resources::CpuImageResult>> {
        if self.resources.is_revoked() || self.operation.is_none() {
            return None;
        }
        self.result.as_ref().map(|result| &result.result)
    }

    pub fn result_is_fresh(&self) -> bool {
        self.resources.is_fresh()
            && !self.resources.is_revoked()
            && self.result.as_ref().is_some_and(|result| {
                self.operation == Some(result.result.operation())
                    && result.source_revision == self.resources.revision()
                    && self.resources.committed().is_some_and(|source| {
                        std::sync::Weak::ptr_eq(&result.source, &std::sync::Arc::downgrade(source))
                    })
            })
    }

    pub fn diagnostic(&self) -> Option<&ImageSessionDiagnostic> {
        self.diagnostic.as_ref()
    }

    pub fn scene_preview_is_current(&self, preview: &PreparedImageScenePreview) -> bool {
        !preview.cancellation.is_cancelled()
            && self.scene_generation == preview.scene_generation
            && self.edit_generation == preview.edit_generation
            && self.resources.revision() == preview.source_revision
            && self.resources.is_fresh()
            && !self.resources.is_revoked()
            && self
                .resources
                .committed()
                .is_some_and(|source| std::sync::Arc::ptr_eq(source, &preview.source))
            && match (self.operation, preview.result.as_ref()) {
                (None, None) => true,
                (Some(_), Some(prepared_result)) => {
                    self.result_is_fresh()
                        && self
                            .result()
                            .is_some_and(|result| std::sync::Arc::ptr_eq(result, prepared_result))
                }
                _ => false,
            }
    }

    pub fn prepare_scene_preview(
        &mut self,
        configuration: graphics_core::render::CpuSvgSceneConfiguration,
        render_scale: f32,
        max_preview_bytes: usize,
        cx: &mut gpui::Context<Self>,
    ) -> Result<gpui::Task<Result<PreparedImageScenePreview, String>>, String> {
        self.prepare_scene_preview_with_rasterization(
            configuration,
            ScenePreviewRasterization::Legacy { render_scale },
            max_preview_bytes,
            cx,
        )
    }

    pub fn prepare_scene_preview_at_target(
        &mut self,
        configuration: graphics_core::render::CpuSvgSceneConfiguration,
        max_preview_bytes: usize,
        cx: &mut gpui::Context<Self>,
    ) -> Result<gpui::Task<Result<PreparedImageScenePreview, String>>, String> {
        self.prepare_scene_preview_with_rasterization(
            configuration,
            ScenePreviewRasterization::ExactCanvas,
            max_preview_bytes,
            cx,
        )
    }

    fn prepare_scene_preview_with_rasterization(
        &mut self,
        configuration: graphics_core::render::CpuSvgSceneConfiguration,
        rasterization: ScenePreviewRasterization,
        max_preview_bytes: usize,
        cx: &mut gpui::Context<Self>,
    ) -> Result<gpui::Task<Result<PreparedImageScenePreview, String>>, String> {
        if !self.resources.is_fresh() || self.resources.is_revoked() {
            return Err("Image source is not current".into());
        }
        let source = self
            .resources
            .committed()
            .cloned()
            .ok_or_else(|| "Image source is unavailable".to_owned())?;
        let result = if self.operation.is_some() {
            if !self.result_is_fresh() {
                return Err("Image edit result is not current".into());
            }
            Some(
                self.result()
                    .cloned()
                    .ok_or_else(|| "Image edit result is unavailable".to_owned())?,
            )
        } else {
            None
        };
        let scene_generation = self
            .scene_generation
            .checked_add(1)
            .ok_or_else(|| "Scene preview generation is exhausted".to_owned())?;
        self.cancel_scene();
        self.scene_generation = scene_generation;
        let cancellation = std::sync::Arc::new(sim_tensor::CancellationToken::default());
        self.pending_scene = Some(cancellation.clone());
        let source_revision = self.resources.revision();
        let edit_generation = self.edit_generation;
        let exact_target = matches!(rasterization, ScenePreviewRasterization::ExactCanvas);
        let renderer = cx.svg_renderer();
        let background = cx.background_spawn({
            let source = source.clone();
            let result = result.clone();
            let cancellation = cancellation.clone();
            async move {
                cancellation.check().map_err(|error| error.to_string())?;
                let layer = match result {
                    Some(result) => {
                        graphics_core::render::CpuSvgSceneLayer::Image((*result).clone())
                    }
                    None => graphics_core::render::CpuSvgSceneLayer::SourceImage(source),
                };
                let scene =
                    graphics_core::render::prepare_cpu_svg_composition(&[layer], configuration)
                        .map_err(|error| error.to_string())?;
                cancellation.check().map_err(|error| error.to_string())?;
                let image = match rasterization {
                    ScenePreviewRasterization::Legacy { render_scale } => {
                        crate::render::prepare_cpu_svg_preview(
                            &scene,
                            &renderer,
                            render_scale,
                            max_preview_bytes,
                        )
                    }
                    ScenePreviewRasterization::ExactCanvas => {
                        crate::render::prepare_cpu_svg_preview_at_target(
                            &scene,
                            &renderer,
                            max_preview_bytes,
                        )
                    }
                }
                .map_err(|error| error.to_string())?;
                cancellation.check().map_err(|error| error.to_string())?;
                Ok::<_, String>((std::sync::Arc::new(scene), image))
            }
        });
        Ok(cx.spawn(async move |session, cx| {
            let outcome = background.await;
            session
                .update(cx, |session, _cx| {
                    if !session
                        .pending_scene
                        .as_ref()
                        .is_some_and(|pending| std::sync::Arc::ptr_eq(pending, &cancellation))
                    {
                        return Err("Scene preview request was superseded or cancelled".into());
                    }
                    session.pending_scene = None;
                    cancellation.check().map_err(|error| error.to_string())?;
                    let (scene, image) = outcome?;
                    let preview = PreparedImageScenePreview {
                        image,
                        scene,
                        source,
                        result,
                        source_revision,
                        edit_generation,
                        scene_generation,
                        cancellation,
                        exact_target,
                    };
                    if !session.scene_preview_is_current(&preview) {
                        return Err(
                            "Image source or operation changed during scene preparation".into()
                        );
                    }
                    Ok(preview)
                })
                .map_err(|error| error.to_string())?
        }))
    }

    pub fn scene_png_is_current(&self, prepared: &PreparedScenePngAsset) -> bool {
        !prepared.cancellation.is_cancelled() && self.scene_preview_is_current(&prepared.preview)
    }

    /// Prepares a caller-target export-style SVG snapshot; it does not publish
    /// an asset version or encode a complete editable document.
    pub fn prepare_scene_png(
        &mut self,
        preview: PreparedImageScenePreview,
        max_pixel_bytes: usize,
        max_encoded_bytes: usize,
        cx: &mut gpui::Context<Self>,
    ) -> Result<gpui::Task<Result<PreparedScenePngAsset, String>>, String> {
        if !preview.exact_target || !self.scene_preview_is_current(&preview) {
            return Err("PNG requires a current exact-target scene snapshot".into());
        }
        self.cancel_png();
        let cancellation = std::sync::Arc::new(sim_tensor::CancellationToken::default());
        self.pending_png = Some(cancellation.clone());
        let background = cx.background_spawn({
            let image = preview.image.clone();
            let scene = preview.scene.clone();
            let cancellation = cancellation.clone();
            let scene_cancellation = preview.cancellation.clone();
            async move {
                cancellation.check().map_err(|error| error.to_string())?;
                scene_cancellation
                    .check()
                    .map_err(|error| error.to_string())?;
                let encoded = crate::render::prepare_cpu_svg_scene_png(
                    &scene,
                    &image,
                    max_pixel_bytes,
                    max_encoded_bytes,
                );
                cancellation.check().map_err(|error| error.to_string())?;
                scene_cancellation
                    .check()
                    .map_err(|error| error.to_string())?;
                encoded
            }
        });
        Ok(cx.spawn(async move |session, cx| {
            let outcome = background.await;
            session
                .update(cx, |session, _cx| {
                    if !session
                        .pending_png
                        .as_ref()
                        .is_some_and(|pending| std::sync::Arc::ptr_eq(pending, &cancellation))
                    {
                        return Err("Scene PNG request was superseded or cancelled".into());
                    }
                    session.pending_png = None;
                    cancellation.check().map_err(|error| error.to_string())?;
                    if !session.scene_preview_is_current(&preview) {
                        return Err(
                            "Scene target, source or edit changed during PNG preparation".into(),
                        );
                    }
                    Ok(PreparedScenePngAsset {
                        preview,
                        encoded: outcome?,
                        cancellation,
                    })
                })
                .map_err(|error| error.to_string())?
        }))
    }

    pub fn prepare_png(
        &mut self,
        max_encoded_bytes: usize,
        cx: &mut gpui::Context<Self>,
    ) -> Result<gpui::Task<Result<graphics_core::export::PreparedPngAsset, String>>, String> {
        if !self.resources.is_fresh() || self.resources.is_revoked() {
            return Err("Image source is not current".into());
        }
        let source = self
            .resources
            .committed()
            .cloned()
            .ok_or_else(|| "Image source is unavailable".to_owned())?;
        let result = if self.operation.is_some() {
            if !self.result_is_fresh() {
                return Err("Image edit result is not current".into());
            }
            Some(
                self.result()
                    .cloned()
                    .ok_or_else(|| "Image edit result is unavailable".to_owned())?,
            )
        } else {
            None
        };
        self.cancel_png();
        let cancellation = std::sync::Arc::new(sim_tensor::CancellationToken::default());
        self.pending_png = Some(cancellation.clone());
        let source_revision = self.resources.revision();
        let operation = self.operation;
        let edit_generation = self.edit_generation;
        let background = cx.background_spawn({
            let source = source.clone();
            let result = result.clone();
            let cancellation = cancellation.clone();
            async move {
                cancellation.check().map_err(|error| error.to_string())?;
                let prepared = match &result {
                    Some(result) => {
                        graphics_core::export::prepare_cpu_result_png(result, max_encoded_bytes)
                    }
                    None => graphics_core::export::prepare_png(&source, max_encoded_bytes),
                }
                .map_err(|error| error.to_string());
                cancellation.check().map_err(|error| error.to_string())?;
                prepared
            }
        });
        Ok(cx.spawn(async move |session, cx| {
            let outcome = background.await;
            session
                .update(cx, |session, _cx| {
                    cancellation.check().map_err(|error| error.to_string())?;
                    let current_result = match &result {
                        Some(result) => {
                            session.result_is_fresh()
                                && session
                                    .result()
                                    .is_some_and(|current| std::sync::Arc::ptr_eq(current, result))
                        }
                        None => session.operation.is_none(),
                    };
                    if !session
                        .pending_png
                        .as_ref()
                        .is_some_and(|pending| std::sync::Arc::ptr_eq(pending, &cancellation))
                        || !session.resources.is_fresh()
                        || session.resources.is_revoked()
                        || session.resources.revision() != source_revision
                        || session.operation != operation
                        || session.edit_generation != edit_generation
                        || !current_result
                        || !session
                            .resources
                            .committed()
                            .is_some_and(|current| std::sync::Arc::ptr_eq(current, &source))
                    {
                        return Err(
                            "Image source or operation changed during PNG preparation".into()
                        );
                    }
                    session.pending_png = None;
                    outcome
                })
                .map_err(|error| error.to_string())?
        }))
    }

    pub fn load_png(
        &mut self,
        load: GraphitePngLoad,
        cx: &mut gpui::Context<Self>,
    ) -> Result<(), crate::resources::ImageResourceError> {
        let decision = match self.resources.request(load.identity, load.descriptor) {
            Ok(decision) => decision,
            Err(error) => {
                self.diagnostic = Some(ImageSessionDiagnostic::Resource(error.clone()));
                self.changed(cx);
                return Err(error);
            }
        };
        self.cancel_work();
        self.diagnostic = None;
        let crate::resources::ImageRequestDecision::Load(request) = decision else {
            self.start_edit(cx);
            return Ok(());
        };
        let cancellation = sim_tensor::CancellationToken::default();
        let background = cx.background_spawn({
            let request = request.clone();
            let cancellation = cancellation.clone();
            async move {
                crate::resources::resolve_graphite_png_request(
                    &load.assets,
                    &load.asset_identity,
                    &load.authorization,
                    &cancellation,
                    &request,
                    load.encoded_budget,
                )
                .map(ImageLoadOutcome::Known)
            }
        });
        self.schedule_load(
            PendingImageRequest::Known(request),
            cancellation,
            background,
            cx,
        );
        Ok(())
    }

    pub fn load_encoded_png(
        &mut self,
        load: GraphiteEncodedPngLoad,
        cx: &mut gpui::Context<Self>,
    ) -> Result<(), crate::resources::ImageResourceError> {
        let decision = match self.resources.request_encoded_png(
            load.identity,
            load.transform,
            load.encoded_budget,
        ) {
            Ok(decision) => decision,
            Err(error) => {
                self.diagnostic = Some(ImageSessionDiagnostic::Resource(error.clone()));
                self.changed(cx);
                return Err(error);
            }
        };
        self.cancel_work();
        self.diagnostic = None;
        let crate::resources::ImageRequestDecision::Load(request) = decision else {
            self.start_edit(cx);
            return Ok(());
        };
        let cancellation = sim_tensor::CancellationToken::default();
        let background = cx.background_spawn({
            let request = request.clone();
            let cancellation = cancellation.clone();
            async move {
                crate::resources::resolve_graphite_encoded_png_request(
                    &load.assets,
                    &load.asset_identity,
                    &load.authorization,
                    &cancellation,
                    &request,
                )
                .map(ImageLoadOutcome::Encoded)
            }
        });
        self.schedule_load(
            PendingImageRequest::Encoded(request),
            cancellation,
            background,
            cx,
        );
        Ok(())
    }

    fn schedule_load(
        &mut self,
        request: PendingImageRequest,
        cancellation: sim_tensor::CancellationToken,
        background: gpui::Task<Result<ImageLoadOutcome, crate::resources::GraphiteAssetReadError>>,
        cx: &mut gpui::Context<Self>,
    ) {
        let task = cx.spawn({
            let request = request.clone();
            let cancellation = cancellation.clone();
            async move |session, cx| {
                let outcome = background.await;
                match session.update(cx, |session, cx| {
                    session.finish_load(&request, &cancellation, outcome, cx);
                }) {
                    Ok(()) => {}
                    Err(_released) => {
                        cancellation.cancel();
                    }
                }
            }
        });
        self.pending_load = Some(PendingImageLoad {
            request,
            cancellation,
            _task: task,
        });
        self.changed(cx);
    }

    pub fn set_operation(
        &mut self,
        operation: Option<graphics_core::resources::CpuImageEdit>,
        cx: &mut gpui::Context<Self>,
    ) {
        if self.operation == operation && (operation.is_none() || self.result_is_fresh()) {
            return;
        }
        self.operation = operation;
        self.start_edit(cx);
    }

    pub fn cancel(&mut self, cx: &mut gpui::Context<Self>) {
        let active = self
            .resources
            .active_request()
            .cloned()
            .map(PendingImageRequest::Known)
            .or_else(|| {
                self.resources
                    .active_encoded_request()
                    .cloned()
                    .map(PendingImageRequest::Encoded)
            });
        let had_work = self.pending_load.is_some()
            || self.pending_edit.is_some()
            || self.pending_png.is_some()
            || self.pending_scene.is_some();
        self.cancel_work();
        if had_work {
            self.diagnostic = Some(ImageSessionDiagnostic::Resource(
                crate::resources::ImageResourceError::Cancelled,
            ));
        }
        if let Some(request) = active {
            match request.reject(
                &mut self.resources,
                crate::resources::ImageResourceFailure::Cancelled,
            ) {
                Ok(()) => {}
                Err(error) => self.diagnostic = Some(ImageSessionDiagnostic::Resource(error)),
            }
        }
        self.changed(cx);
    }

    pub fn revoke(
        &mut self,
        identity: &graphics_core::resources::ImageResourceIdentity,
        cx: &mut gpui::Context<Self>,
    ) -> Result<(), crate::resources::ImageResourceError> {
        match self.resources.revoke(identity) {
            Ok(()) => {
                self.cancel_edit();
                if self
                    .pending_load
                    .as_ref()
                    .is_some_and(|pending| pending.request.identity() == identity)
                    && let Some(pending) = self.pending_load.take()
                {
                    pending.cancellation.cancel();
                }
                self.diagnostic = Some(ImageSessionDiagnostic::Resource(
                    crate::resources::ImageResourceError::Revoked,
                ));
                self.changed(cx);
                Ok(())
            }
            Err(error) => {
                self.diagnostic = Some(ImageSessionDiagnostic::Resource(error.clone()));
                self.changed(cx);
                Err(error)
            }
        }
    }

    pub(crate) fn revoke_all_resources(
        &mut self,
        cx: &mut gpui::Context<Self>,
    ) -> Vec<crate::resources::ImageResourceError> {
        let mut identities = Vec::new();
        if let Some(image) = self.resources.committed() {
            identities.push(image.identity().clone());
        }
        let active_identity = self
            .resources
            .active_request()
            .map(|request| request.identity())
            .or_else(|| {
                self.resources
                    .active_encoded_request()
                    .map(|request| request.identity())
            });
        if let Some(identity) = active_identity
            && !identities.contains(identity)
        {
            identities.push(identity.clone());
        }
        let mut errors = Vec::new();
        for identity in identities {
            if let Err(error) = self.revoke(&identity, cx) {
                errors.push(error);
            }
        }
        self.cancel_work();
        self.changed(cx);
        errors
    }

    fn finish_load(
        &mut self,
        request: &PendingImageRequest,
        cancellation: &sim_tensor::CancellationToken,
        outcome: Result<ImageLoadOutcome, crate::resources::GraphiteAssetReadError>,
        cx: &mut gpui::Context<Self>,
    ) {
        if cancellation.is_cancelled()
            || !self.pending_load.as_ref().is_some_and(|pending| {
                pending.request.generation() == request.generation()
                    && pending.request.identity() == request.identity()
            })
        {
            return;
        }
        self.pending_load = None;
        match outcome {
            Ok(image) => match request.accept(&mut self.resources, image) {
                Ok(()) => {
                    self.diagnostic = None;
                    self.start_edit(cx);
                    return;
                }
                Err(error) => self.diagnostic = Some(ImageSessionDiagnostic::Resource(error)),
            },
            Err(error) => {
                let failure = match &error {
                    crate::resources::GraphiteAssetReadError::Asset(
                        sim_runtime::assets::AssetError::PermissionDenied { .. },
                    ) => crate::resources::ImageResourceFailure::Denied,
                    crate::resources::GraphiteAssetReadError::Asset(
                        sim_runtime::assets::AssetError::Cancelled,
                    ) => crate::resources::ImageResourceFailure::Cancelled,
                    _ => crate::resources::ImageResourceFailure::Failed(error.to_string()),
                };
                match request.reject(&mut self.resources, failure) {
                    Ok(()) => {
                        self.diagnostic = Some(ImageSessionDiagnostic::Read(error.to_string()))
                    }
                    Err(state_error) => {
                        self.diagnostic = Some(ImageSessionDiagnostic::Resource(state_error));
                    }
                }
            }
        }
        self.changed(cx);
    }

    fn start_edit(&mut self, cx: &mut gpui::Context<Self>) {
        self.cancel_edit();
        let Some(operation) = self.operation else {
            self.result = None;
            self.diagnostic = self
                .resources
                .diagnostic()
                .cloned()
                .map(ImageSessionDiagnostic::Resource);
            self.changed(cx);
            return;
        };
        if self.resources.is_revoked() || !self.resources.is_fresh() {
            self.changed(cx);
            return;
        }
        let Some(source) = self.resources.committed().cloned() else {
            self.changed(cx);
            return;
        };
        if self.result_is_fresh() {
            self.diagnostic = self
                .resources
                .diagnostic()
                .cloned()
                .map(ImageSessionDiagnostic::Resource);
            self.changed(cx);
            return;
        }
        let Some(generation) = self.edit_generation.checked_add(1) else {
            self.diagnostic = Some(ImageSessionDiagnostic::Resource(
                crate::resources::ImageResourceError::GenerationExhausted,
            ));
            self.changed(cx);
            return;
        };
        self.edit_generation = generation;
        let source_revision = self.resources.revision();
        let cancellation = sim_tensor::CancellationToken::default();
        let background = cx.background_spawn({
            let source = source.clone();
            let cancellation = cancellation.clone();
            let max_result_bytes = self.max_result_bytes;
            async move {
                cancellation.check().map_err(|error| error.to_string())?;
                let result = graphics_core::resources::evaluate_cpu_image_edit(
                    &source,
                    operation,
                    max_result_bytes,
                )
                .map_err(|error| error.to_string())?;
                cancellation.check().map_err(|error| error.to_string())?;
                Ok::<_, String>(result)
            }
        });
        let task = cx.spawn({
            let cancellation = cancellation.clone();
            async move |session, cx| {
                let outcome = background.await;
                match session.update(cx, |session, cx| {
                    if cancellation.is_cancelled()
                        || session.edit_generation != generation
                        || session.operation != Some(operation)
                        || !session.resources.is_fresh()
                        || session.resources.is_revoked()
                        || session.resources.revision() != source_revision
                        || !session
                            .resources
                            .committed()
                            .is_some_and(|committed| std::sync::Arc::ptr_eq(committed, &source))
                    {
                        return;
                    }
                    session.pending_edit = None;
                    match outcome {
                        Ok(result) => {
                            session.result = Some(CommittedImageEdit {
                                source: std::sync::Arc::downgrade(&source),
                                source_revision,
                                result: std::sync::Arc::new(result),
                            });
                            session.diagnostic = None;
                        }
                        Err(error) => {
                            session.diagnostic = Some(ImageSessionDiagnostic::Evaluation(error))
                        }
                    }
                    session.changed(cx);
                }) {
                    Ok(()) => {}
                    Err(_released) => {
                        cancellation.cancel();
                    }
                }
            }
        });
        self.pending_edit = Some(PendingImageEdit {
            cancellation,
            _task: task,
        });
        self.changed(cx);
    }

    fn cancel_png(&mut self) {
        if let Some(cancellation) = self.pending_png.take() {
            cancellation.cancel();
        }
    }

    fn cancel_scene(&mut self) {
        if let Some(cancellation) = self.pending_scene.take() {
            cancellation.cancel();
        }
    }

    fn cancel_edit(&mut self) {
        self.cancel_scene();
        self.cancel_png();
        if let Some(pending) = self.pending_edit.take() {
            pending.cancellation.cancel();
        }
    }

    fn cancel_work(&mut self) {
        if let Some(pending) = self.pending_load.take() {
            pending.cancellation.cancel();
        }
        self.cancel_edit();
    }

    fn changed(&self, cx: &mut gpui::Context<Self>) {
        cx.emit(ImageSessionEvent::Changed);
        cx.notify();
    }
}

#[derive(Clone)]
struct MixSourceSnapshot {
    sources: [std::sync::Arc<graphics_core::resources::AdmittedCpuImage>; 2],
    revisions: [u64; 2],
}

impl MixSourceSnapshot {
    fn matches(&self, other: &Self) -> bool {
        self.revisions == other.revisions
            && self
                .sources
                .iter()
                .zip(&other.sources)
                .all(|(source, other)| std::sync::Arc::ptr_eq(source, other))
    }
}

#[derive(Clone)]
struct MixEvaluationSnapshot {
    inputs: MixSourceSnapshot,
    parameters: graphics_core::resources::CpuMixParameters,
}

impl MixEvaluationSnapshot {
    fn matches(&self, other: &Self) -> bool {
        self.parameters == other.parameters && self.inputs.matches(&other.inputs)
    }
}

struct CommittedImageMix {
    sources: [std::sync::Weak<graphics_core::resources::AdmittedCpuImage>; 2],
    source_revisions: [u64; 2],
    generation: u64,
    result: std::sync::Arc<graphics_core::resources::CpuMixResult>,
}

pub struct GraphiteMixSession {
    parents: [gpui::Entity<GraphiteImageSession>; 2],
    parameters: graphics_core::resources::CpuMixParameters,
    budget: graphics_core::resources::CpuMixBudget,
    generation: u64,
    pending: Option<PendingImageEdit>,
    pending_snapshot: Option<MixEvaluationSnapshot>,
    result: Option<CommittedImageMix>,
    cancelled: bool,
    diagnostic: Option<String>,
    _subscriptions: Vec<gpui::Subscription>,
}

impl gpui::EventEmitter<ImageSessionEvent> for GraphiteMixSession {}

impl Drop for GraphiteMixSession {
    fn drop(&mut self) {
        self.cancel_pending();
    }
}

impl GraphiteMixSession {
    pub fn new(
        parents: [gpui::Entity<GraphiteImageSession>; 2],
        parameters: graphics_core::resources::CpuMixParameters,
        budget: graphics_core::resources::CpuMixBudget,
        cx: &mut gpui::Context<Self>,
    ) -> Self {
        let subscriptions = parents
            .iter()
            .map(|parent| cx.subscribe(parent, |session, _, _, cx| session.refresh(cx)))
            .collect();
        let mut session = Self {
            parents,
            parameters,
            budget,
            generation: 0,
            pending: None,
            pending_snapshot: None,
            result: None,
            cancelled: false,
            diagnostic: None,
            _subscriptions: subscriptions,
        };
        session.refresh(cx);
        session
    }

    pub fn parents(&self) -> &[gpui::Entity<GraphiteImageSession>; 2] {
        &self.parents
    }

    pub fn parameters(&self) -> graphics_core::resources::CpuMixParameters {
        self.parameters
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    pub fn result_generation(&self) -> Option<u64> {
        self.result.as_ref().map(|result| result.generation)
    }

    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    pub fn is_revoked(&self, cx: &gpui::App) -> bool {
        self.parents
            .iter()
            .any(|parent| parent.read(cx).resource_state().is_revoked())
    }

    pub fn result(
        &self,
        cx: &gpui::App,
    ) -> Option<&std::sync::Arc<graphics_core::resources::CpuMixResult>> {
        if self.is_revoked(cx) {
            return None;
        }
        self.result.as_ref().map(|result| &result.result)
    }

    pub fn result_is_fresh(&self, cx: &gpui::App) -> bool {
        if self.cancelled {
            return false;
        }
        let Ok(current) = self.source_snapshot(cx) else {
            return false;
        };
        self.result.as_ref().is_some_and(|result| {
            result.generation == self.generation
                && result.result.parameters() == self.parameters
                && result.source_revisions == current.revisions
                && result
                    .sources
                    .iter()
                    .zip(&current.sources)
                    .all(|(source, current)| {
                        std::sync::Weak::ptr_eq(source, &std::sync::Arc::downgrade(current))
                    })
        })
    }

    pub fn prepare_png(
        &self,
        max_encoded_bytes: usize,
        cx: &mut gpui::Context<Self>,
    ) -> Result<gpui::Task<Result<graphics_core::export::PreparedMixPngAsset, String>>, String>
    {
        if !self.result_is_fresh(cx) {
            return Err("Mix result is not current".into());
        }
        let inputs = self.source_snapshot(cx)?;
        let result = self
            .result(cx)
            .cloned()
            .ok_or_else(|| "Mix result is unavailable".to_owned())?;
        let generation = self.generation;
        let parameters = self.parameters;
        let background = cx.background_spawn({
            let result = result.clone();
            async move { graphics_core::export::prepare_cpu_mix_png(&result, max_encoded_bytes) }
        });
        Ok(cx.spawn(async move |session, cx| {
            let prepared = background.await.map_err(|error| error.to_string())?;
            session
                .update(cx, |session, cx| {
                    let current = session.source_snapshot(cx)?;
                    if !session.result_is_fresh(cx)
                        || session.generation != generation
                        || session.parameters != parameters
                        || !inputs.matches(&current)
                        || !session
                            .result(cx)
                            .is_some_and(|current| std::sync::Arc::ptr_eq(current, &result))
                    {
                        return Err(
                            "Mix inputs or parameters changed during PNG preparation".into()
                        );
                    }
                    Ok(prepared)
                })
                .map_err(|error| error.to_string())?
        }))
    }

    pub fn diagnostic(&self) -> Option<&str> {
        self.diagnostic.as_deref()
    }

    pub fn set_parameters(
        &mut self,
        parameters: graphics_core::resources::CpuMixParameters,
        cx: &mut gpui::Context<Self>,
    ) {
        self.parameters = parameters;
        self.refresh(cx);
    }

    pub fn cancel(&mut self, cx: &mut gpui::Context<Self>) {
        self.cancel_pending();
        self.cancelled = true;
        self.diagnostic = Some("Mix evaluation cancelled".into());
        self.changed(cx);
    }

    pub fn refresh(&mut self, cx: &mut gpui::Context<Self>) {
        let inputs = match self.source_snapshot(cx) {
            Ok(inputs) => inputs,
            Err(error) => {
                self.cancel_pending();
                self.diagnostic = Some(error);
                self.changed(cx);
                return;
            }
        };
        let snapshot = MixEvaluationSnapshot {
            inputs,
            parameters: self.parameters,
        };
        if self.pending.is_some()
            && self
                .pending_snapshot
                .as_ref()
                .is_some_and(|pending| pending.matches(&snapshot))
        {
            return;
        }
        if self.result_is_fresh(cx) {
            self.diagnostic = None;
            self.changed(cx);
            return;
        }
        self.cancel_pending();
        let Some(generation) = self.generation.checked_add(1) else {
            self.cancelled = true;
            self.diagnostic = Some("Mix evaluation generation exhausted".into());
            self.changed(cx);
            return;
        };
        self.generation = generation;
        self.cancelled = false;
        self.diagnostic = None;
        let cancellation = sim_tensor::CancellationToken::default();
        let background = cx.background_spawn({
            let snapshot = snapshot.clone();
            let cancellation = cancellation.clone();
            let budget = self.budget;
            async move {
                cancellation.check().map_err(|error| error.to_string())?;
                let [over, under] = &snapshot.inputs.sources;
                let result = graphics_core::resources::evaluate_cpu_image_mix(
                    over,
                    under,
                    snapshot.parameters,
                    budget,
                )
                .map_err(|error| error.to_string())?;
                cancellation.check().map_err(|error| error.to_string())?;
                Ok::<_, String>(result)
            }
        });
        let task = cx.spawn({
            let snapshot = snapshot.clone();
            let cancellation = cancellation.clone();
            async move |session, cx| {
                let outcome = background.await;
                match session.update(cx, |session, cx| {
                    if cancellation.is_cancelled()
                        || session.cancelled
                        || session.generation != generation
                        || session.parameters != snapshot.parameters
                        || !session
                            .source_snapshot(cx)
                            .is_ok_and(|current| current.matches(&snapshot.inputs))
                    {
                        return;
                    }
                    session.pending = None;
                    session.pending_snapshot = None;
                    match outcome {
                        Ok(result) => {
                            session.result = Some(CommittedImageMix {
                                sources: snapshot
                                    .inputs
                                    .sources
                                    .each_ref()
                                    .map(std::sync::Arc::downgrade),
                                source_revisions: snapshot.inputs.revisions,
                                generation,
                                result: std::sync::Arc::new(result),
                            });
                            session.diagnostic = None;
                        }
                        Err(error) => session.diagnostic = Some(error),
                    }
                    session.changed(cx);
                }) {
                    Ok(()) => {}
                    Err(_released) => {
                        cancellation.cancel();
                    }
                }
            }
        });
        self.pending_snapshot = Some(snapshot);
        self.pending = Some(PendingImageEdit {
            cancellation,
            _task: task,
        });
        self.changed(cx);
    }

    fn source_snapshot(&self, cx: &gpui::App) -> Result<MixSourceSnapshot, String> {
        let [over, under] = self
            .parents
            .each_ref()
            .map(|parent| parent.read(cx).resource_state());
        for (state, role) in [(over, "Over"), (under, "Under")] {
            if state.is_revoked() {
                return Err(format!("{role} image authorization revoked"));
            }
            if !state.is_fresh() {
                return Err(state.diagnostic().map_or_else(
                    || format!("{role} image is not currently fresh"),
                    |error| format!("{role}: {error}"),
                ));
            }
        }
        let [Some(over_source), Some(under_source)] = [over.committed(), under.committed()] else {
            return Err("Mix requires both source images".into());
        };
        Ok(MixSourceSnapshot {
            sources: [over_source.clone(), under_source.clone()],
            revisions: [over.revision(), under.revision()],
        })
    }

    fn cancel_pending(&mut self) {
        if let Some(pending) = self.pending.take() {
            pending.cancellation.cancel();
        }
        self.pending_snapshot = None;
    }

    fn changed(&self, cx: &mut gpui::Context<Self>) {
        cx.emit(ImageSessionEvent::Changed);
        cx.notify();
    }
}

#[cfg(test)]
mod image_session_tests {
    use super::*;
    use crate::resources::tests::{
        HostReaderFixture, IndexedRequestPng, host_reader_fixture, indexed_request_png,
    };
    use gpui::TestAppContext;
    use graphics_core::resources::{CpuImageEdit, EncodedPngBudget};
    use std::sync::Arc;

    fn load(fixture: &HostReaderFixture, png: &IndexedRequestPng) -> GraphitePngLoad {
        GraphitePngLoad {
            assets: fixture.assets.clone(),
            asset_identity: png.asset_identity.clone(),
            authorization: fixture.input_authorization.clone(),
            identity: png.image_identity.clone(),
            descriptor: png.descriptor,
            encoded_budget: EncodedPngBudget {
                max_encoded_bytes: 16 * 1024,
                max_pixel_bytes: 4096,
                max_metadata_bytes: 4096,
            },
        }
    }

    #[gpui::test]
    fn at_graphite_image_session_loads_real_png_and_explicitly_selects_original_edits(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real local indexed service");
        let png = indexed_request_png(&fixture, 4).expect("actual PNG producer");
        let owner = cx.new(|_| GraphiteImageSession::new(4096));
        owner
            .update(cx, |owner, cx| owner.load_png(load(&fixture, &png), cx))
            .expect("request accepted");
        owner.read_with(cx, |owner, _| {
            assert_eq!(owner.operation(), None);
            assert_eq!(owner.resource_state().revision(), 0);
            assert!(owner.resource_state().committed().is_none());
        });
        cx.run_until_parked();
        let source = owner.read_with(cx, |owner, _| {
            assert_eq!(owner.operation(), None);
            assert!(owner.result().is_none());
            assert!(owner.resource_state().is_fresh());
            assert_eq!(owner.resource_state().revision(), 1);
            assert_eq!(owner.diagnostic(), None);
            owner
                .resource_state()
                .committed()
                .expect("resolved original PNG")
                .clone()
        });
        for operation in [CpuImageEdit::Invert, CpuImageEdit::MakeOpaque] {
            owner.update(cx, |owner, cx| owner.set_operation(Some(operation), cx));
            cx.run_until_parked();
            owner.read_with(cx, |owner, _| {
                assert!(owner.result_is_fresh());
                let result = owner.result().expect("selected original edit");
                let original =
                    graphics_core::resources::evaluate_cpu_image_edit(&source, operation, 4096)
                        .expect("actual original source result");
                assert_eq!(result.image(), original.image());
                assert_eq!(result.pixels_sha256(), original.pixels_sha256());
                assert_eq!(result.operation(), operation);
                assert_eq!(result.input_identity(), source.identity());
                assert_eq!(owner.resource_state().revision(), 1);
                assert_eq!(owner.diagnostic(), None);
            });
        }
        owner.update(cx, |owner, cx| owner.set_operation(None, cx));
        owner.read_with(cx, |owner, _| {
            assert_eq!(owner.operation(), None);
            assert!(owner.result().is_none());
            assert!(Arc::ptr_eq(
                owner.resource_state().committed().expect("source retained"),
                &source
            ));
            assert_eq!(owner.resource_state().revision(), 1);
        });
    }

    #[gpui::test]
    fn at_graphite_image_session_supersedes_before_dispatch_and_reuses_exact_source(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real service");
        let png = indexed_request_png(&fixture, 3).expect("real PNG");
        let owner = cx.new(|_| GraphiteImageSession::new(4096));
        owner
            .update(cx, |owner, cx| owner.load_png(load(&fixture, &png), cx))
            .expect("first request");
        let first_cancel = owner.read_with(cx, |owner, _| {
            owner
                .pending_load
                .as_ref()
                .expect("retained first task")
                .cancellation
                .clone()
        });
        let mut newer = load(&fixture, &png);
        newer.identity.version_id = "caller-supplied-session-version-2".into();
        let expected_identity = newer.identity.clone();
        owner
            .update(cx, |owner, cx| owner.load_png(newer, cx))
            .expect("superseding request");
        assert!(first_cancel.is_cancelled());
        cx.run_until_parked();
        let source = owner.read_with(cx, |owner, _| {
            assert_eq!(owner.resource_state().revision(), 1);
            let source = owner
                .resource_state()
                .committed()
                .expect("only newer completion accepted");
            assert_eq!(source.identity(), &expected_identity);
            source.clone()
        });
        let mut same = load(&fixture, &png);
        same.identity = expected_identity;
        owner
            .update(cx, |owner, cx| owner.load_png(same, cx))
            .expect("exact verified image reuse");
        owner.read_with(cx, |owner, _| {
            assert!(owner.pending_load.is_none());
            assert_eq!(owner.resource_state().revision(), 1);
            assert!(Arc::ptr_eq(
                owner
                    .resource_state()
                    .committed()
                    .expect("same authoritative image"),
                &source
            ));
        });
        owner.update(cx, |owner, cx| {
            owner.set_operation(Some(CpuImageEdit::Invert), cx)
        });
        let edit_cancel = owner.read_with(cx, |owner, _| {
            owner
                .pending_edit
                .as_ref()
                .expect("retained edit")
                .cancellation
                .clone()
        });
        owner.update(cx, |owner, cx| {
            owner.set_operation(Some(CpuImageEdit::MakeOpaque), cx)
        });
        assert!(edit_cancel.is_cancelled());
        cx.run_until_parked();
        owner.read_with(cx, |owner, _| {
            assert!(owner.result_is_fresh());
            assert_eq!(
                owner
                    .result()
                    .expect("only selected edit published")
                    .operation(),
                CpuImageEdit::MakeOpaque
            );
            assert_eq!(owner.resource_state().revision(), 1);
        });
    }

    #[gpui::test]
    fn at_graphite_image_session_read_failure_retains_last_valid_source_and_edit_then_recovers(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real service");
        let png = indexed_request_png(&fixture, 4).expect("real PNG");
        let owner = cx.new(|_| GraphiteImageSession::new(4096));
        owner
            .update(cx, |owner, cx| {
                owner.set_operation(Some(CpuImageEdit::Invert), cx);
                owner.load_png(load(&fixture, &png), cx)
            })
            .expect("load with explicitly selected edit");
        cx.run_until_parked();
        let (source, result) = owner.read_with(cx, |owner, _| {
            (
                owner
                    .resource_state()
                    .committed()
                    .expect("valid source")
                    .clone(),
                owner.result().expect("valid edit").clone(),
            )
        });
        let mut invalid = load(&fixture, &png);
        invalid.identity.content_sha256[0] ^= 1;
        owner
            .update(cx, |owner, cx| owner.load_png(invalid, cx))
            .expect("pending identity can be requested");
        cx.run_until_parked();
        owner.read_with(cx, |owner, _| {
            assert!(Arc::ptr_eq(owner.resource_state().committed().expect("retained source"), &source));
            assert!(Arc::ptr_eq(owner.result().expect("last-valid edit"), &result));
            assert_eq!(owner.resource_state().revision(), 1);
            assert!(!owner.resource_state().is_fresh());
            assert!(!owner.result_is_fresh());
            assert!(owner.diagnostic().is_some());
            assert!(matches!(owner.resource_state().diagnostic(), Some(crate::resources::ImageResourceError::Failed(message)) if message.contains("digest")));
        });
        owner
            .update(cx, |owner, cx| owner.load_png(load(&fixture, &png), cx))
            .expect("explicit recovery");
        cx.run_until_parked();
        owner.read_with(cx, |owner, _| {
            assert_eq!(owner.resource_state().revision(), 2);
            assert!(owner.resource_state().is_fresh());
            assert!(owner.result_is_fresh());
            assert_eq!(owner.diagnostic(), None);
            assert_eq!(
                owner.result().expect("recovered edit").image(),
                result.image()
            );
            assert!(!Arc::ptr_eq(
                owner.result().expect("new parent has new result"),
                &result
            ));
        });
    }

    #[gpui::test]
    fn at_graphite_image_session_edit_failure_retains_last_valid_result_and_recovers(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real service");
        let png = indexed_request_png(&fixture, 4).expect("real PNG");
        let owner = cx.new(|_| GraphiteImageSession::new(4096));
        owner
            .update(cx, |owner, cx| {
                owner.set_operation(Some(CpuImageEdit::Invert), cx);
                owner.load_png(load(&fixture, &png), cx)
            })
            .expect("load");
        cx.run_until_parked();
        let result = owner.read_with(cx, |owner, _| owner.result().expect("valid edit").clone());
        owner.update(cx, |owner, cx| {
            owner.max_result_bytes = 0;
            owner.set_operation(Some(CpuImageEdit::MakeOpaque), cx);
        });
        cx.run_until_parked();
        owner.read_with(cx, |owner, _| {
            assert!(Arc::ptr_eq(
                owner.result().expect("failed edit retains last valid"),
                &result
            ));
            assert!(!owner.result_is_fresh());
            assert_eq!(owner.operation(), Some(CpuImageEdit::MakeOpaque));
            assert!(matches!(
                owner.diagnostic(),
                Some(ImageSessionDiagnostic::Evaluation(_))
            ));
            assert!(owner.resource_state().is_fresh());
            assert_eq!(owner.resource_state().revision(), 1);
        });
        owner.update(cx, |owner, cx| {
            owner.max_result_bytes = 4096;
            owner.set_operation(Some(CpuImageEdit::MakeOpaque), cx);
        });
        cx.run_until_parked();
        owner.read_with(cx, |owner, _| {
            assert!(owner.result_is_fresh());
            assert_eq!(
                owner
                    .result()
                    .expect("recovered source operation")
                    .operation(),
                CpuImageEdit::MakeOpaque
            );
            assert_eq!(owner.diagnostic(), None);
            assert_eq!(owner.resource_state().revision(), 1);
        });
    }

    #[gpui::test]
    fn at_graphite_image_session_cancel_retains_and_revoke_hides_until_valid_completion(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real service");
        let png = indexed_request_png(&fixture, 4).expect("real PNG");
        let owner = cx.new(|_| GraphiteImageSession::new(4096));
        owner
            .update(cx, |owner, cx| {
                owner.set_operation(Some(CpuImageEdit::Invert), cx);
                owner.load_png(load(&fixture, &png), cx)
            })
            .expect("load");
        cx.run_until_parked();
        let result = owner.read_with(cx, |owner, _| owner.result().expect("valid edit").clone());
        let mut next = load(&fixture, &png);
        next.identity.version_id = "caller-supplied-pending-version".into();
        owner
            .update(cx, |owner, cx| owner.load_png(next, cx))
            .expect("pending refresh");
        let cancellation = owner.read_with(cx, |owner, _| {
            owner
                .pending_load
                .as_ref()
                .expect("pending task")
                .cancellation
                .clone()
        });
        owner.update(cx, |owner, cx| owner.cancel(cx));
        assert!(cancellation.is_cancelled());
        cx.run_until_parked();
        owner.read_with(cx, |owner, _| {
            assert!(owner.resource_state().active_request().is_none());
            assert_eq!(owner.resource_state().revision(), 1);
            assert!(!owner.result_is_fresh());
            assert!(Arc::ptr_eq(
                owner.result().expect("last valid after cancellation"),
                &result
            ));
            assert_eq!(
                owner.diagnostic(),
                Some(&ImageSessionDiagnostic::Resource(
                    crate::resources::ImageResourceError::Cancelled
                ))
            );
        });
        owner
            .update(cx, |owner, cx| owner.revoke(&png.image_identity, cx))
            .expect("actual committed identity revocation");
        owner.read_with(cx, |owner, _| {
            assert!(owner.resource_state().is_revoked());
            assert!(owner.result().is_none());
            assert!(!owner.result_is_fresh());
            assert_eq!(owner.resource_state().revision(), 1);
        });
        owner
            .update(cx, |owner, cx| owner.load_png(load(&fixture, &png), cx))
            .expect("explicit fresh validation after revoke");
        cx.run_until_parked();
        owner.read_with(cx, |owner, _| {
            assert!(!owner.resource_state().is_revoked());
            assert!(owner.result_is_fresh());
            assert_eq!(owner.resource_state().revision(), 2);
            assert_eq!(owner.diagnostic(), None);
        });
    }

    #[gpui::test]
    fn at_graphite_image_session_release_cancels_retained_load_and_edit_tasks(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real service");
        let png = indexed_request_png(&fixture, 4).expect("real PNG");
        let owner = cx.new(|_| GraphiteImageSession::new(4096));
        owner
            .update(cx, |owner, cx| owner.load_png(load(&fixture, &png), cx))
            .expect("pending load");
        let cancellation = owner.read_with(cx, |owner, _| {
            owner
                .pending_load
                .as_ref()
                .expect("retained load")
                .cancellation
                .clone()
        });
        let weak = owner.downgrade();
        drop(owner);
        cx.update(|_| {});
        cx.run_until_parked();
        assert!(cancellation.is_cancelled());
        assert!(weak.upgrade().is_none());
        let owner = cx.new(|_| GraphiteImageSession::new(4096));
        owner
            .update(cx, |owner, cx| owner.load_png(load(&fixture, &png), cx))
            .expect("source load");
        cx.run_until_parked();
        owner.update(cx, |owner, cx| {
            owner.set_operation(Some(CpuImageEdit::Invert), cx)
        });
        let cancellation = owner.read_with(cx, |owner, _| {
            owner
                .pending_edit
                .as_ref()
                .expect("retained edit")
                .cancellation
                .clone()
        });
        let weak = owner.downgrade();
        drop(owner);
        cx.update(|_| {});
        cx.run_until_parked();
        assert!(cancellation.is_cancelled());
        assert!(weak.upgrade().is_none());
    }

    fn original_scene_configuration() -> graphics_core::render::CpuSvgSceneConfiguration {
        graphics_core::render::CpuSvgSceneConfiguration {
            viewport: graphite_core_types::transform::Footprint {
                transform: graphite_glam::DAffine2::IDENTITY,
                resolution: graphite_glam::UVec2::new(16, 8),
                quality: graphite_core_types::transform::Footprint::DEFAULT.quality,
            },
            scale: 1.,
            max_svg_bytes: 32 * 1024,
        }
    }

    #[gpui::test]
    async fn at_graphite_original_scene_retains_authenticated_source_without_edit_provenance(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real local indexed service");
        let png = indexed_request_png(&fixture, 4).expect("actual PNG producer");
        let owner = cx.new(|_| GraphiteImageSession::new(4096));
        let mut binding = load(&fixture, &png);
        binding.descriptor.transform = [3., 0., 0., 2., 4., 5.];
        owner
            .update(cx, |owner, cx| owner.load_png(binding, cx))
            .expect("original request");
        cx.run_until_parked();
        let source = owner.read_with(cx, |owner, _| {
            assert_eq!(owner.operation(), None);
            assert!(owner.result().is_none());
            assert!(!owner.result_is_fresh());
            owner
                .resource_state()
                .committed()
                .expect("authenticated source")
                .clone()
        });
        let before = (
            source.image().clone(),
            *source.descriptor(),
            source.identity().clone(),
            source.origin(),
        );
        let revision = owner.read_with(cx, |owner, _| owner.resource_state().revision());
        let frame = owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(original_scene_configuration(), 1., 2048, cx)
            })
            .expect("fresh Original scene")
            .await
            .expect("actual source scene conversion");
        assert!(frame.result.is_none());
        assert!(Arc::ptr_eq(&frame.source, &source));
        assert_eq!(frame.source_revision, revision);
        match frame.scene.layers() {
            [graphics_core::render::CpuSvgSceneLayer::SourceImage(retained)] => {
                assert!(Arc::ptr_eq(retained, &source))
            }
            _ => panic!("Original was relabeled as an edit"),
        }
        let original = graphics_core::render::prepare_cpu_svg_composition(
            &[graphics_core::render::CpuSvgSceneLayer::SourceImage(
                source.clone(),
            )],
            original_scene_configuration(),
        )
        .expect("original source scene oracle");
        assert_eq!(frame.scene.svg().svg, original.svg().svg);
        assert!(frame.scene.svg().svg.contains("data:image/png;base64,"));
        assert!(frame.scene.svg().svg.contains("matrix(3,0,0,2,4,5)"));
        let original_frame = cx
            .update(|cx| cx.svg_renderer())
            .render_single_frame(original.svg().svg.as_bytes(), 1.)
            .expect("existing native renderer oracle");
        assert_eq!(frame.image.as_bytes(0), original_frame.as_bytes(0));
        assert!(
            frame
                .image
                .as_bytes(0)
                .expect("source scene frame")
                .as_chunks::<4>()
                .0
                .iter()
                .any(|pixel| pixel[3] > 0)
        );
        assert_eq!(
            (frame.image.size(0).width.0, frame.image.size(0).height.0),
            (32, 16)
        );
        owner.read_with(cx, |owner, _| {
            assert!(owner.scene_preview_is_current(&frame));
            assert_eq!(owner.operation(), None);
            assert!(owner.result().is_none());
            assert!(!owner.result_is_fresh());
            assert_eq!(owner.resource_state().revision(), revision);
        });
        assert_eq!(
            (
                source.image().clone(),
                *source.descriptor(),
                source.identity().clone(),
                source.origin()
            ),
            before
        );
    }

    #[gpui::test]
    async fn at_graphite_original_scene_supersession_and_edit_original_transitions_are_guarded(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real service");
        let png = indexed_request_png(&fixture, 4).expect("real PNG");
        let owner = cx.new(|_| GraphiteImageSession::new(4096));
        owner
            .update(cx, |owner, cx| owner.load_png(load(&fixture, &png), cx))
            .expect("source request");
        cx.run_until_parked();
        let source = owner.read_with(cx, |owner, _| {
            owner.resource_state().committed().expect("source").clone()
        });
        let first = owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(original_scene_configuration(), 1., 2048, cx)
            })
            .expect("first Original job");
        let second = owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(original_scene_configuration(), 0.5, 512, cx)
            })
            .expect("successor Original job");
        assert!(first.await.is_err());
        let original = second.await.expect("successor Original preview");
        assert!(owner.read_with(cx, |owner, _| owner.scene_preview_is_current(&original)));
        owner.update(cx, |owner, cx| {
            owner.set_operation(Some(CpuImageEdit::Invert), cx)
        });
        assert!(!owner.read_with(cx, |owner, _| owner.scene_preview_is_current(&original)));
        cx.run_until_parked();
        let edited = owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(original_scene_configuration(), 1., 2048, cx)
            })
            .expect("fresh edited job")
            .await
            .expect("edited scene");
        owner.read_with(cx, |owner, _| {
            assert!(owner.scene_preview_is_current(&edited));
            assert!(Arc::ptr_eq(
                owner.result().expect("selected edit"),
                edited.result.as_ref().expect("edited receipt")
            ));
        });
        owner.update(cx, |owner, cx| owner.set_operation(None, cx));
        assert!(!owner.read_with(cx, |owner, _| owner.scene_preview_is_current(&edited)));
        let restored = owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(original_scene_configuration(), 1., 2048, cx)
            })
            .expect("restored Original job")
            .await
            .expect("restored source scene");
        assert!(restored.result.is_none());
        assert!(Arc::ptr_eq(&restored.source, &source));
        assert!(owner.read_with(cx, |owner, _| owner.scene_preview_is_current(&restored)));
        let mut reload = load(&fixture, &png);
        reload.identity.version_id = "explicit-successor-fixture-version".into();
        owner
            .update(cx, |owner, cx| owner.load_png(reload, cx))
            .expect("successor version request");
        assert!(!owner.read_with(cx, |owner, _| owner.scene_preview_is_current(&restored)));
        assert!(
            owner
                .update(cx, |owner, cx| owner.prepare_scene_preview(
                    original_scene_configuration(),
                    1.,
                    2048,
                    cx
                ))
                .is_err()
        );
        cx.run_until_parked();
        assert!(!owner.read_with(cx, |owner, _| owner.scene_preview_is_current(&restored)));
    }

    #[gpui::test]
    async fn at_graphite_original_scene_cancel_revoke_and_release_prevent_publication(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real service");
        let png = indexed_request_png(&fixture, 4).expect("real PNG");
        let owner = cx.new(|_| GraphiteImageSession::new(4096));
        owner
            .update(cx, |owner, cx| owner.load_png(load(&fixture, &png), cx))
            .expect("source request");
        cx.run_until_parked();
        let pending = owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(original_scene_configuration(), 1., 2048, cx)
            })
            .expect("pending scene");
        let token = owner.read_with(cx, |owner, _| {
            owner
                .pending_scene
                .as_ref()
                .expect("tracked scene token")
                .clone()
        });
        owner.update(cx, |owner, cx| owner.cancel(cx));
        assert!(token.is_cancelled());
        assert!(pending.await.is_err());
        let valid = owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(original_scene_configuration(), 1., 2048, cx)
            })
            .expect("cancel recovery")
            .await
            .expect("current Original scene");
        owner
            .update(cx, |owner, cx| owner.revoke(&png.image_identity, cx))
            .expect("source revocation");
        assert!(!owner.read_with(cx, |owner, _| owner.scene_preview_is_current(&valid)));
        assert!(
            owner
                .update(cx, |owner, cx| owner.prepare_scene_preview(
                    original_scene_configuration(),
                    1.,
                    2048,
                    cx
                ))
                .is_err()
        );
        owner
            .update(cx, |owner, cx| owner.load_png(load(&fixture, &png), cx))
            .expect("explicit validated refresh");
        cx.run_until_parked();
        let pending = owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(original_scene_configuration(), 1., 2048, cx)
            })
            .expect("release pending scene");
        let token = owner.read_with(cx, |owner, _| {
            owner.pending_scene.as_ref().expect("release token").clone()
        });
        let weak = owner.downgrade();
        drop(owner);
        cx.update(|_| {});
        assert!(token.is_cancelled());
        assert!(weak.upgrade().is_none());
        assert!(pending.await.is_err());
    }

    #[gpui::test]
    async fn at_graphite_original_scene_cap_failures_recover_without_mutating_source(
        cx: &mut TestAppContext,
    ) {
        let fixture = host_reader_fixture().expect("real service");
        let png = indexed_request_png(&fixture, 4).expect("real PNG");
        let owner = cx.new(|_| GraphiteImageSession::new(4096));
        assert!(
            owner
                .update(cx, |owner, cx| owner.prepare_scene_preview(
                    original_scene_configuration(),
                    1.,
                    2048,
                    cx
                ))
                .is_err()
        );
        owner
            .update(cx, |owner, cx| owner.load_png(load(&fixture, &png), cx))
            .expect("source request");
        cx.run_until_parked();
        let source = owner.read_with(cx, |owner, _| {
            owner.resource_state().committed().expect("source").clone()
        });
        let pixels = source.image().clone();
        let descriptor = *source.descriptor();
        let identity = source.identity().clone();
        let first = owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(original_scene_configuration(), 1., 2048, cx)
            })
            .expect("first source preview")
            .await
            .expect("first source frame");
        for maximum in [0, 2047] {
            let failed = owner
                .update(cx, |owner, cx| {
                    owner.prepare_scene_preview(original_scene_configuration(), 1., maximum, cx)
                })
                .expect("bounded source scene request");
            assert!(!owner.read_with(cx, |owner, _| owner.scene_preview_is_current(&first)));
            assert!(
                failed
                    .await
                    .err()
                    .expect("preview cap failure")
                    .contains("exceeding")
            );
            owner.read_with(cx, |owner, _| {
                assert!(Arc::ptr_eq(
                    owner.resource_state().committed().expect("retained source"),
                    &source
                ));
                assert!(owner.resource_state().is_fresh());
                assert_eq!(owner.operation(), None);
                assert!(owner.result().is_none());
                assert_eq!(owner.resource_state().revision(), first.source_revision);
            });
        }
        let mut too_small = original_scene_configuration();
        too_small.max_svg_bytes = 0;
        assert!(
            owner
                .update(cx, |owner, cx| owner
                    .prepare_scene_preview(too_small, 1., 2048, cx))
                .expect("source SVG cap request")
                .await
                .is_err()
        );
        let recovered = owner
            .update(cx, |owner, cx| {
                owner.prepare_scene_preview(original_scene_configuration(), 1., 2048, cx)
            })
            .expect("cap recovery")
            .await
            .expect("source scene recovered");
        assert!(recovered.result.is_none());
        assert!(owner.read_with(cx, |owner, _| owner.scene_preview_is_current(&recovered)));
        assert_eq!(recovered.image.as_bytes(0), first.image.as_bytes(0));
        assert_eq!(source.image(), &pixels);
        assert_eq!(source.descriptor(), &descriptor);
        assert_eq!(source.identity(), &identity);
    }
}

#[derive(Clone, Debug)]
pub enum ArtboardSessionError {
    Document(graphics_core::evaluate::CpuSvgDocumentError),
    Acceptance(std::sync::Arc<graphics_core::evaluate::CpuSvgAcceptanceError>),
    Conversion(std::sync::Arc<crate::render::CpuSvgPreviewError>),
    PngPreparation(std::sync::Arc<str>),
    Cancelled,
    Superseded,
    Released,
    GenerationExhausted,
    ResourceRevoked,
    MissingEncodedPngInput,
    GestureAlreadyActive,
    InvalidGesture,
}

impl std::fmt::Display for ArtboardSessionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Document(error) => std::fmt::Display::fmt(error, formatter),
            Self::Acceptance(error) => std::fmt::Display::fmt(error.as_ref(), formatter),
            Self::Conversion(error) => std::fmt::Display::fmt(error.as_ref(), formatter),
            Self::PngPreparation(error) => {
                write!(formatter, "Artboard PNG preparation failed: {error}")
            }
            Self::Cancelled => formatter.write_str("Artboard preview was cancelled"),
            Self::Superseded => formatter.write_str("Artboard preview was superseded"),
            Self::Released => formatter.write_str("Artboard preview owner was released"),
            Self::GenerationExhausted => {
                formatter.write_str("Artboard preview generation is exhausted")
            }
            Self::ResourceRevoked => {
                formatter.write_str("Artboard resource authorization was revoked")
            }
            Self::MissingEncodedPngInput => {
                formatter.write_str("The Artboard has no encoded image to place")
            }
            Self::GestureAlreadyActive => {
                formatter.write_str("Apply or discard the current Artboard edits first")
            }
            Self::InvalidGesture => {
                formatter.write_str("The Artboard edit gesture is no longer owned by this session")
            }
        }
    }
}

impl std::error::Error for ArtboardSessionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Document(error) => Some(error),
            Self::Acceptance(error) => Some(error.as_ref()),
            Self::Conversion(error) => Some(error.as_ref()),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ArtboardSessionEvent {
    Changed,
}

struct ArtboardRequestAuthority {
    generation: u64,
    cancellation: std::sync::Arc<sim_tensor::CancellationToken>,
}

#[derive(Clone)]
pub struct PreparedArtboardPreview {
    image: std::sync::Arc<gpui::RenderImage>,
    output: graphics_core::evaluate::AcceptedCpuSvgOutput,
    authority: std::sync::Arc<ArtboardRequestAuthority>,
}

impl PreparedArtboardPreview {
    pub(crate) fn image(&self) -> &std::sync::Arc<gpui::RenderImage> {
        &self.image
    }

    pub(crate) fn output(&self) -> &graphics_core::evaluate::AcceptedCpuSvgOutput {
        &self.output
    }

    pub(crate) fn generation(&self) -> u64 {
        self.authority.generation
    }
}

struct PendingArtboardPreview {
    authority: std::sync::Arc<ArtboardRequestAuthority>,
    task: gpui::Task<()>,
}

struct ArtboardReceiptCancellation {
    cancellation: std::sync::Arc<sim_tensor::CancellationToken>,
    completed: bool,
}

impl Drop for ArtboardReceiptCancellation {
    fn drop(&mut self) {
        if !self.completed {
            self.cancellation.cancel();
        }
    }
}

#[derive(Clone)]
pub struct PreparedArtboardPngAsset {
    preview: PreparedArtboardPreview,
    encoded: graphics_core::export::PreparedRgba8Png,
    cancellation: std::sync::Arc<sim_tensor::CancellationToken>,
}

impl PreparedArtboardPngAsset {
    pub fn source_output(&self) -> &graphics_core::evaluate::AcceptedCpuSvgOutput {
        self.preview.output()
    }

    pub fn png(&self) -> &graphics_core::export::PreparedPngImage {
        self.encoded.png()
    }

    pub fn pixels_sha256(&self) -> [u8; 32] {
        self.encoded.pixels_sha256()
    }
}

#[derive(Clone)]
pub struct ArtboardGesture {
    identity: std::sync::Arc<()>,
}

struct PendingArtboardGesture {
    token: ArtboardGesture,
    expected_revision: u64,
    candidate: graphics_core::evaluate::CpuSvgArtboardDocument,
}

pub struct GraphiteArtboardSession {
    document: graphics_core::evaluate::CpuSvgArtboardDocument,
    gesture: Option<PendingArtboardGesture>,
    png_cancellation: Option<std::sync::Arc<sim_tensor::CancellationToken>>,
    authority: Option<std::sync::Arc<ArtboardRequestAuthority>>,
    pending: Option<PendingArtboardPreview>,
    preview: Option<PreparedArtboardPreview>,
    diagnostic: Option<ArtboardSessionError>,
    generation: u64,
    resource_revoked: bool,
}

impl gpui::EventEmitter<ArtboardSessionEvent> for GraphiteArtboardSession {}

impl Drop for GraphiteArtboardSession {
    fn drop(&mut self) {
        self.cancel_owned_request();
    }
}

impl GraphiteArtboardSession {
    pub fn new(document: graphics_core::evaluate::CpuSvgArtboardDocument) -> Self {
        Self {
            document,
            gesture: None,
            png_cancellation: None,
            authority: None,
            pending: None,
            preview: None,
            diagnostic: None,
            generation: 0,
            resource_revoked: false,
        }
    }

    pub fn document(&self) -> &graphics_core::evaluate::CpuSvgArtboardDocument {
        &self.document
    }

    pub fn configuration(&self) -> graphite_std::application_io::RenderConfig {
        self.document.configuration()
    }

    pub fn parameters(&self) -> graphics_core::evaluate::CpuSvgArtboardParameters {
        self.document.parameters()
    }

    pub fn revision(&self) -> u64 {
        self.document.revision()
    }

    pub fn gesture(&self) -> Option<ArtboardGesture> {
        self.gesture.as_ref().map(|gesture| gesture.token.clone())
    }

    pub fn authoring_parameters(&self) -> graphics_core::evaluate::CpuSvgArtboardParameters {
        self.gesture.as_ref().map_or_else(
            || self.parameters(),
            |gesture| gesture.candidate.parameters(),
        )
    }

    pub fn authoring_image_placement(
        &self,
    ) -> Option<graphite_graph_craft::document::value::DAffine2> {
        self.gesture.as_ref().map_or_else(
            || self.image_placement(),
            |gesture| {
                gesture
                    .candidate
                    .encoded_png_input()
                    .map(|input| input.placement())
            },
        )
    }

    pub fn begin_gesture(
        &mut self,
        expected_revision: u64,
        cx: &mut gpui::Context<Self>,
    ) -> Result<ArtboardGesture, ArtboardSessionError> {
        if expected_revision != self.revision() {
            return Err(ArtboardSessionError::Document(
                graphics_core::evaluate::CpuSvgDocumentError::StaleRevision,
            ));
        }
        if self.gesture.is_some() {
            return Err(ArtboardSessionError::GestureAlreadyActive);
        }
        let token = ArtboardGesture {
            identity: std::sync::Arc::new(()),
        };
        self.gesture = Some(PendingArtboardGesture {
            token: token.clone(),
            expected_revision,
            candidate: self.document.detach_for_edit(),
        });
        self.changed(cx);
        Ok(token)
    }

    fn owned_gesture(
        &self,
        token: &ArtboardGesture,
    ) -> Result<&PendingArtboardGesture, ArtboardSessionError> {
        self.gesture
            .as_ref()
            .filter(|gesture| std::sync::Arc::ptr_eq(&gesture.token.identity, &token.identity))
            .ok_or(ArtboardSessionError::InvalidGesture)
    }

    pub fn stage_gesture(
        &mut self,
        token: &ArtboardGesture,
        parameters: graphics_core::evaluate::CpuSvgArtboardParameters,
        placement: Option<graphite_graph_craft::document::value::DAffine2>,
        cx: &mut gpui::Context<Self>,
    ) -> Result<bool, ArtboardSessionError> {
        let gesture = self.owned_gesture(token)?;
        if gesture.expected_revision != self.revision() {
            return Err(ArtboardSessionError::Document(
                graphics_core::evaluate::CpuSvgDocumentError::StaleRevision,
            ));
        }
        let input = match (gesture.candidate.encoded_png_input(), placement) {
            (Some(input), Some(placement)) => Some(
                graphics_core::evaluate::CpuSvgEncodedPngInput::new(
                    input.source().clone(),
                    placement,
                )
                .map_err(ArtboardSessionError::Document)?,
            ),
            (None, None) => None,
            _ => return Err(ArtboardSessionError::MissingEncodedPngInput),
        };
        let gesture = self
            .gesture
            .as_mut()
            .ok_or(ArtboardSessionError::InvalidGesture)?;
        let candidate = &mut gesture.candidate;
        let changed = match input {
            Some(input) => candidate.apply_encoded_png(
                candidate.revision(),
                parameters,
                candidate.configuration(),
                input,
            ),
            None => candidate.apply(candidate.revision(), parameters, candidate.configuration()),
        }
        .map_err(ArtboardSessionError::Document)?;
        if changed {
            self.changed(cx);
        }
        Ok(changed)
    }

    pub fn cancel_gesture(
        &mut self,
        token: &ArtboardGesture,
        cx: &mut gpui::Context<Self>,
    ) -> Result<(), ArtboardSessionError> {
        self.owned_gesture(token)?;
        self.gesture = None;
        self.changed(cx);
        Ok(())
    }

    pub fn commit_gesture(
        &mut self,
        token: &ArtboardGesture,
        cx: &mut gpui::Context<Self>,
    ) -> Result<bool, ArtboardSessionError> {
        let gesture = self.owned_gesture(token)?;
        let expected_revision = gesture.expected_revision;
        let parameters = gesture.candidate.parameters();
        let configuration = gesture.candidate.configuration();
        let input = gesture.candidate.encoded_png_input().cloned();
        let changed = match input {
            Some(input) => {
                self.document
                    .apply_encoded_png(expected_revision, parameters, configuration, input)
            }
            None => self
                .document
                .apply(expected_revision, parameters, configuration),
        }
        .map_err(ArtboardSessionError::Document)?;
        self.gesture = None;
        if changed {
            self.cancel_owned_request();
            if !self.resource_revoked {
                self.diagnostic = None;
            }
        }
        self.changed(cx);
        Ok(changed)
    }

    pub fn current_output(&self) -> Option<&graphics_core::evaluate::AcceptedCpuSvgOutput> {
        if self.resource_revoked {
            return None;
        }
        self.document
            .accepted()
            .filter(|output| self.document.owns_current_output(output))
    }

    pub fn diagnostic(&self) -> Option<&ArtboardSessionError> {
        self.diagnostic.as_ref()
    }

    pub fn resource_is_revoked(&self) -> bool {
        self.resource_revoked
    }

    pub(crate) fn revoke_resource(&mut self, cx: &mut gpui::Context<Self>) {
        if self.resource_revoked {
            return;
        }
        self.resource_revoked = true;
        self.cancel_owned_request();
        self.preview = None;
        self.diagnostic = Some(ArtboardSessionError::ResourceRevoked);
        self.changed(cx);
    }

    pub fn is_pending(&self) -> bool {
        self.pending
            .as_ref()
            .is_some_and(|pending| !pending.authority.cancellation.is_cancelled())
    }

    pub fn preview(&self) -> Option<&PreparedArtboardPreview> {
        self.preview.as_ref()
    }

    pub fn preview_is_current(&self, preview: &PreparedArtboardPreview) -> bool {
        self.request_is_current(&preview.authority)
            && self.document.owns_current_output(&preview.output)
            && self
                .preview
                .as_ref()
                .is_some_and(|current| std::sync::Arc::ptr_eq(&current.image, &preview.image))
    }

    pub fn apply(
        &mut self,
        expected_revision: u64,
        parameters: graphics_core::evaluate::CpuSvgArtboardParameters,
        configuration: graphite_std::application_io::RenderConfig,
        cx: &mut gpui::Context<Self>,
    ) -> Result<bool, graphics_core::evaluate::CpuSvgDocumentError> {
        let changed = self
            .document
            .apply(expected_revision, parameters, configuration)?;
        if changed {
            self.cancel_owned_request();
            // Keep editable metadata after permission loss without reviving evaluation authority.
            if !self.resource_revoked {
                self.diagnostic = None;
            }
            self.changed(cx);
        }
        Ok(changed)
    }

    pub fn image_placement(&self) -> Option<graphite_graph_craft::document::value::DAffine2> {
        self.document
            .encoded_png_input()
            .map(|input| input.placement())
    }

    pub(crate) fn replace_encoded_source(
        &mut self,
        expected_revision: u64,
        source: std::sync::Arc<graphics_core::resources::PreparedEncodedPngSource>,
        cx: &mut gpui::Context<Self>,
    ) -> Result<bool, ArtboardSessionError> {
        let placement = self
            .image_placement()
            .ok_or(ArtboardSessionError::MissingEncodedPngInput)?;
        let input = graphics_core::evaluate::CpuSvgEncodedPngInput::new(source, placement)
            .map_err(ArtboardSessionError::Document)?;
        let changed = self
            .document
            .apply_encoded_png(
                expected_revision,
                self.parameters(),
                self.configuration(),
                input,
            )
            .map_err(ArtboardSessionError::Document)?;
        if changed {
            self.cancel_owned_request();
            if !self.resource_revoked {
                self.diagnostic = None;
            }
            self.changed(cx);
        }
        Ok(changed)
    }

    pub fn apply_image_placement(
        &mut self,
        expected_revision: u64,
        placement: graphite_graph_craft::document::value::DAffine2,
        cx: &mut gpui::Context<Self>,
    ) -> Result<bool, ArtboardSessionError> {
        let source = self
            .document
            .encoded_png_input()
            .ok_or(ArtboardSessionError::MissingEncodedPngInput)?
            .source()
            .clone();
        let input = graphics_core::evaluate::CpuSvgEncodedPngInput::new(source, placement)
            .map_err(ArtboardSessionError::Document)?;
        let changed = self
            .document
            .apply_encoded_png(
                expected_revision,
                self.parameters(),
                self.configuration(),
                input,
            )
            .map_err(ArtboardSessionError::Document)?;
        if changed {
            self.cancel_owned_request();
            // Authoring retained work does not restore revoked resource authority.
            if !self.resource_revoked {
                self.diagnostic = None;
            }
            self.changed(cx);
        }
        Ok(changed)
    }

    /// The session retains the pipeline. Retiring a pending receipt future
    /// cancels the same request token; dropping a completed Task does not.
    /// Cancellation does not interrupt the original synchronous compiler or
    /// native renderer while they are running.
    pub fn request_preview(
        &mut self,
        max_preview_bytes: usize,
        cx: &mut gpui::Context<Self>,
    ) -> Result<
        gpui::Task<Result<PreparedArtboardPreview, ArtboardSessionError>>,
        ArtboardSessionError,
    > {
        if self.resource_revoked {
            return Err(ArtboardSessionError::ResourceRevoked);
        }
        let generation = self
            .generation
            .checked_add(1)
            .ok_or(ArtboardSessionError::GenerationExhausted)?;
        self.cancel_owned_request();
        self.generation = generation;
        let authority = std::sync::Arc::new(ArtboardRequestAuthority {
            generation,
            cancellation: std::sync::Arc::new(sim_tensor::CancellationToken::default()),
        });
        self.authority = Some(authority.clone());
        self.diagnostic = None;
        let evaluation = self.document.submit();
        let background = cx.background_spawn({
            let authority = authority.clone();
            async move {
                if authority.cancellation.is_cancelled() {
                    return Err(ArtboardSessionError::Cancelled);
                }
                let completion = futures::executor::block_on(evaluation.evaluate());
                if authority.cancellation.is_cancelled() {
                    return Err(ArtboardSessionError::Cancelled);
                }
                Ok(completion)
            }
        });
        let renderer = cx.svg_renderer();
        let (sender, receiver) = futures::channel::oneshot::channel();
        let task = cx.spawn({
            let authority = authority.clone();
            async move |session, cx| {
                let evaluated = background.await;
                let accepted = match evaluated {
                    Ok(completion) => session
                        .update(cx, |session, _cx| {
                            session.check_request(&authority)?;
                            session
                                .document
                                .accept(completion)
                                .cloned()
                                .map_err(|error| {
                                    ArtboardSessionError::Acceptance(std::sync::Arc::new(error))
                                })
                        })
                        .map_err(|_released| ArtboardSessionError::Released)
                        .and_then(|result| result),
                    Err(error) => Err(error),
                };
                let outcome = match accepted {
                    Ok(output) => {
                        let conversion = cx.background_spawn({
                            let output = output.clone();
                            let authority = authority.clone();
                            async move {
                                if authority.cancellation.is_cancelled() {
                                    return Err(ArtboardSessionError::Cancelled);
                                }
                                let image =
                                    crate::render::prepare_accepted_cpu_svg_preview_at_target(
                                        &output,
                                        &renderer,
                                        max_preview_bytes,
                                    )
                                    .map_err(|error| {
                                        ArtboardSessionError::Conversion(std::sync::Arc::new(error))
                                    })?;
                                if authority.cancellation.is_cancelled() {
                                    return Err(ArtboardSessionError::Cancelled);
                                }
                                Ok(image)
                            }
                        });
                        conversion.await.map(|image| PreparedArtboardPreview {
                            image,
                            output,
                            authority: authority.clone(),
                        })
                    }
                    Err(error) => Err(error),
                };
                let outcome =
                    session
                        .update(cx, |session, cx| {
                            if session.resource_revoked {
                                return Err(ArtboardSessionError::ResourceRevoked);
                            }
                            if !session.owns_request(&authority) {
                                return Err(ArtboardSessionError::Superseded);
                            }
                            let outcome = session.check_request(&authority).and(outcome).and_then(
                                |preview| {
                                    if session.document.owns_current_output(&preview.output) {
                                        Ok(preview)
                                    } else {
                                        Err(ArtboardSessionError::Superseded)
                                    }
                                },
                            );
                            if let Some(pending) = session.pending.take() {
                                pending.task.detach();
                            }
                            if outcome.is_err() {
                                session.document.cancel();
                            }
                            session.diagnostic = outcome.as_ref().err().cloned();
                            if let Ok(preview) = &outcome {
                                session.preview = Some(preview.clone());
                            }
                            session.changed(cx);
                            outcome
                        })
                        .map_err(|_released| ArtboardSessionError::Released)
                        .and_then(|result| result);
                if sender.send(outcome).is_err() {
                    authority.cancellation.cancel();
                    match session.update(cx, |session, cx| {
                        if session.owns_request(&authority) {
                            session.document.cancel();
                            session.diagnostic = Some(ArtboardSessionError::Cancelled);
                            session.changed(cx);
                        }
                    }) {
                        Ok(()) => {}
                        Err(error) => log::debug!("Artboard receipt owner released: {error}"),
                    }
                }
            }
        });
        self.pending = Some(PendingArtboardPreview {
            authority: authority.clone(),
            task,
        });
        self.changed(cx);
        let receipt = ArtboardReceiptCancellation {
            cancellation: authority.cancellation.clone(),
            completed: false,
        };
        Ok(cx.spawn(async move |_session, _cx| {
            let mut receipt = receipt;
            let outcome = receiver
                .await
                .map_err(|_closed| {
                    if receipt.cancellation.is_cancelled() {
                        ArtboardSessionError::Cancelled
                    } else {
                        ArtboardSessionError::Released
                    }
                })
                .and_then(|result| result);
            receipt.completed = true;
            outcome
        }))
    }

    pub fn png_is_current(&self, prepared: &PreparedArtboardPngAsset) -> bool {
        self.owns_png_request(&prepared.cancellation)
            && !prepared.cancellation.is_cancelled()
            && self.preview_is_current(&prepared.preview)
    }

    /// Prepares the captured finite CPU frame through the shared PNG encoder.
    /// The returned task must be retained or awaited; pending task destruction
    /// cancels this PNG request without cancelling its source preview.
    /// This receipt does not allocate or publish a project asset version.
    pub fn prepare_png(
        &mut self,
        preview: PreparedArtboardPreview,
        max_pixel_bytes: usize,
        max_encoded_bytes: usize,
        cx: &mut gpui::Context<Self>,
    ) -> Result<
        gpui::Task<Result<PreparedArtboardPngAsset, ArtboardSessionError>>,
        ArtboardSessionError,
    > {
        if self.resource_revoked {
            return Err(ArtboardSessionError::ResourceRevoked);
        }
        if !self.preview_is_current(&preview) {
            return Err(ArtboardSessionError::Superseded);
        }
        self.cancel_png_request();
        let cancellation = std::sync::Arc::new(sim_tensor::CancellationToken::default());
        self.png_cancellation = Some(cancellation.clone());
        self.diagnostic = None;
        let background = cx.background_spawn({
            let preview = preview.clone();
            let cancellation = cancellation.clone();
            async move {
                if cancellation.is_cancelled() || preview.authority.cancellation.is_cancelled() {
                    return Err(ArtboardSessionError::Cancelled);
                }
                let encoded = crate::render::prepare_artboard_png(
                    &preview,
                    max_pixel_bytes,
                    max_encoded_bytes,
                )
                .map_err(|error| ArtboardSessionError::PngPreparation(error.into()))?;
                if cancellation.is_cancelled() || preview.authority.cancellation.is_cancelled() {
                    return Err(ArtboardSessionError::Cancelled);
                }
                Ok(encoded)
            }
        });
        let receipt = ArtboardReceiptCancellation {
            cancellation: cancellation.clone(),
            completed: false,
        };
        self.changed(cx);
        Ok(cx.spawn(async move |session, cx| {
            let mut receipt = receipt;
            let outcome = background.await;
            let outcome = session
                .update(cx, |session, cx| {
                    if session.resource_revoked {
                        return Err(ArtboardSessionError::ResourceRevoked);
                    }
                    if !session.owns_png_request(&cancellation) {
                        return Err(ArtboardSessionError::Superseded);
                    }
                    let outcome = if cancellation.is_cancelled() {
                        Err(ArtboardSessionError::Cancelled)
                    } else if !session.preview_is_current(&preview) {
                        Err(ArtboardSessionError::Superseded)
                    } else {
                        outcome.map(|encoded| PreparedArtboardPngAsset {
                            preview,
                            encoded,
                            cancellation,
                        })
                    };
                    session.diagnostic = outcome.as_ref().err().cloned();
                    session.changed(cx);
                    outcome
                })
                .map_err(|_released| ArtboardSessionError::Released)
                .and_then(|result| result);
            receipt.completed = true;
            outcome
        }))
    }

    pub fn cancel_png(&mut self, cx: &mut gpui::Context<Self>) -> bool {
        let changed = self.cancel_png_request();
        if changed {
            if !self.resource_revoked {
                self.diagnostic = Some(ArtboardSessionError::Cancelled);
            }
            self.changed(cx);
        }
        changed
    }

    fn owns_png_request(
        &self,
        cancellation: &std::sync::Arc<sim_tensor::CancellationToken>,
    ) -> bool {
        self.png_cancellation
            .as_ref()
            .is_some_and(|current| std::sync::Arc::ptr_eq(current, cancellation))
    }

    fn cancel_png_request(&mut self) -> bool {
        if let Some(cancellation) = self.png_cancellation.take() {
            cancellation.cancel();
            true
        } else {
            false
        }
    }

    pub fn cancel(&mut self, cx: &mut gpui::Context<Self>) -> bool {
        let changed = self.cancel_owned_request();
        if changed {
            if !self.resource_revoked {
                self.diagnostic = Some(ArtboardSessionError::Cancelled);
            }
            self.changed(cx);
        }
        changed
    }

    fn owns_request(&self, authority: &std::sync::Arc<ArtboardRequestAuthority>) -> bool {
        !self.resource_revoked
            && self.generation == authority.generation
            && self.authority.as_ref().is_some_and(|current| {
                std::sync::Arc::ptr_eq(&current.cancellation, &authority.cancellation)
                    && std::sync::Arc::ptr_eq(current, authority)
            })
    }

    fn request_is_current(&self, authority: &std::sync::Arc<ArtboardRequestAuthority>) -> bool {
        self.owns_request(authority) && !authority.cancellation.is_cancelled()
    }

    fn check_request(
        &self,
        authority: &std::sync::Arc<ArtboardRequestAuthority>,
    ) -> Result<(), ArtboardSessionError> {
        if self.resource_revoked {
            Err(ArtboardSessionError::ResourceRevoked)
        } else if !self.owns_request(authority) {
            Err(ArtboardSessionError::Superseded)
        } else if authority.cancellation.is_cancelled() {
            Err(ArtboardSessionError::Cancelled)
        } else {
            Ok(())
        }
    }

    fn cancel_owned_request(&mut self) -> bool {
        let png_changed = self.cancel_png_request();
        let authority = self.authority.take();
        if let Some(authority) = &authority {
            authority.cancellation.cancel();
        }
        let pending = self.pending.take();
        let changed =
            png_changed || authority.is_some() || pending.is_some() || self.document.is_pending();
        self.document.cancel();
        drop(pending);
        changed
    }

    fn changed(&self, cx: &mut gpui::Context<Self>) {
        cx.emit(ArtboardSessionEvent::Changed);
        cx.notify();
    }
}

#[cfg(test)]
mod artboard_session_controls {
    use super::*;
    use gpui::{DevicePixels, TestAppContext, size};
    use graphics_core::evaluate::{
        CpuSvgArtboardDocument, CpuSvgArtboardParameters, CpuSvgEvaluationBudget,
    };
    use graphite_graph_craft::document::{NodeId, value::DVec2};
    use graphite_std::Color;
    use graphite_std::application_io::{ExportFormat, RenderConfig, TimingInformation};
    use graphite_std::transform::Footprint;
    use graphite_std::vector_types::vector::style::RenderMode;
    use std::sync::Arc;

    fn document(scale: f64, max_output_bytes: usize) -> CpuSvgArtboardDocument {
        let configuration = RenderConfig {
            viewport: Footprint {
                transform: graphite_glam::DAffine2::from_translation(DVec2::ONE),
                resolution: graphite_glam::UVec2::new(31, 17),
                quality: Footprint::DEFAULT.quality,
            },
            scale,
            time: TimingInformation {
                time: 0.,
                animation_time: std::time::Duration::ZERO,
            },
            pointer: DVec2::ZERO,
            render_mode: RenderMode::Normal,
            export_format: ExportFormat::Svg,
            for_export: true,
            for_eyedropper: false,
        };
        CpuSvgArtboardDocument::new(
            91,
            NodeId(7001),
            CpuSvgArtboardParameters {
                location: DVec2::new(2., 2.),
                dimensions: DVec2::new(6., 4.),
                background: Color::from_gamma_srgb_channels(1., 0., 0., 0.5),
                clip: true,
            },
            configuration,
            CpuSvgEvaluationBudget {
                max_viewport_pixels: 31 * 17,
                max_output_bytes,
            },
        )
        .expect("actual original template with explicit context")
    }

    fn pixel(preview: &PreparedArtboardPreview, x: usize, y: usize) -> [u8; 4] {
        let pixels = preview.image().as_bytes(0).expect("native single frame");
        pixels
            .get((y * 31 + x) * 4..(y * 31 + x + 1) * 4)
            .expect("fixture interior coordinate")
            .try_into()
            .expect("BGRA pixel")
    }

    #[gpui::test]
    async fn at_r18_2_artboard_session_original_execution_pixels_and_metadata(
        cx: &mut TestAppContext,
    ) {
        for scale in [1., 1.5, 2.] {
            let document = document(scale, 64 * 1024);
            let network_before = document.network().clone();
            let configuration = document.configuration();
            let parameters = document.parameters();
            let session = cx.new(|_| GraphiteArtboardSession::new(document));
            let preparation = session
                .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
                .expect("tracked preparation");
            assert!(session.read_with(cx, |session, _| session.is_pending()));
            let preview = preparation
                .await
                .expect("original compiled and converted receipt");
            assert_eq!(preview.generation(), 1);
            assert_eq!(preview.image().frame_count(), 1);
            assert_eq!(
                preview.image().size(0),
                size(DevicePixels(31), DevicePixels(17))
            );
            assert_eq!(
                preview.image().as_bytes(0).expect("frame").len(),
                31 * 17 * 4
            );
            for (actual, expected) in pixel(&preview, (5. * scale) as usize, (5. * scale) as usize)
                .into_iter()
                .zip([0_u8, 0, 255, 128])
            {
                assert!(
                    actual.abs_diff(expected) <= 1,
                    "native interior {actual} versus {expected}"
                );
            }
            assert_eq!(pixel(&preview, 0, 0), [0, 0, 0, 0]);
            assert_eq!(pixel(&preview, 30, 16), [0, 0, 0, 0]);
            let output_before = preview.output().output().clone();
            let output_lease = preview.output().output_lease();
            assert_eq!(output_before.metadata.backgrounds.len(), 1);
            assert_eq!(
                output_before.metadata.backgrounds[0].location,
                parameters.location
            );
            assert_eq!(
                output_before.metadata.backgrounds[0].dimensions,
                parameters.dimensions
            );
            assert!(output_before.metadata.clip_targets.contains(&NodeId(7001)));
            assert!(
                output_before
                    .metadata
                    .click_targets
                    .contains_key(&NodeId(7001))
            );
            session.read_with(cx, |session, _| {
                assert!(!session.is_pending());
                assert!(session.document().owns_current_output(preview.output()));
                assert!(session.preview_is_current(&preview));
                assert_eq!(session.document().network(), &network_before);
                assert_eq!(session.configuration(), configuration);
                assert!(Arc::ptr_eq(
                    &session.current_output().expect("accepted").output_lease(),
                    &output_lease
                ));
                assert!(Arc::ptr_eq(
                    session.preview().expect("retained frame").image(),
                    preview.image()
                ));
                assert!(session.diagnostic().is_none());
            });
            assert_eq!(preview.output().output(), &output_before);
        }
    }

    #[gpui::test]
    async fn at_r18_2_artboard_session_successors_transaction_and_recovery(
        cx: &mut TestAppContext,
    ) {
        let session = cx.new(|_| GraphiteArtboardSession::new(document(1., 64 * 1024)));
        let pending = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("pending original");
        let token = session.read_with(cx, |session, _| {
            session
                .authority
                .as_ref()
                .expect("authority")
                .cancellation
                .clone()
        });
        session.update(cx, |session, cx| {
            let parameters = session.parameters();
            let configuration = session.configuration();
            assert_eq!(session.apply(0, parameters, configuration, cx), Ok(false));
            assert!(session.is_pending());
            assert!(!token.is_cancelled());
            assert_eq!(
                session.apply(1, parameters, configuration, cx),
                Err(graphics_core::evaluate::CpuSvgDocumentError::StaleRevision)
            );
            assert!(session.is_pending());
            let mut parameters = parameters;
            parameters.background = Color::BLUE;
            assert_eq!(session.apply(0, parameters, configuration, cx), Ok(true));
            assert_eq!(session.revision(), 1);
        });
        assert!(token.is_cancelled());
        assert!(matches!(
            pending.await,
            Err(ArtboardSessionError::Cancelled)
        ));
        let current = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("successor request")
            .await
            .expect("successor frame");
        assert_eq!(pixel(&current, 5, 5), [255, 0, 0, 255]);
        let accepted_before = current.output().output().clone();
        let network_before =
            session.read_with(cx, |session, _| session.document().network().clone());
        session.update(cx, |session, cx| {
            let mut configuration = session.configuration();
            configuration.pointer.x = -0.;
            assert_eq!(
                session.apply(1, session.parameters(), configuration, cx),
                Ok(true)
            );
            assert_eq!(
                session.configuration().pointer.x.to_bits(),
                (-0_f64).to_bits()
            );
            assert_eq!(session.document().network(), &network_before);
            assert!(!session.preview_is_current(&current));
        });
        let replacement = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("context successor")
            .await
            .expect("context conversion");
        assert!(session.read_with(cx, |session, _| session.preview_is_current(&replacement)));
        assert!(!session.read_with(cx, |session, _| session.preview_is_current(&current)));
        assert_eq!(current.output().output(), &accepted_before);
        let superseded = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("first retry");
        let retry = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("latest retry");
        assert!(matches!(
            superseded.await,
            Err(ArtboardSessionError::Cancelled)
        ));
        let latest = retry.await.expect("same checkpoint original retry");
        assert!(!session.read_with(cx, |session, _| {
            session.document().owns_current_output(replacement.output())
        }));
        assert!(session.read_with(cx, |session, _| session.preview_is_current(&latest)));
        let foreign = cx.new(|_| GraphiteArtboardSession::new(document(1., 64 * 1024)));
        let foreign_preview = foreign
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("foreign genuine request")
            .await
            .expect("foreign genuine receipt");
        assert!(foreign.read_with(cx, |session, _| {
            session.preview_is_current(&foreign_preview)
        }));
        assert!(!foreign.read_with(cx, |session, _| session.preview_is_current(&latest)));
        assert!(!session.read_with(cx, |session, _| {
            session.preview_is_current(&foreign_preview)
        }));
        assert_eq!(
            session.read_with(cx, |session, _| session.document().network().clone()),
            network_before
        );
    }

    #[gpui::test]
    async fn at_r18_2_artboard_session_typed_budget_failures_and_retained_pixels(
        cx: &mut TestAppContext,
    ) {
        let session = cx.new(|_| GraphiteArtboardSession::new(document(1., 64 * 1024)));
        let valid = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("request")
            .await
            .expect("valid frame");
        let network_before =
            session.read_with(cx, |session, _| session.document().network().clone());
        for maximum in [0, 31 * 17 * 4 - 1] {
            let failed = session
                .update(cx, |session, cx| session.request_preview(maximum, cx))
                .expect("bounded request")
                .await;
            let Err(ArtboardSessionError::Conversion(error)) = failed else {
                panic!("typed native pixel budget error");
            };
            assert!(
                matches!(error.as_ref(), crate::render::CpuSvgPreviewError::PreviewBudgetExceeded { required, maximum: actual } if *required == 31 * 17 * 4 && *actual == maximum)
            );
            session.read_with(cx, |session, _| {
                assert!(!session.is_pending());
                assert!(matches!(
                    session.diagnostic(),
                    Some(ArtboardSessionError::Conversion(_))
                ));
                assert!(Arc::ptr_eq(
                    session.preview().expect("last valid").image(),
                    valid.image()
                ));
                assert!(!session.preview_is_current(&valid));
                assert_eq!(session.document().network(), &network_before);
            });
        }
        let recovered = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("exact cap recovery")
            .await
            .expect("recovered frame");
        assert!(session.read_with(cx, |session, _| session.preview_is_current(&recovered)));
        assert_eq!(recovered.image().as_bytes(0), valid.image().as_bytes(0));
        let tiny_output = cx.new(|_| GraphiteArtboardSession::new(document(1., 0)));
        let failed = tiny_output
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("original evaluation request")
            .await;
        assert!(
            matches!(failed, Err(ArtboardSessionError::Acceptance(error)) if matches!(error.as_ref(), graphics_core::evaluate::CpuSvgAcceptanceError::Evaluation(graphics_core::evaluate::CpuSvgEvaluationError::OutputBudgetExceeded)))
        );
        assert!(
            tiny_output.read_with(cx, |session, _| session.preview().is_none()
                && session.current_output().is_none()
                && !session.is_pending())
        );
    }

    #[gpui::test]
    async fn at_r18_2_artboard_session_abandoned_receipt_cancel_and_release(
        cx: &mut TestAppContext,
    ) {
        let session = cx.new(|_| GraphiteArtboardSession::new(document(1., 64 * 1024)));
        let abandoned = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("abandonable request");
        let token = session.read_with(cx, |session, _| {
            session
                .authority
                .as_ref()
                .expect("authority")
                .cancellation
                .clone()
        });
        drop(abandoned);
        cx.run_until_parked();
        assert!(token.is_cancelled());
        session.read_with(cx, |session, _| {
            assert!(!session.is_pending());
            assert!(!session.document().is_pending());
            assert!(
                session
                    .preview()
                    .is_none_or(|preview| !session.preview_is_current(preview))
            );
            assert!(
                session
                    .diagnostic()
                    .is_none_or(|error| matches!(error, ArtboardSessionError::Cancelled))
            );
        });
        let pending = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("cancel request");
        session.update(cx, |session, cx| {
            assert!(session.cancel(cx));
            assert!(!session.cancel(cx));
        });
        assert!(matches!(
            pending.await,
            Err(ArtboardSessionError::Cancelled)
        ));
        let recovered = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("recovery")
            .await
            .expect("current receipt");
        let accepted_before = recovered.output().output().clone();
        session.update(cx, |session, cx| {
            assert!(session.cancel(cx));
        });
        assert!(!session.read_with(cx, |session, _| session.preview_is_current(&recovered)));
        assert!(session.read_with(cx, |session, _| {
            session.document().owns_current_output(recovered.output())
        }));
        assert_eq!(recovered.output().output(), &accepted_before);
        let pending = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("release request");
        let token = session.read_with(cx, |session, _| {
            session
                .authority
                .as_ref()
                .expect("release authority")
                .cancellation
                .clone()
        });
        let weak = session.downgrade();
        drop(session);
        cx.update(|_| {});
        cx.run_until_parked();
        assert!(weak.upgrade().is_none());
        assert!(token.is_cancelled());
        assert!(matches!(
            pending.await,
            Err(ArtboardSessionError::Cancelled)
        ));
    }

    #[gpui::test]
    async fn at_r18_2_artboard_session_completed_unawaited_task_drop_retains_authority(
        cx: &mut TestAppContext,
    ) {
        let session = cx.new(|_| GraphiteArtboardSession::new(document(1., 64 * 1024)));
        let completed = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("actual scheduled request");
        cx.run_until_parked();
        assert!(completed.is_ready());
        let preview = session.read_with(cx, |session, _| {
            session.preview().expect("prepared genuine frame").clone()
        });
        let output_before = preview.output().output().clone();
        let network_before =
            session.read_with(cx, |session, _| session.document().network().clone());
        let token = preview.authority.cancellation.clone();
        assert!(!token.is_cancelled());
        assert!(session.read_with(cx, |session, _| session.preview_is_current(&preview)));
        drop(completed);
        assert!(!token.is_cancelled());
        session.read_with(cx, |session, _| {
            assert!(!session.is_pending());
            assert!(session.preview_is_current(&preview));
            assert!(session.document().owns_current_output(preview.output()));
            assert!(session.diagnostic().is_none());
            assert_eq!(session.document().network(), &network_before);
        });
        session.update(cx, |session, cx| {
            assert!(session.cancel(cx));
        });
        assert!(token.is_cancelled());
        assert!(!session.read_with(cx, |session, _| session.preview_is_current(&preview)));
        assert!(session.read_with(cx, |session, _| {
            session.document().owns_current_output(preview.output())
        }));
        assert_eq!(preview.output().output(), &output_before);
    }
    #[gpui::test]
    async fn at_r18_2_artboard_session_internal_install_pending_receipt_abandonment_invalidates_retained_status(
        cx: &mut TestAppContext,
    ) {
        let session = cx.new(|_| GraphiteArtboardSession::new(document(1., 64 * 1024)));
        let network_before =
            session.read_with(cx, |session, _| session.document().network().clone());
        let presentation = cx.new(crate::image_preview::CpuPreviewPresentation::new);
        let window = cx.add_empty_window();
        let receipt = session
            .update(window, |session, cx| {
                session.request_preview(31 * 17 * 4, cx)
            })
            .expect("genuine pending receipt");
        while session.read_with(window, |session, _| session.preview().is_none()) {
            assert!(
                window.executor().tick(),
                "actual tracked pipeline remains runnable"
            );
        }
        assert!(
            !receipt.is_ready(),
            "internal install precedes receipt completion"
        );
        let preview = session.read_with(window, |session, _| {
            session
                .preview()
                .expect("genuine internally installed frame")
                .clone()
        });
        let token = preview.authority.cancellation.clone();
        let output_before = preview.output().output().clone();
        assert!(!token.is_cancelled());
        assert!(session.read_with(window, |session, _| session.preview_is_current(&preview)));
        window
            .update(|window, cx| {
                session.update(cx, |session, cx| {
                    presentation.update(cx, |presentation, cx| {
                        presentation.accept_artboard_preview(session, &preview, window, cx)
                    })
                })
            })
            .expect("primitive accepts genuine current internal snapshot");
        assert!(presentation.read_with(window, |presentation, _| presentation.is_fresh()));
        assert!(
            !receipt.is_ready(),
            "receipt remains pending before abandonment"
        );
        drop(receipt);
        window.run_until_parked();
        assert!(token.is_cancelled());
        session.read_with(window, |session, _| {
            assert!(!session.preview_is_current(&preview));
            assert!(session.document().owns_current_output(preview.output()));
            assert_eq!(session.document().network(), &network_before);
        });
        session.update(window, |session, cx| {
            presentation.update(cx, |presentation, cx| {
                presentation.refresh_artboard_status(session, cx);
            });
        });
        presentation.read_with(window, |presentation, _| {
            assert!(!presentation.is_fresh());
            assert_eq!(
                presentation.last_valid_qualification(),
                Some("Last valid image")
            );
            assert!(Arc::ptr_eq(
                presentation.image().expect("retained last valid"),
                preview.image()
            ));
            assert!(matches!(
                presentation.diagnostic(),
                Some(crate::image_preview::CpuPreviewDiagnostic::Preparation(
                    crate::image_preview::CpuPreviewError::ResultNotFresh
                ))
            ));
        });
        assert_eq!(preview.output().output(), &output_before);
    }
    #[gpui::test]
    async fn at_r22_1_artboard_png_current_receipts_caps_and_source_preservation(
        cx: &mut TestAppContext,
    ) {
        let session = cx.new(|_| GraphiteArtboardSession::new(document(1., 64 * 1024)));
        let preview = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("genuine source request")
            .await
            .expect("original frame");
        let output_before = preview.output().output().clone();
        let network_before =
            session.read_with(cx, |session, _| session.document().network().clone());
        let first = session
            .update(cx, |session, cx| {
                session.prepare_png(preview.clone(), 31 * 17 * 4, 64 * 1024, cx)
            })
            .expect("current source PNG request")
            .await
            .expect("genuine baked PNG");
        assert!(session.read_with(cx, |session, _| session.png_is_current(&first)));
        assert!(Arc::ptr_eq(
            &first.source_output().output_lease(),
            &preview.output().output_lease()
        ));
        let encoded_before = first.png().encoded_bytes().to_vec();
        let encoded_size = encoded_before.len();
        assert_eq!((first.png().width(), first.png().height()), (31, 17));
        assert_eq!(first.png().content_type(), "image/png");
        let exact = session
            .update(cx, |session, cx| {
                session.prepare_png(preview.clone(), 31 * 17 * 4, encoded_size, cx)
            })
            .expect("exact cap request")
            .await
            .expect("exact cap succeeds");
        assert!(!session.read_with(cx, |session, _| session.png_is_current(&first)));
        assert!(session.read_with(cx, |session, _| session.png_is_current(&exact)));
        assert_eq!(exact.png().encoded_bytes(), encoded_before);
        for (pixel_limit, encoded_limit) in [
            (0, 64 * 1024),
            (31 * 17 * 4 - 1, 64 * 1024),
            (31 * 17 * 4, 0),
            (31 * 17 * 4, encoded_size - 1),
        ] {
            let failure = session
                .update(cx, |session, cx| {
                    session.prepare_png(preview.clone(), pixel_limit, encoded_limit, cx)
                })
                .expect("source remains current before bounded bake")
                .await;
            assert!(matches!(
                failure,
                Err(ArtboardSessionError::PngPreparation(_))
            ));
            session.read_with(cx, |session, _| {
                assert!(session.preview_is_current(&preview));
                assert!(!session.png_is_current(&exact));
                assert_eq!(session.document().network(), &network_before);
                assert_eq!(
                    session
                        .document()
                        .accepted()
                        .expect("immutable source")
                        .output(),
                    &output_before
                );
                assert!(matches!(
                    session.diagnostic(),
                    Some(ArtboardSessionError::PngPreparation(_))
                ));
            });
        }
        let recovered = session
            .update(cx, |session, cx| {
                session.prepare_png(preview.clone(), 31 * 17 * 4, encoded_size, cx)
            })
            .expect("recovery request")
            .await
            .expect("recovered baked bytes");
        assert!(session.read_with(cx, |session, _| session.png_is_current(&recovered)));
        assert_eq!(recovered.png().encoded_bytes(), encoded_before);
        assert_eq!(recovered.pixels_sha256(), first.pixels_sha256());
        assert_eq!(first.source_output().output(), &output_before);
        assert_eq!(first.png().encoded_bytes(), encoded_before);
    }

    #[gpui::test]
    async fn at_r22_1_artboard_png_foreign_source_successors_and_supersession_fail_closed(
        cx: &mut TestAppContext,
    ) {
        let session = cx.new(|_| GraphiteArtboardSession::new(document(1., 64 * 1024)));
        let foreign = cx.new(|_| GraphiteArtboardSession::new(document(1., 64 * 1024)));
        let preview = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("source request")
            .await
            .expect("source frame");
        let foreign_preview = foreign
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("colliding foreign request")
            .await
            .expect("foreign frame");
        assert_eq!(foreign_preview.generation(), preview.generation());
        assert_eq!(
            foreign_preview.output().revision(),
            preview.output().revision()
        );
        assert!(matches!(
            session.update(cx, |session, cx| session.prepare_png(
                foreign_preview,
                31 * 17 * 4,
                64 * 1024,
                cx
            )),
            Err(ArtboardSessionError::Superseded)
        ));
        let old_request = session
            .update(cx, |session, cx| {
                session.prepare_png(preview.clone(), 31 * 17 * 4, 64 * 1024, cx)
            })
            .expect("first bake");
        let new_request = session
            .update(cx, |session, cx| {
                session.prepare_png(preview.clone(), 31 * 17 * 4, 64 * 1024, cx)
            })
            .expect("successor bake");
        assert!(matches!(
            old_request.await,
            Err(ArtboardSessionError::Superseded)
        ));
        let old_png = new_request.await.expect("current bake");
        let old_bytes = old_png.png().encoded_bytes().to_vec();
        assert!(session.read_with(cx, |session, _| session.png_is_current(&old_png)));
        let source_changed_request = session
            .update(cx, |session, cx| {
                session.prepare_png(preview.clone(), 31 * 17 * 4, 64 * 1024, cx)
            })
            .expect("pending old source bake");
        session.update(cx, |session, cx| {
            let mut parameters = session.parameters();
            parameters.background = Color::from_gamma_srgb_channels(0., 0., 1., 0.5);
            assert!(
                session
                    .apply(session.revision(), parameters, session.configuration(), cx)
                    .expect("transactional source change")
            );
        });
        assert!(matches!(
            source_changed_request.await,
            Err(ArtboardSessionError::Superseded)
        ));
        assert!(!session.read_with(cx, |session, _| session.png_is_current(&old_png)));
        assert!(matches!(
            session.update(cx, |session, cx| session.prepare_png(
                preview,
                31 * 17 * 4,
                64 * 1024,
                cx
            )),
            Err(ArtboardSessionError::Superseded)
        ));
        let next_preview = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("genuine new source request")
            .await
            .expect("new source frame");
        let next_png = session
            .update(cx, |session, cx| {
                session.prepare_png(next_preview.clone(), 31 * 17 * 4, 64 * 1024, cx)
            })
            .expect("new source bake")
            .await
            .expect("new PNG");
        assert!(session.read_with(cx, |session, _| session.png_is_current(&next_png)));
        assert_ne!(
            next_png.png().content_sha256(),
            old_png.png().content_sha256()
        );
        assert!(Arc::ptr_eq(
            &next_png.source_output().output_lease(),
            &next_preview.output().output_lease()
        ));
        assert_eq!(old_png.png().encoded_bytes(), old_bytes);
    }

    #[gpui::test]
    async fn at_r22_1_artboard_png_pending_abandonment_cancel_completed_task_and_release(
        cx: &mut TestAppContext,
    ) {
        let session = cx.new(|_| GraphiteArtboardSession::new(document(1., 64 * 1024)));
        let preview = session
            .update(cx, |session, cx| session.request_preview(31 * 17 * 4, cx))
            .expect("source request")
            .await
            .expect("source frame");
        let abandoned = session
            .update(cx, |session, cx| {
                session.prepare_png(preview.clone(), 31 * 17 * 4, 64 * 1024, cx)
            })
            .expect("abandonable bake");
        let token = session.read_with(cx, |session, _| {
            session
                .png_cancellation
                .as_ref()
                .expect("PNG authority")
                .clone()
        });
        assert!(!abandoned.is_ready());
        drop(abandoned);
        cx.run_until_parked();
        assert!(token.is_cancelled());
        assert!(session.read_with(cx, |session, _| session.preview_is_current(&preview)));
        let cancelled = session
            .update(cx, |session, cx| {
                session.prepare_png(preview.clone(), 31 * 17 * 4, 64 * 1024, cx)
            })
            .expect("explicitly cancelled bake");
        session.update(cx, |session, cx| {
            assert!(session.cancel_png(cx));
            assert!(!session.cancel_png(cx));
        });
        assert!(matches!(
            cancelled.await,
            Err(ArtboardSessionError::Superseded)
        ));
        assert!(session.read_with(cx, |session, _| session.preview_is_current(&preview)));
        let completed = session
            .update(cx, |session, cx| {
                session.prepare_png(preview.clone(), 31 * 17 * 4, 64 * 1024, cx)
            })
            .expect("scheduled completed bake");
        let completed_token = session.read_with(cx, |session, _| {
            session
                .png_cancellation
                .as_ref()
                .expect("current token")
                .clone()
        });
        cx.run_until_parked();
        assert!(completed.is_ready());
        drop(completed);
        assert!(!completed_token.is_cancelled());
        let recovered = session
            .update(cx, |session, cx| {
                session.prepare_png(preview, 31 * 17 * 4, 64 * 1024, cx)
            })
            .expect("recovery bake")
            .await
            .expect("current PNG");
        assert!(session.read_with(cx, |session, _| session.png_is_current(&recovered)));
        let encoded_before = recovered.png().encoded_bytes().to_vec();
        let weak = session.downgrade();
        drop(session);
        cx.update(|_| {});
        cx.run_until_parked();
        assert!(weak.upgrade().is_none());
        assert!(recovered.cancellation.is_cancelled());
        assert_eq!(recovered.png().encoded_bytes(), encoded_before);
    }
}

/// Observes an original owner without adding evaluation, history, storage or freshness operations.
/// Original graph values can retain shared capabilities; this is not generic evaluation detachment.
#[derive(Debug)]
pub struct OriginalDocumentCapture {
    document_id: graphite_editor::messages::portfolio::document::utility_types::misc::DocumentId,
    network_interface: graphite_editor::messages::portfolio::document::utility_types::network_interface::NodeNetworkInterface,
    observed_metadata: graphite_editor::messages::portfolio::document::utility_types::document_metadata::DocumentMetadata,
    resource_declarations: graphite_graph_craft::application_io::resource::ResourceRegistry,
    camera: graphite_editor::messages::portfolio::document::utility_types::misc::PTZ,
    render_mode: graphite_std::vector_types::vector::style::RenderMode,
    selection_network_path: Vec<graphite_graph_craft::document::NodeId>,
    source_network_hash: u64,
}

impl OriginalDocumentCapture {
    pub fn capture(
        document_id: graphite_editor::messages::portfolio::document::utility_types::misc::DocumentId,
        document: &graphite_editor::messages::portfolio::document::DocumentMessageHandler,
    ) -> Self {
        Self {
            document_id,
            network_interface: document.network_interface.clone(),
            observed_metadata: document.metadata().clone(),
            resource_declarations: document.resources.registry.clone(),
            camera: document.document_ptz,
            render_mode: document.render_mode,
            selection_network_path: document.selection_network_path().to_vec(),
            source_network_hash: document.current_hash(),
        }
    }

    pub fn document_id(
        &self,
    ) -> graphite_editor::messages::portfolio::document::utility_types::misc::DocumentId {
        self.document_id
    }

    pub fn network_interface(
        &self,
    ) -> &graphite_editor::messages::portfolio::document::utility_types::network_interface::NodeNetworkInterface{
        &self.network_interface
    }

    pub fn observed_metadata(
        &self,
    ) -> &graphite_editor::messages::portfolio::document::utility_types::document_metadata::DocumentMetadata{
        &self.observed_metadata
    }

    /// These are original resource declarations, not retained bytes or read/version leases.
    pub fn resource_declarations(
        &self,
    ) -> &graphite_graph_craft::application_io::resource::ResourceRegistry {
        &self.resource_declarations
    }

    pub fn camera(
        &self,
    ) -> graphite_editor::messages::portfolio::document::utility_types::misc::PTZ {
        self.camera
    }

    pub fn render_mode(&self) -> graphite_std::vector_types::vector::style::RenderMode {
        self.render_mode
    }

    pub fn selection_network_path(&self) -> &[graphite_graph_craft::document::NodeId] {
        &self.selection_network_path
    }

    /// Original network hash only; camera, declarations and observed metadata are not a checkpoint.
    pub fn source_network_hash(&self) -> u64 {
        self.source_network_hash
    }
}

#[cfg(test)]
mod original_document_capture_controls {
    use super::OriginalDocumentCapture;
    use graphite_editor::messages::portfolio::document::{
        DocumentMessageHandler,
        node_graph::document_node_definitions::{
            ARTBOARD_DIMENSIONS_INPUT_INDEX, resolve_network_node_type, resolve_proto_node_type,
        },
        utility_types::{
            document_metadata::LayerNodeIdentifier,
            misc::{DocumentId, PTZ},
            network_interface::{InputConnector, TransactionStatus},
        },
    };
    use graphite_graph_craft::document::value::{DAffine2, DVec2};
    use graphite_graph_craft::{
        application_io::resource::{DataSource, ResourceHash, ResourceId},
        document::{NodeId, NodeInput, value::TaggedValue},
    };
    use graphite_std::vector_types::vector::style::RenderMode;
    use graphite_std::{NodeParameter, math_nodes::add};

    fn artboard_document(node_id: NodeId) -> DocumentMessageHandler {
        let definition =
            resolve_network_node_type("Artboard").expect("original Artboard definition");
        let mut source = DocumentMessageHandler::default();
        assert!(
            source
                .network_interface
                .insert_node(node_id, definition.default_node_template(), &[])
                .is_none()
        );
        source.network_interface.set_input(
            &InputConnector::Export(0),
            NodeInput::node(node_id, 0),
            &[],
        );
        source.network_interface.load_structure();
        source
            .network_interface
            .selected_nodes_mut(&[])
            .expect("original root selection")
            .set_selected_nodes(vec![node_id]);
        assert!(source.storage().is_none());
        source
    }

    fn numeric_document(node_id: NodeId, augend: f64, addend: f64) -> DocumentMessageHandler {
        let definition =
            resolve_proto_node_type(add::IDENTIFIER).expect("original generated Add definition");
        let mut template = definition.default_node_template();
        for (index, value) in [
            (add::AugendInput::INDEX, augend),
            (add::AddendInput::INDEX, addend),
        ] {
            let input = template
                .inputs
                .get_mut(index)
                .expect("original ordered Add input");
            let NodeInput::Value { exposed, .. } = input else {
                panic!("original Add parameter is not a literal");
            };
            *input = NodeInput::value(TaggedValue::Number(value), *exposed);
        }
        let mut source = DocumentMessageHandler::default();
        assert!(
            source
                .network_interface
                .insert_node(node_id, template, &[])
                .is_none()
        );
        source
    }

    fn camera_bits(camera: PTZ) -> ([u64; 2], u64, u64, bool) {
        (
            camera.pan.to_array().map(f64::to_bits),
            camera.zoom().to_bits(),
            camera.unmodified_tilt().to_bits(),
            camera.flip,
        )
    }

    #[test]
    fn at_r03_r04_original_capture_retains_authentic_graph_metadata_camera_and_declarations() {
        let node_id = NodeId(7001);
        let document_id = DocumentId(0);
        let mut source = artboard_document(node_id);
        let resource_id = ResourceId::from(9001);
        let declared_hash = ResourceHash::from(&b"declaration without owned storage"[..]);
        source
            .resources
            .registry
            .push_source_back(&resource_id, DataSource::Embedded);
        assert!(
            source
                .resources
                .registry
                .resolve(&resource_id, declared_hash)
                .is_none()
        );
        source.document_ptz.pan = DVec2::new(13.25, -7.5);
        source.document_ptz.set_zoom(2.);
        source.document_ptz.set_tilt(0.25);
        source.document_ptz.flip = true;
        source.render_mode = RenderMode::Outline;
        let transform =
            DAffine2::from_scale_angle_translation(DVec2::new(2., 3.), 0.25, DVec2::new(13., -7.));
        source
            .network_interface
            .set_document_to_viewport_transform(transform);
        source.set_save_state(true);
        let layer = LayerNodeIdentifier::new(node_id, &source.network_interface);
        let network_before = source.document_network().clone();
        let metadata_before = source.network_interface.document_network_metadata().clone();
        let source_hash = source.current_hash();
        let capture = OriginalDocumentCapture::capture(document_id, &source);
        assert_eq!(capture.document_id(), document_id);
        assert_eq!(
            capture.network_interface().document_network(),
            &network_before
        );
        assert_eq!(
            capture.network_interface().document_network_metadata(),
            &metadata_before
        );
        assert_eq!(
            capture.network_interface().document_network().exports,
            vec![NodeInput::node(node_id, 0)]
        );
        let original = source
            .document_network()
            .nodes
            .get(&node_id)
            .expect("authentic Artboard");
        let captured = capture
            .network_interface()
            .document_node(&node_id, &[])
            .expect("captured Artboard");
        assert_eq!(captured, original);
        assert_eq!(captured.inputs.len(), 6);
        assert_eq!(
            capture.network_interface().node_metadata(&node_id, &[]),
            source.network_interface.node_metadata(&node_id, &[])
        );
        assert_eq!(
            capture
                .network_interface()
                .selected_nodes()
                .selected_nodes()
                .copied()
                .collect::<Vec<_>>(),
            vec![node_id]
        );
        assert_eq!(
            capture.selection_network_path(),
            source.selection_network_path()
        );
        assert!(capture.observed_metadata().layer_exists(layer));
        assert_eq!(
            capture.observed_metadata().structure.len(),
            source.metadata().structure.len()
        );
        assert_eq!(
            capture
                .observed_metadata()
                .document_to_viewport
                .to_cols_array()
                .map(f64::to_bits),
            transform.to_cols_array().map(f64::to_bits)
        );
        assert!(
            capture
                .observed_metadata()
                .bounding_box_document(layer)
                .is_none()
        );
        assert!(
            capture
                .network_interface()
                .document_metadata()
                .structure
                .is_empty()
        );
        assert_eq!(capture.resource_declarations(), &source.resources.registry);
        assert_eq!(
            capture.resource_declarations().hash(&resource_id),
            Some(declared_hash)
        );
        assert_eq!(
            capture
                .resource_declarations()
                .info(&resource_id)
                .expect("captured declaration")
                .sources,
            &[DataSource::Embedded]
        );
        assert_eq!(
            camera_bits(capture.camera()),
            camera_bits(source.document_ptz)
        );
        assert_eq!(capture.render_mode(), RenderMode::Outline);
        assert_eq!(capture.source_network_hash(), source_hash);
        assert_eq!(source.document_network(), &network_before);
        assert!(source.is_saved());
        assert!(source.storage().is_none());
        assert!(source.resources.embedded.is_empty());
    }

    #[test]
    fn at_r03_r04_original_capture_survives_source_edits_and_colliding_document_node_ids() {
        let node_id = NodeId(7001);
        let mut source = artboard_document(node_id);
        let other = artboard_document(node_id);
        let source_capture = OriginalDocumentCapture::capture(DocumentId(17), &source);
        let other_capture = OriginalDocumentCapture::capture(DocumentId(18), &other);
        let source_before = source.document_network().clone();
        let other_before = other.document_network().clone();
        let resource_id = ResourceId::from(9010);
        source
            .resources
            .registry
            .push_source_back(&resource_id, DataSource::Embedded);
        source.document_ptz.pan = DVec2::new(9., 17.);
        source.render_mode = RenderMode::SvgPreview;
        assert_eq!(source.current_hash(), source_capture.source_network_hash());
        assert!(source_capture.resource_declarations().is_empty());
        assert_ne!(
            camera_bits(source.document_ptz),
            camera_bits(source_capture.camera())
        );
        let dimensions = DVec2::new(320., 240.);
        source.network_interface.set_input(
            &InputConnector::node_at_index(node_id, ARTBOARD_DIMENSIONS_INPUT_INDEX),
            NodeInput::value(TaggedValue::DVec2(dimensions), false),
            &[],
        );
        source
            .network_interface
            .selected_nodes_mut(&[])
            .expect("source selection")
            .set_selected_nodes(Vec::new());
        source
            .network_interface
            .set_document_to_viewport_transform(DAffine2::IDENTITY);
        assert_ne!(source.current_hash(), source_capture.source_network_hash());
        assert_eq!(
            source_capture.network_interface().document_network(),
            &source_before
        );
        assert_eq!(
            other_capture.network_interface().document_network(),
            &other_before
        );
        assert_eq!(other.document_network(), &other_before);
        assert_ne!(source_capture.document_id(), other_capture.document_id());
        assert_eq!(
            source_capture
                .network_interface()
                .selected_nodes()
                .selected_nodes()
                .copied()
                .collect::<Vec<_>>(),
            vec![node_id]
        );
        assert!(
            source
                .network_interface
                .selected_nodes()
                .selected_nodes()
                .next()
                .is_none()
        );
        let mut editable_clone = source_capture.network_interface().clone();
        editable_clone.set_input(
            &InputConnector::node_at_index(node_id, ARTBOARD_DIMENSIONS_INPUT_INDEX),
            NodeInput::value(TaggedValue::DVec2(DVec2::new(51., 73.)), false),
            &[],
        );
        assert_ne!(editable_clone.document_network(), &source_before);
        assert_eq!(
            source_capture.network_interface().document_network(),
            &source_before
        );
        assert_eq!(other.document_network(), &other_before);
        assert_eq!(
            source
                .document_network()
                .nodes
                .get(&node_id)
                .expect("source Artboard")
                .inputs
                .get(ARTBOARD_DIMENSIONS_INPUT_INDEX)
                .and_then(NodeInput::as_value),
            Some(&TaggedValue::DVec2(dimensions))
        );
        assert!(source.storage().is_none() && other.storage().is_none());
    }

    #[test]
    fn at_r03_r04_original_capture_preserves_live_started_transaction_and_separate_observed_metadata()
     {
        let node_id = NodeId(7001);
        let mut source = artboard_document(node_id);
        let transform = DAffine2::from_translation(DVec2::new(7., 11.));
        source
            .network_interface
            .set_document_to_viewport_transform(transform);
        source.set_save_state(true);
        source.network_interface.start_transaction();
        let before = source.network_interface.clone();
        let hash_before = source.current_hash();
        let layer = LayerNodeIdentifier::new(node_id, &source.network_interface);
        let first = OriginalDocumentCapture::capture(DocumentId(27), &source);
        let second = OriginalDocumentCapture::capture(DocumentId(27), &source);
        assert_eq!(
            source.network_interface.transaction_status(),
            TransactionStatus::Started
        );
        assert_eq!(source.network_interface, before);
        assert_eq!(source.current_hash(), hash_before);
        assert!(source.is_saved());
        for capture in [&first, &second] {
            assert_eq!(
                capture.network_interface().transaction_status(),
                TransactionStatus::Finished
            );
            assert_eq!(
                capture.network_interface().document_network(),
                source.document_network()
            );
            assert_eq!(
                capture.network_interface().document_network_metadata(),
                source.network_interface.document_network_metadata()
            );
            assert!(
                !capture
                    .network_interface()
                    .document_metadata()
                    .layer_exists(layer)
            );
            assert!(
                capture
                    .network_interface()
                    .document_metadata()
                    .structure
                    .is_empty()
            );
            assert!(capture.observed_metadata().layer_exists(layer));
            assert_eq!(
                capture
                    .observed_metadata()
                    .document_to_viewport
                    .to_cols_array()
                    .map(f64::to_bits),
                transform.to_cols_array().map(f64::to_bits)
            );
            assert_eq!(capture.source_network_hash(), hash_before);
            assert_eq!(capture.selection_network_path(), &[] as &[NodeId]);
        }
        source.network_interface.finish_transaction();
        assert_eq!(
            first.network_interface().transaction_status(),
            TransactionStatus::Finished
        );
        assert!(source.storage().is_none());
    }

    #[test]
    fn at_r03_r04_original_capture_preserves_signed_zero_numbers_and_raw_camera_bits() {
        let node_id = NodeId(23);
        for (augend, addend) in [(-0.0_f64, 0.0_f64), (0.0_f64, -0.0_f64)] {
            let mut source = numeric_document(node_id, augend, addend);
            source.document_ptz.pan = DVec2::new(-0.0, 0.0);
            source.document_ptz.set_tilt(-0.0);
            source.document_ptz.set_zoom(2.);
            source.document_ptz.flip = true;
            source.set_save_state(true);
            let bits_before = camera_bits(source.document_ptz);
            let source_before = source.document_network().clone();
            let captured = OriginalDocumentCapture::capture(DocumentId(u64::MAX), &source);
            assert_eq!(captured.document_id(), DocumentId(u64::MAX));
            let node = captured
                .network_interface()
                .document_node(&node_id, &[])
                .expect("captured original Add");
            assert_eq!(
                node.implementation,
                graphite_graph_craft::document::DocumentNodeImplementation::ProtoNode(
                    add::IDENTIFIER
                )
            );
            assert_eq!(node.inputs.len(), 2);
            for (input, bits) in [
                (add::AugendInput::INDEX, augend.to_bits()),
                (add::AddendInput::INDEX, addend.to_bits()),
            ] {
                let Some(TaggedValue::Number(value)) =
                    node.inputs.get(input).and_then(NodeInput::as_value)
                else {
                    panic!("captured numeric slot changed type");
                };
                assert_eq!(value.to_bits(), bits);
                let Some(TaggedValue::Number(source_value)) = source
                    .document_network()
                    .nodes
                    .get(&node_id)
                    .expect("source Add")
                    .inputs
                    .get(input)
                    .and_then(NodeInput::as_value)
                else {
                    panic!("source numeric slot changed type");
                };
                assert_eq!(source_value.to_bits(), bits);
            }
            assert_eq!(camera_bits(captured.camera()), bits_before);
            assert_eq!(
                captured.camera().unmodified_tilt().to_bits(),
                (-0.0_f64).to_bits()
            );
            assert_eq!(source.document_network(), &source_before);
            assert!(source.is_saved());
            source.document_ptz.pan = DVec2::new(31., 41.);
            source.document_ptz.set_tilt(0.5);
            source.network_interface.set_input(
                &InputConnector::node_at_index(node_id, add::AddendInput::INDEX),
                NodeInput::value(TaggedValue::Number(9.), false),
                &[],
            );
            assert_eq!(camera_bits(captured.camera()), bits_before);
            assert_eq!(
                captured.network_interface().document_network(),
                &source_before
            );
            assert_eq!(captured.resource_declarations(), &source.resources.registry);
            assert!(captured.resource_declarations().is_empty());
            assert!(source.storage().is_none());
        }
    }

    #[test]
    fn at_r04_r18_original_capture_numeric_snapshot_preserves_genuine_network_and_supplied_context()
    {
        let node_id = NodeId(23);
        let document_id = DocumentId(17);
        let mut source = numeric_document(node_id, 2., 3.);
        source.network_interface.set_input(
            &InputConnector::Export(0),
            NodeInput::node(node_id, 0),
            &[],
        );
        source.set_save_state(true);
        let before = source.document_network().clone();
        let metadata_before = source.network_interface.document_network_metadata().clone();
        let hash_before = source.current_hash();
        let capture = OriginalDocumentCapture::capture(document_id, &source);
        let snapshot = capture
            .numeric_snapshot(81, node_id)
            .expect("genuine source numeric network");
        assert_eq!(snapshot.document_id_encoding(), document_id.0);
        assert_eq!(snapshot.node_id(), node_id);
        assert_eq!(snapshot.revision(), 81);
        assert_eq!(snapshot.network().generated, before.generated);
        assert_eq!(snapshot.network(), &before);
        assert_eq!(
            snapshot.network().exports,
            vec![NodeInput::node(node_id, 0)]
        );
        let source_node = source
            .document_network()
            .nodes
            .get(&node_id)
            .expect("genuine source Add");
        let snapshot_node = snapshot
            .network()
            .nodes
            .get(&node_id)
            .expect("exact retained source Add");
        assert_eq!(snapshot_node, source_node);
        for index in [add::AugendInput::INDEX, add::AddendInput::INDEX] {
            assert_eq!(
                snapshot_node.inputs.get(index),
                source_node.inputs.get(index)
            );
        }
        assert_eq!(snapshot.inputs().expect("source numeric inputs"), (2., 3.));
        let output = futures::executor::block_on(snapshot.evaluate_network())
            .expect("original merged constructor evaluation");
        assert_eq!(
            output.to_bits(),
            graphics_core::evaluate_add(2., 3.).to_bits()
        );
        assert_eq!(output.to_bits(), 5.0_f64.to_bits());
        assert_eq!(snapshot.network(), &before);
        assert_eq!(source.document_network(), &before);
        assert_eq!(
            source.network_interface.document_network_metadata(),
            &metadata_before
        );
        assert_eq!(source.current_hash(), hash_before);
        assert_eq!(capture.source_network_hash(), hash_before);
        assert_eq!(capture.network_interface().document_network(), &before);
        assert!(source.is_saved());
        assert!(source.storage().is_none());
    }

    #[test]
    fn at_r04_r18_original_capture_numeric_snapshot_rejects_unsupported_source_without_rewriting() {
        use graphics_core::document::NumericDocumentError;

        let node_id = NodeId(23);
        let mut source = numeric_document(node_id, 2., 3.);
        source.set_save_state(true);
        let default_before = source.document_network().clone();
        let default_hash = source.current_hash();
        let unconnected = OriginalDocumentCapture::capture(DocumentId(17), &source);
        assert!(matches!(
            unconnected.numeric_snapshot(4, node_id),
            Err(NumericDocumentError::InvalidNodeContract)
        ));
        assert_eq!(source.document_network(), &default_before);
        assert_eq!(
            unconnected.network_interface().document_network(),
            &default_before
        );
        assert_eq!(source.current_hash(), default_hash);
        assert!(source.is_saved());
        assert!(matches!(
            default_before.exports.first(),
            Some(NodeInput::Value { .. })
        ));
        assert!(matches!(
            unconnected.numeric_snapshot(4, NodeId(999)),
            Err(NumericDocumentError::MissingNode)
        ));
        let artboard = artboard_document(node_id);
        let artboard_before = artboard.document_network().clone();
        let artboard_capture = OriginalDocumentCapture::capture(DocumentId(18), &artboard);
        assert!(matches!(
            artboard_capture.numeric_snapshot(4, node_id),
            Err(NumericDocumentError::InvalidNodeContract)
        ));
        assert_eq!(artboard.document_network(), &artboard_before);
        assert_eq!(
            artboard_capture.network_interface().document_network(),
            &artboard_before
        );

        source.network_interface.set_input(
            &InputConnector::Export(0),
            NodeInput::node(node_id, 0),
            &[],
        );
        for (replacement, expected) in [
            (
                TaggedValue::String("wrong original literal type".to_owned()),
                NumericDocumentError::InvalidInputType,
            ),
            (
                TaggedValue::Number(f64::INFINITY),
                NumericDocumentError::NonFiniteValue,
            ),
            (
                TaggedValue::Number(f64::NAN),
                NumericDocumentError::NonFiniteValue,
            ),
        ] {
            let old = source
                .network_interface
                .input_from_connector(
                    &InputConnector::node_at_index(node_id, add::AugendInput::INDEX),
                    &[],
                )
                .expect("original source Augend");
            let NodeInput::Value { exposed, .. } = old else {
                panic!("source Augend is not a literal");
            };
            let exposed = *exposed;
            source.network_interface.set_input(
                &InputConnector::node_at_index(node_id, add::AugendInput::INDEX),
                NodeInput::value(replacement, exposed),
                &[],
            );
            let rejected_before = source.document_network().clone();
            let rejected_hash = source.current_hash();
            let rejected = OriginalDocumentCapture::capture(DocumentId(17), &source);
            let result = rejected.numeric_snapshot(4, node_id);
            match result {
                Err(error) => assert_eq!(error, expected),
                Ok(_) => panic!("invalid source unexpectedly admitted"),
            }
            assert_eq!(source.current_hash(), rejected_hash);
            let source_value = source
                .document_network()
                .nodes
                .get(&node_id)
                .expect("source Add")
                .inputs
                .get(add::AugendInput::INDEX)
                .and_then(NodeInput::as_value)
                .expect("source literal");
            let captured_value = rejected
                .network_interface()
                .document_node(&node_id, &[])
                .expect("captured source Add")
                .inputs
                .get(add::AugendInput::INDEX)
                .and_then(NodeInput::as_value)
                .expect("captured literal");
            match (source_value, captured_value) {
                (TaggedValue::Number(source), TaggedValue::Number(captured)) => {
                    assert_eq!(source.to_bits(), captured.to_bits())
                }
                (source, captured) => assert_eq!(source, captured),
            }
            assert_eq!(source.document_network().exports, rejected_before.exports);
            assert_eq!(
                source.document_network().nodes.len(),
                rejected_before.nodes.len()
            );
        }
        let extra_definition =
            resolve_proto_node_type(add::IDENTIFIER).expect("genuine second Add definition");
        assert!(
            source
                .network_interface
                .insert_node(NodeId(24), extra_definition.default_node_template(), &[])
                .is_none()
        );
        source.network_interface.set_input(
            &InputConnector::node_at_index(node_id, add::AugendInput::INDEX),
            NodeInput::value(TaggedValue::Number(2.), false),
            &[],
        );
        let two_nodes_before = source.document_network().clone();
        let two_nodes = OriginalDocumentCapture::capture(DocumentId(17), &source);
        assert!(matches!(
            two_nodes.numeric_snapshot(4, node_id),
            Err(NumericDocumentError::InvalidNodeContract)
        ));
        assert_eq!(source.document_network(), &two_nodes_before);
        assert_eq!(
            two_nodes.network_interface().document_network(),
            &two_nodes_before
        );
        assert!(source.storage().is_none() && artboard.storage().is_none());
    }

    #[test]
    fn at_r04_r18_original_capture_numeric_snapshots_survive_source_edits_and_colliding_ids() {
        let node_id = NodeId(23);
        let mut first = numeric_document(node_id, 2., 3.);
        let mut second = numeric_document(node_id, 20., 30.);
        for source in [&mut first, &mut second] {
            source.network_interface.set_input(
                &InputConnector::Export(0),
                NodeInput::node(node_id, 0),
                &[],
            );
            source.set_save_state(true);
        }
        let first_before = first.document_network().clone();
        let second_before = second.document_network().clone();
        let first_capture = OriginalDocumentCapture::capture(DocumentId(17), &first);
        let second_capture = OriginalDocumentCapture::capture(DocumentId(18), &second);
        let first_snapshot = first_capture
            .numeric_snapshot(7, node_id)
            .expect("first original source snapshot");
        let second_snapshot = second_capture
            .numeric_snapshot(7, node_id)
            .expect("second colliding node snapshot");
        assert_ne!(
            first_snapshot.document_id_encoding(),
            second_snapshot.document_id_encoding()
        );
        assert_eq!(first_snapshot.node_id(), second_snapshot.node_id());
        let connector = InputConnector::node_at_index(node_id, add::AugendInput::INDEX);
        let Some(NodeInput::Value { exposed, .. }) = first
            .network_interface
            .input_from_connector(&connector, &[])
        else {
            panic!("source literal Augend");
        };
        let exposed = *exposed;
        first.network_interface.set_input(
            &connector,
            NodeInput::value(TaggedValue::Number(9.), exposed),
            &[],
        );
        let edited_before = first.document_network().clone();
        let edited_capture = OriginalDocumentCapture::capture(DocumentId(17), &first);
        let edited_snapshot = edited_capture
            .numeric_snapshot(8, node_id)
            .expect("actual edited source snapshot");
        assert_eq!(
            futures::executor::block_on(first_snapshot.evaluate_network())
                .expect("old source evaluates"),
            5.
        );
        assert_eq!(
            futures::executor::block_on(second_snapshot.evaluate_network())
                .expect("second source evaluates"),
            50.
        );
        assert_eq!(
            futures::executor::block_on(edited_snapshot.evaluate_network())
                .expect("edited source evaluates"),
            12.
        );
        assert_eq!(first_snapshot.revision(), 7);
        assert_eq!(edited_snapshot.revision(), 8);
        assert_eq!(first_snapshot.network(), &first_before);
        assert_eq!(
            first_capture.network_interface().document_network(),
            &first_before
        );
        assert_eq!(edited_snapshot.network(), &edited_before);
        assert_eq!(first.document_network(), &edited_before);
        assert_eq!(second.document_network(), &second_before);
        assert_eq!(second_snapshot.network(), &second_before);
        assert!(second.is_saved());
        assert!(first.storage().is_none() && second.storage().is_none());
    }

    #[test]
    fn at_r04_r18_original_capture_numeric_snapshot_preserves_signed_zero_and_finite_overflow() {
        let node_id = NodeId(23);
        for (augend, addend) in [
            (-0.0_f64, -0.0_f64),
            (-0.0_f64, 0.0_f64),
            (f64::MAX, f64::MAX),
        ] {
            let mut source = numeric_document(node_id, augend, addend);
            source.network_interface.set_input(
                &InputConnector::Export(0),
                NodeInput::node(node_id, 0),
                &[],
            );
            let source_before = source.document_network().clone();
            let capture = OriginalDocumentCapture::capture(DocumentId(u64::MAX), &source);
            let snapshot = capture
                .numeric_snapshot(u64::MAX, node_id)
                .expect("finite exact source inputs");
            let (observed_augend, observed_addend) =
                snapshot.inputs().expect("original numeric pair");
            assert_eq!(observed_augend.to_bits(), augend.to_bits());
            assert_eq!(observed_addend.to_bits(), addend.to_bits());
            assert_eq!(snapshot.document_id_encoding(), u64::MAX);
            assert_eq!(snapshot.revision(), u64::MAX);
            let expected = graphics_core::evaluate_add(augend, addend);
            for _ in 0..2 {
                let output = futures::executor::block_on(snapshot.evaluate_network())
                    .expect("repeat original network evaluation");
                assert_eq!(output.to_bits(), expected.to_bits());
                if augend == f64::MAX {
                    assert_eq!(output, f64::INFINITY);
                }
            }
            assert_eq!(snapshot.network(), &source_before);
            assert_eq!(
                capture.network_interface().document_network(),
                &source_before
            );
            assert_eq!(source.document_network(), &source_before);
            assert!(source.storage().is_none());
        }
    }
    #[test]
    fn at_r04_r18_original_capture_hidden_numeric_node_preserves_source_passthrough_and_rejects_helper()
     {
        use graphics_core::document::NumericDocumentError;

        let node_id = NodeId(23);
        let mut source = numeric_document(node_id, 2., 3.);
        source.network_interface.set_input(
            &InputConnector::Export(0),
            NodeInput::node(node_id, 0),
            &[],
        );
        assert!(source.network_interface.is_visible(&node_id, &[]));
        let visible_node = source
            .document_network()
            .nodes
            .get(&node_id)
            .expect("original source Add")
            .clone();
        source
            .network_interface
            .set_visibility(&node_id, &[], false);
        assert!(!source.network_interface.is_visible(&node_id, &[]));
        let hidden_node = source
            .document_network()
            .nodes
            .get(&node_id)
            .expect("original hidden source Add");
        assert!(!hidden_node.visible);
        assert_eq!(hidden_node.inputs, visible_node.inputs);
        assert_eq!(hidden_node.implementation, visible_node.implementation);
        assert_eq!(hidden_node.call_argument, visible_node.call_argument);
        assert_eq!(hidden_node.context_features, visible_node.context_features);
        assert_eq!(
            hidden_node.skip_deduplication,
            visible_node.skip_deduplication
        );
        assert_eq!(
            hidden_node.original_location,
            visible_node.original_location
        );
        source.set_save_state(true);
        let network_before = source.document_network().clone();
        let metadata_before = source.network_interface.document_network_metadata().clone();
        let hash_before = source.current_hash();
        let capture = OriginalDocumentCapture::capture(DocumentId(17), &source);
        assert!(!capture.network_interface().is_visible(&node_id, &[]));
        let snapshot = capture
            .numeric_snapshot(9, node_id)
            .expect("exact original hidden numeric network");
        assert_eq!(snapshot.document_id_encoding(), 17);
        assert_eq!(snapshot.node_id(), node_id);
        assert_eq!(snapshot.revision(), 9);
        assert_eq!(snapshot.network().generated, network_before.generated);
        assert_eq!(snapshot.network(), &network_before);
        assert!(
            !snapshot
                .network()
                .nodes
                .get(&node_id)
                .expect("retained hidden Add")
                .visible
        );
        assert_eq!(
            snapshot.inputs().expect("retained source literals"),
            (2., 3.)
        );
        assert!(matches!(
            snapshot.evaluate(),
            Err(NumericDocumentError::InvalidNodeContract)
        ));
        let output = futures::executor::block_on(snapshot.evaluate_network())
            .expect("original hidden passthrough execution");
        assert_eq!(output.to_bits(), 2.0_f64.to_bits());
        assert_ne!(
            output.to_bits(),
            graphics_core::evaluate_add(2., 3.).to_bits()
        );
        assert_eq!(snapshot.network(), &network_before);
        assert_eq!(
            capture.network_interface().document_network(),
            &network_before
        );
        assert_eq!(source.document_network(), &network_before);
        assert_eq!(
            source.network_interface.document_network_metadata(),
            &metadata_before
        );
        assert_eq!(capture.source_network_hash(), hash_before);
        assert_eq!(source.current_hash(), hash_before);
        assert!(!source.network_interface.is_visible(&node_id, &[]));
        assert!(source.is_saved());
        assert!(source.storage().is_none());
    }
}

impl OriginalDocumentCapture {
    /// The revision is caller-supplied process-local context, not inferred from the observed hash.
    pub fn numeric_snapshot(
        &self,
        supplied_revision: u64,
        target_node_id: graphite_graph_craft::document::NodeId,
    ) -> Result<
        graphics_core::document::NumericAddSnapshot,
        graphics_core::document::NumericDocumentError,
    > {
        graphics_core::document::NumericAddSnapshot::from_network(
            self.document_id.0,
            target_node_id,
            supplied_revision,
            self.network_interface.document_network().clone(),
        )
    }
}
