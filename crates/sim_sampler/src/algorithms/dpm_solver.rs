use sim_tensor::{
    CpuBackend, CpuWorkspaceVec, ExecutionContext, Tensor, TensorError,
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32},
};

pub(crate) struct DpmSolverEvaluation {
    pub denoised: Tensor,
    pub epsilon: CpuWorkspaceVec<f32>,
}

pub(crate) struct DpmSolverFirstIntermediate {
    pub evaluation: DpmSolverEvaluation,
}

pub(crate) struct DpmSolverThirdOrder {
    pub values: CpuWorkspaceVec<f32>,
    pub second_evaluation: DpmSolverEvaluation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DpmSolverStage {
    FirstOrder,
    FirstIntermediate,
    SecondDifference,
    SecondOrder,
    ThirdFirstDifference,
    ThirdIntermediate,
    ThirdSecondDifference,
    ThirdOrder,
}

#[derive(Debug)]
pub(crate) enum DpmSolverEquationError {
    Tensor(TensorError),
    TensorKernel(NativeDiffusionTensorError),
    Shape(DpmSolverStage),
    NonFinite {
        stage: DpmSolverStage,
        element: usize,
    },
}

#[derive(Debug)]
pub(crate) enum DpmSolverStepError<E> {
    Equation(DpmSolverEquationError),
    Evaluation(E),
}

pub(crate) fn dpm_solver_first_order(
    backend: &CpuBackend,
    current: &[f32],
    epsilon: &[f32],
    time: f32,
    next_time: f32,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, DpmSolverEquationError> {
    let step = next_time - time;
    let coefficient = -(-next_time).exp() * step.exp_m1();
    combine(
        backend,
        current,
        &[(epsilon, coefficient)],
        DpmSolverStage::FirstOrder,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dpm_solver_first_intermediate<E>(
    backend: &CpuBackend,
    template: &Tensor,
    current: &[f32],
    epsilon: &[f32],
    time: f32,
    next_time: f32,
    ratio: f32,
    context: &ExecutionContext<'_>,
    evaluate: &mut impl FnMut(&Tensor, f32, u8) -> Result<DpmSolverEvaluation, E>,
) -> Result<DpmSolverFirstIntermediate, DpmSolverStepError<E>> {
    let step = next_time - time;
    let intermediate_time = time + ratio * step;
    let coefficient = -(-intermediate_time).exp() * (ratio * step).exp_m1();
    let values = combine(
        backend,
        current,
        &[(epsilon, coefficient)],
        DpmSolverStage::FirstIntermediate,
        context,
    )
    .map_err(DpmSolverStepError::Equation)?;
    let input = tensor_from_f32(backend, template.descriptor().shape(), &values, context)
        .map_err(DpmSolverEquationError::TensorKernel)
        .map_err(DpmSolverStepError::Equation)?;
    let evaluation =
        evaluate(&input, intermediate_time, 1).map_err(DpmSolverStepError::Evaluation)?;
    Ok(DpmSolverFirstIntermediate { evaluation })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dpm_solver_second_order(
    backend: &CpuBackend,
    current: &[f32],
    epsilon: &[f32],
    intermediate_epsilon: &[f32],
    time: f32,
    next_time: f32,
    ratio: f32,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, DpmSolverEquationError> {
    let step = next_time - time;
    let sigma_next = (-next_time).exp();
    let epsilon_coefficient = -sigma_next * step.exp_m1();
    let difference_coefficient = -sigma_next / (2.0 * ratio) * step.exp_m1();
    let difference = difference(
        backend,
        intermediate_epsilon,
        epsilon,
        DpmSolverStage::SecondDifference,
        context,
    )?;
    combine(
        backend,
        current,
        &[
            (epsilon, epsilon_coefficient),
            (&difference, difference_coefficient),
        ],
        DpmSolverStage::SecondOrder,
        context,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dpm_solver_third_order<E>(
    backend: &CpuBackend,
    template: &Tensor,
    current: &[f32],
    epsilon: &[f32],
    first_epsilon: &[f32],
    time: f32,
    next_time: f32,
    context: &ExecutionContext<'_>,
    evaluate: &mut impl FnMut(&Tensor, f32, u8) -> Result<DpmSolverEvaluation, E>,
) -> Result<DpmSolverThirdOrder, DpmSolverStepError<E>> {
    let step = next_time - time;
    let first_ratio = 1.0 / 3.0;
    let second_ratio: f32 = 2.0 / 3.0;
    let second_time = time + second_ratio * step;
    let sigma_second = (-second_time).exp();
    let first_difference = difference(
        backend,
        first_epsilon,
        epsilon,
        DpmSolverStage::ThirdFirstDifference,
        context,
    )
    .map_err(DpmSolverStepError::Equation)?;
    let scaled_step = second_ratio * step;
    let base_coefficient = -sigma_second * scaled_step.exp_m1();
    let phi = scaled_step.exp_m1() / scaled_step - 1.0;
    let difference_coefficient = -sigma_second * (second_ratio / first_ratio) * phi;
    let second_values = combine(
        backend,
        current,
        &[
            (epsilon, base_coefficient),
            (&first_difference, difference_coefficient),
        ],
        DpmSolverStage::ThirdIntermediate,
        context,
    )
    .map_err(DpmSolverStepError::Equation)?;
    let second_input = tensor_from_f32(
        backend,
        template.descriptor().shape(),
        &second_values,
        context,
    )
    .map_err(DpmSolverEquationError::TensorKernel)
    .map_err(DpmSolverStepError::Equation)?;
    let second_evaluation =
        evaluate(&second_input, second_time, 2).map_err(DpmSolverStepError::Evaluation)?;
    let second_difference = difference(
        backend,
        &second_evaluation.epsilon,
        epsilon,
        DpmSolverStage::ThirdSecondDifference,
        context,
    )
    .map_err(DpmSolverStepError::Equation)?;
    let sigma_next = (-next_time).exp();
    let epsilon_coefficient = -sigma_next * step.exp_m1();
    let phi = step.exp_m1() / step - 1.0;
    let difference_coefficient = -sigma_next / second_ratio * phi;
    let values = combine(
        backend,
        current,
        &[
            (epsilon, epsilon_coefficient),
            (&second_difference, difference_coefficient),
        ],
        DpmSolverStage::ThirdOrder,
        context,
    )
    .map_err(DpmSolverStepError::Equation)?;
    Ok(DpmSolverThirdOrder {
        values,
        second_evaluation,
    })
}

fn difference(
    backend: &CpuBackend,
    left: &[f32],
    right: &[f32],
    stage: DpmSolverStage,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, DpmSolverEquationError> {
    if left.len() != right.len() {
        return Err(DpmSolverEquationError::Shape(stage));
    }
    let mut output = backend
        .workspace_vec(context, left.len())
        .map_err(DpmSolverEquationError::Tensor)?;
    for (element, (left, right)) in left.iter().zip(right.iter()).enumerate() {
        if element.is_multiple_of(256) {
            context.check().map_err(DpmSolverEquationError::Tensor)?;
        }
        let value = left - right;
        ensure_finite(value, stage, element)?;
        output
            .try_push(value)
            .map_err(DpmSolverEquationError::Tensor)?;
    }
    Ok(output)
}

fn combine(
    backend: &CpuBackend,
    base: &[f32],
    terms: &[(&[f32], f32)],
    stage: DpmSolverStage,
    context: &ExecutionContext<'_>,
) -> Result<CpuWorkspaceVec<f32>, DpmSolverEquationError> {
    if terms.iter().any(|(values, _)| values.len() != base.len()) {
        return Err(DpmSolverEquationError::Shape(stage));
    }
    let mut output = backend
        .workspace_vec(context, base.len())
        .map_err(DpmSolverEquationError::Tensor)?;
    for element in 0..base.len() {
        if element.is_multiple_of(256) {
            context.check().map_err(DpmSolverEquationError::Tensor)?;
        }
        let mut value = *base
            .get(element)
            .ok_or(DpmSolverEquationError::Shape(stage))?;
        for (values, coefficient) in terms {
            let term = values
                .get(element)
                .ok_or(DpmSolverEquationError::Shape(stage))?;
            value += coefficient * term;
        }
        ensure_finite(value, stage, element)?;
        output
            .try_push(value)
            .map_err(DpmSolverEquationError::Tensor)?;
    }
    Ok(output)
}

fn ensure_finite(
    value: f32,
    stage: DpmSolverStage,
    element: usize,
) -> Result<(), DpmSolverEquationError> {
    if !value.is_finite() {
        return Err(DpmSolverEquationError::NonFinite { stage, element });
    }
    Ok(())
}
