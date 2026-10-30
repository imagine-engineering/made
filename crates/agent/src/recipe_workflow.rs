use anyhow::{Context as _, Result, ensure};
use collections::HashMap;
use serde::{Deserialize, Serialize};

pub const MAX_STEPS: usize = 32;
pub const MAX_STEP_OUTPUT: usize = 64 * 1024;
pub const MAX_WORKFLOW_OUTPUT: usize = 1024 * 1024;

pub fn deserialize_steps<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<RecipeStep>, D::Error> {
    struct Steps;
    impl<'de> serde::de::Visitor<'de> for Steps {
        type Value = Vec<RecipeStep>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("at most 32 recipe steps")
        }
        fn visit_seq<A: serde::de::SeqAccess<'de>>(
            self,
            mut sequence: A,
        ) -> std::result::Result<Self::Value, A::Error> {
            let mut steps = Vec::new();
            while let Some(step) = sequence.next_element()? {
                if steps.len() == MAX_STEPS {
                    return Err(serde::de::Error::custom("Recipe exceeds 32 steps"));
                }
                steps.push(step);
            }
            Ok(steps)
        }
    }
    deserializer.deserialize_seq(Steps)
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeStep {
    pub id: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub when: Option<Condition>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next: Option<String>,
    #[serde(default)]
    pub on_failure: FailurePolicy,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum FailurePolicy {
    #[default]
    Stop,
    Continue,
    GoTo(String),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operand {
    Literal { value: String },
    Parameter { name: String },
    Output { step: String },
    Status { step: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "operator", rename_all = "snake_case", deny_unknown_fields)]
pub enum Condition {
    Equals { left: Operand, right: Operand },
    NotEquals { left: Operand, right: Operand },
    Contains { left: Operand, right: Operand },
    All { conditions: Vec<Condition> },
    Any { conditions: Vec<Condition> },
    Not { condition: Box<Condition> },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Succeeded,
    Failed,
    Skipped,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepOutcome {
    pub status: StepStatus,
    pub output: Option<String>,
}

impl StepStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Skipped => "skipped",
        }
    }
}

impl Operand {
    fn validate(&self, recipe: &crate::recipes::Recipe, index: usize) -> Result<()> {
        match self {
            Self::Literal { value } => ensure!(
                value.len() <= MAX_STEP_OUTPUT,
                "Condition literal exceeds 64 KiB"
            ),
            Self::Parameter { name } => ensure!(
                recipe
                    .parameters
                    .iter()
                    .any(|parameter| &parameter.name == name),
                "Unknown condition parameter: {name}"
            ),
            Self::Output { step } | Self::Status { step } => {
                ensure!(
                    recipe
                        .steps
                        .iter()
                        .take(index)
                        .any(|previous| &previous.id == step),
                    "Condition must reference a prior step: {step}"
                );
            }
        }
        Ok(())
    }

    fn resolve<'a>(
        &'a self,
        parameters: &'a HashMap<String, String>,
        outcomes: &'a HashMap<String, StepOutcome>,
    ) -> Result<&'a str> {
        match self {
            Self::Literal { value } => Ok(value),
            Self::Parameter { name } => parameters
                .get(name)
                .map(String::as_str)
                .context("Condition parameter is unavailable"),
            Self::Output { step } => outcomes
                .get(step)
                .and_then(|outcome| outcome.output.as_deref())
                .with_context(|| format!("Output of step {step} is unavailable")),
            Self::Status { step } => outcomes
                .get(step)
                .map(|outcome| outcome.status.as_str())
                .with_context(|| format!("Status of step {step} is unavailable")),
        }
    }
}

impl Condition {
    fn validate(&self, recipe: &crate::recipes::Recipe, index: usize, depth: usize) -> Result<()> {
        ensure!(depth <= 8, "Conditions exceed maximum nesting depth of 8");
        match self {
            Self::Equals { left, right }
            | Self::NotEquals { left, right }
            | Self::Contains { left, right } => {
                left.validate(recipe, index)?;
                right.validate(recipe, index)?;
            }
            Self::All { conditions } | Self::Any { conditions } => {
                ensure!(
                    !conditions.is_empty() && conditions.len() <= MAX_STEPS,
                    "Boolean conditions require 1–32 operands"
                );
                for condition in conditions {
                    condition.validate(recipe, index, depth + 1)?;
                }
            }
            Self::Not { condition } => condition.validate(recipe, index, depth + 1)?,
        }
        Ok(())
    }

    pub fn evaluate(
        &self,
        parameters: &HashMap<String, String>,
        outcomes: &HashMap<String, StepOutcome>,
    ) -> Result<bool> {
        match self {
            Self::Equals { left, right } => {
                Ok(left.resolve(parameters, outcomes)? == right.resolve(parameters, outcomes)?)
            }
            Self::NotEquals { left, right } => {
                Ok(left.resolve(parameters, outcomes)? != right.resolve(parameters, outcomes)?)
            }
            Self::Contains { left, right } => Ok(left
                .resolve(parameters, outcomes)?
                .contains(right.resolve(parameters, outcomes)?)),
            Self::All { conditions } => {
                for condition in conditions {
                    if !condition.evaluate(parameters, outcomes)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            Self::Any { conditions } => {
                for condition in conditions {
                    if condition.evaluate(parameters, outcomes)? {
                        return Ok(true);
                    }
                }
                Ok(false)
            }
            Self::Not { condition } => Ok(!condition.evaluate(parameters, outcomes)?),
        }
    }
}

pub fn validate(recipe: &crate::recipes::Recipe) -> Result<()> {
    if recipe.version == 1 {
        ensure!(recipe.steps.is_empty(), "Steps require recipe version 2");
        return Ok(());
    }
    ensure!(
        !recipe.steps.is_empty() && recipe.steps.len() <= MAX_STEPS,
        "Version-2 recipes require 1–32 steps"
    );
    ensure!(
        recipe.prompt.is_empty(),
        "Version-2 recipes use step prompts, not a top-level prompt"
    );
    let mut ids = collections::HashSet::default();
    for (index, step) in recipe.steps.iter().enumerate() {
        ensure!(
            crate::recipes::valid_name(&step.id),
            "Step IDs require 1–64 lowercase letters, digits or hyphens"
        );
        ensure!(ids.insert(&step.id), "Duplicate step ID: {}", step.id);
        ensure!(
            !step.prompt.trim().is_empty(),
            "Step {} requires a prompt",
            step.id
        );
        for target in step.next.iter().chain(match &step.on_failure {
            FailurePolicy::GoTo(target) => Some(target),
            _ => None,
        }) {
            ensure!(
                recipe
                    .steps
                    .iter()
                    .skip(index + 1)
                    .any(|step| &step.id == target),
                "Step {} target must name a later step: {target}",
                step.id
            );
        }
        if let Some(condition) = &step.when {
            condition.validate(recipe, index, 1)?;
        }
        crate::recipes::render_template(&step.prompt, |name| {
            if let Some(id) = output_reference(name) {
                ensure!(
                    recipe.steps.iter().take(index).any(|step| step.id == id),
                    "Output must reference a prior step: {id}"
                );
            } else {
                ensure!(
                    recipe
                        .parameters
                        .iter()
                        .any(|parameter| parameter.name == name),
                    "Undeclared recipe parameter: {name}"
                );
            }
            Ok(String::new())
        })?;
    }
    Ok(())
}

fn output_reference(name: &str) -> Option<&str> {
    name.strip_prefix("steps.")?.strip_suffix(".output")
}

pub fn render_step(
    step: &RecipeStep,
    parameters: &HashMap<String, String>,
    outcomes: &HashMap<String, StepOutcome>,
) -> Result<String> {
    crate::recipes::render_template(&step.prompt, |name| {
        if let Some(id) = output_reference(name) {
            outcomes
                .get(id)
                .and_then(|outcome| outcome.output.clone())
                .with_context(|| format!("Output of step {id} is unavailable"))
        } else {
            parameters
                .get(name)
                .cloned()
                .with_context(|| format!("Missing recipe parameter: {name}"))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipes::Recipe;

    const WORKFLOW: &str = "version: 2\ntitle: Review\ndescription: Review then report\nsteps:\n- id: review\n  prompt: Review this project\n- id: report\n  prompt: 'Explain {{steps.review.output}}'\n";

    #[test]
    fn recipe_workflow_round_trip_and_literal_output() -> Result<()> {
        let recipe = Recipe::from_yaml(WORKFLOW)?;
        assert_eq!(Recipe::from_yaml(&recipe.to_yaml()?)?, recipe);
        let outcomes = HashMap::from_iter([(
            "review".into(),
            StepOutcome {
                status: StepStatus::Succeeded,
                output: Some("{{secret}} $(touch /host)".into()),
            },
        )]);
        let step = recipe.steps.last().context("report")?;
        assert_eq!(
            render_step(step, &HashMap::default(), &outcomes)?,
            "Explain {{secret}} $(touch /host)"
        );
        assert!(render_step(step, &HashMap::default(), &HashMap::default()).is_err());
        Ok(())
    }

    #[test]
    fn recipe_workflow_rejects_cycles_and_future_outputs() {
        for yaml in [
            WORKFLOW.replace("id: report", "id: review"),
            WORKFLOW.replace("id: report", "id: report\n  next: review"),
            WORKFLOW.replace("Review this project", "'{{steps.report.output}}'"),
            WORKFLOW.replace("version: 2", "version: 1"),
            WORKFLOW.replace(
                "  prompt: Review this project",
                "  prompt: Review this project\n  tools: [terminal]",
            ),
        ] {
            assert!(Recipe::from_yaml(&yaml).is_err());
        }
    }

    #[test]
    fn recipe_workflow_conditions_require_actual_results() -> Result<()> {
        let condition = Condition::Equals {
            left: Operand::Status {
                step: "review".into(),
            },
            right: Operand::Literal {
                value: "failed".into(),
            },
        };
        assert!(
            condition
                .evaluate(&HashMap::default(), &HashMap::default())
                .is_err()
        );
        let outcomes = HashMap::from_iter([(
            "review".into(),
            StepOutcome {
                status: StepStatus::Failed,
                output: None,
            },
        )]);
        assert!(condition.evaluate(&HashMap::default(), &outcomes)?);
        Ok(())
    }

    #[test]
    fn recipe_workflow_bounds_and_failure_targets() -> Result<()> {
        let mut recipe = Recipe::from_yaml(WORKFLOW)?;
        let first = recipe.steps.first_mut().context("first")?;
        first.on_failure = FailurePolicy::GoTo("report".into());
        recipe.validate()?;
        assert_eq!(Recipe::from_yaml(&recipe.to_yaml()?)?, recipe);
        let first = recipe.steps.first_mut().context("first")?;
        first.on_failure = FailurePolicy::GoTo("missing".into());
        assert!(recipe.validate().is_err());
        first_reset(&mut recipe)?;
        let mut condition = Condition::Equals {
            left: Operand::Literal { value: "a".into() },
            right: Operand::Literal { value: "a".into() },
        };
        for _ in 0..8 {
            condition = Condition::Not {
                condition: Box::new(condition),
            };
        }
        recipe.steps.first_mut().context("first")?.when = Some(condition);
        assert!(recipe.validate().is_err());
        let many = format!(
            "version: 2\ntitle: Bounded\ndescription: Bounded\nsteps:\n{}",
            (0..33)
                .map(|index| format!("- id: step-{index}\n  prompt: review\n"))
                .collect::<String>()
        );
        assert!(Recipe::from_yaml(&many).is_err());
        Ok(())
    }

    fn first_reset(recipe: &mut Recipe) -> Result<()> {
        recipe.steps.first_mut().context("first")?.on_failure = FailurePolicy::Stop;
        Ok(())
    }
}
