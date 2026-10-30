use std::{cell::Cell, rc::Rc, sync::Arc, time::Duration};

use agent_client_protocol::schema::v1 as acp;
use anyhow::{Context as _, Result, ensure};
use futures::{FutureExt as _, select_biased};
use gpui::{App, SharedString, Task, WeakEntity};
use language_model::{LanguageModelRegistry, SelectedModel};

use crate::recipes::{RecipeProposalInput, RecipeReviewStatus, RecipeStore};
use crate::{AgentTool, Thread, ToolCallEventStream, ToolInput};

pub struct ProposeRecipeTool {
    thread: WeakEntity<Thread>,
}

impl ProposeRecipeTool {
    pub fn new(thread: WeakEntity<Thread>) -> Self {
        Self { thread }
    }
}

struct PendingReview {
    status: Rc<Cell<RecipeReviewStatus>>,
    acknowledged: bool,
}

impl Drop for PendingReview {
    fn drop(&mut self) {
        if !self.acknowledged
            && matches!(
                self.status.get(),
                RecipeReviewStatus::Pending | RecipeReviewStatus::Opened
            )
        {
            self.status.set(RecipeReviewStatus::Cancelled);
        }
    }
}

impl AgentTool for ProposeRecipeTool {
    type Input = RecipeProposalInput;
    type Output = String;
    const NAME: &'static str = "propose_recipe";

    fn kind() -> acp::ToolKind {
        acp::ToolKind::Other
    }

    fn initial_title(
        &self,
        _: Result<Self::Input, serde_json::Value>,
        _: &mut App,
    ) -> SharedString {
        "Draft recipe for review".into()
    }

    fn run(
        self: Arc<Self>,
        input: ToolInput<Self::Input>,
        events: ToolCallEventStream,
        cx: &mut App,
    ) -> Task<Result<String, String>> {
        cx.spawn(async move |cx| {
            let result: Result<String> = async {
                let input = input.recv().await?;
                ensure!(
                    !events.was_cancelled_by_user(),
                    "Recipe draft request cancelled"
                );
                let mut pending = PendingReview {
                    status: Rc::new(Cell::new(RecipeReviewStatus::Pending)),
                    acknowledged: false,
                };
                let task = cx.update(|cx| {
                    let thread = self
                        .thread
                        .upgrade()
                        .context("Conversation is unavailable")?;
                    ensure!(
                        thread.read(cx).can_propose_recipe(),
                        "Recipe drafts require an interactive root conversation"
                    );
                    let project = thread.read(cx).project().clone();
                    let selected = thread.read(cx).model().map(|model| SelectedModel {
                        provider: model.provider_id(),
                        model: model.id(),
                    });
                    let model = selected.and_then(|selected| {
                        LanguageModelRegistry::global(cx)
                            .update(cx, |registry, cx| registry.select_model(&selected, cx))
                    });
                    let store =
                        RecipeStore::try_global(cx).context("Recipe management is unavailable")?;
                    Ok::<_, anyhow::Error>(store.update(cx, |store, cx| {
                        store.propose(
                            thread.entity_id(),
                            project,
                            input,
                            model,
                            pending.status.clone(),
                            cx,
                        )
                    }))
                })?;
                let proposal = task.await?;

                let wait = async {
                    while proposal.status.get() == RecipeReviewStatus::Pending {
                        cx.background_executor()
                            .timer(Duration::from_millis(25))
                            .await;
                    }
                }
                .fuse();
                let timeout = cx
                    .background_executor()
                    .timer(Duration::from_secs(10))
                    .fuse();
                let cancelled = events.cancelled_by_user().fuse();
                futures::pin_mut!(wait, timeout, cancelled);
                select_biased! {
                    _ = cancelled => { anyhow::bail!("Recipe draft request cancelled"); }
                    _ = wait => {}
                    _ = timeout => { proposal.status.set(RecipeReviewStatus::Unavailable); }
                }

                ensure!(
                    matches!(
                        proposal.status.get(),
                        RecipeReviewStatus::Opened | RecipeReviewStatus::Dismissed
                    ),
                    "{}",
                    proposal.message()
                );
                pending.acknowledged = true;
                Ok(proposal.message())
            }
            .await;
            result.map_err(|error| error.to_string())
        })
    }
}
