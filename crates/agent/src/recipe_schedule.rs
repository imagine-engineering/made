use crate::recipes::Recipe;
use anyhow::{Context as _, Result, ensure};
use chrono::{DateTime, LocalResult, TimeZone as _, Utc};
use collections::HashMap;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr as _;
use util::path_list::SerializedPathList;
use uuid::Uuid;

pub fn next_run(cron: &str, timezone: &str, after: DateTime<Utc>) -> Result<DateTime<Utc>> {
    ensure!(
        cron.split_whitespace().count() == 5,
        "Use five cron fields: minute hour day month weekday"
    );
    let schedule = cron::Schedule::from_str(&format!("0 {cron}"))
        .map_err(|_| anyhow::anyhow!("Invalid cron expression"))?;
    let timezone = chrono_tz::Tz::from_str(timezone)
        .map_err(|_| anyhow::anyhow!("Choose an IANA time zone, such as Europe/London"))?;
    let wall_time = after.with_timezone(&timezone).naive_local().and_utc();
    for candidate in schedule.after(&wall_time) {
        let instant = match timezone.from_local_datetime(&candidate.naive_utc()) {
            LocalResult::None => continue,
            LocalResult::Single(instant) => instant,
            LocalResult::Ambiguous(first, second) => first.min(second),
        }
        .with_timezone(&Utc);
        if instant > after {
            return Ok(instant);
        }
    }
    anyhow::bail!("This cron expression has no future occurrence")
}

pub fn local_timezone() -> Result<String> {
    let timezone = iana_time_zone::get_timezone()
        .context("Could not determine the local time zone; choose an IANA zone")?;
    chrono_tz::Tz::from_str(&timezone)
        .context("Local time zone is not supported; choose an IANA zone")?;
    Ok(timezone)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeSchedule {
    pub id: Uuid,
    pub title: String,
    pub project_paths: SerializedPathList,
    pub worktree: PathBuf,
    pub cron: String,
    pub timezone: String,
    pub recipe: Recipe,
    pub parameters: HashMap<String, String>,
    pub provider: String,
    pub model: String,
    pub paused: bool,
    pub consented: bool,
    pub deleted: bool,
    pub next_run: DateTime<Utc>,
    pub last_run: Option<DateTime<Utc>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecipeRunStatus {
    Admitted,
    Running,
    Succeeded,
    Failed,
    Cancelled,
    Interrupted,
    NeedsAttention,
    Missed,
    SkippedOverlap,
}

impl RecipeRunStatus {
    pub fn is_active(self) -> bool {
        matches!(self, Self::Admitted | Self::Running)
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Admitted => "Starting",
            Self::Running => "Running",
            Self::Succeeded => "Completed",
            Self::Failed => "Failed",
            Self::Cancelled => "Cancelled",
            Self::Interrupted => "Interrupted",
            Self::NeedsAttention => "Needs attention",
            Self::Missed => "Missed",
            Self::SkippedOverlap => "Skipped: another schedule was running",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeRun {
    pub id: Uuid,
    pub schedule_id: Uuid,
    pub scheduled_for: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub finished_at: Option<DateTime<Utc>>,
    pub session_id: Option<String>,
    pub status: RecipeRunStatus,
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecipeScheduleState {
    pub version: u32,
    pub schedules: Vec<RecipeSchedule>,
    pub runs: Vec<RecipeRun>,
    #[serde(default)]
    pub scans: Vec<crate::recipe_scan::RecipeScanRecord>,
}

impl Default for RecipeScheduleState {
    fn default() -> Self {
        Self {
            version: 1,
            schedules: Vec::new(),
            runs: Vec::new(),
            scans: Vec::new(),
        }
    }
}

impl RecipeScheduleState {
    pub fn active_run(&self) -> Option<&RecipeRun> {
        self.runs.iter().find(|run| run.status.is_active())
    }

    pub fn save_schedule(
        &mut self,
        mut schedule: RecipeSchedule,
        now: DateTime<Utc>,
    ) -> Result<()> {
        ensure!(
            !schedule.title.trim().is_empty(),
            "Give the schedule a title"
        );
        ensure!(
            schedule.worktree.is_absolute(),
            "Choose an absolute worktree path"
        );
        ensure!(
            !schedule.provider.is_empty() && !schedule.model.is_empty(),
            "Choose a provider and model"
        );
        schedule.parameters = schedule.recipe.resolve_parameters(&schedule.parameters)?;
        schedule.recipe.render(&schedule.parameters)?;
        schedule.next_run = next_run(&schedule.cron, &schedule.timezone, now)?;
        schedule.deleted = false;
        schedule.paused = true;
        schedule.consented = false;
        if let Some(existing) = self
            .schedules
            .iter_mut()
            .find(|entry| entry.id == schedule.id)
        {
            ensure!(!existing.deleted, "This schedule was deleted");
            schedule.last_run = existing.last_run;
            *existing = schedule;
        } else {
            self.schedules.push(schedule);
        }
        Ok(())
    }

    pub fn set_enabled(
        &mut self,
        id: Uuid,
        enabled: bool,
        consent: bool,
        now: DateTime<Utc>,
    ) -> Result<()> {
        let schedule = self
            .schedules
            .iter_mut()
            .find(|entry| entry.id == id && !entry.deleted)
            .context("Schedule is unavailable")?;
        if enabled {
            ensure!(
                consent,
                "Review the recipe, project, model and permissions before enabling this schedule"
            );
            schedule.next_run = next_run(&schedule.cron, &schedule.timezone, now)?;
            schedule.consented = true;
        }
        schedule.paused = !enabled;
        Ok(())
    }

    pub fn admit(&mut self, id: Uuid, now: DateTime<Utc>) -> Result<Uuid> {
        ensure!(
            self.active_run().is_none(),
            "Another scheduled run is active"
        );
        let schedule = self
            .schedules
            .iter_mut()
            .find(|entry| entry.id == id && !entry.deleted)
            .context("Schedule is unavailable")?;
        ensure!(
            schedule.consented,
            "Review and enable this schedule before running it"
        );
        let run_id = Uuid::new_v4();
        schedule.last_run = Some(now);
        self.runs.push(RecipeRun {
            id: run_id,
            schedule_id: id,
            scheduled_for: now,
            started_at: Some(now),
            finished_at: None,
            session_id: None,
            status: RecipeRunStatus::Admitted,
            detail: None,
        });
        Ok(run_id)
    }

    pub fn finish(
        &mut self,
        id: Uuid,
        status: RecipeRunStatus,
        detail: Option<String>,
        now: DateTime<Utc>,
    ) -> Result<()> {
        ensure!(
            !status.is_active(),
            "A finished run needs a terminal status"
        );
        let run = self
            .runs
            .iter_mut()
            .find(|run| run.id == id)
            .context("Run is unavailable")?;
        ensure!(run.status.is_active(), "Run has already finished");
        run.status = status;
        run.detail = detail;
        run.finished_at = Some(now);
        if status == RecipeRunStatus::NeedsAttention {
            if let Some(schedule) = self
                .schedules
                .iter_mut()
                .find(|schedule| schedule.id == run.schedule_id)
            {
                schedule.paused = true;
                schedule.consented = false;
            }
        }
        Ok(())
    }

    pub fn recover(&mut self, now: DateTime<Utc>) -> Result<()> {
        for scan in &mut self.scans {
            scan.recover(now)?;
        }
        for run in &mut self.runs {
            if run.status.is_active() {
                run.status = RecipeRunStatus::Interrupted;
                run.finished_at = Some(now);
                run.detail =
                    Some("Made closed before this run finished. It will not be replayed.".into());
            }
        }
        self.tick(now, true)?;
        Ok(())
    }

    pub fn tick(&mut self, now: DateTime<Utc>, missed: bool) -> Result<Option<Uuid>> {
        let mut occupied = self.active_run().is_some();
        let mut admitted = None;
        for schedule in &mut self.schedules {
            if schedule.deleted || schedule.paused || !schedule.consented || schedule.next_run > now
            {
                continue;
            }
            let scheduled_for = schedule.next_run;
            schedule.next_run = next_run(&schedule.cron, &schedule.timezone, now)?;
            let status = if missed || now.signed_duration_since(scheduled_for).num_seconds() >= 60 {
                RecipeRunStatus::Missed
            } else if occupied {
                RecipeRunStatus::SkippedOverlap
            } else {
                occupied = true;
                RecipeRunStatus::Admitted
            };
            let id = Uuid::new_v4();
            if status.is_active() {
                admitted = Some(id);
                schedule.last_run = Some(now);
            }
            self.runs.push(RecipeRun {
                id,
                schedule_id: schedule.id,
                scheduled_for,
                started_at: status.is_active().then_some(now),
                finished_at: (!status.is_active()).then_some(now),
                session_id: None,
                status,
                detail: (status == RecipeRunStatus::Missed).then(|| {
                    format!("Missed occurrences from {scheduled_for} through {now}; no catch-up work was started.")
                }),
            });
        }
        Ok(admitted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn instant(value: &str) -> DateTime<Utc> {
        DateTime::parse_from_rfc3339(value)
            .expect("test instant")
            .with_timezone(&Utc)
    }

    fn schedule(now: DateTime<Utc>) -> RecipeSchedule {
        RecipeSchedule {
            id: Uuid::new_v4(),
            title: "Review".into(),
            project_paths: util::path_list::PathList::new(&["/project"]).serialize(),
            worktree: "/project".into(),
            cron: "* * * * *".into(),
            timezone: "UTC".into(),
            recipe: crate::recipes::builtin_recipes()
                .pop()
                .expect("built-in")
                .recipe
                .expect("valid"),
            parameters: HashMap::default(),
            provider: "fake".into(),
            model: "fake".into(),
            paused: false,
            consented: true,
            deleted: false,
            next_run: now,
            last_run: None,
        }
    }

    #[test]
    fn recipe_schedule_workflow_snapshot_survives_restart_and_requires_new_consent() -> Result<()> {
        let now = instant("2026-09-12T12:00:00Z");
        let mut workflow = schedule(now);
        workflow.recipe = Recipe::from_yaml(
            "version: 2\ntitle: Review\ndescription: Test saved workflow\nsteps:\n  - id: inspect\n    prompt: Inspect saved changes\n  - id: report\n    prompt: 'Report {{steps.inspect.output}}'\n",
        )?;
        let id = workflow.id;
        let snapshot = workflow.recipe.clone();
        let mut state = RecipeScheduleState::default();
        state.save_schedule(workflow.clone(), now)?;
        state.save_schedule(schedule(now), now)?;
        state.set_enabled(id, true, true, now)?;
        let run_id = state.admit(id, now)?;
        state.runs.first_mut().expect("run").session_id = Some("workflow-session".into());
        workflow.recipe.steps.first_mut().expect("step").prompt = "Changed source".into();
        let mut restored: RecipeScheduleState =
            serde_json::from_str(&serde_json::to_string(&state)?)?;
        assert_eq!(
            restored.schedules.first().expect("snapshot").recipe,
            snapshot
        );
        assert_eq!(restored.schedules.last().expect("legacy").recipe.version, 1);
        restored.recover(now + chrono::Duration::hours(1))?;
        let run = restored
            .runs
            .iter()
            .find(|run| run.id == run_id)
            .expect("history");
        assert_eq!(run.status, RecipeRunStatus::Interrupted);
        assert_eq!(run.session_id.as_deref(), Some("workflow-session"));
        assert!(restored.active_run().is_none());
        restored.save_schedule(workflow, now + chrono::Duration::hours(1))?;
        assert!(restored.admit(id, now).is_err());
        assert!(restored.set_enabled(id, true, false, now).is_err());
        Ok(())
    }

    #[test]
    fn recipe_schedule_admission_overlap_and_no_catch_up() -> Result<()> {
        let now = instant("2026-09-12T12:00:00Z");
        let first = schedule(now);
        let second = schedule(now);
        let mut state = RecipeScheduleState {
            schedules: vec![first, second],
            ..Default::default()
        };
        let id = state.tick(now, false)?.expect("admission");
        assert_eq!(state.runs.len(), 2);
        assert_eq!(
            state.runs.last().expect("overlap").status,
            RecipeRunStatus::SkippedOverlap
        );
        assert!(
            state
                .admit(state.schedules.first().expect("schedule").id, now)
                .is_err()
        );
        assert!(state.tick(now, false)?.is_none());
        state.finish(id, RecipeRunStatus::Succeeded, None, now)?;
        assert!(
            state
                .tick(now + chrono::Duration::hours(2), true)?
                .is_none()
        );
        assert_eq!(state.runs.len(), 4);
        assert!(
            state
                .runs
                .iter()
                .skip(2)
                .all(|run| run.status == RecipeRunStatus::Missed)
        );
        assert!(
            state
                .schedules
                .iter()
                .all(|schedule| schedule.next_run > now + chrono::Duration::hours(2))
        );
        Ok(())
    }

    #[test]
    fn recipe_recovery_keeps_session_and_never_replays() -> Result<()> {
        let now = instant("2026-09-12T12:00:00Z");
        let mut state = RecipeScheduleState {
            schedules: vec![schedule(now)],
            ..Default::default()
        };
        let id = state.tick(now, false)?.expect("admission");
        state.runs.first_mut().expect("run").session_id = Some("native-session".into());
        state.recover(now + chrono::Duration::days(3))?;
        let run = state
            .runs
            .iter()
            .find(|run| run.id == id)
            .expect("recovered run");
        assert_eq!(run.status, RecipeRunStatus::Interrupted);
        assert_eq!(run.session_id.as_deref(), Some("native-session"));
        assert!(state.active_run().is_none());
        assert_eq!(
            state.runs.last().expect("missed").status,
            RecipeRunStatus::Missed
        );
        let count = state.runs.len();
        state.recover(now + chrono::Duration::days(3))?;
        assert_eq!(state.runs.len(), count);
        Ok(())
    }

    #[test]
    fn recipe_schedule_edits_and_attention_require_consent() -> Result<()> {
        let now = instant("2026-09-12T12:00:00Z");
        let mut state = RecipeScheduleState::default();
        let schedule = schedule(now);
        let id = schedule.id;
        state.save_schedule(schedule.clone(), now)?;
        assert!(state.admit(id, now).is_err());
        assert!(state.set_enabled(id, true, false, now).is_err());
        state.set_enabled(id, true, true, now)?;
        let run = state.admit(id, now)?;
        state.set_enabled(id, false, false, now)?;
        assert!(state.active_run().is_some());
        state.finish(run, RecipeRunStatus::NeedsAttention, None, now)?;
        assert!(state.admit(id, now).is_err());
        state.set_enabled(id, true, true, now)?;
        state.save_schedule(schedule, now)?;
        assert!(state.admit(id, now).is_err());
        Ok(())
    }

    #[test]
    fn recipe_schedule_skips_spring_gap() -> Result<()> {
        assert_eq!(
            next_run(
                "30 2 * * *",
                "America/New_York",
                instant("2026-03-08T00:00:00Z")
            )?,
            instant("2026-03-09T06:30:00Z")
        );
        Ok(())
    }

    #[test]
    fn recipe_schedule_runs_only_first_fall_occurrence() -> Result<()> {
        assert_eq!(
            next_run(
                "30 1 * * *",
                "America/New_York",
                instant("2026-11-01T04:00:00Z")
            )?,
            instant("2026-11-01T05:30:00Z")
        );
        assert_eq!(
            next_run(
                "30 1 * * *",
                "America/New_York",
                instant("2026-11-01T06:00:00Z")
            )?,
            instant("2026-11-02T06:30:00Z")
        );
        Ok(())
    }

    #[test]
    fn recipe_schedule_validates_cron_and_zone() {
        let now = instant("2026-09-12T12:00:00Z");
        for (cron, zone) in [
            ("* * *", "UTC"),
            ("61 * * * *", "UTC"),
            ("* * * * *", "not-a-zone"),
            ("0 0 31 2 *", "UTC"),
        ] {
            assert!(next_run(cron, zone, now).is_err());
        }
    }
}
