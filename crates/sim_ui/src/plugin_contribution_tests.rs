use crate::{
    GeneratedFrontendExtensionDispositionKind, LegacyExtensionPlaceholder,
    MAX_PLUGIN_CONTRIBUTION_SCHEMA_BYTES, PluginContributionInput, PluginContributionSource,
    PluginContributionSurface, frontend_extension_dispositions, plugin_contribution_snapshot,
    register_plugin_contribution_source,
};
use gpui::{Context, IntoElement, Render, TestAppContext, Window};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    sync::Arc,
};

const SOURCE_CATALOG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.agents/specs/comfy-parity/catalogs/frontend-extensions.csv"
));
const DISPOSITION_CATALOG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.agents/specs/comfy-parity/catalogs/frontend-extension-dispositions.csv"
));
const POLICY: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../.agents/specs/comfy-parity/frontend-extension-policy.json"
));
const FIXTURE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../sim_test_support/fixtures/legacy_frontend_extensions/compatibility_cases.json"
));
const GENERATED_SOURCE: &[u8] = include_bytes!("generated_frontend_extension_catalog.rs");
const PROJECTION_SOURCE: &str = include_str!("plugin_contributions.rs");
const PLACEHOLDER_SOURCE: &[u8] = include_bytes!("legacy_extension_placeholder.rs");
const ZED_SOURCE: &str = include_str!("../../zed/src/zed.rs");

#[derive(Clone)]
enum TestSourceResult {
    Inputs(Vec<PluginContributionInput>),
    Failure,
}

#[derive(Clone)]
struct TestSource(TestSourceResult);

impl PluginContributionSource for TestSource {
    fn verified_contributions(&self) -> anyhow::Result<Vec<PluginContributionInput>> {
        match &self.0 {
            TestSourceResult::Inputs(inputs) => Ok(inputs.clone()),
            TestSourceResult::Failure => anyhow::bail!("sanitized inventory failure"),
        }
    }
}

fn input(id: &str, surface: &str, schema: &str) -> PluginContributionInput {
    PluginContributionInput::from_verified_manifest(
        "test.frontend-plugin",
        "a".repeat(64),
        id,
        surface,
        schema,
    )
    .expect("construct checked verified-manifest projection")
}

fn catalog_case() -> Result<Value, String> {
    let source_ids = SOURCE_CATALOG
        .lines()
        .skip(1)
        .filter_map(|line| line.split_once(',').map(|(feature_id, _)| feature_id))
        .collect::<BTreeSet<_>>();
    let dispositions = frontend_extension_dispositions();
    if source_ids.len() != 59 || dispositions.len() != 59 {
        return Err(format!(
            "catalog closure changed: source={}, generated={}",
            source_ids.len(),
            dispositions.len()
        ));
    }
    let generated_ids = dispositions
        .iter()
        .map(|disposition| disposition.feature_id)
        .collect::<BTreeSet<_>>();
    if generated_ids != source_ids {
        return Err("generated disposition identities differ from the source catalog".to_owned());
    }
    let mut counts = BTreeMap::new();
    for disposition in dispositions {
        *counts.entry(disposition.classification).or_insert(0usize) += 1;
        match disposition.classification {
            GeneratedFrontendExtensionDispositionKind::DeclarativeRustWasm => {
                let surface = disposition
                    .native_surface
                    .and_then(PluginContributionSurface::parse)
                    .ok_or_else(|| {
                        format!(
                            "{} lacks a typed declarative surface",
                            disposition.feature_id
                        )
                    })?;
                if surface.as_str() != disposition.native_surface.unwrap_or_default() {
                    return Err(format!(
                        "{} changed its canonical surface",
                        disposition.feature_id
                    ));
                }
            }
            GeneratedFrontendExtensionDispositionKind::LosslessPlaceholder => {
                if disposition.native_surface.is_some() {
                    return Err(format!(
                        "{} placeholder advertised an executable surface",
                        disposition.feature_id
                    ));
                }
            }
            _ => {}
        }
    }
    let expected = BTreeMap::from([
        (
            GeneratedFrontendExtensionDispositionKind::DeclarativeRustWasm,
            12,
        ),
        (
            GeneratedFrontendExtensionDispositionKind::LegacyIdentifierMapping,
            37,
        ),
        (
            GeneratedFrontendExtensionDispositionKind::LosslessPlaceholder,
            8,
        ),
        (
            GeneratedFrontendExtensionDispositionKind::DeliberateDefer,
            2,
        ),
    ]);
    if counts != expected {
        return Err(format!("classification counts changed: {counts:?}"));
    }
    Ok(json!({
        "name": "59-row-generated-disposition-closure",
        "passed": true,
        "rows": dispositions.len(),
        "classifications": {
            "declarative_rust_wasm": 12,
            "legacy_identifier_mapping": 37,
            "lossless_placeholder": 8,
            "deliberate_defer": 2,
        }
    }))
}

fn projection_case(cx: &mut TestAppContext) -> Result<Value, String> {
    let inputs = vec![
        input(
            "panel.valid",
            "bottom-panel",
            r#"{"type":"object","properties":{"message":{"type":"string"}}}"#,
        ),
        input("node.panel", "node-panel", r#"{"type":"object"}"#),
        input("legacy.unknown", "dom-webview", r#"{"type":"object"}"#),
        input("schema.invalid", "menu", "not-json"),
        input(
            "panel.valid",
            "bottom-panel",
            r#"{"type":"object","title":"conflicting duplicate"}"#,
        ),
    ];
    cx.update(|cx| {
        register_plugin_contribution_source(
            Arc::new(TestSource(TestSourceResult::Inputs(inputs))),
            cx,
        );
    });
    let snapshot = cx.update(|cx| plugin_contribution_snapshot(cx));
    if snapshot.declarative().len() != 1
        || snapshot.placeholders().len() != 4
        || snapshot.diagnostics().len() != 4
        || snapshot.source_error().is_some()
    {
        return Err(format!(
            "projection isolation changed: declarative={}, placeholders={}, diagnostics={}, source_error={:?}",
            snapshot.declarative().len(),
            snapshot.placeholders().len(),
            snapshot.diagnostics().len(),
            snapshot.source_error()
        ));
    }
    let contribution = snapshot
        .declarative()
        .first()
        .ok_or_else(|| "the unique signed node-panel contribution is absent".to_owned())?;
    if contribution.surface() != PluginContributionSurface::NodePanel
        || contribution.input().contribution_id() != "node.panel"
        || contribution.input().manifest_digest_sha256() != "a".repeat(64)
        || !contribution.parsed_state_schema().is_object()
    {
        return Err("supported signed projection changed semantics".to_owned());
    }
    let duplicate_payloads = snapshot
        .placeholders()
        .iter()
        .filter(|placeholder| placeholder.hook_identity() == "panel.valid")
        .map(|placeholder| placeholder.exact_payload())
        .collect::<BTreeSet<_>>();
    if duplicate_payloads
        != BTreeSet::from([
            br#"{"type":"object","properties":{"message":{"type":"string"}}}"#.as_slice(),
            br#"{"type":"object","title":"conflicting duplicate"}"#.as_slice(),
        ])
    {
        return Err("ambiguous contribution payloads were not both retained byte-exact".to_owned());
    }
    let unsupported = snapshot
        .placeholders()
        .iter()
        .find(|placeholder| placeholder.hook_identity() == "legacy.unknown")
        .ok_or_else(|| "unknown surface did not become a placeholder".to_owned())?;
    if unsupported.exact_payload() != br#"{"type":"object"}"# {
        return Err("unknown surface payload was not byte-exact".to_owned());
    }
    Ok(json!({
        "name": "signed-projection-and-placeholder-isolation",
        "passed": true,
        "declarative": 1,
        "placeholders": 4,
    }))
}

fn source_failure_case(cx: &mut TestAppContext) -> Result<Value, String> {
    cx.update(|cx| {
        register_plugin_contribution_source(Arc::new(TestSource(TestSourceResult::Failure)), cx);
    });
    let snapshot = cx.update(|cx| plugin_contribution_snapshot(cx));
    let source_error = snapshot
        .source_error()
        .ok_or_else(|| "source failure was silently discarded".to_owned())?;
    if !snapshot.declarative().is_empty()
        || !snapshot.placeholders().is_empty()
        || !source_error.contains("sanitized inventory failure")
    {
        return Err("source failure leaked stale or executable state".to_owned());
    }
    Ok(json!({
        "name": "component-inventory-failure-isolated",
        "passed": true,
    }))
}

fn bounds_case() -> Result<Value, String> {
    let oversized = "x".repeat(MAX_PLUGIN_CONTRIBUTION_SCHEMA_BYTES + 1);
    if PluginContributionInput::from_verified_manifest(
        "test.frontend-plugin",
        "a".repeat(64),
        "oversized.schema",
        "menu",
        oversized,
    )
    .is_ok()
    {
        return Err("oversized schema crossed the UI projection boundary".to_owned());
    }
    if LegacyExtensionPlaceholder::new(
        "legacy.web-extension",
        "setup",
        [],
        "unsafe\naccessible diagnostic",
        [],
    )
    .is_ok()
    {
        return Err("control characters crossed the accessible placeholder boundary".to_owned());
    }
    Ok(json!({
        "name": "projection-bounds-before-json-parse",
        "passed": true,
    }))
}

struct PlaceholderView(LegacyExtensionPlaceholder);

impl Render for PlaceholderView {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
        self.0.clone()
    }
}

fn placeholder_render_case(cx: &mut TestAppContext) -> Result<Value, String> {
    let payload = br#"{"unknown":{"nested":[1,2,3]},"widget":"DOMWidget"}"#.to_vec();
    let placeholder = LegacyExtensionPlaceholder::new(
        "legacy.web-extension",
        "beforeConfigureGraph",
        payload.clone(),
        "Imperative JavaScript hook is unavailable in native GPUI",
        [],
    )
    .map_err(|error| error.to_string())?;
    if placeholder.exact_payload() != payload {
        return Err("placeholder changed its opaque payload".to_owned());
    }
    cx.update(|cx| {
        let settings_store = settings::SettingsStore::test(cx);
        cx.set_global(settings_store);
        theme_settings::init(theme::LoadThemes::JustBase, cx);
    });
    let (_view, window) = cx.add_window_view(|_, _| PlaceholderView(placeholder));
    window.run_until_parked();
    window.update(|window, cx| {
        let _arena_clear_needed = window.draw(cx);
    });
    Ok(json!({
        "name": "accessible-lossless-placeholder-renders",
        "passed": true,
        "role": "alert",
    }))
}

fn production_boundary_case() -> Result<Value, String> {
    if ZED_SOURCE
        .matches("impl sim_ui::PluginContributionSource")
        .count()
        != 1
        || !ZED_SOURCE.contains("router.current()?.installed_plugins()?")
        || !ZED_SOURCE.contains("PluginContributionInput::from_verified_manifest")
        || PROJECTION_SOURCE.contains("eval(")
        || PROJECTION_SOURCE.contains("Command::new")
        || PROJECTION_SOURCE.contains("open_url")
    {
        return Err("production contribution ownership or native boundary changed".to_owned());
    }
    Ok(json!({
        "name": "single-read-only-production-adapter",
        "passed": true,
        "lifecycle_owner": "sim_plugin_host::ComponentHost",
        "ui_adapter": "sim_ui::PluginContributionSource",
    }))
}

fn write_artifact(cases: Vec<Value>) -> anyhow::Result<()> {
    let passed = cases
        .iter()
        .filter(|case| case.get("passed") == Some(&Value::Bool(true)))
        .count();
    let artifact = json!({
        "validation": "VAL-GPUI-015",
        "environment": {
            "operating_system": std::env::consts::OS,
            "architecture": std::env::consts::ARCH,
            "backend": "gpui-test",
            "javascript_execution": false,
            "browser_execution": false,
            "external_processes": false,
        },
        "fixture_digests": {
            "source_catalog": format!("{:x}", Sha256::digest(SOURCE_CATALOG.as_bytes())),
            "disposition_catalog": format!("{:x}", Sha256::digest(DISPOSITION_CATALOG)),
            "policy": format!("{:x}", Sha256::digest(POLICY)),
            "legacy_fixture": format!("{:x}", Sha256::digest(FIXTURE)),
            "generated_rust": format!("{:x}", Sha256::digest(GENERATED_SOURCE)),
            "projection_source": format!("{:x}", Sha256::digest(PROJECTION_SOURCE.as_bytes())),
            "placeholder_source": format!("{:x}", Sha256::digest(PLACEHOLDER_SOURCE)),
            "zed_adapter_source": format!("{:x}", Sha256::digest(ZED_SOURCE.as_bytes())),
        },
        "summary": {
            "passed": passed,
            "failed": cases.len().saturating_sub(passed),
            "skipped": 0,
        },
        "cases": cases,
        "skipped": [],
    });
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target"))
        .join("comfy-parity");
    fs::create_dir_all(&target)?;
    let mut bytes = serde_json::to_vec_pretty(&artifact)?;
    bytes.push(b'\n');
    fs::write(target.join("val-gpui-015.json"), bytes)?;
    Ok(())
}

#[gpui::test(seed = 16029)]
fn val_gpui_015(cx: &mut TestAppContext) {
    let cases = vec![
        catalog_case().expect("reconcile all frontend extension rows"),
        projection_case(cx).expect("project signed declarative UI and isolate unsupported data"),
        source_failure_case(cx).expect("isolate component inventory failure"),
        bounds_case().expect("enforce projection bounds"),
        placeholder_render_case(cx).expect("render accessible lossless placeholder"),
        production_boundary_case().expect("verify the single production adapter"),
    ];
    write_artifact(cases).expect("write VAL-GPUI-015 artifact");
}
