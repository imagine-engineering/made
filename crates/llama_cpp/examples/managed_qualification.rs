use anyhow::{Context as _, Result, ensure};
use futures::{AsyncReadExt as _, FutureExt as _};
use http_client::{AsyncBody, HttpClient as _, Method, Request};
use llama_cpp::managed::{
    DecodingMode, Hardware, LoadOptions, ManagedRuntime, ManagedStore, catalog,
};
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64},
    },
};

fn main() -> Result<()> {
    let mut arguments = std::env::args().skip(1);
    let root = PathBuf::from(
        arguments
            .next()
            .context("Supply a disposable managed storage directory")?,
    );
    let action = arguments
        .next()
        .context("Choose runtime, download, infer, or infer-cpu explicitly")?;
    let model_id = arguments.next();
    smol::block_on(async move {
        let http = reqwest_client::ReqwestClient::user_agent("Copper-managed-local-qualification")?;
        let store = ManagedStore::open(root)?;
        let catalog = catalog()?;
        let hardware = Hardware::detect(store.root())?;
        println!("Hardware: {hardware:?}");
        if action == "runtime" {
            let executable = store.install_runtime(&catalog, &http).await?;
            for argument in ["--version", "--list-devices"] {
                let output = smol::process::Command::new(&executable)
                    .arg(argument)
                    .output()
                    .await?;
                ensure!(output.status.success(), "Runtime {argument} failed");
                println!(
                    "{}{}",
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            return Ok(());
        }
        let model = catalog
            .models
            .iter()
            .find(|model| Some(&model.id) == model_id.as_ref())
            .context("Supply a catalog model identifier")?;
        if action == "download" {
            println!(
                "Explicit test download: {} bytes, {}",
                model.bytes, model.filename
            );
            return store
                .download(
                    model,
                    &http,
                    Arc::new(AtomicU64::new(0)),
                    Arc::new(AtomicBool::new(false)),
                )
                .await;
        }
        ensure!(
            action == "infer" || action == "infer-cpu",
            "Unknown qualification action"
        );
        store.verify(model)?;
        let executable = store.install_runtime(&catalog, &http).await?;
        let mut runtime = ManagedRuntime::launch(
            &executable,
            model,
            &store.model_path(model)?,
            store.root(),
            LoadOptions {
                cpu_only: action == "infer-cpu",
                decoding: DecodingMode::Standard,
            },
        )?;
        let local_http = reqwest_client::ReqwestClient::loopback()?;
        runtime.wait_ready(&local_http, qualification_timer).await?;
        let unauthenticated = local_http
            .get(
                &format!("{}/v1/models", runtime.endpoint),
                AsyncBody::empty(),
                false,
            )
            .await?;
        ensure!(
            unauthenticated.status() == 401,
            "Managed inference must reject requests without its temporary credential"
        );
        println!("Unauthenticated model API correctly rejected with HTTP 401");
        let request = Request::builder()
            .method(Method::POST)
            .uri(format!("{}/v1/chat/completions", runtime.endpoint))
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", runtime.api_key))
            .body(AsyncBody::from(serde_json::to_string(&serde_json::json!({"model":model.id,"messages":[{"role":"user","content":"Reply with the word ready."}],"max_tokens":32,"stream":false}))?))?;
        let mut response = local_http.send(request).await?;
        ensure!(
            response.status().is_success(),
            "Runtime inference returned {}",
            response.status()
        );
        let mut body = String::new();
        response.body_mut().read_to_string(&mut body).await?;
        println!("{body}");
        Ok(())
    })
}

#[allow(
    clippy::disallowed_methods,
    reason = "Standalone qualification runs without a GPUI executor"
)]
fn qualification_timer(duration: std::time::Duration) -> futures::future::BoxFuture<'static, ()> {
    async move {
        smol::Timer::after(duration).await;
    }
    .boxed()
}
