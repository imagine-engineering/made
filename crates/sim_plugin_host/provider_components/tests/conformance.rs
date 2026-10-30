use std::{error::Error, fs, path::PathBuf};

use sim_provider_components::{
    DISCOVERED_PROVIDER_COMPONENTS,
    common::{
        CanonicalTypeId, ConformanceError, CostRequest, CostResponse, EncodedOutput,
        ExpectedOutput, Header, HttpMethod, InvocationPlan, MaterializationError,
        MaterializationSink, MaterializedBatch, MultipartField, PluginValue, Progress, RequestHead,
        ResponseEvent, ScalarValue, ScriptStep, ScriptedActuator, StreamHandle, TransportLimits,
        TypeRegistry, ValueFamily, execute_plan,
    },
};

#[derive(Default)]
struct RecordingSink {
    batches: Vec<MaterializedBatch>,
}

impl MaterializationSink for RecordingSink {
    fn publish(&mut self, batch: MaterializedBatch) -> Result<(), MaterializationError> {
        self.batches.push(batch);
        Ok(())
    }
}

#[test]
fn common_harness_covers_exact_transport_retry_poll_and_materialization()
-> Result<(), Box<dyn Error>> {
    let fixture = common_fixture()?;
    assert!(fixture.contains("method=POST\n"));
    assert!(fixture.contains("path=/v2/jobs\n"));
    let limits = TransportLimits {
        maximum_chunk_bytes: 1024,
        ..TransportLimits::conformance()
    };
    let plan = plan()?;
    let mut actuator = ScriptedActuator::new(successful_steps(&plan, expected_body()));
    let mut sink = RecordingSink::default();
    execute_plan(&plan, limits, &mut actuator, &mut sink)?;
    let observed = actuator.finish()?;
    assert_eq!(
        observed
            .iter()
            .filter(|step| matches!(step, ScriptStep::Start(..)))
            .count(),
        2
    );
    assert!(observed.iter().all(step_uses_fresh_handle));
    assert_eq!(sink.batches.len(), 1);
    assert_eq!(
        sink.batches[0].resident_bytes(),
        plan.outputs[0].bytes.len()
    );
    assert_eq!(sink.batches[0].outputs()[0], plan.outputs[0]);
    Ok(())
}

#[test]
fn malformed_and_oversized_plans_fail_before_transport_or_publication() -> Result<(), Box<dyn Error>>
{
    let mut invalid_path = plan()?;
    invalid_path.request.path = "https://network.invalid/v2".into();
    assert_preflight_failure(invalid_path, ConformanceError::InvalidRequest);

    let mut oversized_body = plan()?;
    oversized_body.multipart_fields[0].bytes = vec![0; 2 * 1024 * 1024];
    assert_preflight_failure(oversized_body, ConformanceError::BodyLimit);

    let mut malformed_value = plan()?;
    malformed_value.outputs[0].bytes = b"not-plugin-value-abi".to_vec();
    assert_preflight_failure(
        malformed_value,
        ConformanceError::Materialization(MaterializationError::MalformedValue),
    );

    let mut wrong_family = plan()?;
    wrong_family.outputs[0].family = ValueFamily::Artifact;
    assert_preflight_failure(
        wrong_family,
        ConformanceError::Materialization(MaterializationError::TypeMismatch),
    );

    let mut actuator = ScriptedActuator::new(Vec::new());
    let mut sink = RecordingSink::default();
    let error = execute_plan(
        &plan()?,
        TransportLimits {
            maximum_chunk_bytes: 0,
            ..TransportLimits::conformance()
        },
        &mut actuator,
        &mut sink,
    )
    .expect_err("zero chunk bound must fail before chunking");
    assert_eq!(error, ConformanceError::InvalidRequest);
    assert!(actuator.observed().is_empty());
    assert!(sink.batches.is_empty());
    Ok(())
}

#[test]
fn response_headers_and_receipts_are_bounded_before_publication() -> Result<(), Box<dyn Error>> {
    let plan = plan()?;
    let mut bad_cost = successful_steps(&plan, expected_body());
    for step in &mut bad_cost {
        if let ScriptStep::Cost(_, _, response) = step {
            response.receipt = vec![1; 32 * 1024 + 1];
        }
    }
    assert_response_failure(&plan, bad_cost, ConformanceError::InvalidCost);

    let mut bad_headers = successful_steps(&plan, expected_body());
    for step in &mut bad_headers {
        if let ScriptStep::Wait {
            outcome: ResponseEvent::Head { headers, .. },
            ..
        } = step
        {
            *headers = vec![Header {
                name: "bad\r\nheader".into(),
                value: "x".into(),
            }];
        }
    }
    assert_response_failure(&plan, bad_headers, ConformanceError::InvalidResponse);

    let mut bad_terminal = successful_steps(&plan, expected_body());
    for step in &mut bad_terminal {
        if let ScriptStep::Wait {
            outcome: ResponseEvent::Completed(receipt),
            ..
        } = step
        {
            *receipt = vec![1; 32 * 1024 + 1];
        }
    }
    assert_response_failure(&plan, bad_terminal, ConformanceError::InvalidResponse);
    Ok(())
}

#[test]
fn cancellation_and_stale_handles_are_typed_and_atomic() -> Result<(), Box<dyn Error>> {
    let plan = plan()?;
    let mut cancelled = ScriptedActuator::new(vec![ScriptStep::Cancel(None)]);
    let mut sink = RecordingSink::default();
    let error = execute_plan(
        &plan,
        TransportLimits::conformance(),
        &mut cancelled,
        &mut sink,
    )
    .expect_err("cancelled invocation must fail");
    assert_eq!(error, ConformanceError::Cancelled);
    assert!(cancelled.observed().is_empty());
    assert!(sink.batches.is_empty());

    let stale = StreamHandle {
        generation: handle().generation - 1,
        ..handle()
    };
    let steps = successful_steps(&plan, expected_body());
    let post_start_indices = steps
        .iter()
        .enumerate()
        .filter_map(|(index, step)| step_handle(step).map(|_| index))
        .collect::<Vec<_>>();
    assert!(!post_start_indices.is_empty());
    for index in post_start_indices {
        let mut stale_steps = steps.clone();
        set_step_handle(&mut stale_steps[index], stale);
        let mut stale_actuator = ScriptedActuator::new(stale_steps);
        let error = execute_plan(
            &plan,
            TransportLimits {
                maximum_chunk_bytes: 1024,
                ..TransportLimits::conformance()
            },
            &mut stale_actuator,
            &mut sink,
        )
        .expect_err("stale generation must fail");
        assert_eq!(error, ConformanceError::InvalidHandle, "step {index}");
        assert!(sink.batches.is_empty());
    }
    Ok(())
}

#[test]
fn registry_and_fixture_discovery_are_symmetric_without_shared_edits() -> Result<(), Box<dyn Error>>
{
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let fixture_root = root.join("../../sim_test_support/fixtures/providers");
    let mut fixtures = fs::read_dir(fixture_root)?
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.file_type().is_ok_and(|kind| kind.is_dir()) && entry.file_name() != "common"
        })
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    fixtures.sort();
    let names = DISCOVERED_PROVIDER_COMPONENTS
        .iter()
        .map(|provider| provider.name)
        .collect::<Vec<_>>();
    assert_eq!(names, fixtures);
    for provider in DISCOVERED_PROVIDER_COMPONENTS {
        provider.run_conformance()?;
    }
    Ok(())
}

fn assert_preflight_failure(plan: InvocationPlan, expected: ConformanceError) {
    let mut actuator = ScriptedActuator::new(Vec::new());
    let mut sink = RecordingSink::default();
    let error = execute_plan(
        &plan,
        TransportLimits::conformance(),
        &mut actuator,
        &mut sink,
    )
    .expect_err("invalid plan must fail");
    assert_eq!(error, expected);
    assert!(actuator.observed().is_empty());
    assert!(sink.batches.is_empty());
}

fn assert_response_failure(
    plan: &InvocationPlan,
    steps: Vec<ScriptStep>,
    expected: ConformanceError,
) {
    let mut actuator = ScriptedActuator::new(steps);
    let mut sink = RecordingSink::default();
    let error = execute_plan(
        plan,
        TransportLimits {
            maximum_chunk_bytes: 1024,
            ..TransportLimits::conformance()
        },
        &mut actuator,
        &mut sink,
    )
    .expect_err("malformed response must fail");
    assert_eq!(error, expected);
    assert!(sink.batches.is_empty());
}

fn plan() -> Result<InvocationPlan, Box<dyn Error>> {
    let registry = TypeRegistry::built_in()?;
    let type_id = CanonicalTypeId::new("sim", "string", 1)?;
    let value = PluginValue::scalar(
        type_id.clone(),
        ScalarValue::String("hello".into()),
        &registry,
    )?;
    Ok(InvocationPlan {
        request: RequestHead {
            method: HttpMethod::Post,
            path: "/v2/jobs".into(),
            secret_id: Some("fixture-secret".into()),
            headers: vec![
                Header {
                    name: "authorization".into(),
                    value: "Bearer fixture".into(),
                },
                Header {
                    name: "x-request-id".into(),
                    value: "request-1".into(),
                },
            ],
        },
        multipart_boundary: "sim-boundary".into(),
        multipart_fields: vec![
            MultipartField {
                name: "prompt".into(),
                file_name: None,
                media_type: None,
                bytes: b"hello".to_vec(),
            },
            MultipartField {
                name: "reference".into(),
                file_name: Some("image.bin".into()),
                media_type: Some("application/octet-stream".into()),
                bytes: vec![1, 2, 3],
            },
        ],
        upload: vec![9, 8, 7],
        cost: CostRequest {
            operation: "generate".into(),
            currency: "USD".into(),
            maximum_microunits: 10,
        },
        progress: Progress {
            sequence: 0,
            completed: 1,
            total: 1,
            message: Some("complete".into()),
        },
        wait_milliseconds: 25,
        expected_outputs: vec![ExpectedOutput {
            port_id: "output_0".into(),
            type_id: type_id.clone(),
            family: ValueFamily::Scalar,
        }],
        outputs: vec![EncodedOutput {
            port_id: "output_0".into(),
            type_id,
            family: ValueFamily::Scalar,
            bytes: value.abi_bytes()?,
        }],
    })
}

fn handle() -> StreamHandle {
    StreamHandle {
        invocation: 1,
        slot: 1,
        generation: 2,
    }
}

fn expected_body() -> Vec<u8> {
    b"--sim-boundary\r\nContent-Disposition: form-data; name=\"prompt\"\r\n\r\nhello\r\n--sim-boundary\r\nContent-Type: application/octet-stream\r\nContent-Disposition: form-data; name=\"reference\"; filename=\"image.bin\"\r\n\r\n\x01\x02\x03\r\n--sim-boundary--\r\n".to_vec()
}

fn successful_steps(plan: &InvocationPlan, body: Vec<u8>) -> Vec<ScriptStep> {
    let handle = handle();
    vec![
        ScriptStep::CheckCancelled(None),
        ScriptStep::Start(plan.request.clone(), Err(ConformanceError::WorkerLost)),
        ScriptStep::CheckCancelled(None),
        ScriptStep::Start(plan.request.clone(), Ok(handle)),
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::RequestChunk {
            handle,
            sequence: 0,
            bytes: body,
            end: true,
        },
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Upload(handle, plan.upload.clone()),
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Cost(
            handle,
            plan.cost.clone(),
            CostResponse {
                accepted: true,
                approved_microunits: 7,
                receipt: b"cost-receipt".to_vec(),
            },
        ),
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Progress(handle, plan.progress.clone()),
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Wait {
            handle,
            after_sequence: None,
            timeout_milliseconds: 25,
            outcome: ResponseEvent::TimedOut,
        },
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Wait {
            handle,
            after_sequence: None,
            timeout_milliseconds: 25,
            outcome: ResponseEvent::Head {
                status: 200,
                headers: vec![Header {
                    name: "content-type".into(),
                    value: "application/json".into(),
                }],
            },
        },
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Wait {
            handle,
            after_sequence: Some(0),
            timeout_milliseconds: 25,
            outcome: ResponseEvent::Chunk(b"{\"ok\":".to_vec()),
        },
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Wait {
            handle,
            after_sequence: Some(1),
            timeout_milliseconds: 25,
            outcome: ResponseEvent::Chunk(b"true}".to_vec()),
        },
        ScriptStep::CheckCancelled(Some(handle)),
        ScriptStep::Wait {
            handle,
            after_sequence: Some(2),
            timeout_milliseconds: 25,
            outcome: ResponseEvent::Completed(b"terminal-receipt".to_vec()),
        },
    ]
}

fn step_uses_fresh_handle(step: &ScriptStep) -> bool {
    step_handle(step).is_none_or(|actual| actual == handle())
}

fn step_handle(step: &ScriptStep) -> Option<StreamHandle> {
    match step {
        ScriptStep::CheckCancelled(Some(actual))
        | ScriptStep::Cancel(Some(actual))
        | ScriptStep::RequestChunk { handle: actual, .. }
        | ScriptStep::Upload(actual, _)
        | ScriptStep::CreateStorage(actual, _, _)
        | ScriptStep::PutSigned(actual, _)
        | ScriptStep::Cost(actual, _, _)
        | ScriptStep::Progress(actual, _)
        | ScriptStep::Wait { handle: actual, .. } => Some(*actual),
        _ => None,
    }
}

fn set_step_handle(step: &mut ScriptStep, replacement: StreamHandle) {
    match step {
        ScriptStep::CheckCancelled(Some(actual))
        | ScriptStep::Cancel(Some(actual))
        | ScriptStep::RequestChunk { handle: actual, .. }
        | ScriptStep::Upload(actual, _)
        | ScriptStep::CreateStorage(actual, _, _)
        | ScriptStep::PutSigned(actual, _)
        | ScriptStep::Cost(actual, _, _)
        | ScriptStep::Progress(actual, _)
        | ScriptStep::Wait { handle: actual, .. } => *actual = replacement,
        _ => {}
    }
}

fn common_fixture() -> Result<String, Box<dyn Error>> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../sim_test_support/fixtures/providers/common/conformance.fixture");
    Ok(fs::read_to_string(path)?)
}
