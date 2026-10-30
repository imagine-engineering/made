use sim_model::{
    conditioning::{
        ConditioningEntry, ConditioningEntryOptions, ConditioningIdentity, ConditioningMetadataKey,
        ConditioningMetadataMap, ConditioningMetadataValue, ConditioningReferences,
        ConditioningSet, ConditioningValue,
    },
    generated_native_diffusion::{sd15_latent_format_identity, sd15_model_family_identity},
};
use sim_nodes::{
    NativeHandleStore, NativeHandleStoreError, NativeHandleStoreIdentity, NativeHandleType,
    NativeNodeComputeSession, NativeNodeContext, NativeNodeFailureKind, NativeNodeOutcome,
    NativeNodeServiceIdentity, NativeNodeServices, NativeOpaqueHandle, NativeResolvedPayload,
    NativeStoredPayload, NativeValue,
};
use sim_runtime::{NativeHandleStoreGeneration, generated_native_node_registry_projection};
use sim_tensor::{
    CancellationToken, CpuBackend, CpuWorkspaceAuthority, NativeLatentBundle, NativeLatentMetadata,
    StreamId, Tensor, generated_native_diffusion::tensor_from_f32,
};
use sim_types::{AttemptId, NodeId, PromptId};
use std::{
    collections::BTreeMap,
    error::Error,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};
use uuid::Uuid;
type TestResult = Result<(), Box<dyn Error>>;
const LIMIT: u64 = 8 * 1024 * 1024;
const CLASS: &str = "ReferenceLatent";

const SOURCE27_CAPTURE_SHA256: &str =
    "d1a3958f93701bf9afc60258e13359751d176625d540d83f530a73064fc78735";
const SOURCE27_REVIEW_SHA256: &str =
    "666d2e3943f02796fbe0513aff3bc58ea14bea4e521eb7ca67d77f9f10e0eb26";

fn source27_capture() -> Result<serde_json::Value, Box<dyn Error>> {
    use sha2::{Digest, Sha256};
    let path = std::env::var_os("SIM_REFERENCE_LATENT_SOURCE27_CAPTURE")
        .ok_or("Reviewed genuine Source27 capture path required")?;
    let review_path = std::env::var_os("SIM_REFERENCE_LATENT_SOURCE27_REVIEW")
        .ok_or("Independent actual Source27 review required")?;
    let expected = std::env::var("SIM_REFERENCE_LATENT_SOURCE27_CAPTURE_SHA256")?;
    assert_eq!(expected, SOURCE27_CAPTURE_SHA256);
    if std::fs::metadata(&path)?.len() > 8 * 1024 * 1024
        || std::fs::metadata(&review_path)?.len() > 8 * 1024 * 1024
    {
        return Err("Bounded Source27 evidence exceeds eight MiB".into());
    }
    let bytes = std::fs::read(path)?;
    assert_eq!(format!("{:x}", Sha256::digest(&bytes)), expected);
    let review_bytes = std::fs::read(review_path)?;
    assert_eq!(
        format!("{:x}", Sha256::digest(&review_bytes)),
        SOURCE27_REVIEW_SHA256
    );
    let review: serde_json::Value = serde_json::from_slice(&review_bytes)?;
    assert_eq!(review["accepted_source_capture"], true);
    assert_eq!(review["all_original_groups_closed"], true);
    assert_eq!(review["scope_closed"], true);
    assert_eq!(review["original_exec"]["exit_code"], 0);
    let capture: serde_json::Value = serde_json::from_slice(&bytes)?;
    assert_eq!(
        capture["format"],
        "comfy.source.reference-latent.full-module-cpu-f32.v1"
    );
    assert_eq!(
        capture["producer_sha256"],
        "ef2069429a13344b9be19872bf3b772470f36e9082eb97efe0b2b6c2a57a3317"
    );
    assert_eq!(
        capture["source_sha256"]["comfy_extras/nodes_edit_model.py"],
        "8fe4edaa13707df052ddf883fade1190f7b68416d54c7026cefefd489a13ee7e"
    );
    assert_eq!(capture["case_count"], 27);
    assert_eq!(capture["success_count"], 22);
    assert_eq!(capture["error_count"], 5);
    Ok(capture)
}

struct Source27Graph<'a> {
    roots: &'a serde_json::Value,
    objects: BTreeMap<&'a str, &'a serde_json::Value>,
}

impl<'a> Source27Graph<'a> {
    fn checked(value: &'a serde_json::Value) -> Result<Self, Box<dyn Error>> {
        let roots = value.get("roots").ok_or("Source27 roots required")?;
        let mut objects = BTreeMap::new();
        for object in value
            .get("objects")
            .and_then(serde_json::Value::as_array)
            .ok_or("Source27 objects required")?
        {
            let identifier = object
                .get("id")
                .and_then(serde_json::Value::as_str)
                .ok_or("Source27 object identity required")?;
            if objects.insert(identifier, object).is_some() {
                return Err("Duplicate Source27 object identity".into());
            }
        }
        Ok(Self { roots, objects })
    }

    fn root(&self, name: &str) -> Result<&'a serde_json::Value, Box<dyn Error>> {
        self.roots
            .get(name)
            .ok_or_else(|| "Source27 root required".into())
    }

    fn object(
        &self,
        reference: &serde_json::Value,
    ) -> Result<&'a serde_json::Value, Box<dyn Error>> {
        let identifier = reference
            .get("ref")
            .and_then(serde_json::Value::as_str)
            .ok_or("Source27 object reference required")?;
        self.objects
            .get(identifier)
            .copied()
            .ok_or_else(|| "Unknown Source27 object reference".into())
    }

    fn list(
        &self,
        reference: &serde_json::Value,
    ) -> Result<&'a [serde_json::Value], Box<dyn Error>> {
        let object = self.object(reference)?;
        if object.get("kind").and_then(serde_json::Value::as_str) != Some("list") {
            return Err("Source27 list required".into());
        }
        object
            .get("items")
            .and_then(serde_json::Value::as_array)
            .map(Vec::as_slice)
            .ok_or_else(|| "Source27 list items required".into())
    }

    fn dictionary(
        &self,
        reference: &serde_json::Value,
    ) -> Result<BTreeMap<&'a str, &'a serde_json::Value>, Box<dyn Error>> {
        let object = self.object(reference)?;
        if object.get("kind").and_then(serde_json::Value::as_str) != Some("dict") {
            return Err("Source27 dictionary required".into());
        }
        let mut values = BTreeMap::new();
        for item in object
            .get("items")
            .and_then(serde_json::Value::as_array)
            .ok_or("Source27 dictionary items required")?
        {
            let key = item.get("key").ok_or("Source27 key required")?;
            if key.get("kind").and_then(serde_json::Value::as_str) != Some("str") {
                return Err("Source27 string dictionary key required".into());
            }
            let name = key
                .get("value")
                .and_then(serde_json::Value::as_str)
                .ok_or("Source27 dictionary key value required")?;
            if values
                .insert(name, item.get("value").ok_or("Source27 value required")?)
                .is_some()
            {
                return Err("Duplicate Source27 dictionary key".into());
            }
        }
        Ok(values)
    }
}

fn source27_float(value: &serde_json::Value) -> Result<f64, Box<dyn Error>> {
    if value.get("kind").and_then(serde_json::Value::as_str) != Some("float") {
        return Err("Exact Source27 binary64 required".into());
    }
    let hexadecimal = value
        .get("f64_word_hex")
        .and_then(serde_json::Value::as_str)
        .ok_or("Source27 binary64 word required")?;
    if hexadecimal.len() != 16 || !hexadecimal.is_ascii() {
        return Err("Source27 binary64 word has invalid width".into());
    }
    let bytes = (0..8)
        .map(|index| u8::from_str_radix(&hexadecimal[index * 2..index * 2 + 2], 16))
        .collect::<Result<Vec<_>, _>>()?;
    let bytes: [u8; 8] = bytes.try_into().map_err(|_| "Source27 binary64 width")?;
    let value = f64::from_le_bytes(bytes);
    if !value.is_finite() {
        return Err("Source27 finite binary64 projection required".into());
    }
    Ok(value)
}

fn source27_tensor_spec(
    object: &serde_json::Value,
) -> Result<(sim_tensor::TensorDescriptor, Vec<u8>), Box<dyn Error>> {
    use sha2::{Digest, Sha256};
    assert_eq!(
        object.get("kind").and_then(serde_json::Value::as_str),
        Some("Tensor")
    );
    assert_eq!(
        object.get("dtype").and_then(serde_json::Value::as_str),
        Some("torch.float32")
    );
    assert_eq!(
        object.get("device").and_then(serde_json::Value::as_str),
        Some("cpu")
    );
    assert_eq!(
        object
            .get("storage_offset")
            .and_then(serde_json::Value::as_u64),
        Some(0)
    );
    let shape = object
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .ok_or("Source27 shape required")?
        .iter()
        .map(|dimension| {
            dimension
                .as_u64()
                .ok_or("Source27 unsigned dimension required")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let descriptor = sim_tensor::TensorDescriptor::contiguous(
        shape,
        sim_tensor::DType::F32,
        sim_tensor::DeviceId::CPU,
        StreamId::DEFAULT,
    )?;
    let strides = object
        .get("stride")
        .and_then(serde_json::Value::as_array)
        .ok_or("Source27 strides required")?
        .iter()
        .map(|stride| stride.as_i64().ok_or("Source27 signed stride required"))
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(descriptor.strides(), strides);
    let mut bytes = Vec::new();
    for word in object
        .get("f32_words_hex")
        .and_then(serde_json::Value::as_array)
        .ok_or("Source27 exact F32 words required")?
    {
        let word = word
            .as_str()
            .ok_or("Source27 hexadecimal F32 word required")?;
        if word.len() != 8 || !word.is_ascii() {
            return Err("Source27 F32 word has invalid width".into());
        }
        bytes.extend_from_slice(&u32::from_str_radix(word, 16)?.to_le_bytes());
    }
    assert_eq!(
        object
            .get("storage_bytes")
            .and_then(serde_json::Value::as_u64),
        Some(u64::try_from(bytes.len())?)
    );
    assert_eq!(
        object
            .get("contiguous_value_sha256")
            .and_then(serde_json::Value::as_str),
        Some(format!("{:x}", Sha256::digest(&bytes)).as_str())
    );
    Ok((descriptor, bytes))
}

fn source27_assert_tensor(actual: &Tensor, object: &serde_json::Value) -> TestResult {
    let (descriptor, bytes) = source27_tensor_spec(object)?;
    assert_eq!(actual.descriptor().shape(), descriptor.shape());
    assert_eq!(actual.descriptor().strides(), descriptor.strides());
    assert_eq!(actual.descriptor().dtype(), descriptor.dtype());
    assert_eq!(actual.descriptor().device(), descriptor.device());
    assert_eq!(actual.contiguous_bytes()?, bytes);
    Ok(())
}

fn source27_tensor(
    context: &NativeNodeContext,
    graph: &Source27Graph<'_>,
    reference: &serde_json::Value,
    storages: &mut BTreeMap<String, Tensor>,
) -> Result<Tensor, Box<dyn Error>> {
    let object = graph.object(reference)?;
    let storage = object
        .get("storage")
        .and_then(serde_json::Value::as_str)
        .ok_or("Source27 storage identity required")?;
    if let Some(tensor) = storages.get(storage) {
        source27_assert_tensor(tensor, object)?;
        return Ok(tensor.clone());
    }
    let (descriptor, bytes) = source27_tensor_spec(object)?;
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let tensor = compute
        .backend()
        .upload_bytes(descriptor, &bytes, &execution)?
        .0;
    storages.insert(storage.to_owned(), tensor.clone());
    Ok(tensor)
}

fn source27_conditioning(
    context: &NativeNodeContext,
    graph: &Source27Graph<'_>,
    reference: &serde_json::Value,
    storages: &mut BTreeMap<String, Tensor>,
) -> Result<Arc<ConditioningSet>, Box<dyn Error>> {
    let mut entries = Vec::new();
    for (index, reference) in graph.list(reference)?.iter().enumerate() {
        let [primary, metadata] = graph.list(reference)? else {
            return Err("Source27 two-column conditioning entry required".into());
        };
        let primary = source27_tensor(context, graph, primary, storages)?;
        let values = graph.dictionary(metadata)?;
        if values.keys().any(|key| {
            !matches!(
                *key,
                "pooled_output"
                    | "strength"
                    | "start_percent"
                    | "end_percent"
                    | "reference_latents"
            )
        }) {
            return Err("Unsupported Source27 conditioning metadata projection".into());
        }
        let mut metadata = ConditioningMetadataMap::default();
        if let Some(pool) = values.get("pooled_output") {
            let pool = if pool.get("kind").and_then(serde_json::Value::as_str) == Some("NoneType") {
                None
            } else {
                Some(source27_tensor(context, graph, pool, storages)?)
            };
            metadata.insert(
                ConditioningMetadataKey::PooledOutput,
                ConditioningMetadataValue::PooledOutput(pool),
            )?;
        }
        if let Some(references) = values.get("reference_latents") {
            let references = graph
                .list(references)?
                .iter()
                .map(|reference| source27_tensor(context, graph, reference, storages))
                .collect::<Result<Vec<_>, _>>()?;
            metadata.insert(
                ConditioningMetadataKey::ReferenceLatents,
                ConditioningMetadataValue::TensorList(references),
            )?;
        }
        entries.push(ConditioningEntry::checked_with_metadata(
            format!("source27.{index}"),
            ConditioningValue::cross_attention(primary)?,
            ConditioningEntryOptions {
                strength: values
                    .get("strength")
                    .map(|value| source27_float(value))
                    .transpose()?
                    .unwrap_or(1.0),
                window: sim_model::conditioning::ConditioningWindow::new(
                    values
                        .get("start_percent")
                        .map(|value| source27_float(value))
                        .transpose()?
                        .unwrap_or(0.0),
                    values
                        .get("end_percent")
                        .map(|value| source27_float(value))
                        .transpose()?
                        .unwrap_or(1.0),
                )?,
                ..Default::default()
            },
            ConditioningReferences::default(),
            metadata,
        )?);
    }
    Ok(Arc::new(ConditioningSet::checked(
        ConditioningIdentity::new(
            "source27",
            sd15_model_family_identity()?,
            sd15_latent_format_identity()?,
        )?,
        entries,
        &context.cancellation,
    )?))
}

fn source27_assert_conditioning(
    actual: &ConditioningSet,
    graph: &Source27Graph<'_>,
    reference: &serde_json::Value,
    storages: &BTreeMap<String, Tensor>,
) -> TestResult {
    let entries = graph.list(reference)?;
    assert_eq!(actual.entries().len(), entries.len());
    let assert_tensor = |tensor: &Tensor, reference: &serde_json::Value| -> TestResult {
        let object = graph.object(reference)?;
        source27_assert_tensor(tensor, object)?;
        let source_storage = object
            .get("storage")
            .and_then(serde_json::Value::as_str)
            .ok_or("Source27 storage")?;
        let expected = storages
            .get(source_storage)
            .ok_or("Source27 retained native storage")?;
        assert_eq!(tensor.storage_id(), expected.storage_id());
        Ok(())
    };
    for (entry, reference) in actual.entries().iter().zip(entries) {
        let [primary, metadata] = graph.list(reference)? else {
            return Err("Source27 two-column output entry required".into());
        };
        let ConditioningValue::CrossAttention(tensor) = entry.value() else {
            return Err("Canonical cross-attention required".into());
        };
        assert_tensor(tensor, primary)?;
        let values = graph.dictionary(metadata)?;
        assert_eq!(
            entry.options().strength.to_bits(),
            values
                .get("strength")
                .map(|value| source27_float(value))
                .transpose()?
                .unwrap_or(1.0)
                .to_bits()
        );
        assert_eq!(
            entry.options().window.start_percent().to_bits(),
            values
                .get("start_percent")
                .map(|value| source27_float(value))
                .transpose()?
                .unwrap_or(0.0)
                .to_bits()
        );
        assert_eq!(
            entry.options().window.end_percent().to_bits(),
            values
                .get("end_percent")
                .map(|value| source27_float(value))
                .transpose()?
                .unwrap_or(1.0)
                .to_bits()
        );
        match (
            entry.metadata().get(ConditioningMetadataKey::PooledOutput),
            values.get("pooled_output"),
        ) {
            (None, None) => {}
            (Some(ConditioningMetadataValue::PooledOutput(None)), Some(value)) => {
                assert_eq!(
                    value.get("kind").and_then(serde_json::Value::as_str),
                    Some("NoneType")
                );
            }
            (Some(ConditioningMetadataValue::PooledOutput(Some(tensor))), Some(reference)) => {
                assert_tensor(tensor, reference)?
            }
            _ => return Err("Source27 pooled presence mismatch".into()),
        }
        match (
            entry
                .metadata()
                .get(ConditioningMetadataKey::ReferenceLatents),
            values.get("reference_latents"),
        ) {
            (None, None) => {}
            (Some(ConditioningMetadataValue::TensorList(tensors)), Some(reference)) => {
                let references = graph.list(reference)?;
                assert_eq!(tensors.len(), references.len());
                for (tensor, reference) in tensors.iter().zip(references) {
                    assert_tensor(tensor, reference)?;
                }
            }
            _ => return Err("Source27 reference presence mismatch".into()),
        }
    }
    Ok(())
}

#[test]
#[ignore = "Requires the reviewed genuine Source27 capture and independent closure review; compares only the declared canonical typed projection"]
fn task622_original_source27_exact_typed_reference_append_and_storage() -> TestResult {
    let capture = source27_capture()?;
    let cases = capture
        .get("cases")
        .and_then(serde_json::Value::as_array)
        .ok_or("Source27 cases required")?;
    let names = cases
        .iter()
        .map(|case| {
            case.get("name")
                .and_then(serde_json::Value::as_str)
                .ok_or("Source27 case name required")
        })
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(
        names,
        [
            "absent-latent",
            "explicit-none-latent",
            "rank4",
            "rank5",
            "batch2",
            "existing-reference-append",
            "ordered-multiple-entries",
            "repeated-reference-alias",
            "two-executions-append",
            "pooled-none-and-absence",
            "ignored-latent-fields",
            "shared-entry-alias",
            "old-reference-empty",
            "old-reference-none",
            "reference65",
            "rank3",
            "rank2",
            "zero-dimension",
            "missing-samples",
            "samples-none",
            "nested-samples",
            "latent-integer",
            "old-reference-tuple",
            "null-entry-metadata",
            "short-entry",
            "empty-condition-with-latent",
            "empty-condition-absent",
        ]
    );
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node(CLASS)
        .ok_or("Executable ReferenceLatent binding required")?;
    let mut compared = 0;
    let mut gaps = 0;
    for case in cases {
        let name = case
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or("Source27 name")?;
        assert_eq!(
            case.get("input_immutable_values_storage_and_object_identity")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
        let gap = match name {
            "explicit-none-latent" => {
                Some("Explicit None LATENT is not an absent canonical optional handle")
            }
            "old-reference-empty" => {
                Some("Canonical TensorList does not retain a present empty reference list")
            }
            "old-reference-none" => {
                Some("Canonical TensorList does not retain a present None reference value")
            }
            "reference65" => Some("The original succeeds beyond the canonical 64-reference bound"),
            "rank3" | "rank2" | "zero-dimension" => {
                Some("The original sample rank or extent is outside canonical latent admission")
            }
            "samples-none" | "nested-samples" => {
                Some("The original reference value is not a canonical single Tensor")
            }
            "missing-samples"
            | "latent-integer"
            | "old-reference-tuple"
            | "null-entry-metadata"
            | "short-entry" => Some(
                "Original Python errors and untyped input representations have no exact canonical error comparison",
            ),
            _ => None,
        };
        if let Some(reason) = gap {
            gaps += 1;
            eprintln!("Source27 {name}: unrepresented gap: {reason}");
            continue;
        }
        assert_eq!(
            case.pointer("/outcome/kind")
                .and_then(serde_json::Value::as_str),
            Some("success")
        );
        assert_eq!(
            case.pointer("/outcome/io_type")
                .and_then(serde_json::Value::as_str),
            Some("comfy_api.latest._io.NodeOutput")
        );
        for field in ["result_is_args", "index_0_is_args_0"] {
            assert_eq!(
                case.get("outcome")
                    .and_then(|value| value.get(field))
                    .and_then(serde_json::Value::as_bool),
                Some(true)
            );
        }
        let before =
            Source27Graph::checked(case.get("before_input").ok_or("Source27 before graph")?)?;
        let after = Source27Graph::checked(
            case.get("input_and_output_identity_graph")
                .ok_or("Source27 after graph")?,
        )?;
        for (identifier, object) in &before.objects {
            assert_eq!(after.objects.get(identifier).copied(), Some(*object));
        }
        assert_eq!(after.root("io_args")?, after.root("io_result")?);
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
        let backend = Arc::new(backend);
        let generation = NativeHandleStoreGeneration::new()?;
        let setup = context(
            backend.clone(),
            &authority,
            generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62270))),
            CancellationToken::default(),
        )?;
        let mut storages = BTreeMap::new();
        let original =
            source27_conditioning(&setup, &before, before.root("conditioning")?, &mut storages)?;
        let original_digest = original.digest().to_owned();
        let original_handle = setup.handle_store().publish(
            NativeStoredPayload::Conditioning(original.clone()),
            &setup.cancellation,
        )?;
        let absent = before
            .root("latent_argument_absent")?
            .get("value")
            .and_then(serde_json::Value::as_bool)
            .ok_or("Source27 explicit latent-absence flag")?;
        let mut retained_latent = None;
        let latent_handle = if absent {
            None
        } else {
            let values = before.dictionary(before.root("latent")?)?;
            let samples = source27_tensor(
                &setup,
                &before,
                values.get("samples").ok_or("Source27 latent samples")?,
                &mut storages,
            )?;
            let noise_mask = values
                .get("noise_mask")
                .map(|reference| source27_tensor(&setup, &before, reference, &mut storages))
                .transpose()?;
            let batch_indices = values
                .get("batch_index")
                .map(|reference| -> Result<Vec<u64>, Box<dyn Error>> {
                    before
                        .list(reference)?
                        .iter()
                        .map(|value| {
                            if value.get("kind").and_then(serde_json::Value::as_str) != Some("int")
                            {
                                return Err("Source27 integer batch index required".into());
                            }
                            value
                                .get("value")
                                .and_then(serde_json::Value::as_u64)
                                .ok_or_else(|| "Source27 unsigned batch index required".into())
                        })
                        .collect()
                })
                .transpose()?;
            let compute = setup.compute_session()?;
            let execution = compute.execution_context(&setup)?;
            let bundle = Arc::new(NativeLatentBundle::single(
                samples,
                noise_mask,
                batch_indices,
                NativeLatentMetadata::default(),
                &execution,
            )?);
            let digest = bundle.semantic_digest_sha256().to_owned();
            let handle = setup.handle_store().publish(
                NativeStoredPayload::Latent(bundle.clone()),
                &setup.cancellation,
            )?;
            retained_latent = Some((bundle, digest));
            Some(handle)
        };
        for (left_name, left) in &storages {
            for (right_name, right) in &storages {
                assert_eq!(
                    left.storage_id() == right.storage_id(),
                    left_name == right_name
                );
            }
        }
        source27_assert_conditioning(&original, &before, before.root("conditioning")?, &storages)?;
        let before_node_bytes = backend.memory_snapshot().current_bytes;
        let first_handle = output(smol::block_on(node.execute(
            setup.clone(),
            inputs(&original_handle, latent_handle.as_ref()),
        ))?)?;
        let first = resolved(&setup, &first_handle)?;
        let final_handle = if name == "two-executions-append" {
            source27_assert_conditioning(
                &first,
                &after,
                after.root("first_output_conditioning")?,
                &storages,
            )?;
            output(smol::block_on(node.execute(
                setup.clone(),
                inputs(&first_handle, latent_handle.as_ref()),
            ))?)?
        } else {
            first_handle
        };
        let result = resolved(&setup, &final_handle)?;
        source27_assert_conditioning(
            &result,
            &after,
            after.root("output_conditioning")?,
            &storages,
        )?;
        assert_eq!(final_handle == original_handle, absent);
        assert_eq!(Arc::ptr_eq(&result, &original), absent);
        assert_eq!(
            case.pointer("/outcome/output_is_input_conditioning")
                .and_then(serde_json::Value::as_bool),
            Some(absent)
        );
        assert_eq!(
            resolved(&setup, &original_handle)?.digest(),
            original_digest
        );
        source27_assert_conditioning(&original, &before, before.root("conditioning")?, &storages)?;
        if let Some((bundle, digest)) = retained_latent {
            assert_eq!(bundle.semantic_digest_sha256(), digest);
            let handle = latent_handle
                .as_ref()
                .ok_or("Retained latent handle required")?;
            let payload =
                setup
                    .handle_store()
                    .resolve(handle, handle.handle_type(), &setup.cancellation)?;
            let NativeStoredPayload::Latent(retained) = payload.as_ref() else {
                return Err("Retained canonical latent required".into());
            };
            assert!(Arc::ptr_eq(retained, &bundle));
            assert_eq!(retained.semantic_digest_sha256(), digest);
        }
        assert_eq!(backend.memory_snapshot().current_bytes, before_node_bytes);
        compared += 1;
        eprintln!(
            "Source27 {name}: exact typed F32 words, strides, binary64 options, pooled/reference presence, ordered append, storage aliases and immutable native inputs asserted; Python containers, omitted numeric-key presence and unrelated fields remain outside this projection"
        );
    }
    assert_eq!((compared, gaps), (13, 14));
    eprintln!(
        "Source27 typed projections13; representation/error gaps14. Python container/scalar identity and unrelated metadata remain unproved."
    );
    Ok(())
}
fn context(
    backend: Arc<CpuBackend>,
    authority: &CpuWorkspaceAuthority,
    store: Arc<dyn NativeHandleStore>,
    cancellation: CancellationToken,
) -> Result<NativeNodeContext, Box<dyn Error>> {
    let node = NodeId("reference-latent".to_owned());
    let scratch = authority.authorize_workspace(LIMIT / 2)?;
    let service = NativeNodeServiceIdentity::checked(
        Uuid::from_u128(0x62201),
        store.attempt_id(),
        node.clone(),
    )?;
    let compute = NativeNodeComputeSession::checked(service, backend, StreamId::DEFAULT, &scratch)?;
    Ok(NativeNodeContext::new_with_services(
        PromptId(Uuid::from_u128(0x62202)),
        store.attempt_id(),
        node,
        cancellation,
        scratch,
        store,
        NativeNodeServices::checked(None, None, Some(compute))?,
    )?)
}
fn value(handle: &NativeOpaqueHandle) -> NativeValue {
    NativeValue::Handle {
        value: handle.clone(),
    }
}
fn inputs(
    conditioning: &NativeOpaqueHandle,
    latent: Option<&NativeOpaqueHandle>,
) -> BTreeMap<String, NativeValue> {
    let mut inputs = BTreeMap::from([("conditioning".to_owned(), value(conditioning))]);
    if let Some(latent) = latent {
        inputs.insert("latent".to_owned(), value(latent));
    }
    inputs
}
fn output(outcome: NativeNodeOutcome) -> Result<NativeOpaqueHandle, Box<dyn Error>> {
    let NativeNodeOutcome::Values {
        outputs,
        ui,
        effects,
    } = outcome
    else {
        return Err("Values required".into());
    };
    assert!(ui.is_none() && effects.is_empty());
    let [NativeValue::Handle { value }] = outputs.as_slice() else {
        return Err("Single conditioning output required".into());
    };
    Ok(value.clone())
}
fn resolved(
    context: &NativeNodeContext,
    handle: &NativeOpaqueHandle,
) -> Result<Arc<ConditioningSet>, Box<dyn Error>> {
    let retained =
        context
            .handle_store()
            .resolve(handle, handle.handle_type(), &context.cancellation)?;
    let NativeStoredPayload::Conditioning(conditioning) = retained.as_ref() else {
        return Err("Conditioning required".into());
    };
    Ok(conditioning.clone())
}
fn references(entry: &ConditioningEntry) -> Result<&[Tensor], Box<dyn Error>> {
    match entry
        .metadata()
        .get(ConditioningMetadataKey::ReferenceLatents)
    {
        Some(ConditioningMetadataValue::TensorList(values)) => Ok(values),
        _ => Err("Reference list required".into()),
    }
}
fn prepare(
    context: &NativeNodeContext,
    shape: &[u64],
) -> Result<
    (
        NativeOpaqueHandle,
        NativeOpaqueHandle,
        Arc<ConditioningSet>,
        Tensor,
    ),
    Box<dyn Error>,
> {
    let compute = context.compute_session()?;
    let execution = compute.execution_context(context)?;
    let elements = shape.iter().try_fold(1_u64, |count, dimension| {
        count.checked_mul(*dimension).ok_or("Shape overflow")
    })?;
    let samples = tensor_from_f32(
        compute.backend(),
        shape,
        &vec![0.25; usize::try_from(elements)?],
        &execution,
    )?;
    let mut mask_shape = shape.to_vec();
    *mask_shape.get_mut(1).ok_or("Mask channel dimension")? = 1;
    let mask_elements = mask_shape.iter().try_fold(1_u64, |count, dimension| {
        count.checked_mul(*dimension).ok_or("Mask shape overflow")
    })?;
    let mask = tensor_from_f32(
        compute.backend(),
        &mask_shape,
        &vec![0.5; usize::try_from(mask_elements)?],
        &execution,
    )?;
    let batch = *shape.first().ok_or("Batch dimension")?;
    let latent = NativeLatentBundle::single(
        samples.clone(),
        Some(mask),
        Some((0..batch).collect()),
        NativeLatentMetadata::default(),
        &execution,
    )?;
    let latent = context.handle_store().publish(
        NativeStoredPayload::Latent(Arc::new(latent)),
        &context.cancellation,
    )?;
    let attention = tensor_from_f32(
        compute.backend(),
        &[1, 2, 2],
        &[0.25, -0.0, 1.0, -0.5],
        &execution,
    )?;
    let mut entries = Vec::new();
    for index in 0..2 {
        let mut metadata = ConditioningMetadataMap::default();
        if index == 1 && matches!(shape.len(), 4 | 5) {
            metadata.insert(
                ConditioningMetadataKey::ReferenceLatents,
                ConditioningMetadataValue::TensorList(vec![samples.clone()]),
            )?;
            metadata.insert(
                ConditioningMetadataKey::PooledOutput,
                ConditioningMetadataValue::PooledOutput(None),
            )?;
        }
        entries.push(ConditioningEntry::checked_with_metadata(
            format!("entry.{index}"),
            ConditioningValue::cross_attention(attention.clone())?,
            ConditioningEntryOptions {
                strength: 0.5 + f64::from(index),
                ..Default::default()
            },
            ConditioningReferences::default(),
            metadata,
        )?);
    }
    let conditioning = Arc::new(ConditioningSet::checked(
        ConditioningIdentity::new(
            "reference-test",
            sd15_model_family_identity()?,
            sd15_latent_format_identity()?,
        )?,
        entries,
        &context.cancellation,
    )?);
    let handle = context.handle_store().publish(
        NativeStoredPayload::Conditioning(conditioning.clone()),
        &context.cancellation,
    )?;
    Ok((handle, latent, conditioning, samples))
}
#[test]
fn task622_reference_latent_exact_schema_rank4_rank5_append_preserves_source_aliases() -> TestResult
{
    let registry = generated_native_node_registry_projection(None)?;
    let descriptor = registry.descriptor(CLASS).ok_or("Registered descriptor")?;
    descriptor.validate_exact_schema_v2()?;
    assert_eq!(
        descriptor
            .inputs
            .iter()
            .map(|input| (input.name.as_str(), input.required, input.cardinality))
            .collect::<Vec<_>>(),
        [
            (
                "conditioning",
                true,
                sim_nodes::NativePortCardinality::Mapped
            ),
            ("latent", false, sim_nodes::NativePortCardinality::Mapped)
        ]
    );
    let node = registry.node(CLASS).ok_or("Executable binding")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    for shape in [vec![1, 4, 1, 1], vec![2, 4, 1, 1, 1]] {
        let generation = NativeHandleStoreGeneration::new()?;
        let setup = context(
            backend.clone(),
            &authority,
            generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62210))),
            CancellationToken::default(),
        )?;
        let (conditioning, latent, original, samples) = prepare(&setup, &shape)?;
        let latent_projection = {
            let retained =
                setup
                    .handle_store()
                    .resolve(&latent, latent.handle_type(), &setup.cancellation)?;
            let NativeStoredPayload::Latent(bundle) = retained.as_ref() else {
                return Err("Canonical latent required".into());
            };
            assert!(bundle.noise_mask().is_some());
            assert!(bundle.batch_indices().is_some());
            serde_json::to_value(bundle.projection())?
        };
        let digest = original.digest().to_owned();
        let before = backend.memory_snapshot().current_bytes;
        let arguments = inputs(&conditioning, Some(&latent));
        assert_eq!(
            node.cache_dependencies(&setup, &arguments)?
                .artifact_digests
                .len(),
            2
        );
        let handle = output(smol::block_on(node.execute(setup.clone(), arguments))?)?;
        let actual = resolved(&setup, &handle)?;
        assert_eq!(actual.identity(), original.identity());
        assert_eq!(actual.entries().len(), original.entries().len());
        for (index, (before, after)) in original.entries().iter().zip(actual.entries()).enumerate()
        {
            assert_eq!(before.identifier(), after.identifier());
            assert_reference_options_preserved(before.options(), after.options())?;
            let (
                ConditioningValue::CrossAttention(before),
                ConditioningValue::CrossAttention(after),
            ) = (before.value(), after.value())
            else {
                return Err("Attention tensor required".into());
            };
            assert_eq!(before.storage_id(), after.storage_id());
            assert_eq!(
                references(actual.entries().get(index).ok_or("Entry")?)?
                    .iter()
                    .map(Tensor::storage_id)
                    .collect::<Vec<_>>(),
                vec![samples.storage_id(); index + 1]
            );
        }
        assert_eq!(original.digest(), digest);
        {
            let retained =
                setup
                    .handle_store()
                    .resolve(&latent, latent.handle_type(), &setup.cancellation)?;
            let NativeStoredPayload::Latent(bundle) = retained.as_ref() else {
                return Err("Canonical latent required".into());
            };
            assert_eq!(
                serde_json::to_value(bundle.projection())?,
                latent_projection
            );
        }
        assert!(
            original
                .entries()
                .first()
                .ok_or("First")?
                .metadata()
                .get(ConditioningMetadataKey::ReferenceLatents)
                .is_none()
        );
        assert!(
            actual
                .entries()
                .first()
                .ok_or("First")?
                .metadata()
                .get(ConditioningMetadataKey::PooledOutput)
                .is_none()
        );
        assert!(matches!(
            actual
                .entries()
                .get(1)
                .ok_or("Second")?
                .metadata()
                .get(ConditioningMetadataKey::PooledOutput),
            Some(ConditioningMetadataValue::PooledOutput(None))
        ));
        assert_eq!(actual.resident_parts()?.tensor_allocations().len(), 2);
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        setup.handle_store().revoke(&handle, &setup.cancellation)?;
    }
    Ok(())
}
#[test]
fn task622_reference_latent_absent_branch_is_exact_handle_and_explicit_null_is_admission_gap()
-> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node(CLASS).ok_or("Binding")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::with_capacity(2)?;
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62220))),
        CancellationToken::default(),
    )?;
    let (conditioning, _, original, _) = prepare(&setup, &[1, 4, 1, 1])?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    assert_eq!(
        output(smol::block_on(
            node.execute(setup.clone(), inputs(&conditioning, None))
        )?)?,
        conditioning
    );
    assert_eq!(
        node.cache_dependencies(&setup, &inputs(&conditioning, None))?
            .artifact_digests
            .len(),
        1
    );
    let mut explicit_null = inputs(&conditioning, None);
    explicit_null.insert(
        "latent".to_owned(),
        NativeValue::Primitive {
            value: sim_nodes::NativePrimitive::Null,
        },
    );
    let error = smol::block_on(node.execute(setup.clone(), explicit_null))
        .err()
        .ok_or("Null handle admitted")?;
    assert_eq!(error.code, "invalid_reference_latent_inputs");
    assert_eq!(original.digest(), resolved(&setup, &conditioning)?.digest());
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    Ok(())
}
#[test]
fn task622_reference_latent_foreign_stale_wrong_type_rank3_and_cancellation_do_not_publish()
-> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node(CLASS).ok_or("Binding")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::new()?;
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62230))),
        CancellationToken::default(),
    )?;
    let (conditioning, latent, _, _) = prepare(&setup, &[1, 4, 1, 1])?;
    let foreign_generation = NativeHandleStoreGeneration::new()?;
    let foreign = context(
        backend.clone(),
        &authority,
        foreign_generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62231))),
        CancellationToken::default(),
    )?;
    let (_, foreign_latent, _, _) = prepare(&foreign, &[1, 4, 1, 1])?;
    let (_, rank3, _, _) = prepare(&setup, &[1, 4, 2])?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    for candidate in [&conditioning, &foreign_latent, &rank3] {
        let error =
            smol::block_on(node.execute(setup.clone(), inputs(&conditioning, Some(candidate))))
                .err()
                .ok_or("Unsupported input succeeded")?;
        assert_eq!(error.kind, NativeNodeFailureKind::Failure);
        assert_eq!(
            (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            before
        );
    }
    let mut cancelled = setup.clone();
    cancelled.cancellation = CancellationToken::default();
    cancelled.cancellation.cancel();
    assert_eq!(
        smol::block_on(node.execute(cancelled, inputs(&conditioning, Some(&latent))))
            .err()
            .ok_or("Cancelled succeeded")?
            .kind,
        NativeNodeFailureKind::Interrupted
    );
    setup.handle_store().revoke(&rank3, &setup.cancellation)?;
    let after_revoke = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    assert!(
        node.cache_dependencies(&setup, &inputs(&conditioning, Some(&rank3)))
            .is_err()
    );
    assert!(
        smol::block_on(node.execute(setup.clone(), inputs(&conditioning, Some(&rank3)))).is_err()
    );
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        after_revoke
    );
    let recovered = output(smol::block_on(
        node.execute(setup.clone(), inputs(&conditioning, Some(&latent))),
    )?)?;
    setup
        .handle_store()
        .revoke(&recovered, &setup.cancellation)?;
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        after_revoke
    );
    Ok(())
}
#[derive(Debug)]
struct CancelAfterPublish {
    inner: Arc<dyn NativeHandleStore>,
    armed: AtomicBool,
}
impl NativeHandleStore for CancelAfterPublish {
    fn identity(&self) -> NativeHandleStoreIdentity {
        self.inner.identity()
    }
    fn attempt_id(&self) -> AttemptId {
        self.inner.attempt_id()
    }
    fn resolve(
        &self,
        handle: &NativeOpaqueHandle,
        expected: &NativeHandleType,
        cancellation: &CancellationToken,
    ) -> Result<NativeResolvedPayload, NativeHandleStoreError> {
        self.inner.resolve(handle, expected, cancellation)
    }
    fn publish(
        &self,
        payload: NativeStoredPayload,
        cancellation: &CancellationToken,
    ) -> Result<NativeOpaqueHandle, NativeHandleStoreError> {
        let handle = self.inner.publish(payload, cancellation)?;
        if self.armed.swap(false, Ordering::AcqRel) {
            cancellation.cancel();
        }
        Ok(handle)
    }
    fn revoke(
        &self,
        handle: &NativeOpaqueHandle,
        cancellation: &CancellationToken,
    ) -> Result<(), NativeHandleStoreError> {
        self.inner.revoke(handle, cancellation)
    }
}
#[test]
fn task622_reference_latent_publication_capacity_postpublish_cancel_rollback_and_recovery()
-> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node(CLASS).ok_or("Binding")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    for capacity in [2, 8] {
        let generation = NativeHandleStoreGeneration::with_capacity(capacity)?;
        let store = Arc::new(CancelAfterPublish {
            inner: generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62240))),
            armed: AtomicBool::new(false),
        });
        let setup = context(
            backend.clone(),
            &authority,
            store.clone(),
            CancellationToken::default(),
        )?;
        let (conditioning, latent, _, _) = prepare(&setup, &[1, 4, 1, 1])?;
        let original_digest = resolved(&setup, &conditioning)?.digest().to_owned();
        let before = (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes,
        );
        store.armed.store(capacity == 8, Ordering::Release);
        let invocation = context(
            backend.clone(),
            &authority,
            store.clone(),
            CancellationToken::default(),
        )?;
        let error = smol::block_on(node.execute(invocation, inputs(&conditioning, Some(&latent))))
            .err()
            .ok_or("Failure not reached")?;
        if capacity == 2 {
            assert_eq!(error.kind, NativeNodeFailureKind::Failure);
            assert!(
                error
                    .message
                    .contains("native handle store capacity is exhausted")
            );
        } else {
            assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        }
        assert_eq!(
            (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            before
        );
        if capacity == 2 {
            assert_eq!(
                output(smol::block_on(
                    node.execute(setup.clone(), inputs(&conditioning, None))
                )?)?,
                conditioning
            );
        } else {
            let recovered = output(smol::block_on(
                node.execute(setup.clone(), inputs(&conditioning, Some(&latent))),
            )?)?;
            setup
                .handle_store()
                .revoke(&recovered, &setup.cancellation)?;
        }
        assert_eq!(
            (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            before
        );
        assert_eq!(store.identity(), setup.handle_store().identity());
        assert_eq!(resolved(&setup, &conditioning)?.digest(), original_digest);
    }
    Ok(())
}
#[test]
fn task622_reference_latent_real_producer_mapper_noop_persistence_cache_and_report_leases()
-> TestResult {
    real_producer_reference_mapper_cache_and_report_leases(
        CLASS,
        ConditioningMetadataKey::ReferenceLatents,
        references,
    )
}

#[test]
fn task622_reference_audio_real_producer_mapper_noop_persistence_cache_and_report_leases()
-> TestResult {
    real_producer_reference_mapper_cache_and_report_leases(
        "ReferenceTimbreAudio",
        ConditioningMetadataKey::ReferenceAudioTimbreLatents,
        audio_references,
    )
}

fn real_producer_reference_mapper_cache_and_report_leases(
    class_type: &str,
    reference_key: ConditioningMetadataKey,
    reference_reader: for<'a> fn(&'a ConditioningEntry) -> Result<&'a [Tensor], Box<dyn Error>>,
) -> TestResult {
    const WORKFLOW_BUDGET: u64 = 128 * 1024 * 1024;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(WORKFLOW_BUDGET)?;
    let backend = Arc::new(backend);
    let mut descriptors = BTreeMap::new();
    let executor =
        sim_runtime::NativeImageExecutor::new_with_generated_registry_diffusion_and_registration(
            sim_types::ProfileId(Uuid::from_u128(0x62250)),
            BTreeMap::new(),
            true,
            backend.clone(),
            Arc::new(sim_test_support::NativeDiffusionFixture::checked_in()),
            |registry| {
                for class_type in [
                    "CheckpointLoaderSimple",
                    "CLIPTextEncode",
                    "EmptySD3LatentImage",
                    class_type,
                ] {
                    let descriptor = registry.descriptor(class_type).ok_or_else(|| {
                        sim_runtime::NativeImageRuntimeError::Registry(format!(
                            "Missing real producer/reference {class_type}"
                        ))
                    })?;
                    if class_type != "CheckpointLoaderSimple" {
                        descriptor.validate_exact_schema_v2().map_err(|error| {
                            sim_runtime::NativeImageRuntimeError::Registry(error.to_string())
                        })?;
                    }
                    if registry.binding_disposition(class_type)
                        != Some(sim_nodes::NativeNodeBindingDisposition::Executable)
                    {
                        return Err(sim_runtime::NativeImageRuntimeError::Registry(format!(
                            "Unbound {class_type}"
                        )));
                    }
                    descriptors.insert(class_type.to_owned(), descriptor.clone());
                }
                Ok(())
            },
        )?;
    let link = |identifier: &str, output_index| sim_runtime::InputBinding::Link {
        source: NodeId(identifier.to_owned()),
        output_index,
        lazy: false,
        mode: sim_nodes::NativePortCardinality::Mapped,
    };
    let string = |text: &str| NativeValue::Primitive {
        value: sim_nodes::NativePrimitive::String(text.to_owned()),
    };
    let integer = |number| sim_runtime::InputBinding::Literal {
        value: NativeValue::Primitive {
            value: sim_nodes::NativePrimitive::Integer(number),
        },
    };
    let definitions = vec![
        (
            "checkpoint",
            "CheckpointLoaderSimple",
            BTreeMap::from([(
                "ckpt_name".to_owned(),
                sim_runtime::InputBinding::Literal {
                    value: string("model.safetensors"),
                },
            )]),
        ),
        (
            "encode",
            "CLIPTextEncode",
            BTreeMap::from([
                ("clip".to_owned(), link("checkpoint", 1)),
                (
                    "text".to_owned(),
                    sim_runtime::InputBinding::Literal {
                        value: NativeValue::List {
                            values: vec![string("a test"), string("a test")],
                        },
                    },
                ),
            ]),
        ),
        (
            "latent",
            "EmptySD3LatentImage",
            BTreeMap::from([
                ("width".to_owned(), integer(64)),
                ("height".to_owned(), integer(64)),
                ("batch_size".to_owned(), integer(1)),
            ]),
        ),
        (
            "reference",
            class_type,
            BTreeMap::from([
                ("conditioning".to_owned(), link("encode", 0)),
                ("latent".to_owned(), link("latent", 0)),
            ]),
        ),
        (
            "noop",
            class_type,
            BTreeMap::from([("conditioning".to_owned(), link("reference", 0))]),
        ),
    ];
    let mut nodes = BTreeMap::new();
    let mut order = Vec::new();
    let mut output_nodes = Vec::new();
    for (identifier, class_type, inputs) in definitions {
        let identifier = NodeId(identifier.to_owned());
        let mut descriptor = descriptors
            .get(class_type)
            .ok_or("Actual descriptor")?
            .clone();
        if class_type != "CheckpointLoaderSimple" {
            descriptor.output_node = true;
            output_nodes.push(identifier.clone());
        }
        nodes.insert(
            identifier.clone(),
            sim_runtime::CompiledNode {
                id: identifier.clone(),
                class_type: class_type.to_owned(),
                descriptor,
                inputs,
                unknown: BTreeMap::new(),
            },
        );
        order.push(identifier);
    }
    output_nodes.sort();
    let plan = sim_runtime::CompiledPlan {
        prompt_id: PromptId(Uuid::from_u128(0x62251)),
        client_id: None,
        prompt_number: None,
        extra_data: BTreeMap::new(),
        unknown: BTreeMap::new(),
        nodes,
        static_required_nodes: order.iter().cloned().collect(),
        topological_order: order,
        output_nodes,
        provider_execution: None,
        persistence_unknown_fields: BTreeMap::new(),
    };
    plan.validate_integrity()?;
    let restored: sim_runtime::CompiledPlan = serde_json::from_slice(&serde_json::to_vec(&plan)?)?;
    let generation = executor.handle_store_generation_for_test();
    assert!(generation.is_empty());
    let attempt = AttemptId(Uuid::from_u128(0x62252));
    let first = executor.execute_blocking(
        &restored,
        attempt,
        CancellationToken::default(),
        0,
        authority.authorize_workspace(WORKFLOW_BUDGET)?,
    )?;
    assert_eq!(
        first.report.state,
        sim_runtime::AttemptState::Succeeded,
        "{:?}",
        first.report.error
    );
    assert_eq!(first.report.cache_hits, 0);
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(attempt),
        CancellationToken::default(),
    )?;
    let column = |result: &sim_runtime::NativeImageExecutionResult,
                  identifier: &str|
     -> Result<Vec<NativeOpaqueHandle>, Box<dyn Error>> {
        let [NativeValue::List { values }] = result
            .report
            .outputs
            .get(&NodeId(identifier.to_owned()))
            .ok_or("Committed mapped column")?
            .as_slice()
        else {
            return Err("Actual single mapped output column required".into());
        };
        values
            .iter()
            .map(|value| match value {
                NativeValue::Handle { value } => Ok(value.clone()),
                _ => Err("Actual canonical handle required".into()),
            })
            .collect()
    };
    let source = column(&first, "encode")?;
    let latent_outputs = first
        .report
        .outputs
        .get(&NodeId("latent".to_owned()))
        .ok_or("Committed scalar latent producer output")?;
    let [NativeValue::Handle { value: latent }] = latent_outputs.as_slice() else {
        return Err("Actual single scalar latent producer output required".into());
    };
    let latent = latent.clone();
    let actual = column(&first, "reference")?;
    assert_eq!(source.len(), 2);
    assert_eq!(actual.len(), source.len());
    assert_eq!(column(&first, "noop")?, actual);
    {
        let retained_latent =
            setup
                .handle_store()
                .resolve(&latent, latent.handle_type(), &setup.cancellation)?;
        let NativeStoredPayload::Latent(bundle) = retained_latent.as_ref() else {
            return Err("Canonical latent required".into());
        };
        let samples = bundle.samples().tensor().ok_or("Real tensor samples")?;
        assert_eq!(samples.descriptor().shape(), &[1, 16, 8, 8]);
        for (source, actual) in source.iter().zip(&actual) {
            let source = resolved(&setup, source)?;
            let actual = resolved(&setup, actual)?;
            assert_eq!(source.identity(), actual.identity());
            assert_eq!(source.entries().len(), actual.entries().len());
            for (source, actual) in source.entries().iter().zip(actual.entries()) {
                let (
                    ConditioningValue::CrossAttention(source_tensor),
                    ConditioningValue::CrossAttention(actual_tensor),
                ) = (source.value(), actual.value())
                else {
                    return Err("CLIP tensor aliases required".into());
                };
                assert_eq!(source_tensor.storage_id(), actual_tensor.storage_id());
                assert_reference_options_preserved(source.options(), actual.options())?;
                assert_eq!(
                    reference_reader(actual)?
                        .iter()
                        .map(Tensor::storage_id)
                        .collect::<Vec<_>>(),
                    [samples.storage_id()]
                );
                assert!(source.metadata().get(reference_key).is_none());
            }
        }
    }

    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
        executor.cache_len_for_test(),
    );
    let replay = executor.execute_blocking(
        &restored,
        AttemptId(Uuid::from_u128(0x62253)),
        CancellationToken::default(),
        0,
        authority.authorize_workspace(WORKFLOW_BUDGET)?,
    )?;
    assert_eq!(
        replay.report.state,
        sim_runtime::AttemptState::Succeeded,
        "{:?}",
        replay.report.error
    );
    assert_eq!(replay.report.cache_hits, 5);
    assert_eq!(replay.report.outputs, first.report.outputs);
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes,
            executor.cache_len_for_test()
        ),
        before
    );
    drop(executor);
    assert!(!generation.is_empty());
    drop(first);
    assert!(!generation.is_empty());
    drop(replay);
    assert!(generation.is_empty());
    assert_eq!(generation.resident_bytes(), 0);
    assert_eq!(backend.memory_snapshot().current_bytes, 0);
    for handle in actual {
        assert!(matches!(
            setup
                .handle_store()
                .resolve(&handle, handle.handle_type(), &setup.cancellation),
            Err(NativeHandleStoreError::Missing(_))
        ));
    }
    Ok(())
}

fn assert_reference_options_preserved(
    original: &ConditioningEntryOptions,
    actual: &ConditioningEntryOptions,
) -> TestResult {
    use sim_model::conditioning::ConditioningRegion;
    assert_eq!(original.strength.to_bits(), actual.strength.to_bits());
    match (&original.region, &actual.region) {
        (None, None) => {}
        (
            Some(ConditioningRegion::Absolute {
                sizes: original_sizes,
                offsets: original_offsets,
            }),
            Some(ConditioningRegion::Absolute {
                sizes: actual_sizes,
                offsets: actual_offsets,
            }),
        ) => {
            assert_eq!(original_sizes, actual_sizes);
            assert_eq!(original_offsets, actual_offsets);
        }
        (
            Some(ConditioningRegion::Percentage {
                sizes: original_sizes,
                offsets: original_offsets,
            }),
            Some(ConditioningRegion::Percentage {
                sizes: actual_sizes,
                offsets: actual_offsets,
            }),
        ) => {
            assert_eq!(
                original_sizes
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                actual_sizes
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                original_offsets
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                actual_offsets
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        _ => return Err("Conditioning region presence or representation changed".into()),
    }
    assert_eq!(
        original.window.start_percent().to_bits(),
        actual.window.start_percent().to_bits()
    );
    assert_eq!(
        original.window.end_percent().to_bits(),
        actual.window.end_percent().to_bits()
    );
    assert_eq!(original.default_region, actual.default_region);
    match (&original.mask, &actual.mask) {
        (None, None) => {}
        (Some(original), Some(actual)) => {
            assert_eq!(original.tensor().storage_id(), actual.tensor().storage_id());
            assert_eq!(original.tensor().descriptor(), actual.tensor().descriptor());
            assert_eq!(original.strength().to_bits(), actual.strength().to_bits());
            assert_eq!(original.feather(), actual.feather());
            assert_eq!(
                original.set_region_to_nonzero_bounds(),
                actual.set_region_to_nonzero_bounds()
            );
        }
        _ => return Err("Conditioning mask presence changed".into()),
    }
    Ok(())
}

#[test]
fn task622_reference_latent_empty_conditioning_present_and_absent_after_task618() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node(CLASS).ok_or("Binding")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::with_capacity(4)?;
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62260))),
        CancellationToken::default(),
    )?;
    let (_, latent, original, _) = prepare(&setup, &[1, 4, 1, 1])?;
    let empty = Arc::new(ConditioningSet::checked(
        original.identity().clone(),
        Vec::new(),
        &setup.cancellation,
    )?);
    let empty_handle = setup.handle_store().publish(
        NativeStoredPayload::Conditioning(empty.clone()),
        &setup.cancellation,
    )?;
    let original_digest = empty.digest().to_owned();
    let before = backend.memory_snapshot().current_bytes;
    let absent = inputs(&empty_handle, None);
    assert_eq!(
        output(smol::block_on(node.execute(setup.clone(), absent.clone()))?)?,
        empty_handle
    );
    let absent_value = resolved(&setup, &empty_handle)?;
    assert!(Arc::ptr_eq(&absent_value, &empty));
    assert!(absent_value.entries().is_empty());
    assert_eq!(
        node.cache_dependencies(&setup, &absent)?
            .artifact_digests
            .len(),
        1
    );
    let present = inputs(&empty_handle, Some(&latent));
    let appended = output(smol::block_on(
        node.execute(setup.clone(), present.clone()),
    )?)?;
    let appended_value = resolved(&setup, &appended)?;
    assert!(appended_value.entries().is_empty());
    assert_eq!(appended_value.identity(), empty.identity());
    assert_eq!(appended_value.digest(), original_digest);
    assert_eq!(empty.digest(), original_digest);
    assert_eq!(
        node.cache_dependencies(&setup, &present)?
            .artifact_digests
            .len(),
        2
    );
    assert_eq!(backend.memory_snapshot().current_bytes, before);
    Ok(())
}

fn audio_references(entry: &ConditioningEntry) -> Result<&[Tensor], Box<dyn Error>> {
    match entry
        .metadata()
        .get(ConditioningMetadataKey::ReferenceAudioTimbreLatents)
    {
        Some(ConditioningMetadataValue::ReferenceAudioTimbreLatents(Some(values))) => Ok(values),
        _ => Err("Audio timbre reference list required".into()),
    }
}
#[test]
fn task622_reference_audio_source_schema_cache_and_identity() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ReferenceTimbreAudio")
        .ok_or("Audio binding")?;
    let descriptor = registry
        .descriptor("ReferenceTimbreAudio")
        .ok_or("Audio descriptor")?;
    assert_eq!(node.class_type(), "ReferenceTimbreAudio");
    descriptor.validate_exact_schema_v2()?;
    assert_eq!(descriptor.inputs.len(), 2);
    let latent = descriptor
        .inputs
        .iter()
        .find(|input| input.name == "latent")
        .ok_or("Latent port")?;
    assert!(!latent.required && !latent.hidden && !latent.lazy && !latent.allows_literal);
    assert_eq!(descriptor.outputs.len(), 1);
    assert_eq!(
        node.implementation_version(),
        "source-reference-audio-timbre-0528-v1"
    );
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::with_capacity(8)?;
    let setup = context(
        backend,
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62280))),
        CancellationToken::default(),
    )?;
    let (conditioning, latent, original, _) = prepare(&setup, &[1, 64, 2])?;
    let present = inputs(&conditioning, Some(&latent));
    let absent = inputs(&conditioning, None);
    assert!(node.demanded_lazy_inputs(&setup, &present)?.is_empty());
    assert_eq!(
        node.cache_change_token(&present)?,
        node.implementation_version()
    );
    let dependencies = node.cache_dependencies(&setup, &present)?;
    assert_eq!(dependencies.artifact_digests.len(), 2);
    assert_eq!(
        node.cache_dependencies(&setup, &absent)?
            .artifact_digests
            .len(),
        1
    );
    let result = output(smol::block_on(node.execute(setup.clone(), present))?)?;
    assert_ne!(resolved(&setup, &result)?.digest(), original.digest());
    assert_eq!(resolved(&setup, &conditioning)?.digest(), original.digest());
    Ok(())
}
#[test]
fn task622_reference_audio_absent_noop_and_explicit_null_remains_schema_gap() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ReferenceTimbreAudio")
        .ok_or("Audio binding")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::with_capacity(2)?;
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62281))),
        CancellationToken::default(),
    )?;
    let (conditioning, _, original, _) = prepare(&setup, &[1, 64, 2])?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    assert_eq!(
        output(smol::block_on(
            node.execute(setup.clone(), inputs(&conditioning, None))
        )?)?,
        conditioning
    );
    let mut explicit_null = inputs(&conditioning, None);
    explicit_null.insert(
        "latent".to_owned(),
        NativeValue::Primitive {
            value: sim_nodes::NativePrimitive::Null,
        },
    );
    let error = smol::block_on(node.execute(setup.clone(), explicit_null))
        .err()
        .ok_or("Null admitted")?;
    assert_eq!(error.code, "invalid_reference_audio_timbre_inputs");
    assert_eq!(resolved(&setup, &conditioning)?.digest(), original.digest());
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    Ok(())
}
#[test]
fn task622_reference_audio_absent_none_empty_and_nonempty_metadata_append_aliases() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ReferenceTimbreAudio")
        .ok_or("Audio binding")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::with_capacity(16)?;
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62282))),
        CancellationToken::default(),
    )?;
    let (_, latent, original, samples) = prepare(&setup, &[1, 64, 2])?;
    for old in [
        None,
        Some(None),
        Some(Some(Vec::new())),
        Some(Some(vec![samples.clone(), samples.clone()])),
    ] {
        let previous_count = old
            .as_ref()
            .and_then(Option::as_ref)
            .map(Vec::len)
            .unwrap_or(0);
        let mut metadata = Vec::new();
        for entry in original.entries() {
            let mut values = entry.metadata().clone();
            if let Some(old) = &old {
                values.insert(
                    ConditioningMetadataKey::ReferenceAudioTimbreLatents,
                    ConditioningMetadataValue::ReferenceAudioTimbreLatents(old.clone()),
                )?;
            }
            metadata.push(values);
        }
        let source = Arc::new(original.with_entry_metadata(metadata, &setup.cancellation)?);
        let source_digest = source.digest().to_owned();
        let conditioning = setup.handle_store().publish(
            NativeStoredPayload::Conditioning(source.clone()),
            &setup.cancellation,
        )?;
        let before = backend.memory_snapshot().current_bytes;
        let handle = output(smol::block_on(
            node.execute(setup.clone(), inputs(&conditioning, Some(&latent))),
        )?)?;
        let actual = resolved(&setup, &handle)?;
        for (entry, original_entry) in actual.entries().iter().zip(source.entries()) {
            assert_eq!(entry.identifier(), original_entry.identifier());
            let references = audio_references(entry)?;
            assert_eq!(references.len(), previous_count + 1);
            for reference in references {
                assert_eq!(reference.storage_id(), samples.storage_id());
            }
            assert!(
                entry
                    .metadata()
                    .get(ConditioningMetadataKey::ReferenceLatents)
                    .is_none()
            );
        }
        assert_eq!(source.digest(), source_digest);
        assert_eq!(backend.memory_snapshot().current_bytes, before);
        setup.handle_store().revoke(&handle, &setup.cancellation)?;
        setup
            .handle_store()
            .revoke(&conditioning, &setup.cancellation)?;
    }
    Ok(())
}
#[test]
fn task622_reference_audio_heterogeneous_shapes_and_empty_conditioning() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ReferenceTimbreAudio")
        .ok_or("Audio binding")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::with_capacity(16)?;
    let setup = context(
        backend,
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62283))),
        CancellationToken::default(),
    )?;
    let (_, latent, original, samples) = prepare(&setup, &[1, 64, 2])?;
    let compute = setup.compute_session()?;
    let execution = compute.execution_context(&setup)?;
    let different = tensor_from_f32(compute.backend(), &[2, 7, 3, 1], &[0.5; 42], &execution)?;
    let mut metadata = original
        .entries()
        .first()
        .ok_or("Entry")?
        .metadata()
        .clone();
    metadata.insert(
        ConditioningMetadataKey::ReferenceAudioTimbreLatents,
        ConditioningMetadataValue::ReferenceAudioTimbreLatents(Some(vec![different.clone()])),
    )?;
    let source =
        original.with_entry_metadata(vec![metadata.clone(), metadata], &setup.cancellation)?;
    let source_handle = setup.handle_store().publish(
        NativeStoredPayload::Conditioning(Arc::new(source)),
        &setup.cancellation,
    )?;
    let result = output(smol::block_on(
        node.execute(setup.clone(), inputs(&source_handle, Some(&latent))),
    )?)?;
    for entry in resolved(&setup, &result)?.entries() {
        let references = audio_references(entry)?;
        assert_eq!(references.len(), 2);
        assert_eq!(
            references.first().ok_or("Old reference")?.storage_id(),
            different.storage_id()
        );
        assert_eq!(
            references.get(1).ok_or("New reference")?.storage_id(),
            samples.storage_id()
        );
    }
    let empty =
        ConditioningSet::checked(original.identity().clone(), Vec::new(), &setup.cancellation)?;
    let empty_handle = setup.handle_store().publish(
        NativeStoredPayload::Conditioning(Arc::new(empty)),
        &setup.cancellation,
    )?;
    assert_eq!(
        output(smol::block_on(
            node.execute(setup.clone(), inputs(&empty_handle, None))
        )?)?,
        empty_handle
    );
    let appended = output(smol::block_on(
        node.execute(setup.clone(), inputs(&empty_handle, Some(&latent))),
    )?)?;
    assert!(resolved(&setup, &appended)?.entries().is_empty());
    Ok(())
}
#[test]
fn task622_reference_audio_publication_capacity_postpublish_cancel_rollback_and_recovery()
-> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry.node("ReferenceTimbreAudio").ok_or("Binding")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    for capacity in [2, 8] {
        let generation = NativeHandleStoreGeneration::with_capacity(capacity)?;
        let store = Arc::new(CancelAfterPublish {
            inner: generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62240))),
            armed: AtomicBool::new(false),
        });
        let setup = context(
            backend.clone(),
            &authority,
            store.clone(),
            CancellationToken::default(),
        )?;
        let (conditioning, latent, _, _) = prepare(&setup, &[1, 64, 2])?;
        let original_digest = resolved(&setup, &conditioning)?.digest().to_owned();
        let before = (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes,
        );
        store.armed.store(capacity == 8, Ordering::Release);
        let invocation = context(
            backend.clone(),
            &authority,
            store.clone(),
            CancellationToken::default(),
        )?;
        let error = smol::block_on(node.execute(invocation, inputs(&conditioning, Some(&latent))))
            .err()
            .ok_or("Failure not reached")?;
        if capacity == 2 {
            assert_eq!(error.kind, NativeNodeFailureKind::Failure);
            assert!(
                error
                    .message
                    .contains("native handle store capacity is exhausted")
            );
        } else {
            assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
        }
        assert_eq!(
            (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            before
        );
        if capacity == 2 {
            assert_eq!(
                output(smol::block_on(
                    node.execute(setup.clone(), inputs(&conditioning, None))
                )?)?,
                conditioning
            );
        } else {
            let recovered = output(smol::block_on(
                node.execute(setup.clone(), inputs(&conditioning, Some(&latent))),
            )?)?;
            setup
                .handle_store()
                .revoke(&recovered, &setup.cancellation)?;
        }
        assert_eq!(
            (
                generation.len(),
                generation.resident_bytes(),
                backend.memory_snapshot().current_bytes
            ),
            before
        );
        assert_eq!(store.identity(), setup.handle_store().identity());
        assert_eq!(resolved(&setup, &conditioning)?.digest(), original_digest);
    }
    Ok(())
}

#[test]
fn task622_reference_audio_invalid_stale_and_pre_cancel_leave_inputs_alive() -> TestResult {
    let registry = generated_native_node_registry_projection(None)?;
    let node = registry
        .node("ReferenceTimbreAudio")
        .ok_or("Audio binding")?;
    let (backend, authority) = CpuWorkspaceAuthority::create_backend(LIMIT)?;
    let backend = Arc::new(backend);
    let generation = NativeHandleStoreGeneration::with_capacity(8)?;
    let setup = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62284))),
        CancellationToken::default(),
    )?;
    let (conditioning, latent, original, _) = prepare(&setup, &[1, 64, 2])?;
    let foreign_generation = NativeHandleStoreGeneration::with_capacity(2)?;
    let foreign = context(
        backend.clone(),
        &authority,
        foreign_generation.handle_store_for_attempt(AttemptId(Uuid::from_u128(0x62285))),
        CancellationToken::default(),
    )?;
    let (_, foreign_latent, _, _) = prepare(&foreign, &[1, 64, 2])?;
    let compute = setup.compute_session()?;
    let execution = compute.execution_context(&setup)?;
    let video = tensor_from_f32(compute.backend(), &[1, 1, 1, 1, 1], &[0.25], &execution)?;
    let audio = tensor_from_f32(compute.backend(), &[1, 1, 1], &[0.5], &execution)?;
    let nested = NativeLatentBundle::audio_video(
        video,
        audio,
        None,
        None,
        None,
        NativeLatentMetadata::default(),
        &execution,
    )?;
    let nested = setup.handle_store().publish(
        NativeStoredPayload::Latent(Arc::new(nested)),
        &setup.cancellation,
    )?;
    let boundary_before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    assert!(
        node.cache_dependencies(&setup, &inputs(&conditioning, Some(&foreign_latent)))
            .is_err()
    );
    let foreign_error =
        smol::block_on(node.execute(setup.clone(), inputs(&conditioning, Some(&foreign_latent))))
            .err()
            .ok_or("Foreign latent admitted")?;
    assert_eq!(foreign_error.kind, NativeNodeFailureKind::Failure);
    let nested_error =
        smol::block_on(node.execute(setup.clone(), inputs(&conditioning, Some(&nested))))
            .err()
            .ok_or("Nested latent admitted")?;
    assert_eq!(
        nested_error.code,
        "source_reference_audio_timbre_samples_unbound"
    );
    assert_eq!(nested_error.kind, NativeNodeFailureKind::Failure);
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes,
        ),
        boundary_before,
    );
    assert_eq!(resolved(&setup, &conditioning)?.digest(), original.digest());
    setup.handle_store().revoke(&nested, &setup.cancellation)?;
    drop(foreign);
    drop(foreign_generation);
    let mut wrong_kind = inputs(&conditioning, Some(&conditioning));
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    assert!(smol::block_on(node.execute(setup.clone(), wrong_kind.clone())).is_err());
    wrong_kind.insert("unknown".to_owned(), value(&latent));
    assert!(smol::block_on(node.execute(setup.clone(), wrong_kind)).is_err());
    let cancellation = CancellationToken::default();
    assert!(cancellation.cancel());
    let cancelled = context(
        backend.clone(),
        &authority,
        generation.handle_store_for_attempt(setup.attempt_id),
        cancellation,
    )?;
    let error = smol::block_on(node.execute(cancelled, inputs(&conditioning, Some(&latent))))
        .err()
        .ok_or("Cancellation ignored")?;
    assert_eq!(error.kind, NativeNodeFailureKind::Interrupted);
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    setup.handle_store().revoke(&latent, &setup.cancellation)?;
    let before = (
        generation.len(),
        generation.resident_bytes(),
        backend.memory_snapshot().current_bytes,
    );
    assert!(
        smol::block_on(node.execute(setup.clone(), inputs(&conditioning, Some(&latent)))).is_err()
    );
    assert_eq!(
        (
            generation.len(),
            generation.resident_bytes(),
            backend.memory_snapshot().current_bytes
        ),
        before
    );
    assert_eq!(resolved(&setup, &conditioning)?.digest(), original.digest());
    assert_eq!(
        output(smol::block_on(
            node.execute(setup.clone(), inputs(&conditioning, None))
        )?)?,
        conditioning
    );
    Ok(())
}
