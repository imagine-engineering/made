use super::*;
use futures::executor::block_on;
use graphite_glam::UVec2;
use graphite_std::application_io::TimingInformation;
use graphite_std::transform::{Footprint, RenderQuality};
use graphite_std::vector_types::vector::style::RenderMode;
use std::{collections::HashSet, error::Error, time::Duration};

type TestResult = Result<(), Box<dyn Error>>;

fn configuration() -> RenderConfig {
    RenderConfig {
        viewport: Footprint {
            transform: DAffine2::from_translation(DVec2::new(7., 11.))
                * DAffine2::from_scale(DVec2::splat(0.25)),
            resolution: UVec2::new(672, 384),
            quality: RenderQuality::Full,
        },
        scale: 1.5,
        time: TimingInformation {
            time: 0.,
            animation_time: Duration::ZERO,
        },
        pointer: DVec2::ZERO,
        render_mode: RenderMode::Normal,
        export_format: ExportFormat::Svg,
        for_export: true,
        for_eyedropper: false,
    }
}

fn budget() -> CpuSvgEvaluationBudget {
    CpuSvgEvaluationBudget {
        max_viewport_pixels: 672 * 384,
        max_output_bytes: 64 * 1024,
    }
}

fn document(document_id_encoding: u64) -> Result<CpuSvgArtboardDocument, CpuSvgDocumentError> {
    CpuSvgArtboardDocument::new(
        document_id_encoding,
        NodeId(7001),
        CpuSvgArtboardParameters::default(),
        configuration(),
        budget(),
    )
}

fn svg(output: &RenderOutput) -> &str {
    let RenderOutputType::Svg { svg, image_data } = &output.data else {
        panic!("the actual CPU export must be SVG");
    };
    assert!(image_data.is_empty());
    svg
}

fn assert_artboard(
    output: &RenderOutput,
    node: NodeId,
    location: DVec2,
    dimensions: DVec2,
    clip: bool,
) {
    assert_eq!(output.metadata.backgrounds.len(), 1);
    let background = output
        .metadata
        .backgrounds
        .first()
        .expect("original Artboard background");
    assert_eq!(background.location, location);
    assert_eq!(background.dimensions, dimensions);
    assert_eq!(
        output
            .metadata
            .click_targets
            .keys()
            .copied()
            .collect::<HashSet<_>>(),
        HashSet::from([node])
    );
    let targets = output
        .metadata
        .click_targets
        .get(&node)
        .expect("original reflected layer target");
    assert_eq!(targets.len(), 1);
    assert_eq!(
        targets
            .first()
            .expect("original rectangular target")
            .bounding_box(),
        Some([DVec2::ZERO, dimensions])
    );
    assert_eq!(
        output.metadata.local_transforms.get(&node),
        Some(&DAffine2::from_translation(location))
    );
    assert_eq!(
        output.metadata.clip_targets,
        if clip {
            HashSet::from([node])
        } else {
            HashSet::new()
        }
    );
    let svg = svg(output);
    assert_eq!(svg.matches("<clipPath id=").count(), usize::from(clip));
    if clip {
        let (_, suffix) = svg
            .split_once("<clipPath id=\"")
            .expect("original clip definition");
        let (identifier, _) = suffix.split_once('"').expect("original clip identifier");
        assert!(identifier.starts_with("artboard-"));
        assert_eq!(
            svg.matches(&format!("clip-path=\"url(#{identifier})\""))
                .count(),
            1
        );
    }
}

#[test]
fn authentic_template_evaluation_is_accepted_with_output_and_metadata_unchanged() -> TestResult {
    let mut document = document(0)?;
    let template: NodeNetwork = serde_json::from_str(include_str!("artboard-template.json"))?;
    assert_eq!(document.network(), &template);
    assert_eq!(document.network().nodes.len(), 1);
    assert_eq!(
        document.network().exports,
        vec![NodeInput::node(NodeId(7001), 0)]
    );
    let node = document
        .network()
        .nodes
        .get(&NodeId(7001))
        .expect("authentic template root");
    assert_eq!(node.inputs.len(), 6);
    assert_eq!(
        node.inputs.get(2).and_then(NodeInput::as_value),
        Some(&TaggedValue::DVec2(DVec2::ZERO))
    );
    assert_eq!(
        node.inputs.get(3).and_then(NodeInput::as_value),
        Some(&TaggedValue::DVec2(DVec2::new(1920., 1080.)))
    );
    assert_eq!(
        node.inputs.get(4).and_then(NodeInput::as_value),
        Some(&TaggedValue::Color(Color::WHITE))
    );
    assert_eq!(
        node.inputs.get(5).and_then(NodeInput::as_value),
        Some(&TaggedValue::Bool(true))
    );
    let before = document.network().clone();
    let checkpoint = document.checkpoint.clone();
    let evaluation = document.submit();
    assert_eq!(evaluation.network(), &before);
    assert!(document.is_pending());
    let completion = block_on(evaluation.evaluate());
    let original = completion
        .result()
        .expect("genuine original CPU SVG evaluation")
        .clone();
    assert_artboard(
        &original,
        NodeId(7001),
        DVec2::ZERO,
        DVec2::new(1920., 1080.),
        true,
    );
    assert!(svg(&original).contains("viewBox=\"0 0 448 256\""));
    assert!(svg(&original).contains("width=\"1920\" height=\"1080\""));
    assert!(
        svg(&original)
            .to_ascii_lowercase()
            .contains("fill=\"#ffffff\"")
    );
    let footprint = original
        .metadata
        .upstream_footprints
        .get(&NodeId(7001))
        .expect("original upstream footprint");
    assert_eq!(footprint.resolution, Footprint::DEFAULT.resolution);
    assert_eq!(
        footprint.transform.to_cols_array().map(f64::to_bits),
        configuration()
            .viewport
            .transform
            .to_cols_array()
            .map(f64::to_bits)
    );
    let accepted = document.accept(completion)?;
    assert_eq!(accepted.output(), &original);
    assert_eq!(accepted.output().metadata, original.metadata);
    assert_eq!(accepted.document_id_encoding(), 0);
    assert_eq!(accepted.node_id(), NodeId(7001));
    assert_eq!(accepted.revision(), 0);
    assert_eq!(accepted.configuration(), configuration());
    let lease = accepted.output_lease();
    assert!(Arc::ptr_eq(
        &lease,
        &document.accepted().expect("accepted output").output
    ));
    assert!(!document.is_pending());
    assert!(document.accepted_is_current());
    assert!(Arc::ptr_eq(&checkpoint, &document.checkpoint));
    assert_eq!(document.network(), &before);
    assert_eq!(template, before);
    Ok(())
}

#[test]
fn unchanged_edit_keeps_pending_checkpoint_and_original_nested_template() -> TestResult {
    let mut document = document(u64::MAX)?;
    let checkpoint = document.checkpoint.clone();
    let network = document.network().clone();
    let evaluation = document.submit();
    let submission = document.pending.clone().expect("active submission");
    assert!(!document.apply(0, document.parameters(), document.configuration())?);
    assert_eq!(document.revision(), 0);
    assert!(Arc::ptr_eq(&checkpoint, &document.checkpoint));
    assert!(Arc::ptr_eq(
        &submission,
        document
            .pending
            .as_ref()
            .expect("unchanged active submission")
    ));
    let completion = block_on(evaluation.evaluate());
    let original = completion
        .result()
        .expect("genuine original CPU SVG evaluation")
        .clone();
    assert_eq!(document.accept(completion)?.output(), &original);
    let accepted = document
        .accepted()
        .expect("first accepted output")
        .output_lease();
    let mut parameters = document.parameters();
    parameters.dimensions = DVec2::new(320., 240.);
    assert!(document.apply(0, parameters, document.configuration())?);
    assert!(!document.accepted_is_current());
    assert!(Arc::ptr_eq(
        &accepted,
        &document
            .accepted()
            .expect("retained last valid output")
            .output_lease()
    ));
    let mut expected = network.clone();
    *expected
        .nodes
        .get_mut(&NodeId(7001))
        .expect("authentic root")
        .inputs
        .get_mut(3)
        .expect("original dimensions port") =
        NodeInput::value(TaggedValue::DVec2(parameters.dimensions), false);
    assert_eq!(document.network(), &expected);
    assert_eq!(network, *checkpoint.network);
    let completion = block_on(document.submit().evaluate());
    assert_artboard(
        completion
            .result()
            .expect("genuine original CPU SVG evaluation"),
        NodeId(7001),
        DVec2::ZERO,
        parameters.dimensions,
        true,
    );
    document.accept(completion)?;
    assert_eq!(document.revision(), 1);
    assert_eq!(document.document_id_encoding(), u64::MAX);
    assert!(document.accepted_is_current());
    Ok(())
}

#[test]
fn context_successors_reject_old_completions_and_preserve_last_valid_metadata() -> TestResult {
    let mut document = document(41)?;
    let completion = block_on(document.submit().evaluate());
    let old_output = document.accept(completion)?.output_lease();
    let before = document.network().clone();
    let old_request = document.submit();
    let mut next = document.configuration();
    next.viewport.transform =
        DAffine2::from_translation(DVec2::new(23., 31.)) * DAffine2::from_scale(DVec2::splat(0.5));
    next.scale = 2.;
    assert!(document.apply(0, document.parameters(), next)?);
    assert_eq!(document.network(), &before);
    assert!(!document.accepted_is_current());
    let current_request = document.submit();
    let current_submission = document.pending.clone().expect("successor submission");
    let stale = block_on(old_request.evaluate());
    assert!(matches!(
        document.accept(stale),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    assert!(Arc::ptr_eq(
        &current_submission,
        document
            .pending
            .as_ref()
            .expect("successor remains pending")
    ));
    assert_eq!(
        document.accepted().expect("last valid").output(),
        old_output.as_ref()
    );
    let completion = block_on(current_request.evaluate());
    let expected = completion
        .result()
        .expect("genuine original CPU SVG evaluation")
        .clone();
    assert_eq!(document.accept(completion)?.output(), &expected);
    assert_ne!(expected.data, old_output.data);
    assert_eq!(
        document
            .accepted()
            .expect("current accepted")
            .configuration(),
        next
    );
    assert_eq!(
        expected
            .metadata
            .upstream_footprints
            .get(&NodeId(7001))
            .expect("current footprint")
            .transform,
        next.viewport.transform
    );
    assert!(document.accepted_is_current());
    assert_eq!(document.network(), &before);
    Ok(())
}

#[test]
fn identical_document_and_node_values_cannot_forge_owner_or_submission_identity() -> TestResult {
    let mut first = document(71)?;
    let mut second = document(71)?;
    assert_eq!(first.network(), second.network());
    assert_eq!(first.configuration(), second.configuration());
    let foreign = block_on(first.submit().evaluate());
    let own = second.submit();
    let submission = second.pending.clone().expect("second pending");
    assert!(matches!(
        second.accept(foreign),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    assert!(second.accepted().is_none());
    assert!(Arc::ptr_eq(
        &submission,
        second.pending.as_ref().expect("own request preserved")
    ));
    let mut forged = block_on(own.evaluate());
    let checkpoint = &second.checkpoint;
    forged.checkpoint = Arc::new(SvgCheckpoint {
        document_id_encoding: checkpoint.document_id_encoding,
        node_id: checkpoint.node_id,
        revision: checkpoint.revision,
        network: checkpoint.network.clone(),
        encoded_png: checkpoint.encoded_png.clone(),
        parameters: checkpoint.parameters,
        configuration: checkpoint.configuration,
        budget: checkpoint.budget,
    });
    assert!(matches!(
        second.accept(forged),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    assert!(second.is_pending());
    let superseded = second.submit();
    let current = second.submit();
    assert!(matches!(
        second.accept(block_on(superseded.evaluate())),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    second.accept(block_on(current.evaluate()))?;
    assert!(second.accepted_is_current());
    assert!(first.is_pending());
    assert!(first.accepted().is_none());
    Ok(())
}

#[test]
fn cancellation_rejects_computed_completion_without_erasing_accepted_output() -> TestResult {
    let mut document = document(83)?;
    let completion = block_on(document.submit().evaluate());
    let last_valid = document.accept(completion)?.output_lease();
    let completion = block_on(document.submit().evaluate());
    assert!(completion.result().is_ok());
    assert!(document.cancel());
    assert!(!document.cancel());
    assert!(matches!(
        document.accept(completion),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    assert!(!document.is_pending());
    assert!(Arc::ptr_eq(
        &last_valid,
        &document
            .accepted()
            .expect("last valid after cancel")
            .output_lease()
    ));
    assert!(document.accepted_is_current());
    let dropped_ticket = document.submit();
    drop(dropped_ticket);
    assert!(document.is_pending());
    assert!(document.cancel());
    let completion = block_on(document.submit().evaluate());
    document.accept(completion)?;
    assert!(document.accepted_is_current());
    Ok(())
}

#[test]
fn signed_zero_and_negative_dimensions_follow_original_artboard_normalization() -> TestResult {
    for (dimensions, clip, expected_location, expected_dimensions) in [
        (DVec2::new(-0., 0.), false, DVec2::new(13., 17.), DVec2::ONE),
        (
            DVec2::new(-7., -11.),
            true,
            DVec2::new(6., 6.),
            DVec2::new(7., 11.),
        ),
    ] {
        let parameters = CpuSvgArtboardParameters {
            location: DVec2::new(13., 17.),
            dimensions,
            background: Color::WHITE,
            clip,
        };
        let mut document =
            CpuSvgArtboardDocument::new(97, NodeId(8113), parameters, configuration(), budget())?;
        let before = document.network().clone();
        let Some(TaggedValue::DVec2(retained)) = document
            .network()
            .nodes
            .get(&NodeId(8113))
            .and_then(|node| node.inputs.get(3))
            .and_then(NodeInput::as_value)
        else {
            panic!("original dimensions input must retain its type");
        };
        assert_eq!(
            retained.to_array().map(f64::to_bits),
            dimensions.to_array().map(f64::to_bits)
        );
        let completion = block_on(document.submit().evaluate());
        assert_artboard(
            completion
                .result()
                .expect("genuine original CPU SVG evaluation"),
            NodeId(8113),
            expected_location,
            expected_dimensions,
            clip,
        );
        let original = completion
            .result()
            .expect("genuine original CPU SVG evaluation")
            .clone();
        assert_eq!(document.accept(completion)?.output(), &original);
        assert_eq!(document.parameters(), parameters);
        assert_eq!(document.network(), &before);
    }
    Ok(())
}

#[test]
fn actual_output_cap_failure_preserves_last_valid_and_recovers_without_mutating_input() -> TestResult
{
    let parameters = CpuSvgArtboardParameters {
        clip: false,
        ..CpuSvgArtboardParameters::default()
    };
    let mut probe =
        CpuSvgArtboardDocument::new(101, NodeId(7001), parameters, configuration(), budget())?;
    let completion = block_on(probe.submit().evaluate());
    let unmasked_bytes = svg(completion
        .result()
        .expect("genuine original CPU SVG evaluation"))
    .len();
    // The original clip definition plus reference exceeds this slack; no UUID is reseeded.
    let tight = CpuSvgEvaluationBudget {
        max_output_bytes: unmasked_bytes + 64,
        ..budget()
    };
    let mut document =
        CpuSvgArtboardDocument::new(101, NodeId(7001), parameters, configuration(), tight)?;
    let completion = block_on(document.submit().evaluate());
    let last_valid = document.accept(completion)?.output_lease();
    let masked = CpuSvgArtboardParameters {
        clip: true,
        ..parameters
    };
    document.apply(0, masked, configuration())?;
    let before = document.network().clone();
    let completion = block_on(document.submit().evaluate());
    assert!(matches!(
        completion.result(),
        Err(CpuSvgEvaluationError::OutputBudgetExceeded)
    ));
    assert!(matches!(
        document.accept(completion),
        Err(CpuSvgAcceptanceError::Evaluation(
            CpuSvgEvaluationError::OutputBudgetExceeded
        ))
    ));
    assert!(!document.is_pending());
    assert!(!document.accepted_is_current());
    assert!(Arc::ptr_eq(
        &last_valid,
        &document
            .accepted()
            .expect("last valid after cap failure")
            .output_lease()
    ));
    assert_eq!(document.network(), &before);
    document.apply(1, parameters, configuration())?;
    let completion = block_on(document.submit().evaluate());
    document.accept(completion)?;
    assert!(document.accepted_is_current());
    let mut zero = CpuSvgArtboardDocument::new(
        102,
        NodeId(7001),
        parameters,
        configuration(),
        CpuSvgEvaluationBudget {
            max_output_bytes: 0,
            ..budget()
        },
    )?;
    let completion = block_on(zero.submit().evaluate());
    assert!(matches!(
        zero.accept(completion),
        Err(CpuSvgAcceptanceError::Evaluation(
            CpuSvgEvaluationError::OutputBudgetExceeded
        ))
    ));
    assert!(zero.accepted().is_none());
    assert!(!zero.is_pending());
    Ok(())
}

#[test]
fn invalid_edits_and_revision_overflow_are_transactional_before_original_evaluation() -> TestResult
{
    let mut document = document(113)?;
    let completion = block_on(document.submit().evaluate());
    let accepted = document.accept(completion)?.output_lease();
    let evaluation = document.submit();
    let checkpoint = document.checkpoint.clone();
    let submission = document
        .pending
        .clone()
        .expect("pending before invalid edits");
    let parameters = document.parameters();
    let mut cases = Vec::new();
    for scale in [0., -0., -1., f64::NAN, f64::INFINITY, f64::from_bits(1)] {
        cases.push((
            parameters,
            RenderConfig {
                scale,
                ..configuration()
            },
            CpuSvgDocumentError::InvalidConfiguration,
        ));
    }
    for render_mode in [
        RenderMode::Outline,
        RenderMode::PixelPreview,
        RenderMode::SvgPreview,
    ] {
        cases.push((
            parameters,
            RenderConfig {
                render_mode,
                ..configuration()
            },
            CpuSvgDocumentError::InvalidConfiguration,
        ));
    }
    cases.extend([
        (
            parameters,
            RenderConfig {
                for_export: false,
                ..configuration()
            },
            CpuSvgDocumentError::InvalidConfiguration,
        ),
        (
            parameters,
            RenderConfig {
                for_eyedropper: true,
                ..configuration()
            },
            CpuSvgDocumentError::InvalidConfiguration,
        ),
        (
            parameters,
            RenderConfig {
                export_format: ExportFormat::Raster,
                ..configuration()
            },
            CpuSvgDocumentError::InvalidConfiguration,
        ),
        (
            parameters,
            RenderConfig {
                pointer: DVec2::splat(f64::NAN),
                ..configuration()
            },
            CpuSvgDocumentError::InvalidConfiguration,
        ),
        (
            parameters,
            RenderConfig {
                viewport: Footprint {
                    resolution: UVec2::ZERO,
                    ..configuration().viewport
                },
                ..configuration()
            },
            CpuSvgDocumentError::InvalidConfiguration,
        ),
        (
            parameters,
            RenderConfig {
                viewport: Footprint {
                    resolution: UVec2::new(673, 384),
                    ..configuration().viewport
                },
                ..configuration()
            },
            CpuSvgDocumentError::ViewportBudgetExceeded,
        ),
        (
            parameters,
            RenderConfig {
                viewport: Footprint {
                    transform: DAffine2::from_scale(DVec2::splat(f64::MAX)),
                    ..configuration().viewport
                },
                scale: 2.,
                ..configuration()
            },
            CpuSvgDocumentError::InvalidConfiguration,
        ),
        (
            CpuSvgArtboardParameters {
                location: DVec2::splat(f64::MAX),
                dimensions: DVec2::splat(f64::MAX),
                ..parameters
            },
            configuration(),
            CpuSvgDocumentError::InvalidGeometry,
        ),
        (
            CpuSvgArtboardParameters {
                dimensions: DVec2::splat(f64::NAN),
                ..parameters
            },
            configuration(),
            CpuSvgDocumentError::InvalidGeometry,
        ),
        (
            CpuSvgArtboardParameters {
                background: Color::from_rgbaf32_unchecked(1., 0., 0., f32::NAN),
                ..parameters
            },
            configuration(),
            CpuSvgDocumentError::InvalidBackground,
        ),
        (
            CpuSvgArtboardParameters {
                background: Color::from_rgbaf32_unchecked(1., 0., 0., -f32::MAX),
                ..parameters
            },
            configuration(),
            CpuSvgDocumentError::InvalidBackground,
        ),
    ]);
    for (parameters, configuration, error) in cases {
        assert_eq!(document.apply(0, parameters, configuration), Err(error));
        assert!(Arc::ptr_eq(&checkpoint, &document.checkpoint));
        assert!(Arc::ptr_eq(
            &submission,
            document
                .pending
                .as_ref()
                .expect("invalid edit keeps pending")
        ));
        assert_eq!(document.network(), checkpoint.network.as_ref());
        assert!(Arc::ptr_eq(
            &accepted,
            &document
                .accepted()
                .expect("invalid edit keeps accepted")
                .output_lease()
        ));
        assert!(document.accepted_is_current());
    }
    assert_eq!(
        document.apply(1, parameters, configuration()),
        Err(CpuSvgDocumentError::StaleRevision)
    );
    document.accept(block_on(evaluation.evaluate()))?;
    let checkpoint = &document.checkpoint;
    document.checkpoint = Arc::new(SvgCheckpoint {
        document_id_encoding: checkpoint.document_id_encoding,
        node_id: checkpoint.node_id,
        revision: u64::MAX,
        network: checkpoint.network.clone(),
        encoded_png: checkpoint.encoded_png.clone(),
        parameters: checkpoint.parameters,
        configuration: checkpoint.configuration,
        budget: checkpoint.budget,
    });
    let exhausted = document.checkpoint.clone();
    let accepted = document
        .accepted()
        .expect("last output before exhaustion")
        .output_lease();
    let mut changed = document.parameters();
    changed.clip = !changed.clip;
    let evaluation = document.submit();
    let submission = document.pending.clone().expect("exhausted owner pending");
    assert!(!document.apply(u64::MAX, document.parameters(), document.configuration())?);
    assert_eq!(
        document.apply(u64::MAX, changed, configuration()),
        Err(CpuSvgDocumentError::RevisionExhausted)
    );
    assert!(Arc::ptr_eq(&exhausted, &document.checkpoint));
    assert!(Arc::ptr_eq(
        &submission,
        document
            .pending
            .as_ref()
            .expect("revision failure keeps pending")
    ));
    assert_eq!(document.network(), exhausted.network.as_ref());
    assert_eq!(document.revision(), u64::MAX);
    assert!(Arc::ptr_eq(
        &accepted,
        &document
            .accepted()
            .expect("revision failure keeps last output")
            .output_lease()
    ));
    drop(evaluation);
    assert!(document.cancel());
    Ok(())
}

#[test]
fn signed_zero_edits_replace_exact_checkpoint_and_bitwise_repeats_preserve_pending() -> TestResult {
    let mut document = document(127)?;
    let before = document.network().clone();
    let before_checkpoint = document.checkpoint.clone();
    let old_request = document.submit();
    let mut parameters = document.parameters();
    assert_eq!(parameters.location.x.to_bits(), 0_f64.to_bits());
    parameters.location.x = -0.;
    assert!(document.apply(0, parameters, document.configuration())?);
    assert_eq!(document.revision(), 1);
    assert!(!document.is_pending());
    assert!(!Arc::ptr_eq(&before_checkpoint, &document.checkpoint));
    assert!(!Arc::ptr_eq(
        &before_checkpoint.network,
        &document.checkpoint.network
    ));
    assert_eq!(
        document.parameters().location.x.to_bits(),
        (-0_f64).to_bits()
    );
    let Some(TaggedValue::DVec2(location)) = document
        .network()
        .nodes
        .get(&NodeId(7001))
        .and_then(|node| node.inputs.get(2))
        .and_then(NodeInput::as_value)
    else {
        panic!("original location port must retain DVec2");
    };
    assert_eq!(location.x.to_bits(), (-0_f64).to_bits());
    let Some(TaggedValue::DVec2(old_location)) = before
        .nodes
        .get(&NodeId(7001))
        .and_then(|node| node.inputs.get(2))
        .and_then(NodeInput::as_value)
    else {
        panic!("detached original location port must retain DVec2");
    };
    assert_eq!(old_location.x.to_bits(), 0_f64.to_bits());
    let current_request = document.submit();
    let current_submission = document
        .pending
        .clone()
        .expect("signed-zero successor pending");
    let current_checkpoint = document.checkpoint.clone();
    assert!(!document.apply(1, parameters, document.configuration())?);
    assert!(Arc::ptr_eq(&current_checkpoint, &document.checkpoint));
    assert!(Arc::ptr_eq(
        &current_submission,
        document
            .pending
            .as_ref()
            .expect("bitwise repeat stays pending")
    ));
    let old_completion = block_on(old_request.evaluate());
    assert!(old_completion.result().is_ok());
    assert!(matches!(
        document.accept(old_completion),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    assert!(document.accepted().is_none());
    assert!(Arc::ptr_eq(
        &current_submission,
        document
            .pending
            .as_ref()
            .expect("stale completion keeps signed-zero successor")
    ));
    let current_completion = block_on(current_request.evaluate());
    let actual_output = current_completion
        .result()
        .expect("genuine signed-zero Artboard evaluation")
        .clone();
    assert_eq!(
        document.accept(current_completion)?.output(),
        &actual_output
    );
    let last_valid = document
        .accepted()
        .expect("accepted signed-zero location")
        .output_lease();

    for component in 0..4 {
        let old_checkpoint = document.checkpoint.clone();
        let old_request = document.submit();
        let mut next = document.configuration();
        match component {
            0 => {
                assert_eq!(
                    next.viewport.transform.matrix2.x_axis.y.to_bits(),
                    0_f64.to_bits()
                );
                next.viewport.transform.matrix2.x_axis.y = -0.;
            }
            1 => {
                assert_eq!(next.time.time.to_bits(), 0_f64.to_bits());
                next.time.time = -0.;
            }
            2 => {
                assert_eq!(next.pointer.x.to_bits(), 0_f64.to_bits());
                next.pointer.x = -0.;
            }
            3 => {
                assert_eq!(next.pointer.y.to_bits(), 0_f64.to_bits());
                next.pointer.y = -0.;
            }
            _ => unreachable!("four explicit finite context components"),
        }
        let revision = document.revision();
        assert!(document.apply(revision, parameters, next)?);
        assert_eq!(document.revision(), revision + 1);
        assert!(!Arc::ptr_eq(&old_checkpoint, &document.checkpoint));
        assert!(Arc::ptr_eq(
            &old_checkpoint.network,
            &document.checkpoint.network
        ));
        assert_eq!(
            document
                .configuration()
                .viewport
                .transform
                .to_cols_array()
                .map(f64::to_bits),
            next.viewport.transform.to_cols_array().map(f64::to_bits)
        );
        assert_eq!(
            document.configuration().time.time.to_bits(),
            next.time.time.to_bits()
        );
        assert_eq!(
            document
                .configuration()
                .pointer
                .to_array()
                .map(f64::to_bits),
            next.pointer.to_array().map(f64::to_bits)
        );
        assert!(!document.accepted_is_current());
        assert!(Arc::ptr_eq(
            &last_valid,
            &document
                .accepted()
                .expect("context edit retains last valid")
                .output_lease()
        ));
        let current_request = document.submit();
        let pending = document.pending.clone().expect("context successor pending");
        let checkpoint = document.checkpoint.clone();
        assert!(!document.apply(revision + 1, parameters, next)?);
        assert!(Arc::ptr_eq(&checkpoint, &document.checkpoint));
        assert!(Arc::ptr_eq(
            &pending,
            document
                .pending
                .as_ref()
                .expect("exact context repeat retains pending")
        ));
        let old_completion = block_on(old_request.evaluate());
        assert!(old_completion.result().is_ok());
        assert!(matches!(
            document.accept(old_completion),
            Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
        ));
        assert!(Arc::ptr_eq(
            &pending,
            document
                .pending
                .as_ref()
                .expect("old context completion keeps new request")
        ));
        if component == 3 {
            document.accept(block_on(current_request.evaluate()))?;
            assert!(document.accepted_is_current());
            assert_eq!(
                document
                    .accepted()
                    .expect("current exact-context output")
                    .revision(),
                5
            );
        } else {
            drop(current_request);
        }
    }
    assert_eq!(*before_checkpoint.network, before);
    Ok(())
}

#[test]
fn genuine_threaded_evaluation_transfers_completion_and_rejects_stale_successor() -> TestResult {
    let mut document = document(41)?;
    let foreground_thread = std::thread::current().id();
    let before = document.network().clone();
    let original_checkpoint = document.checkpoint.clone();
    let evaluation = document.submit();
    assert_eq!(evaluation.network(), &before);
    let worker = std::thread::spawn(move || {
        let worker_thread = std::thread::current().id();
        let completion = block_on(evaluation.evaluate());
        (worker_thread, completion)
    });
    let (worker_thread, completion) = worker.join().expect("original evaluation worker exits");
    assert_ne!(worker_thread, foreground_thread);
    assert_eq!(std::thread::current().id(), foreground_thread);
    let original_output = completion
        .result()
        .expect("genuine threaded original SVG")
        .clone();
    assert_artboard(
        &original_output,
        NodeId(7001),
        DVec2::ZERO,
        DVec2::new(1920., 1080.),
        true,
    );
    assert!(svg(&original_output).contains("viewBox=\"0 0 448 256\""));
    let accepted = document.accept(completion)?;
    assert_eq!(accepted.output(), &original_output);
    assert_eq!(accepted.output().metadata, original_output.metadata);
    assert_eq!(accepted.document_id_encoding(), 41);
    assert_eq!(accepted.node_id(), NodeId(7001));
    assert_eq!(accepted.revision(), 0);
    assert_eq!(accepted.configuration(), configuration());
    let last_valid = accepted.output_lease();
    assert!(document.accepted_is_current());
    assert!(!document.is_pending());
    assert_eq!(document.network(), &before);
    assert!(Arc::ptr_eq(&original_checkpoint, &document.checkpoint));

    let old_request = document.submit();
    let old_worker = std::thread::spawn(move || block_on(old_request.evaluate()));
    let parameters = CpuSvgArtboardParameters {
        location: DVec2::new(17., 23.),
        dimensions: DVec2::new(640., 480.),
        background: Color::BLACK,
        clip: false,
    };
    assert!(document.apply(0, parameters, configuration())?);
    let successor_network = document.network().clone();
    let current_request = document.submit();
    let current_submission = document.pending.clone().expect("genuine successor pending");
    let old_completion = old_worker.join().expect("stale original worker exits");
    assert_artboard(
        old_completion.result().expect("genuine stale threaded SVG"),
        NodeId(7001),
        DVec2::ZERO,
        DVec2::new(1920., 1080.),
        true,
    );
    assert!(matches!(
        document.accept(old_completion),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    assert!(!document.accepted_is_current());
    assert!(Arc::ptr_eq(
        &last_valid,
        &document
            .accepted()
            .expect("last valid output retained")
            .output_lease()
    ));
    assert!(Arc::ptr_eq(
        &current_submission,
        document
            .pending
            .as_ref()
            .expect("successor survives stale completion")
    ));
    assert_eq!(document.network(), &successor_network);
    assert_eq!(*original_checkpoint.network, before);

    let current_worker = std::thread::spawn(move || block_on(current_request.evaluate()));
    let current_completion = current_worker
        .join()
        .expect("current original worker exits");
    let current_output = current_completion
        .result()
        .expect("genuine current threaded SVG")
        .clone();
    assert_artboard(
        &current_output,
        NodeId(7001),
        DVec2::new(17., 23.),
        DVec2::new(640., 480.),
        false,
    );
    assert_ne!(current_output.metadata, original_output.metadata);
    assert_ne!(svg(&current_output), svg(&original_output));
    assert_eq!(std::thread::current().id(), foreground_thread);
    let accepted = document.accept(current_completion)?;
    assert_eq!(accepted.output(), &current_output);
    assert_eq!(accepted.output().metadata, current_output.metadata);
    assert_eq!(accepted.revision(), 1);
    assert_eq!(accepted.document_id_encoding(), 41);
    assert_eq!(accepted.node_id(), NodeId(7001));
    assert_eq!(accepted.configuration(), configuration());
    assert!(document.accepted_is_current());
    assert!(!document.is_pending());
    assert_eq!(document.network(), &successor_network);
    assert_eq!(*original_checkpoint.network, before);
    assert_eq!(*last_valid, original_output);
    Ok(())
}

#[test]
fn genuine_owned_output_crosses_thread_and_tracks_exact_document_acceptance() -> TestResult {
    let mut owner = document(51)?;
    let before = owner.network().clone();
    let original_checkpoint = owner.checkpoint.clone();
    let request = owner.submit();
    let completion = block_on(request.evaluate());
    let original = completion
        .result()
        .expect("genuine accepted source output")
        .clone();
    let lease: AcceptedCpuSvgOutput = owner.accept(completion)?.clone();
    assert!(owner.owns_current_output(&lease));
    assert_eq!(lease.output(), &original);
    assert_eq!(lease.output().metadata, original.metadata);
    let original_svg = svg(lease.output()).to_owned();
    let foreground_thread = std::thread::current().id();
    let outbound = lease.clone();
    let worker = std::thread::spawn(move || {
        let worker_thread = std::thread::current().id();
        let original_output = outbound.output().clone();
        (outbound, worker_thread, original_output)
    });
    let (returned, worker_thread, worker_output) = worker.join().expect("owned lease worker exits");
    assert_ne!(worker_thread, foreground_thread);
    assert_eq!(std::thread::current().id(), foreground_thread);
    assert!(owner.owns_current_output(&returned));
    assert!(Arc::ptr_eq(&lease.checkpoint, &returned.checkpoint));
    assert!(Arc::ptr_eq(&lease.output_lease(), &returned.output_lease()));
    assert_eq!(returned.output(), &worker_output);
    assert_eq!(returned.output().metadata, worker_output.metadata);
    assert_eq!(worker_output, original);
    assert_eq!(svg(returned.output()), original_svg);
    assert_eq!(returned.configuration(), configuration());
    assert_eq!(returned.document_id_encoding(), 51);
    assert_eq!(returned.node_id(), NodeId(7001));
    assert_eq!(returned.revision(), 0);
    assert_eq!(owner.network(), &before);

    let mut foreign = document(51)?;
    let foreign_request = foreign.submit();
    let foreign_completion = block_on(foreign_request.evaluate());
    let foreign_lease: AcceptedCpuSvgOutput = foreign.accept(foreign_completion)?.clone();
    assert_eq!(
        foreign_lease.document_id_encoding(),
        returned.document_id_encoding()
    );
    assert_eq!(foreign_lease.node_id(), returned.node_id());
    assert_eq!(foreign_lease.revision(), returned.revision());
    assert_eq!(foreign_lease.configuration(), returned.configuration());
    assert_eq!(foreign.network(), &before);
    assert!(foreign.owns_current_output(&foreign_lease));
    assert!(!foreign.owns_current_output(&returned));
    assert!(!owner.owns_current_output(&foreign_lease));
    assert!(owner.owns_current_output(&returned));

    let cancelled_retry = owner.submit();
    assert!(owner.is_pending());
    assert!(owner.owns_current_output(&returned));
    assert!(owner.cancel());
    assert!(!owner.is_pending());
    assert!(owner.owns_current_output(&returned));
    let cancelled_completion = block_on(cancelled_retry.evaluate());
    assert!(cancelled_completion.result().is_ok());
    assert!(matches!(
        owner.accept(cancelled_completion),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    assert!(owner.owns_current_output(&returned));

    let retry = owner.submit();
    assert!(owner.owns_current_output(&returned));
    let replacement_completion = block_on(retry.evaluate());
    let replacement_output = replacement_completion
        .result()
        .expect("genuine same-checkpoint replacement")
        .clone();
    let replacement: AcceptedCpuSvgOutput = owner.accept(replacement_completion)?.clone();
    assert!(Arc::ptr_eq(&returned.checkpoint, &replacement.checkpoint));
    assert!(!Arc::ptr_eq(
        &returned.output_lease(),
        &replacement.output_lease()
    ));
    assert!(!owner.owns_current_output(&returned));
    assert!(owner.owns_current_output(&replacement));
    assert_eq!(replacement.output(), &replacement_output);
    assert_eq!(replacement.output().metadata, replacement_output.metadata);
    assert_eq!(replacement.output().metadata, original.metadata);
    assert_eq!(replacement.revision(), 0);
    assert_eq!(owner.network(), &before);
    assert_eq!(returned.output(), &original);
    assert_eq!(svg(returned.output()), original_svg);

    let mut next_configuration = owner.configuration();
    next_configuration.pointer = DVec2::new(4., 5.);
    assert!(owner.apply(0, owner.parameters(), next_configuration)?);
    assert!(Arc::ptr_eq(
        &original_checkpoint.network,
        &owner.checkpoint.network
    ));
    assert!(!owner.owns_current_output(&replacement));
    assert!(!owner.owns_current_output(&returned));
    assert!(!owner.accepted_is_current());
    let context_request = owner.submit();
    let context_completion = block_on(context_request.evaluate());
    let context_output = context_completion
        .result()
        .expect("genuine context successor")
        .clone();
    let context_lease: AcceptedCpuSvgOutput = owner.accept(context_completion)?.clone();
    assert!(owner.owns_current_output(&context_lease));
    assert_eq!(context_lease.revision(), 1);
    assert_eq!(context_lease.configuration(), next_configuration);
    assert_eq!(context_lease.output(), &context_output);
    assert_eq!(context_lease.output().metadata, context_output.metadata);
    assert_eq!(owner.network(), &before);

    let next_parameters = CpuSvgArtboardParameters {
        location: DVec2::new(13., 19.),
        dimensions: DVec2::new(640., 480.),
        background: Color::BLACK,
        clip: false,
    };
    assert!(owner.apply(1, next_parameters, next_configuration)?);
    assert!(!Arc::ptr_eq(
        &original_checkpoint.network,
        &owner.checkpoint.network
    ));
    assert!(!owner.owns_current_output(&context_lease));
    assert!(!owner.owns_current_output(&replacement));
    assert!(!owner.owns_current_output(&returned));
    assert!(!owner.accepted_is_current());
    let source_successor = owner.network().clone();
    let final_request = owner.submit();
    let final_completion = block_on(final_request.evaluate());
    let final_output = final_completion
        .result()
        .expect("genuine source successor")
        .clone();
    let recovered: AcceptedCpuSvgOutput = owner.accept(final_completion)?.clone();
    assert!(owner.owns_current_output(&recovered));
    assert!(!owner.owns_current_output(&context_lease));
    assert!(!owner.owns_current_output(&replacement));
    assert!(!owner.owns_current_output(&returned));
    assert!(!foreign.owns_current_output(&recovered));
    assert_eq!(recovered.revision(), 2);
    assert_eq!(recovered.configuration(), next_configuration);
    assert_eq!(recovered.output(), &final_output);
    assert_eq!(recovered.output().metadata, final_output.metadata);
    assert_artboard(
        &final_output,
        NodeId(7001),
        next_parameters.location,
        next_parameters.dimensions,
        false,
    );
    assert_eq!(owner.network(), &source_successor);
    assert_eq!(*original_checkpoint.network, before);
    assert_eq!(foreign.network(), &before);
    assert!(foreign.owns_current_output(&foreign_lease));
    assert_eq!(returned.output(), &original);
    assert_eq!(returned.output().metadata, original.metadata);
    assert_eq!(svg(returned.output()), original_svg);
    assert_eq!(replacement.output(), &replacement_output);
    assert_eq!(context_lease.output(), &context_output);
    assert!(owner.accepted_is_current());
    assert!(!owner.is_pending());
    Ok(())
}

fn encoded_png_fixture(
    width: u32,
    height: u32,
    samples: &[u8],
) -> Result<Arc<[u8]>, Box<dyn Error>> {
    let mut bytes = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut bytes, width, height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_chunk(
            png::chunk::tEXt,
            b"workflow\0original encoded Artboard fixture",
        )?;
        writer.write_image_data(samples)?;
        writer.finish()?;
    }
    Ok(bytes.into())
}

fn encoded_input(
    bytes: Arc<[u8]>,
    version: &str,
    placement: DAffine2,
) -> Result<CpuSvgEncodedPngInput, Box<dyn Error>> {
    use crate::resources::{EncodedPngBudget, ImageResourceIdentity, prepare_encoded_png_source};
    use sha2::{Digest, Sha256};
    let source = Arc::new(prepare_encoded_png_source(
        ImageResourceIdentity {
            asset_id: "test/input/authentic-artboard.png".into(),
            version_id: version.into(),
            content_sha256: Sha256::digest(bytes.as_ref()).into(),
        },
        bytes,
        EncodedPngBudget {
            max_encoded_bytes: 256 * 1024,
            max_pixel_bytes: 1024 * 1024,
            max_metadata_bytes: 4096,
        },
    )?);
    Ok(CpuSvgEncodedPngInput::new(source, placement)?)
}

fn embedded_png(output: &RenderOutput) -> Result<Vec<u8>, Box<dyn Error>> {
    use base64::Engine as _;
    let svg = svg(output);
    assert_eq!(svg.matches("data:image/png;base64,").count(), 1);
    let (_, embedded) = svg
        .split_once("data:image/png;base64,")
        .expect("actual original SVG embedded PNG");
    let (encoded, _) = embedded
        .split_once('"')
        .expect("original image href terminator");
    Ok(base64::engine::general_purpose::STANDARD.decode(encoded)?)
}

const ENCODED_ARTBOARD_RGBA: [u8; 16] = [
    231, 17, 83, 0, 19, 223, 71, 1, 11, 53, 241, 127, 201, 31, 97, 255,
];

#[test]
fn encoded_network_preserves_original_pixels_unit_geometry_and_complete_metadata() -> TestResult {
    let bytes = encoded_png_fixture(2, 2, &ENCODED_ARTBOARD_RGBA)?;
    let before = bytes.to_vec();
    let oracle = graphite_raster_types::Image::<Color>::from_encoded(&bytes)
        .expect("actual original PNG decoder oracle outside production graph");
    assert_eq!((oracle.width, oracle.height, oracle.data.len()), (2, 2, 4));
    let transparent = oracle
        .data
        .first()
        .expect("original transparent colored pixel");
    assert_eq!(transparent.a().to_bits(), 0_f32.to_bits());
    assert!(transparent.r() > 0. && transparent.g() > 0. && transparent.b() > 0.);
    let oracle_before = oracle
        .data
        .iter()
        .map(|color| {
            [
                color.r().to_bits(),
                color.g().to_bits(),
                color.b().to_bits(),
                color.a().to_bits(),
            ]
        })
        .collect::<Vec<_>>();
    let (flattened, width, height) = oracle.to_flat_u8();
    assert_eq!((width, height), (2, 2));
    assert_eq!(flattened.get(..4), Some([0, 0, 0, 0].as_slice()));
    assert_eq!(
        flattened.get(4..),
        Some(ENCODED_ARTBOARD_RGBA[4..].as_ref())
    );
    let expected_png = oracle.to_png();
    for placement in [
        DAffine2::from_cols_array(&[2., 0., 0., 3., 4., 5.]),
        DAffine2::from_cols_array(&[0., 0., 0., 0., -0., 2.]),
        DAffine2::from_cols_array(&[1., 0., 2., 0., 3., 4.]),
        DAffine2::from_cols_array(&[-2., 0., 0., 3., 7., 11.]),
        DAffine2::from_cols_array(&[1., 0.5, -0.25, 1., 3., 2.]),
    ] {
        let input = encoded_input(bytes.clone(), "supplied-test-label", placement)?;
        let source = input.source().clone();
        assert!(Arc::ptr_eq(source.encoded_bytes(), &bytes));
        let mut document = CpuSvgArtboardDocument::new_with_encoded_png(
            151,
            NodeId(7001),
            CpuSvgArtboardParameters::default(),
            configuration(),
            budget(),
            input,
        )?;
        let network = document.network().clone();
        let completion = block_on(document.submit().evaluate());
        let original = completion
            .result()
            .expect("genuine original encoded Artboard SVG")
            .clone();
        assert_artboard(
            &original,
            NodeId(7001),
            DVec2::ZERO,
            DVec2::new(1920., 1080.),
            true,
        );
        assert_eq!(embedded_png(&original)?, expected_png);
        assert!(svg(&original).contains("<image width=\"1\" height=\"1\""));
        assert!(svg(&original).contains("preserveAspectRatio=\"none\""));
        let transform = graphite_rendering::format_transform_matrix(placement);
        assert!(svg(&original).contains(&format!("transform=\"{transform}\"")));
        let accepted = document.accept(completion)?.clone();
        assert_eq!(accepted.output(), &original);
        assert_eq!(accepted.output().metadata, original.metadata);
        assert!(document.owns_current_output(&accepted));
        let retained = document
            .encoded_png_input()
            .expect("encoded source retained");
        assert!(Arc::ptr_eq(retained.source(), &source));
        assert_eq!(
            retained.placement().to_cols_array().map(f64::to_bits),
            placement.to_cols_array().map(f64::to_bits)
        );
        assert_eq!(document.network(), &network);
        assert_eq!(source.encoded_bytes().as_ref(), before.as_slice());
    }
    assert_eq!(
        oracle
            .data
            .iter()
            .map(|color| [
                color.r().to_bits(),
                color.g().to_bits(),
                color.b().to_bits(),
                color.a().to_bits(),
            ])
            .collect::<Vec<_>>(),
        oracle_before
    );
    assert_eq!(bytes.as_ref(), before.as_slice());
    Ok(())
}

#[test]
fn encoded_parameter_and_context_edits_retain_source_and_exact_input_noops() -> TestResult {
    let bytes = encoded_png_fixture(2, 2, &ENCODED_ARTBOARD_RGBA)?;
    let input = encoded_input(bytes.clone(), "original-label", DAffine2::IDENTITY)?;
    let source = input.source().clone();
    let mut document = CpuSvgArtboardDocument::new_with_encoded_png(
        157,
        NodeId(7001),
        CpuSvgArtboardParameters::default(),
        configuration(),
        budget(),
        input.clone(),
    )?;
    let completion = block_on(document.submit().evaluate());
    let old = document.accept(completion)?.clone();
    let old_output = old.output().clone();
    let old_network = document.network().clone();
    let old_checkpoint = document.checkpoint.clone();
    let stale = document.submit();
    let mut parameters = document.parameters();
    parameters.location = DVec2::new(13., 17.);
    parameters.dimensions = DVec2::new(320., 240.);
    assert!(document.apply(0, parameters, document.configuration())?);
    assert!(Arc::ptr_eq(
        document
            .encoded_png_input()
            .expect("parameter edit keeps source")
            .source(),
        &source
    ));
    assert_eq!(
        document
            .encoded_png_input()
            .expect("parameter edit keeps placement")
            .placement()
            .to_cols_array()
            .map(f64::to_bits),
        input.placement().to_cols_array().map(f64::to_bits)
    );
    assert!(matches!(
        document.accept(block_on(stale.evaluate())),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    let completion = block_on(document.submit().evaluate());
    assert_artboard(
        completion
            .result()
            .expect("source survives Artboard rebuild"),
        NodeId(7001),
        parameters.location,
        parameters.dimensions,
        true,
    );
    assert_eq!(
        embedded_png(
            completion
                .result()
                .expect("actual image after parameter edit")
        )?,
        embedded_png(&old_output)?
    );
    document.accept(completion)?;
    let network = document.checkpoint.network.clone();
    let context_stale = document.submit();
    let mut config = document.configuration();
    config.time.time = -0.;
    assert!(document.apply(1, parameters, config)?);
    assert!(Arc::ptr_eq(&network, &document.checkpoint.network));
    assert!(Arc::ptr_eq(
        document
            .encoded_png_input()
            .expect("context edit keeps source")
            .source(),
        &source
    ));
    assert!(matches!(
        document.accept(block_on(context_stale.evaluate())),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    let pending = document.submit();
    let submission = document.pending.clone().expect("genuine pending request");
    assert!(!document.apply_encoded_png(2, parameters, config, input.clone())?);
    assert_eq!(document.revision(), 2);
    assert!(Arc::ptr_eq(
        &submission,
        document
            .pending
            .as_ref()
            .expect("exact input no-op preserves request")
    ));
    document.accept(block_on(pending.evaluate()))?;
    let signed_zero = CpuSvgEncodedPngInput::new(
        source.clone(),
        DAffine2::from_cols_array(&[1., -0., 0., 1., 0., 0.]),
    )?;
    let stale = document.submit();
    assert!(document.apply_encoded_png(2, parameters, config, signed_zero.clone())?);
    assert_eq!(document.revision(), 3);
    assert_eq!(
        document
            .encoded_png_input()
            .expect("signed-zero placement retained")
            .placement()
            .to_cols_array()
            .map(f64::to_bits),
        signed_zero.placement().to_cols_array().map(f64::to_bits)
    );
    assert!(matches!(
        document.accept(block_on(stale.evaluate())),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    let completion = block_on(document.submit().evaluate());
    document.accept(completion)?;
    let cloned_envelope = encoded_input(bytes.clone(), "original-label", signed_zero.placement())?;
    assert_eq!(cloned_envelope.source().identity(), source.identity());
    assert!(!Arc::ptr_eq(cloned_envelope.source(), &source));
    assert!(document.apply_encoded_png(3, parameters, config, cloned_envelope)?);
    assert_eq!(document.revision(), 4);
    assert!(!document.owns_current_output(&old));
    assert_eq!(old.output(), &old_output);
    assert_eq!(*old_checkpoint.network, old_network);
    assert!(Arc::ptr_eq(
        old_checkpoint
            .encoded_png
            .as_ref()
            .expect("detached source")
            .source(),
        &source
    ));
    assert_eq!(source.encoded_bytes().as_ref(), bytes.as_ref());
    Ok(())
}

#[test]
fn encoded_sources_cross_threads_without_forging_colliding_document_authority() -> TestResult {
    let bytes = encoded_png_fixture(2, 2, &ENCODED_ARTBOARD_RGBA)?;
    let input = encoded_input(bytes.clone(), "same-label", DAffine2::IDENTITY)?;
    let mut first = CpuSvgArtboardDocument::new_with_encoded_png(
        163,
        NodeId(7001),
        CpuSvgArtboardParameters::default(),
        configuration(),
        budget(),
        input.clone(),
    )?;
    let mut second = CpuSvgArtboardDocument::new_with_encoded_png(
        163,
        NodeId(7001),
        CpuSvgArtboardParameters::default(),
        configuration(),
        budget(),
        input,
    )?;
    let foreign = first.submit();
    let own = second.submit();
    let completion = std::thread::spawn(move || block_on(foreign.evaluate()))
        .join()
        .expect("original worker completes");
    let original = completion
        .result()
        .expect("actual threaded encoded output")
        .clone();
    assert!(matches!(
        second.accept(completion),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    assert!(second.is_pending());
    let current = second.accept(block_on(own.evaluate()))?.clone();
    assert_eq!(embedded_png(current.output())?, embedded_png(&original)?);
    let first_network = first.network().clone();
    let cancelled = second.submit();
    let completion = std::thread::spawn(move || block_on(cancelled.evaluate()))
        .join()
        .expect("cancel candidate executes genuinely");
    assert!(completion.result().is_ok());
    assert!(second.cancel());
    assert!(matches!(
        second.accept(completion),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    assert!(second.owns_current_output(&current));
    let superseded = second.submit();
    let pending = second.submit();
    assert!(matches!(
        second.accept(block_on(superseded.evaluate())),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    second.accept(block_on(pending.evaluate()))?;
    assert!(!second.owns_current_output(&current));
    let replacement_bytes = encoded_png_fixture(
        2,
        2,
        &[
            3, 7, 251, 255, 17, 231, 29, 127, 223, 31, 53, 1, 71, 97, 201, 0,
        ],
    )?;
    let replacement = encoded_input(
        replacement_bytes.clone(),
        "explicit-successor-label",
        DAffine2::IDENTITY,
    )?;
    let stale = second.submit();
    assert!(second.apply_encoded_png(
        0,
        second.parameters(),
        second.configuration(),
        replacement.clone()
    )?);
    let completion = std::thread::spawn(move || block_on(stale.evaluate()))
        .join()
        .expect("old source evaluates on worker");
    assert!(matches!(
        second.accept(completion),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    let request = second.submit();
    let completion = std::thread::spawn(move || block_on(request.evaluate()))
        .join()
        .expect("successor worker completes");
    let accepted = second.accept(completion)?.clone();
    assert!(second.owns_current_output(&accepted));
    assert_ne!(embedded_png(accepted.output())?, embedded_png(&original)?);
    assert!(Arc::ptr_eq(
        second
            .encoded_png_input()
            .expect("successor source")
            .source(),
        replacement.source()
    ));
    assert_eq!(first.network(), &first_network);
    assert_eq!(current.output().metadata, original.metadata);
    assert_eq!(
        bytes.as_ref(),
        first
            .encoded_png_input()
            .expect("isolated first source")
            .source()
            .encoded_bytes()
            .as_ref()
    );
    assert_eq!(
        replacement_bytes.as_ref(),
        replacement.source().encoded_bytes().as_ref()
    );
    Ok(())
}

#[test]
fn invalid_encoded_placement_and_stale_commands_preserve_full_committed_and_pending_state()
-> TestResult {
    let bytes = encoded_png_fixture(2, 2, &ENCODED_ARTBOARD_RGBA)?;
    let input = encoded_input(bytes.clone(), "finite-source", DAffine2::IDENTITY)?;
    let source = input.source().clone();
    let mut document = CpuSvgArtboardDocument::new_with_encoded_png(
        167,
        NodeId(7001),
        CpuSvgArtboardParameters::default(),
        configuration(),
        budget(),
        input.clone(),
    )?;
    let completion = block_on(document.submit().evaluate());
    let accepted = document.accept(completion)?.clone();
    let network = document.network().clone();
    let checkpoint = document.checkpoint.clone();
    let pending = document.submit();
    let submission = document
        .pending
        .clone()
        .expect("genuine pending source evaluation");
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for index in 0..6 {
            let mut columns = DAffine2::IDENTITY.to_cols_array();
            columns[index] = value;
            assert!(matches!(
                CpuSvgEncodedPngInput::new(source.clone(), DAffine2::from_cols_array(&columns)),
                Err(CpuSvgDocumentError::InvalidPlacement)
            ));
        }
    }
    let overflow = CpuSvgEncodedPngInput::new(
        source.clone(),
        DAffine2::from_cols_array(&[f64::MAX * 0.75, 0., 0., 1., f64::MAX * 0.75, 0.]),
    )?;
    assert_eq!(
        document.apply_encoded_png(0, document.parameters(), document.configuration(), overflow),
        Err(CpuSvgDocumentError::InvalidPlacement)
    );
    assert_eq!(
        document.apply_encoded_png(
            1,
            document.parameters(),
            document.configuration(),
            input.clone()
        ),
        Err(CpuSvgDocumentError::StaleRevision)
    );
    let mut invalid = document.configuration();
    invalid.scale = 0.;
    assert_eq!(
        document.apply_encoded_png(0, document.parameters(), invalid, input),
        Err(CpuSvgDocumentError::InvalidConfiguration)
    );
    assert_eq!(document.revision(), 0);
    assert_eq!(document.network(), &network);
    assert!(Arc::ptr_eq(&checkpoint, &document.checkpoint));
    assert!(Arc::ptr_eq(
        &submission,
        document
            .pending
            .as_ref()
            .expect("invalid commands preserve active request")
    ));
    assert!(document.owns_current_output(&accepted));
    assert!(Arc::ptr_eq(
        document
            .encoded_png_input()
            .expect("exact source preserved")
            .source(),
        &source
    ));
    assert_eq!(
        document
            .accept(block_on(pending.evaluate()))?
            .output()
            .metadata,
        accepted.output().metadata
    );
    assert_eq!(source.encoded_bytes().as_ref(), bytes.as_ref());
    Ok(())
}

#[test]
fn encoded_valid_envelope_empty_decoder_fallback_is_not_accepted_as_blank_artboard() -> TestResult {
    let bytes = encoded_png_fixture(2, 2, &ENCODED_ARTBOARD_RGBA)?;
    let original_input = encoded_input(bytes.clone(), "valid-parent", DAffine2::IDENTITY)?;
    let parameters = CpuSvgArtboardParameters {
        clip: false,
        ..CpuSvgArtboardParameters::default()
    };
    let mut document = CpuSvgArtboardDocument::new_with_encoded_png(
        173,
        NodeId(7001),
        parameters,
        configuration(),
        budget(),
        original_input.clone(),
    )?;
    let completion = block_on(document.submit().evaluate());
    let old = document.accept(completion)?.clone();
    let old_output = old.output().clone();
    let mut malformed = Vec::new();
    {
        let mut encoder = png::Encoder::new(&mut malformed, 2, 2);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header()?;
        writer.write_chunk(png::chunk::IDAT, b"not zlib pixels")?;
        writer.finish()?;
    }
    let malformed: Arc<[u8]> = malformed.into();
    let bad = encoded_input(
        malformed.clone(),
        "valid-envelope-invalid-pixels",
        DAffine2::IDENTITY,
    )?;
    assert_eq!((bad.source().width(), bad.source().height()), (2, 2));
    assert!(document.apply_encoded_png(0, parameters, document.configuration(), bad.clone())?);
    let network = document.network().clone();
    let completion = block_on(document.submit().evaluate());
    assert!(matches!(
        completion.result(),
        Err(CpuSvgEvaluationError::SourceDecodeFailed)
    ));
    assert!(matches!(
        document.accept(completion),
        Err(CpuSvgAcceptanceError::Evaluation(
            CpuSvgEvaluationError::SourceDecodeFailed
        ))
    ));
    assert!(!document.is_pending());
    assert!(!document.owns_current_output(&old));
    assert_eq!(
        document
            .accepted()
            .expect("last valid output retained")
            .output(),
        &old_output
    );
    assert_eq!(document.network(), &network);
    assert_eq!(bad.source().encoded_bytes().as_ref(), malformed.as_ref());
    assert!(document.apply_encoded_png(
        1,
        parameters,
        document.configuration(),
        original_input.clone()
    )?);
    let completion = block_on(document.submit().evaluate());
    let recovered = document.accept(completion)?.clone();
    assert!(document.owns_current_output(&recovered));
    assert_eq!(
        embedded_png(recovered.output())?,
        embedded_png(&old_output)?
    );
    assert!(Arc::ptr_eq(
        document
            .encoded_png_input()
            .expect("original source recovery")
            .source(),
        original_input.source()
    ));
    assert_eq!(old.output(), &old_output);
    assert_eq!(
        bytes.as_ref(),
        original_input.source().encoded_bytes().as_ref()
    );
    Ok(())
}

#[test]
fn encoded_output_cap_failure_keeps_last_valid_source_and_recovers() -> TestResult {
    let bytes = encoded_png_fixture(2, 2, &ENCODED_ARTBOARD_RGBA)?;
    let input = encoded_input(bytes.clone(), "small-source", DAffine2::IDENTITY)?;
    let parameters = CpuSvgArtboardParameters {
        clip: false,
        ..CpuSvgArtboardParameters::default()
    };
    let mut probe = CpuSvgArtboardDocument::new_with_encoded_png(
        179,
        NodeId(7001),
        parameters,
        configuration(),
        budget(),
        input.clone(),
    )?;
    let completion = block_on(probe.submit().evaluate());
    let actual_small = svg(completion.result().expect("actual unclipped source scene")).len();
    let tight = CpuSvgEvaluationBudget {
        max_output_bytes: actual_small + 128,
        ..budget()
    };
    let mut document = CpuSvgArtboardDocument::new_with_encoded_png(
        179,
        NodeId(7001),
        parameters,
        configuration(),
        tight,
        input.clone(),
    )?;
    let completion = block_on(document.submit().evaluate());
    let old = document.accept(completion)?.clone();
    let old_output = old.output().clone();
    let samples = (0..64 * 64)
        .flat_map(|index| {
            [
                (index % 251) as u8,
                ((index * 73) % 253) as u8,
                ((index * 109) % 255) as u8,
                255,
            ]
        })
        .collect::<Vec<_>>();
    let larger_bytes = encoded_png_fixture(64, 64, &samples)?;
    let larger = encoded_input(larger_bytes.clone(), "larger-source", DAffine2::IDENTITY)?;
    let original = graphite_raster_types::Image::<Color>::from_encoded(&larger_bytes)
        .expect("original larger decoder oracle");
    assert!(original.to_png().len() * 4 / 3 > tight.max_output_bytes);
    assert!(document.apply_encoded_png(0, parameters, document.configuration(), larger.clone())?);
    let network = document.network().clone();
    let completion = block_on(document.submit().evaluate());
    assert!(matches!(
        completion.result(),
        Err(CpuSvgEvaluationError::OutputBudgetExceeded)
    ));
    assert!(matches!(
        document.accept(completion),
        Err(CpuSvgAcceptanceError::Evaluation(
            CpuSvgEvaluationError::OutputBudgetExceeded
        ))
    ));
    assert_eq!(document.network(), &network);
    assert_eq!(
        document
            .accepted()
            .expect("cap failure retains old output")
            .output(),
        &old_output
    );
    assert!(!document.owns_current_output(&old));
    assert!(Arc::ptr_eq(
        document
            .encoded_png_input()
            .expect("large source committed unchanged")
            .source(),
        larger.source()
    ));
    assert!(document.apply_encoded_png(1, parameters, document.configuration(), input.clone())?);
    let completion = block_on(document.submit().evaluate());
    let recovered = document.accept(completion)?.clone();
    assert!(document.owns_current_output(&recovered));
    assert_eq!(
        embedded_png(recovered.output())?,
        embedded_png(&old_output)?
    );
    assert_eq!(
        larger.source().encoded_bytes().as_ref(),
        larger_bytes.as_ref()
    );
    assert_eq!(input.source().encoded_bytes().as_ref(), bytes.as_ref());
    Ok(())
}

#[test]
fn encoded_large_scale_export_validates_actual_logical_geometry_without_unused_physical_product()
-> TestResult {
    let bytes = encoded_png_fixture(2, 2, &ENCODED_ARTBOARD_RGBA)?;
    let before = bytes.to_vec();
    let placement = DAffine2::from_scale(DVec2::splat(1e200));
    let input = encoded_input(bytes.clone(), "finite-large-scale-source", placement)?;
    let mut configuration = configuration();
    configuration.scale = 1e200;
    configuration.viewport.transform = DAffine2::IDENTITY;
    let physical =
        DAffine2::from_scale(DVec2::splat(configuration.scale)) * configuration.viewport.transform;
    let logical = DAffine2::from_scale(DVec2::splat(1. / configuration.scale)) * physical;
    assert!(physical.is_finite());
    assert!(!(physical * placement).is_finite());
    assert!((logical * placement).is_finite());
    for point in [DVec2::ZERO, DVec2::X, DVec2::Y, DVec2::ONE] {
        assert!((logical * placement).transform_point2(point).is_finite());
    }
    let mut document = CpuSvgArtboardDocument::new_with_encoded_png(
        181,
        NodeId(7001),
        CpuSvgArtboardParameters::default(),
        configuration,
        budget(),
        input.clone(),
    )?;
    let network = document.network().clone();
    let completion = block_on(document.submit().evaluate());
    let original = completion
        .result()
        .expect("genuine finite logical export")
        .clone();
    assert_artboard(
        &original,
        NodeId(7001),
        DVec2::ZERO,
        DVec2::new(1920., 1080.),
        true,
    );
    let oracle = graphite_raster_types::Image::<Color>::from_encoded(&bytes)
        .expect("original decoder oracle");
    assert_eq!(embedded_png(&original)?, oracle.to_png());
    let matrix = graphite_rendering::format_transform_matrix(placement);
    assert!(svg(&original).contains(&format!("transform=\"{matrix}\"")));
    let (_, image) = svg(&original)
        .split_once("data:image/png;base64,")
        .expect("original encoded image");
    let (image, _) = image.split_once('"').expect("original image terminator");
    let geometry = svg(&original).replace(image, "");
    assert!(!geometry.contains("NaN"));
    assert!(!geometry.contains("inf"));
    assert!(
        original
            .metadata
            .upstream_footprints
            .values()
            .all(|footprint| footprint.transform.is_finite())
    );
    let accepted = document.accept(completion)?.clone();
    assert_eq!(accepted.output(), &original);
    assert_eq!(accepted.output().metadata, original.metadata);
    assert!(document.owns_current_output(&accepted));
    assert_eq!(accepted.configuration(), configuration);
    assert_eq!(
        document
            .encoded_png_input()
            .expect("large placement retained")
            .placement()
            .to_cols_array()
            .map(f64::to_bits),
        placement.to_cols_array().map(f64::to_bits)
    );
    assert!(Arc::ptr_eq(
        document
            .encoded_png_input()
            .expect("source retained")
            .source(),
        input.source()
    ));
    assert_eq!(document.network(), &network);
    assert_eq!(input.source().encoded_bytes().as_ref(), before.as_slice());
    Ok(())
}

#[test]
fn accepted_encoded_provenance_belongs_to_captured_checkpoint_after_source_edits() -> TestResult {
    let bytes = encoded_png_fixture(2, 2, &ENCODED_ARTBOARD_RGBA)?;
    let input = encoded_input(bytes.clone(), "first-input-label", DAffine2::IDENTITY)?;
    let source = input.source().clone();
    let parameters = CpuSvgArtboardParameters::default();
    let configuration = configuration();
    let mut document = CpuSvgArtboardDocument::new_with_encoded_png(
        193,
        NodeId(7001),
        parameters,
        configuration,
        budget(),
        input,
    )?;
    let completion = block_on(document.submit().evaluate());
    let first = document.accept(completion)?.clone();
    let first_output = first.output().clone();
    let first_network = document.network().clone();
    let placement = DAffine2::from_cols_array(&[2., -0., 0., 3., 5., 7.]);
    let successor = encoded_input(bytes.clone(), "second-input-label", placement)?;
    let successor_source = successor.source().clone();
    let mut successor_parameters = parameters;
    successor_parameters.location = DVec2::new(11., 13.);
    let mut successor_configuration = configuration;
    successor_configuration.pointer = DVec2::new(17., 19.);
    assert!(document.apply_encoded_png(
        0,
        successor_parameters,
        successor_configuration,
        successor,
    )?);
    assert!(!document.owns_current_output(&first));
    let completion = block_on(document.submit().evaluate());
    let second = document.accept(completion)?.clone();
    assert!(document.owns_current_output(&second));
    assert_eq!(first.parameters(), parameters);
    assert_eq!(first.configuration(), configuration);
    assert_eq!(first.revision(), 0);
    assert_eq!(second.parameters(), successor_parameters);
    assert_eq!(second.configuration(), successor_configuration);
    assert_eq!(second.revision(), 1);
    assert_eq!(first.document_id_encoding(), second.document_id_encoding());
    assert_eq!(first.node_id(), second.node_id());
    let captured = first.encoded_png_input().expect("first authentic source");
    assert!(Arc::ptr_eq(captured.source(), &source));
    assert_eq!(
        captured.source().identity().version_id.as_ref(),
        "first-input-label"
    );
    assert_eq!(
        captured.placement().to_cols_array().map(f64::to_bits),
        DAffine2::IDENTITY.to_cols_array().map(f64::to_bits)
    );
    let captured_successor = second
        .encoded_png_input()
        .expect("successor authentic source");
    assert!(Arc::ptr_eq(captured_successor.source(), &successor_source));
    assert_eq!(
        captured_successor.source().identity().version_id.as_ref(),
        "second-input-label"
    );
    assert_eq!(
        captured_successor
            .placement()
            .to_cols_array()
            .map(f64::to_bits),
        placement.to_cols_array().map(f64::to_bits)
    );
    assert_eq!(captured.source().encoded_bytes().as_ref(), bytes.as_ref());
    assert_eq!(
        captured_successor.source().encoded_bytes().as_ref(),
        bytes.as_ref()
    );
    assert_eq!(first.output(), &first_output);
    assert_ne!(document.network(), &first_network);
    document.cancel();
    drop(document);
    assert_eq!(first.parameters(), parameters);
    assert!(Arc::ptr_eq(
        first.encoded_png_input().expect("retained parent").source(),
        &source
    ));
    assert_eq!(second.parameters(), successor_parameters);
    assert!(Arc::ptr_eq(
        second
            .encoded_png_input()
            .expect("retained successor")
            .source(),
        &successor_source
    ));
    let mut resource_free =
        CpuSvgArtboardDocument::new(197, NodeId(7001), parameters, configuration, budget())?;
    let completion = block_on(resource_free.submit().evaluate());
    let resource_free = resource_free.accept(completion)?;
    assert!(resource_free.encoded_png_input().is_none());
    assert_eq!(resource_free.parameters(), parameters);
    Ok(())
}

#[test]
fn detached_edit_checkpoint_retains_content_without_live_request_or_output_authority() -> TestResult
{
    let bytes = encoded_png_fixture(2, 2, &ENCODED_ARTBOARD_RGBA)?;
    let source_bytes = bytes.to_vec();
    let input = encoded_input(bytes.clone(), "detached-edit-source", DAffine2::IDENTITY)?;
    let source = input.source().clone();
    let mut live = CpuSvgArtboardDocument::new_with_encoded_png(
        211,
        NodeId(7001),
        CpuSvgArtboardParameters::default(),
        configuration(),
        budget(),
        input,
    )?;
    let completion = block_on(live.submit().evaluate());
    let accepted = live.accept(completion)?.clone();
    let live_network = live.network().clone();
    let evaluation = live.submit();
    let mut draft = live.detach_for_edit();
    assert!(Arc::ptr_eq(&live.checkpoint, &draft.checkpoint));
    assert!(Arc::ptr_eq(
        &live.checkpoint.network,
        &draft.checkpoint.network
    ));
    assert_eq!(draft.document_id_encoding(), live.document_id_encoding());
    assert_eq!(draft.node_id(), live.node_id());
    assert_eq!(draft.revision(), live.revision());
    assert_eq!(draft.parameters(), live.parameters());
    assert_eq!(draft.configuration(), live.configuration());
    assert!(!draft.is_pending());
    assert!(draft.accepted().is_none());
    assert!(!draft.owns_current_output(&accepted));
    assert!(live.owns_current_output(&accepted));
    let completion = block_on(evaluation.evaluate());
    assert!(matches!(
        draft.accept(completion),
        Err(CpuSvgAcceptanceError::ForeignOrStaleCompletion)
    ));
    assert!(live.is_pending());
    assert!(live.owns_current_output(&accepted));
    let mut invalid = draft.parameters();
    invalid.dimensions.x = f64::NAN;
    assert!(matches!(
        draft.apply(0, invalid, draft.configuration()),
        Err(CpuSvgDocumentError::InvalidGeometry)
    ));
    assert_eq!(draft.revision(), 0);
    assert!(Arc::ptr_eq(&live.checkpoint, &draft.checkpoint));
    let mut changed = draft.parameters();
    changed.location = DVec2::new(17., 23.);
    assert!(draft.apply(0, changed, draft.configuration())?);
    assert_eq!(draft.revision(), 1);
    assert_eq!(live.revision(), 0);
    assert_ne!(draft.network(), &live_network);
    assert_eq!(live.network(), &live_network);
    assert!(live.is_pending());
    assert!(live.owns_current_output(&accepted));
    assert!(Arc::ptr_eq(
        draft.encoded_png_input().expect("draft source").source(),
        &source
    ));
    assert!(Arc::ptr_eq(
        live.encoded_png_input().expect("live source").source(),
        &source
    ));
    let completion = block_on(draft.submit().evaluate());
    let draft_output = draft.accept(completion)?.clone();
    assert_artboard(
        draft_output.output(),
        NodeId(7001),
        changed.location,
        changed.dimensions,
        changed.clip,
    );
    assert!(draft.owns_current_output(&draft_output));
    assert!(!live.owns_current_output(&draft_output));
    let completion = block_on(live.submit().evaluate());
    let live_output = live.accept(completion)?.clone();
    assert!(live.owns_current_output(&live_output));
    assert!(!draft.owns_current_output(&live_output));
    drop(draft);
    assert_eq!(live.revision(), 0);
    assert_eq!(live.network(), &live_network);
    assert!(live.owns_current_output(&live_output));
    assert_eq!(source.encoded_bytes().as_ref(), source_bytes.as_slice());
    let resource_free = document(213)?;
    let free_draft = resource_free.detach_for_edit();
    assert!(free_draft.encoded_png_input().is_none());
    assert!(!free_draft.is_pending());
    assert!(free_draft.accepted().is_none());
    assert_eq!(free_draft.network(), resource_free.network());
    Ok(())
}

#[test]
fn linked_engine_payload_uses_original_legacy_templates_and_persistent_resource_id() -> TestResult {
    use crate::formats::capture_artboard_engine_payload;
    use graphite_editor::messages::portfolio::document::DocumentMessageHandler;
    use graphite_graph_craft::application_io::resource::ResourceId;

    let input = encoded_input(
        encoded_png_fixture(2, 2, &ENCODED_ARTBOARD_RGBA)?,
        "supplied-immutable-version",
        DAffine2::from_scale(DVec2::new(16., 12.)),
    )?;
    let owner = CpuSvgArtboardDocument::new_with_encoded_png(
        901,
        NodeId(7103),
        CpuSvgArtboardParameters::default(),
        configuration(),
        budget(),
        input,
    )?;
    let before = owner.network().clone();
    let resource_id = ResourceId::from(902);
    let payload = capture_artboard_engine_payload(&owner, resource_id, 256 * 1024)?;
    let original: DocumentMessageHandler =
        serde_json::from_value(payload.legacy_document().clone())?;
    let network = original.network_interface.document_network();
    assert_eq!(network.exports, vec![NodeInput::node(NodeId(7103), 0)]);
    assert_eq!(network.nodes.len(), 3);
    assert!(network.is_acyclic());
    let [_, decode_id, placement_id] = encoded_png_node_ids(NodeId(7103))?;
    assert!(
        matches!(network.nodes.get(&decode_id).and_then(|node| node.inputs.first()).and_then(NodeInput::as_value), Some(TaggedValue::Resource(id)) if *id == resource_id)
    );
    assert_eq!(
        network
            .nodes
            .get(&placement_id)
            .expect("original placement")
            .inputs
            .first(),
        Some(&NodeInput::node(decode_id, 0))
    );
    assert!(
        network.nodes.values().all(|node| node.implementation
            != DocumentNodeImplementation::ProtoNode(resource::IDENTIFIER))
    );
    assert_eq!(
        original.resources.registry.ids().collect::<Vec<_>>(),
        vec![resource_id]
    );
    let saved_resource = original
        .resources
        .registry
        .info(&resource_id)
        .expect("persistent linked resource");
    assert_eq!(
        saved_resource.hash.copied(),
        Some(ResourceHash::from(
            owner
                .encoded_png_input()
                .expect("PNG")
                .source()
                .encoded_bytes()
                .as_ref()
        ))
    );
    assert!(saved_resource.sources.is_empty());
    assert!(original.resources.embedded.is_empty());
    let metadata = original.network_interface.document_network_metadata();
    let artboard_metadata = &metadata
        .persistent_metadata
        .node_metadata
        .get(&NodeId(7103))
        .expect("complete Artboard metadata")
        .persistent_metadata;
    let templates: serde_json::Value =
        serde_json::from_str(include_str!("persistent-artboard-templates.json"))?;
    let template_metadata = &templates["artboard"]["persistent_node_metadata"];
    let actual_metadata = serde_json::to_value(artboard_metadata)?;
    assert_eq!(
        actual_metadata["input_metadata"],
        template_metadata["input_metadata"]
    );
    assert_eq!(
        actual_metadata["network_metadata"],
        template_metadata["network_metadata"]
    );
    assert_eq!(
        actual_metadata["output_names"],
        template_metadata["output_names"]
    );
    assert_eq!(payload.resource_id(), resource_id);
    assert_eq!(owner.network(), &before);
    assert!(!owner.is_pending());
    assert!(owner.accepted().is_none());
    let wire: serde_json::Value = serde_json::from_slice(payload.serialized_bytes())?;
    assert_eq!(wire["legacy_document"], *payload.legacy_document());
    assert!(wire.get("capture").is_some());
    assert!(!String::from_utf8(payload.serialized_bytes().to_vec())?.contains("ResourceHash"));
    Ok(())
}

#[test]
fn linked_engine_payload_reopens_independently_with_exact_bits_revision_and_original_png()
-> TestResult {
    use crate::formats::{capture_artboard_engine_payload, reopen_artboard_engine_payload};
    use graphite_graph_craft::application_io::resource::ResourceId;

    let bytes = encoded_png_fixture(2, 2, &ENCODED_ARTBOARD_RGBA)?;
    let placement = DAffine2::from_cols_array(&[16., -0., 0., 12., -0., 3.]);
    let mut parameters = CpuSvgArtboardParameters {
        location: DVec2::new(-0., 0.),
        dimensions: DVec2::new(24., 16.),
        background: Color::from_rgbaf32_unchecked(-0., 0.125, 0.75, 0.),
        clip: false,
    };
    let mut config = configuration();
    config.pointer = DVec2::new(-0., 17.);
    config.time.time = -0.;
    let mut owner = CpuSvgArtboardDocument::new_with_encoded_png(
        903,
        NodeId(7001),
        parameters,
        config,
        budget(),
        encoded_input(bytes.clone(), "version-903", placement)?,
    )?;
    parameters.dimensions.x = 25.;
    assert!(owner.apply(0, parameters, config)?);
    let payload = capture_artboard_engine_payload(&owner, ResourceId::from(904), 256 * 1024)?;
    let fresh_input = encoded_input(bytes, "version-903", DAffine2::IDENTITY)?;
    let fresh_source = fresh_input.source().clone();
    let mut reopened = reopen_artboard_engine_payload(
        payload.serialized_bytes(),
        fresh_source.clone(),
        budget(),
        256 * 1024,
    )?;
    assert_eq!(reopened.document_id_encoding(), 903);
    assert_eq!(reopened.node_id(), NodeId(7001));
    assert_eq!(reopened.revision(), 1);
    assert!(same_parameters(reopened.parameters(), parameters));
    assert!(same_configuration(reopened.configuration(), config));
    assert_eq!(
        reopened
            .encoded_png_input()
            .expect("linked PNG")
            .placement()
            .to_cols_array()
            .map(f64::to_bits),
        placement.to_cols_array().map(f64::to_bits)
    );
    assert!(Arc::ptr_eq(
        reopened
            .encoded_png_input()
            .expect("freshly verified PNG")
            .source(),
        &fresh_source
    ));
    assert!(!Arc::ptr_eq(
        reopened.encoded_png_input().expect("fresh source").source(),
        owner.encoded_png_input().expect("old source").source()
    ));
    assert_eq!(owner.network(), reopened.network());
    let original_completion = block_on(owner.submit().evaluate());
    let reopened_completion = block_on(reopened.submit().evaluate());
    let original_output = original_completion.result().expect("real original SVG");
    let reopened_output = reopened_completion.result().expect("real reopened SVG");
    assert_eq!(
        embedded_png(original_output)?,
        embedded_png(reopened_output)?
    );
    assert_eq!(original_output.metadata, reopened_output.metadata);
    assert_eq!(svg(original_output), svg(reopened_output));
    let old_accepted = owner.accept(original_completion)?.clone();
    assert!(!reopened.owns_current_output(&old_accepted));
    reopened.accept(reopened_completion)?;
    let current_output = reopened
        .accepted()
        .expect("own reopened acceptance")
        .clone();
    drop(owner);
    assert!(reopened.owns_current_output(&current_output));
    parameters.dimensions.x = 26.;
    assert!(reopened.apply(1, parameters, config)?);
    assert!(!reopened.owns_current_output(&current_output));
    let recaptured = capture_artboard_engine_payload(&reopened, ResourceId::from(904), 256 * 1024)?;
    let independent = reopen_artboard_engine_payload(
        recaptured.serialized_bytes(),
        fresh_source,
        budget(),
        256 * 1024,
    )?;
    assert_eq!(independent.revision(), 2);
    assert!(!independent.is_pending());
    assert!(independent.accepted().is_none());
    Ok(())
}

#[test]
fn linked_engine_payload_rejects_unsupported_graph_metadata_and_missing_payload_without_normalization()
-> TestResult {
    use crate::formats::{
        ArtboardPersistenceError, capture_artboard_engine_payload, reopen_artboard_engine_payload,
    };
    use graphite_graph_craft::application_io::resource::ResourceId;
    use serde_json::json;

    let input = encoded_input(
        encoded_png_fixture(2, 2, &ENCODED_ARTBOARD_RGBA)?,
        "version-905",
        DAffine2::IDENTITY,
    )?;
    let source = input.source().clone();
    let owner = CpuSvgArtboardDocument::new_with_encoded_png(
        905,
        NodeId(7001),
        CpuSvgArtboardParameters::default(),
        configuration(),
        budget(),
        input,
    )?;
    let before = owner.network().clone();
    let payload = capture_artboard_engine_payload(&owner, ResourceId::from(906), 256 * 1024)?;
    let original: serde_json::Value = serde_json::from_slice(payload.serialized_bytes())?;
    for pointer in [
        "/legacy_document/rulers_visible",
        "/legacy_document/graph_view_overlay_open",
    ] {
        let mut tampered = original.clone();
        let value = tampered.pointer_mut(pointer).expect("source default field");
        *value = json!(!value.as_bool().expect("real boolean source field"));
        assert!(matches!(
            reopen_artboard_engine_payload(
                &serde_json::to_vec(&tampered)?,
                source.clone(),
                budget(),
                256 * 1024
            ),
            Err(ArtboardPersistenceError::UnsupportedPayload)
        ));
    }
    let mut original_owner: graphite_editor::messages::portfolio::document::DocumentMessageHandler =
        serde_json::from_value(payload.legacy_document().clone())?;
    original_owner.network_interface.set_display_name(
        &NodeId(7001),
        "Unsupported metadata edit".into(),
        &[],
    );
    let mut changed_metadata = original.clone();
    changed_metadata["legacy_document"] = serde_json::to_value(&original_owner)?;
    assert!(matches!(
        reopen_artboard_engine_payload(
            &serde_json::to_vec(&changed_metadata)?,
            source.clone(),
            budget(),
            256 * 1024
        ),
        Err(ArtboardPersistenceError::UnsupportedPayload)
    ));
    original_owner.network_interface.set_input(
        &graphite_editor::messages::portfolio::document::utility_types::network_interface::InputConnector::Node { node_id: NodeId(7001), input_index: 3 },
        NodeInput::value(TaggedValue::DVec2(DVec2::new(19., 23.)), false), &[],
    );
    let mut changed_graph = original.clone();
    changed_graph["legacy_document"] = serde_json::to_value(&original_owner)?;
    assert!(matches!(
        reopen_artboard_engine_payload(
            &serde_json::to_vec(&changed_graph)?,
            source.clone(),
            budget(),
            256 * 1024
        ),
        Err(ArtboardPersistenceError::UnsupportedPayload)
    ));
    let mut unknown = original.clone();
    unknown["legacy_document"]["unsupported_original_state"] = json!(true);
    assert!(matches!(
        reopen_artboard_engine_payload(
            &serde_json::to_vec(&unknown)?,
            source.clone(),
            budget(),
            256 * 1024
        ),
        Err(ArtboardPersistenceError::UnsupportedPayload)
    ));
    let mut changed = original.clone();
    changed["capture"]["dimensions"][0] = json!(100_f64.to_bits());
    assert!(matches!(
        reopen_artboard_engine_payload(
            &serde_json::to_vec(&changed)?,
            source.clone(),
            budget(),
            256 * 1024
        ),
        Err(ArtboardPersistenceError::UnsupportedPayload)
    ));
    let mut missing = original.clone();
    missing
        .as_object_mut()
        .expect("payload object")
        .remove("legacy_document");
    assert!(matches!(
        reopen_artboard_engine_payload(
            &serde_json::to_vec(&missing)?,
            source.clone(),
            budget(),
            256 * 1024
        ),
        Err(ArtboardPersistenceError::Json(_))
    ));
    let mut future = original;
    future["profile"] = json!("sim.graphite.unsupported-future.v2");
    assert!(matches!(
        reopen_artboard_engine_payload(
            &serde_json::to_vec(&future)?,
            source.clone(),
            budget(),
            256 * 1024
        ),
        Err(ArtboardPersistenceError::UnsupportedPayload)
    ));
    assert!(matches!(
        reopen_artboard_engine_payload(b"{", source, budget(), 256 * 1024),
        Err(ArtboardPersistenceError::Json(_))
    ));
    assert_eq!(owner.network(), &before);
    assert_eq!(owner.revision(), 0);
    Ok(())
}

#[test]
fn linked_engine_payload_rejects_source_substitution_and_respects_fresh_budgets() -> TestResult {
    use crate::formats::{
        ArtboardPersistenceError, capture_artboard_engine_payload, reopen_artboard_engine_payload,
    };
    use graphite_graph_craft::application_io::resource::ResourceId;

    let bytes = encoded_png_fixture(2, 2, &ENCODED_ARTBOARD_RGBA)?;
    let input = encoded_input(bytes.clone(), "version-907", DAffine2::IDENTITY)?;
    let source = input.source().clone();
    let owner = CpuSvgArtboardDocument::new_with_encoded_png(
        907,
        NodeId(7001),
        CpuSvgArtboardParameters::default(),
        configuration(),
        budget(),
        input,
    )?;
    let payload = capture_artboard_engine_payload(&owner, ResourceId::from(908), 256 * 1024)?;
    let different_version = encoded_input(bytes, "other-version", DAffine2::IDENTITY)?;
    assert!(matches!(
        reopen_artboard_engine_payload(
            payload.serialized_bytes(),
            different_version.source().clone(),
            budget(),
            256 * 1024
        ),
        Err(ArtboardPersistenceError::SourceMismatch)
    ));
    let different_pixels = encoded_input(
        encoded_png_fixture(2, 2, &[255; 16])?,
        "version-907",
        DAffine2::IDENTITY,
    )?;
    assert!(matches!(
        reopen_artboard_engine_payload(
            payload.serialized_bytes(),
            different_pixels.source().clone(),
            budget(),
            256 * 1024
        ),
        Err(ArtboardPersistenceError::SourceMismatch)
    ));
    let different_shape = encoded_input(
        encoded_png_fixture(1, 1, &[255; 4])?,
        "version-907",
        DAffine2::IDENTITY,
    )?;
    assert!(matches!(
        reopen_artboard_engine_payload(
            payload.serialized_bytes(),
            different_shape.source().clone(),
            budget(),
            256 * 1024
        ),
        Err(ArtboardPersistenceError::SourceMismatch)
    ));
    assert!(matches!(
        capture_artboard_engine_payload(&owner, ResourceId::from(908), 1),
        Err(ArtboardPersistenceError::PayloadBudgetExceeded)
    ));
    assert!(matches!(
        reopen_artboard_engine_payload(payload.serialized_bytes(), source.clone(), budget(), 1),
        Err(ArtboardPersistenceError::PayloadBudgetExceeded)
    ));
    assert!(matches!(
        reopen_artboard_engine_payload(
            payload.serialized_bytes(),
            source,
            CpuSvgEvaluationBudget {
                max_viewport_pixels: 1,
                max_output_bytes: 1
            },
            256 * 1024
        ),
        Err(ArtboardPersistenceError::Document(
            CpuSvgDocumentError::ViewportBudgetExceeded
        ))
    ));
    assert!(matches!(
        capture_artboard_engine_payload(&document(909)?, ResourceId::from(908), 256 * 1024),
        Err(ArtboardPersistenceError::MissingEncodedInput)
    ));
    Ok(())
}

#[test]
fn linked_engine_capture_preserves_finite_non_decimal_bit_patterns() -> TestResult {
    use crate::formats::{capture_artboard_engine_payload, reopen_artboard_engine_payload};
    use graphite_graph_craft::application_io::resource::ResourceId;
    let mut configuration = configuration();
    configuration.pointer = DVec2::new(f64::from_bits(0x3feb0e7009b61ce0), -0.);
    configuration.time.time = f64::from_bits(0x3ff0000000000001);
    configuration.viewport.quality = RenderQuality::Scale(f32::from_bits(0x3eaaaaab));
    let placement =
        DAffine2::from_cols_array(&[1., -0., 0., 1., f64::from_bits(0x3feb0e7009b61ce0), -0.]);
    let parameters = CpuSvgArtboardParameters {
        location: DVec2::new(f64::from_bits(0x3ff0000000000001), -0.),
        ..CpuSvgArtboardParameters::default()
    };
    let input = encoded_input(
        encoded_png_fixture(2, 2, &ENCODED_ARTBOARD_RGBA)?,
        "exact-bits",
        placement,
    )?;
    let source = input.source().clone();
    let owner = CpuSvgArtboardDocument::new_with_encoded_png(
        u64::MAX - 1,
        NodeId(u64::MAX - 1),
        parameters,
        configuration,
        budget(),
        input,
    )?;
    let payload =
        capture_artboard_engine_payload(&owner, ResourceId::from(u64::MAX - 2), 256 * 1024)?;
    let restored =
        reopen_artboard_engine_payload(payload.serialized_bytes(), source, budget(), 256 * 1024)?;
    assert_eq!(restored.document_id_encoding(), u64::MAX - 1);
    assert_eq!(restored.node_id(), NodeId(u64::MAX - 1));
    assert!(same_parameters(restored.parameters(), parameters));
    assert!(same_configuration(restored.configuration(), configuration));
    assert_eq!(
        restored
            .encoded_png_input()
            .expect("source input")
            .placement()
            .to_cols_array()
            .map(f64::to_bits),
        placement.to_cols_array().map(f64::to_bits)
    );
    Ok(())
}

#[test]
fn linked_engine_payload_rejects_original_layer_identifier_overflow_before_source_deserialization()
-> TestResult {
    use crate::formats::{
        ArtboardPersistenceError, capture_artboard_engine_payload, reopen_artboard_engine_payload,
    };
    use graphite_graph_craft::application_io::resource::ResourceId;
    let input = encoded_input(
        encoded_png_fixture(2, 2, &ENCODED_ARTBOARD_RGBA)?,
        "overflow",
        DAffine2::IDENTITY,
    )?;
    let owner = CpuSvgArtboardDocument::new_with_encoded_png(
        40,
        NodeId(u64::MAX),
        CpuSvgArtboardParameters::default(),
        configuration(),
        budget(),
        input.clone(),
    )?;
    assert!(matches!(
        capture_artboard_engine_payload(&owner, ResourceId::from(42), 256 * 1024),
        Err(ArtboardPersistenceError::UnsupportedPayload)
    ));
    let valid = CpuSvgArtboardDocument::new_with_encoded_png(
        40,
        NodeId(7001),
        CpuSvgArtboardParameters::default(),
        configuration(),
        budget(),
        input.clone(),
    )?;
    let payload = capture_artboard_engine_payload(&valid, ResourceId::from(42), 256 * 1024)?;
    let mut tampered: serde_json::Value = serde_json::from_slice(payload.serialized_bytes())?;
    tampered["capture"]["node_id"] = serde_json::json!(u64::MAX);
    assert!(matches!(
        reopen_artboard_engine_payload(
            &serde_json::to_vec(&tampered)?,
            input.source().clone(),
            budget(),
            256 * 1024
        ),
        Err(ArtboardPersistenceError::UnsupportedPayload)
    ));
    Ok(())
}
