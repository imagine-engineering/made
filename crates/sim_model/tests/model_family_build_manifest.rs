#[path = "../build.rs"]
#[allow(dead_code)]
mod build_script;

use build_script::ModelFamilyBuildEntry;
use sim_model::{
    GENERATED_MODEL_FAMILIES, GENERATED_MODEL_FAMILY_FEATURE_IDS, GENERATED_MODEL_FAMILY_FIXTURES,
    GENERATED_MODEL_FAMILY_MODULES, GENERATED_MODEL_FAMILY_REGISTRATIONS,
    GENERATED_MODEL_FAMILY_SOURCE_MANIFEST,
};

#[test]
fn generated_registration_and_exact_manifests_share_source_order() {
    let count = GENERATED_MODEL_FAMILY_REGISTRATIONS.len();
    assert_eq!(GENERATED_MODEL_FAMILIES.len(), count);
    assert_eq!(GENERATED_MODEL_FAMILY_SOURCE_MANIFEST.len(), count);
    assert_eq!(GENERATED_MODEL_FAMILY_MODULES.len(), count);
    assert_eq!(GENERATED_MODEL_FAMILY_FEATURE_IDS.len(), count);
    assert_eq!(GENERATED_MODEL_FAMILY_FIXTURES.len(), count);
    assert!(
        GENERATED_MODEL_FAMILY_REGISTRATIONS
            .windows(2)
            .all(|pair| pair[0].source_ordinal < pair[1].source_ordinal)
    );

    for (index, registration) in GENERATED_MODEL_FAMILY_REGISTRATIONS.iter().enumerate() {
        let (module, feature_id, fixture, source_ordinal) =
            GENERATED_MODEL_FAMILY_SOURCE_MANIFEST[index];
        assert_eq!(module, GENERATED_MODEL_FAMILY_MODULES[index]);
        assert_eq!(feature_id, GENERATED_MODEL_FAMILY_FEATURE_IDS[index]);
        assert_eq!(fixture, GENERATED_MODEL_FAMILY_FIXTURES[index]);
        assert_eq!(source_ordinal, registration.source_ordinal);
        assert_eq!(feature_id, registration.definition.feature_id);
        assert_eq!(
            registration.definition.feature_id,
            GENERATED_MODEL_FAMILIES[index].feature_id
        );
    }
}

#[test]
fn synthetic_registrations_are_parsed_and_sorted_by_source_ordinal()
-> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new("synthetic.rs");
    let later = build_script::model_family_build_entry(
        &synthetic_source("Later", "COMFY-MODEL-9002", "later-comfy-model-9002", 2),
        path,
        "later_sim_model_9002",
    )?;
    let earlier = build_script::model_family_build_entry(
        &synthetic_source("Earlier", "COMFY-MODEL-9001", "earlier-comfy-model-9001", 1),
        path,
        "earlier_sim_model_9001",
    )?;
    let mut entries = Vec::new();
    build_script::register_model_family_entry(&mut entries, later)?;
    build_script::register_model_family_entry(&mut entries, earlier)?;
    build_script::sort_model_family_entries(&mut entries);
    assert_eq!(
        entries
            .iter()
            .map(|entry| (entry.module.as_str(), entry.source_ordinal))
            .collect::<Vec<_>>(),
        [("earlier_sim_model_9001", 1), ("later_sim_model_9002", 2),]
    );
    Ok(())
}

#[test]
fn duplicate_manifest_owners_are_rejected() {
    let original = entry(
        "family_sim_model_9001",
        "Family",
        "COMFY-MODEL-9001",
        "family-comfy-model-9001",
        1,
    );
    for duplicate in [
        entry(
            "family_sim_model_9001",
            "Other",
            "COMFY-MODEL-9002",
            "other-comfy-model-9002",
            2,
        ),
        entry(
            "other_sim_model_9002",
            "Family",
            "COMFY-MODEL-9002",
            "other-comfy-model-9002",
            2,
        ),
        entry(
            "other_sim_model_9002",
            "Other",
            "COMFY-MODEL-9001",
            "other-comfy-model-9002",
            2,
        ),
        entry(
            "other_sim_model_9002",
            "Other",
            "COMFY-MODEL-9002",
            "family-comfy-model-9001",
            2,
        ),
        entry(
            "other_sim_model_9002",
            "Other",
            "COMFY-MODEL-9002",
            "other-comfy-model-9002",
            1,
        ),
    ] {
        let mut entries = vec![original.clone()];
        assert!(build_script::register_model_family_entry(&mut entries, duplicate).is_err());
        assert_eq!(entries.as_slice(), std::slice::from_ref(&original));
    }
}

#[test]
fn malformed_source_declarations_are_rejected() {
    let path = std::path::Path::new("malformed.rs");
    let valid = synthetic_source("Family", "COMFY-MODEL-9001", "family-comfy-model-9001", 1);
    assert!(
        build_script::model_family_build_entry(
            &valid.replace("COMFY-MODEL-9001", "MODEL-9001"),
            path,
            "family_sim_model_9001",
        )
        .is_err()
    );
    assert!(
        build_script::model_family_build_entry(
            &valid.replace("MODEL_FAMILY_REGISTRATION", "NOT_A_REGISTRATION"),
            path,
            "family_sim_model_9001",
        )
        .is_err()
    );
    assert!(
        build_script::model_family_build_entry(
            &valid.replace("definition: &MODEL_FAMILY", "definition: &OTHER_FAMILY"),
            path,
            "family_sim_model_9001",
        )
        .is_err()
    );
    assert!(
        build_script::model_family_build_entry(
            &valid.replace("source_ordinal: 1", "source_ordinal: invalid"),
            path,
            "family_sim_model_9001",
        )
        .is_err()
    );
    assert!(build_script::model_family_build_entry(&valid, path, "Invalid-Module").is_err());
}

#[test]
fn synthetic_test_and_fixture_closure_rejects_missing_and_orphan_rows()
-> Result<(), Box<dyn std::error::Error>> {
    let entries = vec![entry(
        "family_sim_model_9001",
        "Family",
        "COMFY-MODEL-9001",
        "family-comfy-model-9001",
        1,
    )];
    let tests = tempfile::tempdir()?;
    assert!(build_script::model_family_test_names_in(&entries, tests.path()).is_err());
    std::fs::write(tests.path().join("family_sim_model_9001.rs"), b"")?;
    std::fs::write(tests.path().join("._family_sim_model_9001.rs"), [0xff])?;
    assert_eq!(
        build_script::model_family_test_names_in(&entries, tests.path())?,
        ["family_sim_model_9001"]
    );
    std::fs::write(tests.path().join("orphan_sim_model_9002.rs"), b"")?;
    assert!(build_script::model_family_test_names_in(&entries, tests.path()).is_err());

    let fixtures = tempfile::tempdir()?;
    let tuple_entries = vec![(
        "family_sim_model_9001".to_owned(),
        "Family".to_owned(),
        "COMFY-MODEL-9001".to_owned(),
        "family-comfy-model-9001".to_owned(),
    )];
    assert!(build_script::model_family_fixture_names_in(&tuple_entries, fixtures.path()).is_err());
    let fixture = fixtures.path().join("family-comfy-model-9001");
    std::fs::create_dir(&fixture)?;
    std::fs::write(fixture.join("family.json"), b"{}")?;
    assert_eq!(
        build_script::model_family_fixture_names_in(&tuple_entries, fixtures.path())?,
        ["family-comfy-model-9001"]
    );
    let orphan = fixtures.path().join("orphan-comfy-model-9002");
    std::fs::create_dir(&orphan)?;
    std::fs::write(orphan.join("family.json"), b"{}")?;
    assert!(build_script::model_family_fixture_names_in(&tuple_entries, fixtures.path()).is_err());
    Ok(())
}

#[test]
fn zero_family_rows_have_valid_empty_test_and_fixture_closure()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    std::fs::write(directory.path().join("._metadata.rs"), [0xff])?;
    assert!(build_script::model_family_test_names_in(&[], directory.path())?.is_empty());
    assert!(build_script::model_family_fixture_names_in(&[], directory.path())?.is_empty());
    Ok(())
}

fn entry(
    module: &str,
    identifier: &str,
    feature_id: &str,
    fixture: &str,
    source_ordinal: u16,
) -> ModelFamilyBuildEntry {
    ModelFamilyBuildEntry {
        module: module.to_owned(),
        identifier: identifier.to_owned(),
        feature_id: feature_id.to_owned(),
        fixture: fixture.to_owned(),
        source_ordinal,
    }
}

fn synthetic_source(identifier: &str, feature_id: &str, fixture: &str, ordinal: u16) -> String {
    format!(
        "pub const MODEL_FAMILY_IDENTIFIER: &str = \"{identifier}\";\n\
         pub const MODEL_FAMILY_FEATURE_ID: &str = \"{feature_id}\";\n\
         pub const MODEL_FAMILY_FIXTURE: &str = \"{fixture}\";\n\
         pub const MODEL_FAMILY: ModelFamilyDefinition = ModelFamilyDefinition {{}};\n\
         pub const MODEL_FAMILY_REGISTRATION: ModelFamilyRegistration = ModelFamilyRegistration {{\n\
             definition: &MODEL_FAMILY,\n\
             source_ordinal: {ordinal},\n\
         }};\n"
    )
}
