use crate::ExecutionFeatureDisposition;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeneratedExecutionCatalogRow {
    pub feature_id: &'static str,
    pub disposition: ExecutionFeatureDisposition,
}

pub static GENERATED_EXECUTION_CATALOG: [GeneratedExecutionCatalogRow; 119] = [
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-001",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-002",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-003",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-004",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-005",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-006",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-007",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-008",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-009",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-010",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-011",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-012",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-013",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-014",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-015",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-016",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-017",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-018",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-019",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-020",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-021",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-022",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-023",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-024",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-025",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-settings-localization-ui",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-026",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-settings-localization-ui",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-027",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-settings-localization-ui",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-028",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-settings-localization-ui",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-029",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-settings-localization-ui",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-030",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-settings-localization-ui",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-031",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-settings-localization-ui",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-032",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-settings-localization-ui",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-033",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-034",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-memory-planner",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-035",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-036",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-037",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-038",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-039",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-040",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-workflow-experience",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-041",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-042",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-workflow-experience",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-043",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-workflow-experience",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-044",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-process-diagnostics",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-045",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-process-diagnostics",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-046",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-047",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-048",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-049",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-workflow-experience",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-050",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-workflow-experience",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-051",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-052",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-053",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-054",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-055",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-056",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-057",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-058",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-059",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-060",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-061",
        disposition: ExecutionFeatureDisposition::Foundation {
            owner: "comfy-parity-native-graph",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-062",
        disposition: ExecutionFeatureDisposition::Foundation {
            owner: "comfy-parity-native-graph",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-063",
        disposition: ExecutionFeatureDisposition::Foundation {
            owner: "comfy-parity-native-graph",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-064",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-065",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-066",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-067",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-068",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-069",
        disposition: ExecutionFeatureDisposition::Foundation {
            owner: "comfy-parity-workflow-formats",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-070",
        disposition: ExecutionFeatureDisposition::Foundation {
            owner: "comfy-parity-native-graph",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-071",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-assets-editors-viewers",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-072",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-073",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-074",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-075",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-076",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-077",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-078",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-079",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-080",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-081",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-performance",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-082",
        disposition: ExecutionFeatureDisposition::Foundation {
            owner: "comfy-parity-workflow-formats",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-083",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-084",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-085",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-086",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-087",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-088",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-089",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-090",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-091",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-092",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-native-api-host",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-093",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-settings-localization-ui",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-094",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-095",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-096",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-097",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-098",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-099",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-100",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-101",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-102",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-103",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-104",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-105",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-106",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-107",
        disposition: ExecutionFeatureDisposition::Foundation {
            owner: "comfy-parity-native-graph",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-108",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-109",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-110",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-111",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-112",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-113",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-execution-e2e",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-114",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-workflow-experience",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-115",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-116",
        disposition: ExecutionFeatureDisposition::Native,
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-117",
        disposition: ExecutionFeatureDisposition::LaterOwned {
            owner: "comfy-parity-workflow-experience",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-118",
        disposition: ExecutionFeatureDisposition::Foundation {
            owner: "comfy-parity-workflow-formats",
        },
    },
    GeneratedExecutionCatalogRow {
        feature_id: "COMFY-QUEUE-119",
        disposition: ExecutionFeatureDisposition::SharedClosure {
            later_owner: "comfy-parity-native-api-host",
        },
    },
];
