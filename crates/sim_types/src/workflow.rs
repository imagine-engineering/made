use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NodeId(pub String);

impl From<&str> for NodeId {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptNode {
    pub class_type: String,
    pub inputs: BTreeMap<String, Value>,
    #[serde(default, flatten)]
    pub unknown: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ApiPrompt(pub BTreeMap<NodeId, PromptNode>);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PromptId(pub Uuid);

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AttemptId(pub Uuid);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PromptSubmission {
    pub prompt: ApiPrompt,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_id: Option<PromptId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub number: Option<f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub extra_data: BTreeMap<String, Value>,
    #[serde(flatten)]
    pub unknown: BTreeMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_keeps_textual_identifiers_and_unknown_fields() {
        let source = br#"{"prompt":{"001":{"class_type":"Native","inputs":{"link":["7",0]},"future":true}},"future_top":9}"#;
        let submission: PromptSubmission = serde_json::from_slice(source).expect("valid prompt");
        assert!(submission.prompt.0.contains_key(&NodeId::from("001")));
        assert_eq!(
            submission.prompt.0[&NodeId::from("001")].unknown["future"],
            true
        );
        assert_eq!(submission.unknown["future_top"], 9);
        let round_trip = serde_json::to_value(submission).expect("serializable prompt");
        assert_eq!(round_trip["prompt"]["001"]["inputs"]["link"][0], "7");
    }
}
