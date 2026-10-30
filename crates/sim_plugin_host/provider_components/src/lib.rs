pub mod common;
pub(crate) mod ideogram_response;

use common::ConformanceError;

pub struct DiscoveredProviderComponent {
    pub name: &'static str,
    pub fixture: &'static str,
    pub conformance: fn(&str) -> Result<(), ConformanceError>,
}

impl DiscoveredProviderComponent {
    pub fn run_conformance(&self) -> Result<(), ConformanceError> {
        (self.conformance)(self.fixture)
    }
}

include!(concat!(env!("OUT_DIR"), "/provider_registry.rs"));
