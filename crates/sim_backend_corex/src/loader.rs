use std::{
    env,
    ffi::OsString,
    marker::PhantomData,
    path::{Component, Path, PathBuf},
};

use thiserror::Error;

use crate::abi::{ABI_FLOOR, AbiManifest};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DiscoverySource {
    SimCoreXRoot,
    IxrtHome,
    SignedPackage,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveryRoot {
    pub source: DiscoverySource,
    pub path: PathBuf,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DiscoveryEnvironment {
    pub sim_corex_root: Option<OsString>,
    pub ixrt_home: Option<OsString>,
}

impl DiscoveryEnvironment {
    pub fn from_process() -> Self {
        Self {
            sim_corex_root: env::var_os("SIM_COREX_ROOT"),
            ixrt_home: env::var_os("IXRT_HOME"),
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct SignedPackageRoot<'certificate> {
    path: PathBuf,
    certificate_lifetime: PhantomData<&'certificate ()>,
}

impl<'certificate> SignedPackageRoot<'certificate> {
    /// Projects a package path already admitted by the canonical runtime trust owner.
    ///
    /// # Safety
    ///
    /// The caller must retain the signer-bound runtime package certificate and immutable package
    /// image for `path`. This constructor cannot turn metadata or a filesystem path into trust.
    pub unsafe fn from_runtime_verified_path<Certificate: ?Sized>(
        _certificate: &'certificate Certificate,
        path: PathBuf,
    ) -> Result<Self, CoreXLoadError> {
        validate_root(&path, DiscoverySource::SignedPackage)?;
        Ok(Self {
            path,
            certificate_lifetime: PhantomData,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveryPlan {
    roots: Vec<DiscoveryRoot>,
}

impl DiscoveryPlan {
    pub fn from_sources(
        environment: &DiscoveryEnvironment,
        signed_package_roots: &[SignedPackageRoot<'_>],
    ) -> Result<Self, CoreXLoadError> {
        Self::from_sources_for_target(env!("SIM_COREX_TARGET"), environment, signed_package_roots)
    }

    pub fn from_sources_for_target(
        target: &str,
        environment: &DiscoveryEnvironment,
        signed_package_roots: &[SignedPackageRoot<'_>],
    ) -> Result<Self, CoreXLoadError> {
        ensure_target(target)?;
        let mut roots = Vec::new();
        push_environment_root(
            &mut roots,
            environment.sim_corex_root.as_ref(),
            "SIM_COREX_ROOT",
            DiscoverySource::SimCoreXRoot,
        )?;
        push_environment_root(
            &mut roots,
            environment.ixrt_home.as_ref(),
            "IXRT_HOME",
            DiscoverySource::IxrtHome,
        )?;
        for package in signed_package_roots {
            push_unique_root(
                &mut roots,
                DiscoveryRoot {
                    source: DiscoverySource::SignedPackage,
                    path: package.path.clone(),
                },
            );
        }
        Ok(Self { roots })
    }

    pub fn roots(&self) -> &[DiscoveryRoot] {
        &self.roots
    }

    pub fn required_library_names(&self) -> [&'static str; 2] {
        ["libixblas.so", "libixrt.so"]
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegistryCertifiedImage {
    pub library_id: String,
    pub digest_sha256: String,
    pub abi_version: String,
    pub required_symbols: Vec<String>,
    pub unsafe_owner: String,
    pub retained_image_path: PathBuf,
}

#[derive(Debug, Eq, PartialEq)]
pub struct CertifiedCoreXImages<'certificate> {
    certificate_lifetime: PhantomData<&'certificate ()>,
}

impl<'certificate> CertifiedCoreXImages<'certificate> {
    /// Attempts to project registry certificates into the CoreX unsafe adapter.
    ///
    /// # Safety
    ///
    /// Every row must remain a direct projection of a live `NativeFfiRegistry` certificate and
    /// its retained immutable image. Until reviewed IXRT 0.8 and IXBLAS declarations exist, this
    /// constructor deliberately rejects every row before any loader or symbol operation.
    pub unsafe fn from_registry_certificates<Certificate: ?Sized>(
        _certificate: &'certificate Certificate,
        _images: impl IntoIterator<Item = RegistryCertifiedImage>,
    ) -> Result<Self, CoreXLoadError> {
        let manifest =
            AbiManifest::embedded().map_err(|error| CoreXLoadError::Manifest(error.to_string()))?;
        Err(CoreXLoadError::MissingReviewedAbiEvidence {
            missing: manifest
                .missing_evidence
                .iter()
                .map(|row| row.id.clone())
                .collect(),
        })
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum CoreXLoadError {
    #[error(
        "Iluvatar CoreX is unavailable on unsupported target {target}; expected x86_64-unknown-linux-gnu"
    )]
    UnsupportedTarget { target: String },
    #[error("{variable} is not valid Unicode")]
    InvalidEnvironment { variable: &'static str },
    #[error(
        "invalid {discovery_source:?} CoreX discovery root {path}: roots must be absolute and traversal-free"
    )]
    InvalidRoot {
        discovery_source: DiscoverySource,
        path: PathBuf,
    },
    #[error("CoreX ABI manifest is invalid: {0}")]
    Manifest(String),
    #[error(
        "CoreX execution is fail-closed because reviewed IXRT 0.8/IXBLAS ABI evidence is missing: {missing:?}"
    )]
    MissingReviewedAbiEvidence { missing: Vec<String> },
}

pub fn supported_target() -> bool {
    supported_target_name(env!("SIM_COREX_TARGET"))
}

pub fn supported_target_name(target: &str) -> bool {
    target == "x86_64-unknown-linux-gnu"
}

pub(crate) fn unavailable_reason() -> String {
    if supported_target() {
        format!(
            "Iluvatar CoreX {ABI_FLOOR} remains unavailable: libixrt.so and libixblas.so are named by the normative profile, but reviewed IXRT/IXBLAS headers, exact symbols, signatures, layouts, and header digests are absent; SIM_COREX_ROOT, IXRT_HOME, signed package metadata, and NativeFfiRegistry certificates cannot self-certify an unknown ABI"
        )
    } else {
        format!(
            "Iluvatar CoreX {ABI_FLOOR} unsupported target {}; expected x86_64-unknown-linux-gnu, reviewed IXRT/IXBLAS headers, and NativeFfiRegistry certification",
            env!("SIM_COREX_TARGET")
        )
    }
}

fn ensure_target(target: &str) -> Result<(), CoreXLoadError> {
    if !supported_target_name(target) {
        return Err(CoreXLoadError::UnsupportedTarget {
            target: target.to_owned(),
        });
    }
    Ok(())
}

fn push_environment_root(
    roots: &mut Vec<DiscoveryRoot>,
    value: Option<&OsString>,
    variable: &'static str,
    source: DiscoverySource,
) -> Result<(), CoreXLoadError> {
    let Some(value) = value else {
        return Ok(());
    };
    let path = value
        .to_str()
        .ok_or(CoreXLoadError::InvalidEnvironment { variable })?;
    if path.is_empty() {
        return Ok(());
    }
    let root = DiscoveryRoot {
        source,
        path: PathBuf::from(path),
    };
    validate_root(&root.path, source)?;
    push_unique_root(roots, root);
    Ok(())
}

fn push_unique_root(roots: &mut Vec<DiscoveryRoot>, root: DiscoveryRoot) {
    if !roots.iter().any(|existing| existing.path == root.path) {
        roots.push(root);
    }
}

fn validate_root(path: &Path, source: DiscoverySource) -> Result<(), CoreXLoadError> {
    if !path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, Component::ParentDir | Component::CurDir))
    {
        return Err(CoreXLoadError::InvalidRoot {
            discovery_source: source,
            path: path.to_owned(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_order_is_exact_and_deduplicated() -> Result<(), CoreXLoadError> {
        let certificate = ();
        let package = unsafe {
            SignedPackageRoot::from_runtime_verified_path(
                &certificate,
                PathBuf::from("/opt/corex-signed"),
            )?
        };
        let plan = DiscoveryPlan::from_sources_for_target(
            "x86_64-unknown-linux-gnu",
            &DiscoveryEnvironment {
                sim_corex_root: Some(OsString::from("/opt/corex-explicit")),
                ixrt_home: Some(OsString::from("/opt/corex-signed")),
            },
            &[package],
        )?;
        assert_eq!(
            plan.roots()
                .iter()
                .map(|root| root.source)
                .collect::<Vec<_>>(),
            [DiscoverySource::SimCoreXRoot, DiscoverySource::IxrtHome,]
        );
        assert_eq!(
            plan.required_library_names(),
            ["libixblas.so", "libixrt.so"]
        );
        Ok(())
    }

    #[test]
    fn discovery_rejects_unsupported_targets_and_relative_roots() {
        let unsupported = DiscoveryPlan::from_sources_for_target(
            "aarch64-unknown-linux-gnu",
            &DiscoveryEnvironment::default(),
            &[],
        );
        assert!(matches!(
            unsupported,
            Err(CoreXLoadError::UnsupportedTarget { .. })
        ));

        let relative = DiscoveryPlan::from_sources_for_target(
            "x86_64-unknown-linux-gnu",
            &DiscoveryEnvironment {
                sim_corex_root: Some(OsString::from("relative/corex")),
                ixrt_home: None,
            },
            &[],
        );
        assert!(matches!(
            relative,
            Err(CoreXLoadError::InvalidRoot {
                discovery_source: DiscoverySource::SimCoreXRoot,
                ..
            })
        ));
    }

    #[test]
    fn registry_projection_cannot_bypass_missing_reviewed_abi() {
        let certificate = ();
        let image = RegistryCertifiedImage {
            library_id: "ixrt".to_owned(),
            digest_sha256: "0".repeat(64),
            abi_version: ABI_FLOOR.to_owned(),
            required_symbols: Vec::new(),
            unsafe_owner: "sim_backend_corex::loader".to_owned(),
            retained_image_path: PathBuf::from("/proc/self/fd/42"),
        };
        let result =
            unsafe { CertifiedCoreXImages::from_registry_certificates(&certificate, [image]) };
        assert!(matches!(
            result,
            Err(CoreXLoadError::MissingReviewedAbiEvidence { missing })
                if missing.contains(&"ixrt-0.8-symbol-signatures".to_owned())
                    && missing.contains(&"ixblas-symbol-signatures".to_owned())
        ));
    }
}
