use feature_flags::{FeatureFlagAppExt as _, RustNativeProfilingExperimentalFeatureFlag};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
#[cfg(target_os = "macos")]
use std::time::Duration;
use std::{collections::VecDeque, path::PathBuf};

#[cfg(target_os = "macos")]
use anyhow::Context as _;
use anyhow::Result;
use collections::{HashMap, HashSet};
use futures::{FutureExt as _, StreamExt as _};
use gpui::{App, AppContext as _, AsyncApp, Context, Entity, Subscription, Task};
#[cfg(any(
    test,
    feature = "test-support",
    all(target_os = "macos", feature = "cargo-workspace")
))]
use parking_lot::Mutex;
use remote::{ConnectionState, RemoteClient, RemoteClientEvent};
use rpc::proto::Message as _;
use rpc::{AnyProtoClient, TypedEnvelope, proto};
#[cfg(target_os = "macos")]
use util::ResultExt as _;

use crate::{
    ProjectPath,
    profile_results::{ProfileHost, ProfileInput, load_samply_profile},
    profile_results_wire::{
        PROFILE_PROTOCOL_VERSION, profile_import_paths, profile_result_pages,
        valid_profile_task_run_id,
    },
    trusted_worktrees::TrustedWorktrees,
    worktree_store::{WorktreeStore, WorktreeStoreEvent},
};

const MAX_HOST_PROFILE_IMPORTS: usize = 4;
static NEXT_PROXY_IMPORT_ID: AtomicU64 = AtomicU64::new(1 << 63);
#[cfg(target_os = "macos")]
const MAX_HOST_PROFILE_TASKS: usize = 4;
type ImportKey = (proto::PeerId, u64);
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
enum ProfileTaskActor {
    Peer(proto::PeerId),
    #[cfg(target_os = "macos")]
    Local(gpui::EntityId),
}
type ProfileTaskKey = (ProfileTaskActor, String);

#[derive(Clone, PartialEq)]
struct ProfileTaskVersion {
    entry: worktree::Entry,
    root_inode: u64,
}

struct NativePairTaskRun {
    #[cfg(target_os = "macos")]
    checked_invocation: Option<Arc<NativePairInvocation>>,
    paths: [ProjectPath; 2],
    versions: Option<[ProfileTaskVersion; 2]>,
    fingerprints: Option<[[u8; 32]; 2]>,
    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    capture_admission: Option<(NativeCaptureAuthority, ProfileTaskHelper)>,
}

enum ProfileTaskPayload {
    #[cfg(target_os = "macos")]
    NativePair(NativePairTaskRun),
    #[cfg(target_os = "macos")]
    BuildReport(BuildReportTaskRun),
}

#[derive(Clone, Eq, PartialEq)]
struct ProfileTaskHelper {
    path: PathBuf,
    mode: ProfileTaskHelperMode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProfileTaskHelperMode {
    RemoteServer,
    #[cfg(target_os = "macos")]
    LocalApplication,
}

struct ProfileTaskRun {
    completion: watch::Sender<()>,
    generation: u64,
    payload: ProfileTaskPayload,
    failed: bool,
    cancellation: Arc<AtomicBool>,
    stop: async_channel::Sender<()>,
    _expiry: Task<()>,
    #[cfg(target_os = "macos")]
    observation: Option<Task<()>>,
}

impl ProfileTaskRun {
    fn native(&self) -> Option<&NativePairTaskRun> {
        // Matching the place also exhausts the uninhabited payload on other platforms.
        match self.payload {
            #[cfg(target_os = "macos")]
            ProfileTaskPayload::NativePair(ref native) => Some(native),
            #[cfg(target_os = "macos")]
            ProfileTaskPayload::BuildReport(_) => None,
        }
    }

    #[cfg(target_os = "macos")]
    fn native_mut(&mut self) -> Option<&mut NativePairTaskRun> {
        match &mut self.payload {
            ProfileTaskPayload::NativePair(native) => Some(native),
            #[cfg(target_os = "macos")]
            ProfileTaskPayload::BuildReport(_) => None,
        }
    }
}

#[cfg(target_os = "macos")]
const MAX_BUILD_REPORT_RESERVED_BYTES: usize = 8 * 1024 * 1024;

#[cfg(target_os = "macos")]
struct BuildReportReservation {
    budget: Arc<std::sync::atomic::AtomicUsize>,
    limit: usize,
    reserved_bytes: usize,
    retirement_sender: async_channel::Sender<()>,
    retirement: async_channel::Receiver<()>,
}

#[cfg(target_os = "macos")]
impl BuildReportReservation {
    fn reserve(budget: Arc<std::sync::atomic::AtomicUsize>, limit: usize) -> Result<Arc<Self>> {
        Self::reserve_for_invocation(budget, limit, false)
    }

    fn reserve_for_invocation(
        budget: Arc<std::sync::atomic::AtomicUsize>,
        limit: usize,
        checked: bool,
    ) -> Result<Arc<Self>> {
        anyhow::ensure!(
            (1..=util::macos_task_completion::MAX_TASK_STDOUT_REPORT_BYTES).contains(&limit),
            "Build report byte limit is invalid"
        );
        let reserved_bytes = limit
            .checked_add(if checked {
                util::macos_task_completion::TASK_BUILD_INVOCATION_RESERVED_BYTES
            } else {
                0
            })
            .ok_or_else(|| anyhow::anyhow!("Build reservation overflow"))?;
        Self::reserve_bytes(budget, limit, reserved_bytes)
    }

    #[cfg(any(feature = "cargo-workspace", test))]
    fn reserve_native_invocation(budget: Arc<std::sync::atomic::AtomicUsize>) -> Result<Arc<Self>> {
        Self::reserve_bytes(
            budget,
            0,
            util::macos_task_completion::TASK_BUILD_INVOCATION_RESERVED_BYTES,
        )
    }

    fn reserve_bytes(
        budget: Arc<std::sync::atomic::AtomicUsize>,
        limit: usize,
        reserved_bytes: usize,
    ) -> Result<Arc<Self>> {
        budget
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |used| {
                used.checked_add(reserved_bytes)
                    .filter(|total| *total <= MAX_BUILD_REPORT_RESERVED_BYTES)
            })
            .map_err(|_| anyhow::anyhow!("Too many reserved Build report bytes"))?;
        let (retirement_sender, retirement) = async_channel::bounded(1);
        Ok(Arc::new(Self {
            budget,
            limit,
            reserved_bytes,
            retirement_sender,
            retirement,
        }))
    }
}

#[cfg(target_os = "macos")]
impl Drop for BuildReportReservation {
    fn drop(&mut self) {
        self.budget
            .fetch_sub(self.reserved_bytes, Ordering::Relaxed);
        self.retirement_sender.close();
    }
}

#[cfg(target_os = "macos")]
struct NativePairInvocation {
    expected: util::macos_task_completion::TaskBuildInvocation,
    reservation: Arc<BuildReportReservation>,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum BuildReportMode {
    Stdout,
    CheckedBuild,
    CheckedCargoRuntime,
}

#[cfg(target_os = "macos")]
struct BuildReportRuntimeAdmission {
    image: LocalNativeTaskHelperImage,
    #[cfg(feature = "cargo-workspace")]
    lease: crate::cargo_workspace_store::NativeCargoRuntimeAdmissionLease,
}

#[cfg(target_os = "macos")]
struct BuildReportTaskRun {
    #[cfg(feature = "cargo-workspace")]
    observation_arm: Option<async_channel::Sender<()>>,
    runtime_admission: Option<Arc<BuildReportRuntimeAdmission>>,
    mode: BuildReportMode,
    expected_invocation: Option<util::macos_task_completion::TaskBuildInvocation>,
    observed_invocation: Option<Arc<util::macos_task_completion::ObservedTaskBuildInvocation>>,
    observed_cargo_runtime: Option<Arc<util::macos_task_completion::ObservedCargoRuntime>>,
    helper: ProfileTaskHelper,
    identity: Option<crate::profile_results::TaskHelperIdentity>,
    report: Option<Arc<Vec<u8>>>,
    fingerprint: Option<[u8; 32]>,
    reservation: Arc<BuildReportReservation>,
}

#[cfg(target_os = "macos")]
struct BuildReportPreparation {
    mode: BuildReportMode,
    expected_invocation: Option<util::macos_task_completion::TaskBuildInvocation>,
    reservation: Arc<BuildReportReservation>,
}

#[cfg(target_os = "macos")]
impl BuildReportPreparation {
    fn into_parts(
        self,
    ) -> (
        Arc<BuildReportReservation>,
        Option<util::macos_task_completion::TaskBuildInvocation>,
        BuildReportMode,
    ) {
        (self.reservation, self.expected_invocation, self.mode)
    }
}

#[cfg(target_os = "macos")]
struct BuildReportObservation {
    armed: Option<async_channel::Receiver<()>>,
    runtime_admission: Option<Arc<BuildReportRuntimeAdmission>>,
    mode: BuildReportMode,
    expected_invocation: Option<util::macos_task_completion::TaskBuildInvocation>,
    key: ProfileTaskKey,
    generation: u64,
    identity: crate::profile_results::TaskHelperIdentity,
    listener: util::macos_task_completion::TaskCompletionListener,
    stopped: async_channel::Receiver<()>,
    filesystem: Arc<dyn fs::Fs>,
    cancellation: Arc<AtomicBool>,
    limit: usize,
    // Keep the quota alive until all fields with retained report state have been dropped.
    reservation: Arc<BuildReportReservation>,
}

#[cfg(target_os = "macos")]
struct ReservedBuildReport {
    cargo_runtime: Option<Arc<util::macos_task_completion::ObservedCargoRuntime>>,
    bytes: Arc<Vec<u8>>,
    fingerprint: [u8; 32],
    invocation: Option<Arc<util::macos_task_completion::ObservedTaskBuildInvocation>>,
    reservation: Arc<BuildReportReservation>,
}

#[cfg(target_os = "macos")]
pub struct LocalNativeTaskHelperImage {
    owner_helper: ProfileTaskHelper,
    purpose: NativeTaskHelperImagePurpose,
    project: gpui::EntityId,
    store: gpui::EntityId,
    epoch: Arc<()>,
    helper: ProfileTaskHelper,
    identity: crate::profile_results::TaskHelperIdentity,
}

#[cfg(target_os = "macos")]
#[derive(Clone)]
enum NativeTaskHelperImagePurpose {
    Application,
    CargoRuntime(Arc<NativeRuntimeHelperPolicy>),
}

#[cfg(target_os = "macos")]
struct NativeRuntimeHelperPolicy {
    app_identity: crate::profile_results::TaskHelperIdentity,
    architectures: Vec<u32>,
    signer: native_runtime_image_policy::Signer,
    system_closure: native_runtime_image_policy::SystemClosure,
}

#[cfg(all(target_os = "macos", any(feature = "cargo-workspace", test)))]
pub(crate) fn validate_native_runtime_loader_environment(
    environment: &std::collections::BTreeMap<String, String>,
) -> Result<()> {
    anyhow::ensure!(
        !environment
            .iter()
            .any(|(key, value)| key.starts_with("DYLD_") && !value.is_empty()),
        "Cargo runtime loader environment requires unsupported pre-main image qualification"
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn native_runtime_observer_path(process: &std::path::Path) -> Result<PathBuf> {
    anyhow::ensure!(
        process.is_absolute()
            && process
                .to_str()
                .is_some_and(|value| value.len() <= 4096 && !value.chars().any(char::is_control))
            && !process.components().any(|component| !matches!(
                component,
                std::path::Component::RootDir | std::path::Component::Normal(_)
            )),
        "Native runtime process route is invalid"
    );
    let directory = process
        .parent()
        .context("Native runtime process has no parent")?;
    let contents = directory
        .parent()
        .context("Native runtime process has no Contents owner")?;
    let bundle = contents
        .parent()
        .context("Native runtime process has no App owner")?;
    anyhow::ensure!(
        directory.file_name().is_some_and(|name| name == "MacOS")
            && contents.file_name().is_some_and(|name| name == "Contents")
            && bundle
                .extension()
                .is_some_and(|extension| extension == "app")
            && process
                .file_name()
                .is_some_and(|name| name != "native-cargo-runtime-observer"),
        "Native runtime helper requires the actual installed App process"
    );
    Ok(directory.join("native-cargo-runtime-observer"))
}

#[cfg(target_os = "macos")]
mod native_runtime_image_policy {
    use super::*;
    use std::{ffi::c_void, io::Read as _, os::unix::ffi::OsStrExt as _, path::Path};
    type Reference = *const c_void;
    struct Owned(Reference);
    impl Drop for Owned {
        fn drop(&mut self) {
            unsafe { CFRelease(self.0) };
        }
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFRelease(value: Reference);
        fn CFGetTypeID(value: Reference) -> usize;
        fn CFStringGetTypeID() -> usize;
        fn CFDictionaryGetTypeID() -> usize;
        fn CFNumberGetTypeID() -> usize;
        fn CFBooleanGetTypeID() -> usize;
        fn CFBooleanGetValue(value: Reference) -> u8;
        fn CFStringCreateWithBytes(
            allocator: Reference,
            bytes: *const u8,
            length: isize,
            encoding: u32,
            external: u8,
        ) -> Reference;
        fn CFStringGetLength(value: Reference) -> isize;
        fn CFStringGetCString(
            value: Reference,
            buffer: *mut i8,
            capacity: isize,
            encoding: u32,
        ) -> u8;
        fn CFURLCreateFromFileSystemRepresentation(
            allocator: Reference,
            bytes: *const u8,
            length: isize,
            directory: u8,
        ) -> Reference;
        fn CFDictionaryCreate(
            allocator: Reference,
            keys: *const Reference,
            values: *const Reference,
            count: isize,
            key_callbacks: Reference,
            value_callbacks: Reference,
        ) -> Reference;
        fn CFDictionaryGetCount(dictionary: Reference) -> isize;
        fn CFDictionaryGetValue(dictionary: Reference, key: Reference) -> Reference;
        fn CFNumberGetValue(number: Reference, kind: isize, output: *mut c_void) -> u8;
    }
    #[link(name = "Security", kind = "framework")]
    unsafe extern "C" {
        static kSecCodeAttributeArchitecture: Reference;
        static kSecCodeInfoFlags: Reference;
        static kSecCodeInfoTeamIdentifier: Reference;
        static kSecCodeInfoEntitlementsDict: Reference;
        fn SecStaticCodeCreateWithPathAndAttributes(
            path: Reference,
            flags: u32,
            attributes: Reference,
            output: *mut Reference,
        ) -> i32;
        fn SecStaticCodeCheckValidity(code: Reference, flags: u32, requirement: Reference) -> i32;
        fn SecRequirementCreateWithString(
            text: Reference,
            flags: u32,
            output: *mut Reference,
        ) -> i32;
        fn SecCodeCopySigningInformation(
            code: Reference,
            flags: u32,
            output: *mut Reference,
        ) -> i32;
    }
    const UTF8: u32 = 0x0800_0100;
    pub(super) const MAX_IMAGE_BYTES: usize = 16 * 1024 * 1024;
    const ARM64: u32 = 0x0100_000c;
    const X86_64: u32 = 0x0100_0007;
    #[derive(Eq, PartialEq)]
    pub(super) enum Signer {
        DeveloperId(String),
        AdHocDevelopment,
    }
    fn owned(reference: Reference) -> Result<Owned> {
        anyhow::ensure!(
            !reference.is_null(),
            "Native runtime signing object is unavailable"
        );
        Ok(Owned(reference))
    }
    fn string(value: &str) -> Result<Owned> {
        anyhow::ensure!(
            value.len() <= 4096,
            "Native runtime signing input exceeds its limit"
        );
        owned(unsafe {
            CFStringCreateWithBytes(
                std::ptr::null(),
                value.as_ptr(),
                value.len() as isize,
                UTF8,
                0,
            )
        })
    }
    fn team(value: Reference) -> Result<String> {
        anyhow::ensure!(
            !value.is_null()
                && unsafe {
                    CFGetTypeID(value) == CFStringGetTypeID() && CFStringGetLength(value) == 10
                },
            "Native runtime signing Team is unavailable"
        );
        let mut buffer = [0_i8; 32];
        anyhow::ensure!(
            unsafe { CFStringGetCString(value, buffer.as_mut_ptr(), buffer.len() as isize, UTF8) }
                != 0,
            "Native runtime signing Team is invalid"
        );
        let bytes = buffer
            .iter()
            .take_while(|byte| **byte != 0)
            .map(|byte| *byte as u8)
            .collect::<Vec<_>>();
        anyhow::ensure!(
            bytes.len() == 10
                && bytes
                    .iter()
                    .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit()),
            "Native runtime signing Team is invalid"
        );
        String::from_utf8(bytes)
            .map_err(|_| anyhow::anyhow!("Native runtime signing Team is invalid"))
    }
    pub(super) fn validate_code(
        path: &Path,
        architecture: Option<u32>,
        require_dyld: bool,
    ) -> Result<Signer> {
        let path_bytes = path.as_os_str().as_bytes();
        anyhow::ensure!(
            path.is_absolute() && path_bytes.len() <= 4096,
            "Native runtime signing path is invalid"
        );
        let url = owned(unsafe {
            CFURLCreateFromFileSystemRepresentation(
                std::ptr::null(),
                path_bytes.as_ptr(),
                path_bytes.len() as isize,
                0,
            )
        })?;
        let architecture_name = architecture
            .map(|architecture| match architecture {
                ARM64 => Ok("arm64"),
                X86_64 => Ok("x86_64"),
                _ => Err(anyhow::anyhow!(
                    "Native runtime architecture is unsupported"
                )),
            })
            .transpose()?
            .map(string)
            .transpose()?;
        let attributes = architecture_name
            .as_ref()
            .map(|architecture| {
                let key = unsafe { kSecCodeAttributeArchitecture };
                owned(unsafe {
                    CFDictionaryCreate(
                        std::ptr::null(),
                        &key,
                        &architecture.0,
                        1,
                        std::ptr::null(),
                        std::ptr::null(),
                    )
                })
            })
            .transpose()?;
        let mut code = std::ptr::null();
        anyhow::ensure!(
            unsafe {
                SecStaticCodeCreateWithPathAndAttributes(
                    url.0,
                    0,
                    attributes
                        .as_ref()
                        .map_or(std::ptr::null(), |attributes| attributes.0),
                    &mut code,
                )
            } == 0,
            "Native runtime signature cannot be inspected"
        );
        let code = owned(code)?;
        let mut info = std::ptr::null();
        anyhow::ensure!(
            unsafe { SecCodeCopySigningInformation(code.0, 1 << 1, &mut info) } == 0,
            "Native runtime signing facts are unavailable"
        );
        let info = owned(info)?;
        anyhow::ensure!(
            unsafe { CFGetTypeID(info.0) == CFDictionaryGetTypeID() },
            "Native runtime signing facts are invalid"
        );
        let flags_value = unsafe { CFDictionaryGetValue(info.0, kSecCodeInfoFlags) };
        anyhow::ensure!(
            !flags_value.is_null() && unsafe { CFGetTypeID(flags_value) == CFNumberGetTypeID() },
            "Native runtime signature flags are unavailable"
        );
        let mut signature_flags = 0_i32;
        anyhow::ensure!(
            unsafe { CFNumberGetValue(flags_value, 3, (&mut signature_flags as *mut i32).cast()) }
                != 0,
            "Native runtime signature flags are invalid"
        );
        let team_value = unsafe { CFDictionaryGetValue(info.0, kSecCodeInfoTeamIdentifier) };
        let requirement = if signature_flags & 0x2 != 0 {
            anyhow::ensure!(
                team_value.is_null(),
                "Ad-hoc runtime image cannot claim a signing Team"
            );
            None
        } else {
            let requirement_text = string(
                "anchor apple generic and certificate leaf[field.1.2.840.113635.100.6.1.13] exists",
            )?;
            let mut requirement = std::ptr::null();
            anyhow::ensure!(
                unsafe { SecRequirementCreateWithString(requirement_text.0, 0, &mut requirement) }
                    == 0,
                "Native runtime signing requirement is unavailable"
            );
            Some(owned(requirement)?)
        };
        // Static validity authenticates the image, not shipping or loaded-code certification.
        let flags = (1 << 29) | (1 << 4) | (1 << 0);
        anyhow::ensure!(
            unsafe {
                SecStaticCodeCheckValidity(
                    code.0,
                    flags,
                    requirement
                        .as_ref()
                        .map_or(std::ptr::null(), |requirement| requirement.0),
                )
            } == 0,
            "Native runtime image signature is not qualified"
        );
        let signer = if requirement.is_some() {
            Signer::DeveloperId(team(team_value)?)
        } else {
            Signer::AdHocDevelopment
        };
        if require_dyld {
            anyhow::ensure!(
                signature_flags & 0x10000 != 0,
                "Native runtime helper requires hardened runtime"
            );
            let entitlements =
                unsafe { CFDictionaryGetValue(info.0, kSecCodeInfoEntitlementsDict) };
            anyhow::ensure!(
                !entitlements.is_null()
                    && unsafe {
                        CFGetTypeID(entitlements) == CFDictionaryGetTypeID()
                            && CFDictionaryGetCount(entitlements) == 1
                    },
                "Native runtime helper requires exactly its approved entitlement"
            );
            let key = string("com.apple.security.cs.allow-dyld-environment-variables")?;
            let allow_dyld = unsafe { CFDictionaryGetValue(entitlements, key.0) };
            anyhow::ensure!(
                !allow_dyld.is_null()
                    && unsafe {
                        CFGetTypeID(allow_dyld) == CFBooleanGetTypeID()
                            && CFBooleanGetValue(allow_dyld) != 0
                    },
                "Native runtime helper DYLD entitlement is unavailable"
            );
        }
        Ok(signer)
    }
    fn word(bytes: &[u8], offset: usize, big: bool) -> Result<u32> {
        let raw: [u8; 4] = bytes
            .get(
                offset
                    ..offset
                        .checked_add(4)
                        .context("Native runtime image offset overflow")?,
            )
            .context("Native runtime image header is truncated")?
            .try_into()?;
        Ok(if big {
            u32::from_be_bytes(raw)
        } else {
            u32::from_le_bytes(raw)
        })
    }
    fn thin(bytes: &[u8]) -> Result<u32> {
        anyhow::ensure!(
            word(bytes, 0, false)? == 0xfeed_facf,
            "Native runtime helper requires a supported Mach-O executable"
        );
        let architecture = word(bytes, 4, false)?;
        let subtype = word(bytes, 8, false)?;
        anyhow::ensure!(
            (architecture == ARM64 && subtype == 0)
                || (architecture == X86_64 && subtype & 0x7fff_ffff == 3),
            "Native runtime helper subarchitecture is unsupported"
        );
        anyhow::ensure!(
            matches!(architecture, ARM64 | X86_64) && word(bytes, 12, false)? == 2,
            "Native runtime helper architecture or file type is unsupported"
        );
        let count = word(bytes, 16, false)? as usize;
        let size = word(bytes, 20, false)? as usize;
        anyhow::ensure!(
            count <= 4096 && size <= 256 * 1024,
            "Native runtime load commands exceed their limit"
        );
        let commands = bytes
            .get(
                32..32_usize
                    .checked_add(size)
                    .context("Native runtime load command overflow")?,
            )
            .context("Native runtime load commands are truncated")?;
        let mut offset = 0;
        let mut has_dynamic_linker = false;
        for _ in 0..count {
            let command = word(commands, offset, false)?;
            anyhow::ensure!(
                matches!(
                    command,
                    0x2 | 0xb
                        | 0xc
                        | 0xe
                        | 0x19
                        | 0x1b
                        | 0x1d
                        | 0x20
                        | 0x22
                        | 0x24
                        | 0x26
                        | 0x29
                        | 0x2a
                        | 0x2b
                        | 0x2d
                        | 0x2e
                        | 0x31
                        | 0x32
                        | 0x36
                        | 0x8000_0018
                        | 0x8000_001f
                        | 0x8000_0022
                        | 0x8000_0023
                        | 0x8000_0028
                        | 0x8000_0033
                        | 0x8000_0034
                ),
                "Native runtime load command requires unsupported policy"
            );
            let length = word(commands, offset + 4, false)? as usize;
            anyhow::ensure!(
                length >= 8 && length.is_multiple_of(8),
                "Native runtime load command is invalid"
            );
            let record = commands
                .get(
                    offset
                        ..offset
                            .checked_add(length)
                            .context("Native runtime load command overflow")?,
                )
                .context("Native runtime load command is truncated")?;
            anyhow::ensure!(
                command != 0x8000_001c && command != 0xf && command != 0x27,
                "Native runtime helper cannot use runtime library search paths or ID dylibs"
            );
            anyhow::ensure!(
                !matches!(command, 0x8000_0018 | 0x20 | 0x8000_0023),
                "Native runtime optional or delayed dependency requires unsupported closure policy"
            );
            if matches!(command, 0xc | 0x8000_001f | 0xe) {
                let name_offset = word(record, 8, false)? as usize;
                anyhow::ensure!(
                    name_offset >= if command == 0xe { 12 } else { 24 },
                    "Native runtime dependency offset is invalid"
                );
                let name = record
                    .get(name_offset..)
                    .context("Native runtime dependency name is truncated")?;
                let end = name
                    .iter()
                    .position(|byte| *byte == 0)
                    .context("Native runtime dependency name is unterminated")?;
                anyhow::ensure!(end <= 4096, "Native runtime dependency exceeds its limit");
                let name = std::str::from_utf8(&name[..end])
                    .context("Native runtime dependency is not UTF-8")?;
                let path = Path::new(name);
                anyhow::ensure!(
                    (name.starts_with("/usr/lib/") || name.starts_with("/System/Library/"))
                        && name
                            .split('/')
                            .skip(1)
                            .all(|part| !part.is_empty() && part != "." && part != "..")
                        && !name.chars().any(char::is_control)
                        && !path.components().any(|component| !matches!(
                            component,
                            std::path::Component::RootDir | std::path::Component::Normal(_)
                        )),
                    "Native runtime helper requires ordinary absolute system dependencies"
                );
                if command == 0xe {
                    anyhow::ensure!(
                        !has_dynamic_linker && name == "/usr/lib/dyld",
                        "Native runtime helper dynamic linker is unsupported or ambiguous"
                    );
                    has_dynamic_linker = true;
                }
            }
            offset += length;
        }
        anyhow::ensure!(
            offset == commands.len(),
            "Native runtime load command accounting is inconsistent"
        );
        Ok(architecture)
    }
    pub(super) fn architectures(bytes: &[u8]) -> Result<Vec<u32>> {
        anyhow::ensure!(
            !bytes.is_empty() && bytes.len() <= MAX_IMAGE_BYTES,
            "Native runtime image exceeds its byte limit"
        );
        if word(bytes, 0, false)? == 0xfeed_facf {
            let architecture = thin(bytes)?;
            #[cfg(target_arch = "aarch64")]
            anyhow::ensure!(
                architecture == ARM64,
                "Native runtime helper does not support the current process architecture"
            );
            #[cfg(target_arch = "x86_64")]
            anyhow::ensure!(
                architecture == X86_64,
                "Native runtime helper does not support the current process architecture"
            );
            #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
            anyhow::bail!("Native runtime process architecture is unsupported");
            Ok(vec![architecture])
        } else {
            universal_architectures(bytes)
        }
    }
    fn universal_architectures(bytes: &[u8]) -> Result<Vec<u32>> {
        anyhow::ensure!(
            word(bytes, 0, true)? == 0xcafe_babe,
            "Native runtime universal image format is unsupported"
        );
        let count = word(bytes, 4, true)? as usize;
        anyhow::ensure!(
            (1..=2).contains(&count),
            "Native runtime universal architecture count is unsupported"
        );
        let header_end = 8 + count * 20;
        let mut result = Vec::new();
        let mut ranges = Vec::new();
        for index in 0..count {
            let record = 8 + index * 20;
            let architecture = word(bytes, record, true)?;
            let offset = word(bytes, record + 8, true)? as usize;
            let size = word(bytes, record + 12, true)? as usize;
            let alignment = word(bytes, record + 16, true)?;
            let end = offset
                .checked_add(size)
                .context("Native runtime slice offset overflow")?;
            anyhow::ensure!(
                offset >= header_end
                    && size >= 32
                    && alignment <= 30
                    && offset.is_multiple_of(1_usize << alignment)
                    && ranges
                        .iter()
                        .all(|(start, finish)| end <= *start || offset >= *finish),
                "Native runtime universal slice is invalid"
            );
            let slice = bytes
                .get(offset..end)
                .context("Native runtime universal slice is truncated")?;
            anyhow::ensure!(
                thin(slice)? == architecture
                    && word(bytes, record + 4, true)? == word(slice, 8, false)?
                    && !result.contains(&architecture),
                "Native runtime universal architecture is ambiguous"
            );
            result.push(architecture);
            ranges.push((offset, end));
        }
        #[cfg(target_arch = "aarch64")]
        let current = ARM64;
        #[cfg(target_arch = "x86_64")]
        let current = X86_64;
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        anyhow::bail!("Native runtime process architecture is unsupported");
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        anyhow::ensure!(
            result.contains(&current),
            "Native runtime helper does not support the current process architecture"
        );
        Ok(result)
    }
    const MAX_CACHE_OUTPUT_BYTES: usize = 16 * 1024 * 1024;
    const MAX_CACHE_IMAGES: usize = 8192;
    const MAX_CACHE_DEPENDENCIES: usize = 131_072;
    const MAX_CLOSURE_IMAGES: usize = 1024;
    const MAX_CLOSURE_DEPENDENCIES: usize = 16_384;
    const MAX_DISK_IMAGE_BYTES: u64 = 64 * 1024 * 1024;

    #[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
    struct SystemDependency {
        path: String,
        attributes: u8,
    }
    #[derive(Clone, Debug, Eq, PartialEq)]
    struct SystemImageFacts {
        path: String,
        architecture: String,
        uuid: uuid::Uuid,
        dependencies: Vec<SystemDependency>,
    }
    enum SystemImageOrigin {
        LoadedCache {
            address: u64,
        },
        CacheText {
            address: u64,
            text_size: u64,
            canonical_path: String,
        },
        Cache,
        Disk(crate::profile_results::TaskHelperIdentity),
    }
    struct ReviewedSystemImage {
        facts: SystemImageFacts,
        origin: SystemImageOrigin,
    }
    enum SystemClosureInspection {
        ObservedLoaded,
        ObservedCacheText,
        Tool {
            discovery: crate::profile_results::TaskHelperIdentity,
            tool: crate::profile_results::TaskHelperIdentity,
        },
    }
    pub(super) struct SystemClosure {
        inspection: SystemClosureInspection,
        operating_system: String,
        cache: ActiveCacheIdentity,
        mapped_version: u32,
        mapped_size: u64,
        mapped_epoch: u64,
        images: Vec<ReviewedSystemImage>,
        absent_weak: std::collections::BTreeSet<String>,
    }

    #[derive(Clone, Eq, PartialEq)]
    struct ActiveCacheIdentity {
        base: u64,
        length: u64,
        uuid: uuid::Uuid,
    }
    struct LoadedCachedImage {
        address: u64,
        facts: SystemImageFacts,
    }
    struct LoadedCacheSnapshot {
        cache: ActiveCacheIdentity,
        version: u32,
        size: u64,
        timestamp: u64,
        images: std::collections::BTreeMap<String, LoadedCachedImage>,
    }
    unsafe extern "C" {
        static mach_task_self_: u32;
        fn task_info(task: u32, flavor: i32, output: *mut u32, count: *mut u32) -> i32;
        fn mach_vm_read_overwrite(
            task: u32,
            address: u64,
            size: u64,
            output: u64,
            copied: *mut u64,
        ) -> i32;
    }
    type CacheRangeFunction = unsafe extern "C" fn(*mut usize) -> *const c_void;
    type CacheUuidFunction = unsafe extern "C" fn(*mut u8) -> bool;
    type ImmutableMemoryFunction = unsafe extern "C" fn(*const c_void, usize) -> bool;
    struct CacheSdkFunctions {
        range: CacheRangeFunction,
        uuid: CacheUuidFunction,
        immutable: ImmutableMemoryFunction,
    }
    impl CacheSdkFunctions {
        fn current() -> Result<Self> {
            let range = unsafe {
                libc::dlsym(
                    (-2_isize) as *mut c_void,
                    c"_dyld_get_shared_cache_range".as_ptr(),
                )
            };
            let uuid = unsafe {
                libc::dlsym(
                    (-2_isize) as *mut c_void,
                    c"_dyld_get_shared_cache_uuid".as_ptr(),
                )
            };
            let immutable = unsafe {
                libc::dlsym(
                    (-2_isize) as *mut c_void,
                    c"_dyld_is_memory_immutable".as_ptr(),
                )
            };
            anyhow::ensure!(
                !range.is_null() && !uuid.is_null() && !immutable.is_null(),
                "Native runtime mapped cache capability is unavailable"
            );
            Ok(Self {
                range: unsafe { std::mem::transmute::<*mut c_void, CacheRangeFunction>(range) },
                uuid: unsafe { std::mem::transmute::<*mut c_void, CacheUuidFunction>(uuid) },
                immutable: unsafe {
                    std::mem::transmute::<*mut c_void, ImmutableMemoryFunction>(immutable)
                },
            })
        }
        fn identity(&self) -> Result<ActiveCacheIdentity> {
            let mut length = 0_usize;
            let base = unsafe { (self.range)(&mut length) } as u64;
            let mut uuid = [0_u8; 16];
            anyhow::ensure!(
                base != 0
                    && length > 0
                    && base.checked_add(length as u64).is_some()
                    && unsafe { (self.uuid)(uuid.as_mut_ptr()) },
                "Native runtime active cache identity is unavailable"
            );
            let uuid = uuid::Uuid::from_bytes(uuid);
            anyhow::ensure!(
                !uuid.is_nil(),
                "Native runtime active cache UUID is unavailable"
            );
            Ok(ActiveCacheIdentity {
                base,
                length: length as u64,
                uuid,
            })
        }
    }
    fn memory_word(bytes: &[u8], offset: usize) -> Result<u64> {
        let end = offset
            .checked_add(8)
            .context("Native runtime mapped metadata overflow")?;
        Ok(u64::from_le_bytes(
            bytes
                .get(offset..end)
                .context("Native runtime mapped metadata is truncated")?
                .try_into()?,
        ))
    }
    fn read_process_memory(address: u64, size: usize) -> Result<Vec<u8>> {
        anyhow::ensure!(
            address != 0
                && size > 0
                && size <= 256 * 1024 + 32
                && address.checked_add(size as u64).is_some(),
            "Native runtime mapped read exceeds its limit"
        );
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(size)?;
        bytes.resize(size, 0);
        let mut copied = 0_u64;
        // Kernel copying can reject an unmapped or concurrently removed image without dereferencing its pointers.
        anyhow::ensure!(
            unsafe {
                mach_vm_read_overwrite(
                    mach_task_self_,
                    address,
                    size as u64,
                    bytes.as_mut_ptr() as u64,
                    &mut copied,
                )
            } == 0
                && copied == size as u64,
            "Native runtime mapped image metadata is unavailable"
        );
        Ok(bytes)
    }
    fn read_image_name(address: u64) -> Result<String> {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        anyhow::ensure!(
            page_size > 0 && page_size <= 64 * 1024,
            "Native runtime mapped page capability is unavailable"
        );
        let mut name = Vec::new();
        while name.len() < 4096 {
            let address = address
                .checked_add(name.len() as u64)
                .context("Native runtime image-name offset overflow")?;
            let remaining_page = page_size as u64 - address % page_size as u64;
            let count = 64_usize.min(4096 - name.len()).min(remaining_page as usize);
            let bytes = read_process_memory(address, count)?;
            if let Some(end) = bytes.iter().position(|byte| *byte == 0) {
                name.extend_from_slice(&bytes[..end]);
                return String::from_utf8(name)
                    .context("Native runtime mapped image name is not UTF-8");
            }
            name.extend_from_slice(&bytes);
        }
        anyhow::bail!("Native runtime mapped image name exceeds its limit")
    }
    fn current_dyld_infos() -> Result<(u64, u64, Vec<u8>)> {
        let mut fields = [0_u32; 5];
        let mut count = fields.len() as u32;
        anyhow::ensure!(
            unsafe { task_info(mach_task_self_, 17, fields.as_mut_ptr(), &mut count) } == 0
                && count == 5
                && fields[4] == 1,
            "Native runtime current-process dyld information is unavailable"
        );
        let address = u64::from(fields[0]) | (u64::from(fields[1]) << 32);
        let size = u64::from(fields[2]) | (u64::from(fields[3]) << 32);
        anyhow::ensure!(
            size >= 192 && size <= 4096,
            "Native runtime dyld information version is unsupported"
        );
        Ok((address, size, read_process_memory(address, 192)?))
    }
    fn cached_header_facts(path: &str, bytes: &[u8]) -> Result<SystemImageFacts> {
        ordinary_image_path(path)?;
        anyhow::ensure!(
            word(bytes, 0, false)? == 0xfeed_facf
                && word(bytes, 12, false)? == 6
                && word(bytes, 24, false)? & 0x8000_0000 != 0,
            "Native runtime mapped dependency is not a cached Mach-O dylib"
        );
        let cpu = word(bytes, 4, false)?;
        let subtype = word(bytes, 8, false)? & 0x00ff_ffff;
        let architecture = match (cpu, subtype) {
            (ARM64, 0) => "arm64",
            (ARM64, 2) => "arm64e",
            (X86_64, 3) => "x86_64",
            (X86_64, 8) => "x86_64h",
            _ => anyhow::bail!("Native runtime mapped cache CPU family is unsupported"),
        };
        current_architecture(cpu)?;
        let count = word(bytes, 16, false)? as usize;
        let size = word(bytes, 20, false)? as usize;
        anyhow::ensure!(
            count <= 4096 && size <= 256 * 1024 && bytes.len() == 32 + size,
            "Native runtime mapped load commands exceed their limit"
        );
        let mut offset = 32_usize;
        let mut identifier = None;
        let mut install_name = None;
        let mut dependencies = Vec::new();
        for _ in 0..count {
            let command = word(bytes, offset, false)?;
            let length = word(bytes, offset + 4, false)? as usize;
            anyhow::ensure!(
                length >= 8 && length.is_multiple_of(8),
                "Native runtime mapped load command is invalid"
            );
            let end = offset
                .checked_add(length)
                .context("Native runtime mapped command overflow")?;
            let record = bytes
                .get(offset..end)
                .context("Native runtime mapped load command is truncated")?;
            if command == 0x1b {
                anyhow::ensure!(
                    length == 24 && identifier.is_none(),
                    "Native runtime mapped UUID is ambiguous"
                );
                let raw: [u8; 16] = record
                    .get(8..24)
                    .context("Native runtime mapped UUID is truncated")?
                    .try_into()?;
                identifier = Some(uuid::Uuid::from_bytes(raw));
            }
            if matches!(
                command,
                0xc | 0xd | 0x8000_0018 | 0x8000_001f | 0x8000_0023 | 0x20
            ) {
                let start = word(record, 8, false)? as usize;
                let alternate =
                    matches!(command, 0xc | 0x8000_0018) && word(record, 12, false)? == 0x1a74_1800;
                let attributes = if alternate {
                    let flags = word(record, 24, false)?;
                    anyhow::ensure!(
                        start == 28
                            && flags & !15 == 0
                            && !(flags & 2 != 0 && flags & (1 | 8) != 0),
                        "Native runtime mapped dependency flags are unsupported"
                    );
                    flags as u8
                } else {
                    match command {
                        0x8000_0018 => 1,
                        0x8000_001f => 2,
                        0x8000_0023 => 4,
                        0x20 => 16,
                        _ => 0,
                    }
                };
                anyhow::ensure!(
                    start >= 24,
                    "Native runtime mapped dependency offset is invalid"
                );
                let raw = record
                    .get(start..)
                    .context("Native runtime mapped dependency is truncated")?;
                let end = raw
                    .iter()
                    .position(|byte| *byte == 0)
                    .context("Native runtime mapped dependency is unterminated")?;
                anyhow::ensure!(
                    end <= 4096,
                    "Native runtime mapped dependency exceeds its limit"
                );
                let name = std::str::from_utf8(&raw[..end])
                    .context("Native runtime mapped dependency is not UTF-8")?;
                ordinary_image_path(name)?;
                if command == 0xd {
                    anyhow::ensure!(
                        install_name.replace(name.to_owned()).is_none(),
                        "Native runtime mapped install name is ambiguous"
                    );
                } else {
                    dependencies.push(SystemDependency {
                        path: name.to_owned(),
                        attributes,
                    });
                }
            }
            anyhow::ensure!(
                !matches!(command, 0x8000_001c | 0x27 | 0x8000_0035),
                "Native runtime mapped dependency requires unsupported additional load policy"
            );
            offset = end;
        }
        anyhow::ensure!(
            offset == bytes.len() && install_name.as_deref() == Some(path),
            "Native runtime mapped cached path does not match its install name"
        );
        let uuid = identifier.context("Native runtime mapped cached UUID is unavailable")?;
        anyhow::ensure!(
            !uuid.is_nil(),
            "Native runtime mapped cached UUID is invalid"
        );
        Ok(SystemImageFacts {
            path: path.to_owned(),
            architecture: architecture.to_owned(),
            uuid,
            dependencies,
        })
    }
    fn loaded_cache_snapshot(
        requested: &std::collections::BTreeSet<String>,
    ) -> Result<LoadedCacheSnapshot> {
        loaded_cache_snapshot_inner(requested, LoadedCachePathEvidence::Membership)
    }
    enum LoadedCachePathEvidence {
        Membership,
        ObservedImmutableHeader,
    }
    fn loaded_cache_snapshot_inner(
        requested: &std::collections::BTreeSet<String>,
        evidence: LoadedCachePathEvidence,
    ) -> Result<LoadedCacheSnapshot> {
        let sdk = CacheSdkFunctions::current()?;
        let cache = sdk.identity()?;
        let (address, size, before) = current_dyld_infos()?;
        let version = word(&before, 0, false)?;
        let count = word(&before, 4, false)? as usize;
        let array = memory_word(&before, 8)?;
        let timestamp = memory_word(&before, 184)?;
        anyhow::ensure!(
            version >= 15
                && count > 0
                && count <= 4096
                && array != 0
                && memory_word(&before, 176)? == cache.base
                && before.get(160..176) == Some(cache.uuid.as_bytes().as_slice())
                && before.get(24) == Some(&0),
            "Native runtime process has no matching active shared cache"
        );
        let descriptors = read_process_memory(
            array,
            count
                .checked_mul(24)
                .context("Native runtime mapped image count overflow")?,
        )?;
        let mut images = std::collections::BTreeMap::new();
        let mut aggregate_bytes = 0_usize;
        for descriptor in descriptors.chunks_exact(24) {
            let name = read_image_name(memory_word(descriptor, 8)?)?;
            if !requested.contains(&name) {
                continue;
            }
            if matches!(evidence, LoadedCachePathEvidence::Membership) && !cache_contains(&name)? {
                continue;
            }
            let header = memory_word(descriptor, 0)?;
            anyhow::ensure!(
                header >= cache.base
                    && header
                        .checked_add(32)
                        .is_some_and(|end| end <= cache.base + cache.length)
                    && unsafe { (sdk.immutable)(header as *const c_void, 32) },
                "Native runtime mapped header has no immutable cache origin"
            );
            let first = read_process_memory(header, 32)?;
            let length = word(&first, 20, false)? as usize;
            anyhow::ensure!(
                length <= 256 * 1024,
                "Native runtime mapped command bytes exceed their limit"
            );
            let total = 32_usize
                .checked_add(length)
                .context("Native runtime mapped image size overflow")?;
            anyhow::ensure!(
                header >= cache.base
                    && header
                        .checked_add(total as u64)
                        .is_some_and(|end| end <= cache.base + cache.length)
                    && unsafe { (sdk.immutable)(header as *const c_void, total) },
                "Native runtime mapped image is not immutable in the active cache"
            );
            aggregate_bytes = aggregate_bytes
                .checked_add(total)
                .context("Native runtime mapped snapshot size overflow")?;
            anyhow::ensure!(
                aggregate_bytes <= MAX_CACHE_OUTPUT_BYTES,
                "Native runtime mapped system snapshot exceeds its byte limit"
            );
            let bytes = read_process_memory(header, total)?;
            let facts = cached_header_facts(&name, &bytes)?;
            anyhow::ensure!(
                images
                    .insert(
                        name,
                        LoadedCachedImage {
                            address: header,
                            facts
                        }
                    )
                    .is_none(),
                "Native runtime mapped system image is duplicated"
            );
        }
        let (after_address, after_size, after) = current_dyld_infos()?;
        anyhow::ensure!(
            address == after_address
                && size == after_size
                && before == after
                && cache == sdk.identity()?,
            "Native runtime loaded cache changed during its bounded snapshot"
        );
        Ok(LoadedCacheSnapshot {
            cache,
            version,
            size,
            timestamp,
            images,
        })
    }

    type CacheTextFunction = for<'callback> unsafe extern "C" fn(
        *const u8,
        *const block2::Block<dyn Fn(*const c_void) + 'callback>,
    ) -> i32;
    type CacheCanonicalPathFunction =
        unsafe extern "C" fn(*const std::ffi::c_char) -> *const std::ffi::c_char;
    struct CacheTextSdk {
        iterate: CacheTextFunction,
        canonical: Option<CacheCanonicalPathFunction>,
    }
    impl CacheTextSdk {
        fn current() -> Result<Self> {
            let iterate = unsafe {
                libc::dlsym(
                    (-2_isize) as *mut c_void,
                    c"dyld_shared_cache_iterate_text".as_ptr(),
                )
            };
            anyhow::ensure!(
                !iterate.is_null(),
                "Native runtime cache text capability is unavailable"
            );
            let canonical = unsafe {
                libc::dlsym(
                    (-2_isize) as *mut c_void,
                    c"_dyld_shared_cache_real_path".as_ptr(),
                )
            };
            Ok(Self {
                iterate: unsafe { std::mem::transmute::<*mut c_void, CacheTextFunction>(iterate) },
                canonical: if canonical.is_null() {
                    None
                } else {
                    Some(unsafe {
                        std::mem::transmute::<*mut c_void, CacheCanonicalPathFunction>(canonical)
                    })
                },
            })
        }
        fn canonical_path(&self, path: &str) -> Result<Option<String>> {
            system_path(path)?;
            let Some(canonical) = self.canonical else {
                return Ok(Some(path.to_owned()));
            };
            let path = std::ffi::CString::new(path)?;
            let address = unsafe { canonical(path.as_ptr()) } as u64;
            if address == 0 {
                return Ok(None);
            }
            let path = read_image_name(address)?;
            system_path(&path)?;
            Ok(Some(path))
        }
    }
    struct CacheTextImage {
        address: u64,
        text_size: u64,
        canonical_path: String,
        facts: SystemImageFacts,
    }
    struct CacheTextSnapshot {
        cache: ActiveCacheIdentity,
        version: u32,
        size: u64,
        timestamp: u64,
        images: std::collections::BTreeMap<String, CacheTextImage>,
    }
    fn cache_text_record_image(
        sdk: &CacheSdkFunctions,
        cache: &ActiveCacheIdentity,
        canonical_path: &str,
        record: &[u8],
        aggregate_bytes: &mut usize,
    ) -> Result<CacheTextImage> {
        anyhow::ensure!(
            record.len() == 56 && memory_word(record, 0)? == 2,
            "Native runtime cache text record version is unsupported"
        );
        let text_size = memory_word(record, 16)?;
        let offset = memory_word(record, 48)?;
        let address = cache
            .base
            .checked_add(offset)
            .context("Native runtime cache text offset overflow")?;
        anyhow::ensure!(
            (32..=MAX_DISK_IMAGE_BYTES).contains(&text_size)
                && offset
                    .checked_add(text_size)
                    .is_some_and(|end| end <= cache.length)
                && unsafe { (sdk.immutable)(address as *const c_void, 32) },
            "Native runtime cache text header has no immutable active origin"
        );
        let first = read_process_memory(address, 32)?;
        let command_size = word(&first, 20, false)? as usize;
        anyhow::ensure!(
            command_size <= 256 * 1024,
            "Native runtime cache text load commands exceed their limit"
        );
        let total = 32_usize
            .checked_add(command_size)
            .context("Native runtime cache text header size overflow")?;
        anyhow::ensure!(
            total as u64 <= text_size
                && unsafe { (sdk.immutable)(address as *const c_void, total) },
            "Native runtime cache text commands are not immutable"
        );
        *aggregate_bytes = aggregate_bytes
            .checked_add(total)
            .context("Native runtime cache text snapshot size overflow")?;
        anyhow::ensure!(
            *aggregate_bytes <= MAX_CACHE_OUTPUT_BYTES,
            "Native runtime cache text snapshot exceeds its byte limit"
        );
        let bytes = read_process_memory(address, total)?;
        let facts = cached_header_facts(canonical_path, &bytes)?;
        anyhow::ensure!(
            record.get(24..40) == Some(facts.uuid.as_bytes().as_slice()),
            "Native runtime cache text UUID does not match its actual mapped header"
        );
        Ok(CacheTextImage {
            address,
            text_size,
            canonical_path: canonical_path.to_owned(),
            facts,
        })
    }
    struct CacheTextDiscovery {
        cache: ActiveCacheIdentity,
        version: u32,
        size: u64,
        timestamp: u64,
        records: std::collections::BTreeMap<String, Vec<u8>>,
    }
    struct CacheTextDiscoveryState {
        count: usize,
        aggregate_bytes: usize,
        records: std::collections::BTreeMap<String, Vec<u8>>,
    }
    fn cache_text_metadata_epoch(
        cache: &ActiveCacheIdentity,
        renewed_cache: &ActiveCacheIdentity,
        before: (u64, u64, &[u8]),
        after: (u64, u64, &[u8]),
    ) -> Result<u64> {
        let (address, size, before) = before;
        let (after_address, after_size, after) = after;
        anyhow::ensure!(
            address != 0
                && address == after_address
                && (192..=4096).contains(&size)
                && size == after_size
                && before.len() == 192
                && after.len() == 192
                && word(before, 0, false)? >= 15
                && word(before, 0, false)? == word(after, 0, false)?,
            "Native runtime cache text metadata layout changed"
        );
        anyhow::ensure!(
            cache.base != 0
                && cache.length > 0
                && cache.base.checked_add(cache.length).is_some()
                && !cache.uuid.is_nil()
                && cache == renewed_cache
                && memory_word(before, 176)? == cache.base
                && memory_word(after, 176)? == cache.base
                && before.get(160..176) == Some(cache.uuid.as_bytes().as_slice())
                && after.get(160..176) == Some(cache.uuid.as_bytes().as_slice())
                && before.get(24) == Some(&0)
                && after.get(24) == Some(&0),
            "Native runtime cache text metadata has no stable attached active cache"
        );
        let epoch = memory_word(after, 184)?;
        anyhow::ensure!(
            epoch >= memory_word(before, 184)?,
            "Native runtime cache text process epoch regressed"
        );
        // Cache-text discovery never reads the loaded-image array; unrelated loads may advance its epoch.
        Ok(epoch)
    }
    fn cache_text_discovery() -> Result<CacheTextDiscovery> {
        let sdk = CacheSdkFunctions::current()?;
        let text_sdk = CacheTextSdk::current()?;
        let cache = sdk.identity()?;
        let (address, size, before) = current_dyld_infos()?;
        let version = word(&before, 0, false)?;
        anyhow::ensure!(
            version >= 15
                && memory_word(&before, 176)? == cache.base
                && before.get(160..176) == Some(cache.uuid.as_bytes().as_slice())
                && before.get(24) == Some(&0),
            "Native runtime cache text process has no matching active cache"
        );
        let state: std::cell::RefCell<Result<CacheTextDiscoveryState>> =
            std::cell::RefCell::new(Ok(CacheTextDiscoveryState {
                count: 0,
                aggregate_bytes: 0,
                records: std::collections::BTreeMap::new(),
            }));
        let reentered = std::cell::Cell::new(false);
        let status = {
            let callback = block2::StackBlock::new(|raw: *const c_void| {
                let Ok(mut state) = state.try_borrow_mut() else {
                    reentered.set(true);
                    return;
                };
                let Ok(CacheTextDiscoveryState {
                    count,
                    aggregate_bytes,
                    records,
                }) = state.as_mut()
                else {
                    return;
                };
                let result = (|| -> Result<()> {
                    *count = count
                        .checked_add(1)
                        .context("Native runtime cache text record count overflow")?;
                    anyhow::ensure!(
                        *count <= MAX_CACHE_IMAGES,
                        "Native runtime cache text image count exceeds its limit"
                    );
                    let record = read_process_memory(raw as u64, 56)?;
                    anyhow::ensure!(
                        memory_word(&record, 0)? == 2,
                        "Native runtime cache text record version is unsupported"
                    );
                    let canonical = read_image_name(memory_word(&record, 40)?)?;
                    ordinary_image_path(&canonical)?;
                    *aggregate_bytes = aggregate_bytes
                        .checked_add(canonical.len())
                        .and_then(|total| total.checked_add(record.len()))
                        .context("Native runtime cache text discovery size overflow")?;
                    anyhow::ensure!(
                        *aggregate_bytes <= MAX_CACHE_OUTPUT_BYTES,
                        "Native runtime cache text discovery exceeds its byte limit"
                    );
                    anyhow::ensure!(
                        records.insert(canonical, record).is_none(),
                        "Native runtime cache text record path is duplicated"
                    );
                    Ok(())
                })();
                if let Err(error) = result {
                    *state = Err(error);
                }
            });
            // The pinned dyld API invokes the block synchronously and invalidates each path after its callback.
            unsafe { (text_sdk.iterate)(cache.uuid.as_bytes().as_ptr(), &*callback) }
        };
        anyhow::ensure!(
            status == 0 && !reentered.get(),
            "Native runtime cache text iteration failed"
        );
        let records = state.into_inner()?.records;
        anyhow::ensure!(
            !records.is_empty(),
            "Native runtime cache text records are unavailable"
        );
        let (after_address, after_size, after) = current_dyld_infos()?;
        #[cfg(any(test, feature = "test-support"))]
        if address != after_address || size != after_size || before != after {
            let (cache_available, cache_stable) = match sdk.identity() {
                Ok(current) => (true, current == cache),
                Err(_) => (false, false),
            };
            eprintln!(
                "Native runtime cache-text discovery metadata changed: version_changed={} image_count_changed={} image_array_changed={} timestamp_changed={} cache_available={} cache_stable={} metadata_address_stable={} metadata_size_stable={}",
                before.get(0..4) != after.get(0..4),
                before.get(4..8) != after.get(4..8),
                before.get(8..16) != after.get(8..16),
                before.get(184..192) != after.get(184..192),
                cache_available,
                cache_stable,
                address == after_address,
                size == after_size,
            );
        }
        let timestamp = cache_text_metadata_epoch(
            &cache,
            &sdk.identity()?,
            (address, size, &before),
            (after_address, after_size, &after),
        )?;
        Ok(CacheTextDiscovery {
            cache,
            version,
            size,
            timestamp,
            records,
        })
    }
    fn cache_text_snapshot_from_discovery(
        discovery: &CacheTextDiscovery,
        requested: &std::collections::BTreeSet<String>,
    ) -> Result<CacheTextSnapshot> {
        anyhow::ensure!(
            requested.len() <= MAX_CLOSURE_IMAGES,
            "Native runtime cache text requests exceed their limit"
        );
        let sdk = CacheSdkFunctions::current()?;
        let text_sdk = CacheTextSdk::current()?;
        anyhow::ensure!(
            discovery.cache == sdk.identity()?,
            "Native runtime cache text discovery origin changed"
        );
        let (address, size, before) = current_dyld_infos()?;
        let version = word(&before, 0, false)?;
        let timestamp = memory_word(&before, 184)?;
        anyhow::ensure!(
            version == discovery.version
                && size == discovery.size
                && timestamp >= discovery.timestamp
                && memory_word(&before, 176)? == discovery.cache.base
                && before.get(160..176) == Some(discovery.cache.uuid.as_bytes().as_slice())
                && before.get(24) == Some(&0),
            "Native runtime cache text process no longer matches its discovery"
        );
        let mut images = std::collections::BTreeMap::new();
        let mut aggregate_bytes = 0_usize;
        for path in requested {
            let Some(canonical_path) = text_sdk.canonical_path(path)? else {
                continue;
            };
            let Some(record) = discovery.records.get(&canonical_path) else {
                continue;
            };
            let mut image = cache_text_record_image(
                &sdk,
                &discovery.cache,
                &canonical_path,
                record,
                &mut aggregate_bytes,
            )?;
            image.facts.path.clone_from(path);
            anyhow::ensure!(
                text_sdk.canonical_path(path)?.as_ref() == Some(&canonical_path),
                "Native runtime cache canonical path changed during its mapped review"
            );
            images.insert(path.clone(), image);
        }
        let (after_address, after_size, after) = current_dyld_infos()?;
        let timestamp = cache_text_metadata_epoch(
            &discovery.cache,
            &sdk.identity()?,
            (address, size, &before),
            (after_address, after_size, &after),
        )?;
        Ok(CacheTextSnapshot {
            cache: discovery.cache.clone(),
            version,
            size,
            timestamp,
            images,
        })
    }
    fn cache_text_snapshot(
        requested: &std::collections::BTreeSet<String>,
    ) -> Result<CacheTextSnapshot> {
        cache_text_snapshot_from_discovery(&cache_text_discovery()?, requested)
    }
    fn observed_cache_text_closure(
        dependencies: Vec<(u32, String)>,
        operating_system: String,
    ) -> Result<Option<SystemClosure>> {
        anyhow::ensure!(
            !dependencies.is_empty() && dependencies.len() <= MAX_CLOSURE_DEPENDENCIES,
            "Native runtime cache text closure roots exceed their limit"
        );
        for (_, path) in &dependencies {
            system_path(path)?;
        }
        if dependencies
            .iter()
            .any(|(architecture, _)| current_architecture(*architecture).is_err())
        {
            return Ok(None);
        }
        let mut requested = dependencies
            .into_iter()
            .map(|(_, path)| path)
            .collect::<std::collections::BTreeSet<_>>();
        anyhow::ensure!(
            requested.len() <= MAX_CLOSURE_IMAGES,
            "Native runtime cache text closure roots exceed their image limit"
        );
        let mut original: Option<(ActiveCacheIdentity, u32, u64, u64)> = None;
        let mut absent_weak = std::collections::BTreeSet::new();
        let discovery = cache_text_discovery()?;
        for _ in 0..MAX_CLOSURE_IMAGES {
            let mapped = cache_text_snapshot_from_discovery(&discovery, &requested)?;
            if let Some((cache, version, size, timestamp)) = &original {
                anyhow::ensure!(
                    mapped.cache == *cache
                        && mapped.version == *version
                        && mapped.size == *size
                        && mapped.timestamp >= *timestamp,
                    "Native runtime cache text closure changed between snapshots"
                );
            } else {
                original = Some((
                    mapped.cache.clone(),
                    mapped.version,
                    mapped.size,
                    mapped.timestamp,
                ));
            }
            if requested
                .iter()
                .any(|path| !mapped.images.contains_key(path))
            {
                return Ok(None);
            }
            let mut needed = requested.clone();
            let mut dependency_count = 0_usize;
            for image in mapped.images.values() {
                dependency_count = dependency_count
                    .checked_add(image.facts.dependencies.len())
                    .context("Native runtime cache text dependency count overflow")?;
                anyhow::ensure!(
                    dependency_count <= MAX_CLOSURE_DEPENDENCIES,
                    "Native runtime cache text dependency count exceeds its limit"
                );
                for dependency in &image.facts.dependencies {
                    system_path(&dependency.path)?;
                    if dependency.attributes & 1 != 0
                        && cache_membership_if_available(&dependency.path)? == Some(false)
                    {
                        absent_weak.insert(dependency.path.clone());
                    } else {
                        needed.insert(dependency.path.clone());
                    }
                    anyhow::ensure!(
                        needed.len() <= MAX_CLOSURE_IMAGES
                            && absent_weak.len() <= MAX_CLOSURE_DEPENDENCIES,
                        "Native runtime cache text closure exceeds its image or weak-edge limit"
                    );
                }
            }
            if needed == requested {
                return Ok(Some(SystemClosure {
                    inspection: SystemClosureInspection::ObservedCacheText,
                    operating_system,
                    cache: mapped.cache,
                    mapped_version: mapped.version,
                    mapped_size: mapped.size,
                    mapped_epoch: mapped.timestamp,
                    images: mapped
                        .images
                        .into_values()
                        .map(|image| ReviewedSystemImage {
                            facts: image.facts,
                            origin: SystemImageOrigin::CacheText {
                                address: image.address,
                                text_size: image.text_size,
                                canonical_path: image.canonical_path,
                            },
                        })
                        .collect(),
                    absent_weak,
                }));
            }
            requested = needed;
        }
        anyhow::bail!("Native runtime cache text closure did not converge within its limit")
    }
    fn revalidate_cache_text_images(
        closure: &SystemClosure,
        mapped: &CacheTextSnapshot,
    ) -> Result<()> {
        anyhow::ensure!(
            mapped.cache == closure.cache
                && mapped.version == closure.mapped_version
                && mapped.size == closure.mapped_size
                && mapped.timestamp >= closure.mapped_epoch,
            "Native runtime cache text identity changed"
        );
        for image in &closure.images {
            if let SystemImageOrigin::CacheText {
                address,
                text_size,
                canonical_path,
            } = &image.origin
            {
                let current = mapped
                    .images
                    .get(&image.facts.path)
                    .context("Native runtime cache text dependency disappeared")?;
                anyhow::ensure!(
                    current.address == *address
                        && current.text_size == *text_size
                        && current.canonical_path == *canonical_path
                        && current.facts == image.facts,
                    "Native runtime immutable cache text dependency changed"
                );
            }
        }
        Ok(())
    }
    fn revalidate_observed_cache_text_closure(closure: &SystemClosure) -> Result<()> {
        anyhow::ensure!(
            observed_operating_system_version()? == closure.operating_system
                && closure
                    .images
                    .iter()
                    .all(|image| matches!(&image.origin, SystemImageOrigin::CacheText { .. })),
            "Native runtime cache text closure policy changed"
        );
        let requested = closure
            .images
            .iter()
            .map(|image| image.facts.path.clone())
            .collect();
        let mapped = cache_text_snapshot(&requested)?;
        revalidate_cache_text_images(closure, &mapped)?;
        for path in &closure.absent_weak {
            anyhow::ensure!(
                cache_membership_if_available(path)? == Some(false),
                "Native runtime absent cached weak dependency changed"
            );
        }
        Ok(())
    }

    fn ordinary_image_path(path: &str) -> Result<()> {
        anyhow::ensure!(
            path.len() <= 4096
                && path.starts_with('/')
                && !path.chars().any(char::is_control)
                && path
                    .split('/')
                    .skip(1)
                    .all(|part| !part.is_empty() && part != "." && part != ".."),
            "Native runtime dependency requires an ordinary absolute system path"
        );
        Ok(())
    }
    fn system_path(path: &str) -> Result<()> {
        ordinary_image_path(path)?;
        anyhow::ensure!(
            path.starts_with("/usr/lib/") || path.starts_with("/System/Library/"),
            "Native runtime disk dependency requires its original system location"
        );
        Ok(())
    }
    fn parse_dependency(line: &str) -> Result<SystemDependency> {
        let start = line
            .find('/')
            .context("Native runtime dependency path is unavailable")?;
        let path = &line[start..];
        ordinary_image_path(path)?;
        let mut attributes = 0;
        for attribute in line[..start].split_ascii_whitespace() {
            let bit = match attribute {
                "weak-link" => 1,
                "re-export" => 2,
                "upward" => 4,
                "delay-init" => 8,
                "lazy-load" => 16,
                _ => anyhow::bail!("Native runtime dependency attribute is unsupported"),
            };
            anyhow::ensure!(
                attributes & bit == 0,
                "Native runtime dependency attribute is ambiguous"
            );
            attributes |= bit;
        }
        anyhow::ensure!(
            attributes & 2 == 0 || attributes & (1 | 8) == 0,
            "Native runtime re-export dependency has unsupported conflicting attributes"
        );
        Ok(SystemDependency {
            path: path.to_owned(),
            attributes,
        })
    }
    fn parse_system_images(
        bytes: &[u8],
    ) -> Result<std::collections::BTreeMap<(String, String), SystemImageFacts>> {
        anyhow::ensure!(
            !bytes.is_empty() && bytes.len() <= MAX_CACHE_OUTPUT_BYTES,
            "Native runtime cache output exceeds its limit"
        );
        let text =
            std::str::from_utf8(bytes).context("Native runtime cache output is not UTF-8")?;
        let mut result = std::collections::BTreeMap::new();
        let mut lines = text.lines().peekable();
        let mut dependency_count = 0_usize;
        while let Some(header) = lines.next() {
            if header.is_empty() {
                continue;
            }
            anyhow::ensure!(
                result.len() < MAX_CACHE_IMAGES,
                "Native runtime cache image count exceeds its limit"
            );
            let (path, architecture) = header
                .split_once(" [")
                .context("Native runtime cache header is unsupported")?;
            ordinary_image_path(path)?;
            let architecture = architecture
                .strip_suffix("]:")
                .context("Native runtime cache architecture is unavailable")?;
            anyhow::ensure!(
                matches!(architecture, "arm64" | "arm64e" | "x86_64" | "x86_64h"),
                "Native runtime cache architecture is unsupported"
            );
            anyhow::ensure!(
                lines.next() == Some("    -uuid:"),
                "Native runtime cache UUID section is unavailable"
            );
            let uuid_text = lines
                .next()
                .and_then(|line| line.strip_prefix("        "))
                .context("Native runtime cache UUID is unavailable")?;
            let uuid =
                uuid::Uuid::parse_str(uuid_text).context("Native runtime cache UUID is invalid")?;
            anyhow::ensure!(
                !uuid.is_nil()
                    && uuid_text.len() == 36
                    && uuid
                        .hyphenated()
                        .to_string()
                        .eq_ignore_ascii_case(uuid_text),
                "Native runtime cache UUID is noncanonical"
            );
            anyhow::ensure!(
                lines.next() == Some("    -linked_dylibs:")
                    && lines.next() == Some("        attributes     load path"),
                "Native runtime cache dependency section is unavailable"
            );
            let mut dependencies = Vec::new();
            while lines
                .peek()
                .is_some_and(|line| line.starts_with("        "))
            {
                let line = lines
                    .next()
                    .context("Native runtime dependency disappeared")?;
                dependency_count = dependency_count
                    .checked_add(1)
                    .context("Native runtime dependency count overflow")?;
                anyhow::ensure!(
                    dependency_count <= MAX_CACHE_DEPENDENCIES,
                    "Native runtime cache dependency count exceeds its limit"
                );
                dependencies.push(parse_dependency(line)?);
            }
            anyhow::ensure!(
                result
                    .insert(
                        (path.to_owned(), architecture.to_owned()),
                        SystemImageFacts {
                            path: path.to_owned(),
                            architecture: architecture.to_owned(),
                            uuid,
                            dependencies,
                        }
                    )
                    .is_none(),
                "Native runtime cache image is duplicated"
            );
        }
        anyhow::ensure!(
            !result.is_empty(),
            "Native runtime cache graph is unavailable"
        );
        Ok(result)
    }
    fn cache_architecture_matches(architecture: u32, name: &str) -> bool {
        // The active arm64e cache supports arm64 executables; a foreign CPU family does not.
        match architecture {
            ARM64 => matches!(name, "arm64" | "arm64e"),
            X86_64 => matches!(name, "x86_64" | "x86_64h"),
            _ => false,
        }
    }
    fn current_architecture(architecture: u32) -> Result<&'static str> {
        #[cfg(target_arch = "aarch64")]
        if architecture == ARM64 {
            return Ok("arm64");
        }
        #[cfg(target_arch = "x86_64")]
        if architecture == X86_64 {
            return Ok("x86_64");
        }
        anyhow::bail!("Native runtime helper slice has no current-host cache qualification")
    }
    fn observed_operating_system_version() -> Result<String> {
        let mut buffer = [0_u8; 64];
        let mut size = buffer.len();
        anyhow::ensure!(
            unsafe {
                libc::sysctlbyname(
                    c"kern.osproductversion".as_ptr(),
                    buffer.as_mut_ptr().cast(),
                    &mut size,
                    std::ptr::null_mut(),
                    0,
                )
            } == 0
                && size > 1
                && size <= buffer.len()
                && buffer.get(size - 1) == Some(&0),
            "Native runtime cache capability cannot determine the operating system"
        );
        let version = std::str::from_utf8(&buffer[..size - 1])
            .context("Native runtime operating system version is invalid")?;
        anyhow::ensure!(
            version
                .split('.')
                .all(|part| !part.is_empty() && part.bytes().all(|byte| byte.is_ascii_digit()))
                && version.split('.').count() <= 3,
            "Native runtime operating system version is unsupported"
        );
        let mut components = version.split('.');
        let major = components
            .next()
            .context("Native runtime OS major is missing")?
            .parse::<u32>()?;
        let minor = components.next().unwrap_or("0").parse::<u32>()?;
        anyhow::ensure!(
            major > 10 || (major == 10 && minor >= 15),
            "Native runtime loader admission requires macOS 10.15 or later"
        );
        Ok(version.to_owned())
    }
    fn operating_system_version() -> Result<String> {
        let version = observed_operating_system_version()?;
        anyhow::ensure!(
            version
                .split('.')
                .next()
                .and_then(|major| major.parse::<u32>().ok())
                .is_some_and(|major| major >= 14),
            "Native runtime installed cache tool currently requires macOS 14 or later"
        );
        Ok(version)
    }
    fn cache_contains(path: &str) -> Result<bool> {
        cache_membership_if_available(path)?
            .context("Native runtime active cache API is unavailable on this operating system")
    }
    fn cache_membership_if_available(path: &str) -> Result<Option<bool>> {
        ordinary_image_path(path)?;
        let path = std::ffi::CString::new(path).context("Native runtime cache path is invalid")?;
        let address = unsafe {
            libc::dlsym(
                (-2_isize) as *mut c_void,
                c"_dyld_shared_cache_contains_path".as_ptr(),
            )
        };
        if address.is_null() {
            return Ok(None);
        }
        // Dynamic lookup preserves the application's macOS 10.15 deployment compatibility.
        let contains: unsafe extern "C" fn(*const std::ffi::c_char) -> bool =
            unsafe { std::mem::transmute(address) };
        Ok(Some(unsafe { contains(path.as_ptr()) }))
    }
    fn validate_apple_code(path: &Path, identifier: Option<&str>) -> Result<()> {
        let bytes = path.as_os_str().as_bytes();
        anyhow::ensure!(
            path.is_absolute() && bytes.len() <= 4096,
            "Native runtime Apple image path is invalid"
        );
        let url = owned(unsafe {
            CFURLCreateFromFileSystemRepresentation(
                std::ptr::null(),
                bytes.as_ptr(),
                bytes.len() as isize,
                0,
            )
        })?;
        let mut code = std::ptr::null();
        anyhow::ensure!(
            unsafe {
                SecStaticCodeCreateWithPathAndAttributes(url.0, 0, std::ptr::null(), &mut code)
            } == 0,
            "Native runtime Apple image signature is unavailable"
        );
        let code = owned(code)?;
        let text = match identifier {
            Some("com.apple.xcrun") => "anchor apple and identifier \"com.apple.xcrun\"",
            Some("com.apple.dyld_info") => "anchor apple and identifier \"com.apple.dyld_info\"",
            None => "anchor apple",
            _ => anyhow::bail!("Native runtime Apple image role is unsupported"),
        };
        let text = string(text)?;
        let mut requirement = std::ptr::null();
        anyhow::ensure!(
            unsafe { SecRequirementCreateWithString(text.0, 0, &mut requirement) } == 0,
            "Native runtime Apple image trust requirement is unavailable"
        );
        let requirement = owned(requirement)?;
        anyhow::ensure!(
            unsafe { SecStaticCodeCheckValidity(code.0, (1 << 29) | (1 << 4) | 1, requirement.0) }
                == 0,
            "Native runtime Apple image does not satisfy strict native trust"
        );
        Ok(())
    }
    #[cfg(feature = "cargo-workspace")]
    async fn tool_output(
        path: &Path,
        arguments: &[&str],
        executor: &gpui::BackgroundExecutor,
    ) -> Result<Vec<u8>> {
        let limit = if arguments == ["--find", "dyld_info"] {
            4097
        } else {
            MAX_CACHE_OUTPUT_BYTES
        };
        let mut command = util::command::new_command(path);
        command
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("LC_ALL", "C")
            .current_dir("/")
            .args(arguments);
        let output = crate::cargo_workspace_store::bounded_cargo_command_output(
            command,
            "Native runtime Apple cache review",
            limit,
            Duration::from_secs(10),
            executor,
        )
        .await?;
        anyhow::ensure!(
            output.status.success() && output.stderr.is_empty(),
            "Native runtime Apple cache review was unsuccessful"
        );
        Ok(output.stdout)
    }
    #[cfg(not(feature = "cargo-workspace"))]
    async fn tool_output(
        _path: &Path,
        _arguments: &[&str],
        _executor: &gpui::BackgroundExecutor,
    ) -> Result<Vec<u8>> {
        anyhow::bail!("Native runtime cache admission requires the Cargo workspace capability")
    }
    async fn bounded_apple_identity(
        filesystem: &dyn fs::Fs,
        requested: PathBuf,
    ) -> Result<crate::profile_results::TaskHelperIdentity> {
        let canonical = filesystem.canonicalize(&requested).await?;
        let metadata = filesystem
            .metadata(&canonical)
            .await?
            .context("Native runtime Apple image disappeared")?;
        anyhow::ensure!(
            !metadata.is_dir
                && !metadata.is_fifo
                && !metadata.is_symlink
                && metadata.len > 0
                && metadata.len <= MAX_DISK_IMAGE_BYTES,
            "Native runtime Apple image exceeds its byte policy"
        );
        crate::profile_results::prepare_task_helper_identity(
            filesystem,
            requested,
            &AtomicBool::new(false),
        )
        .await
    }
    async fn prepare_tool(
        filesystem: &dyn fs::Fs,
        executor: &gpui::BackgroundExecutor,
    ) -> Result<(
        crate::profile_results::TaskHelperIdentity,
        crate::profile_results::TaskHelperIdentity,
    )> {
        let cancellation = AtomicBool::new(false);
        let discovery = bounded_apple_identity(filesystem, PathBuf::from("/usr/bin/xcrun")).await?;
        validate_apple_code(discovery.canonical_path(), Some("com.apple.xcrun"))?;
        let output = tool_output(
            discovery.canonical_path(),
            &["--find", "dyld_info"],
            executor,
        )
        .await?;
        anyhow::ensure!(
            output.len() <= 4097,
            "Native runtime installed tool path exceeds its limit"
        );
        let text = std::str::from_utf8(&output)
            .context("Native runtime installed tool path is not UTF-8")?;
        let text = text.strip_suffix('\n').unwrap_or(text);
        anyhow::ensure!(
            !text.is_empty()
                && !text.chars().any(char::is_control)
                && Path::new(text).is_absolute(),
            "Native runtime installed tool path is invalid"
        );
        let tool = bounded_apple_identity(filesystem, PathBuf::from(text)).await?;
        validate_apple_code(tool.canonical_path(), Some("com.apple.dyld_info"))?;
        crate::profile_results::revalidate_task_helper_identity(
            filesystem,
            &discovery,
            &cancellation,
        )
        .await?;
        crate::profile_results::revalidate_task_helper_identity(filesystem, &tool, &cancellation)
            .await?;
        Ok((discovery, tool))
    }
    fn select_cached<'a>(
        index: &'a std::collections::BTreeMap<(String, String), SystemImageFacts>,
        path: &str,
        architecture: u32,
    ) -> Result<&'a SystemImageFacts> {
        let names: &[&str] = match architecture {
            ARM64 => &["arm64", "arm64e"],
            X86_64 => &["x86_64", "x86_64h"],
            _ => anyhow::bail!("Native runtime dependency CPU family is unsupported"),
        };
        let mut selected = None;
        for name in names {
            if let Some(facts) = index.get(&(path.to_owned(), (*name).to_owned())) {
                anyhow::ensure!(
                    selected.is_none(),
                    "Native runtime cache dependency architecture is ambiguous"
                );
                selected = Some(facts);
            }
        }
        selected.context("Native runtime dependency has no matching cache image")
    }
    fn helper_dependencies(bytes: &[u8], architectures: &[u32]) -> Result<Vec<(u32, String)>> {
        let mut result = Vec::new();
        let mut slices = Vec::new();
        if word(bytes, 0, false)? == 0xfeed_facf {
            slices.push(bytes);
        } else {
            for index in 0..architectures.len() {
                let record = 8 + index * 20;
                let start = word(bytes, record + 8, true)? as usize;
                let length = word(bytes, record + 12, true)? as usize;
                slices.push(
                    bytes
                        .get(
                            start
                                ..start
                                    .checked_add(length)
                                    .context("Native runtime slice overflow")?,
                        )
                        .context("Native runtime slice is truncated")?,
                );
            }
        }
        for slice in slices {
            let architecture = thin(slice)?;
            current_architecture(architecture)?;
            let mut offset = 32_usize;
            for _ in 0..word(slice, 16, false)? {
                let command = word(slice, offset, false)?;
                let length = word(slice, offset + 4, false)? as usize;
                let record = slice
                    .get(
                        offset
                            ..offset
                                .checked_add(length)
                                .context("Native runtime command overflow")?,
                    )
                    .context("Native runtime command is truncated")?;
                if matches!(command, 0xc | 0x8000_001f) {
                    let start = word(record, 8, false)? as usize;
                    let name = record
                        .get(start..)
                        .context("Native runtime dependency is truncated")?;
                    let end = name
                        .iter()
                        .position(|byte| *byte == 0)
                        .context("Native runtime dependency is unterminated")?;
                    let name = std::str::from_utf8(&name[..end])
                        .context("Native runtime dependency is not UTF-8")?;
                    system_path(name)?;
                    result.push((architecture, name.to_owned()));
                }
                offset = offset
                    .checked_add(length)
                    .context("Native runtime command overflow")?;
            }
            result.push((architecture, "/usr/lib/libSystem.B.dylib".into()));
        }
        Ok(result)
    }
    async fn read_helper(
        filesystem: &dyn fs::Fs,
        helper: &crate::profile_results::TaskHelperIdentity,
    ) -> Result<Vec<u8>> {
        let path = helper.canonical_path();
        let parent = path
            .parent()
            .context("Native runtime helper parent is unavailable")?;
        let relative = path
            .file_name()
            .context("Native runtime helper name is unavailable")?;
        let root = filesystem
            .metadata(parent)
            .await?
            .context("Native runtime helper parent disappeared")?;
        let entry = filesystem
            .metadata(path)
            .await?
            .context("Native runtime helper disappeared")?;
        anyhow::ensure!(
            entry.len <= MAX_IMAGE_BYTES as u64,
            "Native runtime helper exceeds its byte limit"
        );
        let mut reader = filesystem
            .open_sync_confined_file(parent, Path::new(relative), root.inode, entry.inode)
            .await?
            .take(MAX_IMAGE_BYTES as u64 + 1);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        anyhow::ensure!(
            bytes.len() <= MAX_IMAGE_BYTES,
            "Native runtime helper exceeds its byte limit"
        );
        Ok(bytes)
    }
    fn observed_loaded_closure(
        dependencies: Vec<(u32, String)>,
        operating_system: String,
    ) -> Result<Option<SystemClosure>> {
        anyhow::ensure!(
            !dependencies.is_empty() && dependencies.len() <= MAX_CLOSURE_DEPENDENCIES,
            "Native runtime loaded closure root declarations exceed their limit"
        );
        for (_, path) in &dependencies {
            system_path(path)?;
        }
        if dependencies
            .iter()
            .any(|(architecture, _)| current_architecture(*architecture).is_err())
        {
            return Ok(None);
        }
        let mut requested = dependencies
            .into_iter()
            .map(|(_, path)| path)
            .collect::<std::collections::BTreeSet<_>>();
        anyhow::ensure!(
            !requested.is_empty() && requested.len() <= MAX_CLOSURE_IMAGES,
            "Native runtime loaded closure roots exceed their limit"
        );
        for path in &requested {
            system_path(path)?;
        }
        let mut absent_weak = std::collections::BTreeSet::new();
        let mut original: Option<(ActiveCacheIdentity, u32, u64, u64)> = None;
        for _ in 0..MAX_CLOSURE_IMAGES {
            let mapped = loaded_cache_snapshot_inner(
                &requested,
                LoadedCachePathEvidence::ObservedImmutableHeader,
            )?;
            if let Some((cache, version, size, timestamp)) = &original {
                anyhow::ensure!(
                    mapped.cache == *cache
                        && mapped.version == *version
                        && mapped.size == *size
                        && mapped.timestamp >= *timestamp,
                    "Native runtime loaded closure changed between bounded snapshots"
                );
            } else {
                original = Some((
                    mapped.cache.clone(),
                    mapped.version,
                    mapped.size,
                    mapped.timestamp,
                ));
            }
            if requested
                .iter()
                .any(|path| !mapped.images.contains_key(path))
            {
                return Ok(None);
            }
            let mut needed = requested.clone();
            let mut dependency_count = 0_usize;
            for image in mapped.images.values() {
                dependency_count = dependency_count
                    .checked_add(image.facts.dependencies.len())
                    .context("Native runtime loaded dependency count overflow")?;
                anyhow::ensure!(
                    dependency_count <= MAX_CLOSURE_DEPENDENCIES,
                    "Native runtime loaded closure exceeds its dependency limit"
                );
                for dependency in &image.facts.dependencies {
                    system_path(&dependency.path)?;
                    if dependency.attributes & 1 != 0
                        && cache_membership_if_available(&dependency.path)? == Some(false)
                    {
                        absent_weak.insert(dependency.path.clone());
                    } else {
                        needed.insert(dependency.path.clone());
                    }
                    anyhow::ensure!(
                        needed.len() <= MAX_CLOSURE_IMAGES
                            && absent_weak.len() <= MAX_CLOSURE_DEPENDENCIES,
                        "Native runtime loaded closure exceeds its image or weak-edge limit"
                    );
                }
            }
            if needed == requested {
                return Ok(Some(SystemClosure {
                    inspection: SystemClosureInspection::ObservedLoaded,
                    operating_system,
                    cache: mapped.cache,
                    mapped_version: mapped.version,
                    mapped_size: mapped.size,
                    mapped_epoch: mapped.timestamp,
                    images: mapped
                        .images
                        .into_values()
                        .map(|image| ReviewedSystemImage {
                            facts: image.facts,
                            origin: SystemImageOrigin::LoadedCache {
                                address: image.address,
                            },
                        })
                        .collect(),
                    absent_weak,
                }));
            }
            requested = needed;
        }
        anyhow::bail!("Native runtime loaded closure did not converge within its limit")
    }
    fn revalidate_observed_loaded_closure(closure: &SystemClosure) -> Result<()> {
        anyhow::ensure!(
            observed_operating_system_version()? == closure.operating_system,
            "Native runtime operating system cache policy changed"
        );
        let requested = closure
            .images
            .iter()
            .map(|image| image.facts.path.clone())
            .collect();
        let mapped = loaded_cache_snapshot_inner(
            &requested,
            LoadedCachePathEvidence::ObservedImmutableHeader,
        )?;
        anyhow::ensure!(
            mapped.cache == closure.cache
                && mapped.version == closure.mapped_version
                && mapped.size == closure.mapped_size
                && mapped.timestamp >= closure.mapped_epoch,
            "Native runtime observed loaded cache identity changed"
        );
        for image in &closure.images {
            let SystemImageOrigin::LoadedCache { address } = &image.origin else {
                anyhow::bail!("Native runtime loaded closure contains a different origin");
            };
            let current = mapped
                .images
                .get(&image.facts.path)
                .context("Native runtime loaded dependency disappeared")?;
            anyhow::ensure!(
                current.address == *address && current.facts == image.facts,
                "Native runtime immutable loaded dependency changed"
            );
        }
        for path in &closure.absent_weak {
            anyhow::ensure!(
                cache_membership_if_available(path)? == Some(false),
                "Native runtime absent cached weak dependency changed"
            );
        }
        Ok(())
    }
    pub(super) async fn prepare_system_closure(
        filesystem: &dyn fs::Fs,
        helper: &crate::profile_results::TaskHelperIdentity,
        architectures: &[u32],
        executor: &gpui::BackgroundExecutor,
    ) -> Result<SystemClosure> {
        let operating_system = observed_operating_system_version()?;
        let bytes = read_helper(filesystem, helper).await?;
        anyhow::ensure!(
            self::architectures(&bytes)? == architectures,
            "Native runtime helper slices changed during cache review"
        );
        if let Some(closure) = observed_loaded_closure(
            helper_dependencies(&bytes, architectures)?,
            operating_system,
        )? {
            revalidate_observed_loaded_closure(&closure)?;
            crate::profile_results::revalidate_task_helper_identity(
                filesystem,
                helper,
                &AtomicBool::new(false),
            )
            .await?;
            return Ok(closure);
        }
        if let Some(closure) = observed_cache_text_closure(
            helper_dependencies(&bytes, architectures)?,
            observed_operating_system_version()?,
        )? {
            revalidate_observed_cache_text_closure(&closure)?;
            crate::profile_results::revalidate_task_helper_identity(
                filesystem,
                helper,
                &AtomicBool::new(false),
            )
            .await?;
            return Ok(closure);
        }
        let operating_system = operating_system_version()?;
        let mapped = loaded_cache_snapshot(&std::collections::BTreeSet::new())?;
        let (discovery, tool) = prepare_tool(filesystem, executor).await?;
        let output = tool_output(
            tool.canonical_path(),
            &["-all_dyld_cache", "-uuid", "-linked_dylibs"],
            executor,
        )
        .await?;
        let index = parse_system_images(&output)?;
        let bytes = read_helper(filesystem, helper).await?;
        anyhow::ensure!(
            self::architectures(&bytes)? == architectures,
            "Native runtime helper slices changed during cache review"
        );
        let text_discovery = cache_text_discovery()?;
        let mut pending = helper_dependencies(&bytes, architectures)?;
        let mut visited = std::collections::BTreeSet::new();
        let mut images = Vec::new();
        let mut absent_weak = std::collections::BTreeSet::new();
        let mut dependency_count = 0_usize;
        let cancellation = AtomicBool::new(false);
        while let Some((architecture, path)) = pending.pop() {
            if !visited.insert((architecture, path.clone())) {
                continue;
            }
            anyhow::ensure!(
                visited.len() <= MAX_CLOSURE_IMAGES,
                "Native runtime system closure exceeds its image limit"
            );
            let (facts, origin) = if cache_contains(&path)? {
                let text = cache_text_snapshot_from_discovery(
                    &text_discovery,
                    &std::collections::BTreeSet::from([path.clone()]),
                )?;
                anyhow::ensure!(
                    text.cache == mapped.cache
                        && text.version == mapped.version
                        && text.size == mapped.size
                        && text.timestamp >= mapped.timestamp,
                    "Native runtime cache text identity changed during tool review"
                );
                if let Some(image) = text.images.get(&path) {
                    anyhow::ensure!(
                        cache_architecture_matches(architecture, &image.facts.architecture),
                        "Native runtime cache text dependency CPU family is unsupported"
                    );
                    (
                        image.facts.clone(),
                        SystemImageOrigin::CacheText {
                            address: image.address,
                            text_size: image.text_size,
                            canonical_path: image.canonical_path.clone(),
                        },
                    )
                } else if filesystem.metadata(Path::new(&path)).await?.is_some() {
                    let requested = std::collections::BTreeSet::from([path.clone()]);
                    let current = loaded_cache_snapshot(&requested)?;
                    anyhow::ensure!(
                        current.cache == mapped.cache
                            && current.version == mapped.version
                            && current.size == mapped.size
                            && current.timestamp >= mapped.timestamp,
                        "Native runtime loaded cache identity changed during review"
                    );
                    #[cfg(any(test, feature = "test-support"))]
                    if !current.images.contains_key(&path) {
                        match system_path(&path) {
                            Ok(()) => eprintln!(
                                "Native runtime immutable loaded image missing for system dependency: {path}"
                            ),
                            Err(_) => eprintln!(
                                "Native runtime immutable loaded image missing for a non-system dependency"
                            ),
                        }
                    }
                    let image = current.images.get(&path)
                        .context("Native runtime overlapping cache dependency is not an observed immutable loaded image")?;
                    anyhow::ensure!(
                        cache_architecture_matches(architecture, &image.facts.architecture),
                        "Native runtime loaded dependency CPU family is unsupported"
                    );
                    (
                        image.facts.clone(),
                        SystemImageOrigin::LoadedCache {
                            address: image.address,
                        },
                    )
                } else {
                    let facts = match select_cached(&index, &path, architecture) {
                        Ok(facts) => facts.clone(),
                        Err(_) => {
                            let requested = current_architecture(architecture)?;
                            let output = tool_output(
                                tool.canonical_path(),
                                &["-arch", requested, "-uuid", "-linked_dylibs", &path],
                                executor,
                            )
                            .await?;
                            let alias = parse_system_images(&output)?;
                            anyhow::ensure!(
                                alias.len() == 1,
                                "Native runtime cached alias slice is ambiguous"
                            );
                            select_cached(&alias, &path, architecture)?.clone()
                        }
                    };
                    anyhow::ensure!(
                        cache_contains(&path)?
                            && filesystem.metadata(Path::new(&path)).await?.is_none(),
                        "Native runtime cache-only dependency origin changed"
                    );
                    (facts, SystemImageOrigin::Cache)
                }
            } else {
                system_path(&path)?;
                let canonical = filesystem.canonicalize(Path::new(&path)).await?;
                system_path(
                    canonical
                        .to_str()
                        .context("Native runtime system identity is not UTF-8")?,
                )?;
                let metadata = filesystem
                    .metadata(&canonical)
                    .await?
                    .context("Native runtime system dependency is missing")?;
                anyhow::ensure!(
                    !metadata.is_dir && !metadata.is_fifo && metadata.len <= MAX_DISK_IMAGE_BYTES,
                    "Native runtime disk dependency exceeds its supported image policy"
                );
                let identity = bounded_apple_identity(filesystem, PathBuf::from(&path)).await?;
                system_path(
                    identity
                        .canonical_path()
                        .to_str()
                        .context("Native runtime system identity is not UTF-8")?,
                )?;
                validate_apple_code(identity.canonical_path(), None)?;
                let requested = current_architecture(architecture)?;
                let output = tool_output(
                    tool.canonical_path(),
                    &["-arch", requested, "-uuid", "-linked_dylibs", &path],
                    executor,
                )
                .await?;
                let disk = parse_system_images(&output)?;
                anyhow::ensure!(
                    disk.len() == 1,
                    "Native runtime disk dependency slice is ambiguous"
                );
                let facts = select_cached(&disk, &path, architecture)?.clone();
                anyhow::ensure!(
                    cache_architecture_matches(architecture, &facts.architecture),
                    "Native runtime disk dependency CPU family is unsupported"
                );
                crate::profile_results::revalidate_task_helper_identity(
                    filesystem,
                    &identity,
                    &cancellation,
                )
                .await?;
                (facts, SystemImageOrigin::Disk(identity))
            };
            let cached = matches!(
                &origin,
                SystemImageOrigin::Cache
                    | SystemImageOrigin::LoadedCache { .. }
                    | SystemImageOrigin::CacheText { .. }
            );
            dependency_count = dependency_count
                .checked_add(facts.dependencies.len())
                .context("Native runtime closure dependency count overflow")?;
            anyhow::ensure!(
                dependency_count <= MAX_CLOSURE_DEPENDENCIES,
                "Native runtime closure exceeds its dependency limit"
            );
            for dependency in &facts.dependencies {
                if cached && dependency.attributes & 1 != 0 && !cache_contains(&dependency.path)? {
                    // Apple's cached loader skips broken weak edges absent from its own active cache.
                    absent_weak.insert(dependency.path.clone());
                } else {
                    pending.push((architecture, dependency.path.clone()));
                    anyhow::ensure!(
                        pending.len() <= MAX_CLOSURE_DEPENDENCIES,
                        "Native runtime pending system dependencies exceed their limit"
                    );
                }
            }
            images.push(ReviewedSystemImage { facts, origin });
        }
        let closure = SystemClosure {
            inspection: SystemClosureInspection::Tool { discovery, tool },
            operating_system,
            cache: mapped.cache,
            mapped_version: mapped.version,
            mapped_size: mapped.size,
            mapped_epoch: mapped.timestamp,
            images,
            absent_weak,
        };
        revalidate_system_closure(filesystem, &closure, executor).await?;
        crate::profile_results::revalidate_task_helper_identity(filesystem, helper, &cancellation)
            .await?;
        Ok(closure)
    }
    pub(super) async fn revalidate_system_closure(
        filesystem: &dyn fs::Fs,
        closure: &SystemClosure,
        executor: &gpui::BackgroundExecutor,
    ) -> Result<()> {
        let (discovery, tool) = match &closure.inspection {
            SystemClosureInspection::ObservedLoaded => {
                return revalidate_observed_loaded_closure(closure);
            }
            SystemClosureInspection::ObservedCacheText => {
                return revalidate_observed_cache_text_closure(closure);
            }
            SystemClosureInspection::Tool { discovery, tool } => (discovery, tool),
        };
        anyhow::ensure!(
            operating_system_version()? == closure.operating_system,
            "Native runtime operating system cache policy changed"
        );
        let cancellation = AtomicBool::new(false);
        crate::profile_results::revalidate_task_helper_identity(
            filesystem,
            discovery,
            &cancellation,
        )
        .await?;
        crate::profile_results::revalidate_task_helper_identity(filesystem, tool, &cancellation)
            .await?;
        validate_apple_code(discovery.canonical_path(), Some("com.apple.xcrun"))?;
        validate_apple_code(tool.canonical_path(), Some("com.apple.dyld_info"))?;
        let selected = tool_output(
            discovery.canonical_path(),
            &["--find", "dyld_info"],
            executor,
        )
        .await?;
        anyhow::ensure!(
            selected.len() <= 4097,
            "Native runtime installed tool path exceeds its limit"
        );
        let selected =
            std::str::from_utf8(&selected).context("Native runtime tool selection is not UTF-8")?;
        let selected = selected.strip_suffix('\n').unwrap_or(selected);
        anyhow::ensure!(
            filesystem.canonicalize(Path::new(selected)).await? == tool.canonical_path(),
            "Native runtime selected Apple tool changed"
        );
        let requested = closure
            .images
            .iter()
            .filter(|image| matches!(&image.origin, SystemImageOrigin::LoadedCache { .. }))
            .map(|image| image.facts.path.clone())
            .collect();
        let mapped = loaded_cache_snapshot(&requested)?;
        anyhow::ensure!(
            mapped.cache == closure.cache
                && mapped.version == closure.mapped_version
                && mapped.size == closure.mapped_size
                && mapped.timestamp >= closure.mapped_epoch,
            "Native runtime active mapped cache identity changed"
        );
        let text_requested = closure
            .images
            .iter()
            .filter(|image| matches!(&image.origin, SystemImageOrigin::CacheText { .. }))
            .map(|image| image.facts.path.clone())
            .collect();
        let text_mapped = cache_text_snapshot(&text_requested)?;
        revalidate_cache_text_images(closure, &text_mapped)?;
        for image in &closure.images {
            if matches!(&image.origin, SystemImageOrigin::Cache) {
                anyhow::ensure!(
                    cache_contains(&image.facts.path)?
                        && filesystem
                            .metadata(Path::new(&image.facts.path))
                            .await?
                            .is_none(),
                    "Native runtime cache-only dependency changed before tool observation"
                );
            }
        }
        let output = tool_output(
            tool.canonical_path(),
            &["-all_dyld_cache", "-uuid", "-linked_dylibs"],
            executor,
        )
        .await?;
        let index = parse_system_images(&output)?;
        for image in &closure.images {
            match &image.origin {
                SystemImageOrigin::LoadedCache { address } => {
                    let current = mapped
                        .images
                        .get(&image.facts.path)
                        .context("Native runtime reviewed mapped dependency is unavailable")?;
                    anyhow::ensure!(
                        current.address == *address
                            && current.facts == image.facts
                            && cache_contains(&image.facts.path)?,
                        "Native runtime actual immutable cached dependency changed"
                    );
                }
                SystemImageOrigin::CacheText { .. } => {}
                SystemImageOrigin::Cache => {
                    anyhow::ensure!(
                        cache_contains(&image.facts.path)?
                            && filesystem
                                .metadata(Path::new(&image.facts.path))
                                .await?
                                .is_none(),
                        "Native runtime cache-only dependency origin changed"
                    );
                    let key = (image.facts.path.clone(), image.facts.architecture.clone());
                    if let Some(current) = index.get(&key) {
                        anyhow::ensure!(
                            current == &image.facts,
                            "Native runtime active cached dependency changed"
                        );
                    } else {
                        let requested = match image.facts.architecture.as_str() {
                            "arm64" | "arm64e" => current_architecture(ARM64)?,
                            "x86_64" | "x86_64h" => current_architecture(X86_64)?,
                            _ => anyhow::bail!(
                                "Native runtime cached alias architecture is unsupported"
                            ),
                        };
                        let output = tool_output(
                            tool.canonical_path(),
                            &[
                                "-arch",
                                requested,
                                "-uuid",
                                "-linked_dylibs",
                                &image.facts.path,
                            ],
                            executor,
                        )
                        .await?;
                        let current = parse_system_images(&output)?;
                        anyhow::ensure!(
                            current.len() == 1 && current.get(&key) == Some(&image.facts),
                            "Native runtime cached alias changed"
                        );
                    }
                    anyhow::ensure!(
                        cache_contains(&image.facts.path)?
                            && filesystem
                                .metadata(Path::new(&image.facts.path))
                                .await?
                                .is_none(),
                        "Native runtime cache-only dependency origin changed"
                    );
                }
                SystemImageOrigin::Disk(identity) => {
                    anyhow::ensure!(
                        !cache_contains(&image.facts.path)?,
                        "Native runtime system dependency origin changed"
                    );
                    crate::profile_results::revalidate_task_helper_identity(
                        filesystem,
                        identity,
                        &cancellation,
                    )
                    .await?;
                    validate_apple_code(identity.canonical_path(), None)?;
                }
            }
        }
        for path in &closure.absent_weak {
            anyhow::ensure!(
                !cache_contains(path)?,
                "Native runtime absent cached weak dependency changed"
            );
        }
        let renewed = loaded_cache_snapshot(&requested)?;
        anyhow::ensure!(
            renewed.cache == mapped.cache
                && renewed.version == mapped.version
                && renewed.size == mapped.size
                && renewed.timestamp >= mapped.timestamp,
            "Native runtime mapped cache changed after dependency review"
        );
        for (path, image) in &mapped.images {
            let current = renewed
                .images
                .get(path)
                .context("Native runtime mapped dependency was removed")?;
            anyhow::ensure!(
                current.address == image.address && current.facts == image.facts,
                "Native runtime mapped dependency changed after review"
            );
        }
        crate::profile_results::revalidate_task_helper_identity(
            filesystem,
            discovery,
            &cancellation,
        )
        .await?;
        let renewed_text = cache_text_snapshot(&text_requested)?;
        revalidate_cache_text_images(closure, &renewed_text)?;
        crate::profile_results::revalidate_task_helper_identity(filesystem, tool, &cancellation)
            .await?;
        Ok(())
    }

    #[cfg(test)]
    mod closure_tests {
        use super::*;
        fn image(path: &str, architecture: &str, dependencies: &str) -> String {
            format!(
                "{path} [{architecture}]:\n    -uuid:\n        4FED5EE2-5D3E-35B1-A170-9859C4B683BB\n    -linked_dylibs:\n        attributes     load path\n{dependencies}"
            )
        }
        #[test]
        fn cache_text_metadata_requires_used_origin_fields_and_nondecreasing_epoch() {
            let cache = ActiveCacheIdentity {
                base: 0x1000,
                length: 0x8000,
                uuid: uuid::Uuid::from_bytes([1; 16]),
            };
            let mut before = vec![0_u8; 192];
            before
                .get_mut(0..4)
                .expect("version field")
                .copy_from_slice(&15_u32.to_le_bytes());
            before
                .get_mut(4..8)
                .expect("count field")
                .copy_from_slice(&3_u32.to_le_bytes());
            before
                .get_mut(8..16)
                .expect("array field")
                .copy_from_slice(&0x2000_u64.to_le_bytes());
            before
                .get_mut(160..176)
                .expect("cache UUID field")
                .copy_from_slice(cache.uuid.as_bytes());
            before
                .get_mut(176..184)
                .expect("cache base field")
                .copy_from_slice(&cache.base.to_le_bytes());
            before
                .get_mut(184..192)
                .expect("epoch field")
                .copy_from_slice(&100_u64.to_le_bytes());
            let mut after = before.clone();
            after
                .get_mut(4..8)
                .expect("count field")
                .copy_from_slice(&4_u32.to_le_bytes());
            after
                .get_mut(8..16)
                .expect("array field")
                .copy_from_slice(&0x3000_u64.to_le_bytes());
            after
                .get_mut(184..192)
                .expect("epoch field")
                .copy_from_slice(&101_u64.to_le_bytes());
            let unchanged = cache_text_metadata_epoch(
                &cache,
                &cache,
                (0x4000, 192, &before),
                (0x4000, 192, &before),
            )
            .expect("unchanged used source fields");
            assert_eq!(unchanged, 100);
            assert_eq!(
                cache_text_metadata_epoch(
                    &cache,
                    &cache,
                    (0x4000, 192, &before),
                    (0x4000, 192, &after)
                )
                .expect("unrelated loaded-image array and monotone epoch"),
                101
            );
            for (offset, value) in [(0, 14_u32), (0, 16_u32)] {
                let mut changed = after.clone();
                changed
                    .get_mut(offset..offset + 4)
                    .expect("version field")
                    .copy_from_slice(&value.to_le_bytes());
                assert!(
                    cache_text_metadata_epoch(
                        &cache,
                        &cache,
                        (0x4000, 192, &before),
                        (0x4000, 192, &changed)
                    )
                    .is_err(),
                    "supported unchanged layout version is required"
                );
            }
            for offset in [24, 160, 176] {
                let mut changed = before.clone();
                *changed.get_mut(offset).expect("protected source field") ^= 1;
                assert!(
                    cache_text_metadata_epoch(
                        &cache,
                        &cache,
                        (0x4000, 192, &changed),
                        (0x4000, 192, &after)
                    )
                    .is_err(),
                    "before cache UUID/base/no-detach is required"
                );
                let mut changed = after.clone();
                *changed.get_mut(offset).expect("protected source field") ^= 1;
                assert!(
                    cache_text_metadata_epoch(
                        &cache,
                        &cache,
                        (0x4000, 192, &before),
                        (0x4000, 192, &changed)
                    )
                    .is_err(),
                    "after cache UUID/base/no-detach is required"
                );
            }
            let mut regressed = after.clone();
            regressed
                .get_mut(184..192)
                .expect("epoch field")
                .copy_from_slice(&99_u64.to_le_bytes());
            assert!(
                cache_text_metadata_epoch(
                    &cache,
                    &cache,
                    (0x4000, 192, &before),
                    (0x4000, 192, &regressed)
                )
                .is_err()
            );
            for (address, size) in [
                (0, 192),
                (0x5000, 192),
                (0x4000, 191),
                (0x4000, 193),
                (0x4000, 4097),
            ] {
                assert!(
                    cache_text_metadata_epoch(
                        &cache,
                        &cache,
                        (0x4000, 192, &before),
                        (address, size, &after)
                    )
                    .is_err(),
                    "same bounded metadata address and size is required"
                );
            }
            for bytes in [&before[..0], &before[..191]] {
                assert!(
                    cache_text_metadata_epoch(
                        &cache,
                        &cache,
                        (0x4000, 192, bytes),
                        (0x4000, 192, &after)
                    )
                    .is_err()
                );
                assert!(
                    cache_text_metadata_epoch(
                        &cache,
                        &cache,
                        (0x4000, 192, &before),
                        (0x4000, 192, bytes)
                    )
                    .is_err()
                );
            }
            for renewed in [
                ActiveCacheIdentity {
                    uuid: uuid::Uuid::nil(),
                    ..cache.clone()
                },
                ActiveCacheIdentity {
                    base: 0x2000,
                    ..cache.clone()
                },
                ActiveCacheIdentity {
                    length: 0x9000,
                    ..cache.clone()
                },
            ] {
                assert!(
                    cache_text_metadata_epoch(
                        &cache,
                        &renewed,
                        (0x4000, 192, &before),
                        (0x4000, 192, &after)
                    )
                    .is_err(),
                    "actual SDK UUID/base/range must renew exactly"
                );
            }
        }
        #[test]
        fn observed_cache_text_closure_renews_actual_mapped_origin_without_loaded_list() {
            #[cfg(target_arch = "aarch64")]
            let architecture = ARM64;
            #[cfg(target_arch = "x86_64")]
            let architecture = X86_64;
            let mut closure = observed_cache_text_closure(
                vec![(architecture, "/usr/lib/libSystem.B.dylib".into())],
                observed_operating_system_version().expect("actual OS"),
            )
            .expect("actual cache-text graph")
            .expect("complete current libSystem graph");
            assert!(matches!(
                &closure.inspection,
                SystemClosureInspection::ObservedCacheText
            ));
            assert!(!closure.images.is_empty());
            assert!(
                closure
                    .images
                    .iter()
                    .all(|image| matches!(&image.origin, SystemImageOrigin::CacheText { .. }))
            );
            revalidate_observed_cache_text_closure(&closure)
                .expect("same actual cache-text origin");
            closure.cache.uuid = uuid::Uuid::nil();
            assert!(
                revalidate_observed_cache_text_closure(&closure).is_err(),
                "wrong actual cache identity refuses"
            );
        }
        #[test]
        fn cache_text_mapping_uses_actual_canonical_api_and_refuses_unissued_paths() {
            let sdk = CacheTextSdk::current().expect("actual cache-text API");
            let path =
                "/System/Library/PrivateFrameworks/MLCompilerServices.framework/MLCompilerServices";
            let requested = std::collections::BTreeSet::from([path.to_owned()]);
            let mapped = cache_text_snapshot(&requested).expect("bounded actual cache mapping");
            if let Some(image) = mapped.images.get(path) {
                assert_eq!(
                    sdk.canonical_path(path).expect("actual alias API"),
                    Some(image.canonical_path.clone())
                );
                assert!(!image.facts.uuid.is_nil());
                assert_eq!(image.facts.path, path);
                assert!(image.address >= mapped.cache.base);
                let renewed = cache_text_snapshot(&requested).expect("actual source renewal");
                let current = renewed.images.get(path).expect("same alias still present");
                assert_eq!(current.canonical_path, image.canonical_path);
                assert_eq!(current.facts, image.facts);
                assert_eq!(current.address, image.address);
            } else {
                assert!(
                    sdk.canonical_path(path)
                        .expect("actual alias capability")
                        .is_none_or(|canonical| canonical == path),
                    "missing alias capability does not invent a versioned mapping"
                );
            }
            let missing = std::collections::BTreeSet::from([
                "/usr/lib/zed-unissued-cache-text-origin.dylib".to_owned(),
            ]);
            assert!(
                cache_text_snapshot(&missing)
                    .expect("bounded missing path")
                    .images
                    .is_empty()
            );
            let foreign = if cfg!(target_arch = "aarch64") {
                X86_64
            } else {
                ARM64
            };
            assert!(
                observed_cache_text_closure(
                    vec![(foreign, "/usr/lib/libSystem.B.dylib".into())],
                    observed_operating_system_version().expect("actual OS")
                )
                .expect("bounded foreign architecture")
                .is_none()
            );
            assert!(
                cache_text_snapshot(&std::collections::BTreeSet::from([
                    "/private/tmp/unissued.dylib".to_owned()
                ]))
                .is_err()
            );
        }
        #[test]
        fn cache_text_record_refuses_wrong_uuid_version_range_and_oversized_text() {
            let discovery = cache_text_discovery().expect("actual current-cache discovery");
            let text_sdk = CacheTextSdk::current().expect("actual API");
            let canonical = text_sdk
                .canonical_path("/usr/lib/libSystem.B.dylib")
                .expect("actual cache path")
                .expect("actual libSystem");
            let record = discovery
                .records
                .get(&canonical)
                .expect("actual cache record");
            let sdk = CacheSdkFunctions::current().expect("actual cache source");
            cache_text_record_image(&sdk, &discovery.cache, &canonical, record, &mut 0)
                .expect("actual immutable header matches the record");
            for (offset, replacement) in
                [(0, 1_u64), (16, MAX_DISK_IMAGE_BYTES + 1), (48, u64::MAX)]
            {
                let mut changed = record.clone();
                changed
                    .get_mut(offset..offset + 8)
                    .expect("fixed actual ABI field")
                    .copy_from_slice(&replacement.to_le_bytes());
                assert!(
                    cache_text_record_image(&sdk, &discovery.cache, &canonical, &changed, &mut 0)
                        .is_err(),
                    "unsupported ABI or outside actual cache never grants origin"
                );
            }
            let mut changed = record.clone();
            changed.get_mut(24..40).expect("actual UUID field").fill(0);
            assert!(
                cache_text_record_image(&sdk, &discovery.cache, &canonical, &changed, &mut 0)
                    .is_err(),
                "record UUID must match actual immutable header"
            );
            assert!(
                cache_text_record_image(
                    &sdk,
                    &discovery.cache,
                    "/usr/lib/zed-unissued-cache-text-origin.dylib",
                    record,
                    &mut 0
                )
                .is_err(),
                "claimed install-name must match actual header"
            );
            let mut aggregate_bytes = MAX_CACHE_OUTPUT_BYTES;
            assert!(
                cache_text_record_image(
                    &sdk,
                    &discovery.cache,
                    &canonical,
                    record,
                    &mut aggregate_bytes
                )
                .is_err(),
                "aggregate selected header bound remains enforced"
            );
        }
        #[test]
        fn observed_loaded_closure_uses_actual_process_identity_and_renews() {
            #[cfg(target_arch = "aarch64")]
            let architecture = ARM64;
            #[cfg(target_arch = "x86_64")]
            let architecture = X86_64;
            let closure = observed_loaded_closure(
                vec![(
                    architecture,
                    "/usr/lib/system/libsystem_kernel.dylib".into(),
                )],
                observed_operating_system_version().expect("Actual OS identity"),
            )
            .expect("Bounded actual loaded closure")
            .expect("Actual process-listed kernel closure is complete");
            assert!(matches!(
                &closure.inspection,
                SystemClosureInspection::ObservedLoaded
            ));
            assert!(!closure.images.is_empty());
            assert!(closure.images.len() <= MAX_CLOSURE_IMAGES);
            revalidate_observed_loaded_closure(&closure).expect("Actual immutable graph renewal");
            let mut changed = closure;
            changed.cache.uuid = uuid::Uuid::nil();
            assert!(
                revalidate_observed_loaded_closure(&changed).is_err(),
                "Wrong cache UUID cannot renew"
            );
        }
        #[test]
        fn observed_loaded_closure_refuses_unloaded_and_foreign_slice_roots() {
            #[cfg(target_arch = "aarch64")]
            let (architecture, foreign) = (ARM64, X86_64);
            #[cfg(target_arch = "x86_64")]
            let (architecture, foreign) = (X86_64, ARM64);
            let version = observed_operating_system_version().expect("Actual OS identity");
            assert!(
                observed_loaded_closure(
                    vec![(
                        architecture,
                        "/usr/lib/zed-unissued-native-dependency.dylib".into()
                    )],
                    version.clone()
                )
                .expect("Bounded missing loaded root")
                .is_none()
            );
            assert!(
                observed_loaded_closure(
                    vec![(foreign, "/usr/lib/libSystem.B.dylib".into())],
                    version
                )
                .expect("Bounded foreign slice")
                .is_none()
            );
        }
        #[test]
        fn observed_loaded_closure_refuses_empty_excess_and_non_system_roots() {
            let version = observed_operating_system_version().expect("Actual OS identity");
            assert!(observed_loaded_closure(Vec::new(), version.clone()).is_err());
            #[cfg(target_arch = "aarch64")]
            let architecture = ARM64;
            #[cfg(target_arch = "x86_64")]
            let architecture = X86_64;
            assert!(
                observed_loaded_closure(
                    vec![(architecture, "/private/tmp/unissued.dylib".into())],
                    version.clone()
                )
                .is_err()
            );
            let roots = (0..=MAX_CLOSURE_IMAGES)
                .map(|index| (architecture, format!("/usr/lib/unissued-{index}.dylib")))
                .collect();
            assert!(observed_loaded_closure(roots, version).is_err());
        }
        #[test]
        fn cache_graph_requires_complete_unambiguous_observed_headers() {
            let valid = image(
                "/usr/lib/libSystem.B.dylib",
                "arm64e",
                "        re-export      /usr/lib/system/libdyld.dylib\n",
            );
            let parsed = parse_system_images(valid.as_bytes()).expect("bounded actual-format data");
            assert_eq!(parsed.len(), 1);
            let facts = parsed.values().next().expect("one controlled node");
            assert_eq!(facts.architecture, "arm64e");
            assert_eq!(facts.dependencies[0].attributes, 2);
            for changed in [
                String::new(),
                valid.replace("arm64e", "unknown"),
                valid.replace("    -uuid:\n", ""),
                valid.replace(
                    "4FED5EE2-5D3E-35B1-A170-9859C4B683BB",
                    "00000000-0000-0000-0000-000000000000",
                ),
                valid.replace("    -linked_dylibs:", "    -rpaths:"),
                valid.clone() + &valid,
                valid.replace(
                    "/usr/lib/system/libdyld.dylib",
                    "/usr/lib/../private/foreign.dylib",
                ),
                valid.replace("re-export", "unknown"),
                valid.replace("re-export", "re-export weak-link"),
            ] {
                assert!(
                    parse_system_images(changed.as_bytes()).is_err(),
                    "invalid cache data must refuse"
                );
            }
        }
        #[test]
        fn cache_architecture_requires_actual_cpu_family_and_unique_image() {
            let arm = image("/usr/lib/libSystem.B.dylib", "arm64e", "");
            let parsed = parse_system_images(arm.as_bytes()).expect("controlled cache data");
            assert!(
                select_cached(&parsed, "/usr/lib/libSystem.B.dylib", ARM64).is_ok(),
                "arm64 executables use the genuine arm64e system cache"
            );
            assert!(
                select_cached(&parsed, "/usr/lib/libSystem.B.dylib", X86_64).is_err(),
                "the installed tool's ignored x86 request cannot certify an Intel slice"
            );
            let both = arm + &image("/usr/lib/libSystem.B.dylib", "arm64", "");
            let parsed = parse_system_images(both.as_bytes()).expect("both distinct data headers");
            assert!(
                select_cached(&parsed, "/usr/lib/libSystem.B.dylib", ARM64).is_err(),
                "one path cannot silently choose between ambiguous cache slices"
            );
            #[cfg(target_arch = "aarch64")]
            assert!(current_architecture(X86_64).is_err());
            #[cfg(target_arch = "x86_64")]
            assert!(current_architecture(ARM64).is_err());
        }
        #[test]
        fn cache_graph_preserves_optional_upward_and_delayed_edges() {
            let text = image(
                "/usr/lib/libSystem.B.dylib",
                "arm64e",
                "        upward delay-init weak-link    /usr/lib/system/libdyld.dylib\n        lazy-load    /usr/lib/system/libsystem_c.dylib\n",
            );
            let parsed =
                parse_system_images(text.as_bytes()).expect("controlled dependency attributes");
            let facts = parsed.values().next().expect("one controlled node");
            assert_eq!(facts.dependencies.len(), 2);
            assert_eq!(facts.dependencies[0].attributes, 1 | 4 | 8);
            assert_eq!(facts.dependencies[1].attributes, 16);
            assert!(
                parse_dependency("        weak-link weak-link /usr/lib/libSystem.B.dylib").is_err()
            );
            assert!(parse_dependency("        /usr/lib/libSystem.B.dylib\0").is_err());
        }
        #[test]
        fn cache_graph_limits_reject_before_unbounded_image_retention() {
            let mut text = String::new();
            for index in 0..=MAX_CACHE_IMAGES {
                text.push_str(&image(
                    &format!("/usr/lib/controlled-{index}.dylib"),
                    "arm64e",
                    "",
                ));
            }
            assert!(parse_system_images(text.as_bytes()).is_err());
            let oversized_path = format!("/usr/lib/{}", "x".repeat(4096));
            assert!(parse_dependency(&format!("        {oversized_path}")).is_err());
            assert!(parse_system_images(&vec![b' '; MAX_CACHE_OUTPUT_BYTES + 1]).is_err());
        }
        fn mapped_data_image(path: &str, dependency: Option<(u32, &str, Option<u32>)>) -> Vec<u8> {
            fn dylib(command: u32, path: &str, flags: Option<u32>) -> Vec<u8> {
                let start = if flags.is_some() { 28_usize } else { 24 };
                let length = (start + path.len() + 1).next_multiple_of(8);
                let mut record = vec![0_u8; length];
                record[..4].copy_from_slice(&command.to_le_bytes());
                record[4..8].copy_from_slice(&(length as u32).to_le_bytes());
                record[8..12].copy_from_slice(&(start as u32).to_le_bytes());
                if let Some(flags) = flags {
                    record[12..16].copy_from_slice(&0x1a74_1800_u32.to_le_bytes());
                    record[24..28].copy_from_slice(&flags.to_le_bytes());
                }
                record[start..start + path.len()].copy_from_slice(path.as_bytes());
                record
            }
            let mut records = vec![0_u8; 24];
            records[..4].copy_from_slice(&0x1b_u32.to_le_bytes());
            records[4..8].copy_from_slice(&24_u32.to_le_bytes());
            records[8..24].copy_from_slice(uuid::Uuid::from_u128(1).as_bytes());
            records.extend(dylib(0xd, path, None));
            if let Some((command, path, flags)) = dependency {
                records.extend(dylib(command, path, flags));
            }
            let mut result = vec![0_u8; 32];
            result[..4].copy_from_slice(&0xfeed_facf_u32.to_le_bytes());
            #[cfg(target_arch = "aarch64")]
            let (cpu, subtype) = (ARM64, 0x8000_0002_u32);
            #[cfg(target_arch = "x86_64")]
            let (cpu, subtype) = (X86_64, 3_u32);
            result[4..8].copy_from_slice(&cpu.to_le_bytes());
            result[8..12].copy_from_slice(&subtype.to_le_bytes());
            result[12..16].copy_from_slice(&6_u32.to_le_bytes());
            result[16..20]
                .copy_from_slice(&(if dependency.is_some() { 3_u32 } else { 2 }).to_le_bytes());
            result[20..24].copy_from_slice(&(records.len() as u32).to_le_bytes());
            result[24..28].copy_from_slice(&0x8000_0000_u32.to_le_bytes());
            result.extend(records);
            result
        }
        #[test]
        fn mapped_cache_header_requires_exact_origin_uuid_path_and_commands() {
            let path = "/usr/lib/system/libsystem_kernel.dylib";
            let valid = mapped_data_image(path, None);
            let parsed = cached_header_facts(path, &valid).expect("controlled mapped HEADER data");
            assert_eq!(parsed.path, path);
            assert_eq!(parsed.uuid, uuid::Uuid::from_u128(1));
            assert!(parsed.dependencies.is_empty());
            assert!(
                cached_header_facts("/usr/lib/system/libsystem_pthread.dylib", &valid).is_err()
            );
            for (offset, value) in [
                (0, 0_u32),
                (12, 2),
                (24, 0),
                (4, 0),
                (16, 4097),
                (20, 256 * 1024 + 1),
            ] {
                let mut changed = valid.clone();
                changed[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                assert!(cached_header_facts(path, &changed).is_err());
            }
            let mut changed = valid.clone();
            changed[40..56].fill(0);
            assert!(cached_header_facts(path, &changed).is_err());
            for end in [0, 31, valid.len() - 1] {
                assert!(cached_header_facts(path, &valid[..end]).is_err());
            }
            let mut trailing = valid;
            trailing.push(0);
            assert!(cached_header_facts(path, &trailing).is_err());
        }
        #[test]
        fn mapped_cache_dependencies_preserve_weak_reexport_and_modern_flags() {
            let path = "/usr/lib/libSystem.B.dylib";
            let dependency = "/usr/lib/system/libsystem_c.dylib";
            for (command, expected) in [
                (0xc, 0),
                (0x8000_0018, 1),
                (0x8000_001f, 2),
                (0x8000_0023, 4),
                (0x20, 16),
            ] {
                let data = mapped_data_image(path, Some((command, dependency, None)));
                let parsed = cached_header_facts(path, &data)
                    .expect("controlled original dependency encoding");
                assert_eq!(
                    parsed.dependencies,
                    vec![SystemDependency {
                        path: dependency.into(),
                        attributes: expected
                    }]
                );
            }
            let data = mapped_data_image(path, Some((0xc, dependency, Some(1 | 4 | 8))));
            let parsed = cached_header_facts(path, &data)
                .expect("controlled current SDK alternate encoding");
            assert_eq!(parsed.dependencies[0].attributes, 1 | 4 | 8);
            for flags in [16, 1 | 2, 2 | 8] {
                assert!(
                    cached_header_facts(
                        path,
                        &mapped_data_image(path, Some((0xc, dependency, Some(flags))))
                    )
                    .is_err()
                );
            }
            let escaped = mapped_data_image(path, Some((0xc, "/usr/lib/../foreign.dylib", None)));
            assert!(cached_header_facts(path, &escaped).is_err());
        }
        #[test]
        fn mapped_process_reads_refuse_invalid_bounds_before_kernel_access() {
            for (address, size) in [(0_u64, 32), (1, 0), (1, 256 * 1024 + 33), (u64::MAX, 32)] {
                assert!(read_process_memory(address, size).is_err());
            }
        }
        #[test]
        fn apple_cache_sdk_rejects_unsigned_actual_tool_role() {
            let directory =
                tempfile::tempdir_in("/private/tmp").expect("private actual unsigned tool fixture");
            let path = directory.path().join("dyld_info");
            std::fs::write(&path, b"not a signed installed Apple tool")
                .expect("owned refusal input");
            assert!(validate_apple_code(&path, Some("com.apple.dyld_info")).is_err());
            assert!(validate_apple_code(&path, Some("com.apple.xcrun")).is_err());
            assert!(validate_apple_code(&path, Some("caller-role")).is_err());
        }
    }

    pub(super) async fn validate(
        filesystem: &dyn fs::Fs,
        app: &crate::profile_results::TaskHelperIdentity,
        helper: &crate::profile_results::TaskHelperIdentity,
    ) -> Result<(Vec<u32>, Signer)> {
        anyhow::ensure!(
            !filesystem.is_fake(),
            "Native runtime signing requires actual installed files"
        );
        let cancellation = AtomicBool::new(false);
        crate::profile_results::revalidate_task_helper_identity(filesystem, app, &cancellation)
            .await?;
        crate::profile_results::revalidate_task_helper_identity(filesystem, helper, &cancellation)
            .await?;
        let path = helper.canonical_path();
        let parent = path
            .parent()
            .context("Native runtime helper parent is unavailable")?;
        let relative = path
            .file_name()
            .context("Native runtime helper name is unavailable")?;
        let root = filesystem
            .metadata(parent)
            .await?
            .context("Native runtime helper parent disappeared")?;
        let entry = filesystem
            .metadata(path)
            .await?
            .context("Native runtime helper disappeared")?;
        anyhow::ensure!(
            entry.len <= MAX_IMAGE_BYTES as u64,
            "Native runtime helper exceeds its byte limit"
        );
        let mut reader = filesystem
            .open_sync_confined_file(parent, Path::new(relative), root.inode, entry.inode)
            .await?
            .take(MAX_IMAGE_BYTES as u64 + 1);
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        let architectures = architectures(&bytes)?;
        let app_signer = validate_code(app.canonical_path(), None, false)?;
        for architecture in &architectures {
            anyhow::ensure!(
                validate_code(path, Some(*architecture), true)? == app_signer,
                "Native runtime helper signer differs from the installed App"
            );
        }
        crate::profile_results::revalidate_task_helper_identity(filesystem, app, &cancellation)
            .await?;
        crate::profile_results::revalidate_task_helper_identity(filesystem, helper, &cancellation)
            .await?;
        Ok((architectures, app_signer))
    }
}

#[cfg(target_os = "macos")]
impl LocalNativeTaskHelperImage {
    pub fn helper_path(&self) -> &std::path::Path {
        self.identity.canonical_path()
    }

    #[cfg(any(feature = "cargo-workspace", test))]
    pub(crate) fn same_issuance(&self, other: &Self) -> bool {
        self.project == other.project
            && self.store == other.store
            && Arc::ptr_eq(&self.epoch, &other.epoch)
            && self.helper == other.helper
            && self.identity.canonical_path() == other.identity.canonical_path()
            && self.identity.fingerprint() == other.identity.fingerprint()
            && self.owner_helper == other.owner_helper
            && match (&self.purpose, &other.purpose) {
                (
                    NativeTaskHelperImagePurpose::Application,
                    NativeTaskHelperImagePurpose::Application,
                ) => true,
                (
                    NativeTaskHelperImagePurpose::CargoRuntime(left),
                    NativeTaskHelperImagePurpose::CargoRuntime(right),
                ) => Arc::ptr_eq(left, right),
                _ => false,
            }
    }

    fn duplicate(&self) -> Self {
        Self {
            owner_helper: self.owner_helper.clone(),
            purpose: self.purpose.clone(),
            project: self.project,
            store: self.store,
            epoch: self.epoch.clone(),
            helper: self.helper.clone(),
            identity: self.identity.clone(),
        }
    }
}

#[cfg(target_os = "macos")]
pub struct BuildTaskReportBinding {
    runtime_admission: Option<Arc<BuildReportRuntimeAdmission>>,
    helper_image: Option<LocalNativeTaskHelperImage>,
    mode: BuildReportMode,
    expected_invocation: Option<util::macos_task_completion::TaskBuildInvocation>,
    store: gpui::EntityId,
    key: ProfileTaskKey,
    generation: u64,
    helper: ProfileTaskHelper,
    identity: crate::profile_results::TaskHelperIdentity,
    socket: PathBuf,
    token: [u8; 32],
    reservation: Arc<BuildReportReservation>,
    cancellation: Arc<AtomicBool>,
    stop: async_channel::Sender<()>,
}

#[cfg(target_os = "macos")]
impl BuildTaskReportBinding {
    pub fn requires_cargo_runtime_observation(&self) -> bool {
        self.mode == BuildReportMode::CheckedCargoRuntime
    }
    pub fn requires_checked_invocation(&self) -> bool {
        self.expected_invocation.is_some()
    }
    pub fn run_id(&self) -> &str {
        &self.key.1
    }
    pub fn helper_path(&self) -> &std::path::Path {
        self.identity.canonical_path()
    }
    pub fn helper_mode(&self) -> ProfileTaskHelperMode {
        self.helper.mode
    }
    pub fn configured_helper_fingerprint(&self) -> [u8; 32] {
        self.identity.fingerprint()
    }
    pub fn socket_path(&self) -> &std::path::Path {
        &self.socket
    }
    pub fn token(&self) -> &[u8; 32] {
        &self.token
    }
    pub fn report_limit(&self) -> usize {
        self.reservation.limit
    }
}

#[cfg(target_os = "macos")]
impl Drop for BuildTaskReportBinding {
    fn drop(&mut self) {
        self.cancellation.store(true, Ordering::Relaxed);
        self.stop.close();
    }
}

#[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
enum SelectedNativeCargoIntent {
    Build(Box<crate::cargo_workspace_store::CargoNativeBuildIntent>),
    Runtime(Box<crate::cargo_workspace_store::CargoNativeRuntimeIntent>),
}

#[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
impl SelectedNativeCargoIntent {
    fn program(&self) -> &std::path::Path {
        match self {
            Self::Build(intent) => intent.program(),
            Self::Runtime(intent) => intent.program(),
        }
    }
    fn arguments(&self) -> &[String] {
        match self {
            Self::Build(intent) => intent.arguments(),
            Self::Runtime(intent) => intent.arguments(),
        }
    }
    fn working_directory(&self) -> &std::path::Path {
        match self {
            Self::Build(intent) => intent.working_directory(),
            Self::Runtime(intent) => intent.working_directory(),
        }
    }
}

#[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
enum NativeTerminalTaskAuthority {
    CargoRuntime(Arc<BuildReportRuntimeAdmission>),
    Capture {
        authority: NativeCaptureAuthority,
        invocation: Arc<NativePairInvocation>,
    },
}

#[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
pub(crate) struct NativeTerminalTaskAdmission {
    project: gpui::EntityId,
    store: gpui::WeakEntity<ProfileResultsStore>,
    key: ProfileTaskKey,
    generation: u64,
    cancellation: Arc<AtomicBool>,
    helper_image: LocalNativeTaskHelperImage,
    authority: NativeTerminalTaskAuthority,
    launch: task::SpawnInTerminal,
    environment: HashMap<String, String>,
}

#[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
impl NativeTerminalTaskAdmission {
    fn from_issued(
        store: gpui::WeakEntity<ProfileResultsStore>,
        key: ProfileTaskKey,
        generation: u64,
        cancellation: Arc<AtomicBool>,
        helper_image: LocalNativeTaskHelperImage,
        authority: NativeTerminalTaskAuthority,
        launch: &mut task::SpawnInTerminal,
    ) -> Result<Self> {
        let expected = match &authority {
            NativeTerminalTaskAuthority::CargoRuntime(admission) => {
                anyhow::ensure!(
                    admission.image.same_issuance(&helper_image)
                        && admission.lease.matches_helper_image(&helper_image)
                        && admission.lease.run_id().to_string() == key.1,
                    "Native terminal runtime issuance changed"
                );
                admission.lease.expected_invocation()
            }
            NativeTerminalTaskAuthority::Capture { invocation, .. } => &invocation.expected,
        };
        expected.validate().map_err(|_| {
            anyhow::anyhow!("Native terminal invocation exceeds its admitted bounds")
        })?;
        anyhow::ensure!(
            helper_image.store == store.entity_id()
                && key.0 == ProfileTaskActor::Local(store.entity_id())
                && launch.command.as_deref() == helper_image.helper_path().to_str()
                && launch.cwd.as_deref() == Some(expected.working_directory())
                && launch.literal_prefix_args == launch.args.len(),
            "Native terminal helper recipe differs from its issuing owner"
        );
        let mut environment = expected
            .environment()
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect::<HashMap<_, _>>();
        for name in [
            util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV,
            util::macos_task_completion::TASK_BUILD_INVOCATION_ENV,
            util::macos_task_completion::TASK_NATIVE_INVOCATION_ENV,
        ] {
            if let Some(value) = launch.env.get(name) {
                environment.insert(name.to_owned(), value.clone());
            }
        }
        anyhow::ensure!(
            environment.contains_key(util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV)
                && match &authority {
                    NativeTerminalTaskAuthority::CargoRuntime(_) => {
                        environment
                            .contains_key(util::macos_task_completion::TASK_BUILD_INVOCATION_ENV)
                            && !environment.contains_key(
                                util::macos_task_completion::TASK_NATIVE_INVOCATION_ENV,
                            )
                    }
                    NativeTerminalTaskAuthority::Capture { .. } => {
                        environment
                            .contains_key(util::macos_task_completion::TASK_NATIVE_INVOCATION_ENV)
                            && !environment.contains_key(
                                util::macos_task_completion::TASK_BUILD_INVOCATION_ENV,
                            )
                    }
                },
            "Native terminal private invocation mode changed"
        );
        launch.env.clear();
        Ok(Self {
            project: helper_image.project,
            store,
            key,
            generation,
            cancellation,
            helper_image,
            authority,
            launch: launch.clone(),
            environment,
        })
    }

    pub(crate) fn validate_launch(&self, launch: &task::SpawnInTerminal) -> Result<()> {
        anyhow::ensure!(
            launch == &self.launch && launch.env.is_empty(),
            "Native terminal logical launch changed after owner preparation"
        );
        Ok(())
    }

    fn check_current(&self, store: &ProfileResultsStore, cx: &App) -> Result<()> {
        anyhow::ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling experimental opt-in was disabled"
        );
        anyhow::ensure!(
            !self.cancellation.load(Ordering::Relaxed)
                && store.helper_image_current(&self.helper_image, cx),
            "Native terminal image or cancellation owner changed"
        );
        let run = store
            .task_runs
            .get(&self.key)
            .filter(|run| {
                run.generation == self.generation
                    && !run.failed
                    && Arc::ptr_eq(&run.cancellation, &self.cancellation)
                    && !run.cancellation.load(Ordering::Relaxed)
            })
            .context("Native terminal producing Task owner expired")?;
        match (&self.authority, &run.payload) {
            (
                NativeTerminalTaskAuthority::CargoRuntime(admission),
                ProfileTaskPayload::BuildReport(build),
            ) => {
                anyhow::ensure!(
                    build.mode == BuildReportMode::CheckedCargoRuntime
                        && build
                            .runtime_admission
                            .as_ref()
                            .is_some_and(|current| Arc::ptr_eq(current, admission))
                        && admission.image.same_issuance(&self.helper_image)
                        && build.expected_invocation.as_ref()
                            == Some(admission.lease.expected_invocation()),
                    "Native terminal runtime admission changed"
                );
            }
            (
                NativeTerminalTaskAuthority::Capture {
                    authority,
                    invocation,
                },
                ProfileTaskPayload::NativePair(native),
            ) => {
                anyhow::ensure!(
                    native
                        .checked_invocation
                        .as_ref()
                        .is_some_and(|current| Arc::ptr_eq(current, invocation))
                        && native
                            .capture_admission
                            .as_ref()
                            .is_some_and(|(current, helper)| {
                                Arc::ptr_eq(current, authority)
                                    && helper == &self.helper_image.helper
                            }),
                    "Native terminal Capture admission changed"
                );
            }
            _ => anyhow::bail!("Native terminal producing Task kind changed"),
        }
        Ok(())
    }

    pub(crate) fn renew_for_terminal(
        mut self,
        project: Entity<crate::Project>,
        launch: task::SpawnInTerminal,
        cx: &mut AsyncApp,
    ) -> Result<Task<Result<crate::terminals::NativeTerminalSpawnData>>> {
        self.validate_launch(&launch)?;
        anyhow::ensure!(
            project.entity_id() == self.project,
            "Native terminal Project or transport owner changed"
        );
        project.read_with(cx, |project, _cx| {
            anyhow::ensure!(
                project.is_local()
                    && !project.is_shared()
                    && project
                        .profile_results_store
                        .as_ref()
                        .is_some_and(|store| store.entity_id() == self.store.entity_id()),
                "Native terminal Project or transport owner changed"
            );
            Ok::<_, anyhow::Error>(())
        })?;
        let store = self
            .store
            .upgrade()
            .context("Native terminal producing owner disappeared")?;
        store.read_with(cx, |store, cx| self.check_current(store, cx))?;
        let renewed = store.update(cx, |store, cx| match &self.authority {
            NativeTerminalTaskAuthority::CargoRuntime(_) => {
                store.check_build_report_registration(&self.key, self.generation, cx)
            }
            NativeTerminalTaskAuthority::Capture { .. } => {
                store.check_native_capture_registration(&self.key, self.generation, cx)
            }
        })?;
        Ok(cx.spawn(async move |cx| {
            renewed.await?;
            let renewed_image = project.update(cx, |project, cx| match &self.authority {
                NativeTerminalTaskAuthority::CargoRuntime(_) => project
                    .revalidate_local_native_cargo_runtime_helper_image(&self.helper_image, cx),
                NativeTerminalTaskAuthority::Capture { .. } => {
                    project.revalidate_local_native_task_helper_image(&self.helper_image, cx)
                }
            })?;
            renewed_image.await?;
            store.read_with(cx, |store, cx| self.check_current(store, cx))?;
            let data = project.read_with(cx, |project, cx| {
                anyhow::ensure!(
                    project.is_local()
                        && !project.is_shared()
                        && project.native_task_helper_image_current(
                            &self.helper_image,
                            self.project,
                            cx
                        ),
                    "Native terminal original Project image owner changed during renewal"
                );
                match &self.authority {
                    NativeTerminalTaskAuthority::CargoRuntime(admission) => {
                        admission.lease.validate_outer_loader_environment()?;
                    }
                    NativeTerminalTaskAuthority::Capture {
                        authority,
                        invocation,
                    } => {
                        let authority = authority.lock();
                        let plan = authority
                            .as_ref()
                            .context("Native capture loader authority expired")?;
                        plan.validate_terminal_loader_environment(&invocation.expected)?;
                    }
                }
                let uses_pty = !terminal::HeadlessTerminal::is_enabled(cx);
                let environment = terminal::NativeTerminalEnvironment::for_spawn(
                    std::mem::take(&mut self.environment),
                    self.project.as_u64(),
                    uses_pty,
                )?;
                Ok(crate::terminals::NativeTerminalSpawnData {
                    program: launch
                        .command
                        .context("Native terminal direct helper disappeared")?,
                    arguments: launch.args,
                    working_directory: launch
                        .cwd
                        .context("Native terminal exact cwd disappeared")?,
                    environment,
                })
            })?;
            if matches!(
                &self.authority,
                NativeTerminalTaskAuthority::CargoRuntime(_)
            ) {
                store.update(cx, |store, cx| {
                    self.check_current(store, cx)?;
                    let run = store
                        .task_runs
                        .get_mut(&self.key)
                        .context("Native terminal producing Task owner disappeared")?;
                    let ProfileTaskPayload::BuildReport(build) = &mut run.payload else {
                        anyhow::bail!("Native terminal producing Task kind changed");
                    };
                    let arm = build
                        .observation_arm
                        .take()
                        .context("Native runtime observation was already armed")?;
                    arm.try_send(()).map_err(|_| {
                        anyhow::anyhow!(
                            "Native runtime observation owner disappeared before launch"
                        )
                    })
                })?;
            }
            Ok(data)
        }))
    }
}

pub struct PreparedNativeCargoBuildTask {
    launch: task::SpawnInTerminal,
    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    terminal_admission: Option<NativeTerminalTaskAdmission>,
    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    intent: SelectedNativeCargoIntent,
    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    binding: BuildTaskReportBinding,
}

impl PreparedNativeCargoBuildTask {
    pub fn launch(&self) -> &task::SpawnInTerminal {
        &self.launch
    }

    pub fn take_native_terminal_launch(
        &mut self,
    ) -> Result<Option<crate::terminals::PreparedNativeTerminalLaunch>> {
        #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
        if matches!(&self.intent, SelectedNativeCargoIntent::Runtime(_)) {
            let admission = self
                .terminal_admission
                .take()
                .context("Native terminal runtime admission was already consumed")?;
            admission.validate_launch(&self.launch)?;
            return Ok(Some(
                crate::terminals::PreparedNativeTerminalLaunch::from_admission(
                    admission.project,
                    admission,
                ),
            ));
        }
        Ok(None)
    }
}

#[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
struct LocalNativePairBinding {
    checked_invocation: Arc<NativePairInvocation>,
    key: ProfileTaskKey,
    generation: u64,
    helper: ProfileTaskHelper,
    binding: ProfileTaskBinding,
    cancellation: Arc<AtomicBool>,
    stop: async_channel::Sender<()>,
    armed: bool,
}
#[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
impl Drop for LocalNativePairBinding {
    fn drop(&mut self) {
        if self.armed {
            self.cancellation.store(true, Ordering::Relaxed);
            self.stop.close();
        }
    }
}

#[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
async fn check_native_capture_outputs(
    filesystem: Arc<dyn fs::Fs>,
    roots: Vec<(PathBuf, PathBuf, u64, u64)>,
    paths: Vec<PathBuf>,
    executor: gpui::BackgroundExecutor,
) -> Result<()> {
    smol::future::race(
        async move {
            for (root, parent, root_inode, parent_inode) in roots {
                let root_metadata = filesystem
                    .metadata(&root)
                    .await?
                    .context("Native capture root disappeared")?;
                let parent_metadata = filesystem
                    .metadata(&parent)
                    .await?
                    .context("Native capture parent disappeared")?;
                anyhow::ensure!(
                    root_metadata.is_dir
                        && !root_metadata.is_symlink
                        && root_metadata.inode == root_inode
                        && parent_metadata.is_dir
                        && !parent_metadata.is_symlink
                        && parent_metadata.inode == parent_inode
                        && filesystem.canonicalize(&root).await? == root
                        && filesystem.canonicalize(&parent).await? == parent,
                    "Native capture output root changed"
                );
            }
            for path in paths {
                anyhow::ensure!(
                    filesystem.metadata(&path).await?.is_none(),
                    "Native capture output already exists"
                );
            }
            Ok(())
        },
        async move {
            executor.timer(Duration::from_secs(30)).await;
            Err(anyhow::anyhow!("Native capture output admission timed out"))
        },
    )
    .await
}

#[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
type NativeCaptureAuthority =
    Arc<Mutex<Option<crate::cargo_workspace_store::NativeCargoCapturePlan>>>;

pub struct PreparedNativeCapturePlan {
    resolved: task::ResolvedTask,
    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    authority: crate::cargo_workspace_store::NativeCargoCapturePlan,
}
impl PreparedNativeCapturePlan {
    pub fn resolved_task(&self) -> &task::ResolvedTask {
        &self.resolved
    }
}
pub struct PreparedNativeCaptureTask {
    launch: task::SpawnInTerminal,
    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    native_terminal_required: bool,
    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    terminal_admission: Option<NativeTerminalTaskAdmission>,
    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    authority: NativeCaptureAuthority,
    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    binding: LocalNativePairBinding,
}
impl PreparedNativeCaptureTask {
    pub fn launch(&self) -> &task::SpawnInTerminal {
        &self.launch
    }

    pub fn take_native_terminal_launch(
        &mut self,
    ) -> Result<Option<crate::terminals::PreparedNativeTerminalLaunch>> {
        #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
        {
            if !self.native_terminal_required {
                return Ok(None);
            }
            let admission = self
                .terminal_admission
                .take()
                .context("Native terminal Capture admission was already consumed")?;
            admission.validate_launch(&self.launch)?;
            return Ok(Some(
                crate::terminals::PreparedNativeTerminalLaunch::from_admission(
                    admission.project,
                    admission,
                ),
            ));
        }
        #[cfg(not(all(target_os = "macos", feature = "cargo-workspace")))]
        Ok(None)
    }
}

pub struct NativeCargoWorkloadReceipt {
    _private: (),
    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    workload: crate::cargo_workspace_store::NativeCargoWorkload,
}

impl NativeCargoWorkloadReceipt {
    pub fn executable_path(&self) -> Result<PathBuf> {
        #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
        {
            Ok(self.workload.executable_path())
        }
        #[cfg(not(all(target_os = "macos", feature = "cargo-workspace")))]
        {
            Err(anyhow::anyhow!(
                "Selected native Cargo workloads are unavailable on this host"
            ))
        }
    }
}

#[cfg(target_os = "macos")]
pub struct ObservedBuildTaskReport {
    cargo_runtime: Option<Arc<util::macos_task_completion::ObservedCargoRuntime>>,
    invocation: Option<Arc<util::macos_task_completion::ObservedTaskBuildInvocation>>,
    bytes: Arc<Vec<u8>>,
    fingerprint: [u8; 32],
    binding: BuildTaskReportBinding,
}

#[cfg(target_os = "macos")]
impl ObservedBuildTaskReport {
    #[cfg(feature = "cargo-workspace")]
    pub(crate) fn runtime_admission_matches(
        &self,
        lease: &crate::cargo_workspace_store::NativeCargoRuntimeAdmissionLease,
    ) -> Result<bool> {
        self.bytes()?;
        Ok(self.binding.mode == BuildReportMode::CheckedCargoRuntime
            && self
                .binding
                .runtime_admission
                .as_ref()
                .is_some_and(|admission| admission.lease.same_issuance(lease)))
    }

    pub fn observed_cargo_runtime(
        &self,
    ) -> Result<Option<&util::macos_task_completion::ObservedCargoRuntime>> {
        self.bytes()?;
        Ok(self.cargo_runtime.as_deref())
    }
    pub fn observed_invocation(
        &self,
    ) -> Result<Option<&util::macos_task_completion::ObservedTaskBuildInvocation>> {
        self.bytes()?;
        Ok(self.invocation.as_deref())
    }
    pub fn run_id(&self) -> &str {
        self.binding.run_id()
    }
    pub fn bytes(&self) -> Result<&[u8]> {
        anyhow::ensure!(
            !self.binding.cancellation.load(Ordering::Relaxed),
            "Build Task report lease was retired"
        );
        Ok(&self.bytes)
    }
    pub fn fingerprint(&self) -> [u8; 32] {
        self.fingerprint
    }
    pub fn configured_helper_fingerprint(&self) -> [u8; 32] {
        self.binding.configured_helper_fingerprint()
    }
}

impl Drop for ProfileTaskRun {
    fn drop(&mut self) {
        self.cancellation.store(true, Ordering::Relaxed);
        self.stop.close();
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProfileImportPurpose {
    Diagnostic,
    Native,
}

struct ProfileImport {
    purpose: ProfileImportPurpose,
    task_run_id: Option<String>,
    generation: u64,
    cancellation: Arc<AtomicBool>,
    paths: HashSet<ProjectPath>,
    pages: VecDeque<proto::ProfileResultPage>,
    next_page: u32,
    revision: u64,
}

struct RemoteProfileProxy {
    client: AnyProtoClient,
    project_id: u64,
}

struct ProxiedImport {
    remote_id: u64,
    generation: u64,
    next_page: u32,
    total_bytes: usize,
    paths: HashSet<ProjectPath>,
    cancellation: RemoteImportCancellation,
}

impl Drop for ProfileImport {
    fn drop(&mut self) {
        self.cancellation.store(true, Ordering::Relaxed);
    }
}

struct ImportCancellation(Option<Arc<AtomicBool>>);

impl ImportCancellation {
    fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for ImportCancellation {
    fn drop(&mut self) {
        if let Some(cancellation) = &self.0 {
            cancellation.store(true, Ordering::Relaxed);
        }
    }
}

pub const DIAGNOSTIC_PROFILE_BYTES: usize = 10 * 1024 * 1024;

pub struct DiagnosticProfileImport {
    store: gpui::EntityId,
    paths: [ProjectPath; 2],
    entries: [worktree::Entry; 2],
    scans: Vec<(settings::WorktreeId, usize)>,
    inputs: [ProfileInput; 2],
}

impl DiagnosticProfileImport {
    pub fn paths(&self) -> &[ProjectPath; 2] {
        &self.paths
    }
    pub fn input_identity(&self) -> String {
        format!(
            "{:?}:{:?}:{:?}",
            self.paths,
            self.scans,
            self.entries
                .iter()
                .map(|entry| (entry.inode, entry.mtime, entry.size))
                .collect::<Vec<_>>()
        )
    }
}

pub struct DiagnosticProfileResult {
    store: gpui::EntityId,
    entries: [worktree::Entry; 2],
    scans: Vec<(settings::WorktreeId, usize)>,
    paths: [ProjectPath; 2],
    result: crate::profile_results::ProfileResults,
}

impl DiagnosticProfileResult {
    pub fn total_samples(&self) -> u64 {
        self.result.total_samples
    }
    pub fn unattributed_samples(&self) -> u64 {
        self.result.unattributed_samples
    }
    pub fn unresolved_frames(&self) -> usize {
        self.result.unresolved_frames
    }
    pub fn into_parts(self) -> ([ProjectPath; 2], crate::profile_results::ProfileResults) {
        (self.paths, self.result)
    }
}

impl ProfileResultsStore {
    fn diagnostic_inputs(
        &self,
        paths: &[ProjectPath; 2],
        cx: &App,
    ) -> Result<(
        [ProfileInput; 2],
        [worktree::Entry; 2],
        Vec<(settings::WorktreeId, usize)>,
    )> {
        anyhow::ensure!(
            cfg!(target_os = "macos")
                && self.remote_proxy.is_none()
                && self.worktree_store.read(cx).fs().is_some(),
            "Doctor profile import requires the selected local macOS host; bounded remote import is unavailable"
        );
        anyhow::ensure!(
            !TrustedWorktrees::has_restricted_worktrees(&self.worktree_store, cx),
            "Project trust is required for selected diagnostic profile inputs"
        );
        anyhow::ensure!(
            paths[0].worktree_id == paths[1].worktree_id
                && paths[0] != paths[1]
                && paths[0]
                    .path
                    .as_std_path()
                    .extension()
                    .is_some_and(|extension| extension == "json")
                && !paths[0]
                    .path
                    .as_std_path()
                    .to_string_lossy()
                    .ends_with(".syms.json")
                && paths[1]
                    .path
                    .as_std_path()
                    .to_string_lossy()
                    .ends_with(".syms.json"),
            "Select a Samply JSON profile and explicit symbol sidecar from the same worktree; dumps and other capture formats are unsupported"
        );
        let mut inputs = Vec::new();
        let mut entries = Vec::new();
        let mut scans = Vec::new();
        let worktrees = self.worktree_store.read(cx);
        for path in paths {
            let worktree = worktrees
                .visible_worktrees(cx)
                .find(|worktree| worktree.read(cx).id() == path.worktree_id)
                .ok_or_else(|| anyhow::anyhow!("Selected profile worktree is unavailable"))?;
            let worktree = worktree.read(cx);
            let snapshot = worktree.snapshot();
            inputs.push(ProfileInput::from_snapshot(&snapshot, path.path.as_ref())?);
            entries.push(
                snapshot
                    .entry_for_path(path.path.as_ref())
                    .ok_or_else(|| anyhow::anyhow!("Selected profile input disappeared"))?
                    .clone(),
            );
            scans.push((worktree.id(), worktree.scan_id()));
        }
        let entries: [worktree::Entry; 2] = entries
            .try_into()
            .map_err(|_| anyhow::anyhow!("Select exactly two profile inputs"))?;
        let total = entries.iter().try_fold(0u64, |total, entry| {
            total
                .checked_add(entry.size)
                .ok_or_else(|| anyhow::anyhow!("Profile size exceeds the supported range"))
        })?;
        anyhow::ensure!(
            total <= DIAGNOSTIC_PROFILE_BYTES as u64,
            "Selected profile and symbols exceed Doctor's combined 10 MiB limit"
        );
        Ok((
            inputs
                .try_into()
                .map_err(|_| anyhow::anyhow!("Select exactly two profile inputs"))?,
            entries,
            scans,
        ))
    }

    fn prepare_diagnostic_import(
        &self,
        paths: [ProjectPath; 2],
        cx: &Context<Self>,
    ) -> Result<DiagnosticProfileImport> {
        let (inputs, entries, scans) = self.diagnostic_inputs(&paths, cx)?;
        Ok(DiagnosticProfileImport {
            store: cx.entity_id(),
            paths,
            entries,
            scans,
            inputs,
        })
    }

    fn import_diagnostic_profile(
        &mut self,
        plan: DiagnosticProfileImport,
        cancellation: Arc<AtomicBool>,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<DiagnosticProfileResult>>> {
        anyhow::ensure!(
            plan.store == cx.entity_id(),
            "Diagnostic profile belongs to another owner"
        );
        let (_, entries, scans) = self.diagnostic_inputs(&plan.paths, cx)?;
        anyhow::ensure!(
            entries == plan.entries && scans == plan.scans,
            "Diagnostic profile inputs changed; review a fresh import"
        );
        anyhow::ensure!(
            !cancellation.load(Ordering::Relaxed),
            "Diagnostic profile import was cancelled"
        );
        anyhow::ensure!(
            self.imports.len() < MAX_HOST_PROFILE_IMPORTS,
            "The existing profile owner is busy"
        );
        let fs = self
            .worktree_store
            .read(cx)
            .fs()
            .ok_or_else(|| anyhow::anyhow!("Local profile filesystem is unavailable"))?;
        let import_id = NEXT_PROXY_IMPORT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| anyhow::anyhow!("Profile import identity exhausted"))?;
        let key = (proto::REMOTE_SERVER_PEER_ID, import_id);
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("Profile import generation exhausted"))?;
        self.imports.insert(
            key,
            ProfileImport {
                purpose: ProfileImportPurpose::Diagnostic,
                task_run_id: None,
                generation: self.generation,
                cancellation: cancellation.clone(),
                paths: plan.paths.iter().cloned().collect(),
                pages: VecDeque::new(),
                next_page: 0,
                revision: self.revision,
            },
        );
        let mut cancel_on_drop = ImportCancellation(Some(cancellation.clone()));
        let cancellation_for_parsing = cancellation.clone();
        let worktrees = self
            .worktree_store
            .read(cx)
            .visible_worktrees(cx)
            .map(|worktree| {
                let worktree = worktree.read(cx);
                (worktree.id(), worktree.abs_path())
            })
            .collect::<Vec<_>>();
        let paths = plan.paths.clone();
        #[cfg(any(test, feature = "test-support"))]
        let processing_gate = self.import_processing_gate.take();
        let parsing = cx.background_spawn(async move {
            #[cfg(any(test, feature = "test-support"))]
            if let Some((started, release)) = processing_gate {
                started
                    .send(())
                    .map_err(|_| anyhow::anyhow!("Profile test gate observer disappeared"))?;
                release.await?;
            }
            for (path, expected) in paths.iter().zip(&plan.entries) {
                let root = worktrees
                    .iter()
                    .find(|(id, _)| *id == path.worktree_id)
                    .ok_or_else(|| anyhow::anyhow!("Selected profile worktree is unavailable"))?;
                let metadata = fs
                    .metadata(&root.1.join(path.path.as_std_path()))
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("Profile input is unavailable"))?;
                anyhow::ensure!(
                    !metadata.is_dir
                        && !metadata.is_fifo
                        && metadata.inode == expected.inode
                        && Some(metadata.mtime) == expected.mtime
                        && metadata.len == expected.size,
                    "Profile input changed after review"
                );
            }
            let result = crate::profile_results::load_samply_profile_with_byte_limit(
                fs.as_ref(),
                [&plan.inputs[0], &plan.inputs[1]],
                ProfileHost::MacOs,
                &cancellation_for_parsing,
                DIAGNOSTIC_PROFILE_BYTES,
                |_, _| None,
            )
            .await?;
            for (path, expected) in paths.iter().zip(&plan.entries) {
                let root = worktrees
                    .iter()
                    .find(|(id, _)| *id == path.worktree_id)
                    .ok_or_else(|| anyhow::anyhow!("Selected profile worktree is unavailable"))?;
                let metadata = fs
                    .metadata(&root.1.join(path.path.as_std_path()))
                    .await?
                    .ok_or_else(|| anyhow::anyhow!("Profile input is unavailable"))?;
                anyhow::ensure!(
                    metadata.inode == expected.inode
                        && Some(metadata.mtime) == expected.mtime
                        && metadata.len == expected.size,
                    "Profile input changed during import"
                );
            }
            anyhow::Ok((plan, result))
        });
        let (completed, completion) = futures::channel::oneshot::channel();
        let caller_cancellation = cancellation.clone();
        cx.spawn(async move |this, cx| {
            let result = async {
                let parsed = parsing.await;
                let result = this.update(cx, |this, cx| {
                    anyhow::ensure!(
                        this.imports.contains_key(&key) && !cancellation.load(Ordering::Relaxed),
                        "Diagnostic profile import cancelled or authority changed"
                    );
                    let (plan, result) = parsed?;
                    let (_, entries, scans) = this.diagnostic_inputs(&plan.paths, cx)?;
                    anyhow::ensure!(
                        entries == plan.entries && scans == plan.scans,
                        "Diagnostic profile authority or inputs changed during import"
                    );
                    Ok::<_, anyhow::Error>(DiagnosticProfileResult {
                        store: plan.store,
                        entries: plan.entries,
                        scans: plan.scans,
                        paths: plan.paths,
                        result,
                    })
                });
                this.update(cx, |this, _| {
                    this.imports.remove(&key);
                })
                .map_err(|_| anyhow::anyhow!("Profile owner disappeared during cleanup"))?;
                cancel_on_drop.disarm();
                result.map_err(|_| anyhow::anyhow!("Profile owner disappeared during import"))?
            }
            .await;
            if completed.send(result).is_err() {
                log::debug!("Diagnostic profile caller closed; import cleanup completed");
            }
        })
        .detach();
        let mut caller_guard = ImportCancellation(Some(caller_cancellation));
        Ok(cx.background_spawn(async move {
            let result = completion
                .await
                .map_err(|_| anyhow::anyhow!("Diagnostic profile import was interrupted"))?;
            caller_guard.disarm();
            result
        }))
    }
}

impl crate::Project {
    pub fn prepare_diagnostic_profile_import(
        &self,
        paths: [ProjectPath; 2],
        cx: &mut App,
    ) -> Result<DiagnosticProfileImport> {
        anyhow::ensure!(
            self.is_local() && !self.is_shared() && !self.is_isolated_recipe_test(),
            "Doctor profile import requires an exclusive normal local development project"
        );
        self.profile_results_store
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Native profile owner unavailable"))?
            .update(cx, |store, cx| store.prepare_diagnostic_import(paths, cx))
    }
    pub fn revalidate_diagnostic_profile_result(
        &self,
        result: &DiagnosticProfileResult,
        cx: &App,
    ) -> Result<()> {
        anyhow::ensure!(
            self.is_local() && !self.is_shared() && !self.is_isolated_recipe_test(),
            "Diagnostic profile project authority changed"
        );
        let store = self
            .profile_results_store
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Native profile owner unavailable"))?;
        anyhow::ensure!(
            store.entity_id() == result.store,
            "Profile result belongs to another owner"
        );
        let (_, entries, scans) = store.read(cx).diagnostic_inputs(&result.paths, cx)?;
        anyhow::ensure!(
            entries == result.entries && scans == result.scans,
            "Profile inputs changed after import"
        );
        Ok(())
    }

    pub fn import_diagnostic_profile(
        &self,
        plan: DiagnosticProfileImport,
        cancellation: Arc<AtomicBool>,
        cx: &mut App,
    ) -> Result<Task<Result<DiagnosticProfileResult>>> {
        anyhow::ensure!(
            self.is_local() && !self.is_shared() && !self.is_isolated_recipe_test(),
            "Doctor profile import requires the reviewed exclusive local development project"
        );
        self.profile_results_store
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Native profile owner unavailable"))?
            .update(cx, |store, cx| {
                store.import_diagnostic_profile(plan, cancellation, cx)
            })
    }
}

pub struct ProfileResultsStore {
    task_helper: Option<ProfileTaskHelper>,
    #[cfg(target_os = "macos")]
    helper_admission_epoch: Arc<()>,
    #[cfg(target_os = "macos")]
    build_report_budget: Arc<std::sync::atomic::AtomicUsize>,
    task_runs: HashMap<ProfileTaskKey, ProfileTaskRun>,
    worktree_store: Entity<WorktreeStore>,
    imports: HashMap<ImportKey, ProfileImport>,
    remote_proxy: Option<RemoteProfileProxy>,
    proxied_imports: HashMap<ImportKey, ProxiedImport>,
    generation: u64,
    revision: u64,
    invalidation: watch::Sender<()>,
    transport_connected: bool,
    cancelled: HashSet<ImportKey>,
    _collaboration_session: Option<Task<()>>,
    #[cfg(any(test, feature = "test-support"))]
    acknowledged_host_cancellations: Arc<Mutex<HashSet<(u64, u64)>>>,
    #[cfg(any(test, feature = "test-support"))]
    import_processing_gate: Option<(
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
    )>,
    #[cfg(any(test, feature = "test-support"))]
    page_processing_gate: Option<(
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
    )>,
    #[cfg(any(test, feature = "test-support"))]
    task_preparation_response_gate: Option<(
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
    )>,
    #[cfg(any(test, feature = "test-support"))]
    task_preparation_acceptance_gate: Option<(
        futures::channel::oneshot::Sender<()>,
        futures::channel::oneshot::Receiver<()>,
    )>,
    _subscriptions: Vec<Subscription>,
}

impl ProfileResultsStore {
    pub fn has_active_work(&self) -> bool {
        !self.task_runs.is_empty() || !self.imports.is_empty() || !self.proxied_imports.is_empty()
    }

    pub fn init(client: &AnyProtoClient) {
        client.add_entity_request_handler(Self::handle_prepare_task);
        client.add_entity_request_handler(Self::handle_get_task);
        client.add_entity_request_handler(Self::handle_cancel_task);
        client.add_entity_request_handler(Self::handle_import);
        client.add_entity_request_handler(Self::handle_page);
        client.add_entity_request_handler(Self::handle_cancel);
    }

    pub fn local(worktree_store: Entity<WorktreeStore>, cx: &mut Context<Self>) -> Self {
        let mut subscriptions = vec![cx.observe_global::<feature_flags::FeatureFlagStore>(|this, cx| {
            if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
                #[cfg(target_os = "macos")]
                { this.helper_admission_epoch = Arc::new(()); }
                for run in this.task_runs.values() {
                    let native = run.native().is_some();
                    #[cfg(target_os = "macos")]
                    let native = native || matches!(&run.payload, ProfileTaskPayload::BuildReport(build)
                        if build.mode != BuildReportMode::Stdout);
                    if native {
                        run.cancellation.store(true, Ordering::Relaxed);
                        run.stop.close();
                    }
                }
                this.imports.retain(|_, import| {
                    import.purpose == ProfileImportPurpose::Diagnostic
                });
                this.proxied_imports.clear();
                if this.invalidation.send(()).is_err() {
                    log::trace!("No pending native profile imports to revoke");
                }
            }
        }), cx.subscribe(&worktree_store, |this, _, event, cx| {
            #[cfg(target_os = "macos")]
            if matches!(
                event,
                WorktreeStoreEvent::WorktreeAdded(..)
                    | WorktreeStoreEvent::WorktreeRemoved(..)
                    | WorktreeStoreEvent::WorktreeReleased(..)
                    | WorktreeStoreEvent::WorktreeOrderChanged
            ) {
                this.helper_admission_epoch = Arc::new(());
                this.task_runs.retain(|_, run| !matches!(&run.payload,
                    ProfileTaskPayload::BuildReport(build) if build.mode == BuildReportMode::CheckedCargoRuntime));
            }
            if this.imports.is_empty()
                && this.proxied_imports.is_empty()
                && this.cancelled.is_empty()
                && this.task_runs.is_empty()
            {
                return;
            }
            match event {
                WorktreeStoreEvent::WorktreeRemoved(..)
                | WorktreeStoreEvent::WorktreeReleased(..)
                | WorktreeStoreEvent::WorktreeOrderChanged => this.invalidate(),
                WorktreeStoreEvent::WorktreeUpdatedEntries(worktree_id, changes) => {
                    this.revision = this.revision.wrapping_add(1);
                    let revoked = changes
                        .iter()
                        .map(|(path, _, _)| ProjectPath {
                            worktree_id: *worktree_id,
                            path: path.clone(),
                        })
                        .filter(|path| {
                            this.imports
                                .values()
                                .any(|import| import.paths.contains(path))
                                || this
                                    .proxied_imports
                                    .values()
                                    .any(|import| import.paths.contains(path))
                        })
                        .filter(|path| !this.permitted(path, cx))
                        .collect::<HashSet<_>>();
                    this.imports
                        .retain(|_, import| import.paths.is_disjoint(&revoked));
                    this.proxied_imports
                        .retain(|_, import| import.paths.is_disjoint(&revoked));
                    let stale_runs = this
                        .task_runs
                        .iter()
                        .filter_map(|(key, run)| {
                            (run.native().is_some_and(|native| native.versions.is_some())
                                && !this.task_run_current(run, cx))
                            .then_some(key.clone())
                        })
                        .collect::<Vec<_>>();
                    for key in stale_runs {
                        this.remove_task_run(&key);
                    }
                }
                _ => {}
            }
        })];
        if let Some(trust) = TrustedWorktrees::try_get_global(cx) {
            subscriptions.push(cx.subscribe(&trust, |this, _, _, cx| {
                if TrustedWorktrees::has_restricted_worktrees(&this.worktree_store, cx) {
                    this.invalidate();
                }
            }));
        }
        Self {
            task_helper: None,
            #[cfg(target_os = "macos")]
            helper_admission_epoch: Arc::new(()),
            #[cfg(target_os = "macos")]
            build_report_budget: Arc::default(),
            task_runs: HashMap::default(),
            worktree_store,
            imports: HashMap::default(),
            remote_proxy: None,
            proxied_imports: HashMap::default(),
            generation: 0,
            revision: 0,
            invalidation: watch::channel(()).0,
            transport_connected: true,
            cancelled: HashSet::default(),
            _collaboration_session: None,
            #[cfg(any(test, feature = "test-support"))]
            acknowledged_host_cancellations: Arc::default(),
            #[cfg(any(test, feature = "test-support"))]
            import_processing_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            page_processing_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            task_preparation_response_gate: None,
            #[cfg(any(test, feature = "test-support"))]
            task_preparation_acceptance_gate: None,
            _subscriptions: subscriptions,
        }
    }

    pub(crate) fn collaboration(
        project_id: u64,
        worktree_store: Entity<WorktreeStore>,
        client: &Arc<client::Client>,
        cx: &mut Context<Self>,
    ) -> Self {
        let mut store = Self::local(worktree_store, cx);
        // A joined peer owns import invalidation, but cannot issue local profile Tasks.
        store.remote_proxy = Some(RemoteProfileProxy {
            client: client.clone().into(),
            project_id,
        });
        let mut statuses = client.status();
        let mut previous_status = *statuses.borrow();
        store.transport_connected = previous_status.is_connected();
        store._collaboration_session = Some(cx.spawn(async move |this, cx| {
            while let Some(status) = statuses.next().await {
                if status == previous_status {
                    continue;
                }
                previous_status = status;
                if let Err(error) = this.update(cx, |store, _| {
                    store.invalidate();
                    store.transport_connected = status.is_connected();
                }) {
                    log::debug!("Profile collaboration session owner retired: {error:#}");
                    break;
                }
            }
        }));
        store
    }

    pub fn remote(
        worktree_store: Entity<WorktreeStore>,
        client: AnyProtoClient,
        remote: &Entity<RemoteClient>,
        cx: &mut Context<Self>,
    ) -> Self {
        let mut store = Self::local(worktree_store, cx);
        store.remote_proxy = Some(RemoteProfileProxy {
            client,
            project_id: proto::REMOTE_SERVER_PROJECT_ID,
        });
        store.transport_connected =
            remote.read(cx).connection_state() == ConnectionState::Connected;
        store
            ._subscriptions
            .push(cx.subscribe(remote, |this, _, event, _| match event {
                RemoteClientEvent::ConnectionStateChanged(ConnectionState::Connected) => {
                    this.transport_connected = true;
                }
                RemoteClientEvent::ConnectionStateChanged(_)
                | RemoteClientEvent::Disconnected { .. } => {
                    this.invalidate();
                    this.transport_connected = false;
                }
            }));
        store
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn gate_next_import_for_test(
        &mut self,
        started: futures::channel::oneshot::Sender<()>,
        release: futures::channel::oneshot::Receiver<()>,
    ) {
        self.import_processing_gate = Some((started, release));
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn gate_next_page_for_test(
        &mut self,
        started: futures::channel::oneshot::Sender<()>,
        release: futures::channel::oneshot::Receiver<()>,
    ) {
        self.page_processing_gate = Some((started, release));
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn gate_next_task_preparation_response_for_test(
        &mut self,
        started: futures::channel::oneshot::Sender<()>,
        release: futures::channel::oneshot::Receiver<()>,
    ) {
        self.task_preparation_response_gate = Some((started, release));
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn gate_next_task_preparation_acceptance_for_test(
        &mut self,
        started: futures::channel::oneshot::Sender<()>,
        release: futures::channel::oneshot::Receiver<()>,
    ) {
        self.task_preparation_acceptance_gate = Some((started, release));
    }

    #[cfg(all(target_os = "macos", any(test, feature = "test-support")))]
    pub fn checked_native_resources_for_test(&self) -> (usize, usize) {
        (
            self.build_report_budget.load(Ordering::Relaxed),
            self.task_runs
                .values()
                .filter(|run| {
                    run.native()
                        .is_some_and(|native| native.checked_invocation.is_some())
                })
                .count(),
        )
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn is_import_only_remote_proxy_for_test(&self, project_id: u64) -> bool {
        self.remote_proxy
            .as_ref()
            .is_some_and(|proxy| proxy.project_id == project_id)
            && self.task_helper.is_none()
            && self.task_runs.is_empty()
            && self.imports.is_empty()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn task_binding_count_for_peer_for_test(&self, peer: proto::PeerId) -> usize {
        self.task_runs
            .keys()
            .filter(|(owner, _)| *owner == ProfileTaskActor::Peer(peer))
            .count()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn import_cancellation_for_test(
        &self,
        peer: proto::PeerId,
        import_id: u64,
    ) -> Option<Arc<AtomicBool>> {
        self.imports
            .get(&(peer, import_id))
            .map(|import| import.cancellation.clone())
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn active_import_count_for_peer_for_test(&self, peer: proto::PeerId) -> usize {
        self.imports
            .keys()
            .filter(|(owner, _)| *owner == peer)
            .count()
            + self
                .proxied_imports
                .keys()
                .filter(|(owner, _)| *owner == peer)
                .count()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn active_import_count_for_test(&self) -> usize {
        self.imports.len() + self.proxied_imports.len()
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn proxied_import_identity_for_test(
        &self,
        peer: proto::PeerId,
        import_id: u64,
    ) -> Option<(u64, u64)> {
        self.proxied_imports
            .get(&(peer, import_id))
            .map(|import| (import.remote_id, import.generation))
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn host_cancellation_acknowledged_for_test(&self, import_id: u64, generation: u64) -> bool {
        self.acknowledged_host_cancellations
            .lock()
            .contains(&(import_id, generation))
    }

    pub fn observe_transport<T: gpui::EventEmitter<bool> + 'static>(
        &mut self,
        transport: &Entity<T>,
        connected: bool,
        cx: &mut Context<Self>,
    ) {
        self.invalidate();
        self.transport_connected = connected;
        self._subscriptions
            .push(cx.subscribe(transport, |this, _, connected, _| {
                this.invalidate();
                this.transport_connected = *connected;
            }));
    }

    pub fn invalidate(&mut self) {
        #[cfg(target_os = "macos")]
        {
            self.helper_admission_epoch = Arc::new(());
        }
        self.task_runs.clear();
        self.revision = self.revision.wrapping_add(1);
        self.imports.clear();
        self.proxied_imports.clear();
        self.cancelled.clear();
        if self.invalidation.send(()).is_err() {
            log::trace!("No pending profile proxy requests to invalidate");
        }
    }

    pub fn forget_peer(&mut self, peer: proto::PeerId) {
        self.task_runs
            .retain(|(owner, _), _| *owner != ProfileTaskActor::Peer(peer));
        self.imports.retain(|(owner, _), _| *owner != peer);
        self.proxied_imports.retain(|(owner, _), _| *owner != peer);
        self.cancelled.retain(|(owner, _)| *owner != peer);
    }

    pub fn set_task_helper(&mut self, path: PathBuf) {
        self.replace_task_helper(ProfileTaskHelper {
            path,
            mode: ProfileTaskHelperMode::RemoteServer,
        });
    }

    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    fn ensure_local_task_helper(&mut self) -> Result<()> {
        if let Some(helper) = &self.task_helper {
            anyhow::ensure!(
                helper.mode == ProfileTaskHelperMode::LocalApplication,
                "Selected local Task cannot use a remote helper route"
            );
        } else {
            let path = std::env::current_exe()
                .map_err(|_| anyhow::anyhow!("Local Task helper image is unavailable"))?;
            self.set_local_task_helper(path);
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    pub fn set_local_task_helper(&mut self, path: PathBuf) {
        self.replace_task_helper(ProfileTaskHelper {
            path,
            mode: ProfileTaskHelperMode::LocalApplication,
        });
    }

    fn replace_task_helper(&mut self, helper: ProfileTaskHelper) {
        if self.task_helper.as_ref() != Some(&helper) {
            #[cfg(target_os = "macos")]
            {
                self.helper_admission_epoch = Arc::new(());
            }
            #[cfg(target_os = "macos")]
            self.task_runs.retain(|_, run| run.native().is_some());
            self.task_helper = Some(helper);
        }
    }

    fn remove_task_run(&mut self, key: &ProfileTaskKey) {
        self.task_runs.remove(key);
        self.imports.retain(|(owner, _), import| {
            ProfileTaskActor::Peer(*owner) != key.0 || import.task_run_id.as_ref() != Some(&key.1)
        });
    }

    #[cfg(target_os = "macos")]
    fn prune_cancelled_task_runs(&mut self) -> Vec<async_channel::Receiver<()>> {
        let mut retirements = Vec::new();
        // A cancelled capture's admission can own a Build binding whose drop retires another row.
        for _ in 0..MAX_HOST_PROFILE_TASKS {
            let before = self.task_runs.len();
            self.task_runs.retain(|_, run| {
                let remove = run.native().is_some() && run.cancellation.load(Ordering::Relaxed);
                if remove {
                    if let Some(checked) = run
                        .native()
                        .and_then(|native| native.checked_invocation.as_ref())
                    {
                        retirements.push(checked.reservation.retirement.clone());
                    }
                }
                !remove
            });
            self.task_runs.retain(|_, run| {
                if !run.cancellation.load(Ordering::Relaxed) {
                    return true;
                }
                if let ProfileTaskPayload::BuildReport(build) = &run.payload {
                    retirements.push(build.reservation.retirement.clone());
                }
                false
            });
            if self.task_runs.len() == before {
                break;
            }
        }
        retirements
    }

    #[cfg(target_os = "macos")]
    fn helper_image_current(&self, image: &LocalNativeTaskHelperImage, cx: &App) -> bool {
        self.remote_proxy.is_none()
            && self.transport_connected
            && !TrustedWorktrees::has_restricted_worktrees(&self.worktree_store, cx)
            && self.task_helper.as_ref() == Some(&image.owner_helper)
            && Arc::ptr_eq(&self.helper_admission_epoch, &image.epoch)
    }

    #[cfg(target_os = "macos")]
    fn prepare_local_native_task_helper_image(
        &self,
        project: gpui::EntityId,
        cx: &mut Context<Self>,
    ) -> Task<Result<LocalNativeTaskHelperImage>> {
        let prepared = (|| {
            anyhow::ensure!(
                self.remote_proxy.is_none()
                    && self.transport_connected
                    && !TrustedWorktrees::has_restricted_worktrees(&self.worktree_store, cx),
                "Native helper image owner is unavailable"
            );
            let helper = self
                .task_helper
                .clone()
                .context("Native helper image is unconfigured")?;
            anyhow::ensure!(
                helper.mode == ProfileTaskHelperMode::LocalApplication,
                "Native helper image requires local application mode"
            );
            let filesystem = self
                .worktree_store
                .read(cx)
                .fs()
                .context("Native helper image filesystem is unavailable")?;
            Ok((helper, filesystem))
        })();
        let (helper, filesystem) = match prepared {
            Ok(value) => value,
            Err(error) => return Task::ready(Err(error)),
        };
        let epoch = self.helper_admission_epoch.clone();
        let store = cx.entity_id();
        let background = cx.background_spawn({
            let helper = helper.clone();
            async move {
                crate::profile_results::prepare_task_helper_identity(
                    filesystem.as_ref(),
                    helper.path,
                    &AtomicBool::new(false),
                )
                .await
                .map_err(|_| anyhow::anyhow!("Native helper image could not be sealed"))
            }
        });
        cx.spawn(async move |this, cx| {
            let identity = background.await?;
            let image = LocalNativeTaskHelperImage {
                owner_helper: helper.clone(),
                purpose: NativeTaskHelperImagePurpose::Application,
                project,
                store,
                epoch,
                helper,
                identity,
            };
            this.read_with(cx, |this, cx| {
                anyhow::ensure!(
                    this.helper_image_current(&image, cx),
                    "Native helper image owner changed during preparation"
                );
                Ok(image)
            })?
        })
    }

    #[cfg(target_os = "macos")]
    fn build_report_helper_current(
        &self,
        helper: &ProfileTaskHelper,
        runtime: Option<&BuildReportRuntimeAdmission>,
        cx: &App,
    ) -> bool {
        match runtime {
            Some(runtime) => {
                self.helper_image_current(&runtime.image, cx)
                    && helper == &runtime.image.helper
                    && matches!(
                        runtime.image.purpose,
                        NativeTaskHelperImagePurpose::CargoRuntime(_)
                    )
            }
            None => self.task_helper.as_ref() == Some(helper),
        }
    }

    #[cfg(target_os = "macos")]
    fn build_binding_current(&self, binding: &BuildTaskReportBinding, cx: &App) -> bool {
        (binding.mode == BuildReportMode::Stdout || cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>())
            && binding.key.0 == ProfileTaskActor::Local(binding.store)
            && binding.helper_image.as_ref().is_none_or(|image| self.helper_image_current(image, cx))
            && self.remote_proxy.is_none()
            && self.transport_connected
            && !binding.cancellation.load(Ordering::Relaxed)
            && !TrustedWorktrees::has_restricted_worktrees(&self.worktree_store, cx)
            && self.build_report_helper_current(&binding.helper, binding.runtime_admission.as_deref(), cx)
            && self.task_runs.get(&binding.key).is_some_and(|run| {
                run.generation == binding.generation
                    && !run.failed
                    && Arc::ptr_eq(&run.cancellation, &binding.cancellation)
                    && matches!(&run.payload, ProfileTaskPayload::BuildReport(build)
                        if build.helper == binding.helper
                            && build.expected_invocation == binding.expected_invocation
                            && build.mode == binding.mode
                            && match (&build.runtime_admission, &binding.runtime_admission) {
                                (Some(current), Some(bound)) => Arc::ptr_eq(current, bound),
                                (None, None) => true,
                                _ => false,
                            }
                            && Arc::ptr_eq(&build.reservation, &binding.reservation)
                            && build.identity.as_ref().is_some_and(|identity|
                                identity.fingerprint() == binding.identity.fingerprint()
                                    && identity.canonical_path() == binding.identity.canonical_path()))
            })
    }

    #[cfg(target_os = "macos")]
    fn prepare_local_build_report(
        &mut self,
        run_id: String,
        report_limit: usize,
        cx: &mut Context<Self>,
    ) -> Task<Result<BuildTaskReportBinding>> {
        self.prepare_local_build_report_context(
            run_id,
            report_limit,
            None,
            BuildReportMode::Stdout,
            None,
            cx,
        )
    }

    #[cfg(target_os = "macos")]
    fn prepare_checked_local_build_report(
        &mut self,
        run_id: String,
        report_limit: usize,
        expected: util::macos_task_completion::TaskBuildInvocation,
        cx: &mut Context<Self>,
    ) -> Task<Result<BuildTaskReportBinding>> {
        self.prepare_local_build_report_context(
            run_id,
            report_limit,
            Some(expected),
            BuildReportMode::CheckedBuild,
            None,
            cx,
        )
    }

    #[cfg(all(target_os = "macos", test))]
    fn prepare_checked_local_cargo_runtime_report(
        &mut self,
        run_id: String,
        report_limit: usize,
        expected: util::macos_task_completion::TaskBuildInvocation,
        cx: &mut Context<Self>,
    ) -> Task<Result<BuildTaskReportBinding>> {
        self.prepare_local_build_report_context(
            run_id,
            report_limit,
            Some(expected),
            BuildReportMode::CheckedCargoRuntime,
            None,
            cx,
        )
    }

    #[cfg(target_os = "macos")]
    fn prepare_local_build_report_context(
        &mut self,
        run_id: String,
        report_limit: usize,
        expected_invocation: Option<util::macos_task_completion::TaskBuildInvocation>,
        mode: BuildReportMode,
        runtime_admission: Option<Arc<BuildReportRuntimeAdmission>>,
        cx: &mut Context<Self>,
    ) -> Task<Result<BuildTaskReportBinding>> {
        let retirements = self.prune_cancelled_task_runs();
        if retirements.is_empty() {
            return self.prepare_local_build_report_after_retirement(
                run_id,
                report_limit,
                expected_invocation,
                mode,
                runtime_admission,
                cx,
            );
        }
        let deadline = cx.background_executor().timer(Duration::from_secs(30));
        cx.spawn(async move |this, cx| {
            smol::future::race(
                async move {
                    for retirement in retirements {
                        anyhow::ensure!(
                            retirement.recv().await.is_err(),
                            "Build report retirement notification was invalid"
                        );
                    }
                    anyhow::Ok(())
                },
                async move {
                    deadline.await;
                    anyhow::bail!(
                        "Cancelled Build report observation did not retire within 30 seconds"
                    )
                },
            )
            .await?;
            this.update(cx, |this, cx| {
                this.prepare_local_build_report_after_retirement(
                    run_id,
                    report_limit,
                    expected_invocation,
                    mode,
                    runtime_admission,
                    cx,
                )
            })?
            .await
        })
    }

    #[cfg(target_os = "macos")]
    fn prepare_local_build_report_after_retirement(
        &mut self,
        run_id: String,
        report_limit: usize,
        expected_invocation: Option<util::macos_task_completion::TaskBuildInvocation>,
        mode: BuildReportMode,
        runtime_admission: Option<Arc<BuildReportRuntimeAdmission>>,
        cx: &mut Context<Self>,
    ) -> Task<Result<BuildTaskReportBinding>> {
        let prepare = (|| {
            anyhow::ensure!(
                (mode == BuildReportMode::Stdout) == expected_invocation.is_none(),
                "Build report invocation mode is inconsistent"
            );
            if let Some(expected) = &expected_invocation {
                expected
                    .validate()
                    .map_err(|_| anyhow::anyhow!("Invalid checked Build invocation"))?;
            }
            anyhow::ensure!(
                valid_profile_task_run_id(&run_id),
                "Build Task run identity is invalid"
            );
            anyhow::ensure!(
                self.remote_proxy.is_none() && self.transport_connected,
                "Build report owner is unavailable"
            );
            anyhow::ensure!(
                !TrustedWorktrees::has_restricted_worktrees(&self.worktree_store, cx),
                "Build report requires trusted worktrees"
            );
            let helper = match runtime_admission.as_ref() {
                Some(admission) => {
                    anyhow::ensure!(
                        mode == BuildReportMode::CheckedCargoRuntime
                            && self.build_report_helper_current(
                                &admission.image.helper,
                                Some(admission),
                                cx
                            ),
                        "Cargo runtime report requires its original dedicated image"
                    );
                    admission.image.helper.clone()
                }
                None => self
                    .task_helper
                    .clone()
                    .context("Build Task helper is unavailable")?,
            };
            anyhow::ensure!(
                helper.mode == ProfileTaskHelperMode::LocalApplication,
                "Build report requires the local application helper mode"
            );
            let filesystem = self
                .worktree_store
                .read(cx)
                .fs()
                .ok_or_else(|| anyhow::anyhow!("Build report filesystem is unavailable"))?;
            let key = (ProfileTaskActor::Local(cx.entity_id()), run_id.clone());
            anyhow::ensure!(
                !self.task_runs.contains_key(&key) && self.task_runs.len() < MAX_HOST_PROFILE_TASKS,
                "Build Task run is already reserved or the owner is full"
            );
            let reservation = if expected_invocation.is_some() {
                BuildReportReservation::reserve_for_invocation(
                    self.build_report_budget.clone(),
                    report_limit,
                    true,
                )?
            } else {
                BuildReportReservation::reserve(self.build_report_budget.clone(), report_limit)?
            };
            let generation = self
                .generation
                .checked_add(1)
                .ok_or_else(|| anyhow::anyhow!("Build Task generation is exhausted"))?;
            self.generation = generation;
            Ok((helper, filesystem, key, reservation, generation))
        })();
        let (helper, filesystem, key, reservation, generation) = match prepare {
            Ok(prepared) => prepared,
            Err(error) => return Task::ready(Err(error)),
        };
        let cancellation = Arc::new(AtomicBool::new(false));
        let (stop, stopped) = async_channel::bounded(1);
        #[cfg(feature = "cargo-workspace")]
        let (observation_arm, armed) =
            if mode == BuildReportMode::CheckedCargoRuntime && runtime_admission.is_some() {
                let (arm, armed) = async_channel::bounded(1);
                (Some(arm), Some(armed))
            } else {
                (None, None)
            };
        #[cfg(not(feature = "cargo-workspace"))]
        let armed = None;
        let expiry = self.task_expiry(key.clone(), generation, Duration::from_secs(30), cx);
        self.task_runs.insert(
            key.clone(),
            ProfileTaskRun {
                completion: watch::channel(()).0,
                generation,
                payload: ProfileTaskPayload::BuildReport(BuildReportTaskRun {
                    #[cfg(feature = "cargo-workspace")]
                    observation_arm,
                    runtime_admission: runtime_admission.clone(),
                    mode,
                    expected_invocation: expected_invocation.clone(),
                    observed_invocation: None,
                    observed_cargo_runtime: None,
                    helper: helper.clone(),
                    identity: None,
                    reservation: reservation.clone(),
                    report: None,
                    fingerprint: None,
                }),
                failed: false,
                cancellation: cancellation.clone(),
                stop: stop.clone(),
                _expiry: expiry,
                observation: None,
            },
        );
        let store = cx.entity_id();
        let mut cancel_on_drop = ImportCancellation(Some(cancellation.clone()));
        let token = rand::random::<[u8; 32]>();
        let background = cx.background_spawn({
            let cancellation = cancellation.clone();
            let helper = helper.clone();
            let filesystem = filesystem.clone();
            async move {
                let identity = crate::profile_results::prepare_task_helper_identity(
                    filesystem.as_ref(),
                    helper.path,
                    &cancellation,
                )
                .await?;
                anyhow::ensure!(
                    !cancellation.load(Ordering::Relaxed),
                    "Build report preparation cancelled"
                );
                let listener = util::macos_task_completion::TaskCompletionListener::bind(token)
                    .map_err(|_| {
                        anyhow::anyhow!("Cannot prepare the private Build Task listener")
                    })?;
                anyhow::Ok((identity, listener))
            }
        });
        let preparation = BuildReportPreparation {
            mode,
            expected_invocation,
            reservation,
        };
        cx.spawn(async move |this, cx| {
            let (reservation, expected_invocation, mode) = preparation.into_parts();
            let prepared = background.await;
            let binding = this.update(cx, |this, cx| {
                let pending = this.task_runs.get(&key).is_some_and(|run| {
                    run.generation == generation && !run.cancellation.load(Ordering::Relaxed)
                        && matches!(&run.payload, ProfileTaskPayload::BuildReport(build) if build.helper == helper)
                });
                if !pending || !this.build_report_helper_current(&helper, runtime_admission.as_deref(), cx)
                    || !this.transport_connected || this.remote_proxy.is_some()
                    || TrustedWorktrees::has_restricted_worktrees(&this.worktree_store, cx) {
                    if pending { this.remove_task_run(&key); }
                    anyhow::bail!("Build report preparation became stale");
                }
                let (identity, listener) = match prepared {
                    Ok(prepared) => prepared,
                    Err(_) => { this.remove_task_run(&key); anyhow::bail!("Build Task helper could not be sealed"); }
                };
                let socket = listener.path();
                let observation = this.observe_build_report(BuildReportObservation {
                    armed,
                    runtime_admission: runtime_admission.clone(),
                    mode,
                    reservation: reservation.clone(),
                    expected_invocation: expected_invocation.clone(),
                    key: key.clone(), generation, identity: identity.clone(), listener, stopped,
                    filesystem, cancellation: cancellation.clone(), limit: report_limit,
                }, cx);
                let expiry = this.task_expiry(key.clone(), generation, Duration::from_secs(3630), cx);
                if let Some(run) = this.task_runs.get_mut(&key) {
                    if let ProfileTaskPayload::BuildReport(build) = &mut run.payload { build.identity = Some(identity.clone()); }
                    run.observation = Some(observation);
                    run._expiry = expiry;
                }
                Ok(BuildTaskReportBinding { runtime_admission, helper_image: None, mode, expected_invocation, store, key, generation, helper, identity, socket, token,
                    reservation, cancellation, stop })
            })?;
            if binding.is_ok() { cancel_on_drop.disarm(); }
            binding
        })
    }

    #[cfg(target_os = "macos")]
    async fn observe_reserved_build_report(
        observation: BuildReportObservation,
        registration_requests: async_channel::Sender<futures::channel::oneshot::Sender<Result<()>>>,
    ) -> Result<ReservedBuildReport> {
        // The first local is dropped after the later observation futures and buffers.
        let BuildReportObservation {
            armed,
            runtime_admission,
            mode,
            reservation,
            expected_invocation,
            key: _,
            generation: _,
            identity,
            listener,
            stopped,
            filesystem,
            cancellation,
            limit,
        } = observation;
        if let Some(armed) = armed {
            smol::future::race(
                async {
                    armed
                        .recv()
                        .await
                        .map_err(|_| anyhow::anyhow!("Build Task launch observation was not armed"))
                },
                async {
                    if stopped.recv().await.is_err() {
                        log::trace!("Unarmed Build Task observation channel closed");
                    }
                    anyhow::bail!("Unarmed Build Task observation cancelled")
                },
            )
            .await?;
            anyhow::ensure!(
                !cancellation.load(Ordering::Relaxed),
                "Build Task launch observation was cancelled before arming"
            );
        }
        let check = {
            let filesystem = filesystem.clone();
            let cancellation = cancellation.clone();
            let identity = identity.clone();
            async move |kernel_executable: PathBuf| {
                let actual = filesystem
                    .canonicalize(&kernel_executable)
                    .await
                    .map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            "Cannot identify the registered Build Task helper",
                        )
                    })?;
                if actual.as_path() != identity.canonical_path() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "Registered Build Task helper does not match its prepared executable",
                    ));
                }
                crate::profile_results::revalidate_task_helper_identity(
                    filesystem.as_ref(),
                    &identity,
                    &cancellation,
                )
                .await
                .map_err(|_| {
                    std::io::Error::new(
                        std::io::ErrorKind::PermissionDenied,
                        "Prepared Build Task helper identity changed",
                    )
                })?;
                if runtime_admission.is_some() {
                    let (answer, answered) = futures::channel::oneshot::channel();
                    registration_requests.send(answer).await.map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::Interrupted,
                            "Cargo runtime registration owner disappeared",
                        )
                    })?;
                    answered
                        .await
                        .map_err(|_| {
                            std::io::Error::new(
                                std::io::ErrorKind::Interrupted,
                                "Cargo runtime registration was cancelled",
                            )
                        })?
                        .map_err(|_| {
                            std::io::Error::new(
                                std::io::ErrorKind::PermissionDenied,
                                "Cargo runtime registration became stale",
                            )
                        })?;
                }
                Ok(())
            }
        };
        let checked_observation = match (mode, expected_invocation) {
            (BuildReportMode::CheckedCargoRuntime, Some(expected)) => listener
                .observe_checked_cargo_runtime_stdout_with_helper_check(limit, expected, check)
                .boxed(),
            (BuildReportMode::CheckedBuild, Some(expected)) => listener
                .observe_checked_build_stdout_with_helper_check(limit, expected, check)
                .boxed(),
            (BuildReportMode::Stdout, None) => listener
                .observe_stdout_with_helper_check(limit, check)
                .boxed(),
            _ => anyhow::bail!("Build report invocation mode is inconsistent"),
        };
        let report = smol::future::race(checked_observation, async {
            if stopped.recv().await.is_err() {
                log::trace!("Build Task observation channel closed");
            }
            Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "Build Task observation cancelled",
            ))
        })
        .await
        .map_err(|error| {
            #[cfg(any(test, feature = "test-support"))]
            eprintln!("native Build Task kernel observation failed: {error:?}");
            #[cfg(not(any(test, feature = "test-support")))]
            drop(error);
            anyhow::anyhow!("Build Task observation failed")
        })?;
        let (status, bytes, invocation, cargo_runtime) = report.into_parts_with_cargo_runtime();
        anyhow::ensure!(status.success(), "Build Task child did not succeed");
        anyhow::ensure!(
            (mode == BuildReportMode::CheckedCargoRuntime) == cargo_runtime.is_some(),
            "Build report runtime observation mode is inconsistent"
        );
        anyhow::ensure!(
            bytes.len() <= limit,
            "Build Task report exceeds its reserved byte limit"
        );
        let fingerprint = crate::profile_results::profile_fingerprint(&bytes, &cancellation)?;
        crate::profile_results::revalidate_task_helper_identity(
            filesystem.as_ref(),
            &identity,
            &cancellation,
        )
        .await?;
        anyhow::Ok(ReservedBuildReport {
            cargo_runtime: cargo_runtime.map(Arc::new),
            bytes: Arc::new(bytes),
            fingerprint,
            invocation: invocation.map(Arc::new),
            reservation,
        })
    }

    #[cfg(target_os = "macos")]
    fn check_build_report_registration(
        &self,
        key: &ProfileTaskKey,
        generation: u64,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<()>>> {
        anyhow::ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling experimental opt-in was disabled"
        );

        let admission = self
            .task_runs
            .get(key)
            .filter(|run| {
                run.generation == generation
                    && !run.failed
                    && !run.cancellation.load(Ordering::Relaxed)
            })
            .and_then(|run| match &run.payload {
                ProfileTaskPayload::BuildReport(build)
                    if build.mode == BuildReportMode::CheckedCargoRuntime =>
                {
                    build.runtime_admission.clone()
                }
                _ => None,
            })
            .context("Cargo runtime registration admission disappeared")?;
        anyhow::ensure!(
            self.helper_image_current(&admission.image, cx),
            "Cargo runtime helper image owner changed"
        );
        let key = key.clone();
        #[cfg(feature = "cargo-workspace")]
        return Ok(cx.spawn(async move |this, cx| {
            anyhow::ensure!(admission.lease.matches_helper_image(&admission.image),
                "Cargo runtime helper admission identity changed");
            let renewed = admission.lease.renew(cx)?;
            renewed.await?;
            this.read_with(cx, |store, cx| {
                anyhow::ensure!(store.helper_image_current(&admission.image, cx)
                    && store.task_runs.get(&key).is_some_and(|run| run.generation == generation && !run.failed
                        && !run.cancellation.load(Ordering::Relaxed)
                        && matches!(&run.payload, ProfileTaskPayload::BuildReport(build)
                            if build.mode == BuildReportMode::CheckedCargoRuntime
                                && build.runtime_admission.as_ref().is_some_and(|current| Arc::ptr_eq(current, &admission)))),
                    "Cargo runtime registration was retired during renewal");
                Ok(())
            })?
        }));
        #[cfg(not(feature = "cargo-workspace"))]
        {
            let (_admission, _key, _cx) = (admission, key, cx);
            anyhow::bail!("Selected Cargo runtime authority requires enabled Cargo support")
        }
    }

    #[cfg(target_os = "macos")]
    fn observe_build_report(
        &self,
        observation: BuildReportObservation,
        cx: &mut Context<Self>,
    ) -> Task<()> {
        let key = observation.key.clone();
        let generation = observation.generation;
        let (registration_requests, registration_requested) = async_channel::bounded(1);
        let observation = cx.background_spawn(Self::observe_reserved_build_report(
            observation,
            registration_requests,
        ));
        cx.spawn(async move |this, cx| {
            let mut observation = observation;
            let observed = loop {
                match futures::future::select(
                    (&mut observation).boxed(),
                    registration_requested.recv().boxed(),
                )
                .await
                {
                    futures::future::Either::Left((observed, _)) => break observed,
                    futures::future::Either::Right((request, pending_observation)) => {
                        let Ok(answer) = request else {
                            break pending_observation.await;
                        };
                        let checked = match this.update(cx, |store, cx| {
                            store.check_build_report_registration(&key, generation, cx)
                        }) {
                            Ok(Ok(check)) => check.await,
                            Ok(Err(error)) => Err(error),
                            Err(error) => Err(error),
                        };
                        if answer.send(checked).is_err() {
                            log::trace!("Cargo runtime registration helper departed");
                        }
                    }
                }
            };
            this.update(cx, |this, cx| {
                let current = this.transport_connected
                    && this.remote_proxy.is_none()
                    && !TrustedWorktrees::has_restricted_worktrees(&this.worktree_store, cx);
                let expiry =
                    this.task_expiry(key.clone(), generation, Duration::from_secs(300), cx);
                let helper_current = this.task_runs.get(&key).is_some_and(|run| {
                    matches!(&run.payload, ProfileTaskPayload::BuildReport(build)
                        if this.build_report_helper_current(&build.helper, build.runtime_admission.as_deref(), cx))
                });
                if let Some(run) = this
                    .task_runs
                    .get_mut(&key)
                    .filter(|run| run.generation == generation)
                {
                    let ProfileTaskPayload::BuildReport(build) = &mut run.payload else {
                        return;
                    };
                    let observed = match observed {
                        Ok(observed) => Some(observed),
                        Err(_) => {
                            log::debug!(
                                "Build Task observation did not produce a successful bounded report"
                            );
                            None
                        }
                    }
                    .filter(|observed| {
                        current
                            && !run.cancellation.load(Ordering::Relaxed)
                            && helper_current
                            && Arc::ptr_eq(&observed.reservation, &build.reservation)
                    });
                    run.failed = observed.is_none();
                    if let Some(ReservedBuildReport {
                        reservation,
                        bytes,
                        fingerprint,
                        invocation,
                        cargo_runtime,
                    }) = observed
                    {
                        build.observed_invocation = invocation;
                        build.observed_cargo_runtime = cargo_runtime;
                        build.report = Some(bytes);
                        build.fingerprint = Some(fingerprint);
                        drop(reservation);
                    }
                    run._expiry = expiry;
                    if run.completion.send(()).is_err() {
                        log::trace!("Build Task completion has no pending waiter");
                    }
                }
            })
            .log_err();
        })
    }

    #[cfg(target_os = "macos")]
    fn await_build_report(
        &self,
        binding: BuildTaskReportBinding,
        cx: &mut Context<Self>,
    ) -> Task<Result<ObservedBuildTaskReport>> {
        if binding.store != cx.entity_id() || !self.build_binding_current(&binding, cx) {
            return Task::ready(Err(anyhow::anyhow!("Build report binding became stale")));
        }
        let pending = self.task_runs.get(&binding.key).and_then(|run| {
            matches!(&run.payload, ProfileTaskPayload::BuildReport(build) if build.report.is_none())
                .then(|| run.completion.receiver())
        });
        cx.spawn(async move |this, cx| {
            if let Some(mut completion) = pending {
                completion
                    .changed()
                    .await
                    .map_err(|_| anyhow::anyhow!("Build report observation was retired"))?;
            }
            this.update(cx, |this, cx| {
                anyhow::ensure!(
                    binding.store == cx.entity_id() && this.build_binding_current(&binding, cx),
                    "Build report observation became stale or unsuccessful"
                );
                let Some(run) = this.task_runs.get(&binding.key) else {
                    anyhow::bail!("Build report run disappeared");
                };
                let ProfileTaskPayload::BuildReport(build) = &run.payload else {
                    anyhow::bail!("Build report mode changed");
                };
                let bytes = build
                    .report
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("Build Task has no observed report"))?;
                let fingerprint = build
                    .fingerprint
                    .ok_or_else(|| anyhow::anyhow!("Build Task report identity is absent"))?;
                Ok(ObservedBuildTaskReport {
                    cargo_runtime: build.observed_cargo_runtime.clone(),
                    invocation: build.observed_invocation.clone(),
                    binding,
                    bytes,
                    fingerprint,
                })
            })?
        })
    }

    #[cfg(target_os = "macos")]
    fn observed_build_report_current(&self, report: &ObservedBuildTaskReport, cx: &App) -> bool {
        self.build_binding_current(&report.binding, cx)
            && self.task_runs.get(&report.binding.key).is_some_and(|run| {
                matches!(&run.payload, ProfileTaskPayload::BuildReport(build)
                if build.fingerprint == Some(report.fingerprint)
                    && build.report.as_ref().is_some_and(|bytes| Arc::ptr_eq(bytes, &report.bytes))
                    && match (&build.observed_cargo_runtime, &report.cargo_runtime) {
                        (Some(current), Some(observed)) => Arc::ptr_eq(current, observed),
                        (None, None) => true,
                        _ => false,
                    }
                    && match (&build.observed_invocation, &report.invocation) {
                        (Some(current), Some(observed)) => Arc::ptr_eq(current, observed),
                        (None, None) => true,
                        _ => false,
                    })
            })
    }

    fn task_versions(&self, paths: &[ProjectPath; 2], cx: &App) -> Option<[ProfileTaskVersion; 2]> {
        paths
            .iter()
            .map(|path| {
                if !self.permitted(path, cx) {
                    return None;
                }
                let worktree = self
                    .worktree_store
                    .read(cx)
                    .visible_worktrees(cx)
                    .find(|worktree| worktree.read(cx).id() == path.worktree_id)?;
                let snapshot = worktree.read(cx).snapshot();
                Some(ProfileTaskVersion {
                    entry: snapshot.entry_for_path(path.path.as_ref())?.clone(),
                    root_inode: snapshot.root_entry()?.inode,
                })
            })
            .collect::<Option<Vec<_>>>()?
            .try_into()
            .ok()
    }

    fn task_run_current(&self, run: &ProfileTaskRun, cx: &App) -> bool {
        let Some(native) = run.native() else {
            return false;
        };
        cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>()
            && self.transport_connected
            && !run.cancellation.load(Ordering::Relaxed)
            && !TrustedWorktrees::has_restricted_worktrees(&self.worktree_store, cx)
            && native.fingerprints.is_some()
            && native.versions.as_ref().is_some_and(|versions| {
                self.task_versions(&native.paths, cx).as_ref() == Some(versions)
            })
    }

    fn task_status(
        run_id: String,
        status: proto::ProfileResultStatus,
        complete: bool,
    ) -> proto::ProfileTaskBindingResponse {
        proto::ProfileTaskBindingResponse {
            protocol_version: PROFILE_PROTOCOL_VERSION,
            task_run_id: if valid_profile_task_run_id(&run_id) {
                run_id
            } else {
                String::new()
            },
            status: status as i32,
            complete,
            ..Default::default()
        }
    }

    async fn handle_get_task(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::GetProfileTaskBinding>,
        mut cx: AsyncApp,
    ) -> Result<proto::ProfileTaskBindingResponse> {
        let peer = envelope.original_sender_id.unwrap_or(envelope.sender_id);
        let request = envelope.payload;
        if this.read_with(&cx, |store, _| store.remote_proxy.is_some()) {
            return Ok(Self::task_status(
                request.task_run_id,
                proto::ProfileResultStatus::Unsupported,
                true,
            ));
        }
        if !valid_profile_task_run_id(&request.task_run_id) {
            return Ok(Self::task_status(
                String::new(),
                proto::ProfileResultStatus::Error,
                true,
            ));
        }
        let key = (ProfileTaskActor::Peer(peer), request.task_run_id.clone());
        let pending = this.read_with(&cx, |this, _| {
            (request.protocol_version == PROFILE_PROTOCOL_VERSION)
                .then(|| {
                    this.task_runs
                        .get(&key)
                        .filter(|run| {
                            !run.failed
                                && run.native().is_some_and(|native| native.versions.is_none())
                                && !run.cancellation.load(Ordering::Relaxed)
                        })
                        .map(|run| (run.generation, run.completion.receiver()))
                })
                .flatten()
        });
        let expected_generation = if let Some((generation, mut completion)) = pending {
            if completion.changed().await.is_err() {
                return Ok(Self::task_status(
                    request.task_run_id,
                    proto::ProfileResultStatus::Stale,
                    true,
                ));
            }
            Some(generation)
        } else {
            None
        };
        Ok(this.update(&mut cx, |this, cx| {
            let key = (ProfileTaskActor::Peer(peer), request.task_run_id.clone());
            let (status, complete) = if request.protocol_version != PROFILE_PROTOCOL_VERSION {
                (proto::ProfileResultStatus::Mismatch, true)
            } else if let Some(run) = this.task_runs.get(&key) {
                if expected_generation.is_some_and(|generation| generation != run.generation) {
                    (proto::ProfileResultStatus::Stale, true)
                } else if run.failed {
                    (proto::ProfileResultStatus::Error, true)
                } else if run.native().is_some_and(|native| native.versions.is_none())
                    && !run.cancellation.load(Ordering::Relaxed)
                {
                    (proto::ProfileResultStatus::Ready, false)
                } else if this.task_run_current(run, cx) {
                    (proto::ProfileResultStatus::Ready, true)
                } else {
                    (proto::ProfileResultStatus::Stale, true)
                }
            } else {
                (proto::ProfileResultStatus::Stale, true)
            };
            Self::task_status(request.task_run_id, status, complete)
        }))
    }

    async fn handle_cancel_task(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::CancelProfileTaskBinding>,
        mut cx: AsyncApp,
    ) -> Result<proto::Ack> {
        let peer = envelope.original_sender_id.unwrap_or(envelope.sender_id);
        if valid_profile_task_run_id(&envelope.payload.task_run_id) {
            this.update(&mut cx, |this, _| {
                this.remove_task_run(&(ProfileTaskActor::Peer(peer), envelope.payload.task_run_id))
            });
        }
        Ok(proto::Ack {})
    }

    async fn handle_prepare_task(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::PrepareProfileTaskBinding>,
        mut cx: AsyncApp,
    ) -> Result<proto::ProfileTaskBindingResponse> {
        let peer = envelope.original_sender_id.unwrap_or(envelope.sender_id);
        let response = this
            .update(&mut cx, |this, cx| {
                this.prepare_task(peer, envelope.payload, cx)
            })
            .await;
        #[cfg(any(test, feature = "test-support"))]
        if let Some((started, release)) = this.update(&mut cx, |store, _| {
            store.task_preparation_response_gate.take()
        }) {
            anyhow::ensure!(
                response.status == proto::ProfileResultStatus::Ready as i32
                    && !response.complete
                    && response
                        .helper_path
                        .as_ref()
                        .is_some_and(|path| !path.is_empty())
                    && response
                        .socket_path
                        .as_ref()
                        .is_some_and(|path| !path.is_empty())
                    && response
                        .token
                        .as_ref()
                        .is_some_and(|token| token.len() == 64),
                "Held native preparation did not issue an actual host binding"
            );
            started.send(()).map_err(|_| {
                anyhow::anyhow!("Native preparation response start receiver was dropped")
            })?;
            release
                .await
                .map_err(|_| anyhow::anyhow!("Native preparation response release was dropped"))?;
        }
        Ok(response)
    }

    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    fn prepare_checked_local_native_pair(
        &mut self,
        run_id: String,
        paths: [ProjectPath; 2],
        expected: util::macos_task_completion::TaskBuildInvocation,
        cx: &mut Context<Self>,
    ) -> Task<Result<LocalNativePairBinding>> {
        let retirements = self.prune_cancelled_task_runs();
        if retirements.is_empty() {
            return self
                .prepare_checked_local_native_pair_after_retirement(run_id, paths, expected, cx);
        }
        let deadline = cx.background_executor().timer(Duration::from_secs(30));
        cx.spawn(async move |this, cx| {
            smol::future::race(
                async move {
                    for retirement in retirements {
                        anyhow::ensure!(
                            retirement.recv().await.is_err(),
                            "Native invocation retirement notification was invalid"
                        );
                    }
                    anyhow::Ok(())
                },
                async move {
                    deadline.await;
                    anyhow::bail!("Cancelled native invocation did not retire within 30 seconds")
                },
            )
            .await?;
            this.update(cx, |this, cx| {
                this.prepare_checked_local_native_pair_after_retirement(run_id, paths, expected, cx)
            })?
            .await
        })
    }

    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    fn prepare_checked_local_native_pair_after_retirement(
        &mut self,
        run_id: String,
        paths: [ProjectPath; 2],
        expected: util::macos_task_completion::TaskBuildInvocation,
        cx: &mut Context<Self>,
    ) -> Task<Result<LocalNativePairBinding>> {
        let checked = (|| {
            expected.validate().map_err(|_| {
                anyhow::anyhow!("Native invocation is outside its supported bounds")
            })?;
            anyhow::ensure!(
                !expected.environment().contains_key("_NT_SYMBOL_PATH"),
                "Native invocation contains an unsupported symbol service environment"
            );
            Ok::<_, anyhow::Error>(Arc::new(NativePairInvocation {
                expected,
                reservation: BuildReportReservation::reserve_native_invocation(
                    self.build_report_budget.clone(),
                )?,
            }))
        })();
        let checked = match checked {
            Ok(checked) => checked,
            Err(error) => return Task::ready(Err(error)),
        };
        let key = (ProfileTaskActor::Local(cx.entity_id()), run_id.clone());
        let prepared = self.prepare_task_for_actor_with_native_invocation(
            key.0,
            ProfileTaskHelperMode::LocalApplication,
            proto::PrepareProfileTaskBinding {
                project_id: 0,
                protocol_version: PROFILE_PROTOCOL_VERSION,
                task_run_id: run_id.clone(),
                profile_path: Some(paths[0].to_proto()),
                symbols_path: Some(paths[1].to_proto()),
                artifact_format: "samply-0.13.1".into(),
            },
            Some(checked.clone()),
            cx,
        );
        let expected_generation = self.task_runs.get(&key).map(|run| run.generation);
        cx.spawn(async move |this, cx| {
            let response = check_task_binding_response(prepared.await, &run_id, false)?;
            this.update(cx, |store, _| {
                let result = (|| {
                    let run = store
                        .task_runs
                        .get(&key)
                        .context("Native pair owner expired during preparation")?;
                    let helper = store
                        .task_helper
                        .as_ref()
                        .filter(|helper| helper.mode == ProfileTaskHelperMode::LocalApplication)
                        .context("Native pair helper route changed")?
                        .clone();
                    anyhow::ensure!(
                        response.helper_path.as_deref() == helper.path.to_str()
                            && !run.cancellation.load(Ordering::Relaxed),
                        "Native pair helper binding changed"
                    );
                    anyhow::ensure!(
                        run.native()
                            .and_then(|native| native.checked_invocation.as_ref())
                            .is_some_and(|current| Arc::ptr_eq(current, &checked)),
                        "Native checked invocation ownership changed"
                    );
                    Ok(LocalNativePairBinding {
                        checked_invocation: checked,
                        key: key.clone(),
                        generation: run.generation,
                        helper,
                        binding: ProfileTaskBinding {
                            helper_path: response
                                .helper_path
                                .context("Native helper path is unavailable")?,
                            socket_path: response
                                .socket_path
                                .context("Native helper socket is unavailable")?,
                            token: response
                                .token
                                .context("Native helper token is unavailable")?,
                        },
                        cancellation: run.cancellation.clone(),
                        stop: run.stop.clone(),
                        armed: true,
                    })
                })();
                if result.is_err()
                    && store
                        .task_runs
                        .get(&key)
                        .is_some_and(|run| Some(run.generation) == expected_generation)
                {
                    store.remove_task_run(&key);
                }
                result
            })?
        })
    }

    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    fn check_native_capture_registration(
        &self,
        key: &ProfileTaskKey,
        generation: u64,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<()>>> {
        anyhow::ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling experimental opt-in was disabled"
        );

        let run = self
            .task_runs
            .get(key)
            .filter(|run| {
                run.generation == generation
                    && !run.failed
                    && !run.cancellation.load(Ordering::Relaxed)
            })
            .context("Native capture registration owner expired")?;
        let native = run.native().context("Native capture pair owner changed")?;
        let checked_invocation = native
            .checked_invocation
            .as_ref()
            .context("Native checked invocation is unavailable")?
            .clone();
        let (authority, helper) = native
            .capture_admission
            .as_ref()
            .context("Native capture authority is unavailable")?;
        anyhow::ensure!(
            self.task_helper.as_ref() == Some(helper)
                && self.transport_connected
                && !TrustedWorktrees::has_restricted_worktrees(&self.worktree_store, cx),
            "Native capture registration route or trust changed"
        );
        let authority = authority.clone();
        let helper = helper.clone();
        let key = key.clone();
        let paths = native.paths.clone();
        let plan = authority
            .lock()
            .take()
            .context("Native capture authority is already in use")?;
        let project = plan.project()?;
        let cargo = project.read(cx).cargo_workspace_store().clone();
        let bound = plan.resolved.bind_artifact_invocation(&key.1)?;
        let group = bound
            .resolved_artifact_group()
            .context("Native capture bound pair disappeared")?;
        let artifact_directory = bound
            .artifact_directory()
            .or(bound.resolved.cwd.as_deref())
            .context("Native capture artifact directory disappeared")?;
        let reviewed =
            project
                .read(cx)
                .review_native_capture_outputs(artifact_directory, group, cx)?;
        anyhow::ensure!(
            reviewed.1 == paths,
            "Native capture bound pair route changed"
        );
        let absolute = group
            .files
            .values()
            .map(|artifact| artifact_directory.join(&artifact.path))
            .collect::<Vec<_>>();
        let executor = cx.background_executor().clone();
        Ok(cx.spawn(async move |this, cx| {
            let (filesystem, _, roots) = reviewed;
            cx.background_spawn(check_native_capture_outputs(
                filesystem, roots, absolute, executor,
            ))
            .await?;
            let checked = cargo.update(cx, |store, cx| {
                store.revalidate_native_capture_plan(plan, cx)
            })?;
            let plan = checked.await?;
            anyhow::ensure!(
                plan.expected_capture_invocation(&bound.resolved)? == checked_invocation.expected,
                "Native capture environment or invocation changed before helper authorization"
            );
            if plan.has_observed_runtime() {
                plan.validate_terminal_loader_environment(&checked_invocation.expected)?;
            }
            this.update(cx, |store, cx| {
                let current = store
                    .task_runs
                    .get(&key)
                    .filter(|run| {
                        run.generation == generation
                            && !run.failed
                            && !run.cancellation.load(Ordering::Relaxed)
                    })
                    .and_then(ProfileTaskRun::native)
                    .and_then(|native| native.capture_admission.as_ref())
                    .is_some_and(|(current, _)| Arc::ptr_eq(current, &authority));
                anyhow::ensure!(
                    current
                        && store.transport_connected
                        && store.task_helper.as_ref() == Some(&helper)
                        && !TrustedWorktrees::has_restricted_worktrees(&store.worktree_store, cx),
                    "Native capture registration was retired"
                );
                let project = plan.project()?;
                anyhow::ensure!(
                    project.read(cx).is_local() && !project.read(cx).is_shared(),
                    "Native capture Project route changed"
                );
                *authority.lock() = Some(plan);
                Ok(())
            })?
        }))
    }

    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    fn await_local_native_pair(
        &self,
        binding: &LocalNativePairBinding,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<()>>> {
        anyhow::ensure!(
            binding.key.0 == ProfileTaskActor::Local(cx.entity_id())
                && self.task_helper.as_ref() == Some(&binding.helper),
            "Native pair owner or helper changed"
        );
        let run = self
            .task_runs
            .get(&binding.key)
            .filter(|run| {
                run.generation == binding.generation
                    && Arc::ptr_eq(&run.cancellation, &binding.cancellation)
                    && !run.cancellation.load(Ordering::Relaxed)
                    && run
                        .native()
                        .and_then(|native| native.checked_invocation.as_ref())
                        .is_some_and(|checked| Arc::ptr_eq(checked, &binding.checked_invocation))
            })
            .context("Native pair capture became stale")?;
        let pending = run.native().is_some_and(|native| native.versions.is_none()) && !run.failed;
        let mut completed = run.completion.receiver();
        let key = binding.key.clone();
        let generation = binding.generation;
        let helper = binding.helper.clone();
        let checked_invocation = binding.checked_invocation.clone();
        Ok(cx.spawn(async move |this, cx| {
            if pending {
                completed
                    .changed()
                    .await
                    .map_err(|_| anyhow::anyhow!("Native pair capture cancelled"))?;
            }
            this.read_with(cx, |store, cx| {
                anyhow::ensure!(
                    store.task_helper.as_ref() == Some(&helper)
                        && store.task_runs.get(&key).is_some_and(|run| run.generation
                            == generation
                            && !run.failed
                            && store.task_run_current(run, cx)
                            && run
                                .native()
                                .and_then(|native| native.checked_invocation.as_ref())
                                .is_some_and(|current| Arc::ptr_eq(current, &checked_invocation))),
                    "Native pair capture did not complete with current sealed artifacts"
                );
                Ok(())
            })?
        }))
    }

    fn prepare_task(
        &mut self,
        peer: proto::PeerId,
        request: proto::PrepareProfileTaskBinding,
        cx: &mut Context<Self>,
    ) -> Task<proto::ProfileTaskBindingResponse> {
        self.prepare_task_for_actor(
            ProfileTaskActor::Peer(peer),
            ProfileTaskHelperMode::RemoteServer,
            request,
            cx,
        )
    }

    fn prepare_task_for_actor(
        &mut self,
        actor: ProfileTaskActor,
        helper_mode: ProfileTaskHelperMode,
        request: proto::PrepareProfileTaskBinding,
        cx: &mut Context<Self>,
    ) -> Task<proto::ProfileTaskBindingResponse> {
        #[cfg(not(target_os = "macos"))]
        {
            _ = (actor, helper_mode, cx);
            return Task::ready(Self::task_status(
                request.task_run_id,
                proto::ProfileResultStatus::Unsupported,
                true,
            ));
        }
        #[cfg(target_os = "macos")]
        {
            self.prepare_task_for_actor_with_native_invocation(
                actor,
                helper_mode,
                request,
                None,
                cx,
            )
        }
    }

    #[cfg(target_os = "macos")]
    fn prepare_task_for_actor_with_native_invocation(
        &mut self,
        actor: ProfileTaskActor,
        helper_mode: ProfileTaskHelperMode,
        request: proto::PrepareProfileTaskBinding,
        checked_invocation: Option<Arc<NativePairInvocation>>,
        cx: &mut Context<Self>,
    ) -> Task<proto::ProfileTaskBindingResponse> {
        {
            let run_id = request.task_run_id.clone();
            let fail = |status| Task::ready(Self::task_status(run_id.clone(), status, true));
            if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
                return fail(proto::ProfileResultStatus::Unsupported);
            }
            if checked_invocation.is_some()
                && (!matches!(actor, ProfileTaskActor::Local(_))
                    || helper_mode != ProfileTaskHelperMode::LocalApplication)
            {
                return fail(proto::ProfileResultStatus::Unsupported);
            }
            let Some(helper) = self
                .task_helper
                .as_ref()
                .filter(|helper| helper.mode == helper_mode)
                .and_then(|helper| helper.path.to_str())
                .map(str::to_owned)
            else {
                return fail(proto::ProfileResultStatus::Unsupported);
            };
            let paths = match profile_import_paths(
                &proto::ImportProfileResults {
                    project_id: request.project_id,
                    import_id: 1,
                    protocol_version: request.protocol_version,
                    profile_path: request.profile_path,
                    symbols_path: request.symbols_path,
                    consent_to_transfer: true,
                    task_run_id: Some(run_id.clone()),
                    artifact_format: request.artifact_format,
                },
                ProfileHost::MacOs,
            ) {
                Ok(paths) => paths,
                Err(status) => return fail(status),
            };
            if !self.transport_connected {
                return fail(proto::ProfileResultStatus::Stale);
            }
            if TrustedWorktrees::has_restricted_worktrees(&self.worktree_store, cx)
                || !paths.iter().all(|path| {
                    self.worktree_store
                        .read(cx)
                        .visible_worktrees(cx)
                        .any(|tree| {
                            tree.read(cx).id() == path.worktree_id
                                && tree.read(cx).as_local().is_some()
                        })
                })
            {
                return fail(proto::ProfileResultStatus::Restricted);
            }
            drop(self.prune_cancelled_task_runs());
            let key = (actor, run_id.clone());
            if self.task_runs.contains_key(&key) || self.task_runs.len() >= MAX_HOST_PROFILE_TASKS {
                return fail(proto::ProfileResultStatus::Error);
            }
            let Some(generation) = self.generation.checked_add(1) else {
                return fail(proto::ProfileResultStatus::Error);
            };
            self.generation = generation;
            let cancellation = Arc::new(AtomicBool::new(false));
            let (stop, stopped) = async_channel::bounded(1);
            let expiry = self.task_expiry(key.clone(), generation, Duration::from_secs(30), cx);
            self.task_runs.insert(
                key.clone(),
                ProfileTaskRun {
                    completion: watch::channel(()).0,
                    generation,
                    payload: ProfileTaskPayload::NativePair(NativePairTaskRun {
                        checked_invocation: checked_invocation.clone(),
                        paths: paths.clone(),
                        versions: None,
                        fingerprints: None,
                        #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
                        capture_admission: None,
                    }),
                    failed: false,
                    cancellation: cancellation.clone(),
                    stop,
                    _expiry: expiry,
                    observation: None,
                },
            );
            let checked_filesystem = if helper_mode == ProfileTaskHelperMode::LocalApplication {
                match self.worktree_store.read(cx).fs() {
                    Some(filesystem) => Some(filesystem),
                    None => {
                        self.remove_task_run(&key);
                        return fail(proto::ProfileResultStatus::Unsupported);
                    }
                }
            } else {
                None
            };
            let helper_path = PathBuf::from(&helper);
            let identity_cancellation = cancellation.clone();
            let mut cancel_on_drop = ImportCancellation(Some(cancellation));
            let token = rand::random::<[u8; 32]>();
            let binding = cx.background_spawn(async move {
                let checked_helper = match checked_filesystem {
                    Some(filesystem) => {
                        let identity = crate::profile_results::prepare_task_helper_identity(
                            filesystem.as_ref(),
                            helper_path,
                            &identity_cancellation,
                        )
                        .await?;
                        Some((filesystem, identity, identity_cancellation))
                    }
                    None => None,
                };
                let listener = util::macos_task_completion::TaskCompletionListener::bind(token)?;
                anyhow::Ok((listener, checked_helper))
            });
            cx.spawn(async move |this, cx| {
                let binding = binding.await;
                let response = this
                    .update(cx, |this, cx| {
                        if !this.task_runs.get(&key).is_some_and(|run| {
                            run.generation == generation
                                && !run.cancellation.load(Ordering::Relaxed)
                        }) {
                            return Self::task_status(
                                run_id.clone(),
                                proto::ProfileResultStatus::Stale,
                                true,
                            );
                        }
                        let Ok((listener, checked_helper)) = binding else {
                            this.remove_task_run(&key);
                            return Self::task_status(
                                run_id.clone(),
                                proto::ProfileResultStatus::Error,
                                true,
                            );
                        };
                        let socket = listener.path().to_string_lossy().into_owned();
                        let observation = this.observe_task(
                            key.clone(),
                            generation,
                            paths,
                            listener,
                            stopped,
                            checked_helper,
                            checked_invocation,
                            cx,
                        );
                        let expiry = this.task_expiry(
                            key.clone(),
                            generation,
                            Duration::from_secs(3630),
                            cx,
                        );
                        if let Some(run) = this.task_runs.get_mut(&key) {
                            run.observation = Some(observation);
                            run._expiry = expiry;
                        }
                        proto::ProfileTaskBindingResponse {
                            helper_path: Some(helper),
                            socket_path: Some(socket),
                            token: Some(token.iter().map(|byte| format!("{byte:02x}")).collect()),
                            ..Self::task_status(
                                run_id.clone(),
                                proto::ProfileResultStatus::Ready,
                                false,
                            )
                        }
                    })
                    .unwrap_or_else(|_| {
                        Self::task_status(run_id, proto::ProfileResultStatus::Cancelled, true)
                    });
                cancel_on_drop.disarm();
                response
            })
        }
    }

    #[cfg(target_os = "macos")]
    fn task_expiry(
        &self,
        key: ProfileTaskKey,
        generation: u64,
        duration: Duration,
        cx: &mut Context<Self>,
    ) -> Task<()> {
        let timer = cx.background_executor().timer(duration);
        cx.spawn(async move |this, cx| {
            timer.await;
            this.update(cx, |this, _| {
                if this
                    .task_runs
                    .get(&key)
                    .is_some_and(|run| run.generation == generation)
                {
                    this.remove_task_run(&key);
                }
            })
            .log_err();
        })
    }

    #[cfg(target_os = "macos")]
    async fn observe_native_pair_process(
        listener: util::macos_task_completion::TaskCompletionListener,
        stopped: async_channel::Receiver<()>,
        checked_helper: Option<(
            Arc<dyn fs::Fs>,
            crate::profile_results::TaskHelperIdentity,
            Arc<AtomicBool>,
        )>,
        checked_invocation: Option<Arc<NativePairInvocation>>,
        registration_requests: async_channel::Sender<futures::channel::oneshot::Sender<Result<()>>>,
    ) -> std::io::Result<std::process::ExitStatus> {
        let observed = match checked_helper {
            Some((filesystem, identity, cancellation)) => {
                let is_checked_native = checked_invocation.is_some();
                let check = async move |executable: PathBuf| {
                    let actual = filesystem.canonicalize(&executable).await.map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            "Cannot identify the registered Native Task helper",
                        )
                    })?;
                    if actual.as_path() != identity.canonical_path() {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            "Native Task helper image mismatch",
                        ));
                    }
                    crate::profile_results::revalidate_task_helper_identity(
                        filesystem.as_ref(),
                        &identity,
                        &cancellation,
                    )
                    .await
                    .map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::PermissionDenied,
                            "Native Task helper image changed",
                        )
                    })?;
                    if !is_checked_native {
                        return Ok(());
                    }
                    let (answer, response) = futures::channel::oneshot::channel();
                    registration_requests.send(answer).await.map_err(|_| {
                        std::io::Error::new(
                            std::io::ErrorKind::Interrupted,
                            "Native capture registration owner disappeared",
                        )
                    })?;
                    response
                        .await
                        .map_err(|_| {
                            std::io::Error::new(
                                std::io::ErrorKind::Interrupted,
                                "Native capture registration was cancelled",
                            )
                        })?
                        .map_err(|_| {
                            std::io::Error::new(
                                std::io::ErrorKind::PermissionDenied,
                                "Native capture registration became stale",
                            )
                        })
                };
                match checked_invocation.as_ref() {
                    Some(checked) => listener
                        .observe_checked_native_profile_with_helper_check(
                            checked.expected.clone(),
                            check,
                        )
                        .boxed(),
                    None => listener
                        .observe_native_profile_with_helper_check(check)
                        .boxed(),
                }
            }
            None if checked_invocation.is_some() => {
                futures::future::ready(Err(std::io::Error::new(
                    std::io::ErrorKind::PermissionDenied,
                    "Checked native capture requires its sealed local helper",
                )))
                .boxed()
            }
            None => listener.observe().boxed(),
        };
        let result = smol::future::race(observed, async {
            if let Err(async_channel::RecvError) = stopped.recv().await {
                log::trace!("Profile Task observation channel closed");
            }
            Err(std::io::Error::new(
                std::io::ErrorKind::Interrupted,
                "Profile Task observation cancelled",
            ))
        })
        .await;
        // Precise async capture must not release the reservation while the
        // observed frame, helper or kernel-exit future can still retain it.
        drop(checked_invocation);
        result
    }

    #[cfg(target_os = "macos")]
    fn observe_task(
        &self,
        key: ProfileTaskKey,
        generation: u64,
        paths: [ProjectPath; 2],
        listener: util::macos_task_completion::TaskCompletionListener,
        stopped: async_channel::Receiver<()>,
        checked_helper: Option<(
            Arc<dyn fs::Fs>,
            crate::profile_results::TaskHelperIdentity,
            Arc<AtomicBool>,
        )>,
        checked_invocation: Option<Arc<NativePairInvocation>>,
        cx: &mut Context<Self>,
    ) -> Task<()> {
        use futures::StreamExt as _;
        let (registration_requests, registration_requested) =
            async_channel::bounded::<futures::channel::oneshot::Sender<Result<()>>>(1);
        let observation = cx.background_spawn(Self::observe_native_pair_process(
            listener,
            stopped,
            checked_helper,
            checked_invocation,
            registration_requests,
        ));
        cx.spawn(async move |this, cx| {
            let mut observation = observation;
            let observed = loop {
                match futures::future::select(
                    (&mut observation).boxed(),
                    registration_requested.recv().boxed(),
                )
                .await
                {
                    futures::future::Either::Left((observed, _)) => break observed,
                    futures::future::Either::Right((request, pending_observation)) => {
                        let Ok(answer) = request else {
                            break pending_observation.await;
                        };
                        #[cfg(feature = "cargo-workspace")]
                        let checked = match this.update(cx, |store, cx| {
                            store.check_native_capture_registration(&key, generation, cx)
                        }) {
                            Ok(Ok(check)) => check.await,
                            Ok(Err(error)) => Err(error),
                            Err(error) => Err(error),
                        };
                        #[cfg(not(feature = "cargo-workspace"))]
                        let checked =
                            Err(anyhow::anyhow!("Selected native capture is unavailable"));
                        #[cfg(any(test, feature = "test-support"))]
                        if checked.is_err() {
                            eprintln!(
                                "native capture registration renewal: category=authority-refused"
                            );
                        }
                        if answer.send(checked).is_err() {
                            log::trace!("Native capture registration helper departed");
                        }
                    }
                }
            };
            #[cfg(any(test, feature = "test-support"))]
            {
                let category = match &observed {
                    Ok(status) if status.success() => "kernel-success",
                    Ok(_) => "kernel-non-success",
                    Err(error) => match error.kind() {
                        std::io::ErrorKind::TimedOut => "registration-timeout",
                        std::io::ErrorKind::UnexpectedEof => "registration-eof",
                        std::io::ErrorKind::PermissionDenied => "registration-refused",
                        _ => "observation-refused",
                    },
                };
                eprintln!("native capture kernel observation: category={category}");
            }
            let success = observed.is_ok_and(|status| status.success());

            let versions = if success {
                let refreshes = this
                    .update(cx, |this, cx| {
                        paths
                            .iter()
                            .map(|path| {
                                let tree = this
                                    .worktree_store
                                    .read(cx)
                                    .visible_worktrees(cx)
                                    .find(|tree| tree.read(cx).id() == path.worktree_id)?;
                                Some(
                                    tree.read(cx)
                                        .as_local()?
                                        .refresh_entries_for_paths(vec![path.path.clone()]),
                                )
                            })
                            .collect::<Option<Vec<_>>>()
                    })
                    .ok()
                    .flatten();
                if let Some(refreshes) = refreshes {
                    let mut refreshed = true;
                    for mut refresh in refreshes {
                        refreshed &= refresh.next().await.is_some();
                    }
                    if refreshed {
                        this.read_with(cx, |this, cx| this.task_versions(&paths, cx))
                            .ok()
                            .flatten()
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
            let fingerprinting = versions.as_ref().and_then(|_| {
                this.read_with(cx, |this, cx| {
                    let cancellation = this.task_runs.get(&key)?.cancellation.clone();
                    let fs = this.worktree_store.read(cx).fs()?;
                    let inputs = paths
                        .iter()
                        .map(|path| {
                            let tree = this
                                .worktree_store
                                .read(cx)
                                .visible_worktrees(cx)
                                .find(|tree| tree.read(cx).id() == path.worktree_id)?;
                            ProfileInput::from_snapshot(
                                &tree.read(cx).snapshot(),
                                path.path.as_ref(),
                            )
                            .ok()
                        })
                        .collect::<Option<Vec<_>>>()?;
                    let inputs: [ProfileInput; 2] = inputs.try_into().ok()?;
                    Some(cx.background_spawn(async move {
                        crate::profile_results::fingerprint_profile_inputs(
                            fs.as_ref(),
                            &inputs,
                            &cancellation,
                        )
                        .await
                    }))
                })
                .ok()
                .flatten()
            });
            let fingerprints = match fingerprinting {
                Some(task) => task.await.log_err(),
                None => None,
            };
            this.update(cx, |this, cx| {
                let versions = versions.filter(|versions| {
                    fingerprints.is_some()
                        && this.task_versions(&paths, cx).as_ref() == Some(versions)
                });
                let expiry =
                    this.task_expiry(key.clone(), generation, Duration::from_secs(300), cx);
                if let Some(run) = this
                    .task_runs
                    .get_mut(&key)
                    .filter(|run| run.generation == generation)
                {
                    run.failed = versions.is_none();
                    if let Some(native) = run.native_mut() {
                        native.versions = versions;
                        native.fingerprints = fingerprints;
                    }
                    run._expiry = expiry;
                    if run.completion.send(()).is_err() {
                        log::trace!("Profile Task completion has no pending waiter");
                    }
                }
            })
            .log_err();
        })
    }

    fn permitted(&self, path: &ProjectPath, cx: &App) -> bool {
        self.worktree_store
            .read(cx)
            .visible_worktrees(cx)
            .any(|worktree| {
                let worktree = worktree.read(cx);
                worktree.id() == path.worktree_id
                    && worktree
                        .snapshot()
                        .entry_for_path(path.path.as_ref())
                        .is_some_and(|entry| {
                            entry.is_file()
                                && !entry.is_private
                                && !entry.is_fifo
                                && !entry.is_external
                        })
            })
    }

    fn status(
        import_id: u64,
        generation: u64,
        status: proto::ProfileResultStatus,
    ) -> proto::ProfileResultPage {
        proto::ProfileResultPage {
            import_id,
            generation,
            protocol_version: PROFILE_PROTOCOL_VERSION,
            status: status as i32,
            ..Default::default()
        }
    }

    fn import(
        &mut self,
        peer: proto::PeerId,
        request: proto::ImportProfileResults,
        cx: &mut Context<Self>,
    ) -> Task<proto::ProfileResultPage> {
        let host = if cfg!(target_os = "macos") {
            ProfileHost::MacOs
        } else if cfg!(target_os = "windows") {
            ProfileHost::Windows
        } else {
            ProfileHost::Linux
        };
        let import_id = request.import_id;
        if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
            return Task::ready(Self::status(
                import_id,
                0,
                proto::ProfileResultStatus::Unsupported,
            ));
        }
        if !self.transport_connected {
            return Task::ready(Self::status(
                import_id,
                0,
                proto::ProfileResultStatus::Stale,
            ));
        }
        let key = (peer, import_id);
        if self.cancelled.remove(&key) {
            return Task::ready(Self::status(
                import_id,
                0,
                proto::ProfileResultStatus::Cancelled,
            ));
        }
        self.imports
            .retain(|_, import| !import.cancellation.load(Ordering::Relaxed));
        let inputs = match profile_import_paths(&request, host) {
            Ok(inputs) => inputs,
            Err(status) => return Task::ready(Self::status(import_id, 0, status)),
        };
        if let Some(run_id) = &request.task_run_id
            && !self
                .task_runs
                .get(&(ProfileTaskActor::Peer(peer), run_id.clone()))
                .is_some_and(|run| {
                    run.native().is_some_and(|native| native.paths == inputs)
                        && self.task_run_current(run, cx)
                })
        {
            return Task::ready(Self::status(
                import_id,
                0,
                proto::ProfileResultStatus::Stale,
            ));
        }
        if TrustedWorktrees::has_restricted_worktrees(&self.worktree_store, cx)
            || !inputs.iter().all(|path| self.permitted(path, cx))
        {
            return Task::ready(Self::status(
                import_id,
                0,
                proto::ProfileResultStatus::Restricted,
            ));
        }
        let Some(fs) = self.worktree_store.read(cx).fs() else {
            return Task::ready(Self::status(
                import_id,
                0,
                proto::ProfileResultStatus::Unsupported,
            ));
        };
        if !self.imports.contains_key(&key) && self.imports.len() >= MAX_HOST_PROFILE_IMPORTS {
            return Task::ready(Self::status(
                import_id,
                0,
                proto::ProfileResultStatus::Error,
            ));
        }
        let Some(generation) = self.generation.checked_add(1) else {
            return Task::ready(Self::status(
                import_id,
                0,
                proto::ProfileResultStatus::Error,
            ));
        };
        self.generation = generation;
        let worktrees = self
            .worktree_store
            .read(cx)
            .visible_worktrees(cx)
            .map(|worktree| {
                let worktree = worktree.read(cx);
                (worktree.id(), worktree.abs_path(), worktree.snapshot())
            })
            .collect::<Vec<_>>();
        let admitted_inputs = inputs
            .iter()
            .map(|path| {
                worktrees
                    .iter()
                    .find(|(id, _, _)| *id == path.worktree_id)
                    .and_then(|(_, _, snapshot)| {
                        ProfileInput::from_snapshot(snapshot, path.path.as_ref()).ok()
                    })
            })
            .collect::<Option<Vec<_>>>();
        let Some(mut admitted_inputs) = admitted_inputs else {
            return Task::ready(Self::status(
                import_id,
                0,
                proto::ProfileResultStatus::Restricted,
            ));
        };
        if let Some(run_id) = &request.task_run_id {
            let Some(fingerprints) = self
                .task_runs
                .get(&(ProfileTaskActor::Peer(peer), run_id.clone()))
                .and_then(|run| run.native()?.fingerprints)
            else {
                return Task::ready(Self::status(
                    import_id,
                    0,
                    proto::ProfileResultStatus::Stale,
                ));
            };
            admitted_inputs = admitted_inputs
                .into_iter()
                .zip(fingerprints)
                .map(|(input, fingerprint)| input.with_fingerprint(fingerprint))
                .collect();
        }
        let cancellation = Arc::new(AtomicBool::new(false));
        self.imports.insert(
            key,
            ProfileImport {
                purpose: ProfileImportPurpose::Native,
                task_run_id: request.task_run_id,
                generation,
                cancellation: cancellation.clone(),
                paths: inputs.iter().cloned().collect(),
                pages: VecDeque::new(),
                next_page: 0,
                revision: self.revision,
            },
        );
        let mut cancel_on_drop = ImportCancellation(Some(cancellation.clone()));
        #[cfg(any(test, feature = "test-support"))]
        let processing_gate = self.import_processing_gate.take();
        let parsing = cx.background_spawn(async move {
            #[cfg(any(test, feature = "test-support"))]
            if let Some((started, release)) = processing_gate {
                started
                    .send(())
                    .map_err(|_| anyhow::anyhow!("Profile import start receiver was dropped"))?;
                release.await?;
            }
            let profile_input = admitted_inputs
                .first()
                .ok_or_else(|| anyhow::anyhow!("Missing profile input"))?;
            let symbols_input = admitted_inputs
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("Missing symbol input"))?;
            let result = load_samply_profile(
                fs.as_ref(),
                [profile_input, symbols_input],
                host,
                &cancellation,
                |path, _| {
                    worktrees.iter().find_map(|(id, root, snapshot)| {
                        let relative = path.strip_prefix(root).ok()?;
                        let relative =
                            util::rel_path::RelPath::new(relative, util::paths::PathStyle::local())
                                .ok()?;
                        let entry = snapshot.entry_for_path(relative.as_ref())?;
                        (entry.is_file()
                            && !entry.is_private
                            && !entry.is_fifo
                            && !entry.is_external)
                            .then(|| ProjectPath {
                                worktree_id: *id,
                                path: Arc::from(relative.as_ref()),
                            })
                    })
                },
            )
            .await?;
            let mut paths = inputs.into_iter().collect::<HashSet<_>>();
            paths.extend(
                result
                    .frames
                    .iter()
                    .filter_map(|frame| frame.source.as_ref().map(|(path, _)| path.clone())),
            );
            let pages =
                profile_result_pages(&result, import_id, generation, &cancellation, |path| {
                    worktrees.iter().any(|(id, _, snapshot)| {
                        *id == path.worktree_id
                            && snapshot
                                .entry_for_path(path.path.as_ref())
                                .is_some_and(|entry| {
                                    entry.is_file()
                                        && !entry.is_private
                                        && !entry.is_fifo
                                        && !entry.is_external
                                })
                    })
                })?;
            anyhow::Ok((pages, paths))
        });
        cx.spawn(async move |this, cx| {
            let result = parsing.await;
            let response = this.update(cx, |this, cx| {
                let Some(import) = this.imports.get(&key) else {
                    return Self::status(
                        import_id,
                        generation,
                        proto::ProfileResultStatus::Cancelled,
                    );
                };
                if import.generation != generation {
                    return Self::status(import_id, generation, proto::ProfileResultStatus::Stale);
                }
                let (pages, paths) = match result {
                    Ok(result) => result,
                    Err(_) => {
                        this.imports.remove(&key);
                        return Self::status(
                            import_id,
                            generation,
                            proto::ProfileResultStatus::Error,
                        );
                    }
                };
                if import.revision != this.revision {
                    this.imports.remove(&key);
                    return Self::status(import_id, generation, proto::ProfileResultStatus::Stale);
                }
                if TrustedWorktrees::has_restricted_worktrees(&this.worktree_store, cx)
                    || !paths.iter().all(|path| this.permitted(path, cx))
                {
                    this.imports.remove(&key);
                    return Self::status(
                        import_id,
                        generation,
                        proto::ProfileResultStatus::Restricted,
                    );
                }
                if let Some(import) = this.imports.get_mut(&key) {
                    import.paths = paths;
                    import.pages = pages.into();
                }
                this.page(key, generation, 0, cx)
            });
            cancel_on_drop.disarm();
            response.unwrap_or_else(|_| {
                Self::status(import_id, generation, proto::ProfileResultStatus::Cancelled)
            })
        })
    }

    fn page(
        &mut self,
        key: ImportKey,
        generation: u64,
        page_index: u32,
        cx: &App,
    ) -> proto::ProfileResultPage {
        if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
            return Self::status(key.1, generation, proto::ProfileResultStatus::Unsupported);
        }
        let Some(import) = self.imports.get(&key) else {
            return Self::status(key.1, generation, proto::ProfileResultStatus::Stale);
        };
        if import.generation != generation || import.next_page != page_index {
            return Self::status(key.1, generation, proto::ProfileResultStatus::Stale);
        }
        if let Some(run_id) = &import.task_run_id
            && !self
                .task_runs
                .get(&(ProfileTaskActor::Peer(key.0), run_id.clone()))
                .is_some_and(|run| self.task_run_current(run, cx))
        {
            self.imports.remove(&key);
            return Self::status(key.1, generation, proto::ProfileResultStatus::Stale);
        }
        if TrustedWorktrees::has_restricted_worktrees(&self.worktree_store, cx)
            || !import.paths.iter().all(|path| self.permitted(path, cx))
        {
            self.imports.remove(&key);
            return Self::status(key.1, generation, proto::ProfileResultStatus::Restricted);
        }
        let page = self.imports.get_mut(&key).and_then(|import| {
            let page = import.pages.pop_front()?;
            import.next_page += 1;
            Some(page)
        });
        match page {
            Some(page) => {
                if page.complete {
                    self.imports.remove(&key);
                }
                page
            }
            None => Self::status(key.1, generation, proto::ProfileResultStatus::Stale),
        }
    }

    fn proxied_page_paths(page: &proto::ProfileResultPage) -> Option<HashSet<ProjectPath>> {
        page.frames
            .iter()
            .filter_map(|frame| frame.source.as_ref())
            .map(|source| ProjectPath::from_proto(source.clone()))
            .collect()
    }

    fn accept_proxied_page(
        &mut self,
        key: ImportKey,
        remote_id: u64,
        expected_page: u32,
        mut page: proto::ProfileResultPage,
        cx: &App,
    ) -> proto::ProfileResultPage {
        let guest_id = key.1;
        if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
            self.proxied_imports.remove(&key);
            return Self::status(guest_id, 0, proto::ProfileResultStatus::Unsupported);
        }
        let Some(import) = self.proxied_imports.get(&key) else {
            return Self::status(guest_id, 0, proto::ProfileResultStatus::Cancelled);
        };
        if import.remote_id != remote_id || import.next_page != expected_page {
            return Self::status(
                guest_id,
                import.generation,
                proto::ProfileResultStatus::Stale,
            );
        }
        if !self.transport_connected {
            self.proxied_imports.remove(&key);
            return Self::status(guest_id, 0, proto::ProfileResultStatus::Stale);
        }
        if TrustedWorktrees::has_restricted_worktrees(&self.worktree_store, cx)
            || !import.paths.iter().all(|path| self.permitted(path, cx))
        {
            self.proxied_imports.remove(&key);
            return Self::status(guest_id, 0, proto::ProfileResultStatus::Restricted);
        }
        if page.status != proto::ProfileResultStatus::Ready as i32 {
            self.proxied_imports.remove(&key);
            let status = match page.status {
                status if status == proto::ProfileResultStatus::Unsupported as i32 => {
                    proto::ProfileResultStatus::Unsupported
                }
                status if status == proto::ProfileResultStatus::Restricted as i32 => {
                    proto::ProfileResultStatus::Restricted
                }
                status if status == proto::ProfileResultStatus::Cancelled as i32 => {
                    proto::ProfileResultStatus::Cancelled
                }
                status if status == proto::ProfileResultStatus::Stale as i32 => {
                    proto::ProfileResultStatus::Stale
                }
                status if status == proto::ProfileResultStatus::Mismatch as i32 => {
                    proto::ProfileResultStatus::Mismatch
                }
                _ => proto::ProfileResultStatus::Error,
            };
            return Self::status(guest_id, page.generation, status);
        }
        let Some(source_paths) = Self::proxied_page_paths(&page) else {
            self.proxied_imports.remove(&key);
            return Self::status(guest_id, 0, proto::ProfileResultStatus::Error);
        };
        if !source_paths.iter().all(|path| self.permitted(path, cx)) {
            self.proxied_imports.remove(&key);
            return Self::status(guest_id, 0, proto::ProfileResultStatus::Restricted);
        }
        let valid = page.protocol_version == PROFILE_PROTOCOL_VERSION
            && page.import_id == remote_id
            && page.page_index == expected_page
            && page.generation != 0
            && (import.generation == 0 || import.generation == page.generation)
            && page.encoded_len() <= crate::profile_results_wire::MAX_PROFILE_PAGE_BYTES
            && import
                .total_bytes
                .checked_add(page.encoded_len())
                .is_some_and(|bytes| {
                    bytes <= crate::profile_results_wire::MAX_PROFILE_RESULT_BYTES
                })
            && expected_page < 128;
        if !valid {
            self.proxied_imports.remove(&key);
            return Self::status(guest_id, 0, proto::ProfileResultStatus::Error);
        }
        if let Some(import) = self.proxied_imports.get_mut(&key) {
            import.generation = page.generation;
            import.next_page += 1;
            import.total_bytes += page.encoded_len();
            import.paths.extend(source_paths);
            import.cancellation.set_generation(page.generation);
            if page.complete {
                import.cancellation.complete();
                self.proxied_imports.remove(&key);
            }
        }
        page.import_id = guest_id;
        page
    }

    async fn handle_proxy_import(
        this: Entity<Self>,
        peer: proto::PeerId,
        request: proto::ImportProfileResults,
        mut cx: AsyncApp,
    ) -> Result<proto::ProfileResultPage> {
        let guest_id = request.import_id;
        let prepared = this.update(&mut cx, |store, cx| {
            if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
                return Err(proto::ProfileResultStatus::Unsupported);
            }
            if !cfg!(target_os = "macos") {
                return Err(proto::ProfileResultStatus::Unsupported);
            }
            let paths = profile_import_paths(&request, ProfileHost::MacOs)?;
            if request.task_run_id.is_some() {
                return Err(proto::ProfileResultStatus::Unsupported);
            }
            if !store.transport_connected {
                return Err(proto::ProfileResultStatus::Stale);
            }
            let key = (peer, guest_id);
            if store.cancelled.remove(&key) {
                return Err(proto::ProfileResultStatus::Cancelled);
            }
            if TrustedWorktrees::has_restricted_worktrees(&store.worktree_store, cx)
                || !paths.iter().all(|path| store.permitted(path, cx))
            {
                return Err(proto::ProfileResultStatus::Restricted);
            }
            if !store.proxied_imports.contains_key(&key)
                && store.proxied_imports.len() >= MAX_HOST_PROFILE_IMPORTS
            {
                return Err(proto::ProfileResultStatus::Error);
            }
            let remote_id = NEXT_PROXY_IMPORT_ID
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
                .map_err(|_| proto::ProfileResultStatus::Error)?;
            let proxy = store
                .remote_proxy
                .as_ref()
                .ok_or(proto::ProfileResultStatus::Unsupported)?;
            let client = proxy.client.clone();
            let project_id = proxy.project_id;
            let cancellation = Arc::new(AtomicBool::new(false));
            let mut guard = RemoteImportCancellation {
                client: Some(client.clone()),
                executor: cx.background_executor().clone(),
                cancellation,
                project_id,
                import_id: remote_id,
                generation: 0,
                started: false,
                #[cfg(any(test, feature = "test-support"))]
                acknowledged_host_cancellations: Some(
                    store.acknowledged_host_cancellations.clone(),
                ),
            };
            guard.start();
            store.proxied_imports.insert(
                key,
                ProxiedImport {
                    remote_id,
                    generation: 0,
                    next_page: 0,
                    total_bytes: 0,
                    paths: paths.into_iter().collect(),
                    cancellation: guard,
                },
            );
            let mut forwarded = request;
            forwarded.project_id = project_id;
            forwarded.import_id = remote_id;
            Ok((client, forwarded, remote_id))
        });
        let (client, forwarded, remote_id) = match prepared {
            Ok(prepared) => prepared,
            Err(status) => return Ok(Self::status(guest_id, 0, status)),
        };
        let response = client.request(forwarded).await;
        #[cfg(any(test, feature = "test-support"))]
        if let Some((started, release)) =
            this.update(&mut cx, |store, _| store.import_processing_gate.take())
        {
            started
                .send(())
                .map_err(|_| anyhow::anyhow!("Profile proxy import start receiver was dropped"))?;
            release.await?;
        }
        Ok(this.update(&mut cx, |store, cx| match response {
            Ok(page) => store.accept_proxied_page((peer, guest_id), remote_id, 0, page, cx),
            Err(error) => {
                let key = (peer, guest_id);
                if store
                    .proxied_imports
                    .get(&key)
                    .is_some_and(|import| import.remote_id == remote_id)
                {
                    store.proxied_imports.remove(&key);
                }
                let status = match RemoteProfileError::from_rpc(error) {
                    RemoteProfileError::Unsupported => proto::ProfileResultStatus::Unsupported,
                    _ => proto::ProfileResultStatus::Stale,
                };
                Self::status(guest_id, 0, status)
            }
        }))
    }

    async fn handle_import(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::ImportProfileResults>,
        mut cx: AsyncApp,
    ) -> Result<proto::ProfileResultPage> {
        let peer = envelope.original_sender_id.unwrap_or(envelope.sender_id);
        if this.read_with(&cx, |store, _| store.remote_proxy.is_some()) {
            return Self::handle_proxy_import(this, peer, envelope.payload, cx).await;
        }
        Ok(this
            .update(&mut cx, |this, cx| this.import(peer, envelope.payload, cx))
            .await)
    }

    async fn handle_page(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::GetProfileResultPage>,
        mut cx: AsyncApp,
    ) -> Result<proto::ProfileResultPage> {
        let peer = envelope.original_sender_id.unwrap_or(envelope.sender_id);
        let request = envelope.payload;
        if request.protocol_version != PROFILE_PROTOCOL_VERSION {
            return Ok(Self::status(
                request.import_id,
                request.generation,
                proto::ProfileResultStatus::Mismatch,
            ));
        }
        #[cfg(any(test, feature = "test-support"))]
        if let Some((started, release)) =
            this.update(&mut cx, |this, _| this.page_processing_gate.take())
        {
            started
                .send(())
                .map_err(|_| anyhow::anyhow!("Profile page start receiver was dropped"))?;
            release.await?;
        }
        #[cfg(all(feature = "profile-page-loss-fixture", target_os = "macos"))]
        if let Some(directory) = std::env::var_os("ZED_TEST_PROFILE_PAGE_GATE_DIR") {
            let directory = PathBuf::from(directory);
            let armed = directory.join("armed");
            if armed.exists() {
                std::fs::rename(&armed, directory.join("started"))?;
                while !directory.join("release").exists() {
                    cx.background_executor()
                        .timer(Duration::from_millis(20))
                        .await;
                }
            }
        }
        if this.read_with(&cx, |store, _| store.remote_proxy.is_some()) {
            return Self::handle_proxy_page(this, peer, request, cx).await;
        }
        Ok(this.update(&mut cx, |this, cx| {
            this.page(
                (peer, request.import_id),
                request.generation,
                request.page_index,
                cx,
            )
        }))
    }

    async fn handle_proxy_page(
        this: Entity<Self>,
        peer: proto::PeerId,
        request: proto::GetProfileResultPage,
        mut cx: AsyncApp,
    ) -> Result<proto::ProfileResultPage> {
        let key = (peer, request.import_id);
        let prepared = this.update(&mut cx, |store, cx| {
            if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
                return Err(proto::ProfileResultStatus::Unsupported);
            }
            let Some(import) = store.proxied_imports.get(&key) else {
                return Err(proto::ProfileResultStatus::Stale);
            };
            let remote_id = import.remote_id;
            if import.generation != request.generation || import.next_page != request.page_index {
                return Err(proto::ProfileResultStatus::Stale);
            }
            if !store.transport_connected {
                store.proxied_imports.remove(&key);
                return Err(proto::ProfileResultStatus::Stale);
            }
            if TrustedWorktrees::has_restricted_worktrees(&store.worktree_store, cx)
                || !import.paths.iter().all(|path| store.permitted(path, cx))
            {
                store.proxied_imports.remove(&key);
                return Err(proto::ProfileResultStatus::Restricted);
            }
            let proxy = store
                .remote_proxy
                .as_ref()
                .ok_or(proto::ProfileResultStatus::Unsupported)?;
            Ok((
                proxy.client.clone(),
                proxy.project_id,
                remote_id,
                store.invalidation.receiver(),
            ))
        });
        let (client, project_id, remote_id, mut invalidation) = match prepared {
            Ok(prepared) => prepared,
            Err(status) => return Ok(Self::status(request.import_id, request.generation, status)),
        };
        let response = futures::select_biased! {
            _ = invalidation.changed().fuse() => None,
            response = client.request(proto::GetProfileResultPage {
                project_id,
                import_id: remote_id,
                generation: request.generation,
                protocol_version: request.protocol_version,
                page_index: request.page_index,
            }).fuse() => Some(response),
        };
        Ok(this.update(&mut cx, |store, cx| match response {
            Some(Ok(page)) => {
                store.accept_proxied_page(key, remote_id, request.page_index, page, cx)
            }
            Some(Err(error)) => {
                if store
                    .proxied_imports
                    .get(&key)
                    .is_some_and(|import| import.remote_id == remote_id)
                {
                    store.proxied_imports.remove(&key);
                }
                let status = match RemoteProfileError::from_rpc(error) {
                    RemoteProfileError::Unsupported => proto::ProfileResultStatus::Unsupported,
                    _ => proto::ProfileResultStatus::Stale,
                };
                Self::status(request.import_id, request.generation, status)
            }
            None => Self::status(
                request.import_id,
                request.generation,
                proto::ProfileResultStatus::Stale,
            ),
        }))
    }

    async fn handle_cancel(
        this: Entity<Self>,
        envelope: TypedEnvelope<proto::CancelProfileImport>,
        mut cx: AsyncApp,
    ) -> Result<proto::Ack> {
        let peer = envelope.original_sender_id.unwrap_or(envelope.sender_id);
        let request = envelope.payload;
        this.update(&mut cx, |this, _| {
            let key = (peer, request.import_id);
            if this.remote_proxy.is_some() {
                if this.proxied_imports.get(&key).is_some_and(|import| {
                    request.generation == 0 || request.generation == import.generation
                }) {
                    this.proxied_imports.remove(&key);
                } else if request.generation == 0 && !this.proxied_imports.contains_key(&key) {
                    anyhow::ensure!(
                        this.cancelled.len() < 64 || this.cancelled.contains(&key),
                        "Too many pending profile cancellations"
                    );
                    this.cancelled.insert(key);
                }
                return anyhow::Ok(());
            }
            if this.imports.get(&key).is_some_and(|import| {
                request.generation == 0 || request.generation == import.generation
            }) {
                this.imports.remove(&key);
            } else if request.generation == 0 && !this.imports.contains_key(&key) {
                anyhow::ensure!(
                    this.cancelled.len() < 64 || this.cancelled.contains(&key),
                    "Too many pending profile cancellations"
                );
                this.cancelled.insert(key);
            }
            anyhow::Ok(())
        })?;
        Ok(proto::Ack {})
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RemoteProfileError {
    Unsupported,
    Restricted,
    Disconnected,
    Cancelled,
    Stale,
    Invalid,
}

impl std::fmt::Display for RemoteProfileError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Unsupported => {
                "The authoritative host does not support native macOS profile import"
            }
            Self::Restricted => {
                "Profile import requires consent and trusted, visible project files"
            }
            Self::Disconnected => "The profile host is disconnected or could not respond",
            Self::Cancelled => "Profile import was cancelled",
            Self::Stale => "The profile session or permissions changed; import again",
            Self::Invalid => "The host returned an invalid or unsupported profile result",
        })
    }
}

impl std::error::Error for RemoteProfileError {}

impl RemoteProfileError {
    fn from_assembly(cancellation: &AtomicBool) -> Self {
        if cancellation.load(Ordering::Relaxed) {
            Self::Cancelled
        } else {
            Self::Invalid
        }
    }

    fn from_rpc(error: anyhow::Error) -> Self {
        if error
            .downcast_ref::<proto::RpcError>()
            .is_some_and(|error| {
                error.raw_message().contains("was not handled")
                    || error
                        .raw_message()
                        .starts_with("no handler registered for ")
            })
        {
            Self::Unsupported
        } else {
            Self::Disconnected
        }
    }

    fn check_status(page: &proto::ProfileResultPage) -> Result<()> {
        use proto::ProfileResultStatus;
        match page.status {
            status if status == ProfileResultStatus::Ready as i32 => Ok(()),
            status if status == ProfileResultStatus::Unsupported as i32 => {
                Err(Self::Unsupported.into())
            }
            status if status == ProfileResultStatus::Restricted as i32 => {
                Err(Self::Restricted.into())
            }
            status if status == ProfileResultStatus::Cancelled as i32 => {
                Err(Self::Cancelled.into())
            }
            status if status == ProfileResultStatus::Stale as i32 => Err(Self::Stale.into()),
            _ => Err(Self::Invalid.into()),
        }
    }
}

struct PendingRemoteProfileTaskBinding {
    binding: ProfileTaskBinding,
    acceptance: futures::channel::oneshot::Sender<()>,
}

pub struct ProfileTaskBinding {
    pub helper_path: String,
    pub socket_path: String,
    pub token: String,
}

fn check_task_binding_response(
    response: proto::ProfileTaskBindingResponse,
    run_id: &str,
    complete: bool,
) -> Result<proto::ProfileTaskBindingResponse> {
    if response.protocol_version != PROFILE_PROTOCOL_VERSION || response.task_run_id != run_id {
        return Err(RemoteProfileError::Invalid.into());
    }
    use proto::ProfileResultStatus;
    match response.status {
        status if status == ProfileResultStatus::Ready as i32 && response.complete == complete => {
            Ok(response)
        }
        status if status == ProfileResultStatus::Unsupported as i32 => {
            Err(RemoteProfileError::Unsupported.into())
        }
        status if status == ProfileResultStatus::Restricted as i32 => {
            Err(RemoteProfileError::Restricted.into())
        }
        status if status == ProfileResultStatus::Cancelled as i32 => {
            Err(RemoteProfileError::Cancelled.into())
        }
        status if status == ProfileResultStatus::Stale as i32 => {
            Err(RemoteProfileError::Stale.into())
        }
        _ => Err(RemoteProfileError::Invalid.into()),
    }
}

struct RemoteImportCancellation {
    client: Option<AnyProtoClient>,
    executor: gpui::BackgroundExecutor,
    cancellation: Arc<AtomicBool>,
    project_id: u64,
    import_id: u64,
    generation: u64,
    started: bool,
    #[cfg(any(test, feature = "test-support"))]
    acknowledged_host_cancellations: Option<Arc<Mutex<HashSet<(u64, u64)>>>>,
}

impl RemoteImportCancellation {
    fn start(&mut self) {
        self.started = true;
    }

    fn set_generation(&mut self, generation: u64) {
        self.generation = generation;
    }

    fn complete(&mut self) {
        self.client = None;
    }
}

impl Drop for RemoteImportCancellation {
    fn drop(&mut self) {
        let Some(client) = self.client.take() else {
            return;
        };
        self.cancellation.store(true, Ordering::Relaxed);
        if !self.started {
            return;
        }
        let request = proto::CancelProfileImport {
            project_id: self.project_id,
            import_id: self.import_id,
            generation: self.generation,
        };
        #[cfg(any(test, feature = "test-support"))]
        let acknowledged_host_cancellations = self.acknowledged_host_cancellations.take();
        #[cfg(any(test, feature = "test-support"))]
        let import_id = self.import_id;
        #[cfg(any(test, feature = "test-support"))]
        let generation = self.generation;
        self.executor
            .spawn(async move {
                match client.request(request).await {
                    Ok(_) => {
                        #[cfg(any(test, feature = "test-support"))]
                        if let Some(acknowledged_host_cancellations) =
                            acknowledged_host_cancellations
                        {
                            acknowledged_host_cancellations
                                .lock()
                                .insert((import_id, generation));
                        }
                    }
                    Err(_) => {
                        log::debug!("Native profile cancellation could not reach the host");
                    }
                }
            })
            .detach();
    }
}

impl crate::Project {
    fn remote_profile_task_client(&self, cx: &App) -> Result<(AnyProtoClient, u64)> {
        if self.is_disconnected(cx) {
            return Err(RemoteProfileError::Disconnected.into());
        }
        if TrustedWorktrees::has_restricted_worktrees(&self.worktree_store(), cx) {
            return Err(RemoteProfileError::Restricted.into());
        }
        if let Some(remote) = self.remote_client() {
            Ok((
                remote.read(cx).proto_client(),
                proto::REMOTE_SERVER_PROJECT_ID,
            ))
        } else if self.is_via_collab() {
            Ok((
                self.client().into(),
                self.remote_id().ok_or(RemoteProfileError::Disconnected)?,
            ))
        } else {
            Err(RemoteProfileError::Unsupported.into())
        }
    }

    #[cfg(target_os = "macos")]
    pub fn prepare_local_cargo_build_report(
        &self,
        run_id: String,
        report_limit: usize,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<BuildTaskReportBinding>>> {
        self.prepare_local_cargo_build_report_context(run_id, report_limit, None, cx)
    }

    pub fn prepare_native_capture_plan(
        &mut self,
        receipt: NativeCargoWorkloadReceipt,
        declaration: task::TaskNativeProfileContinuation,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<PreparedNativeCapturePlan>>> {
        anyhow::ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
        {
            anyhow::ensure!(
                self.is_local() && !self.is_shared(),
                "Native capture requires an unshared local macOS Project"
            );
            declaration.validate()?;
            let store = self.cargo_workspace_store.clone();
            Ok(cx.spawn(async move |_, cx| {
                let prepared = store.update(cx, |store, cx| {
                    store.prepare_native_capture_plan(receipt.workload, declaration, cx)
                })?;
                let authority = prepared.await?;
                Ok(PreparedNativeCapturePlan {
                    resolved: authority.resolved.clone(),
                    authority,
                })
            }))
        }
        #[cfg(not(all(target_os = "macos", feature = "cargo-workspace")))]
        {
            let (_receipt, _declaration, _cx) = (receipt, declaration, cx);
            Err(anyhow::anyhow!(
                "Native capture requires the enabled macOS host"
            ))
        }
    }

    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    fn review_native_capture_outputs(
        &self,
        artifact_directory: &std::path::Path,
        group: &task::TaskArtifactGroup,
        cx: &App,
    ) -> Result<(
        Arc<dyn fs::Fs>,
        [ProjectPath; 2],
        Vec<(PathBuf, PathBuf, u64, u64)>,
    )> {
        anyhow::ensure!(
            self.is_local()
                && !self.is_shared()
                && !TrustedWorktrees::has_restricted_worktrees(&self.worktree_store(), cx),
            "Native capture Project route or trust changed"
        );
        anyhow::ensure!(
            artifact_directory.is_absolute(),
            "Native capture artifact directory must be absolute"
        );
        let filesystem = self
            .worktree_store()
            .read(cx)
            .fs()
            .context("Native capture filesystem is unavailable")?
            .clone();
        let mut paths = Vec::new();
        let mut roots = Vec::new();
        for role in ["profile", "symbols"] {
            let artifact = group
                .files
                .get(role)
                .context("Native capture role is unavailable")?;
            let absolute = artifact_directory.join(&artifact.path);
            let path = self
                .find_project_path(&absolute, cx)
                .context("Native capture output is outside its Project")?;
            let trees = self.worktree_store();
            let tree = trees
                .read(cx)
                .worktree_for_id(path.worktree_id, cx)
                .context("Native capture output worktree disappeared")?;
            let tree = tree.read(cx);
            let local = tree
                .as_local()
                .context("Native capture output must be local")?;
            let root = local.abs_path().to_path_buf();
            let snapshot = tree.snapshot();
            let root_inode = snapshot
                .root_entry()
                .context("Native capture root is unavailable")?
                .inode;
            let parent = absolute
                .parent()
                .context("Native capture output parent is unavailable")?
                .to_path_buf();
            let relative = parent.strip_prefix(&root).map_err(|_| {
                anyhow::anyhow!("Native capture output parent escaped its worktree")
            })?;
            let relative = util::rel_path::RelPath::new(relative, util::paths::PathStyle::local())?;
            let entry = snapshot
                .entry_for_path(relative.as_ref())
                .context("Native capture output parent must already be indexed")?;
            anyhow::ensure!(
                entry.is_dir() && entry.is_ignored && !entry.is_private && !entry.is_external,
                "Native capture output parent is not a permitted ignored directory"
            );
            roots.push((root, parent, root_inode, entry.inode));
            paths.push(path);
        }
        let paths: [ProjectPath; 2] = paths
            .try_into()
            .map_err(|_| anyhow::anyhow!("Native capture pair changed"))?;
        anyhow::ensure!(
            paths[0] != paths[1],
            "Native capture pair paths are duplicated"
        );
        Ok((filesystem, paths, roots))
    }

    pub fn admit_native_capture_task(
        &mut self,
        plan: PreparedNativeCapturePlan,
        run_id: uuid::Uuid,
        launch: task::SpawnInTerminal,
        group: task::TaskArtifactGroup,
        artifact_directory: Option<PathBuf>,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<PreparedNativeCaptureTask>>> {
        anyhow::ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
        {
            anyhow::ensure!(!run_id.is_nil(), "Native capture run identity is invalid");
            let bound = plan
                .resolved
                .bind_artifact_invocation(&run_id.to_string())?;
            anyhow::ensure!(
                bound.resolved_artifact_group() == Some(&group)
                    && bound.id == launch.id
                    && bound.resolved.command == launch.command
                    && bound.resolved.args == launch.args
                    && bound.resolved.env == launch.env
                    && bound.resolved.cwd == launch.cwd
                    && bound.resolved.literal_command == launch.literal_command
                    && bound.resolved.literal_prefix_args == launch.literal_prefix_args,
                "Native capture invocation no longer matches its prepared plan"
            );
            let expected_artifact_directory = bound
                .artifact_directory()
                .or(bound.resolved.cwd.as_deref())
                .context("Native capture prepared artifact directory is unavailable")?;
            let artifact_directory = artifact_directory
                .or_else(|| launch.cwd.clone())
                .context("Native capture disclosed artifact directory is unavailable")?;
            anyhow::ensure!(
                artifact_directory.as_path() == expected_artifact_directory,
                "Native capture artifact directory no longer matches its prepared plan"
            );
            let reviewed = self.review_native_capture_outputs(&artifact_directory, &group, cx)?;
            let store = self.cargo_workspace_store.clone();
            let pair_store = self
                .profile_results_store
                .as_ref()
                .context("Native pair owner is unavailable")?
                .clone();
            let executor = cx.background_executor().clone();
            Ok(cx.spawn(async move |this, cx| {
                let checked = store.update(cx, |store, cx| {
                    store.revalidate_native_capture_plan(plan.authority, cx)
                })?;
                let authority = checked.await?;
                let (filesystem, paths, roots) = reviewed;
                let absolute_paths = group
                    .files
                    .values()
                    .map(|artifact| artifact_directory.join(&artifact.path))
                    .collect::<Vec<_>>();
                cx.background_spawn(check_native_capture_outputs(
                    filesystem.clone(),
                    roots.clone(),
                    absolute_paths.clone(),
                    executor.clone(),
                ))
                .await?;
                let reviewed = this.read_with(cx, |project, cx| {
                    project.review_native_capture_outputs(&artifact_directory, &group, cx)
                })??;
                anyhow::ensure!(
                    reviewed.1 == paths && reviewed.2 == roots,
                    "Native capture output route changed"
                );
                let checked = store.update(cx, |store, cx| {
                    store.revalidate_native_capture_plan(authority, cx)
                })?;
                let authority = checked.await?;
                cx.background_spawn(check_native_capture_outputs(
                    filesystem,
                    roots.clone(),
                    absolute_paths,
                    executor,
                ))
                .await?;
                let reviewed = this.read_with(cx, |project, cx| {
                    project.review_native_capture_outputs(&artifact_directory, &group, cx)
                })??;
                anyhow::ensure!(
                    reviewed.1 == paths && reviewed.2 == roots,
                    "Native capture output route changed before helper preparation"
                );
                // The helper cannot register until preparation returns. Start its bounded
                // registration window after the expensive source/probe checks have finished.
                let mut authority = authority;
                authority.resolved = bound;
                let expected_invocation = authority.expected_capture_invocation(&launch)?;
                let invocation_envelope =
                    expected_invocation
                        .encode_helper_environment()
                        .map_err(|_| {
                            anyhow::anyhow!(
                                "Native capture invocation exceeds its serialized budget"
                            )
                        })?;
                let native_terminal_required = authority.has_observed_runtime();
                let helper_image = if native_terminal_required {
                    let image_preparation = this.update(cx, |project, cx| {
                        project.prepare_local_native_task_helper_image(cx)
                    })??;
                    Some(image_preparation.await?)
                } else {
                    None
                };
                let binding = pair_store
                    .update(cx, |store, cx| {
                        store.prepare_checked_local_native_pair(
                            run_id.to_string(),
                            paths.clone(),
                            expected_invocation,
                            cx,
                        )
                    })
                    .await?;
                let reviewed = this.read_with(cx, |project, cx| {
                    project.review_native_capture_outputs(&artifact_directory, &group, cx)
                })??;
                anyhow::ensure!(
                    reviewed.1 == paths && reviewed.2 == roots,
                    "Native capture output route changed after helper preparation"
                );

                let mut wrapped = launch;
                let program = wrapped
                    .command
                    .take()
                    .context("Native capture collector disappeared")?;
                let mut arguments = vec![
                    "--internal-task-observer".into(),
                    "--socket".into(),
                    binding.binding.socket_path.clone(),
                    "--checked-native-profile-capture".into(),
                    "--".into(),
                    program,
                ];
                arguments.append(&mut wrapped.args);
                wrapped.command = Some(binding.binding.helper_path.clone());
                wrapped.args = arguments;
                wrapped.literal_prefix_args = wrapped.args.len();
                wrapped.preserve_command_label = true;
                wrapped.env.insert(
                    util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV.into(),
                    binding.binding.token.clone(),
                );
                wrapped.env.insert(
                    util::macos_task_completion::TASK_NATIVE_INVOCATION_ENV.into(),
                    invocation_envelope,
                );
                let authority = Arc::new(Mutex::new(Some(authority)));
                pair_store.update(cx, |store, _| {
                    let run = store
                        .task_runs
                        .get_mut(&binding.key)
                        .filter(|run| {
                            run.generation == binding.generation
                                && !run.cancellation.load(Ordering::Relaxed)
                        })
                        .context("Native capture admission owner expired")?;
                    let native = run
                        .native_mut()
                        .context("Native capture pair owner changed")?;
                    anyhow::ensure!(
                        native.capture_admission.is_none(),
                        "Native capture authority was already installed"
                    );
                    native.capture_admission = Some((authority.clone(), binding.helper.clone()));
                    Ok::<_, anyhow::Error>(())
                })?;
                let terminal_admission = if let Some(helper_image) = helper_image {
                    let admission = NativeTerminalTaskAdmission::from_issued(
                        pair_store.downgrade(),
                        binding.key.clone(),
                        binding.generation,
                        binding.cancellation.clone(),
                        helper_image,
                        NativeTerminalTaskAuthority::Capture {
                            authority: authority.clone(),
                            invocation: binding.checked_invocation.clone(),
                        },
                        &mut wrapped,
                    )?;
                    pair_store.read_with(cx, |store, cx| admission.check_current(store, cx))?;
                    Some(admission)
                } else {
                    None
                };
                Ok(PreparedNativeCaptureTask {
                    launch: wrapped,
                    native_terminal_required,
                    terminal_admission,
                    authority,
                    binding,
                })
            }))
        }
        #[cfg(not(all(target_os = "macos", feature = "cargo-workspace")))]
        {
            let (_plan, _run, _launch, _group, _artifact_directory, _cx) =
                (plan, run_id, launch, group, artifact_directory, cx);
            Err(anyhow::anyhow!(
                "Native capture requires the enabled macOS host"
            ))
        }
    }

    pub fn finish_native_capture_task(
        &mut self,
        prepared: PreparedNativeCaptureTask,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<()>>> {
        anyhow::ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
        {
            let pair_store = self
                .profile_results_store
                .as_ref()
                .context("Native pair owner is unavailable")?
                .clone();
            let awaited = pair_store.update(cx, |store, cx| {
                store.await_local_native_pair(&prepared.binding, cx)
            })?;
            let store = self.cargo_workspace_store.clone();
            Ok(cx.spawn(async move |this, cx| {
                awaited.await?;
                let checked = store.update(cx, |store, cx| {
                    store.revalidate_native_capture_plan(
                        prepared
                            .authority
                            .lock()
                            .take()
                            .context("Native capture authority was retired")?,
                        cx,
                    )
                })?;
                let authority = checked.await?;
                this.read_with(cx, |project, cx| {
                    anyhow::ensure!(
                        project.is_local()
                            && !project.is_shared()
                            && !TrustedWorktrees::has_restricted_worktrees(
                                &project.worktree_store(),
                                cx
                            ),
                        "Native capture completion route or trust changed"
                    );
                    Ok::<_, anyhow::Error>(())
                })??;
                pair_store.update(cx, |store, cx| {
                    let run = store
                        .task_runs
                        .get(&prepared.binding.key)
                        .filter(|run| {
                            run.generation == prepared.binding.generation
                                && Arc::ptr_eq(&run.cancellation, &prepared.binding.cancellation)
                                && !run.failed
                                && store.task_run_current(run, cx)
                        })
                        .context("Native capture completed owner changed")?;
                    let native = run.native().context("Native capture pair owner changed")?;
                    anyhow::ensure!(
                        native.checked_invocation.as_ref().is_some_and(|current| {
                            Arc::ptr_eq(current, &prepared.binding.checked_invocation)
                        }) && native
                            .capture_admission
                            .as_ref()
                            .is_some_and(|(current, _)| {
                                Arc::ptr_eq(current, &prepared.authority)
                            }),
                        "Native capture completed invocation or admission owner changed"
                    );
                    let native = store
                        .task_runs
                        .get_mut(&prepared.binding.key)
                        .and_then(ProfileTaskRun::native_mut)
                        .context("Native capture completed pair disappeared")?;
                    // Sealed artifacts remain available to import; their completed
                    // issuer and full child map must not consume future capture quota.
                    native.capture_admission = None;
                    native.checked_invocation = None;
                    Ok::<_, anyhow::Error>(())
                })?;
                drop(prepared.launch);
                let mut binding = prepared.binding;
                binding.armed = false;
                drop(authority);
                Ok(())
            }))
        }
        #[cfg(not(all(target_os = "macos", feature = "cargo-workspace")))]
        {
            let (_prepared, _cx) = (prepared, cx);
            Err(anyhow::anyhow!(
                "Native capture requires the enabled macOS host"
            ))
        }
    }

    pub fn prepare_selected_native_cargo_build_task(
        &mut self,
        run_id: uuid::Uuid,
        launch: task::SpawnInTerminal,
        report_limit: usize,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<PreparedNativeCargoBuildTask>>> {
        anyhow::ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
        {
            self.prepare_selected_native_cargo_build_task_inner(
                run_id,
                launch,
                report_limit,
                None,
                cx,
            )
        }
        #[cfg(not(all(target_os = "macos", feature = "cargo-workspace")))]
        {
            let (_run_id, _launch, _report_limit, _context) = (run_id, launch, report_limit, cx);
            Err(anyhow::anyhow!(
                "Selected native Cargo Build requires an enabled local macOS host"
            ))
        }
    }

    pub fn prepare_selected_native_cargo_runtime_task(
        &mut self,
        run_id: uuid::Uuid,
        launch: task::SpawnInTerminal,
        report_limit: usize,
        continuation: task::TaskNativeProfileContinuation,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<PreparedNativeCargoBuildTask>>> {
        anyhow::ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        continuation.validate()?;
        #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
        {
            self.prepare_selected_native_cargo_build_task_inner(
                run_id,
                launch,
                report_limit,
                Some(continuation),
                cx,
            )
        }
        #[cfg(not(all(target_os = "macos", feature = "cargo-workspace")))]
        {
            let (_run_id, _launch, _report_limit, _continuation, _context) =
                (run_id, launch, report_limit, continuation, cx);
            Err(anyhow::anyhow!(
                "Selected native Cargo runtime requires an enabled local macOS host"
            ))
        }
    }

    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    fn prepare_selected_native_cargo_build_task_inner(
        &mut self,
        run_id: uuid::Uuid,
        launch: task::SpawnInTerminal,
        report_limit: usize,
        continuation: Option<task::TaskNativeProfileContinuation>,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<PreparedNativeCargoBuildTask>>> {
        anyhow::ensure!(
            self.is_local() && !self.is_shared(),
            "Selected native Build requires an unshared local Project"
        );
        anyhow::ensure!(
            (task::MIN_TASK_STDOUT_REPORT_BYTES..=task::MAX_TASK_STDOUT_REPORT_BYTES)
                .contains(&report_limit),
            "Native Build declared report limit is invalid"
        );
        self.profile_results_store
            .as_ref()
            .context("Native Build report owner is unavailable")?
            .update(cx, |store, _| store.ensure_local_task_helper())?;
        let store = self.cargo_workspace_store.clone();
        Ok(cx.spawn(async move |this, cx| {
            let project = this
                .upgrade()
                .ok_or_else(|| anyhow::anyhow!("Selected native Build Project disappeared"))?;
            let runtime_preparation = continuation.is_some();
            let (intent, binding) = match continuation {
                Some(continuation) => {
                    let prepared = store.update(cx, |store, cx| {
                        store.prepare_native_cargo_runtime_intent_for_launch(
                            project,
                            run_id,
                            &launch,
                            &continuation,
                            cx,
                        )
                    })?;
                    let intent = prepared.await?;
                    let checked = store.update(cx, |store, cx| {
                        store.prepare_native_cargo_runtime_binding(
                            intent,
                            &launch,
                            report_limit,
                            cx,
                        )
                    })?;
                    let (intent, binding) = checked.await?;
                    (
                        SelectedNativeCargoIntent::Runtime(Box::new(intent)),
                        binding,
                    )
                }
                None => {
                    let prepared = store.update(cx, |store, cx| {
                        store.prepare_native_cargo_build_intent_for_launch(
                            project, run_id, &launch, cx,
                        )
                    })?;
                    let intent = prepared.await?;
                    let checked = store.update(cx, |store, cx| {
                        store.prepare_native_cargo_build_binding(intent, &launch, report_limit, cx)
                    })?;
                    let (intent, binding) = checked.await?;
                    (SelectedNativeCargoIntent::Build(Box::new(intent)), binding)
                }
            };
            anyhow::ensure!(
                binding.requires_cargo_runtime_observation() == runtime_preparation,
                "Selected Cargo preparation report mode changed"
            );
            anyhow::ensure!(
                binding.helper_mode() == ProfileTaskHelperMode::LocalApplication,
                "Selected native Build needs its prepared local helper image"
            );
            let mut wrapped = launch;
            wrapped.command = Some(
                binding
                    .helper_path()
                    .to_str()
                    .ok_or_else(|| {
                        anyhow::anyhow!("Selected native Build helper path is not UTF-8")
                    })?
                    .to_owned(),
            );
            wrapped.args = vec![
                "--internal-task-observer".into(),
                "--socket".into(),
                binding
                    .socket_path()
                    .to_str()
                    .ok_or_else(|| {
                        anyhow::anyhow!("Selected native Build socket path is not UTF-8")
                    })?
                    .to_owned(),
                if binding.requires_cargo_runtime_observation() {
                    "--checked-cargo-runtime-stdout".into()
                } else {
                    "--capture-build-stdout".into()
                },
                "--".into(),
                intent
                    .program()
                    .to_str()
                    .ok_or_else(|| {
                        anyhow::anyhow!("Selected native Build program path is not UTF-8")
                    })?
                    .to_owned(),
            ];
            wrapped.args.extend_from_slice(intent.arguments());
            wrapped.literal_prefix_args = wrapped.args.len();
            wrapped.preserve_command_label = true;
            wrapped.cwd = Some(intent.working_directory().to_path_buf());
            let invocation_environment = binding
                .expected_invocation
                .as_ref()
                .context("Selected native Build invocation is unavailable")?
                .encode_helper_environment()
                .map_err(|_| {
                    anyhow::anyhow!(
                        "Selected native Build invocation exceeds its private environment bound"
                    )
                })?;
            wrapped.env.insert(
                util::macos_task_completion::TASK_BUILD_INVOCATION_ENV.into(),
                invocation_environment,
            );
            wrapped.env.insert(
                util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV.into(),
                binding
                    .token()
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect(),
            );
            this.read_with(cx, |project, _| {
                anyhow::ensure!(
                    project.is_local() && !project.is_shared(),
                    "Selected native Build Project route changed"
                );
                Ok::<_, anyhow::Error>(())
            })??;
            let terminal_admission = if runtime_preparation {
                let admission = binding
                    .runtime_admission
                    .as_ref()
                    .context("Native terminal runtime authority disappeared")?
                    .clone();
                let pair_store = this
                    .read_with(cx, |project, _| project.profile_results_store.clone())?
                    .context("Native terminal producing owner disappeared")?;
                let terminal = NativeTerminalTaskAdmission::from_issued(
                    pair_store.downgrade(),
                    binding.key.clone(),
                    binding.generation,
                    binding.cancellation.clone(),
                    admission.image.duplicate(),
                    NativeTerminalTaskAuthority::CargoRuntime(admission),
                    &mut wrapped,
                )?;
                pair_store.read_with(cx, |store, cx| terminal.check_current(store, cx))?;
                Some(terminal)
            } else {
                None
            };
            Ok(PreparedNativeCargoBuildTask {
                launch: wrapped,
                terminal_admission,
                intent,
                binding,
            })
        }))
    }

    pub fn finish_selected_native_cargo_build_task(
        &mut self,
        prepared: PreparedNativeCargoBuildTask,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<NativeCargoWorkloadReceipt>>> {
        anyhow::ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
        {
            let runtime = matches!(&prepared.intent, SelectedNativeCargoIntent::Runtime(_));
            anyhow::ensure!(
                prepared.binding.requires_cargo_runtime_observation() == runtime
                    && prepared.binding.runtime_admission.is_some() == runtime,
                "Selected native Cargo completion admission mode changed"
            );
            let report = self.await_local_cargo_build_report(prepared.binding, cx)?;
            let store = self.cargo_workspace_store.clone();
            Ok(cx.spawn(async move |_, cx| {
                let report = report.await?;
                let finished = store.update(cx, |store, cx| match prepared.intent {
                    SelectedNativeCargoIntent::Build(intent) => {
                        store.finish_native_cargo_build(*intent, report, cx)
                    }
                    SelectedNativeCargoIntent::Runtime(intent) => {
                        store.finish_native_cargo_runtime(*intent, report, cx)
                    }
                })?;
                Ok(NativeCargoWorkloadReceipt {
                    _private: (),
                    workload: finished.await?,
                })
            }))
        }
        #[cfg(not(all(target_os = "macos", feature = "cargo-workspace")))]
        {
            let (_prepared, _context) = (prepared, cx);
            Err(anyhow::anyhow!(
                "Selected native Cargo Build is unavailable on this host"
            ))
        }
    }

    pub fn revalidate_selected_native_cargo_workload(
        &mut self,
        receipt: NativeCargoWorkloadReceipt,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<NativeCargoWorkloadReceipt>>> {
        anyhow::ensure!(
            cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
        {
            let store = self.cargo_workspace_store.clone();
            Ok(cx.spawn(async move |_, cx| {
                let checked = store.update(cx, |store, cx| {
                    store.revalidate_native_cargo_workload(receipt.workload, cx)
                })?;
                Ok(NativeCargoWorkloadReceipt {
                    _private: (),
                    workload: checked.await?,
                })
            }))
        }
        #[cfg(not(all(target_os = "macos", feature = "cargo-workspace")))]
        {
            let (_receipt, _context) = (receipt, cx);
            Err(anyhow::anyhow!(
                "Selected native Cargo workload is unavailable on this host"
            ))
        }
    }

    #[cfg(target_os = "macos")]
    fn native_task_helper_image_current(
        &self,
        image: &LocalNativeTaskHelperImage,
        project: gpui::EntityId,
        cx: &App,
    ) -> bool {
        self.is_local()
            && !self.is_shared()
            && image.project == project
            && self.profile_results_store.as_ref().is_some_and(|store| {
                store.entity_id() == image.store && store.read(cx).helper_image_current(image, cx)
            })
    }

    #[cfg(target_os = "macos")]
    pub fn prepare_local_native_task_helper_image(
        &self,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<LocalNativeTaskHelperImage>>> {
        anyhow::ensure!(
            self.is_local() && !self.is_shared(),
            "Native helper image requires an unshared local Project"
        );
        let store = self
            .profile_results_store
            .as_ref()
            .context("Native helper image owner is unavailable")?;
        #[cfg(feature = "cargo-workspace")]
        store.update(cx, |store, _| store.ensure_local_task_helper())?;
        let project = cx.entity_id();
        let prepared = store.update(cx, |store, cx| {
            store.prepare_local_native_task_helper_image(project, cx)
        });
        Ok(cx.spawn(async move |this, cx| {
            let image = prepared.await?;
            this.read_with(cx, |project_state, cx| {
                anyhow::ensure!(
                    project_state.native_task_helper_image_current(&image, project, cx),
                    "Native helper image Project changed during preparation"
                );
                Ok(image)
            })?
        }))
    }

    #[cfg(target_os = "macos")]
    pub fn revalidate_local_native_task_helper_image(
        &self,
        image: &LocalNativeTaskHelperImage,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<()>>> {
        let project = cx.entity_id();
        anyhow::ensure!(
            matches!(image.purpose, NativeTaskHelperImagePurpose::Application),
            "Dedicated runtime images require their signing-policy renewal"
        );
        anyhow::ensure!(
            self.native_task_helper_image_current(image, project, cx),
            "Native helper image owner became stale"
        );
        let filesystem = self
            .worktree_store
            .read(cx)
            .fs()
            .context("Native helper image filesystem is unavailable")?;
        let image = image.duplicate();
        let background = cx.background_spawn({
            let identity = image.identity.clone();
            async move {
                crate::profile_results::revalidate_task_helper_identity(
                    filesystem.as_ref(),
                    &identity,
                    &AtomicBool::new(false),
                )
                .await
                .map_err(|_| anyhow::anyhow!("Native helper image identity changed"))
            }
        });
        Ok(cx.spawn(async move |this, cx| {
            background.await?;
            this.read_with(cx, |project_state, cx| {
                anyhow::ensure!(
                    project_state.native_task_helper_image_current(&image, project, cx),
                    "Native helper image owner changed during revalidation"
                );
                Ok(())
            })?
        }))
    }

    #[cfg(target_os = "macos")]
    pub fn prepare_local_native_cargo_runtime_helper_image(
        &self,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<LocalNativeTaskHelperImage>>> {
        anyhow::ensure!(
            self.is_local() && !self.is_shared(),
            "Native runtime helper requires an unshared local Project"
        );
        let store = self
            .profile_results_store
            .as_ref()
            .context("Native runtime helper owner is unavailable")?;
        #[cfg(feature = "cargo-workspace")]
        store.update(cx, |store, _| store.ensure_local_task_helper())?;
        let (helper, epoch, filesystem) = store.read_with(cx, |store, cx| {
            anyhow::ensure!(
                store.remote_proxy.is_none()
                    && store.transport_connected
                    && !TrustedWorktrees::has_restricted_worktrees(&store.worktree_store, cx),
                "Native runtime helper owner is unavailable"
            );
            let helper = store
                .task_helper
                .clone()
                .context("Native runtime helper is unconfigured")?;
            anyhow::ensure!(
                helper.mode == ProfileTaskHelperMode::LocalApplication,
                "Native runtime helper cannot use a remote route"
            );
            Ok::<_, anyhow::Error>((
                helper,
                store.helper_admission_epoch.clone(),
                store
                    .worktree_store
                    .read(cx)
                    .fs()
                    .context("Native runtime helper filesystem is unavailable")?,
            ))
        })?;
        let project = cx.entity_id();
        let store_id = store.entity_id();
        let process =
            std::env::current_exe().context("Native runtime process image is unavailable")?;
        let executor = cx.background_executor().clone();
        let deadline = executor.timer(Duration::from_secs(30));
        let prepared = cx.background_spawn(async move {
            let process = filesystem.canonicalize(&process).await?;
            let requested = native_runtime_observer_path(&process)?;
            let configured = filesystem.canonicalize(&helper.path).await?;
            anyhow::ensure!(
                configured == process,
                "Native runtime helper cannot use a configured image override"
            );
            let cancel = AtomicBool::new(false);
            let app_identity = crate::profile_results::prepare_task_helper_identity(
                filesystem.as_ref(),
                process,
                &cancel,
            )
            .await?;
            let identity = crate::profile_results::prepare_task_helper_identity(
                filesystem.as_ref(),
                requested.clone(),
                &cancel,
            )
            .await?;
            anyhow::ensure!(
                identity.canonical_path() == requested,
                "Native runtime observer cannot be a bundle-external alias"
            );
            let (architectures, signer) = native_runtime_image_policy::validate(
                filesystem.as_ref(),
                &app_identity,
                &identity,
            )
            .await?;
            let system_closure = native_runtime_image_policy::prepare_system_closure(
                filesystem.as_ref(),
                &identity,
                &architectures,
                &executor,
            )
            .await?;
            crate::profile_results::revalidate_task_helper_identity(
                filesystem.as_ref(),
                &app_identity,
                &cancel,
            )
            .await?;
            Ok::<_, anyhow::Error>(LocalNativeTaskHelperImage {
                project,
                store: store_id,
                epoch,
                owner_helper: helper,
                helper: ProfileTaskHelper {
                    path: requested,
                    mode: ProfileTaskHelperMode::LocalApplication,
                },
                purpose: NativeTaskHelperImagePurpose::CargoRuntime(Arc::new(
                    NativeRuntimeHelperPolicy {
                        app_identity,
                        architectures,
                        signer,
                        system_closure,
                    },
                )),
                identity,
            })
        });
        Ok(cx.spawn(async move |this, cx| {
            let image = smol::future::race(prepared, async move {
                deadline.await;
                anyhow::bail!("Native runtime signing preparation exceeded 30 seconds")
            })
            .await?;
            this.read_with(cx, |state, cx| {
                anyhow::ensure!(
                    state.native_task_helper_image_current(&image, project, cx),
                    "Native runtime helper owner changed during preparation"
                );
                Ok(image)
            })?
        }))
    }

    #[cfg(target_os = "macos")]
    pub fn revalidate_local_native_cargo_runtime_helper_image(
        &self,
        image: &LocalNativeTaskHelperImage,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<()>>> {
        let project = cx.entity_id();
        anyhow::ensure!(
            self.native_task_helper_image_current(image, project, cx),
            "Native runtime helper owner became stale"
        );
        let NativeTaskHelperImagePurpose::CargoRuntime(policy) = &image.purpose else {
            anyhow::bail!("Native runtime helper requires its dedicated issued image");
        };
        let filesystem = self
            .worktree_store
            .read(cx)
            .fs()
            .context("Native runtime helper filesystem is unavailable")?;
        let image = image.duplicate();
        let policy = policy.clone();
        let executor = cx.background_executor().clone();
        let deadline = executor.timer(Duration::from_secs(30));
        let renewed = cx.background_spawn({
            let image = image.duplicate();
            async move {
                let actual_process = filesystem.canonicalize(&std::env::current_exe()?).await?;
                anyhow::ensure!(
                    actual_process == policy.app_identity.canonical_path()
                        && native_runtime_observer_path(&actual_process)?
                            == image.identity.canonical_path(),
                    "Native runtime installed process route changed"
                );
                let (architectures, signer) = native_runtime_image_policy::validate(
                    filesystem.as_ref(),
                    &policy.app_identity,
                    &image.identity,
                )
                .await?;
                anyhow::ensure!(
                    architectures == policy.architectures && signer == policy.signer,
                    "Native runtime architecture or signer policy changed"
                );
                native_runtime_image_policy::revalidate_system_closure(
                    filesystem.as_ref(),
                    &policy.system_closure,
                    &executor,
                )
                .await?;
                crate::profile_results::revalidate_task_helper_identity(
                    filesystem.as_ref(),
                    &image.identity,
                    &AtomicBool::new(false),
                )
                .await?;
                crate::profile_results::revalidate_task_helper_identity(
                    filesystem.as_ref(),
                    &policy.app_identity,
                    &AtomicBool::new(false),
                )
                .await?;
                Ok(())
            }
        });
        Ok(cx.spawn(async move |this, cx| {
            smol::future::race(renewed, async move {
                deadline.await;
                anyhow::bail!("Native runtime signing renewal exceeded 30 seconds")
            })
            .await?;
            this.read_with(cx, |state, cx| {
                anyhow::ensure!(
                    state.native_task_helper_image_current(&image, project, cx),
                    "Native runtime helper owner changed during renewal"
                );
                Ok(())
            })?
        }))
    }

    #[cfg(target_os = "macos")]
    pub fn prepare_checked_local_cargo_build_report(
        &self,
        run_id: String,
        report_limit: usize,
        expected: util::macos_task_completion::TaskBuildInvocation,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<BuildTaskReportBinding>>> {
        self.prepare_local_cargo_build_report_context(run_id, report_limit, Some(expected), cx)
    }

    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    pub(crate) fn prepare_checked_local_cargo_runtime_report(
        &self,
        run_id: String,
        report_limit: usize,
        expected: util::macos_task_completion::TaskBuildInvocation,
        image: &LocalNativeTaskHelperImage,
        lease: crate::cargo_workspace_store::NativeCargoRuntimeAdmissionLease,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<BuildTaskReportBinding>>> {
        anyhow::ensure!(
            lease.project()?.entity_id() == cx.entity_id()
                && lease.matches_helper_image(image)
                && lease.run_id().to_string() == run_id
                && lease.expected_invocation() == &expected,
            "Cargo runtime report does not match its original issued admission"
        );
        lease.validate_outer_loader_environment()?;
        if !matches!(image.purpose, NativeTaskHelperImagePurpose::CargoRuntime(_)) {
            validate_native_runtime_loader_environment(expected.environment())?;
            anyhow::bail!("Cargo runtime report requires an issued dedicated loader policy");
        }
        let renewed = self.revalidate_local_native_cargo_runtime_helper_image(image, cx)?;
        let image = image.duplicate();
        let project = cx.entity_id();
        Ok(cx.spawn(async move |this, cx| {
            renewed.await?;
            let prepared = this.update(cx, |project_state, cx| {
                anyhow::ensure!(
                    project_state.native_task_helper_image_current(&image, project, cx),
                    "Native helper image changed before report preparation"
                );
                let admission = Arc::new(BuildReportRuntimeAdmission {
                    image: image.duplicate(),
                    lease,
                });
                project_state.prepare_local_cargo_build_report_mode(
                    run_id,
                    report_limit,
                    Some(expected),
                    BuildReportMode::CheckedCargoRuntime,
                    Some(admission),
                    cx,
                )
            })??;
            let mut binding = prepared.await?;
            anyhow::ensure!(
                binding.store == image.store
                    && binding.helper == image.helper
                    && binding.identity.canonical_path() == image.identity.canonical_path()
                    && binding.identity.fingerprint() == image.identity.fingerprint(),
                "Native helper binding did not retain the prepared image"
            );
            let renewed = this.update(cx, |project_state, cx| {
                project_state.revalidate_local_native_cargo_runtime_helper_image(&image, cx)
            })??;
            renewed.await?;
            this.read_with(cx, |project_state, cx| {
                anyhow::ensure!(
                    project_state.native_task_helper_image_current(&image, project, cx),
                    "Native helper image owner changed after report preparation"
                );
                binding.helper_image = Some(image);
                Ok(binding)
            })?
        }))
    }

    #[cfg(target_os = "macos")]
    fn prepare_local_cargo_build_report_context(
        &self,
        run_id: String,
        report_limit: usize,
        expected: Option<util::macos_task_completion::TaskBuildInvocation>,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<BuildTaskReportBinding>>> {
        let mode = if expected.is_some() {
            BuildReportMode::CheckedBuild
        } else {
            BuildReportMode::Stdout
        };
        self.prepare_local_cargo_build_report_mode(run_id, report_limit, expected, mode, None, cx)
    }

    #[cfg(target_os = "macos")]
    fn prepare_local_cargo_build_report_mode(
        &self,
        run_id: String,
        report_limit: usize,
        expected: Option<util::macos_task_completion::TaskBuildInvocation>,
        mode: BuildReportMode,
        runtime_admission: Option<Arc<BuildReportRuntimeAdmission>>,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<BuildTaskReportBinding>>> {
        anyhow::ensure!(
            mode == BuildReportMode::Stdout
                || cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>(),
            "Native Rust profiling requires local experimental opt-in"
        );

        anyhow::ensure!(
            self.is_local() && !self.is_shared(),
            "Build report observation requires an unshared local Project"
        );
        let store = self
            .profile_results_store
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Build report owner is unavailable"))?;
        let prepared = store.update(cx, |store, cx| match (mode, expected) {
            (BuildReportMode::CheckedCargoRuntime, Some(expected)) => store
                .prepare_local_build_report_context(
                    run_id,
                    report_limit,
                    Some(expected),
                    BuildReportMode::CheckedCargoRuntime,
                    runtime_admission,
                    cx,
                ),
            (BuildReportMode::CheckedBuild, Some(expected)) => {
                store.prepare_checked_local_build_report(run_id, report_limit, expected, cx)
            }
            (BuildReportMode::Stdout, None) => {
                store.prepare_local_build_report(run_id, report_limit, cx)
            }
            _ => Task::ready(Err(anyhow::anyhow!(
                "Build report invocation mode is inconsistent"
            ))),
        });
        Ok(cx.spawn(async move |this, cx| {
            let binding = prepared.await?;
            this.read_with(cx, |project, cx| {
                anyhow::ensure!(
                    project.is_local() && !project.is_shared(),
                    "Build report Project route changed during preparation"
                );
                let store = project
                    .profile_results_store
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Build report owner disappeared"))?;
                anyhow::ensure!(
                    store.entity_id() == binding.store
                        && store.read(cx).build_binding_current(&binding, cx),
                    "Build report owner changed during preparation"
                );
                Ok(binding)
            })?
        }))
    }

    #[cfg(target_os = "macos")]
    pub fn await_local_cargo_build_report(
        &self,
        binding: BuildTaskReportBinding,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<ObservedBuildTaskReport>>> {
        anyhow::ensure!(
            self.is_local() && !self.is_shared(),
            "Build report observation requires an unshared local Project"
        );
        anyhow::ensure!(
            binding.helper_image.as_ref().is_none_or(
                |image| self.native_task_helper_image_current(image, cx.entity_id(), cx)
            ),
            "Native helper image Project changed before report wait"
        );
        let store = self
            .profile_results_store
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Build report owner is unavailable"))?;
        let observed = store.update(cx, |store, cx| store.await_build_report(binding, cx));
        Ok(cx.spawn(async move |this, cx| {
            let report = observed.await?;
            this.read_with(cx, |project, cx| {
                anyhow::ensure!(
                    project.is_local() && !project.is_shared(),
                    "Build report Project route changed while waiting"
                );
                let store = project
                    .profile_results_store
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Build report owner disappeared"))?;
                anyhow::ensure!(
                    store.entity_id() == report.binding.store
                        && store.read(cx).observed_build_report_current(&report, cx),
                    "Build report owner changed while waiting"
                );
                Ok(report)
            })?
        }))
    }

    #[cfg(target_os = "macos")]
    pub fn revalidate_local_cargo_build_report(
        &self,
        report: &ObservedBuildTaskReport,
        cx: &mut Context<Self>,
    ) -> Result<Task<Result<()>>> {
        anyhow::ensure!(
            self.is_local() && !self.is_shared(),
            "Build report observation requires an unshared local Project"
        );
        let store = self
            .profile_results_store
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Build report owner is unavailable"))?;
        anyhow::ensure!(
            report.binding.helper_image.as_ref().is_none_or(
                |image| self.native_task_helper_image_current(image, cx.entity_id(), cx)
            ),
            "Native helper image Project changed before report renewal"
        );
        anyhow::ensure!(
            store.read(cx).observed_build_report_current(report, cx),
            "Build report observation became stale"
        );
        let filesystem = self
            .worktree_store
            .read(cx)
            .fs()
            .ok_or_else(|| anyhow::anyhow!("Build report filesystem is unavailable"))?;
        let identity = report.binding.identity.clone();
        let cancellation = report.binding.cancellation.clone();
        let key = report.binding.key.clone();
        let generation = report.binding.generation;
        let bytes = report.bytes.clone();
        let reservation = report.binding.reservation.clone();
        let helper = report.binding.helper.clone();
        let runtime_admission = report.binding.runtime_admission.clone();
        let store = store.downgrade();
        let background = cx.background_spawn(async move {
            crate::profile_results::revalidate_task_helper_identity(
                filesystem.as_ref(),
                &identity,
                &cancellation,
            )
            .await
            .map_err(|_| anyhow::anyhow!("Build Task helper identity changed"))
        });
        Ok(cx.spawn(async move |this, cx| {
            background.await?;
            this.read_with(cx, |project, cx| {
                anyhow::ensure!(project.is_local() && !project.is_shared(), "Build report Project route changed");
                store.read_with(cx, |store, cx| {
                    anyhow::ensure!(store.transport_connected && store.remote_proxy.is_none()
                        && !TrustedWorktrees::has_restricted_worktrees(&store.worktree_store, cx)
                        && store.build_report_helper_current(&helper, runtime_admission.as_deref(), cx)
                        && store.task_runs.get(&key).is_some_and(|run| run.generation == generation
                            && !run.failed && !run.cancellation.load(Ordering::Relaxed)
                            && matches!(&run.payload, ProfileTaskPayload::BuildReport(build)
                                if Arc::ptr_eq(&build.reservation, &reservation)
                                    && build.report.as_ref().is_some_and(|report| Arc::ptr_eq(report, &bytes)))),
                        "Build report run changed during helper revalidation");
                    anyhow::Ok(())
                })?
            })?
        }))
    }

    pub fn prepare_remote_profile_task(
        &self,
        run_id: String,
        paths: [ProjectPath; 2],
        cx: &App,
    ) -> Result<Task<Result<ProfileTaskBinding>>> {
        if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
            return Err(RemoteProfileError::Unsupported.into());
        }
        if !valid_profile_task_run_id(&run_id) || paths[0] == paths[1] {
            return Err(RemoteProfileError::Invalid.into());
        }
        let (client, project_id) = self.remote_profile_task_client(cx)?;
        let [profile_path, symbols_path] = paths;
        let request = proto::PrepareProfileTaskBinding {
            project_id,
            protocol_version: PROFILE_PROTOCOL_VERSION,
            task_run_id: run_id.clone(),
            profile_path: Some(profile_path.to_proto()),
            symbols_path: Some(symbols_path.to_proto()),
            artifact_format: "samply-0.13.1".into(),
        };
        let store = self
            .profile_results_store
            .as_ref()
            .ok_or(RemoteProfileError::Unsupported)?;
        let (invalidation, mut acceptance_invalidation) = {
            let store = store.read(cx);
            (store.invalidation.receiver(), store.invalidation.receiver())
        };
        #[cfg(any(test, feature = "test-support"))]
        let store = store.downgrade();
        let pending = observe_native_remote_result(
            move |cx| {
                let (response_sender, response_receiver) = futures::channel::oneshot::channel();
                cx.background_executor()
                    .spawn(async move {
                        let response = client
                            .request(request)
                            .await
                            .map_err(RemoteProfileError::from_rpc);
                        let issued = response.as_ref().is_ok_and(|response| {
                            response.protocol_version == PROFILE_PROTOCOL_VERSION
                                && response.task_run_id == run_id
                                && response.status == proto::ProfileResultStatus::Ready as i32
                                && !response.complete
                        });
                        let (acceptance, accepted) = futures::channel::oneshot::channel();
                        let result = (|| {
                            let response = check_task_binding_response(response?, &run_id, false)?;
                            let binding = ProfileTaskBinding {
                                helper_path: response.helper_path.ok_or(RemoteProfileError::Invalid)?,
                                socket_path: response.socket_path.ok_or(RemoteProfileError::Invalid)?,
                                token: response.token.ok_or(RemoteProfileError::Invalid)?,
                            };
                            if binding.helper_path.is_empty()
                                || binding.socket_path.is_empty()
                                || binding.token.len() != 64
                                || !binding.token.bytes().all(|byte| byte.is_ascii_hexdigit())
                            {
                                return Err(RemoteProfileError::Invalid.into());
                            }
                            Ok::<_, anyhow::Error>(PendingRemoteProfileTaskBinding {
                                binding,
                                acceptance,
                            })
                        })();
                        if response_sender.send(result).is_err() {
                            log::trace!("Native preparation receiver was dropped");
                        }
                        // A dropped client future cannot retract a request the host already received.
                        // Drain its response before cancelling only the binding this request issued.
                        if issued && accepted.await.is_err() {
                            if let Err(error) = client
                                .request(proto::CancelProfileTaskBinding {
                                    project_id,
                                    task_run_id: run_id,
                                })
                                .await
                            {
                                let error = RemoteProfileError::from_rpc(error);
                                log::warn!("Revoked native Task preparation could not cancel its host binding: {error}");
                            }
                        }
                    })
                    .detach();
                cx.background_executor().spawn(async move {
                    response_receiver
                        .await
                        .map_err(|_| RemoteProfileError::Disconnected)?
                })
            },
            Arc::new(AtomicBool::new(false)),
            invalidation,
            cx,
        );
        Ok(cx.spawn(async move |cx| {
            let pending = pending.await?;
            #[cfg(any(test, feature = "test-support"))]
            if let Some((started, release)) =
                store.update(cx, |store, _| store.task_preparation_acceptance_gate.take())?
            {
                started.send(()).map_err(|_| {
                    anyhow::anyhow!("Native preparation acceptance start receiver was dropped")
                })?;
                release.await.map_err(|_| {
                    anyhow::anyhow!("Native preparation acceptance release was dropped")
                })?;
            }
            cx.update(|cx| {
                if acceptance_invalidation.changed().now_or_never().is_some()
                    || !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>()
                {
                    return Err(RemoteProfileError::Unsupported.into());
                }
                pending
                    .acceptance
                    .send(())
                    .map_err(|_| RemoteProfileError::Disconnected)?;
                Ok(pending.binding)
            })
        }))
    }

    pub fn get_remote_profile_task(&self, run_id: String, cx: &App) -> Result<Task<Result<()>>> {
        if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
            return Err(RemoteProfileError::Unsupported.into());
        }
        if !valid_profile_task_run_id(&run_id) {
            return Err(RemoteProfileError::Invalid.into());
        }
        let (client, project_id) = self.remote_profile_task_client(cx)?;
        let receive = move |cx: &App| {
            cx.background_executor().spawn(async move {
                let response = client
                    .request(proto::GetProfileTaskBinding {
                        project_id,
                        protocol_version: PROFILE_PROTOCOL_VERSION,
                        task_run_id: run_id.clone(),
                    })
                    .await
                    .map_err(RemoteProfileError::from_rpc)?;
                check_task_binding_response(response, &run_id, true)?;
                Ok(())
            })
        };
        let invalidation = self
            .profile_results_store
            .as_ref()
            .ok_or(RemoteProfileError::Unsupported)?
            .read(cx)
            .invalidation
            .receiver();
        Ok(observe_native_remote_result(
            receive,
            Arc::new(AtomicBool::new(false)),
            invalidation,
            cx,
        ))
    }

    pub fn cancel_remote_profile_task(&self, run_id: String, cx: &App) -> Result<Task<Result<()>>> {
        if !valid_profile_task_run_id(&run_id) {
            return Err(RemoteProfileError::Invalid.into());
        }
        let (client, project_id) = self.remote_profile_task_client(cx)?;
        Ok(cx.background_executor().spawn(async move {
            client
                .request(proto::CancelProfileTaskBinding {
                    project_id,
                    task_run_id: run_id,
                })
                .await
                .map_err(RemoteProfileError::from_rpc)?;
            Ok(())
        }))
    }

    pub fn import_remote_profile(
        &self,
        paths: [ProjectPath; 2],
        consent_to_transfer: bool,
        cancellation: Arc<AtomicBool>,
        cx: &App,
    ) -> Result<Task<Result<crate::profile_results::ProfileResults>>> {
        self.import_remote_profile_internal(paths, None, consent_to_transfer, cancellation, cx)
    }

    pub fn import_remote_task_profile(
        &self,
        paths: [ProjectPath; 2],
        task_run_id: String,
        cancellation: Arc<AtomicBool>,
        cx: &App,
    ) -> Result<Task<Result<crate::profile_results::ProfileResults>>> {
        if !valid_profile_task_run_id(&task_run_id) {
            return Err(RemoteProfileError::Invalid.into());
        }
        self.import_remote_profile_internal(paths, Some(task_run_id), true, cancellation, cx)
    }

    fn import_remote_profile_internal(
        &self,
        paths: [ProjectPath; 2],
        task_run_id: Option<String>,
        consent_to_transfer: bool,
        cancellation: Arc<AtomicBool>,
        cx: &App,
    ) -> Result<Task<Result<crate::profile_results::ProfileResults>>> {
        use std::sync::atomic::AtomicU64;
        static NEXT_IMPORT_ID: AtomicU64 = AtomicU64::new(1);

        if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
            return Err(RemoteProfileError::Unsupported.into());
        }
        if self.is_disconnected(cx) {
            return Err(RemoteProfileError::Disconnected.into());
        }
        if !consent_to_transfer
            || TrustedWorktrees::has_restricted_worktrees(&self.worktree_store(), cx)
        {
            return Err(RemoteProfileError::Restricted.into());
        }
        let (client, project_id) = if let Some(remote) = self.remote_client() {
            (
                remote.read(cx).proto_client(),
                proto::REMOTE_SERVER_PROJECT_ID,
            )
        } else if self.is_via_collab() {
            (
                self.client().into(),
                self.remote_id().ok_or(RemoteProfileError::Disconnected)?,
            )
        } else {
            return Err(RemoteProfileError::Unsupported.into());
        };
        let worktrees = self
            .visible_worktrees(cx)
            .map(|worktree| {
                let worktree = worktree.read(cx);
                (worktree.id(), worktree.snapshot())
            })
            .collect::<Vec<_>>();
        let permitted = move |path: &ProjectPath| {
            worktrees.iter().any(|(id, snapshot)| {
                *id == path.worktree_id
                    && snapshot
                        .entry_for_path(path.path.as_ref())
                        .is_some_and(|entry| {
                            entry.is_file()
                                && !entry.is_private
                                && !entry.is_fifo
                                && !entry.is_external
                        })
            })
        };
        if !paths.iter().all(&permitted) {
            return Err(RemoteProfileError::Restricted.into());
        }
        let import_id = NEXT_IMPORT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| RemoteProfileError::Invalid)?;
        let [profile_path, symbols_path] = paths;
        let request = proto::ImportProfileResults {
            project_id,
            import_id,
            protocol_version: PROFILE_PROTOCOL_VERSION,
            profile_path: Some(profile_path.to_proto()),
            symbols_path: Some(symbols_path.to_proto()),
            consent_to_transfer,
            task_run_id,
            artifact_format: "samply-0.13.1".into(),
        };
        profile_import_paths(&request, ProfileHost::MacOs)
            .map_err(|_| RemoteProfileError::Invalid)?;
        let cancellation_for_receiving = cancellation.clone();
        let invalidation = self
            .profile_results_store
            .as_ref()
            .ok_or(RemoteProfileError::Unsupported)?
            .read(cx)
            .invalidation
            .receiver();
        Ok(observe_native_remote_result(
            move |cx| {
                receive_remote_profile(
                    request,
                    client,
                    cancellation_for_receiving,
                    permitted,
                    cx.background_executor(),
                )
            },
            cancellation,
            invalidation,
            cx,
        ))
    }
}

fn observe_native_remote_result<T: 'static>(
    receive: impl FnOnce(&App) -> Task<Result<T>> + 'static,
    cancellation: Arc<AtomicBool>,
    mut invalidation: watch::Receiver<()>,
    cx: &App,
) -> Task<Result<T>> {
    cx.spawn(async move |cx| {
        if invalidation.changed().now_or_never().is_some() {
            cancellation.store(true, Ordering::Relaxed);
            return Err(RemoteProfileError::Unsupported.into());
        }
        let (stop, stopped) = async_channel::bounded::<()>(1);
        let revoked = Arc::new(AtomicBool::new(false));
        let _subscription = cx.update(|cx| {
            if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
                cancellation.store(true, Ordering::Relaxed);
                return Err(anyhow::Error::from(RemoteProfileError::Unsupported));
            }
            Ok(cx.observe_global::<feature_flags::FeatureFlagStore>({
                let cancellation = cancellation.clone();
                let revoked = revoked.clone();
                move |cx| {
                    if !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>() {
                        revoked.store(true, Ordering::Relaxed);
                        cancellation.store(true, Ordering::Relaxed);
                        stop.close();
                    }
                }
            }))
        })?;
        let receiving = cx.update(|cx| receive(cx));
        let revoked_by_owner = async {
            drop(
                futures::future::select(
                    invalidation.changed().boxed_local(),
                    stopped.recv().boxed_local(),
                )
                .await,
            );
        };
        let result = {
            match futures::future::select(receiving.boxed_local(), revoked_by_owner.boxed_local())
                .await
            {
                futures::future::Either::Left((result, _)) => result,
                futures::future::Either::Right(_) => {
                    cancellation.store(true, Ordering::Relaxed);
                    Err(RemoteProfileError::Unsupported.into())
                }
            }
        };
        cx.update(|cx| {
            if revoked.load(Ordering::Relaxed)
                || invalidation.changed().now_or_never().is_some()
                || !cx.has_flag::<RustNativeProfilingExperimentalFeatureFlag>()
            {
                return Err(RemoteProfileError::Unsupported.into());
            }
            result
        })
    })
}

fn receive_remote_profile(
    request: proto::ImportProfileResults,
    client: AnyProtoClient,
    cancellation: Arc<AtomicBool>,
    permitted: impl Fn(&ProjectPath) -> bool + Send + 'static,
    executor: &gpui::BackgroundExecutor,
) -> Task<Result<crate::profile_results::ProfileResults>> {
    use crate::profile_results_wire::ProfilePageAssembler;
    let project_id = request.project_id;
    let import_id = request.import_id;
    let mut guard = RemoteImportCancellation {
        client: Some(client.clone()),
        executor: executor.clone(),
        cancellation: cancellation.clone(),
        project_id,
        import_id,
        generation: 0,
        started: false,
        #[cfg(any(test, feature = "test-support"))]
        acknowledged_host_cancellations: None,
    };
    executor.spawn(async move {
        if cancellation.load(Ordering::Relaxed) {
            return Err(RemoteProfileError::Cancelled.into());
        }
        guard.start();
        let mut page = client
            .request(request)
            .await
            .map_err(RemoteProfileError::from_rpc)?;
        if let Err(error) = RemoteProfileError::check_status(&page) {
            guard.complete();
            return Err(error);
        }
        let generation = page.generation;
        guard.set_generation(generation);
        let mut assembler = ProfilePageAssembler::new(import_id, generation)
            .map_err(|_| RemoteProfileError::Invalid)?;
        let mut next_page = 0u32;
        loop {
            if cancellation.load(Ordering::Relaxed) {
                return Err(RemoteProfileError::Cancelled.into());
            }
            RemoteProfileError::check_status(&page)?;
            let complete = page.complete;
            assembler
                .push(page, &cancellation, &permitted)
                .map_err(|_| RemoteProfileError::from_assembly(&cancellation))?;
            if complete {
                let result = assembler
                    .finish(&cancellation, permitted)
                    .map_err(|_| RemoteProfileError::from_assembly(&cancellation))?;
                guard.complete();
                return Ok(result);
            }
            next_page = next_page
                .checked_add(1)
                .ok_or(RemoteProfileError::Invalid)?;
            if next_page >= 128 {
                return Err(RemoteProfileError::Invalid.into());
            }
            page = client
                .request(proto::GetProfileResultPage {
                    project_id,
                    import_id,
                    generation,
                    protocol_version: PROFILE_PROTOCOL_VERSION,
                    page_index: next_page,
                })
                .await
                .map_err(RemoteProfileError::from_rpc)?;
        }
    })
}

#[cfg(test)]
mod client_tests {
    use super::*;
    use futures::future::BoxFuture;
    use parking_lot::Mutex;
    use proto::EnvelopedMessage as _;

    #[cfg(target_os = "macos")]
    #[test]
    fn native_pair_checked_reservation_bounds_and_owned_map_retention() {
        let budget = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let build = BuildReportReservation::reserve_for_invocation(
            budget.clone(),
            util::macos_task_completion::MAX_TASK_STDOUT_REPORT_BYTES,
            true,
        )
        .expect("checked Build");
        let expected = util::macos_task_completion::TaskBuildInvocation::new(
            "/bin/sh".into(),
            vec!["-c".into(), "exit 0".into()],
            std::env::current_dir().expect("actual cwd"),
            std::collections::BTreeMap::new(),
        )
        .expect("bounded complete invocation");
        let context = Arc::new(NativePairInvocation {
            expected,
            reservation: BuildReportReservation::reserve_native_invocation(budget.clone())
                .expect("checked capture fits shared budget"),
        });
        let retired = context.reservation.retirement.clone();
        assert_eq!(
            budget.load(Ordering::Relaxed),
            MAX_BUILD_REPORT_RESERVED_BYTES
        );
        assert!(BuildReportReservation::reserve_native_invocation(budget.clone()).is_err());
        let retained = context.clone();
        drop(context);
        assert_eq!(
            budget.load(Ordering::Relaxed),
            MAX_BUILD_REPORT_RESERVED_BYTES
        );
        assert!(!retired.is_closed());
        drop(retained);
        assert!(retired.is_closed());
        assert_eq!(budget.load(Ordering::Relaxed), build.reserved_bytes);
        let next = BuildReportReservation::reserve_native_invocation(budget.clone())
            .expect("truly dropped invocation permits next Capture");
        drop(next);
        drop(build);
        assert_eq!(budget.load(Ordering::Relaxed), 0);
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn native_pair_checked_mode_rejects_peer_and_remote_helper(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            crate::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        let filesystem = fs::FakeFs::new(cx.executor());
        let worktrees = cx.new(|_| WorktreeStore::local(true, filesystem, Default::default()));
        let store = cx.new(|cx| ProfileResultsStore::local(worktrees, cx));
        let expected = util::macos_task_completion::TaskBuildInvocation::new(
            "/bin/sh".into(),
            vec![],
            std::env::current_dir().expect("actual cwd"),
            std::collections::BTreeMap::new(),
        )
        .expect("bounded map");
        let (context, budget) = store.read_with(cx, |store, _| {
            let budget = store.build_report_budget.clone();
            (
                Arc::new(NativePairInvocation {
                    expected,
                    reservation: BuildReportReservation::reserve_native_invocation(budget.clone())
                        .expect("bounded capture reservation"),
                }),
                budget,
            )
        });
        for peer in [true, false] {
            let response = store
                .update(cx, |store, cx| {
                    store.prepare_task_for_actor_with_native_invocation(
                        if peer {
                            ProfileTaskActor::Peer(proto::PeerId::default())
                        } else {
                            ProfileTaskActor::Local(cx.entity_id())
                        },
                        if peer {
                            ProfileTaskHelperMode::LocalApplication
                        } else {
                            ProfileTaskHelperMode::RemoteServer
                        },
                        proto::PrepareProfileTaskBinding {
                            protocol_version: PROFILE_PROTOCOL_VERSION,
                            task_run_id: "checked-native-mode".into(),
                            ..Default::default()
                        },
                        Some(context.clone()),
                        cx,
                    )
                })
                .await;
            assert_eq!(
                response.status,
                proto::ProfileResultStatus::Unsupported as i32
            );
            assert!(response.complete);
            assert!(response.token.is_none());
            assert!(response.socket_path.is_none());
            assert!(store.read_with(cx, |store, _| store.task_runs.is_empty()));
        }
        drop(context);
        assert_eq!(budget.load(Ordering::Relaxed), 0);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_pair_checked_helper_process_entry() {
        let Ok(socket) = std::env::var("ZED_NATIVE_PAIR_TEST_SOCKET") else {
            return;
        };
        let command = std::env::var("ZED_NATIVE_PAIR_TEST_COMMAND").expect("controlled child");
        let mut child = std::process::Command::new("/bin/sh");
        child.args(["-c", command.as_str()]);
        let error = util::macos_task_completion::register_and_exec_checked_native_profile_capture(
            std::path::Path::new(&socket),
            &mut child,
        );
        assert_ne!(error.kind(), std::io::ErrorKind::TimedOut);
        std::process::exit(2);
    }

    #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
    #[gpui::test]
    async fn native_pair_checked_observer_retains_quota_and_requires_issued_admission(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            crate::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        cx.executor().allow_parking();
        let filesystem = Arc::new(fs::RealFs::new(None, cx.background_executor.clone()));
        let worktrees =
            cx.new(|_| WorktreeStore::local(true, filesystem.clone(), Default::default()));
        let store = cx.new(|cx| ProfileResultsStore::local(worktrees, cx));
        let helper = std::env::current_exe().expect("actual helper executable");
        let identity_cancellation = Arc::new(AtomicBool::new(false));
        let identity = crate::profile_results::prepare_task_helper_identity(
            filesystem.as_ref(),
            helper.clone(),
            &identity_cancellation,
        )
        .await
        .expect("actual helper identity");
        let root = tempfile::tempdir().expect("private process fixture");
        for scenario in [
            "unissued",
            "changed-environment",
            "changed-argv",
            "tampered-envelope",
            "cancelled-before-registration",
        ] {
            let marker = root.path().join(scenario);
            let command = format!("printf forbidden > '{}'", marker.display());
            let expected = util::macos_task_completion::TaskBuildInvocation::new(
                "/bin/sh".into(),
                vec!["-c".into(), command.clone()],
                std::env::current_dir().expect("actual cwd"),
                std::collections::BTreeMap::new(),
            )
            .expect("bounded complete invocation");
            let mut changed_environment = std::collections::BTreeMap::new();
            if scenario == "changed-environment" {
                changed_environment.insert("RUSTFLAGS".into(), "unadmitted".into());
            }
            let observed = util::macos_task_completion::TaskBuildInvocation::new(
                "/bin/sh".into(),
                vec![
                    "-c".into(),
                    if scenario == "changed-argv" {
                        "exit 0".into()
                    } else {
                        command.clone()
                    },
                ],
                std::env::current_dir().expect("actual cwd"),
                changed_environment,
            )
            .expect("bounded peer invocation");
            let envelope = if scenario == "tampered-envelope" {
                "{\"success\":true}".to_owned()
            } else {
                observed
                    .encode_helper_environment()
                    .expect("bounded private envelope")
            };
            let token = [0x57; 32];
            let listener = util::macos_task_completion::TaskCompletionListener::bind(token)
                .expect("actual listener");
            let socket = listener.path();
            let (stop, stopped) = async_channel::bounded(1);
            let (observation, retirement, budget, build) = store.update(cx, |store, cx| {
                let budget = store.build_report_budget.clone();
                let build = BuildReportReservation::reserve_for_invocation(
                    budget.clone(),
                    util::macos_task_completion::MAX_TASK_STDOUT_REPORT_BYTES,
                    true,
                )
                .expect("full checked Build quota");
                let context = Arc::new(NativePairInvocation {
                    expected,
                    reservation: BuildReportReservation::reserve_native_invocation(budget.clone())
                        .expect("checked capture quota"),
                });
                let retirement = context.reservation.retirement.clone();
                let paths = ["profile.json", "profile.syms.json"].map(|path| {
                    ProjectPath::from_proto(proto::ProjectPath {
                        worktree_id: 1,
                        path: path.into(),
                    })
                    .expect("portable pair path")
                });
                let observation = store.observe_task(
                    (
                        ProfileTaskActor::Local(cx.entity_id()),
                        format!("checked-{scenario}"),
                    ),
                    1,
                    paths,
                    listener,
                    stopped,
                    Some((
                        filesystem.clone(),
                        identity.clone(),
                        identity_cancellation.clone(),
                    )),
                    Some(context),
                    cx,
                );
                (observation, retirement, budget, build)
            });
            cx.run_until_parked();
            assert_eq!(
                budget.load(Ordering::Relaxed),
                MAX_BUILD_REPORT_RESERVED_BYTES
            );
            assert!(
                !retirement.is_closed(),
                "pending real listener still owns issued map quota"
            );
            assert!(BuildReportReservation::reserve_native_invocation(budget.clone()).is_err());
            let observation = if scenario == "cancelled-before-registration" {
                stop.close();
                observation.await;
                assert!(
                    retirement.is_closed(),
                    "cancelled observer actually retired before peer connects"
                );
                assert_eq!(budget.load(Ordering::Relaxed), build.reserved_bytes);
                None
            } else {
                Some(observation)
            };
            let mut process = smol::process::Command::new(&helper);
            process
                .args([
                    "--exact",
                    "profile_results_store::client_tests::native_pair_checked_helper_process_entry",
                    "--nocapture",
                ])
                .env("ZED_NATIVE_PAIR_TEST_SOCKET", &socket)
                .env(
                    "ZED_NATIVE_PAIR_TEST_COMMAND",
                    if scenario == "changed-argv" {
                        "exit 0"
                    } else {
                        command.as_str()
                    },
                )
                .env(
                    util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV,
                    "57".repeat(32),
                )
                .env(
                    util::macos_task_completion::TASK_NATIVE_INVOCATION_ENV,
                    envelope,
                )
                .kill_on_drop(true);
            let deadline = cx.background_executor.timer(Duration::from_secs(10));
            let output = smol::future::race(
                async move { process.output().await.expect("actual helper result") },
                async move {
                    deadline.await;
                    panic!("checked native denial must not await registration expiry");
                },
            )
            .await;
            assert_eq!(
                output.status.code(),
                Some(2),
                "real helper rejected: {scenario}"
            );
            assert!(
                !marker.exists(),
                "rejected helper launched collector: {scenario}"
            );
            if scenario == "tampered-envelope" {
                stop.close();
            }
            if let Some(observation) = observation {
                observation.await;
            }
            assert!(
                retirement.is_closed(),
                "actual observation dropped its complete map"
            );
            assert_eq!(budget.load(Ordering::Relaxed), build.reserved_bytes);
            assert!(store.read_with(cx, |store, _| store.task_runs.is_empty()));
            assert!(
                !identity_cancellation.load(Ordering::Relaxed),
                "peer denial does not forge local cancellation"
            );
            drop(stop);
            drop(build);
            assert_eq!(budget.load(Ordering::Relaxed), 0);
        }
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn native_pair_checked_future_drop_retires_unpolled_and_registered_observation(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.update(|cx| {
            crate::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        cx.executor().allow_parking();
        let filesystem = Arc::new(fs::RealFs::new(None, cx.background_executor.clone()));
        let helper = std::env::current_exe().expect("actual helper image");
        let cancellation = Arc::new(AtomicBool::new(false));
        let identity = crate::profile_results::prepare_task_helper_identity(
            filesystem.as_ref(),
            helper.clone(),
            &cancellation,
        )
        .await
        .expect("actual helper identity");
        let temporary = tempfile::tempdir().expect("private child marker fixture");
        for registered in [false, true] {
            let marker = temporary
                .path()
                .join(if registered { "registered" } else { "unpolled" });
            let command = format!("printf forbidden > '{}'", marker.display());
            let expected = util::macos_task_completion::TaskBuildInvocation::new(
                "/bin/sh".into(),
                vec!["-c".into(), command.clone()],
                std::env::current_dir().expect("actual cwd"),
                std::collections::BTreeMap::new(),
            )
            .expect("bounded complete map");
            let envelope = expected
                .encode_helper_environment()
                .expect("bounded transient envelope");
            let budget = Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let build = BuildReportReservation::reserve_for_invocation(
                budget.clone(),
                util::macos_task_completion::MAX_TASK_STDOUT_REPORT_BYTES,
                true,
            )
            .expect("full checked Build lease");
            let context = Arc::new(NativePairInvocation {
                expected,
                reservation: BuildReportReservation::reserve_native_invocation(budget.clone())
                    .expect("admitted capture context"),
            });
            let retired = context.reservation.retirement.clone();
            let listener = util::macos_task_completion::TaskCompletionListener::bind([0x63; 32])
                .expect("actual local listener");
            let socket = listener.path();
            let (stop, stopped) = async_channel::bounded(1);
            let (registration_requests, registration_requested) = async_channel::bounded(1);
            let mut observation = Box::pin(ProfileResultsStore::observe_native_pair_process(
                listener,
                stopped,
                Some((filesystem.clone(), identity.clone(), cancellation.clone())),
                Some(context.clone()),
                registration_requests,
            ));
            drop(context);
            assert_eq!(
                budget.load(Ordering::Relaxed),
                MAX_BUILD_REPORT_RESERVED_BYTES
            );
            assert!(
                !retired.is_closed(),
                "unpolled observation owns its complete map"
            );
            let mut child = None;
            let mut held_admission = None;
            if registered {
                let process = smol::process::Command::new(&helper)
                    .args(["--exact", "profile_results_store::client_tests::native_pair_checked_helper_process_entry", "--nocapture"])
                    .env("ZED_NATIVE_PAIR_TEST_SOCKET", &socket)
                    .env("ZED_NATIVE_PAIR_TEST_COMMAND", &command)
                    .env(util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV, "63".repeat(32))
                    .env(util::macos_task_completion::TASK_NATIVE_INVOCATION_ENV, envelope)
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .kill_on_drop(true)
                    .spawn().expect("actual helper process");
                child = Some(process);
                let deadline = cx.background_executor.timer(Duration::from_secs(10));
                let admitted = smol::future::race(
                    async {
                        match futures::future::select(
                            (&mut observation).boxed(),
                            registration_requested.recv().boxed(),
                        )
                        .await
                        {
                            futures::future::Either::Right((request, _)) => {
                                request.expect("checked image/frame reached foreground admission")
                            }
                            futures::future::Either::Left((result, _)) => {
                                panic!("observer ended before held admission: {result:?}")
                            }
                        }
                    },
                    async {
                        deadline.await;
                        panic!("actual checked helper failed to reach held admission");
                    },
                )
                .await;
                held_admission = Some(admitted);
                assert!(
                    child
                        .as_mut()
                        .expect("registered helper")
                        .try_status()
                        .expect("nonblocking status")
                        .is_none()
                );
                assert!(
                    !marker.exists(),
                    "no fabricated capture approval before future Drop"
                );
                assert!(
                    !retired.is_closed(),
                    "registered frame and observer still own map quota"
                );
                assert_eq!(
                    budget.load(Ordering::Relaxed),
                    MAX_BUILD_REPORT_RESERVED_BYTES
                );
            }
            assert!(BuildReportReservation::reserve_native_invocation(budget.clone()).is_err());
            // A handle or actor reference disappearing is not a quota signal;
            // only destruction of the actual observation future retires its frame.
            drop(observation);
            assert!(
                retired.is_closed(),
                "actual inner-future destruction retires invocation"
            );
            assert_eq!(budget.load(Ordering::Relaxed), build.reserved_bytes);
            let fresh = BuildReportReservation::reserve_native_invocation(budget.clone())
                .expect("fresh capture fits only after actual observer destruction");
            assert_eq!(
                budget.load(Ordering::Relaxed),
                MAX_BUILD_REPORT_RESERVED_BYTES
            );
            drop(fresh);
            drop(held_admission);
            assert!(!marker.exists());
            if let Some(mut child) = child {
                let deadline = cx.background_executor.timer(Duration::from_secs(10));
                let status = smol::future::race(
                    async {
                        child
                            .status()
                            .await
                            .expect("actual helper rejection status")
                    },
                    async {
                        deadline.await;
                        panic!("helper remained alive after observer destruction");
                    },
                )
                .await;
                assert_eq!(status.code(), Some(2));
                assert!(
                    !marker.exists(),
                    "destroyed observer never approves collector exec"
                );
            }
            drop(stop);
            drop(build);
            assert_eq!(budget.load(Ordering::Relaxed), 0);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_runtime_dedicated_image_loader_requires_exact_system_route() {
        for name in [
            "/usr/lib/other",
            "/usr/lib//dyld",
            "/usr/lib/./dyld",
            "/usr/lib/dyld/",
            "/usr/lib/dyld\n",
        ] {
            assert!(
                native_runtime_image_policy::architectures(&native_runtime_test_image(Some(name)))
                    .is_err()
            );
        }
        let mut duplicated = native_runtime_test_image(Some("/usr/lib/dyld"));
        let command = duplicated[32..].to_vec();
        duplicated[16..20].copy_from_slice(&2_u32.to_le_bytes());
        duplicated[20..24].copy_from_slice(&(2 * command.len() as u32).to_le_bytes());
        duplicated.extend_from_slice(&command);
        assert!(native_runtime_image_policy::architectures(&duplicated).is_err());
        native_runtime_image_policy::architectures(&native_runtime_test_image(Some(
            "/usr/lib/dyld",
        )))
        .expect("ordinary system loader header, not closure authority");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_runtime_dedicated_image_optional_dependency_cannot_qualify_as_strong() {
        fn with_dependency(command_kind: u32) -> Vec<u8> {
            let mut image = native_runtime_test_image(None);
            let name = b"/usr/lib/libSystem.B.dylib";
            let length = (24 + name.len() + 1 + 7) & !7;
            image[16..20].copy_from_slice(&1_u32.to_le_bytes());
            image[20..24].copy_from_slice(&(length as u32).to_le_bytes());
            for value in [command_kind, length as u32, 24, 0, 0, 0] {
                image.extend_from_slice(&value.to_le_bytes());
            }
            image.extend_from_slice(name);
            image.resize(32 + length, 0);
            image
        }
        for kind in [0x8000_0018, 0x20, 0x8000_0023] {
            assert!(native_runtime_image_policy::architectures(&with_dependency(kind)).is_err());
        }
        for kind in [0xc, 0x8000_001f] {
            native_runtime_image_policy::architectures(&with_dependency(kind))
                .expect("strong system dependency data, not transitive authority");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_runtime_dedicated_image_other_slice_cannot_hide_loader_refusal() {
        let current = native_runtime_test_image(None);
        let current_architecture =
            u32::from_le_bytes(current[4..8].try_into().expect("fixture architecture"));
        let current_subtype =
            u32::from_le_bytes(current[8..12].try_into().expect("fixture subtype"));
        let (other_architecture, other_subtype) = if current_architecture == 0x0100_000c {
            (0x0100_0007_u32, 3_u32)
        } else {
            (0x0100_000c_u32, 0_u32)
        };
        let mut other = native_runtime_test_image(Some("/usr/lib/other-loader"));
        other[4..8].copy_from_slice(&other_architecture.to_le_bytes());
        other[8..12].copy_from_slice(&other_subtype.to_le_bytes());
        let mut universal = Vec::new();
        for value in [
            0xcafe_babe_u32,
            2,
            current_architecture,
            current_subtype,
            48,
            current.len() as u32,
            0,
            other_architecture,
            other_subtype,
            48 + current.len() as u32,
            other.len() as u32,
            0,
        ] {
            universal.extend_from_slice(&value.to_be_bytes());
        }
        universal.extend_from_slice(&current);
        universal.extend_from_slice(&other);
        assert!(
            native_runtime_image_policy::architectures(&universal).is_err(),
            "unused executable slice cannot qualify with an unsupported loader"
        );
        native_runtime_image_policy::architectures(&current)
            .expect("unchanged current-slice data recovers");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_runtime_dedicated_image_path_is_only_the_installed_process_sibling() {
        assert_eq!(
            native_runtime_observer_path(std::path::Path::new(
                "/Applications/Made.app/Contents/MacOS/made"
            ))
            .expect("installed path data"),
            PathBuf::from("/Applications/Made.app/Contents/MacOS/native-cargo-runtime-observer")
        );
        for path in [
            "/private/tmp/helper",
            "/Applications/Made.app/Contents/Resources/made",
            "/Applications/Made.app/Contents/MacOS/native-cargo-runtime-observer",
            "Made.app/Contents/MacOS/made",
            "/Applications/Made.app/Contents/MacOS/../made",
        ] {
            assert!(
                native_runtime_observer_path(std::path::Path::new(path)).is_err(),
                "{path}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    fn native_runtime_test_image(dependency: Option<&str>) -> Vec<u8> {
        #[cfg(target_arch = "aarch64")]
        let (architecture, subtype) = (0x0100_000c_u32, 0_u32);
        #[cfg(target_arch = "x86_64")]
        let (architecture, subtype) = (0x0100_0007_u32, 3_u32);
        let mut command = Vec::new();
        if let Some(dependency) = dependency {
            let length = (12 + dependency.len() + 1 + 7) & !7;
            command.extend_from_slice(&0xe_u32.to_le_bytes());
            command.extend_from_slice(&(length as u32).to_le_bytes());
            command.extend_from_slice(&12_u32.to_le_bytes());
            command.extend_from_slice(dependency.as_bytes());
            command.resize(length, 0);
        }
        let mut image = Vec::new();
        for value in [
            0xfeed_facf_u32,
            architecture,
            subtype,
            2,
            u32::from(dependency.is_some()),
            command.len() as u32,
            0,
            0,
        ] {
            image.extend_from_slice(&value.to_le_bytes());
        }
        image.extend_from_slice(&command);
        image
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_runtime_dedicated_image_rejects_dynamic_paths_and_bounded_header_mutation() {
        let original = native_runtime_test_image(Some("/usr/lib/dyld"));
        let facts = native_runtime_image_policy::architectures(&original)
            .expect("supported header facts, not signing authority");
        assert_eq!(facts.len(), 1);
        for dependency in [
            "@rpath/libSystem.B.dylib",
            "/private/tmp/libSystem.B.dylib",
            "/usr/lib/../private/other",
            "/System/Library/../other",
        ] {
            assert!(
                native_runtime_image_policy::architectures(&native_runtime_test_image(Some(
                    dependency
                )))
                .is_err()
            );
        }
        for (offset, value) in [
            (0, 0_u32),
            (8, 99),
            (12, 6),
            (16, 4097),
            (20, 256 * 1024 + 1),
            (32, 0x27),
            (32, 0x2c),
            (32, 0x8000_0035),
            (36, 7),
            (40, 0),
        ] {
            let mut changed = original.clone();
            changed[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
            assert!(
                native_runtime_image_policy::architectures(&changed).is_err(),
                "offset {offset}"
            );
        }
        for length in [0, 3, 31, original.len() - 1] {
            assert!(native_runtime_image_policy::architectures(&original[..length]).is_err());
        }
        assert_eq!(
            native_runtime_image_policy::architectures(&original)
                .expect("fresh unchanged header recovery"),
            facts
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_runtime_dedicated_image_universal_roles_cannot_hide_or_overlap_slices() {
        let thin = native_runtime_test_image(None);
        let architecture = u32::from_le_bytes(thin[4..8].try_into().expect("controlled header"));
        let subtype = u32::from_le_bytes(thin[8..12].try_into().expect("controlled subtype"));
        let mut image = Vec::new();
        for value in [
            0xcafe_babe_u32,
            1,
            architecture,
            subtype,
            32,
            thin.len() as u32,
            0,
        ] {
            image.extend_from_slice(&value.to_be_bytes());
        }
        image.resize(32, 0);
        image.extend_from_slice(&thin);
        assert_eq!(
            native_runtime_image_policy::architectures(&image).expect("bounded slice facts"),
            [architecture]
        );
        for (offset, value) in [(4, 3_u32), (8, 0), (16, 8), (20, u32::MAX), (24, 31)] {
            let mut changed = image.clone();
            changed[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
            assert!(
                native_runtime_image_policy::architectures(&changed).is_err(),
                "offset {offset}"
            );
        }
        let mut duplicate = Vec::new();
        for value in [
            0xcafe_babe_u32,
            2,
            architecture,
            subtype,
            48,
            thin.len() as u32,
            0,
            architecture,
            subtype,
            48,
            thin.len() as u32,
            0,
        ] {
            duplicate.extend_from_slice(&value.to_be_bytes());
        }
        duplicate.extend_from_slice(&thin);
        assert!(
            native_runtime_image_policy::architectures(&duplicate).is_err(),
            "duplicate or overlapping slice cannot qualify"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_runtime_dedicated_image_loader_environment_is_explicitly_unqualified() {
        for key in [
            "DYLD_INSERT_LIBRARIES",
            "DYLD_LIBRARY_PATH",
            "DYLD_FALLBACK_LIBRARY_PATH",
            "DYLD_FRAMEWORK_PATH",
            "DYLD_IMAGE_SUFFIX",
            "DYLD_UNKNOWN",
        ] {
            let mut environment = std::collections::BTreeMap::from([(
                String::from("PATH"),
                String::from("/usr/bin:/bin"),
            )]);
            environment.insert(key.into(), "/private/tmp/foreign".into());
            assert!(
                validate_native_runtime_loader_environment(&environment).is_err(),
                "{key}"
            );
            environment.insert(key.into(), String::new());
            validate_native_runtime_loader_environment(&environment)
                .expect("empty loader variable has no injected search path");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_runtime_dedicated_image_sdk_refuses_unsigned_actual_image() {
        let directory =
            tempfile::tempdir_in("/private/tmp").expect("private signing refusal fixture");
        let path = directory.path().join("native-cargo-runtime-observer");
        std::fs::write(&path, native_runtime_test_image(None)).expect("actual unsigned bytes");
        assert!(
            native_runtime_image_policy::validate_code(&path, None, true).is_err(),
            "actual SDK refusal, not synthesized signing facts"
        );
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn native_runtime_dedicated_image_public_issuer_rejects_override_and_preserves_ordinary_recovery(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt as _;
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let filesystem = Arc::new(fs::RealFs::new(None, cx.background_executor.clone()));
        let project =
            crate::Project::test(filesystem, std::iter::empty::<&std::path::Path>(), cx).await;
        let store = project.read_with(cx, |project, _| {
            project
                .profile_results_store
                .clone()
                .expect("existing local helper owner")
        });
        let directory = tempfile::tempdir_in("/private/tmp").expect("private real image fixture");
        let helper = directory.path().join("helper");
        std::fs::write(&helper, b"#!/bin/sh\nprintf original\n").expect("actual helper bytes");
        std::fs::set_permissions(&helper, std::fs::Permissions::from_mode(0o755))
            .expect("actual mode");
        store.update(cx, |store, _| store.set_local_task_helper(helper));
        let ordinary = project
            .update(cx, |project, cx| {
                project.prepare_local_native_task_helper_image(cx)
            })
            .expect("ordinary image request")
            .await
            .expect("ordinary real sealed identity");
        assert!(
            project
                .update(cx, |project, cx| project
                    .revalidate_local_native_cargo_runtime_helper_image(
                        &ordinary, cx
                    ))
                .is_err(),
            "ordinary image cannot be relabelled as a runtime image"
        );
        assert!(
            project
                .update(cx, |project, cx| project
                    .prepare_local_native_cargo_runtime_helper_image(cx))
                .expect("real dedicated public request")
                .await
                .is_err(),
            "actual test image outside installed bundle must refuse"
        );
        project
            .update(cx, |project, cx| {
                project.revalidate_local_native_task_helper_image(&ordinary, cx)
            })
            .expect("ordinary recovery request")
            .await
            .expect("unchanged ordinary helper remains admitted");
        fn assert_send<T: Send>() {}
        assert_send::<LocalNativeTaskHelperImage>();
        assert!(ordinary.same_issuance(&ordinary.duplicate()));
        assert_eq!(
            store.read_with(cx, |store, _| store
                .build_report_budget
                .load(Ordering::Relaxed)),
            0
        );
        assert!(
            store.read_with(cx, |store, _| store.task_runs.is_empty()),
            "refused image reserves no run or quota"
        );
        project.update(cx, |project, cx| project.close(cx));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn build_task_report_reservation_bounds_and_retained_leases() {
        let budget = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let limit = util::macos_task_completion::MAX_TASK_STDOUT_REPORT_BYTES;
        assert!(BuildReportReservation::reserve(budget.clone(), 0).is_err());
        assert!(BuildReportReservation::reserve(budget.clone(), limit + 1).is_err());
        let checked = BuildReportReservation::reserve_for_invocation(budget.clone(), limit, true)
            .expect("checked metadata accounted");
        assert_eq!(
            budget.load(Ordering::Relaxed),
            limit + util::macos_task_completion::TASK_BUILD_INVOCATION_RESERVED_BYTES
        );
        assert!(
            BuildReportReservation::reserve_for_invocation(budget.clone(), limit, true).is_err()
        );
        let retained_checked = checked.clone();
        drop(checked);
        assert_ne!(
            budget.load(Ordering::Relaxed),
            0,
            "retained invocation lease holds its quota"
        );
        drop(retained_checked);
        assert_eq!(budget.load(Ordering::Relaxed), 0);
        let first =
            BuildReportReservation::reserve(budget.clone(), limit).expect("first bounded report");
        let retained = first.clone();
        let second =
            BuildReportReservation::reserve(budget.clone(), limit).expect("second bounded report");
        assert_eq!(
            budget.load(Ordering::Relaxed),
            MAX_BUILD_REPORT_RESERVED_BYTES
        );
        assert!(BuildReportReservation::reserve(budget.clone(), 1).is_err());
        drop(first);
        assert!(
            BuildReportReservation::reserve(budget.clone(), 1).is_err(),
            "external retained report still owns its reservation"
        );
        drop(retained);
        let replacement = BuildReportReservation::reserve(budget.clone(), limit)
            .expect("released reservation permits replacement");
        drop(second);
        drop(replacement);
        assert_eq!(budget.load(Ordering::Relaxed), 0);
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn build_task_report_local_actor_cancellation_and_helper_mode_admission(
        cx: &mut gpui::TestAppContext,
    ) {
        let filesystem = fs::FakeFs::new(cx.executor());
        let worktrees = cx.new(|_| WorktreeStore::local(true, filesystem, Default::default()));
        let store = cx.new(|cx| ProfileResultsStore::local(worktrees, cx));
        let missing = store.update(cx, |store, cx| {
            store.prepare_local_build_report("build-1".into(), 4096, cx)
        });
        assert!(missing.await.is_err());
        store.update(cx, |store, _| {
            store.set_task_helper("/remote/helper".into())
        });
        let wrong_mode = store.update(cx, |store, cx| {
            store.prepare_local_build_report("build-1".into(), 4096, cx)
        });
        assert!(
            wrong_mode.await.is_err(),
            "legacy native helper cannot be silently launched with local helper syntax"
        );
        store.update(cx, |store, _| {
            store.set_local_task_helper("/local/helper".into())
        });
        let bad_limit = store.update(cx, |store, cx| {
            store.prepare_local_build_report("build-1".into(), 0, cx)
        });
        assert!(bad_limit.await.is_err());
        let bad_id = store.update(cx, |store, cx| {
            store.prepare_local_build_report("bad\nrun".into(), 4096, cx)
        });
        assert!(bad_id.await.is_err());
        assert!(store.read_with(cx, |store, _| store.task_runs.is_empty()));
        let cancellation = Arc::new(AtomicBool::new(false));
        let (stop, stopped) = async_channel::bounded(1);
        let retained = store.update(cx, |store, cx| {
            let reservation =
                BuildReportReservation::reserve(store.build_report_budget.clone(), 4096)
                    .expect("bounded reservation");
            let key = (
                ProfileTaskActor::Local(cx.entity_id()),
                "held-build".to_owned(),
            );
            store.task_runs.insert(
                key.clone(),
                ProfileTaskRun {
                    completion: watch::channel(()).0,
                    generation: 1,
                    payload: ProfileTaskPayload::BuildReport(BuildReportTaskRun {
                        #[cfg(feature = "cargo-workspace")]
                        observation_arm: None,
                        runtime_admission: None,
                        mode: BuildReportMode::Stdout,
                        expected_invocation: None,
                        observed_invocation: None,
                        observed_cargo_runtime: None,
                        helper: store.task_helper.clone().expect("configured local mode"),
                        identity: None,
                        reservation: reservation.clone(),
                        report: None,
                        fingerprint: None,
                    }),
                    failed: false,
                    cancellation: cancellation.clone(),
                    stop,
                    _expiry: Task::ready(()),
                    observation: None,
                },
            );
            assert_eq!(store.task_runs.len(), 1);
            assert!(
                store
                    .task_runs
                    .get(&key)
                    .expect("held build")
                    .native()
                    .is_none()
            );
            store.forget_peer(proto::PeerId::default());
            assert_eq!(
                store.task_runs.len(),
                1,
                "peer revocation cannot consume a local actor reservation"
            );
            let native_key = (
                ProfileTaskActor::Peer(proto::PeerId::default()),
                "held-build".to_owned(),
            );
            let native_paths = ["profile.json", "profile.syms.json"].map(|path| {
                ProjectPath::from_proto(proto::ProjectPath {
                    worktree_id: 1,
                    path: path.into(),
                })
                .expect("native pair path")
            });
            let (native_stop, _) = async_channel::bounded(1);
            store.task_runs.insert(
                native_key.clone(),
                ProfileTaskRun {
                    completion: watch::channel(()).0,
                    generation: 2,
                    payload: ProfileTaskPayload::NativePair(NativePairTaskRun {
                        #[cfg(target_os = "macos")]
                        checked_invocation: None,
                        paths: native_paths,
                        versions: None,
                        fingerprints: None,
                        #[cfg(all(target_os = "macos", feature = "cargo-workspace"))]
                        capture_admission: None,
                    }),
                    failed: false,
                    cancellation: Arc::new(AtomicBool::new(false)),
                    stop: native_stop,
                    _expiry: Task::ready(()),
                    observation: None,
                },
            );
            store.set_local_task_helper("/replacement/helper".into());
            assert_eq!(
                store.task_runs.len(),
                1,
                "helper replacement retires only Build data ownership"
            );
            assert!(
                store
                    .task_runs
                    .get(&native_key)
                    .expect("legacy native pair retained")
                    .native()
                    .is_some()
            );
            store.forget_peer(proto::PeerId::default());
            assert!(
                store.task_runs.is_empty(),
                "peer invalidation still owns the native pair"
            );
            assert!(cancellation.load(Ordering::Relaxed));
            assert!(stopped.is_closed());
            reservation
        });
        assert_eq!(
            store.read_with(cx, |store, _| store
                .build_report_budget
                .load(Ordering::Relaxed)),
            4096
        );
        drop(retained);
        assert_eq!(
            store.read_with(cx, |store, _| store
                .build_report_budget
                .load(Ordering::Relaxed)),
            0
        );
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    fn build_task_report_expiry_generation_cannot_revoke_replacement(
        cx: &mut gpui::TestAppContext,
    ) {
        let filesystem = fs::FakeFs::new(cx.executor());
        let worktrees = cx.new(|_| WorktreeStore::local(true, filesystem, Default::default()));
        let store = cx.new(|cx| ProfileResultsStore::local(worktrees, cx));
        let cancellation = Arc::new(AtomicBool::new(false));
        let (stop, stopped) = async_channel::bounded(1);
        let (old_expiry, retained) = store.update(cx, |store, cx| {
            let key = (
                ProfileTaskActor::Local(cx.entity_id()),
                "same-build".to_owned(),
            );
            let old_expiry = store.task_expiry(key.clone(), 1, Duration::from_secs(1), cx);
            let expiry = store.task_expiry(key.clone(), 2, Duration::from_secs(10), cx);
            let reservation =
                BuildReportReservation::reserve(store.build_report_budget.clone(), 4096)
                    .expect("pending run reservation");
            store.task_runs.insert(
                key,
                ProfileTaskRun {
                    completion: watch::channel(()).0,
                    generation: 2,
                    payload: ProfileTaskPayload::BuildReport(BuildReportTaskRun {
                        #[cfg(feature = "cargo-workspace")]
                        observation_arm: None,
                        runtime_admission: None,
                        mode: BuildReportMode::Stdout,
                        expected_invocation: None,
                        observed_invocation: None,
                        observed_cargo_runtime: None,
                        helper: ProfileTaskHelper {
                            path: "/helper".into(),
                            mode: ProfileTaskHelperMode::LocalApplication,
                        },
                        identity: None,
                        reservation: reservation.clone(),
                        report: None,
                        fingerprint: None,
                    }),
                    failed: false,
                    cancellation: cancellation.clone(),
                    stop,
                    _expiry: expiry,
                    observation: None,
                },
            );
            (old_expiry, reservation)
        });
        cx.run_until_parked();
        cx.executor().advance_clock(Duration::from_secs(1));
        cx.run_until_parked();
        assert_eq!(store.read_with(cx, |store, _| store.task_runs.len()), 1);
        assert!(!cancellation.load(Ordering::Relaxed));
        drop(old_expiry);
        cx.executor().advance_clock(Duration::from_secs(9));
        cx.run_until_parked();
        assert!(store.read_with(cx, |store, _| store.task_runs.is_empty()));
        assert!(cancellation.load(Ordering::Relaxed));
        assert!(stopped.is_closed());
        assert_eq!(
            store.read_with(cx, |store, _| store
                .build_report_budget
                .load(Ordering::Relaxed)),
            4096
        );
        drop(retained);
        assert_eq!(
            store.read_with(cx, |store, _| store
                .build_report_budget
                .load(Ordering::Relaxed)),
            0
        );
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn build_task_report_cancelled_observer_retires_before_full_replacement(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        cx.update(|cx| {
            crate::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        let filesystem = Arc::new(fs::RealFs::new(None, cx.background_executor.clone()));
        let worktrees = cx.new(|_| WorktreeStore::local(true, filesystem, Default::default()));
        let store = cx.new(|cx| ProfileResultsStore::local(worktrees, cx));
        let helper = std::env::current_exe().expect("real test-helper executable");
        store.update(cx, |store, _| store.set_local_task_helper(helper));
        let expected = util::macos_task_completion::TaskBuildInvocation::new(
            PathBuf::from("/bin/sh"),
            vec!["-c".into(), "printf selected".into()],
            std::env::current_dir().expect("actual cwd"),
            std::collections::BTreeMap::new(),
        )
        .expect("bounded invocation");
        let limit = util::macos_task_completion::MAX_TASK_STDOUT_REPORT_BYTES;
        let first = store
            .update(cx, |store, cx| {
                store.prepare_checked_local_build_report(
                    "retiring-build".into(),
                    limit,
                    expected.clone(),
                    cx,
                )
            })
            .await
            .expect("actual listener admission");
        let retirement = first.reservation.retirement.clone();
        assert!(!retirement.is_closed());
        let rejected = store.update(cx, |store, cx| {
            store.prepare_checked_local_build_report(
                "concurrent-build".into(),
                limit,
                expected.clone(),
                cx,
            )
        });
        assert!(
            rejected.await.is_err(),
            "two full checked reports exceed the unchanged cap"
        );
        drop(first);
        let replacement = store
            .update(cx, |store, cx| {
                store.prepare_checked_local_build_report(
                    "replacement-build".into(),
                    limit,
                    expected,
                    cx,
                )
            })
            .await
            .expect("replacement waits for actual cancelled observation retirement");
        assert!(retirement.is_closed());
        assert!(store.read_with(cx, |store, cx| {
            store.task_runs.len() == 1 && store.build_binding_current(&replacement, cx)
        }));
        assert_eq!(
            store.read_with(cx, |store, _| store
                .build_report_budget
                .load(Ordering::Relaxed)),
            limit + util::macos_task_completion::TASK_BUILD_INVOCATION_RESERVED_BYTES,
        );
        let replacement_retirement = replacement.reservation.retirement.clone();
        drop(replacement);
        store.update(cx, |store, _| store.invalidate());
        assert!(replacement_retirement.recv().await.is_err());
        assert_eq!(
            store.read_with(cx, |store, _| store
                .build_report_budget
                .load(Ordering::Relaxed)),
            0,
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn build_task_report_deferred_helper_process_entry() {
        let Some(marker) = std::env::var_os("ZED_DEFERRED_BUILD_HELPER_MARKER") else {
            return;
        };
        std::fs::write(marker, std::process::id().to_string())
            .expect("actual early helper process marker");
        build_task_report_helper_process_entry();
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn build_task_report_deferred_observer_cancels_early_helper_and_recovers(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        let filesystem = Arc::new(fs::RealFs::new(None, cx.background_executor.clone()));
        let helper = std::env::current_exe().expect("actual existing helper image");
        let identity = cx
            .background_spawn({
                let filesystem = filesystem.clone();
                let helper = helper.clone();
                async move {
                    crate::profile_results::prepare_task_helper_identity(
                        filesystem.as_ref(),
                        helper,
                        &AtomicBool::new(false),
                    )
                    .await
                }
            })
            .await
            .expect("actual sealed helper image");
        let worktrees =
            cx.new(|_| WorktreeStore::local(true, filesystem.clone(), Default::default()));
        let store = cx.new(|cx| ProfileResultsStore::local(worktrees, cx));
        let actor = ProfileTaskActor::Local(store.entity_id());
        let directory = tempfile::tempdir_in("/private/tmp").expect("owned deferred fixture");
        let budget = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let executor = cx.background_executor.clone();
        for released in [false, true] {
            let name = if released { "fresh" } else { "cancelled" };
            let marker = directory.path().join(format!("child-{name}"));
            let helper_marker = directory.path().join(format!("helper-{name}"));
            let token = rand::random::<[u8; 32]>();
            let listener = util::macos_task_completion::TaskCompletionListener::bind(token)
                .expect("actual private listener");
            let socket = listener.path();
            let (arm, armed) = async_channel::bounded(1);
            let (stop, stopped) = async_channel::bounded(1);
            let (registration, _requests) = async_channel::bounded(1);
            let cancellation = Arc::new(AtomicBool::new(false));
            let reservation = BuildReportReservation::reserve(budget.clone(), 1024)
                .expect("real bounded observer reservation");
            let retired = reservation.retirement.clone();
            let observed = cx.background_spawn(ProfileResultsStore::observe_reserved_build_report(
                BuildReportObservation {
                    armed: Some(armed),
                    runtime_admission: None,
                    mode: BuildReportMode::Stdout,
                    expected_invocation: None,
                    key: (actor, name.into()),
                    generation: 1,
                    identity: identity.clone(),
                    listener,
                    stopped,
                    filesystem: filesystem.clone(),
                    cancellation: cancellation.clone(),
                    limit: 1024,
                    reservation,
                },
                registration,
            ));
            let token = token
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            let mut child = smol::process::Command::new(&helper)
                .args([
                    "--exact",
                    "profile_results_store::client_tests::build_task_report_deferred_helper_process_entry",
                    "--nocapture",
                ])
                .env_clear()
                .env("PATH", "/usr/bin:/bin")
                .env("ZED_BUILD_REPORT_TEST_SOCKET", socket)
                .env("ZED_DEFERRED_BUILD_HELPER_MARKER", &helper_marker)
                .env("ZED_DEFERRED_BUILD_CHILD_MARKER", &marker)
                .env("ZED_BUILD_REPORT_TEST_COMMAND", "printf child > \"$ZED_DEFERRED_BUILD_CHILD_MARKER\"; printf retained")
                .env(util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV, token)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
                .expect("actual early registered helper attempt");
            smol::future::race(
                async {
                    while !helper_marker.exists() {
                        executor.timer(Duration::from_millis(10)).await;
                    }
                },
                async {
                    executor.timer(Duration::from_secs(10)).await;
                    panic!("actual deferred helper did not start");
                },
            )
            .await;
            assert!(child.try_status().expect("actual helper status").is_none());
            assert!(!marker.exists(), "unarmed helper cannot execute its child");
            assert!(!retired.is_closed());
            if released {
                arm.try_send(()).expect("one actual owner wakeup");
                let report = observed.await.expect("fresh actual observation");
                assert_eq!(report.bytes.as_slice(), b"retained");
                assert!(marker.exists());
                assert!(
                    child
                        .status()
                        .await
                        .expect("actual successful helper exit")
                        .success()
                );
                assert!(
                    !retired.is_closed(),
                    "actual bytes retain their reservation"
                );
                drop(report);
            } else {
                cancellation.store(true, Ordering::Relaxed);
                stop.close();
                assert!(observed.await.is_err());
                assert!(
                    !child
                        .status()
                        .await
                        .expect("actual refused helper exit")
                        .success()
                );
                assert!(!marker.exists(), "cancelled unarmed child never ran");
            }
            assert!(retired.recv().await.is_err());
            assert_eq!(budget.load(Ordering::Relaxed), 0);
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn build_task_report_helper_process_entry() {
        let Ok(socket) = std::env::var("ZED_BUILD_REPORT_TEST_SOCKET") else {
            return;
        };
        let command =
            std::env::var("ZED_BUILD_REPORT_TEST_COMMAND").expect("controlled child workload");
        let mut child = std::process::Command::new("/bin/sh");
        child.args(["-c", command.as_str()]);
        let status = if std::env::var_os("ZED_BUILD_REPORT_TEST_CHECKED").is_some() {
            util::macos_task_completion::register_and_capture_build_stdout(
                std::path::Path::new(&socket),
                child,
            )
        } else {
            util::macos_task_completion::register_and_capture_stdout(
                std::path::Path::new(&socket),
                child,
            )
        }
        .expect("actual test-helper child observation");
        util::macos_task_completion::exit_with_observed_task_status(status);
    }

    #[cfg(all(target_os = "macos", feature = "test-support"))]
    #[gpui::test]
    async fn native_task_helper_image_public_owner_refuses_drift_and_recovers(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::os::unix::fs::PermissionsExt as _;
        cx.executor().allow_parking();
        cx.update(|cx| {
            let settings = settings::SettingsStore::test(cx);
            cx.set_global(settings);
            crate::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            );
            release_channel::init(semver::Version::new(0, 0, 0), cx);
        });
        let filesystem = Arc::new(fs::RealFs::new(None, cx.background_executor.clone()));
        let project =
            crate::Project::test(filesystem, std::iter::empty::<&std::path::Path>(), cx).await;
        let store = project.read_with(cx, |project, _| {
            project
                .profile_results_store
                .clone()
                .expect("local helper owner")
        });
        let directory = tempfile::tempdir_in("/private/tmp").expect("private helper fixture");
        let helper = directory.path().join("helper");
        let replacement = directory.path().join("replacement");
        for path in [&helper, &replacement] {
            std::fs::write(path, b"#!/bin/sh\nprintf first\n").expect("actual helper bytes");
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
                .expect("helper executable mode");
        }
        store.update(cx, |store, _| store.set_local_task_helper(helper.clone()));
        let image = project
            .update(cx, |project, cx| {
                project.prepare_local_native_task_helper_image(cx)
            })
            .expect("real local image admission")
            .await
            .expect("actual sealed image");
        assert_eq!(image.helper_path(), helper);
        project
            .update(cx, |project, cx| {
                project.revalidate_local_native_task_helper_image(&image, cx)
            })
            .expect("original live owner")
            .await
            .expect("unchanged image");
        assert_eq!(
            store.read_with(cx, |store, _| store
                .build_report_budget
                .load(Ordering::Relaxed)),
            0
        );
        assert!(
            store.read_with(cx, |store, _| store.task_runs.is_empty()),
            "image receipt reserves no run or execution authority"
        );
        store.update(cx, |store, _| {
            store.set_local_task_helper(replacement.clone());
            store.set_local_task_helper(helper.clone());
        });
        assert!(
            project
                .update(cx, |project, cx| project
                    .revalidate_local_native_task_helper_image(&image, cx))
                .is_err(),
            "restored configured path must not revive retired owner epoch"
        );
        let image = project
            .update(cx, |project, cx| {
                project.prepare_local_native_task_helper_image(cx)
            })
            .expect("fresh owner")
            .await
            .expect("fresh sealed image");
        std::fs::write(&helper, b"#!/bin/sh\nprintf other\n")
            .expect("same-sized actual helper mutation");
        assert!(
            project
                .update(cx, |project, cx| project
                    .revalidate_local_native_task_helper_image(&image, cx))
                .expect("live owner before filesystem read")
                .await
                .is_err()
        );
        let image = project
            .update(cx, |project, cx| {
                project.prepare_local_native_task_helper_image(cx)
            })
            .expect("fresh file owner")
            .await
            .expect("fresh actual image recovery");
        project
            .update(cx, |project, cx| {
                project.revalidate_local_native_task_helper_image(&image, cx)
            })
            .expect("fresh owner current")
            .await
            .expect("fresh bytes current");
        store.update(cx, |store, _| store.invalidate());
        assert!(
            project
                .update(cx, |project, cx| project
                    .revalidate_local_native_task_helper_image(&image, cx))
                .is_err(),
            "session invalidation retires image epoch"
        );
        let image = project
            .update(cx, |project, cx| {
                project.prepare_local_native_task_helper_image(cx)
            })
            .expect("fresh session owner")
            .await
            .expect("fresh session identity");
        let expected = util::macos_task_completion::TaskBuildInvocation::new(
            "/bin/echo".into(),
            vec!["controlled".into()],
            directory.path().to_path_buf(),
            Default::default(),
        )
        .expect("bounded exact future child data");
        assert!(
            image.same_issuance(&image.duplicate()),
            "retained helper receipt keeps original ownership"
        );
        let binding = project
            .update(cx, |project, cx| {
                // This control verifies the image/observer primitive, without issuing Cargo authority.
                project
                    .profile_results_store
                    .as_ref()
                    .expect("same prepared image owner")
                    .update(cx, |store, cx| {
                        anyhow::ensure!(
                            store.helper_image_current(&image, cx),
                            "same prepared helper identity"
                        );
                        Ok(store.prepare_checked_local_cargo_runtime_report(
                            "sealed-helper-runtime".into(),
                            1024,
                            expected,
                            cx,
                        ))
                    })
            })
            .expect("checked report admission")
            .await
            .expect("same prepared helper identity");
        assert_eq!(binding.helper_path(), image.helper_path());
        assert_eq!(
            binding.configured_helper_fingerprint(),
            image.identity.fingerprint()
        );
        assert!(binding.requires_cargo_runtime_observation());
        drop(binding);
        store.update(cx, |store, _| store.invalidate());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn build_task_runtime_report_helper_process_entry() {
        let Some(socket) = std::env::var_os("ZED_TEST_BUILD_RUNTIME_SOCKET") else {
            return;
        };
        let mut command =
            std::process::Command::new(std::env::current_exe().expect("helper image"));
        command.args([
            "--exact",
            "profile_results_store::client_tests::build_task_runtime_report_child_process_entry",
            "--nocapture",
        ]);
        match util::macos_task_completion::register_and_capture_cargo_runtime_stdout(
            std::path::Path::new(&socket),
            command,
        ) {
            Ok(status) => util::macos_task_completion::exit_with_observed_task_status(status),
            Err(error) => {
                eprintln!("Build runtime fixture failed ({:?})", error.kind());
                std::process::exit(1);
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn build_task_runtime_report_child_process_entry() {
        use std::io::Write as _;
        use std::os::unix::process::CommandExt as _;
        let Some(marker) = std::env::var_os("ZED_TEST_BUILD_RUNTIME_STARTED") else {
            return;
        };
        std::fs::write(marker, std::process::id().to_string()).expect("actual owned child marker");
        std::io::stdout()
            .write_all(b"{\"reason\":\"build-finished\",\"success\":true}\n")
            .expect("actual report bytes");
        std::io::stdout().flush().expect("raw stdout flush");
        let mut runner =
            std::process::Command::new(std::env::current_exe().expect("owned runner image"));
        runner.args([
            "--exact",
            "profile_results_store::client_tests::build_task_runtime_report_runner_process_entry",
            "--nocapture",
        ]).env("CARGO_MANIFEST_DIR", "/actual/runtime/package");
        let error = runner.exec();
        panic!("actual runtime exec failed ({:?})", error.kind());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn build_task_runtime_report_runner_process_entry() {
        if std::env::var_os("ZED_TEST_BUILD_RUNTIME_STARTED").is_none() {
            return;
        }
        let socket = std::env::var_os(util::macos_task_completion::TASK_CARGO_RUNTIME_ENDPOINT_ENV)
            .expect("outer-owned endpoint");
        match util::macos_task_completion::observe_cargo_runtime(
            std::path::Path::new(&socket),
            std::ffi::OsStr::new("./not-executed-workload"),
            &["literal spaces $ ; ' λ".into(), "".into()],
        ) {
            Ok(()) => std::process::exit(0),
            Err(error) => {
                eprintln!("Build runtime runner failed ({:?})", error.kind());
                std::process::exit(1);
            }
        }
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn build_task_runtime_report_owner_joins_actual_success_mode_and_retirement(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        cx.update(|cx| {
            crate::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        let filesystem = Arc::new(fs::RealFs::new(None, cx.background_executor.clone()));
        let worktrees = cx.new(|_| WorktreeStore::local(true, filesystem, Default::default()));
        let store = cx.new(|cx| ProfileResultsStore::local(worktrees, cx));
        store.update(cx, |store, _| {
            store.set_local_task_helper(std::env::current_exe().expect("helper"))
        });
        let directory = tempfile::tempdir_in("/private/tmp").expect("owned runtime fixture");
        for (index, runtime_mode) in [(0, false), (1, true), (2, true)] {
            let marker = directory.path().join(format!("started-{index}"));
            let expected = util::macos_task_completion::TaskBuildInvocation::new(
                std::env::current_exe().expect("actual child image"),
                vec!["--exact".into(), "profile_results_store::client_tests::build_task_runtime_report_child_process_entry".into(), "--nocapture".into()],
                directory.path().to_path_buf(),
                std::collections::BTreeMap::from([
                    ("PATH".into(), "/usr/bin:/bin".into()),
                    ("HOME".into(), "/issued/home".into()),
                    ("ZED_TEST_BUILD_RUNTIME_STARTED".into(), marker.to_str().expect("marker UTF8").into()),
                ]),
            ).expect("bounded complete invocation");
            let binding = store
                .update(cx, |store, cx| {
                    if runtime_mode {
                        store.prepare_checked_local_cargo_runtime_report(
                            format!("runtime-{index}"),
                            4 * 1024 * 1024,
                            expected.clone(),
                            cx,
                        )
                    } else {
                        store.prepare_checked_local_build_report(
                            format!("runtime-{index}"),
                            4 * 1024 * 1024,
                            expected.clone(),
                            cx,
                        )
                    }
                })
                .await
                .expect("real prepared owner");
            assert_eq!(binding.requires_cargo_runtime_observation(), runtime_mode);
            assert!(binding.requires_checked_invocation());
            let retirement = binding.reservation.retirement.clone();
            let token = binding
                .token()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            let child = smol::process::Command::new(binding.helper_path())
                .args(["--exact", "profile_results_store::client_tests::build_task_runtime_report_helper_process_entry", "--nocapture"])
                .env_clear()
                .env("ZED_TEST_BUILD_RUNTIME_SOCKET", binding.socket_path())
                .env(util::macos_task_completion::TASK_COMPLETION_TOKEN_ENV, token)
                .env(util::macos_task_completion::TASK_BUILD_INVOCATION_ENV, expected.encode_helper_environment().expect("transient issued envelope"))
                .current_dir(directory.path())
                .kill_on_drop(true)
                .spawn().expect("actual configured helper process");
            let observed = store.update(cx, |store, cx| store.await_build_report(binding, cx));
            let (observed, output) = futures::join!(observed, child.output());
            let output = output.expect("actual helper output");
            if !runtime_mode {
                assert!(observed.is_err(), "ACK4 must not admit ACK6 runtime bytes");
                assert!(!output.status.success());
                assert!(!marker.exists(), "wrong mode must not spawn the child");
            } else {
                let report = observed.expect("genuine ACK6 successful report");
                assert!(output.status.success());
                assert!(marker.exists());
                let runtime = report
                    .observed_cargo_runtime()
                    .expect("current lease")
                    .expect("genuine runtime record");
                assert_eq!(
                    runtime.program(),
                    directory.path().join("./not-executed-workload")
                );
                assert_eq!(runtime.working_directory(), directory.path());
                assert_eq!(runtime.arguments(), ["literal spaces $ ; ' λ", ""]);
                assert_eq!(
                    runtime
                        .environment()
                        .get("CARGO_MANIFEST_DIR")
                        .map(String::as_str),
                    Some("/actual/runtime/package")
                );
                assert!(
                    !runtime
                        .environment()
                        .keys()
                        .any(|key| key.starts_with("ZED_PROFILE_"))
                );
                assert!(
                    expected.matches_observed(
                        report
                            .observed_invocation()
                            .expect("current invocation lease")
                            .expect("checked invocation")
                    )
                );
                assert!(
                    report
                        .bytes()
                        .expect("leased bytes")
                        .windows(b"build-finished".len())
                        .any(|bytes| bytes == b"build-finished")
                );
                assert!(store.read_with(cx, |store, cx| {
                    store.observed_build_report_current(&report, cx)
                }));
                store.update(cx, |store, _| store.invalidate());
                assert!(
                    report.observed_cargo_runtime().is_err(),
                    "retired lease cannot return runtime authority"
                );
                assert!(
                    store.read_with(cx, |store, _| store
                        .build_report_budget
                        .load(Ordering::Relaxed))
                        > 0,
                    "report buffers retain the reservation after handle invalidation"
                );
                drop(report);
            }
            store.update(cx, |store, _| store.invalidate());
            assert!(
                retirement.recv().await.is_err(),
                "actual retained buffers retire before next admission"
            );
            assert_eq!(
                store.read_with(cx, |store, _| store
                    .build_report_budget
                    .load(Ordering::Relaxed)),
                0
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn build_task_report_checked_invocation_couples_actual_child_and_rejects_replacement(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        cx.update(|cx| {
            crate::task_store::set_experimental_feature_for_test(
                "rust-native-profiling-experimental",
                true,
                cx,
            )
        });
        let filesystem = Arc::new(fs::RealFs::new(None, cx.background_executor.clone()));
        let worktrees = cx.new(|_| WorktreeStore::local(true, filesystem, Default::default()));
        let store = cx.new(|cx| ProfileResultsStore::local(worktrees, cx));
        store.update(cx, |store, _| {
            store.set_local_task_helper(std::env::current_exe().expect("helper"))
        });
        let directory = tempfile::tempdir().expect("fixture");
        let marker = directory.path().join("executed");
        let command = format!("touch '{}'; printf selected", marker.display());
        for (index, changed) in [(0, false), (1, true)] {
            let expected_command = if changed {
                "printf different".to_owned()
            } else {
                command.clone()
            };
            let expected = util::macos_task_completion::TaskBuildInvocation::new(
                PathBuf::from("/bin/sh"),
                vec!["-c".into(), expected_command],
                std::env::current_dir().expect("cwd"),
                std::collections::BTreeMap::from([(
                    "ZED_BUILD_REPORT_REQUIRED".into(),
                    "retained".into(),
                )]),
            )
            .expect("bounded expected invocation");
            let binding = store
                .update(cx, |store, cx| {
                    store.prepare_checked_local_build_report(
                        format!("checked-{index}"),
                        64,
                        expected.clone(),
                        cx,
                    )
                })
                .await
                .expect("actual helper admission");
            assert!(binding.requires_checked_invocation());
            let retirement = binding.reservation.retirement.clone();
            let token: String = binding
                .token()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            let child = smol::process::Command::new(binding.helper_path())
                .args([
                    "--exact",
                    "profile_results_store::client_tests::build_task_report_helper_process_entry",
                    "--nocapture",
                ])
                .env("ZED_BUILD_REPORT_TEST_SOCKET", binding.socket_path())
                .env("ZED_BUILD_REPORT_TEST_COMMAND", &command)
                .env("ZED_BUILD_REPORT_TEST_CHECKED", "1")
                .env("ZED_BUILD_REPORT_REQUIRED", "retained")
                .env("ZED_PROFILE_TASK_TOKEN", token)
                .kill_on_drop(true)
                .spawn()
                .expect("actual helper");
            let observed = store.update(cx, |store, cx| store.await_build_report(binding, cx));
            let (observed, output) = futures::join!(observed, child.output());
            let output = output.expect("helper output");
            if changed {
                assert!(observed.is_err());
                assert!(!output.status.success());
                assert!(!marker.exists());
            } else {
                let report = observed.expect("actual matched child");
                assert!(output.status.success());
                assert_eq!(report.bytes().expect("current report"), b"selected");
                let actual = report
                    .observed_invocation()
                    .expect("current lease")
                    .expect("checked invocation");
                assert!(expected.matches_observed(actual));
                assert!(!actual.environment().contains_key("ZED_PROFILE_TASK_TOKEN"));
                assert!(store.read_with(cx, |store, cx| {
                    store.observed_build_report_current(&report, cx)
                }));
                std::fs::remove_file(&marker).expect("reset private marker");
                store.update(cx, |store, _| store.invalidate());
                assert!(
                    report.observed_invocation().is_err(),
                    "retired lease cannot attest an invocation"
                );
            }
            store.update(cx, |store, _| store.invalidate());
            assert!(retirement.recv().await.is_err());
        }
        assert_eq!(
            store.read_with(cx, |store, _| store
                .build_report_budget
                .load(Ordering::Relaxed)),
            0
        );
    }

    #[cfg(target_os = "macos")]
    #[gpui::test]
    async fn build_task_report_owner_accepts_only_actual_bounded_success(
        cx: &mut gpui::TestAppContext,
    ) {
        cx.executor().allow_parking();
        let filesystem = Arc::new(fs::RealFs::new(None, cx.background_executor.clone()));
        let worktrees = cx.new(|_| WorktreeStore::local(true, filesystem, Default::default()));
        let store = cx.new(|cx| ProfileResultsStore::local(worktrees, cx));
        let helper = std::env::current_exe().expect("real test-helper executable");
        store.update(cx, |store, _| store.set_local_task_helper(helper));
        for (index, command, succeeds) in [
            (
                0,
                "printf '{\"reason\":\"build-finished\",\"success\":true}'; exit 0",
                true,
            ),
            (
                1,
                "printf '{\"reason\":\"build-finished\",\"success\":true}'; exit 7",
                false,
            ),
            (2, "printf '%0100d' 0; exit 0", false),
        ] {
            let run_id = format!("actual-build-{index}");
            let binding = store
                .update(cx, |store, cx| {
                    store.prepare_local_build_report(run_id.clone(), 64, cx)
                })
                .await
                .expect("real helper identity and listener");
            assert_eq!(
                binding.helper_mode(),
                ProfileTaskHelperMode::LocalApplication
            );
            assert_eq!(binding.run_id(), run_id);
            let token: String = binding
                .token()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            let child = smol::process::Command::new(binding.helper_path())
                .args([
                    "--exact",
                    "profile_results_store::client_tests::build_task_report_helper_process_entry",
                    "--nocapture",
                ])
                .env("ZED_BUILD_REPORT_TEST_SOCKET", binding.socket_path())
                .env("ZED_BUILD_REPORT_TEST_COMMAND", command)
                .env("ZED_PROFILE_TASK_TOKEN", token)
                .kill_on_drop(true)
                .spawn()
                .expect("actual kernel-observed helper process");
            let observed = store.update(cx, |store, cx| store.await_build_report(binding, cx));
            let (observed, output) = futures::join!(observed, child.output());
            let output = output.expect("helper process output");
            if succeeds {
                let report = observed.expect("actual successful bounded child report");
                assert!(output.status.success());
                assert_eq!(
                    report.bytes().expect("current report lease"),
                    b"{\"reason\":\"build-finished\",\"success\":true}"
                );
                assert!(store.read_with(cx, |store, cx| {
                    store.observed_build_report_current(&report, cx)
                }));
                store.update(cx, |store, _| store.invalidate());
                assert!(
                    report.bytes().is_err(),
                    "revoked owner cannot expose a fresh report lease"
                );
                assert!(!store.read_with(cx, |store, cx| {
                    store.observed_build_report_current(&report, cx)
                }));
            } else {
                assert!(
                    observed.is_err(),
                    "a JSON success field cannot override actual failure or report overflow"
                );
                assert!(!output.status.success());
                store.update(cx, |store, _| store.invalidate());
            }
        }
    }

    #[test]
    fn task_binding_response_requires_matching_identity_and_phase() {
        let response = proto::ProfileTaskBindingResponse {
            protocol_version: PROFILE_PROTOCOL_VERSION,
            task_run_id: "run-1".into(),
            status: proto::ProfileResultStatus::Ready as i32,
            complete: false,
            ..Default::default()
        };
        assert!(check_task_binding_response(response.clone(), "run-1", false).is_ok());
        assert!(check_task_binding_response(response.clone(), "run-2", false).is_err());
        assert!(check_task_binding_response(response.clone(), "run-1", true).is_err());
        let complete = proto::ProfileTaskBindingResponse {
            complete: true,
            ..response.clone()
        };
        assert!(check_task_binding_response(complete, "run-1", true).is_ok());
        let unsupported = proto::ProfileTaskBindingResponse {
            status: proto::ProfileResultStatus::Unsupported as i32,
            ..response
        };
        assert!(check_task_binding_response(unsupported, "run-1", false).is_err());
    }

    #[derive(Default)]
    struct ScriptedClient {
        handlers: Mutex<rpc::ProtoMessageHandlerSet>,
        requests: Mutex<Vec<proto::Envelope>>,
        replies: Mutex<VecDeque<BoxFuture<'static, Result<proto::Envelope>>>>,
    }

    impl rpc::ProtoClient for ScriptedClient {
        fn request(
            &self,
            envelope: proto::Envelope,
            _: &'static str,
        ) -> BoxFuture<'static, Result<proto::Envelope>> {
            let cancellation = matches!(
                envelope.payload.as_ref(),
                Some(proto::envelope::Payload::CancelProfileImport(_))
            );
            self.requests.lock().push(envelope);
            if cancellation {
                return futures::future::ready(Ok(proto::Ack {}.into_envelope(0, None, None)))
                    .boxed();
            }
            self.replies.lock().pop_front().unwrap_or_else(|| {
                futures::future::ready(Err(anyhow::anyhow!("unexpected profile request"))).boxed()
            })
        }

        fn send(&self, _: proto::Envelope, _: &'static str) -> Result<()> {
            anyhow::bail!("unexpected send")
        }

        fn send_response(&self, _: proto::Envelope, _: &'static str) -> Result<()> {
            anyhow::bail!("unexpected response")
        }

        fn message_handler_set(&self) -> &Mutex<rpc::ProtoMessageHandlerSet> {
            &self.handlers
        }
        fn is_via_collab(&self) -> bool {
            false
        }
        fn has_wsl_interop(&self) -> bool {
            false
        }
    }

    impl ScriptedClient {
        fn page(&self, page: proto::ProfileResultPage) {
            self.replies
                .lock()
                .push_back(futures::future::ready(Ok(page.into_envelope(0, None, None))).boxed());
        }

        fn cancellations(&self) -> Vec<proto::CancelProfileImport> {
            self.requests
                .lock()
                .iter()
                .filter_map(|envelope| match envelope.payload.as_ref() {
                    Some(proto::envelope::Payload::CancelProfileImport(request)) => {
                        Some(request.clone())
                    }
                    _ => None,
                })
                .collect()
        }
    }

    fn page(index: u32) -> proto::ProfileResultPage {
        proto::ProfileResultPage {
            import_id: 41,
            generation: 7,
            protocol_version: PROFILE_PROTOCOL_VERSION,
            page_index: index,
            status: proto::ProfileResultStatus::Ready as i32,
            summary: (index == 0).then(|| proto::ProfileResultSummary {
                frame_count: 129,
                ..Default::default()
            }),
            frame_offset: index,
            frames: vec![proto::ProfileResultFrame {
                label: "frame".into(),
                resolved: true,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn import(
        client: &Arc<ScriptedClient>,
        cancel: Arc<AtomicBool>,
        cx: &gpui::TestAppContext,
    ) -> Task<Result<crate::profile_results::ProfileResults>> {
        receive_remote_profile(
            proto::ImportProfileResults {
                project_id: 29,
                import_id: 41,
                protocol_version: PROFILE_PROTOCOL_VERSION,
                ..Default::default()
            },
            AnyProtoClient::new(client.clone()),
            cancel,
            |_| true,
            &cx.executor(),
        )
    }

    #[gpui::test]
    async fn profile_results_client_drop_cancels_pending_requests(cx: &mut gpui::TestAppContext) {
        for received_first_page in [false, true] {
            let client = Arc::new(ScriptedClient::default());
            if received_first_page {
                client.page(page(0));
            }
            let (sender, receiver) = futures::channel::oneshot::channel::<proto::Envelope>();
            client
                .replies
                .lock()
                .push_back(async move { Ok(receiver.await?) }.boxed());
            let cancellation = Arc::new(AtomicBool::new(false));
            let task = import(&client, cancellation.clone(), cx);
            cx.run_until_parked();
            assert_eq!(
                client.requests.lock().len(),
                if received_first_page { 2 } else { 1 }
            );
            drop(task);
            cx.run_until_parked();
            assert!(cancellation.load(Ordering::Relaxed));
            let cancellations = client.cancellations();
            assert_eq!(cancellations.len(), 1);
            let request = cancellations.first().expect("cancellation request");
            assert_eq!((request.project_id, request.import_id), (29, 41));
            assert_eq!(request.generation, if received_first_page { 7 } else { 0 });
            assert!(
                sender.send(page(0).into_envelope(0, None, None)).is_err(),
                "dropped import must not accept a late response"
            );
        }
    }

    #[gpui::test]
    async fn profile_results_client_bounds_fragmented_and_stale_responses(
        cx: &mut gpui::TestAppContext,
    ) {
        let client = Arc::new(ScriptedClient::default());
        for index in 0..128 {
            client.page(page(index));
        }
        let error = import(&client, Arc::new(AtomicBool::new(false)), cx)
            .await
            .expect_err("fragmented host stream exceeds page ceiling");
        assert_eq!(
            error.downcast_ref::<RemoteProfileError>(),
            Some(&RemoteProfileError::Invalid)
        );
        cx.run_until_parked();
        assert_eq!(client.requests.lock().len(), 129);
        assert_eq!(client.cancellations().len(), 1);

        let client = Arc::new(ScriptedClient::default());
        client.page(page(0));
        let mut stale = page(1);
        stale.generation = 6;
        client.page(stale);
        let error = import(&client, Arc::new(AtomicBool::new(false)), cx)
            .await
            .expect_err("stale generation rejected");
        assert_eq!(
            error.downcast_ref::<RemoteProfileError>(),
            Some(&RemoteProfileError::Invalid)
        );
        cx.run_until_parked();
        assert_eq!(
            client
                .cancellations()
                .first()
                .expect("cancel current stream")
                .generation,
            7
        );
    }

    #[gpui::test]
    async fn profile_results_client_reports_cancellation_during_page_assembly(
        cx: &mut gpui::TestAppContext,
    ) {
        use std::sync::atomic::AtomicUsize;

        for cancel_on_lookup in [1, 2] {
            let client = Arc::new(ScriptedClient::default());
            let mut response = page(0);
            response.complete = true;
            response
                .summary
                .as_mut()
                .expect("first-page summary")
                .frame_count = 2;
            response.frames[0].source = Some(proto::ProjectPath {
                worktree_id: 1,
                path: "source.rs".into(),
            });
            response.frames[0].source_line = Some(1);
            response.frames.push(proto::ProfileResultFrame {
                label: "later frame".into(),
                resolved: true,
                ..Default::default()
            });
            client.page(response);

            let cancellation = Arc::new(AtomicBool::new(false));
            let cancellation_on_source = cancellation.clone();
            let source_lookups = Arc::new(AtomicUsize::new(0));
            let source_lookups_for_import = source_lookups.clone();
            let result = receive_remote_profile(
                proto::ImportProfileResults {
                    project_id: 29,
                    import_id: 41,
                    protocol_version: PROFILE_PROTOCOL_VERSION,
                    ..Default::default()
                },
                AnyProtoClient::new(client.clone()),
                cancellation.clone(),
                move |_| {
                    if source_lookups_for_import.fetch_add(1, Ordering::Relaxed) + 1
                        == cancel_on_lookup
                    {
                        cancellation_on_source.store(true, Ordering::Relaxed);
                    }
                    true
                },
                &cx.executor(),
            )
            .await
            .expect_err("cancellation during page assembly must discard the partial model");
            assert_eq!(
                result.downcast_ref::<RemoteProfileError>(),
                Some(&RemoteProfileError::Cancelled)
            );
            cx.run_until_parked();
            assert_eq!(source_lookups.load(Ordering::Relaxed), cancel_on_lookup);
            assert!(cancellation.load(Ordering::Relaxed));
            assert_eq!(client.cancellations().len(), 1);
        }
    }

    #[gpui::test]
    async fn profile_results_client_completion_does_not_cancel(cx: &mut gpui::TestAppContext) {
        let client = Arc::new(ScriptedClient::default());
        let mut complete = page(0);
        complete.complete = true;
        complete
            .summary
            .as_mut()
            .expect("first-page summary")
            .frame_count = 1;
        client.page(complete);
        let cancellation = Arc::new(AtomicBool::new(false));
        let result = import(&client, cancellation.clone(), cx)
            .await
            .expect("complete empty profile");
        assert_eq!(result.total_samples, 0);
        cx.run_until_parked();
        assert!(!cancellation.load(Ordering::Relaxed));
        assert!(client.cancellations().is_empty());
    }
}
