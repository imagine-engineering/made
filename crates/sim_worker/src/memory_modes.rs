use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sim_runtime::MemoryPolicy;
use sim_tensor::{
    BackendCapabilityMatrix, CachedAllocationOwner, CancellationToken, DeviceId, TensorError,
};
use sim_types::DeviceKind;
use thiserror::Error;

use crate::{
    MemoryPlan, MemoryPlanError, MemoryPlanRequest, MemoryPlanner, MemoryReservationKind,
    MemoryRetryTracker,
};

pub const CATALOG_MEMORY_MODE_COUNT: usize = 5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CatalogMemoryMode {
    pub source_identifier: &'static str,
    pub native_disposition: &'static str,
}

pub const CATALOG_MEMORY_MODES: [CatalogMemoryMode; CATALOG_MEMORY_MODE_COUNT] = [
    CatalogMemoryMode {
        source_identifier: "Dynamic VRAM",
        native_disposition: "dynamic",
    },
    CatalogMemoryMode {
        source_identifier: "asynchronous offload",
        native_disposition: "asynchronous_offload",
    },
    CatalogMemoryMode {
        source_identifier: "highvram/gpu-only",
        native_disposition: "high_vram_or_gpu_only",
    },
    CatalogMemoryMode {
        source_identifier: "lowvram/novram",
        native_disposition: "low_vram_or_no_vram",
    },
    CatalogMemoryMode {
        source_identifier: "pinned memory",
        native_disposition: "pinned_staging",
    },
];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelResidencyMode {
    #[default]
    Dynamic,
    HighVram,
    GpuOnly,
    LowVram,
    NoVram,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MemoryModeRequest {
    pub residency: ModelResidencyMode,
    pub asynchronous_offload: bool,
    pub pinned_staging: bool,
    pub mmap_weights: bool,
}

impl Default for MemoryModeRequest {
    fn default() -> Self {
        Self {
            residency: ModelResidencyMode::Dynamic,
            asynchronous_offload: false,
            pinned_staging: false,
            mmap_weights: true,
        }
    }
}

impl MemoryModeRequest {
    pub const fn from_runtime_policy(policy: MemoryPolicy) -> Self {
        match policy {
            MemoryPolicy::Conservative => Self {
                residency: ModelResidencyMode::NoVram,
                asynchronous_offload: false,
                pinned_staging: false,
                mmap_weights: true,
            },
            MemoryPolicy::Balanced => Self {
                residency: ModelResidencyMode::Dynamic,
                asynchronous_offload: false,
                pinned_staging: false,
                mmap_weights: true,
            },
            MemoryPolicy::Performance => Self {
                residency: ModelResidencyMode::HighVram,
                asynchronous_offload: false,
                pinned_staging: false,
                mmap_weights: false,
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MemoryModeCapabilities {
    pub device: DeviceId,
    pub supports_asynchronous_offload: bool,
    pub supports_pinned_staging: bool,
    pub supports_mmap_weights: bool,
}

impl MemoryModeCapabilities {
    pub fn from_backend(backend: &BackendCapabilityMatrix) -> Self {
        Self {
            device: backend.device(),
            supports_asynchronous_offload: false,
            supports_pinned_staging: false,
            supports_mmap_weights: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OffloadTarget {
    None,
    HostPinned,
    HostPageable,
    MemoryMapped,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OffloadGranularity {
    None,
    Layer,
    Group,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EffectiveMemoryMode {
    pub residency: ModelResidencyMode,
    pub asynchronous_offload: bool,
    pub pinned_staging: bool,
    pub mmap_weights: bool,
    pub offload_target: OffloadTarget,
    pub offload_granularity: OffloadGranularity,
}

impl EffectiveMemoryMode {
    pub fn resolve(
        request: MemoryModeRequest,
        capabilities: MemoryModeCapabilities,
    ) -> Result<Self, MemoryPolicyError> {
        for (enabled, supported, mode) in [
            (
                request.asynchronous_offload,
                capabilities.supports_asynchronous_offload,
                "asynchronous offload",
            ),
            (
                request.pinned_staging,
                capabilities.supports_pinned_staging,
                "pinned memory",
            ),
            (
                request.mmap_weights,
                capabilities.supports_mmap_weights,
                "mmap weights",
            ),
        ] {
            if enabled && !supported {
                return Err(MemoryPolicyError::UnsupportedMode {
                    mode,
                    device: capabilities.device,
                });
            }
        }
        if request.asynchronous_offload
            && matches!(
                request.residency,
                ModelResidencyMode::HighVram | ModelResidencyMode::GpuOnly
            )
        {
            return Err(MemoryPolicyError::ConflictingModes {
                first: "asynchronous offload",
                second: "resident-only",
            });
        }
        if request.mmap_weights && request.residency == ModelResidencyMode::GpuOnly {
            return Err(MemoryPolicyError::ConflictingModes {
                first: "mmap weights",
                second: "gpu-only",
            });
        }
        let offload_granularity = match request.residency {
            ModelResidencyMode::Dynamic | ModelResidencyMode::LowVram => OffloadGranularity::Group,
            ModelResidencyMode::NoVram => OffloadGranularity::Layer,
            ModelResidencyMode::HighVram | ModelResidencyMode::GpuOnly => OffloadGranularity::None,
        };
        let offload_target = if offload_granularity == OffloadGranularity::None {
            OffloadTarget::None
        } else if request.pinned_staging {
            OffloadTarget::HostPinned
        } else if request.mmap_weights {
            OffloadTarget::MemoryMapped
        } else {
            OffloadTarget::HostPageable
        };
        Ok(Self {
            residency: request.residency,
            asynchronous_offload: request.asynchronous_offload,
            pinned_staging: request.pinned_staging,
            mmap_weights: request.mmap_weights,
            offload_target,
            offload_granularity,
        })
    }

    pub fn configuration_token(self) -> String {
        format!(
            "residency={:?};async={};pinned={};mmap={};target={:?};granularity={:?}",
            self.residency,
            self.asynchronous_offload,
            self.pinned_staging,
            self.mmap_weights,
            self.offload_target,
            self.offload_granularity,
        )
        .to_ascii_lowercase()
    }

    pub const fn patch_uses_weight_dtype(self) -> bool {
        matches!(
            self.residency,
            ModelResidencyMode::LowVram | ModelResidencyMode::NoVram
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct MemoryResourceId(u64);

impl MemoryResourceId {
    pub fn new(value: u64) -> Result<Self, MemoryPolicyError> {
        (value != 0)
            .then_some(Self(value))
            .ok_or(MemoryPolicyError::ZeroIdentifier("memory resource"))
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct MemoryFenceId(u64);

impl MemoryFenceId {
    pub fn new(value: u64) -> Result<Self, MemoryPolicyError> {
        (value != 0)
            .then_some(Self(value))
            .ok_or(MemoryPolicyError::ZeroIdentifier("memory fence"))
    }

    pub const fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryResourceKind {
    Weights,
    Patches,
    Workspace,
    Activations,
    Staging,
    Preview,
    DecodedMedia,
    Cache,
    Codec,
    Output,
    ModelPage,
    IpcExport,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "device")]
pub enum MemoryLocation {
    Device(DeviceId),
    HostPinned,
    HostPageable,
    MemoryMapped,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MemoryResource {
    id: MemoryResourceId,
    kind: MemoryResourceKind,
    bytes: u64,
    location: MemoryLocation,
    last_used: u64,
    durable: bool,
    pinned: bool,
    in_flight: bool,
    active: bool,
    offloadable: bool,
    fence: Option<MemoryFenceId>,
}

impl MemoryResource {
    pub fn new(
        id: MemoryResourceId,
        kind: MemoryResourceKind,
        bytes: u64,
        location: MemoryLocation,
    ) -> Result<Self, MemoryPolicyError> {
        if bytes == 0 {
            return Err(MemoryPolicyError::ZeroBytes(id));
        }
        Ok(Self {
            id,
            kind,
            bytes,
            location,
            last_used: 0,
            durable: matches!(
                kind,
                MemoryResourceKind::Weights | MemoryResourceKind::Patches
            ),
            pinned: false,
            in_flight: false,
            active: true,
            offloadable: matches!(
                kind,
                MemoryResourceKind::Activations | MemoryResourceKind::ModelPage
            ),
            fence: None,
        })
    }

    pub const fn id(&self) -> MemoryResourceId {
        self.id
    }

    pub const fn kind(&self) -> MemoryResourceKind {
        self.kind
    }

    pub const fn bytes(&self) -> u64 {
        self.bytes
    }

    pub const fn location(&self) -> MemoryLocation {
        self.location
    }

    pub const fn fence(&self) -> Option<MemoryFenceId> {
        self.fence
    }

    pub fn with_durable(mut self, durable: bool) -> Self {
        self.durable = durable;
        self
    }

    pub fn with_pinned(mut self, pinned: bool) -> Self {
        self.pinned = pinned;
        self
    }

    pub fn with_in_flight(mut self, in_flight: bool) -> Self {
        self.in_flight = in_flight;
        self
    }

    pub fn with_active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    pub fn with_offloadable(mut self, offloadable: bool) -> Self {
        self.offloadable = offloadable;
        self
    }

    fn protected(&self) -> bool {
        self.pinned || self.in_flight || self.fence.is_some()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DeviceMemoryBudget {
    pub device: DeviceId,
    pub capacity_bytes: u64,
    pub durable_baseline_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MemoryTopology {
    devices: Vec<DeviceMemoryBudget>,
    host_pinned_capacity_bytes: u64,
    host_pageable_capacity_bytes: u64,
    peer_links: Vec<(DeviceId, DeviceId)>,
}

impl MemoryTopology {
    pub fn new(
        devices: Vec<DeviceMemoryBudget>,
        host_pinned_capacity_bytes: u64,
        host_pageable_capacity_bytes: u64,
        peer_links: Vec<(DeviceId, DeviceId)>,
    ) -> Result<Self, MemoryPolicyError> {
        if devices.is_empty() {
            return Err(MemoryPolicyError::NoDevices);
        }
        for (index, budget) in devices.iter().enumerate() {
            if budget.durable_baseline_bytes > budget.capacity_bytes {
                return Err(MemoryPolicyError::BaselineExceedsCapacity {
                    device: budget.device,
                    baseline_bytes: budget.durable_baseline_bytes,
                    capacity_bytes: budget.capacity_bytes,
                });
            }
            if devices
                .get(..index)
                .is_some_and(|prior| prior.iter().any(|value| value.device == budget.device))
            {
                return Err(MemoryPolicyError::DuplicateDevice(budget.device));
            }
        }
        for (first, second) in &peer_links {
            if first == second
                || !contains_device(&devices, *first)
                || !contains_device(&devices, *second)
            {
                return Err(MemoryPolicyError::InvalidPeerLink {
                    first: *first,
                    second: *second,
                });
            }
        }
        Ok(Self {
            devices,
            host_pinned_capacity_bytes,
            host_pageable_capacity_bytes,
            peer_links,
        })
    }

    pub fn single_device(
        device: DeviceId,
        capacity_bytes: u64,
        durable_baseline_bytes: u64,
    ) -> Result<Self, MemoryPolicyError> {
        Self::new(
            vec![DeviceMemoryBudget {
                device,
                capacity_bytes,
                durable_baseline_bytes,
            }],
            capacity_bytes,
            capacity_bytes,
            Vec::new(),
        )
    }

    pub fn devices(&self) -> &[DeviceMemoryBudget] {
        &self.devices
    }

    pub fn supports_peer_copy(&self, first: DeviceId, second: DeviceId) -> bool {
        first == second
            || self.peer_links.iter().any(|(left, right)| {
                (*left == first && *right == second) || (*left == second && *right == first)
            })
    }
}

fn contains_device(devices: &[DeviceMemoryBudget], device: DeviceId) -> bool {
    devices.iter().any(|budget| budget.device == device)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvictionActionKind {
    Dropped,
    OffloadedToPinnedHost,
    OffloadedToPageableHost,
    PagedOutToMapping,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EvictionAction {
    pub resource_id: MemoryResourceId,
    pub resource_kind: MemoryResourceKind,
    pub bytes_reclaimed: u64,
    pub action: EvictionActionKind,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MemoryAccountingSnapshot {
    pub device_bytes: Vec<(DeviceId, u64)>,
    pub host_pinned_bytes: u64,
    pub host_pageable_bytes: u64,
    pub mapped_bytes: u64,
    pub durable_bytes: u64,
    pub fenced_bytes: u64,
    pub resource_count: usize,
}

impl MemoryAccountingSnapshot {
    pub fn source_memory_stats(
        &self,
        device: DeviceId,
    ) -> Result<BTreeMap<String, u64>, MemoryPolicyError> {
        let mut matching_bytes = self
            .device_bytes
            .iter()
            .filter_map(|(candidate, bytes)| (*candidate == device).then_some(*bytes));
        let active_bytes = matching_bytes
            .next()
            .ok_or(MemoryPolicyError::UnknownDevice(device))?;
        if matching_bytes.next().is_some() {
            return Err(MemoryPolicyError::DuplicateDevice(device));
        }
        Ok(BTreeMap::from([
            ("active_bytes.all.current".to_owned(), active_bytes),
            ("reserved_bytes.all.current".to_owned(), active_bytes),
        ]))
    }
}

pub fn npu_memory_stats_exact_native(
    capabilities: &BackendCapabilityMatrix,
    snapshot: &MemoryAccountingSnapshot,
    cancellation: &CancellationToken,
) -> Result<BTreeMap<String, u64>, NativeMemoryStatsError> {
    source_memory_stats_exact_native(capabilities, snapshot, &[DeviceKind::Npu], cancellation)
}

pub fn xpu_memory_stats_exact_native(
    capabilities: &BackendCapabilityMatrix,
    snapshot: &MemoryAccountingSnapshot,
    cancellation: &CancellationToken,
) -> Result<BTreeMap<String, u64>, NativeMemoryStatsError> {
    cancellation.check()?;
    source_memory_stats_exact_native(capabilities, snapshot, &[DeviceKind::Xpu], cancellation)
}

pub fn mlu_memory_stats_exact_native(
    capabilities: &BackendCapabilityMatrix,
    snapshot: &MemoryAccountingSnapshot,
    cancellation: &CancellationToken,
) -> Result<BTreeMap<String, u64>, NativeMemoryStatsError> {
    cancellation.check()?;
    source_memory_stats_exact_native(capabilities, snapshot, &[DeviceKind::Mlu], cancellation)
}

pub fn cuda_memory_stats_exact_native(
    capabilities: &BackendCapabilityMatrix,
    snapshot: &MemoryAccountingSnapshot,
    cancellation: &CancellationToken,
) -> Result<BTreeMap<String, u64>, NativeMemoryStatsError> {
    cancellation.check()?;
    source_memory_stats_exact_native(
        capabilities,
        snapshot,
        &[DeviceKind::Cuda, DeviceKind::Rocm],
        cancellation,
    )
}

fn source_memory_stats_exact_native(
    capabilities: &BackendCapabilityMatrix,
    snapshot: &MemoryAccountingSnapshot,
    expected_kinds: &[DeviceKind],
    cancellation: &CancellationToken,
) -> Result<BTreeMap<String, u64>, NativeMemoryStatsError> {
    cancellation.check()?;
    let device = capabilities.device();
    if !expected_kinds.contains(&device.kind()) {
        return Err(NativeMemoryStatsError::UnsupportedDevice(device));
    }
    let stats = snapshot.source_memory_stats(device)?;
    cancellation.check()?;
    Ok(stats)
}

pub fn cuda_memory_summary_exact_native(
    capabilities: &BackendCapabilityMatrix,
    snapshot: &MemoryAccountingSnapshot,
    cancellation: &CancellationToken,
) -> Result<String, NativeMemoryStatsError> {
    cancellation.check()?;
    let device = capabilities.device();
    if !matches!(device.kind(), DeviceKind::Cuda | DeviceKind::Rocm) {
        return Err(NativeMemoryStatsError::UnsupportedDevice(device));
    }
    let stats = snapshot.source_memory_stats(device)?;
    let active = stats.get("active_bytes.all.current").copied().ok_or(
        NativeMemoryStatsError::MissingStat("active_bytes.all.current"),
    )?;
    let reserved = stats.get("reserved_bytes.all.current").copied().ok_or(
        NativeMemoryStatsError::MissingStat("reserved_bytes.all.current"),
    )?;
    cancellation.check()?;
    Ok(format!(
        "Zed native {:?} memory summary, device {}\nactive_bytes.all.current: {active}\nreserved_bytes.all.current: {reserved}\n",
        device.kind(),
        device.ordinal()
    ))
}

pub fn xpu_device_count_exact_native(
    topology: &MemoryTopology,
    cancellation: &CancellationToken,
) -> Result<u32, NativeMemoryStatsError> {
    source_device_count_exact_native(topology, &[DeviceKind::Xpu], cancellation)
}

pub fn mlu_device_count_exact_native(
    topology: &MemoryTopology,
    cancellation: &CancellationToken,
) -> Result<u32, NativeMemoryStatsError> {
    source_device_count_exact_native(topology, &[DeviceKind::Mlu], cancellation)
}

pub fn npu_device_count_exact_native(
    topology: &MemoryTopology,
    cancellation: &CancellationToken,
) -> Result<u32, NativeMemoryStatsError> {
    source_device_count_exact_native(topology, &[DeviceKind::Npu], cancellation)
}

pub fn npu_is_available_exact_native(
    topology: &MemoryTopology,
    cancellation: &CancellationToken,
) -> Result<bool, NativeMemoryStatsError> {
    Ok(npu_device_count_exact_native(topology, cancellation)? > 0)
}

pub fn xpu_is_available_exact_native(
    topology: &MemoryTopology,
    cancellation: &CancellationToken,
) -> Result<bool, NativeMemoryStatsError> {
    cancellation.check()?;
    Ok(xpu_device_count_exact_native(topology, cancellation)? > 0)
}

pub fn mlu_is_available_exact_native(
    topology: &MemoryTopology,
    cancellation: &CancellationToken,
) -> Result<bool, NativeMemoryStatsError> {
    Ok(mlu_device_count_exact_native(topology, cancellation)? > 0)
}

pub fn mlu_mem_get_info_exact_native(
    capabilities: &BackendCapabilityMatrix,
    topology: &MemoryTopology,
    snapshot: &MemoryAccountingSnapshot,
    cancellation: &CancellationToken,
) -> Result<(u64, u64), NativeMemoryStatsError> {
    source_mem_get_info_exact_native(
        capabilities,
        topology,
        snapshot,
        &[DeviceKind::Mlu],
        cancellation,
    )
}

pub fn npu_mem_get_info_exact_native(
    capabilities: &BackendCapabilityMatrix,
    topology: &MemoryTopology,
    snapshot: &MemoryAccountingSnapshot,
    cancellation: &CancellationToken,
) -> Result<(u64, u64), NativeMemoryStatsError> {
    cancellation.check()?;
    source_mem_get_info_exact_native(
        capabilities,
        topology,
        snapshot,
        &[DeviceKind::Npu],
        cancellation,
    )
}

pub fn cuda_mem_get_info_exact_native(
    capabilities: &BackendCapabilityMatrix,
    topology: &MemoryTopology,
    snapshot: &MemoryAccountingSnapshot,
    cancellation: &CancellationToken,
) -> Result<(u64, u64), NativeMemoryStatsError> {
    cancellation.check()?;
    source_mem_get_info_exact_native(
        capabilities,
        topology,
        snapshot,
        &[DeviceKind::Cuda, DeviceKind::Rocm],
        cancellation,
    )
}

fn source_mem_get_info_exact_native(
    capabilities: &BackendCapabilityMatrix,
    topology: &MemoryTopology,
    snapshot: &MemoryAccountingSnapshot,
    expected_kinds: &[DeviceKind],
    cancellation: &CancellationToken,
) -> Result<(u64, u64), NativeMemoryStatsError> {
    cancellation.check()?;
    let device = capabilities.device();
    if !expected_kinds.contains(&device.kind()) {
        return Err(NativeMemoryStatsError::UnsupportedDevice(device));
    }
    let mut matching_budgets = topology
        .devices()
        .iter()
        .filter(|budget| budget.device == device);
    let budget = matching_budgets
        .next()
        .ok_or(MemoryPolicyError::UnknownDevice(device))?;
    if matching_budgets.next().is_some() {
        return Err(MemoryPolicyError::DuplicateDevice(device).into());
    }
    let active = snapshot
        .source_memory_stats(device)?
        .get("active_bytes.all.current")
        .copied()
        .ok_or(NativeMemoryStatsError::MissingStat(
            "active_bytes.all.current",
        ))?;
    let free = budget.capacity_bytes.checked_sub(active).ok_or(
        MemoryPolicyError::LocationOutOfMemory {
            location: MemoryLocation::Device(device),
            required_bytes: active,
            capacity_bytes: budget.capacity_bytes,
        },
    )?;
    cancellation.check()?;
    Ok((free, budget.capacity_bytes))
}

pub fn cuda_device_count_exact_native(
    topology: &MemoryTopology,
    cancellation: &CancellationToken,
) -> Result<u32, NativeMemoryStatsError> {
    cancellation.check()?;
    source_device_count_exact_native(
        topology,
        &[DeviceKind::Cuda, DeviceKind::Rocm],
        cancellation,
    )
}

pub fn cuda_is_available_exact_native(
    topology: &MemoryTopology,
    cancellation: &CancellationToken,
) -> Result<bool, NativeMemoryStatsError> {
    cancellation.check()?;
    Ok(cuda_device_count_exact_native(topology, cancellation)? > 0)
}

fn source_device_count_exact_native(
    topology: &MemoryTopology,
    kinds: &[DeviceKind],
    cancellation: &CancellationToken,
) -> Result<u32, NativeMemoryStatsError> {
    cancellation.check()?;
    let count = topology
        .devices()
        .iter()
        .filter(|budget| kinds.contains(&budget.device.kind()))
        .count();
    let count = u32::try_from(count).map_err(|_| NativeMemoryStatsError::DeviceCountOverflow)?;
    cancellation.check()?;
    Ok(count)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeCacheRelease {
    pub allocator_bytes_released: u64,
    pub accounting_actions: Vec<EvictionAction>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NativeIpcCollection {
    pub bytes_released: u64,
    pub accounting_actions: Vec<EvictionAction>,
}

pub fn cuda_ipc_collect_exact_native(
    inventory: &mut MemoryPlacementInventory,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<NativeIpcCollection, NativeMemoryStatsError> {
    cancellation.check()?;
    if !matches!(device.kind(), DeviceKind::Cuda | DeviceKind::Rocm) {
        return Err(NativeMemoryStatsError::UnsupportedDevice(device));
    }
    let (proposed, accounting_actions) = inventory.stage_ipc_collection(device, cancellation)?;
    let bytes_released = accounting_actions.iter().try_fold(0_u64, |total, action| {
        total
            .checked_add(action.bytes_reclaimed)
            .ok_or(MemoryPolicyError::AccountingOverflow(
                "IPC collection bytes",
            ))
    })?;
    cancellation.check()?;
    *inventory = proposed;
    Ok(NativeIpcCollection {
        bytes_released,
        accounting_actions,
    })
}

pub fn mps_empty_cache_exact_native(
    backend: &dyn CachedAllocationOwner,
    inventory: &mut MemoryPlacementInventory,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<NativeCacheRelease, NativeMemoryStatsError> {
    cancellation.check()?;
    device_empty_cache_exact_native(backend, inventory, device, DeviceKind::Metal, cancellation)
}

pub fn cuda_empty_cache_exact_native(
    backend: &dyn CachedAllocationOwner,
    inventory: &mut MemoryPlacementInventory,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<NativeCacheRelease, NativeMemoryStatsError> {
    cancellation.check()?;
    if !matches!(device.kind(), DeviceKind::Cuda | DeviceKind::Rocm) {
        return Err(NativeMemoryStatsError::UnsupportedDevice(device));
    }
    device_empty_cache_exact_native(backend, inventory, device, device.kind(), cancellation)
}

pub fn xpu_empty_cache_exact_native(
    backend: &dyn CachedAllocationOwner,
    inventory: &mut MemoryPlacementInventory,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<NativeCacheRelease, NativeMemoryStatsError> {
    cancellation.check()?;
    device_empty_cache_exact_native(backend, inventory, device, DeviceKind::Xpu, cancellation)
}

pub fn npu_empty_cache_exact_native(
    backend: &dyn CachedAllocationOwner,
    inventory: &mut MemoryPlacementInventory,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<NativeCacheRelease, NativeMemoryStatsError> {
    cancellation.check()?;
    device_empty_cache_exact_native(backend, inventory, device, DeviceKind::Npu, cancellation)
}

pub fn mlu_empty_cache_exact_native(
    backend: &dyn CachedAllocationOwner,
    inventory: &mut MemoryPlacementInventory,
    device: DeviceId,
    cancellation: &CancellationToken,
) -> Result<NativeCacheRelease, NativeMemoryStatsError> {
    device_empty_cache_exact_native(backend, inventory, device, DeviceKind::Mlu, cancellation)
}

fn device_empty_cache_exact_native(
    backend: &dyn CachedAllocationOwner,
    inventory: &mut MemoryPlacementInventory,
    device: DeviceId,
    expected_kind: DeviceKind,
    cancellation: &CancellationToken,
) -> Result<NativeCacheRelease, NativeMemoryStatsError> {
    if device.kind() != expected_kind || backend.cache_device() != device {
        return Err(NativeMemoryStatsError::UnsupportedDevice(device));
    }
    let (proposed, accounting_actions) =
        inventory.stage_reclaimable_cache_release(device, cancellation)?;
    let allocator_bytes_released = backend.release_cached_allocations(cancellation)?;
    cancellation.check()?;
    *inventory = proposed;
    Ok(NativeCacheRelease {
        allocator_bytes_released,
        accounting_actions,
    })
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum NativeMemoryStatsError {
    #[error(transparent)]
    Policy(#[from] MemoryPolicyError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error("native memory statistics were cancelled")]
    Cancelled,
    #[error("native source memory diagnostics do not support device {0:?}")]
    UnsupportedDevice(DeviceId),
    #[error("native source device count does not fit u32")]
    DeviceCountOverflow,
    #[error("canonical memory accounting omitted required source key {0}")]
    MissingStat(&'static str),
}

impl From<sim_types::CancellationError> for NativeMemoryStatsError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

#[derive(Clone, Debug)]
pub struct MemoryPlacementInventory {
    topology: MemoryTopology,
    resources: Vec<MemoryResource>,
    lost_devices: Vec<DeviceId>,
    clock: u64,
}

impl MemoryPlacementInventory {
    pub fn new(topology: MemoryTopology) -> Self {
        Self {
            topology,
            resources: Vec::new(),
            lost_devices: Vec::new(),
            clock: 0,
        }
    }

    pub fn topology(&self) -> &MemoryTopology {
        &self.topology
    }

    pub fn resources(&self) -> &[MemoryResource] {
        &self.resources
    }

    pub fn register(&mut self, mut resource: MemoryResource) -> Result<(), MemoryPolicyError> {
        if self.resources.iter().any(|value| value.id == resource.id) {
            return Err(MemoryPolicyError::DuplicateResource(resource.id));
        }
        self.ensure_location_capacity(resource.location, resource.bytes, None)?;
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or(MemoryPolicyError::AccountingOverflow("usage clock"))?;
        resource.last_used = self.clock;
        self.resources.push(resource);
        Ok(())
    }

    pub fn touch(&mut self, id: MemoryResourceId) -> Result<(), MemoryPolicyError> {
        self.clock = self
            .clock
            .checked_add(1)
            .ok_or(MemoryPolicyError::AccountingOverflow("usage clock"))?;
        let clock = self.clock;
        self.resource_mut(id)?.last_used = clock;
        Ok(())
    }

    pub fn attach_fence(
        &mut self,
        id: MemoryResourceId,
        fence: MemoryFenceId,
    ) -> Result<(), MemoryPolicyError> {
        let resource = self.resource_mut(id)?;
        if let Some(existing) = resource.fence {
            return Err(MemoryPolicyError::ResourceAlreadyFenced {
                resource: id,
                fence: existing,
            });
        }
        resource.fence = Some(fence);
        resource.in_flight = true;
        Ok(())
    }

    pub fn complete_fence(&mut self, fence: MemoryFenceId) -> usize {
        let mut completed = 0;
        for resource in &mut self.resources {
            if resource.fence == Some(fence) {
                resource.fence = None;
                resource.in_flight = false;
                completed += 1;
            }
        }
        completed
    }

    pub fn reclaim_device(
        &mut self,
        device: DeviceId,
        bytes: u64,
        mode: EffectiveMemoryMode,
    ) -> Result<Vec<EvictionAction>, MemoryPolicyError> {
        let mut proposed = self.clone();
        let actions = proposed.reclaim_device_inner(device, bytes, mode)?;
        *self = proposed;
        Ok(actions)
    }

    fn stage_reclaimable_cache_release(
        &self,
        device: DeviceId,
        cancellation: &CancellationToken,
    ) -> Result<(Self, Vec<EvictionAction>), NativeMemoryStatsError> {
        cancellation.check()?;
        if !contains_device(&self.topology.devices, device) {
            return Err(MemoryPolicyError::UnknownDevice(device).into());
        }
        let mut removed = BTreeSet::new();
        let mut actions = Vec::new();
        for resource in &self.resources {
            cancellation.check()?;
            if resource.location == MemoryLocation::Device(device)
                && resource.kind == MemoryResourceKind::Cache
                && !resource.protected()
            {
                removed.insert(resource.id);
                actions.push(EvictionAction {
                    resource_id: resource.id,
                    resource_kind: resource.kind,
                    bytes_reclaimed: resource.bytes,
                    action: EvictionActionKind::Dropped,
                });
            }
        }
        cancellation.check()?;
        let mut proposed = self.clone();
        proposed
            .resources
            .retain(|resource| !removed.contains(&resource.id));
        Ok((proposed, actions))
    }

    fn stage_ipc_collection(
        &self,
        device: DeviceId,
        cancellation: &CancellationToken,
    ) -> Result<(Self, Vec<EvictionAction>), NativeMemoryStatsError> {
        cancellation.check()?;
        if !contains_device(&self.topology.devices, device) {
            return Err(MemoryPolicyError::UnknownDevice(device).into());
        }
        let mut removed = BTreeSet::new();
        let mut actions = Vec::new();
        for resource in &self.resources {
            cancellation.check()?;
            if resource.location == MemoryLocation::Device(device)
                && resource.kind == MemoryResourceKind::IpcExport
                && !resource.protected()
            {
                removed.insert(resource.id);
                actions.push(EvictionAction {
                    resource_id: resource.id,
                    resource_kind: resource.kind,
                    bytes_reclaimed: resource.bytes,
                    action: EvictionActionKind::Dropped,
                });
            }
        }
        cancellation.check()?;
        let mut proposed = self.clone();
        proposed
            .resources
            .retain(|resource| !removed.contains(&resource.id));
        Ok((proposed, actions))
    }

    fn reclaim_device_inner(
        &mut self,
        device: DeviceId,
        bytes: u64,
        mode: EffectiveMemoryMode,
    ) -> Result<Vec<EvictionAction>, MemoryPolicyError> {
        if !contains_device(&self.topology.devices, device) {
            return Err(MemoryPolicyError::UnknownDevice(device));
        }
        if bytes == 0 {
            return Ok(Vec::new());
        }
        let mut candidates: Vec<_> = self
            .resources
            .iter()
            .enumerate()
            .filter_map(|(index, resource)| {
                if resource.location != MemoryLocation::Device(device) || resource.protected() {
                    None
                } else {
                    eviction_rank(resource)
                        .map(|rank| (rank, resource.last_used, resource.id, index))
                }
            })
            .collect();
        candidates.sort_by_key(|(rank, last_used, id, _)| (*rank, *last_used, *id));

        let mut actions = Vec::new();
        let mut reclaimed = 0_u64;
        let mut removed = BTreeSet::new();
        for (_, _, _, index) in candidates {
            if reclaimed >= bytes {
                break;
            }
            let resource = self
                .resources
                .get(index)
                .cloned()
                .ok_or(MemoryPolicyError::AccountingOverflow("eviction index"))?;
            let action = match resource.kind {
                MemoryResourceKind::Preview
                | MemoryResourceKind::DecodedMedia
                | MemoryResourceKind::Cache => {
                    removed.insert(resource.id);
                    EvictionActionKind::Dropped
                }
                MemoryResourceKind::Activations if resource.offloadable => {
                    let target = host_target(mode, false)?;
                    self.ensure_location_capacity(target, resource.bytes, Some(resource.id))?;
                    self.resource_mut(resource.id)?.location = target;
                    action_for_location(target)?
                }
                MemoryResourceKind::ModelPage if resource.offloadable && !resource.active => {
                    let target = host_target(mode, true)?;
                    self.ensure_location_capacity(target, resource.bytes, Some(resource.id))?;
                    self.resource_mut(resource.id)?.location = target;
                    action_for_location(target)?
                }
                MemoryResourceKind::IpcExport => continue,
                _ => continue,
            };
            reclaimed = reclaimed
                .checked_add(resource.bytes)
                .ok_or(MemoryPolicyError::AccountingOverflow("reclaimed bytes"))?;
            actions.push(EvictionAction {
                resource_id: resource.id,
                resource_kind: resource.kind,
                bytes_reclaimed: resource.bytes,
                action,
            });
        }
        if !removed.is_empty() {
            self.resources
                .retain(|resource| !removed.contains(&resource.id));
        }
        if reclaimed < bytes {
            return Err(MemoryPolicyError::InsufficientReclaim {
                requested_bytes: bytes,
                reclaimed_bytes: reclaimed,
            });
        }
        Ok(actions)
    }

    pub fn discard_attempt_resources(&mut self) -> Result<u64, MemoryPolicyError> {
        let mut released = 0_u64;
        for resource in &self.resources {
            if !resource.durable && resource.fence.is_none() {
                released = released.checked_add(resource.bytes).ok_or(
                    MemoryPolicyError::AccountingOverflow("released attempt bytes"),
                )?;
            }
        }
        self.resources
            .retain(|resource| resource.durable || resource.fence.is_some());
        Ok(released)
    }

    pub fn invalidate_device(&mut self, device: DeviceId) -> Result<u64, MemoryPolicyError> {
        if !contains_device(&self.topology.devices, device) {
            return Err(MemoryPolicyError::UnknownDevice(device));
        }
        if self.lost_devices.contains(&device) {
            return Err(MemoryPolicyError::DeviceAlreadyLost(device));
        }
        let mut invalidated = 0_u64;
        for resource in &self.resources {
            if resource.location == MemoryLocation::Device(device) {
                invalidated = invalidated.checked_add(resource.bytes).ok_or(
                    MemoryPolicyError::AccountingOverflow("invalidated device bytes"),
                )?;
            }
        }
        self.resources
            .retain(|resource| resource.location != MemoryLocation::Device(device));
        self.lost_devices.push(device);
        Ok(invalidated)
    }

    pub fn snapshot(&self) -> Result<MemoryAccountingSnapshot, MemoryPolicyError> {
        let mut device_bytes = Vec::with_capacity(self.topology.devices.len());
        for budget in &self.topology.devices {
            device_bytes.push((
                budget.device,
                self.bytes_at(MemoryLocation::Device(budget.device))?,
            ));
        }
        let durable_bytes = checked_sum(
            self.resources
                .iter()
                .filter(|resource| resource.durable)
                .map(|resource| resource.bytes),
            "durable bytes",
        )?
        .checked_add(checked_sum(
            self.topology
                .devices
                .iter()
                .filter(|budget| !self.lost_devices.contains(&budget.device))
                .map(|budget| budget.durable_baseline_bytes),
            "topology durable baseline",
        )?)
        .ok_or(MemoryPolicyError::AccountingOverflow("total durable bytes"))?;
        let fenced_bytes = checked_sum(
            self.resources
                .iter()
                .filter(|resource| resource.fence.is_some())
                .map(|resource| resource.bytes),
            "fenced bytes",
        )?;
        Ok(MemoryAccountingSnapshot {
            device_bytes,
            host_pinned_bytes: self.bytes_at(MemoryLocation::HostPinned)?,
            host_pageable_bytes: self.bytes_at(MemoryLocation::HostPageable)?,
            mapped_bytes: self.bytes_at(MemoryLocation::MemoryMapped)?,
            durable_bytes,
            fenced_bytes,
            resource_count: self.resources.len(),
        })
    }

    fn resource_mut(
        &mut self,
        id: MemoryResourceId,
    ) -> Result<&mut MemoryResource, MemoryPolicyError> {
        self.resources
            .iter_mut()
            .find(|resource| resource.id == id)
            .ok_or(MemoryPolicyError::UnknownResource(id))
    }

    fn bytes_at(&self, location: MemoryLocation) -> Result<u64, MemoryPolicyError> {
        let resource_bytes = checked_sum(
            self.resources
                .iter()
                .filter(|resource| resource.location == location)
                .map(|resource| resource.bytes),
            "location bytes",
        )?;
        let baseline_bytes = match location {
            MemoryLocation::Device(device) if self.lost_devices.contains(&device) => 0,
            MemoryLocation::Device(device) => self
                .topology
                .devices
                .iter()
                .find(|budget| budget.device == device)
                .map(|budget| budget.durable_baseline_bytes)
                .ok_or(MemoryPolicyError::UnknownDevice(device))?,
            MemoryLocation::HostPinned
            | MemoryLocation::HostPageable
            | MemoryLocation::MemoryMapped => 0,
        };
        resource_bytes
            .checked_add(baseline_bytes)
            .ok_or(MemoryPolicyError::AccountingOverflow(
                "location bytes including baseline",
            ))
    }

    fn ensure_location_capacity(
        &self,
        location: MemoryLocation,
        additional_bytes: u64,
        excluding: Option<MemoryResourceId>,
    ) -> Result<(), MemoryPolicyError> {
        if location == MemoryLocation::MemoryMapped {
            return Ok(());
        }
        if let MemoryLocation::Device(device) = location
            && self.lost_devices.contains(&device)
        {
            return Err(MemoryPolicyError::DeviceLost(device));
        }
        let resource_bytes = checked_sum(
            self.resources
                .iter()
                .filter(|resource| resource.location == location && Some(resource.id) != excluding)
                .map(|resource| resource.bytes),
            "location capacity",
        )?;
        let baseline_bytes = match location {
            MemoryLocation::Device(device) => self
                .topology
                .devices
                .iter()
                .find(|budget| budget.device == device)
                .map(|budget| budget.durable_baseline_bytes)
                .ok_or(MemoryPolicyError::UnknownDevice(device))?,
            MemoryLocation::HostPinned
            | MemoryLocation::HostPageable
            | MemoryLocation::MemoryMapped => 0,
        };
        let current = resource_bytes.checked_add(baseline_bytes).ok_or(
            MemoryPolicyError::AccountingOverflow("location baseline capacity"),
        )?;
        let required = current
            .checked_add(additional_bytes)
            .ok_or(MemoryPolicyError::AccountingOverflow("location capacity"))?;
        let capacity = match location {
            MemoryLocation::Device(device) => self
                .topology
                .devices
                .iter()
                .find(|budget| budget.device == device)
                .map(|budget| budget.capacity_bytes)
                .ok_or(MemoryPolicyError::UnknownDevice(device))?,
            MemoryLocation::HostPinned => self.topology.host_pinned_capacity_bytes,
            MemoryLocation::HostPageable => self.topology.host_pageable_capacity_bytes,
            MemoryLocation::MemoryMapped => u64::MAX,
        };
        if required > capacity {
            return Err(MemoryPolicyError::LocationOutOfMemory {
                location,
                required_bytes: required,
                capacity_bytes: capacity,
            });
        }
        Ok(())
    }
}

fn checked_sum(
    values: impl IntoIterator<Item = u64>,
    context: &'static str,
) -> Result<u64, MemoryPolicyError> {
    values.into_iter().try_fold(0_u64, |total, value| {
        total
            .checked_add(value)
            .ok_or(MemoryPolicyError::AccountingOverflow(context))
    })
}

fn eviction_rank(resource: &MemoryResource) -> Option<u8> {
    match resource.kind {
        MemoryResourceKind::Preview => Some(0),
        MemoryResourceKind::DecodedMedia => Some(1),
        MemoryResourceKind::Cache if !resource.pinned => Some(2),
        MemoryResourceKind::Activations if resource.offloadable => Some(3),
        MemoryResourceKind::ModelPage if resource.offloadable && !resource.active => Some(4),
        _ => None,
    }
}

fn host_target(
    mode: EffectiveMemoryMode,
    model_page: bool,
) -> Result<MemoryLocation, MemoryPolicyError> {
    match mode.offload_target {
        OffloadTarget::HostPinned => Ok(MemoryLocation::HostPinned),
        OffloadTarget::HostPageable => Ok(MemoryLocation::HostPageable),
        OffloadTarget::MemoryMapped if model_page => Ok(MemoryLocation::MemoryMapped),
        OffloadTarget::MemoryMapped => Ok(MemoryLocation::HostPageable),
        OffloadTarget::None => Err(MemoryPolicyError::OffloadDisabled),
    }
}

fn action_for_location(location: MemoryLocation) -> Result<EvictionActionKind, MemoryPolicyError> {
    match location {
        MemoryLocation::HostPinned => Ok(EvictionActionKind::OffloadedToPinnedHost),
        MemoryLocation::HostPageable => Ok(EvictionActionKind::OffloadedToPageableHost),
        MemoryLocation::MemoryMapped => Ok(EvictionActionKind::PagedOutToMapping),
        MemoryLocation::Device(_) => Err(MemoryPolicyError::InvalidOffloadTarget(location)),
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PlacementGroup {
    pub group_id: u64,
    pub bytes: u64,
    pub preferred_devices: Vec<DeviceId>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferRoute {
    None,
    PeerCopy,
    HostPinnedStaging,
    HostPageableStaging,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct EffectivePlacement {
    pub group_id: u64,
    pub device: DeviceId,
    pub transfer_from_previous: TransferRoute,
    pub transfer_bytes: u64,
}

pub struct MemoryPlacementPlanner;

impl MemoryPlacementPlanner {
    pub fn place(
        inventory: &MemoryPlacementInventory,
        groups: &[PlacementGroup],
        mode: EffectiveMemoryMode,
    ) -> Result<Vec<EffectivePlacement>, MemoryPolicyError> {
        let snapshot = inventory.snapshot()?;
        let mut available = Vec::with_capacity(inventory.topology.devices.len());
        for budget in &inventory.topology.devices {
            if inventory.lost_devices.contains(&budget.device) {
                continue;
            }
            let used = snapshot
                .device_bytes
                .iter()
                .find(|(device, _)| *device == budget.device)
                .map(|(_, bytes)| *bytes)
                .ok_or(MemoryPolicyError::UnknownDevice(budget.device))?;
            available.push((
                budget.device,
                budget.capacity_bytes.checked_sub(used).ok_or(
                    MemoryPolicyError::AccountingOverflow("placement available bytes"),
                )?,
            ));
        }
        let mut placements = Vec::with_capacity(groups.len());
        let mut previous = None;
        let mut seen = BTreeSet::new();
        for group in groups {
            if group.group_id == 0 {
                return Err(MemoryPolicyError::ZeroIdentifier("placement group"));
            }
            if !seen.insert(group.group_id) {
                return Err(MemoryPolicyError::DuplicatePlacementGroup(group.group_id));
            }
            if group.bytes == 0 {
                return Err(MemoryPolicyError::ZeroPlacementBytes(group.group_id));
            }
            let selected = select_device(&available, group).ok_or(
                MemoryPolicyError::PlacementUnavailable {
                    group_id: group.group_id,
                    bytes: group.bytes,
                },
            )?;
            let entry = available
                .iter_mut()
                .find(|(device, _)| *device == selected)
                .ok_or(MemoryPolicyError::UnknownDevice(selected))?;
            entry.1 =
                entry
                    .1
                    .checked_sub(group.bytes)
                    .ok_or(MemoryPolicyError::AccountingOverflow(
                        "placement reservation",
                    ))?;
            let transfer_from_previous = match previous {
                None => TransferRoute::None,
                Some(source) if source == selected => TransferRoute::None,
                Some(source) if inventory.topology.supports_peer_copy(source, selected) => {
                    TransferRoute::PeerCopy
                }
                Some(_) if mode.pinned_staging => TransferRoute::HostPinnedStaging,
                Some(_) => TransferRoute::HostPageableStaging,
            };
            let transfer_bytes = match transfer_from_previous {
                TransferRoute::None => 0,
                TransferRoute::PeerCopy => group.bytes,
                TransferRoute::HostPinnedStaging => {
                    inventory.ensure_location_capacity(
                        MemoryLocation::HostPinned,
                        group.bytes,
                        None,
                    )?;
                    group.bytes
                }
                TransferRoute::HostPageableStaging => {
                    inventory.ensure_location_capacity(
                        MemoryLocation::HostPageable,
                        group.bytes,
                        None,
                    )?;
                    group.bytes
                }
            };
            placements.push(EffectivePlacement {
                group_id: group.group_id,
                device: selected,
                transfer_from_previous,
                transfer_bytes,
            });
            previous = Some(selected);
        }
        Ok(placements)
    }
}

fn select_device(available: &[(DeviceId, u64)], group: &PlacementGroup) -> Option<DeviceId> {
    for preferred in &group.preferred_devices {
        if available
            .iter()
            .any(|(device, bytes)| device == preferred && *bytes >= group.bytes)
        {
            return Some(*preferred);
        }
    }
    available
        .iter()
        .enumerate()
        .filter(|(_, (_, bytes))| *bytes >= group.bytes)
        .max_by_key(|(index, (_, bytes))| (*bytes, std::cmp::Reverse(*index)))
        .map(|(_, (device, _))| *device)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryRecoveryPhase {
    Initial,
    PressureReplan,
    ReducedWorkspace,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MemoryPlanRecord {
    pub phase: MemoryRecoveryPhase,
    pub committed_target_bytes: u64,
    pub workspace_bytes: u64,
    pub failure: Option<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptMemoryState {
    Planned,
    Running,
    Cancelling,
    Cancelled,
    Completed,
    Failed,
    DeviceLost,
}

#[derive(Debug)]
pub struct PlannedWorkspaceAuthorization {
    workspace_bytes: u64,
}

impl PlannedWorkspaceAuthorization {
    pub const fn bytes(&self) -> u64 {
        self.workspace_bytes
    }
}

#[derive(Debug)]
pub struct AttemptMemoryController {
    capacity_bytes: u64,
    durable_baseline_bytes: u64,
    request: MemoryPlanRequest,
    current_plan: MemoryPlan,
    retry_tracker: MemoryRetryTracker,
    records: Vec<MemoryPlanRecord>,
    pending_fences: BTreeSet<MemoryFenceId>,
    state: AttemptMemoryState,
    workspace_authorization_issued: bool,
}

impl AttemptMemoryController {
    pub fn new(
        capacity_bytes: u64,
        durable_baseline_bytes: u64,
        request: MemoryPlanRequest,
    ) -> Result<Self, MemoryPolicyError> {
        let current_plan = MemoryPlanner::plan(capacity_bytes, durable_baseline_bytes, request)?;
        let target = current_plan.committed_target_bytes;
        Ok(Self {
            capacity_bytes,
            durable_baseline_bytes,
            request,
            current_plan,
            retry_tracker: MemoryRetryTracker::new(target),
            records: vec![MemoryPlanRecord {
                phase: MemoryRecoveryPhase::Initial,
                committed_target_bytes: target,
                workspace_bytes: request.workspace_bytes,
                failure: None,
            }],
            pending_fences: BTreeSet::new(),
            state: AttemptMemoryState::Planned,
            workspace_authorization_issued: false,
        })
    }

    pub fn current_plan(&self) -> &MemoryPlan {
        &self.current_plan
    }

    pub fn workspace_authorization_bytes(&self) -> u64 {
        self.current_plan
            .reservations
            .iter()
            .find(|reservation| reservation.kind == MemoryReservationKind::Workspace)
            .map_or(0, |reservation| reservation.bytes)
    }

    pub fn issue_workspace_authorization(
        &mut self,
    ) -> Result<PlannedWorkspaceAuthorization, MemoryPolicyError> {
        self.require_state(AttemptMemoryState::Planned, "issue workspace authorization")?;
        if self.workspace_authorization_issued {
            return Err(MemoryPolicyError::WorkspaceAuthorizationAlreadyIssued);
        }
        self.workspace_authorization_issued = true;
        Ok(PlannedWorkspaceAuthorization {
            workspace_bytes: self.workspace_authorization_bytes(),
        })
    }

    pub fn records(&self) -> &[MemoryPlanRecord] {
        &self.records
    }

    pub const fn state(&self) -> AttemptMemoryState {
        self.state
    }

    pub fn begin(&mut self) -> Result<(), MemoryPolicyError> {
        self.require_state(AttemptMemoryState::Planned, "begin")?;
        self.state = AttemptMemoryState::Running;
        Ok(())
    }

    pub fn pressure_replan(
        &mut self,
        failure: impl Into<String>,
        reclaimed_durable_bytes: u64,
    ) -> Result<&MemoryPlan, MemoryPolicyError> {
        self.require_recoverable("pressure replan")?;
        if self.retry_tracker.retries_used() != 0 {
            return Err(MemoryPolicyError::RecoveryOrder);
        }
        if reclaimed_durable_bytes == 0 {
            return Err(MemoryPolicyError::RecoveryDidNotReclaim);
        }
        let baseline = self
            .durable_baseline_bytes
            .checked_sub(reclaimed_durable_bytes)
            .ok_or(MemoryPolicyError::ReclaimExceedsBaseline {
                reclaimed_bytes: reclaimed_durable_bytes,
                baseline_bytes: self.durable_baseline_bytes,
            })?;
        let plan = MemoryPlanner::plan(self.capacity_bytes, baseline, self.request)?;
        self.retry_tracker
            .accept_lower_target(plan.committed_target_bytes)?;
        self.record_failure(failure.into());
        self.records.push(MemoryPlanRecord {
            phase: MemoryRecoveryPhase::PressureReplan,
            committed_target_bytes: plan.committed_target_bytes,
            workspace_bytes: self.request.workspace_bytes,
            failure: None,
        });
        self.durable_baseline_bytes = baseline;
        self.current_plan = plan;
        Ok(&self.current_plan)
    }

    pub fn reduced_workspace_retry(
        &mut self,
        failure: impl Into<String>,
        reduced_workspace_bytes: u64,
    ) -> Result<&MemoryPlan, MemoryPolicyError> {
        self.require_recoverable("reduced workspace retry")?;
        if self.retry_tracker.retries_used() != 1 {
            return Err(MemoryPolicyError::RecoveryOrder);
        }
        if reduced_workspace_bytes >= self.request.workspace_bytes {
            return Err(MemoryPolicyError::WorkspaceDidNotReduce {
                previous_bytes: self.request.workspace_bytes,
                proposed_bytes: reduced_workspace_bytes,
            });
        }
        let mut request = self.request;
        request.workspace_bytes = reduced_workspace_bytes;
        let plan = MemoryPlanner::plan(self.capacity_bytes, self.durable_baseline_bytes, request)?;
        self.retry_tracker
            .accept_lower_target(plan.committed_target_bytes)?;
        self.record_failure(failure.into());
        self.records.push(MemoryPlanRecord {
            phase: MemoryRecoveryPhase::ReducedWorkspace,
            committed_target_bytes: plan.committed_target_bytes,
            workspace_bytes: reduced_workspace_bytes,
            failure: None,
        });
        self.request = request;
        self.current_plan = plan;
        Ok(&self.current_plan)
    }

    pub fn terminal_oom(&mut self, failure: impl Into<String>) -> MemoryPolicyError {
        self.record_failure(failure.into());
        self.state = AttemptMemoryState::Failed;
        MemoryPolicyError::TerminalOutOfMemory {
            attempts: self.records.len(),
            last_target_bytes: self.current_plan.committed_target_bytes,
        }
    }

    pub fn register_fence(&mut self, fence: MemoryFenceId) -> Result<(), MemoryPolicyError> {
        if !matches!(
            self.state,
            AttemptMemoryState::Running | AttemptMemoryState::Cancelling
        ) {
            return Err(MemoryPolicyError::InvalidAttemptTransition {
                from: self.state,
                operation: "register fence",
            });
        }
        if !self.pending_fences.insert(fence) {
            return Err(MemoryPolicyError::DuplicateFence(fence));
        }
        Ok(())
    }

    pub fn cancel(&mut self) -> Result<AttemptMemoryState, MemoryPolicyError> {
        if !matches!(
            self.state,
            AttemptMemoryState::Planned | AttemptMemoryState::Running
        ) {
            return Err(MemoryPolicyError::InvalidAttemptTransition {
                from: self.state,
                operation: "cancel",
            });
        }
        self.state = if self.pending_fences.is_empty() {
            AttemptMemoryState::Cancelled
        } else {
            AttemptMemoryState::Cancelling
        };
        Ok(self.state)
    }

    pub fn complete_fence(
        &mut self,
        fence: MemoryFenceId,
    ) -> Result<AttemptMemoryState, MemoryPolicyError> {
        if !self.pending_fences.remove(&fence) {
            return Err(MemoryPolicyError::UnknownFence(fence));
        }
        if self.state == AttemptMemoryState::Cancelling && self.pending_fences.is_empty() {
            self.state = AttemptMemoryState::Cancelled;
        }
        Ok(self.state)
    }

    pub fn complete(&mut self) -> Result<(), MemoryPolicyError> {
        if self.state != AttemptMemoryState::Running || !self.pending_fences.is_empty() {
            return Err(MemoryPolicyError::InvalidAttemptTransition {
                from: self.state,
                operation: "complete",
            });
        }
        self.state = AttemptMemoryState::Completed;
        Ok(())
    }

    pub fn device_lost(&mut self) -> Result<(), MemoryPolicyError> {
        if matches!(
            self.state,
            AttemptMemoryState::Cancelled
                | AttemptMemoryState::Completed
                | AttemptMemoryState::Failed
                | AttemptMemoryState::DeviceLost
        ) {
            return Err(MemoryPolicyError::InvalidAttemptTransition {
                from: self.state,
                operation: "device loss",
            });
        }
        self.pending_fences.clear();
        self.state = AttemptMemoryState::DeviceLost;
        Ok(())
    }

    pub fn fail(&mut self) -> Result<(), MemoryPolicyError> {
        if matches!(
            self.state,
            AttemptMemoryState::Cancelled
                | AttemptMemoryState::Completed
                | AttemptMemoryState::Failed
                | AttemptMemoryState::DeviceLost
        ) {
            return Err(MemoryPolicyError::InvalidAttemptTransition {
                from: self.state,
                operation: "fail",
            });
        }
        self.pending_fences.clear();
        self.state = AttemptMemoryState::Failed;
        Ok(())
    }

    pub fn accepts_value(&self) -> bool {
        self.state == AttemptMemoryState::Running
    }

    fn require_state(
        &self,
        expected: AttemptMemoryState,
        operation: &'static str,
    ) -> Result<(), MemoryPolicyError> {
        if self.state != expected {
            return Err(MemoryPolicyError::InvalidAttemptTransition {
                from: self.state,
                operation,
            });
        }
        Ok(())
    }

    fn require_recoverable(&self, operation: &'static str) -> Result<(), MemoryPolicyError> {
        if self.workspace_authorization_issued {
            return Err(MemoryPolicyError::WorkspaceAuthorizationAlreadyIssued);
        }
        if !matches!(
            self.state,
            AttemptMemoryState::Planned | AttemptMemoryState::Running
        ) {
            return Err(MemoryPolicyError::InvalidAttemptTransition {
                from: self.state,
                operation,
            });
        }
        Ok(())
    }

    fn record_failure(&mut self, failure: String) {
        if let Some(record) = self.records.last_mut() {
            record.failure = Some(failure);
        }
    }
}

pub fn native_image_memory_request(
    input_asset_bytes: u64,
    node_count: u64,
    previews_enabled: bool,
) -> Result<MemoryPlanRequest, MemoryPolicyError> {
    native_image_memory_request_with_codec(input_asset_bytes, node_count, previews_enabled, 0)
}

pub fn native_image_memory_request_with_codec(
    input_asset_bytes: u64,
    node_count: u64,
    previews_enabled: bool,
    codec_bytes: u64,
) -> Result<MemoryPlanRequest, MemoryPolicyError> {
    let decoded_bytes =
        input_asset_bytes
            .checked_mul(4)
            .ok_or(MemoryPolicyError::AccountingOverflow(
                "decoded image estimate",
            ))?;
    let workspace_bytes = node_count
        .checked_mul(1024 * 1024)
        .and_then(|bytes| bytes.checked_add(decoded_bytes))
        .ok_or(MemoryPolicyError::AccountingOverflow(
            "native image workspace estimate",
        ))?;
    Ok(MemoryPlanRequest {
        workspace_bytes,
        activations_bytes: decoded_bytes,
        staging_bytes: input_asset_bytes,
        preview_bytes: if previews_enabled { decoded_bytes } else { 0 },
        output_bytes: decoded_bytes,
        codec_bytes,
        ..MemoryPlanRequest::default()
    })
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum MemoryPolicyError {
    #[error(transparent)]
    Plan(#[from] MemoryPlanError),
    #[error("{0} identifier must be non-zero")]
    ZeroIdentifier(&'static str),
    #[error("memory resource {0:?} must reserve non-zero bytes")]
    ZeroBytes(MemoryResourceId),
    #[error("memory resource {0:?} is duplicated")]
    DuplicateResource(MemoryResourceId),
    #[error("memory resource {0:?} is unknown")]
    UnknownResource(MemoryResourceId),
    #[error("memory resource {resource:?} already carries fence {fence:?}")]
    ResourceAlreadyFenced {
        resource: MemoryResourceId,
        fence: MemoryFenceId,
    },
    #[error("memory topology must contain at least one device")]
    NoDevices,
    #[error("memory topology contains duplicate device {0:?}")]
    DuplicateDevice(DeviceId),
    #[error("memory topology does not contain device {0:?}")]
    UnknownDevice(DeviceId),
    #[error("memory device {0:?} is lost and cannot accept placement")]
    DeviceLost(DeviceId),
    #[error("memory device {0:?} was already invalidated")]
    DeviceAlreadyLost(DeviceId),
    #[error("durable baseline {baseline_bytes} exceeds capacity {capacity_bytes} on {device:?}")]
    BaselineExceedsCapacity {
        device: DeviceId,
        baseline_bytes: u64,
        capacity_bytes: u64,
    },
    #[error("invalid peer link from {first:?} to {second:?}")]
    InvalidPeerLink { first: DeviceId, second: DeviceId },
    #[error("memory accounting overflowed while computing {0}")]
    AccountingOverflow(&'static str),
    #[error(
        "memory location {location:?} requires {required_bytes} bytes but has capacity {capacity_bytes}"
    )]
    LocationOutOfMemory {
        location: MemoryLocation,
        required_bytes: u64,
        capacity_bytes: u64,
    },
    #[error(
        "memory pressure requested {requested_bytes} bytes but only {reclaimed_bytes} bytes were reclaimable"
    )]
    InsufficientReclaim {
        requested_bytes: u64,
        reclaimed_bytes: u64,
    },
    #[error("selected memory mode disables offload")]
    OffloadDisabled,
    #[error("invalid offload target {0:?}")]
    InvalidOffloadTarget(MemoryLocation),
    #[error("memory mode {mode} is unsupported on {device:?}")]
    UnsupportedMode {
        mode: &'static str,
        device: DeviceId,
    },
    #[error("memory modes {first} and {second} conflict")]
    ConflictingModes {
        first: &'static str,
        second: &'static str,
    },
    #[error("placement group {0} is duplicated")]
    DuplicatePlacementGroup(u64),
    #[error("placement group {0} must reserve non-zero bytes")]
    ZeroPlacementBytes(u64),
    #[error("placement group {group_id} requiring {bytes} bytes has no eligible device")]
    PlacementUnavailable { group_id: u64, bytes: u64 },
    #[error("memory recovery phases must run as pressure replan then reduced workspace")]
    RecoveryOrder,
    #[error("memory pressure replan must reclaim durable bytes")]
    RecoveryDidNotReclaim,
    #[error(
        "memory pressure reclaimed {reclaimed_bytes} bytes from baseline {baseline_bytes} bytes"
    )]
    ReclaimExceedsBaseline {
        reclaimed_bytes: u64,
        baseline_bytes: u64,
    },
    #[error(
        "reduced workspace {proposed_bytes} must be lower than previous workspace {previous_bytes}"
    )]
    WorkspaceDidNotReduce {
        previous_bytes: u64,
        proposed_bytes: u64,
    },
    #[error("the attempt workspace authorization was already issued")]
    WorkspaceAuthorizationAlreadyIssued,
    #[error(
        "native OOM is terminal after {attempts} plans; last committed target was {last_target_bytes} bytes"
    )]
    TerminalOutOfMemory {
        attempts: usize,
        last_target_bytes: u64,
    },
    #[error("cannot {operation} while attempt memory state is {from:?}")]
    InvalidAttemptTransition {
        from: AttemptMemoryState,
        operation: &'static str,
    },
    #[error("memory fence {0:?} is duplicated")]
    DuplicateFence(MemoryFenceId),
    #[error("memory fence {0:?} is unknown")]
    UnknownFence(MemoryFenceId),
}
