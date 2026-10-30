use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
};

use sha2::{Digest, Sha256};
use sim_runtime::MemoryPolicy;
use sim_tensor::{
    BackendCapabilityMatrix, CachedAllocationOwner, CancellationToken, CpuBackend,
    CpuWorkspaceAuthority, DeviceId, TensorError,
};
use sim_types::DeviceKind;
use sim_worker::{
    AttemptMemoryController, AttemptMemoryState, CATALOG_MEMORY_MODES, DeviceMemoryBudget,
    EffectiveMemoryMode, EvictionActionKind, MemoryAccountingSnapshot, MemoryFenceId,
    MemoryLocation, MemoryModeCapabilities, MemoryModeRequest, MemoryPlacementInventory,
    MemoryPlacementPlanner, MemoryPolicyError, MemoryRecoveryPhase, MemoryResource,
    MemoryResourceId, MemoryResourceKind, MemoryTopology, ModelResidencyMode,
    NativeMemoryStatsError, OffloadGranularity, PlacementGroup, TransferRoute,
    cuda_device_count_exact_native, cuda_empty_cache_exact_native, cuda_ipc_collect_exact_native,
    cuda_is_available_exact_native, cuda_mem_get_info_exact_native, cuda_memory_stats_exact_native,
    cuda_memory_summary_exact_native, mlu_device_count_exact_native, mlu_empty_cache_exact_native,
    mlu_is_available_exact_native, mlu_mem_get_info_exact_native, mlu_memory_stats_exact_native,
    mps_empty_cache_exact_native, native_image_memory_request, npu_device_count_exact_native,
    npu_empty_cache_exact_native, npu_is_available_exact_native, npu_mem_get_info_exact_native,
    npu_memory_stats_exact_native, xpu_device_count_exact_native, xpu_empty_cache_exact_native,
    xpu_is_available_exact_native, xpu_memory_stats_exact_native,
};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

#[test]
fn residency_projects_patch_compute_policy_without_a_second_dtype_owner()
-> Result<(), Box<dyn Error>> {
    let normal = EffectiveMemoryMode::resolve(
        MemoryModeRequest::default(),
        MemoryModeCapabilities::from_backend(&CpuBackend::capability_matrix()),
    )?;
    let low_vram = EffectiveMemoryMode::resolve(
        MemoryModeRequest {
            residency: ModelResidencyMode::LowVram,
            ..MemoryModeRequest::default()
        },
        MemoryModeCapabilities::from_backend(&CpuBackend::capability_matrix()),
    )?;
    let no_vram = EffectiveMemoryMode::resolve(
        MemoryModeRequest {
            residency: ModelResidencyMode::NoVram,
            ..MemoryModeRequest::default()
        },
        MemoryModeCapabilities::from_backend(&CpuBackend::capability_matrix()),
    )?;
    assert!(!normal.patch_uses_weight_dtype());
    assert!(low_vram.patch_uses_weight_dtype());
    assert!(no_vram.patch_uses_weight_dtype());
    Ok(())
}

struct FixtureCacheOwner {
    device: DeviceId,
    fail: bool,
    released: AtomicBool,
}

struct CancellingCacheOwner {
    device: DeviceId,
}

#[test]
fn task_62_cuda_empty_cache_reuses_the_canonical_cache_transaction() -> Result<(), Box<dyn Error>> {
    let cuda = DeviceId::new(DeviceKind::Cuda, 0);
    let topology = MemoryTopology::single_device(cuda, 1_000, 0)?;
    let mut inventory = MemoryPlacementInventory::new(topology);
    inventory.register(MemoryResource::new(
        resource_id(992)?,
        MemoryResourceKind::Cache,
        64,
        MemoryLocation::Device(cuda),
    )?)?;
    let backend = FixtureCacheOwner {
        device: cuda,
        fail: false,
        released: AtomicBool::new(false),
    };
    let cancellation = CancellationToken::default();

    let cancelled_before_validation = CancellationToken::default();
    assert!(cancelled_before_validation.cancel());
    assert_eq!(
        cuda_empty_cache_exact_native(
            &backend,
            &mut inventory,
            DeviceId::new(DeviceKind::Xpu, 0),
            &cancelled_before_validation,
        ),
        Err(NativeMemoryStatsError::Cancelled)
    );
    assert_eq!(inventory.resources().len(), 1);

    let release = cuda_empty_cache_exact_native(&backend, &mut inventory, cuda, &cancellation)?;
    assert_eq!(release.allocator_bytes_released, 128);
    assert_eq!(release.accounting_actions.len(), 1);
    assert!(inventory.resources().is_empty());

    let rocm = DeviceId::new(DeviceKind::Rocm, 1);
    let mut rocm_inventory =
        MemoryPlacementInventory::new(MemoryTopology::single_device(rocm, 1_000, 0)?);
    rocm_inventory.register(MemoryResource::new(
        resource_id(993)?,
        MemoryResourceKind::Cache,
        32,
        MemoryLocation::Device(rocm),
    )?)?;
    let rocm_backend = FixtureCacheOwner {
        device: rocm,
        fail: false,
        released: AtomicBool::new(false),
    };
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(
        cuda_empty_cache_exact_native(&rocm_backend, &mut rocm_inventory, rocm, &cancelled,)
            .is_err()
    );
    assert_eq!(rocm_inventory.resources().len(), 1);
    assert!(
        cuda_empty_cache_exact_native(
            &rocm_backend,
            &mut rocm_inventory,
            DeviceId::new(DeviceKind::Xpu, 1),
            &cancellation,
        )
        .is_err()
    );
    assert_eq!(rocm_inventory.resources().len(), 1);

    let cancellation_during_release = CancellationToken::default();
    let cancelling_backend = CancellingCacheOwner { device: rocm };
    assert!(
        cuda_empty_cache_exact_native(
            &cancelling_backend,
            &mut rocm_inventory,
            rocm,
            &cancellation_during_release,
        )
        .is_err()
    );
    assert!(cancellation_during_release.is_cancelled());
    assert_eq!(
        rocm_inventory.resources().len(),
        1,
        "cancellation after allocator release must prevent accounting commit"
    );
    Ok(())
}

#[test]
fn mlu_count_and_xpu_cache_release_use_the_canonical_memory_owners() -> Result<(), Box<dyn Error>> {
    let xpu = DeviceId::new(DeviceKind::Xpu, 0);
    let mlu = DeviceId::new(DeviceKind::Mlu, 2);
    let topology = MemoryTopology::new(
        vec![
            DeviceMemoryBudget {
                device: xpu,
                capacity_bytes: 1_000,
                durable_baseline_bytes: 0,
            },
            DeviceMemoryBudget {
                device: mlu,
                capacity_bytes: 1_000,
                durable_baseline_bytes: 0,
            },
        ],
        1_000,
        1_000,
        Vec::new(),
    )?;
    let cancellation = CancellationToken::default();
    assert_eq!(mlu_device_count_exact_native(&topology, &cancellation)?, 1);
    let pre_cancelled = CancellationToken::default();
    pre_cancelled.cancel();
    assert!(matches!(
        mlu_device_count_exact_native(&topology, &pre_cancelled),
        Err(NativeMemoryStatsError::Cancelled)
    ));

    let mut inventory = MemoryPlacementInventory::new(topology);
    inventory.register(MemoryResource::new(
        resource_id(990)?,
        MemoryResourceKind::Cache,
        64,
        MemoryLocation::Device(xpu),
    )?)?;
    let backend = FixtureCacheOwner {
        device: xpu,
        fail: false,
        released: AtomicBool::new(false),
    };
    assert!(matches!(
        xpu_empty_cache_exact_native(&backend, &mut inventory, mlu, &pre_cancelled),
        Err(NativeMemoryStatsError::Cancelled)
    ));
    assert!(!backend.released.load(Ordering::Acquire));
    assert_eq!(inventory.resources().len(), 1);
    let release = xpu_empty_cache_exact_native(&backend, &mut inventory, xpu, &cancellation)?;
    assert_eq!(release.allocator_bytes_released, 128);
    assert_eq!(release.accounting_actions.len(), 1);
    assert!(backend.released.load(Ordering::Acquire));
    assert!(inventory.resources().is_empty());
    assert!(xpu_empty_cache_exact_native(&backend, &mut inventory, mlu, &cancellation).is_err());
    Ok(())
}

#[test]
fn task_59_mlu_empty_cache_is_only_an_adapter_to_canonical_accounting() -> Result<(), Box<dyn Error>>
{
    let mlu = DeviceId::new(DeviceKind::Mlu, 0);
    let topology = MemoryTopology::single_device(mlu, 1_000, 0)?;
    let mut inventory = MemoryPlacementInventory::new(topology);
    inventory.register(MemoryResource::new(
        resource_id(991)?,
        MemoryResourceKind::Cache,
        64,
        MemoryLocation::Device(mlu),
    )?)?;
    let backend = FixtureCacheOwner {
        device: mlu,
        fail: false,
        released: AtomicBool::new(false),
    };
    let cancellation = CancellationToken::default();
    let release = mlu_empty_cache_exact_native(&backend, &mut inventory, mlu, &cancellation)?;
    assert_eq!(release.allocator_bytes_released, 128);
    assert_eq!(release.accounting_actions.len(), 1);
    assert!(backend.released.load(Ordering::Acquire));
    assert!(inventory.resources().is_empty());
    assert!(
        mlu_empty_cache_exact_native(
            &backend,
            &mut inventory,
            DeviceId::new(DeviceKind::Xpu, 0),
            &cancellation,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn task_60_cuda_queries_and_ipc_collection_reuse_canonical_memory_owners()
-> Result<(), Box<dyn Error>> {
    let cuda = DeviceId::new(DeviceKind::Cuda, 0);
    let xpu = DeviceId::new(DeviceKind::Xpu, 0);
    let topology = MemoryTopology::new(
        vec![
            DeviceMemoryBudget {
                device: cuda,
                capacity_bytes: 1_000,
                durable_baseline_bytes: 0,
            },
            DeviceMemoryBudget {
                device: xpu,
                capacity_bytes: 2_000,
                durable_baseline_bytes: 0,
            },
        ],
        1_000,
        1_000,
        Vec::new(),
    )?;
    let snapshot = MemoryAccountingSnapshot {
        device_bytes: vec![(cuda, 250), (xpu, 100)],
        host_pinned_bytes: 0,
        host_pageable_bytes: 0,
        mapped_bytes: 0,
        durable_bytes: 0,
        fenced_bytes: 0,
        resource_count: 2,
    };
    let cancellation = CancellationToken::default();
    let capabilities = BackendCapabilityMatrix::new(cuda, vec![], vec![])?;
    assert_eq!(
        cuda_mem_get_info_exact_native(&capabilities, &topology, &snapshot, &cancellation)?,
        (750, 1_000)
    );
    assert!(xpu_is_available_exact_native(&topology, &cancellation)?);

    let mut inventory = MemoryPlacementInventory::new(topology);
    inventory.register(MemoryResource::new(
        resource_id(992)?,
        MemoryResourceKind::IpcExport,
        100,
        MemoryLocation::Device(cuda),
    )?)?;
    inventory.register(
        MemoryResource::new(
            resource_id(993)?,
            MemoryResourceKind::IpcExport,
            200,
            MemoryLocation::Device(cuda),
        )?
        .with_pinned(true),
    )?;
    inventory.register(MemoryResource::new(
        resource_id(994)?,
        MemoryResourceKind::Cache,
        50,
        MemoryLocation::Device(cuda),
    )?)?;
    let collection = cuda_ipc_collect_exact_native(&mut inventory, cuda, &cancellation)?;
    assert_eq!(collection.bytes_released, 100);
    assert_eq!(collection.accounting_actions.len(), 1);
    assert_eq!(inventory.resources().len(), 2);
    assert!(
        inventory
            .resources()
            .iter()
            .any(|resource| resource.kind() == MemoryResourceKind::IpcExport)
    );
    assert!(
        inventory
            .resources()
            .iter()
            .any(|resource| resource.kind() == MemoryResourceKind::Cache)
    );

    inventory.register(MemoryResource::new(
        resource_id(995)?,
        MemoryResourceKind::IpcExport,
        75,
        MemoryLocation::Device(cuda),
    )?)?;
    let resource_count = inventory.resources().len();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        cuda_ipc_collect_exact_native(&mut inventory, xpu, &cancelled),
        Err(NativeMemoryStatsError::Cancelled)
    ));
    assert_eq!(inventory.resources().len(), resource_count);
    let wrong_capabilities = BackendCapabilityMatrix::new(DeviceId::CPU, vec![], vec![])?;
    assert!(matches!(
        cuda_mem_get_info_exact_native(
            &wrong_capabilities,
            inventory.topology(),
            &snapshot,
            &cancelled,
        ),
        Err(NativeMemoryStatsError::Cancelled)
    ));
    assert!(matches!(
        xpu_is_available_exact_native(inventory.topology(), &cancelled),
        Err(NativeMemoryStatsError::Cancelled)
    ));
    assert!(cuda_ipc_collect_exact_native(&mut inventory, xpu, &cancellation).is_err());
    Ok(())
}

#[test]
fn task_56_mlu_queries_reuse_canonical_topology_and_accounting() -> Result<(), Box<dyn Error>> {
    let mlu = DeviceId::new(DeviceKind::Mlu, 1);
    let topology = MemoryTopology::single_device(mlu, 1_000, 0)?;
    let snapshot = MemoryAccountingSnapshot {
        device_bytes: vec![(mlu, 250)],
        host_pinned_bytes: 0,
        host_pageable_bytes: 0,
        mapped_bytes: 0,
        durable_bytes: 0,
        fenced_bytes: 0,
        resource_count: 1,
    };
    let capabilities = BackendCapabilityMatrix::new(mlu, vec![], vec![])?;
    let cancellation = CancellationToken::default();
    assert!(mlu_is_available_exact_native(&topology, &cancellation)?);
    assert_eq!(
        mlu_mem_get_info_exact_native(&capabilities, &topology, &snapshot, &cancellation,)?,
        (750, 1_000)
    );
    let wrong = BackendCapabilityMatrix::new(DeviceId::CPU, vec![], vec![])?;
    assert!(mlu_mem_get_info_exact_native(&wrong, &topology, &snapshot, &cancellation).is_err());
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    assert!(mlu_is_available_exact_native(&topology, &cancelled).is_err());
    Ok(())
}

#[test]
fn task_57_npu_count_reuses_canonical_topology() -> Result<(), Box<dyn Error>> {
    let npu = DeviceId::new(DeviceKind::Npu, 0);
    let topology = MemoryTopology::new(
        vec![
            DeviceMemoryBudget {
                device: npu,
                capacity_bytes: 1_000,
                durable_baseline_bytes: 0,
            },
            DeviceMemoryBudget {
                device: DeviceId::new(DeviceKind::Npu, 1),
                capacity_bytes: 2_000,
                durable_baseline_bytes: 0,
            },
        ],
        4_000,
        4_000,
        Vec::new(),
    )?;
    let cancellation = CancellationToken::default();
    assert_eq!(npu_device_count_exact_native(&topology, &cancellation)?, 2);
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    assert!(npu_device_count_exact_native(&topology, &cancelled).is_err());
    Ok(())
}

#[test]
fn task_58_npu_availability_reuses_canonical_topology() -> Result<(), Box<dyn Error>> {
    let cancellation = CancellationToken::default();
    let topology = MemoryTopology::new(
        vec![DeviceMemoryBudget {
            device: DeviceId::new(DeviceKind::Npu, 0),
            capacity_bytes: 1_000,
            durable_baseline_bytes: 0,
        }],
        4_000,
        4_000,
        Vec::new(),
    )?;
    assert!(npu_is_available_exact_native(&topology, &cancellation)?);
    let cpu_only = MemoryTopology::single_device(DeviceId::CPU, 1_000, 0)?;
    assert!(!npu_is_available_exact_native(&cpu_only, &cancellation)?);
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    assert!(npu_is_available_exact_native(&topology, &cancelled).is_err());
    Ok(())
}

impl CachedAllocationOwner for FixtureCacheOwner {
    fn cache_device(&self) -> DeviceId {
        self.device
    }

    fn release_cached_allocations(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<u64, TensorError> {
        cancellation.check()?;
        if self.fail {
            return Err(TensorError::UnsupportedCapability {
                operation: "fixture-cache-release".to_owned(),
                device: self.device,
                reason: "injected allocator failure".to_owned(),
            });
        }
        self.released.store(true, Ordering::Release);
        Ok(128)
    }
}

impl CachedAllocationOwner for CancellingCacheOwner {
    fn cache_device(&self) -> DeviceId {
        self.device
    }

    fn release_cached_allocations(
        &self,
        cancellation: &CancellationToken,
    ) -> Result<u64, TensorError> {
        cancellation.check()?;
        cancellation.cancel();
        Ok(128)
    }
}

fn workspace_root() -> Result<PathBuf, Box<dyn Error>> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| "sim_worker manifest has no workspace root".into())
}

fn resource_id(value: u64) -> Result<MemoryResourceId, Box<dyn Error>> {
    Ok(MemoryResourceId::new(value)?)
}

fn fence_id(value: u64) -> Result<MemoryFenceId, Box<dyn Error>> {
    Ok(MemoryFenceId::new(value)?)
}

fn cuda(ordinal: u32) -> DeviceId {
    DeviceId::new(DeviceKind::Cuda, ordinal)
}

#[test]
fn npu_memory_stats_are_a_checked_projection_of_worker_accounting() -> Result<(), Box<dyn Error>> {
    let npu = DeviceId::new(DeviceKind::Npu, 2);
    let snapshot = MemoryAccountingSnapshot {
        device_bytes: vec![(npu, 4096)],
        host_pinned_bytes: 0,
        host_pageable_bytes: 0,
        mapped_bytes: 0,
        durable_bytes: 0,
        fenced_bytes: 0,
        resource_count: 1,
    };
    let capabilities = BackendCapabilityMatrix::new(npu, Vec::new(), Vec::new())?;
    let cancellation = CancellationToken::default();
    let stats = npu_memory_stats_exact_native(&capabilities, &snapshot, &cancellation)?;
    assert_eq!(stats.get("active_bytes.all.current"), Some(&4096));
    assert_eq!(stats.get("reserved_bytes.all.current"), Some(&4096));

    let mut duplicated = snapshot.clone();
    duplicated.device_bytes.push((npu, 8192));
    assert!(matches!(
        npu_memory_stats_exact_native(&capabilities, &duplicated, &cancellation),
        Err(NativeMemoryStatsError::Policy(
            MemoryPolicyError::DuplicateDevice(device)
        )) if device == npu
    ));

    let cpu = BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?;
    assert!(matches!(
        npu_memory_stats_exact_native(&cpu, &snapshot, &cancellation),
        Err(NativeMemoryStatsError::UnsupportedDevice(DeviceId::CPU))
    ));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        npu_memory_stats_exact_native(&cpu, &duplicated, &cancelled),
        Err(NativeMemoryStatsError::Cancelled)
    ));
    Ok(())
}

#[test]
fn task_64_mlu_memory_stats_reuse_canonical_worker_accounting() -> Result<(), Box<dyn Error>> {
    let mlu = DeviceId::new(DeviceKind::Mlu, 0);
    let snapshot = MemoryAccountingSnapshot {
        device_bytes: vec![(mlu, 4096)],
        host_pinned_bytes: 0,
        host_pageable_bytes: 0,
        mapped_bytes: 0,
        durable_bytes: 0,
        fenced_bytes: 0,
        resource_count: 1,
    };
    let capabilities = BackendCapabilityMatrix::new(mlu, Vec::new(), Vec::new())?;
    let cancellation = CancellationToken::default();
    let stats = mlu_memory_stats_exact_native(&capabilities, &snapshot, &cancellation)?;
    assert_eq!(stats.get("active_bytes.all.current"), Some(&4096));
    assert_eq!(stats.get("reserved_bytes.all.current"), Some(&4096));
    assert!(matches!(
        mlu_memory_stats_exact_native(
            &BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?,
            &snapshot,
            &cancellation,
        ),
        Err(NativeMemoryStatsError::UnsupportedDevice(DeviceId::CPU))
    ));
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let mut duplicated = snapshot;
    duplicated.device_bytes.push((mlu, 2048));
    assert!(matches!(
        mlu_memory_stats_exact_native(
            &BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?,
            &duplicated,
            &cancelled,
        ),
        Err(NativeMemoryStatsError::Cancelled)
    ));
    Ok(())
}

#[test]
fn task_65_cuda_stats_and_npu_info_reuse_canonical_worker_accounting() -> Result<(), Box<dyn Error>>
{
    let cuda = DeviceId::new(DeviceKind::Cuda, 2);
    let cuda_snapshot = MemoryAccountingSnapshot {
        device_bytes: vec![(cuda, 3_000)],
        host_pinned_bytes: 0,
        host_pageable_bytes: 0,
        mapped_bytes: 0,
        durable_bytes: 0,
        fenced_bytes: 0,
        resource_count: 1,
    };
    let cuda_capabilities = BackendCapabilityMatrix::new(cuda, Vec::new(), Vec::new())?;
    let cancellation = CancellationToken::default();
    let stats = cuda_memory_stats_exact_native(&cuda_capabilities, &cuda_snapshot, &cancellation)?;
    assert_eq!(stats.get("active_bytes.all.current"), Some(&3_000));
    assert_eq!(stats.get("reserved_bytes.all.current"), Some(&3_000));

    let npu = DeviceId::new(DeviceKind::Npu, 1);
    let topology = MemoryTopology::single_device(npu, 10_000, 0)?;
    let npu_snapshot = MemoryAccountingSnapshot {
        device_bytes: vec![(npu, 4_000)],
        host_pinned_bytes: 0,
        host_pageable_bytes: 0,
        mapped_bytes: 0,
        durable_bytes: 0,
        fenced_bytes: 0,
        resource_count: 1,
    };
    let npu_capabilities = BackendCapabilityMatrix::new(npu, Vec::new(), Vec::new())?;
    assert_eq!(
        npu_mem_get_info_exact_native(&npu_capabilities, &topology, &npu_snapshot, &cancellation,)?,
        (6_000, 10_000)
    );

    let cpu = BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?;
    assert!(matches!(
        cuda_memory_stats_exact_native(&cpu, &cuda_snapshot, &cancellation),
        Err(NativeMemoryStatsError::UnsupportedDevice(DeviceId::CPU))
    ));
    assert!(matches!(
        npu_mem_get_info_exact_native(&cpu, &topology, &npu_snapshot, &cancellation),
        Err(NativeMemoryStatsError::UnsupportedDevice(DeviceId::CPU))
    ));
    let cancelled = CancellationToken::default();
    assert!(cancelled.cancel());
    let mut duplicated_cuda_snapshot = cuda_snapshot;
    duplicated_cuda_snapshot.device_bytes.push((cuda, 1));
    assert!(matches!(
        cuda_memory_stats_exact_native(&cpu, &duplicated_cuda_snapshot, &cancelled),
        Err(NativeMemoryStatsError::Cancelled)
    ));
    assert!(matches!(
        npu_mem_get_info_exact_native(&cpu, &topology, &npu_snapshot, &cancelled,),
        Err(NativeMemoryStatsError::Cancelled)
    ));
    Ok(())
}

#[test]
fn task_54_part_eleven_npu_cache_and_xpu_stats_reuse_canonical_worker_transactions()
-> Result<(), Box<dyn Error>> {
    let npu = DeviceId::new(DeviceKind::Npu, 1);
    let xpu = DeviceId::new(DeviceKind::Xpu, 4);
    let topology = MemoryTopology::new(
        vec![
            DeviceMemoryBudget {
                device: npu,
                capacity_bytes: 1_000,
                durable_baseline_bytes: 0,
            },
            DeviceMemoryBudget {
                device: xpu,
                capacity_bytes: 2_000,
                durable_baseline_bytes: 0,
            },
        ],
        1_000,
        1_000,
        Vec::new(),
    )?;
    let mut inventory = MemoryPlacementInventory::new(topology);
    inventory.register(MemoryResource::new(
        resource_id(991)?,
        MemoryResourceKind::Cache,
        64,
        MemoryLocation::Device(npu),
    )?)?;
    let cancellation = CancellationToken::default();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_owner = FixtureCacheOwner {
        device: xpu,
        fail: false,
        released: AtomicBool::new(false),
    };
    assert!(matches!(
        npu_empty_cache_exact_native(&cancelled_owner, &mut inventory, xpu, &cancelled),
        Err(NativeMemoryStatsError::Cancelled)
    ));
    assert!(!cancelled_owner.released.load(Ordering::SeqCst));
    assert_eq!(inventory.resources().len(), 1);
    let failing_owner = FixtureCacheOwner {
        device: npu,
        fail: true,
        released: AtomicBool::new(false),
    };
    assert!(
        npu_empty_cache_exact_native(&failing_owner, &mut inventory, npu, &cancellation).is_err()
    );
    assert_eq!(inventory.resources().len(), 1);

    let owner = FixtureCacheOwner {
        device: npu,
        fail: false,
        released: AtomicBool::new(false),
    };
    let release = npu_empty_cache_exact_native(&owner, &mut inventory, npu, &cancellation)?;
    assert_eq!(release.allocator_bytes_released, 128);
    assert_eq!(release.accounting_actions.len(), 1);
    assert!(inventory.resources().is_empty());
    assert!(npu_empty_cache_exact_native(&owner, &mut inventory, xpu, &cancellation).is_err());

    let snapshot = MemoryAccountingSnapshot {
        device_bytes: vec![(xpu, 777)],
        host_pinned_bytes: 0,
        host_pageable_bytes: 0,
        mapped_bytes: 0,
        durable_bytes: 0,
        fenced_bytes: 0,
        resource_count: 1,
    };
    let capabilities = BackendCapabilityMatrix::new(xpu, Vec::new(), Vec::new())?;
    let stats = xpu_memory_stats_exact_native(&capabilities, &snapshot, &cancellation)?;
    assert_eq!(stats.get("active_bytes.all.current"), Some(&777));
    assert_eq!(stats.get("reserved_bytes.all.current"), Some(&777));
    assert!(
        xpu_memory_stats_exact_native(
            &BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?,
            &snapshot,
            &cancellation
        )
        .is_err()
    );
    assert!(matches!(
        xpu_memory_stats_exact_native(
            &BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?,
            &snapshot,
            &cancelled,
        ),
        Err(NativeMemoryStatsError::Cancelled)
    ));
    Ok(())
}

#[test]
fn source_device_and_memory_diagnostics_are_checked_worker_projections()
-> Result<(), Box<dyn Error>> {
    let cuda = cuda(1);
    let xpu_zero = DeviceId::new(DeviceKind::Xpu, 0);
    let xpu_two = DeviceId::new(DeviceKind::Xpu, 2);
    let topology = MemoryTopology::new(
        vec![
            DeviceMemoryBudget {
                device: cuda,
                capacity_bytes: 16 * GIB,
                durable_baseline_bytes: 0,
            },
            DeviceMemoryBudget {
                device: xpu_zero,
                capacity_bytes: 8 * GIB,
                durable_baseline_bytes: 0,
            },
            DeviceMemoryBudget {
                device: xpu_two,
                capacity_bytes: 8 * GIB,
                durable_baseline_bytes: 0,
            },
        ],
        8 * GIB,
        16 * GIB,
        Vec::new(),
    )?;
    let snapshot = MemoryAccountingSnapshot {
        device_bytes: vec![(cuda, 8192), (xpu_zero, 0), (xpu_two, 0)],
        host_pinned_bytes: 0,
        host_pageable_bytes: 0,
        mapped_bytes: 0,
        durable_bytes: 0,
        fenced_bytes: 0,
        resource_count: 1,
    };
    let capabilities = BackendCapabilityMatrix::new(cuda, Vec::new(), Vec::new())?;
    let cancellation = CancellationToken::default();
    assert_eq!(
        cuda_memory_summary_exact_native(&capabilities, &snapshot, &cancellation)?,
        "Zed native Cuda memory summary, device 1\nactive_bytes.all.current: 8192\nreserved_bytes.all.current: 8192\n"
    );
    assert_eq!(xpu_device_count_exact_native(&topology, &cancellation)?, 2);

    let cpu = BackendCapabilityMatrix::for_native_device(DeviceId::CPU)?;
    assert!(matches!(
        cuda_memory_summary_exact_native(&cpu, &snapshot, &cancellation),
        Err(NativeMemoryStatsError::UnsupportedDevice(DeviceId::CPU))
    ));
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    assert!(matches!(
        cuda_memory_summary_exact_native(&cpu, &snapshot, &cancelled),
        Err(NativeMemoryStatsError::Cancelled)
    ));
    assert!(matches!(
        xpu_device_count_exact_native(&topology, &cancelled),
        Err(NativeMemoryStatsError::Cancelled)
    ));
    Ok(())
}

fn capabilities(device: DeviceId) -> MemoryModeCapabilities {
    MemoryModeCapabilities {
        device,
        supports_asynchronous_offload: true,
        supports_pinned_staging: true,
        supports_mmap_weights: true,
    }
}

fn dynamic_mode(device: DeviceId) -> Result<EffectiveMemoryMode, Box<dyn Error>> {
    Ok(EffectiveMemoryMode::resolve(
        MemoryModeRequest::default(),
        capabilities(device),
    )?)
}

fn parse_csv_row(row: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut characters = row.chars().peekable();
    let mut quoted = false;
    while let Some(character) = characters.next() {
        match character {
            '"' if quoted && characters.peek() == Some(&'"') => {
                characters.next();
                field.push('"');
            }
            '"' => quoted = !quoted,
            ',' if !quoted => fields.push(std::mem::take(&mut field)),
            '\r' if !quoted => {}
            _ => field.push(character),
        }
    }
    if quoted {
        return Err("unterminated CSV quote".into());
    }
    fields.push(field);
    Ok(fields)
}

fn catalog_memory_modes() -> Result<BTreeSet<String>, Box<dyn Error>> {
    let catalog = fs::read_to_string(
        workspace_root()?.join("crates/sim_test_support/parity/catalogs/backend-models.csv"),
    )?;
    let mut modes = BTreeSet::new();
    for row in catalog.lines().skip(1) {
        let fields = parse_csv_row(row)?;
        if fields.first().map(String::as_str) == Some("memory mode") {
            let identifier = fields
                .get(1)
                .ok_or("memory-mode catalog row has no identifier")?;
            modes.insert(identifier.clone());
        }
    }
    Ok(modes)
}

fn catalog_modes_are_exact() -> Result<bool, Box<dyn Error>> {
    let catalog = catalog_memory_modes()?;
    let native: BTreeSet<_> = CATALOG_MEMORY_MODES
        .iter()
        .map(|mode| mode.source_identifier.to_owned())
        .collect();
    Ok(catalog.len() == 5 && catalog == native)
}

fn mode_resolution_is_typed() -> Result<bool, Box<dyn Error>> {
    let device = cuda(0);
    let cpu_capabilities = MemoryModeCapabilities::from_backend(&CpuBackend::capability_matrix());
    let dynamic = dynamic_mode(device)?;
    let low = EffectiveMemoryMode::resolve(
        MemoryModeRequest {
            residency: ModelResidencyMode::LowVram,
            asynchronous_offload: true,
            pinned_staging: true,
            mmap_weights: false,
        },
        capabilities(device),
    )?;
    let no_vram = EffectiveMemoryMode::resolve(
        MemoryModeRequest {
            residency: ModelResidencyMode::NoVram,
            asynchronous_offload: false,
            pinned_staging: false,
            mmap_weights: true,
        },
        capabilities(device),
    )?;
    let unsupported = EffectiveMemoryMode::resolve(
        MemoryModeRequest {
            asynchronous_offload: true,
            ..MemoryModeRequest::default()
        },
        MemoryModeCapabilities {
            supports_asynchronous_offload: false,
            ..capabilities(device)
        },
    );
    let conflict = EffectiveMemoryMode::resolve(
        MemoryModeRequest {
            residency: ModelResidencyMode::GpuOnly,
            asynchronous_offload: true,
            pinned_staging: false,
            mmap_weights: false,
        },
        capabilities(device),
    );
    let conservative = EffectiveMemoryMode::resolve(
        MemoryModeRequest::from_runtime_policy(MemoryPolicy::Conservative),
        capabilities(device),
    )?;
    let balanced = EffectiveMemoryMode::resolve(
        MemoryModeRequest::from_runtime_policy(MemoryPolicy::Balanced),
        capabilities(device),
    )?;
    let performance = EffectiveMemoryMode::resolve(
        MemoryModeRequest::from_runtime_policy(MemoryPolicy::Performance),
        capabilities(device),
    )?;
    Ok(dynamic.offload_granularity == OffloadGranularity::Group
        && low.offload_granularity == OffloadGranularity::Group
        && no_vram.offload_granularity == OffloadGranularity::Layer
        && matches!(unsupported, Err(MemoryPolicyError::UnsupportedMode { .. }))
        && matches!(conflict, Err(MemoryPolicyError::ConflictingModes { .. }))
        && conservative.residency == ModelResidencyMode::NoVram
        && balanced.residency == ModelResidencyMode::Dynamic
        && performance.residency == ModelResidencyMode::HighVram
        && conservative
            .configuration_token()
            .contains("residency=novram")
        && cpu_capabilities.device == DeviceId::CPU
        && !cpu_capabilities.supports_asynchronous_offload
        && !cpu_capabilities.supports_pinned_staging
        && cpu_capabilities.supports_mmap_weights)
}

fn pressure_inventory() -> Result<MemoryPlacementInventory, Box<dyn Error>> {
    let device = cuda(0);
    let topology = MemoryTopology::new(
        vec![DeviceMemoryBudget {
            device,
            capacity_bytes: 2_000,
            durable_baseline_bytes: 0,
        }],
        2_000,
        2_000,
        Vec::new(),
    )?;
    let mut inventory = MemoryPlacementInventory::new(topology);
    for (id, kind, bytes, active) in [
        (1, MemoryResourceKind::Preview, 50, true),
        (2, MemoryResourceKind::DecodedMedia, 60, true),
        (3, MemoryResourceKind::Cache, 70, true),
        (4, MemoryResourceKind::Activations, 80, true),
        (5, MemoryResourceKind::ModelPage, 90, false),
    ] {
        inventory.register(
            MemoryResource::new(
                resource_id(id)?,
                kind,
                bytes,
                MemoryLocation::Device(device),
            )?
            .with_active(active),
        )?;
    }
    inventory.register(
        MemoryResource::new(
            resource_id(6)?,
            MemoryResourceKind::Preview,
            100,
            MemoryLocation::Device(device),
        )?
        .with_pinned(true),
    )?;
    inventory.register(MemoryResource::new(
        resource_id(7)?,
        MemoryResourceKind::Preview,
        110,
        MemoryLocation::Device(device),
    )?)?;
    inventory.attach_fence(resource_id(7)?, fence_id(7)?)?;
    Ok(inventory)
}

fn eviction_order_and_protection_are_exact() -> Result<bool, Box<dyn Error>> {
    let device = cuda(0);
    let mut inventory = pressure_inventory()?;
    let actions = inventory.reclaim_device(device, 350, dynamic_mode(device)?)?;
    let kinds: Vec<_> = actions
        .iter()
        .map(|action| (action.resource_kind, action.action))
        .collect();
    let remaining: BTreeSet<_> = inventory
        .resources()
        .iter()
        .map(MemoryResource::id)
        .collect();
    Ok(kinds
        == [
            (MemoryResourceKind::Preview, EvictionActionKind::Dropped),
            (
                MemoryResourceKind::DecodedMedia,
                EvictionActionKind::Dropped,
            ),
            (MemoryResourceKind::Cache, EvictionActionKind::Dropped),
            (
                MemoryResourceKind::Activations,
                EvictionActionKind::OffloadedToPageableHost,
            ),
            (
                MemoryResourceKind::ModelPage,
                EvictionActionKind::PagedOutToMapping,
            ),
        ]
        && remaining.contains(&resource_id(6)?)
        && remaining.contains(&resource_id(7)?)
        && inventory.snapshot()?.fenced_bytes == 110)
}

fn fences_block_reclaim_until_completion() -> Result<bool, Box<dyn Error>> {
    let device = cuda(0);
    let mut inventory = pressure_inventory()?;
    let before = inventory.reclaim_device(device, 500, dynamic_mode(device)?);
    let completed = inventory.complete_fence(fence_id(7)?);
    let after = inventory.reclaim_device(device, 460, dynamic_mode(device)?)?;
    let fenced_resource = resource_id(7)?;
    Ok(
        matches!(before, Err(MemoryPolicyError::InsufficientReclaim { .. }))
            && completed == 1
            && after
                .iter()
                .any(|action| action.resource_id == fenced_resource),
    )
}

fn placement_routes_are_effective() -> Result<bool, Box<dyn Error>> {
    let first = cuda(0);
    let second = cuda(1);
    let budgets = vec![
        DeviceMemoryBudget {
            device: first,
            capacity_bytes: 1_000,
            durable_baseline_bytes: 0,
        },
        DeviceMemoryBudget {
            device: second,
            capacity_bytes: 1_000,
            durable_baseline_bytes: 0,
        },
    ];
    let groups = vec![
        PlacementGroup {
            group_id: 1,
            bytes: 800,
            preferred_devices: vec![first],
        },
        PlacementGroup {
            group_id: 2,
            bytes: 800,
            preferred_devices: vec![first],
        },
    ];
    let peer_inventory = MemoryPlacementInventory::new(MemoryTopology::new(
        budgets.clone(),
        2_000,
        2_000,
        vec![(first, second)],
    )?);
    let peer = MemoryPlacementPlanner::place(&peer_inventory, &groups, dynamic_mode(first)?)?;
    let pinned_mode = EffectiveMemoryMode::resolve(
        MemoryModeRequest {
            pinned_staging: true,
            mmap_weights: false,
            ..MemoryModeRequest::default()
        },
        capabilities(first),
    )?;
    let staged_inventory =
        MemoryPlacementInventory::new(MemoryTopology::new(budgets, 2_000, 2_000, Vec::new())?);
    let staged = MemoryPlacementPlanner::place(&staged_inventory, &groups, pinned_mode)?;
    let insufficient_staging = MemoryPlacementPlanner::place(
        &MemoryPlacementInventory::new(MemoryTopology::new(
            vec![
                DeviceMemoryBudget {
                    device: first,
                    capacity_bytes: 1_000,
                    durable_baseline_bytes: 0,
                },
                DeviceMemoryBudget {
                    device: second,
                    capacity_bytes: 1_000,
                    durable_baseline_bytes: 0,
                },
            ],
            799,
            2_000,
            Vec::new(),
        )?),
        &groups,
        pinned_mode,
    );
    Ok(peer[0].device == first
        && peer[1].device == second
        && peer[1].transfer_from_previous == TransferRoute::PeerCopy
        && peer[1].transfer_bytes == 800
        && staged[1].transfer_from_previous == TransferRoute::HostPinnedStaging
        && staged[1].transfer_bytes == 800
        && matches!(
            insufficient_staging,
            Err(MemoryPolicyError::LocationOutOfMemory {
                location: MemoryLocation::HostPinned,
                ..
            })
        ))
}

fn runtime_policy_reaches_worker_configuration() -> Result<bool, Box<dyn Error>> {
    let root = workspace_root()?;
    let runtime_source =
        fs::read_to_string(root.join("crates/sim_runtime/src/native_execution_controller.rs"))?;
    let worker_source = fs::read_to_string(root.join("crates/sim_worker/src/sim_worker.rs"))?;
    let zed_source = fs::read_to_string(root.join("crates/zed/src/zed.rs"))?;
    Ok(runtime_source.contains("pub memory_policy: MemoryPolicy")
        && runtime_source.contains("NativeImageWorkerPlan::new_with_memory_policy")
        && runtime_source.contains("execute_blocking_with_event_bus_and_configuration")
        && worker_source.contains("worker_plan.memory_policy")
        && worker_source.contains("effective_mode.configuration_token()")
        && zed_source.contains("with_memory_policy(profile.memory_policy)"))
}

fn oom_policy_is_bounded_and_monotonic() -> Result<bool, Box<dyn Error>> {
    let mut controller = AttemptMemoryController::new(
        2 * GIB,
        512 * MIB,
        sim_worker::MemoryPlanRequest {
            workspace_bytes: 256 * MIB,
            activations_bytes: 128 * MIB,
            output_bytes: 64 * MIB,
            ..sim_worker::MemoryPlanRequest::default()
        },
    )?;
    let initial = controller.current_plan().committed_target_bytes;
    let pressure = controller
        .pressure_replan("injected device pressure", 128 * MIB)?
        .committed_target_bytes;
    let reduced = controller
        .reduced_workspace_retry("injected allocation OOM", 64 * MIB)?
        .committed_target_bytes;
    let terminal = controller.terminal_oom("injected terminal OOM");
    let phases: Vec<_> = controller
        .records()
        .iter()
        .map(|record| record.phase)
        .collect();
    Ok(initial > pressure
        && pressure > reduced
        && phases
            == [
                MemoryRecoveryPhase::Initial,
                MemoryRecoveryPhase::PressureReplan,
                MemoryRecoveryPhase::ReducedWorkspace,
            ]
        && controller.state() == AttemptMemoryState::Failed
        && matches!(
            terminal,
            MemoryPolicyError::TerminalOutOfMemory { attempts: 3, .. }
        ))
}

fn workspace_authorization_follows_the_only_retry_owner() -> Result<bool, Box<dyn Error>> {
    let mut controller = AttemptMemoryController::new(
        2 * GIB,
        256 * MIB,
        sim_worker::MemoryPlanRequest {
            workspace_bytes: 192 * MIB,
            activations_bytes: 64 * MIB,
            ..sim_worker::MemoryPlanRequest::default()
        },
    )?;
    let initial = controller.workspace_authorization_bytes();
    controller.pressure_replan("workspace pressure", 64 * MIB)?;
    let after_pressure = controller.workspace_authorization_bytes();
    controller.reduced_workspace_retry("workspace allocation", 48 * MIB)?;
    let after_reduction = controller.workspace_authorization_bytes();
    let duplicate_reduction = matches!(
        controller.reduced_workspace_retry("duplicate", 16 * MIB),
        Err(MemoryPolicyError::RecoveryOrder)
    );
    let planned_authorization = controller.issue_workspace_authorization()?;
    let duplicate_authorization = matches!(
        controller.issue_workspace_authorization(),
        Err(MemoryPolicyError::WorkspaceAuthorizationAlreadyIssued)
    );
    let issued_plan_is_frozen = matches!(
        controller.pressure_replan("late pressure", MIB),
        Err(MemoryPolicyError::WorkspaceAuthorizationAlreadyIssued)
    );
    Ok(initial == 192 * MIB
        && after_pressure == initial
        && after_reduction == 48 * MIB
        && planned_authorization.bytes() == after_reduction
        && duplicate_authorization
        && issued_plan_is_frozen
        && duplicate_reduction)
}

#[test]
fn workspace_authorization_tracks_the_only_retry_owner() -> Result<(), Box<dyn Error>> {
    assert!(workspace_authorization_follows_the_only_retry_owner()?);
    Ok(())
}

fn recovery_order_rejects_loops() -> Result<bool, Box<dyn Error>> {
    let request = sim_worker::MemoryPlanRequest {
        workspace_bytes: 64 * MIB,
        ..sim_worker::MemoryPlanRequest::default()
    };
    let mut controller = AttemptMemoryController::new(GIB, 128 * MIB, request)?;
    let early_reduction = matches!(
        controller.reduced_workspace_retry("early", 32 * MIB),
        Err(MemoryPolicyError::RecoveryOrder)
    );
    controller.pressure_replan("pressure", 32 * MIB)?;
    let duplicate_pressure = matches!(
        controller.pressure_replan("again", 16 * MIB),
        Err(MemoryPolicyError::RecoveryOrder)
    );
    Ok(early_reduction && duplicate_pressure)
}

fn cancellation_waits_for_named_fences() -> Result<bool, Box<dyn Error>> {
    let mut controller = AttemptMemoryController::new(
        GIB,
        0,
        sim_worker::MemoryPlanRequest {
            workspace_bytes: MIB,
            ..sim_worker::MemoryPlanRequest::default()
        },
    )?;
    controller.begin()?;
    controller.register_fence(fence_id(1)?)?;
    let cancelling = controller.cancel()?;
    let rejects_late_value = !controller.accepts_value();
    let cancelled = controller.complete_fence(fence_id(1)?)?;
    Ok(cancelling == AttemptMemoryState::Cancelling
        && rejects_late_value
        && cancelled == AttemptMemoryState::Cancelled)
}

fn cancellation_converges_to_durable_inventory() -> Result<bool, Box<dyn Error>> {
    let device = cuda(0);
    let mut inventory =
        MemoryPlacementInventory::new(MemoryTopology::single_device(device, 1_000, 0)?);
    inventory.register(
        MemoryResource::new(
            resource_id(1)?,
            MemoryResourceKind::Weights,
            100,
            MemoryLocation::Device(device),
        )?
        .with_durable(true),
    )?;
    inventory.register(
        MemoryResource::new(
            resource_id(2)?,
            MemoryResourceKind::Activations,
            400,
            MemoryLocation::Device(device),
        )?
        .with_durable(false),
    )?;
    let released = inventory.discard_attempt_resources()?;
    let snapshot = inventory.snapshot()?;
    Ok(released == 400
        && snapshot.durable_bytes == 100
        && snapshot.device_bytes == vec![(device, 100)])
}

fn device_loss_revokes_device_resources() -> Result<bool, Box<dyn Error>> {
    let device = cuda(0);
    let mut inventory =
        MemoryPlacementInventory::new(MemoryTopology::single_device(device, 1_000, 100)?);
    inventory.register(MemoryResource::new(
        resource_id(1)?,
        MemoryResourceKind::Output,
        300,
        MemoryLocation::Device(device),
    )?)?;
    let invalidated = inventory.invalidate_device(device)?;
    let mut controller = AttemptMemoryController::new(
        GIB,
        0,
        sim_worker::MemoryPlanRequest {
            workspace_bytes: MIB,
            ..sim_worker::MemoryPlanRequest::default()
        },
    )?;
    controller.begin()?;
    controller.device_lost()?;
    Ok(invalidated == 300
        && inventory.snapshot()?.resource_count == 0
        && inventory.snapshot()?.device_bytes == vec![(device, 0)]
        && inventory.snapshot()?.durable_bytes == 0
        && controller.state() == AttemptMemoryState::DeviceLost
        && !controller.accepts_value())
}

fn image_preflight_is_checked() -> Result<bool, Box<dyn Error>> {
    let request = native_image_memory_request(2 * MIB, 5, true)?;
    let overflow = native_image_memory_request(u64::MAX, 1, true);
    Ok(request.staging_bytes == 2 * MIB
        && request.activations_bytes == 8 * MIB
        && request.workspace_bytes == 13 * MIB
        && request.preview_bytes == 8 * MIB
        && request.output_bytes == 8 * MIB
        && matches!(overflow, Err(MemoryPolicyError::AccountingOverflow(_))))
}

fn canonical_allocator_ownership_is_preserved() -> Result<bool, Box<dyn Error>> {
    let (backend, _workspace_authority) = CpuWorkspaceAuthority::create_backend(GIB)?;
    let before = backend.memory_snapshot();
    let controller = AttemptMemoryController::new(
        before.limit_bytes,
        before.current_bytes,
        sim_worker::MemoryPlanRequest {
            workspace_bytes: MIB,
            ..sim_worker::MemoryPlanRequest::default()
        },
    )?;
    let after = backend.memory_snapshot();
    let root = workspace_root()?;
    let worker_source = fs::read_to_string(root.join("crates/sim_worker/src/sim_worker.rs"))?;
    let policy_source = fs::read_to_string(root.join("crates/sim_worker/src/memory_modes.rs"))?;
    let tensor_operation_source =
        fs::read_to_string(root.join("crates/sim_tensor/src/operation.rs"))?;
    let tensor_cpu_source = fs::read_to_string(root.join("crates/sim_tensor/src/cpu_backend.rs"))?;
    let tensor_rocm_source =
        fs::read_to_string(root.join("crates/sim_tensor/src/backends/amd_rocm_sim_model_0014.rs"))?;
    let model_source = fs::read_to_string(root.join("crates/sim_model/src/model_store.rs"))?;
    let cancellation_declaration = ["struct", "CancellationToken"].join(" ");
    Ok(before == after
        && controller.current_plan().committed_target_bytes > 0
        && worker_source.contains("session.memory_snapshot()")
        && worker_source.contains("AttemptMemoryController::new")
        && !policy_source.contains("struct MemoryTracker")
        && !policy_source.contains(&cancellation_declaration)
        && !policy_source.contains("mmap(")
        && tensor_operation_source
            .matches("struct BackendMemoryTracker")
            .count()
            == 1
        && !tensor_cpu_source.contains("struct BackendMemoryTracker")
        && !tensor_rocm_source.contains("struct BackendMemoryTracker")
        && model_source.contains("mmap("))
}

fn fixture_digest(path: &Path) -> Result<String, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn json_string(value: &str) -> Result<String, Box<dyn Error>> {
    Ok(serde_json::to_string(value)?)
}

fn write_artifact(cases: &BTreeMap<&str, bool>) -> Result<(), Box<dyn Error>> {
    if let Some((name, _)) = cases.iter().find(|(_, passed)| !**passed) {
        return Err(format!("VAL-MEMORY-001 case failed: {name}").into());
    }
    let root = workspace_root()?;
    let catalog_path = root.join("crates/sim_test_support/parity/catalogs/backend-models.csv");
    let digest = fixture_digest(&catalog_path)?;
    let cases_json = cases
        .iter()
        .map(|(name, passed)| Ok(format!("{}:{}", json_string(name)?, passed)))
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?
        .join(",");
    let artifact = format!(
        concat!(
            "{{\n",
            "  \"validation_id\": \"VAL-MEMORY-001\",\n",
            "  \"scope\": \"Task 34 native memory modes, placement, pressure, OOM, cancellation, and ownership\",\n",
            "  \"environment\": {{\"operating_system\": {}, \"architecture\": {}, \"backend\": \"native-rust-cpu-policy-plus-synthetic-multidevice\", \"development_oracle_executed\": false}},\n",
            "  \"fixture_digests\": {{\"crates/sim_test_support/parity/catalogs/backend-models.csv\": {}}},\n",
            "  \"summary\": {{\"passed\": {}, \"failed\": 0, \"skipped\": 0}},\n",
            "  \"cases\": {{{}}},\n",
            "  \"skipped\": [],\n",
            "  \"validation_closure\": {{\"claimed\": true, \"stage\": \"comfy-parity-native-memory-planner\"}},\n",
            "  \"release_closure_claimed\": false,\n",
            "  \"release_closure_required\": true,\n",
            "  \"remaining_release_gates\": [\"comfy-parity-final-validation\"]\n",
            "}}\n"
        ),
        json_string(std::env::consts::OS)?,
        json_string(std::env::consts::ARCH)?,
        json_string(&digest)?,
        cases.len(),
        cases_json,
    );
    let directory = root.join("target/comfy-parity");
    fs::create_dir_all(&directory)?;
    let path = directory.join("val-memory-001.json");
    let temporary = directory.join("val-memory-001.json.tmp");
    match fs::remove_file(&temporary) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    fs::write(&temporary, artifact)?;
    fs::rename(temporary, path)?;
    Ok(())
}

#[test]
fn val_memory_001() -> Result<(), Box<dyn Error>> {
    let cases = BTreeMap::from([
        (
            "allocator_owner_is_not_duplicated",
            canonical_allocator_ownership_is_preserved()?,
        ),
        (
            "cancellation_converges_to_durable_baseline",
            cancellation_converges_to_durable_inventory()?,
        ),
        (
            "cancellation_waits_for_named_fences",
            cancellation_waits_for_named_fences()?,
        ),
        (
            "cataloged_memory_modes_are_exact",
            catalog_modes_are_exact()?,
        ),
        (
            "device_loss_revokes_uncommitted_resources",
            device_loss_revokes_device_resources()?,
        ),
        (
            "eviction_order_and_protection_are_exact",
            eviction_order_and_protection_are_exact()?,
        ),
        (
            "fences_block_reclaim_until_completion",
            fences_block_reclaim_until_completion()?,
        ),
        (
            "memory_mode_resolution_is_typed",
            mode_resolution_is_typed()?,
        ),
        (
            "native_image_preflight_is_checked",
            image_preflight_is_checked()?,
        ),
        (
            "oom_policy_is_bounded_and_monotonic",
            oom_policy_is_bounded_and_monotonic()?,
        ),
        (
            "placement_and_peer_fallback_are_effective",
            placement_routes_are_effective()?,
        ),
        (
            "recovery_order_rejects_restart_loops",
            recovery_order_rejects_loops()?,
        ),
        (
            "runtime_policy_reaches_worker_and_cache_configuration",
            runtime_policy_reaches_worker_configuration()?,
        ),
        (
            "workspace_authorization_tracks_the_only_retry_owner",
            workspace_authorization_follows_the_only_retry_owner()?,
        ),
    ]);
    write_artifact(&cases)
}

#[test]
fn invalid_topologies_and_capacity_fail_before_mutation() -> Result<(), Box<dyn Error>> {
    let duplicate = MemoryTopology::new(
        vec![
            DeviceMemoryBudget {
                device: cuda(0),
                capacity_bytes: 100,
                durable_baseline_bytes: 0,
            },
            DeviceMemoryBudget {
                device: cuda(0),
                capacity_bytes: 100,
                durable_baseline_bytes: 0,
            },
        ],
        0,
        0,
        Vec::new(),
    );
    assert!(matches!(
        duplicate,
        Err(MemoryPolicyError::DuplicateDevice(_))
    ));

    let mut inventory =
        MemoryPlacementInventory::new(MemoryTopology::single_device(cuda(0), 100, 50)?);
    let error = inventory.register(MemoryResource::new(
        resource_id(1)?,
        MemoryResourceKind::Workspace,
        51,
        MemoryLocation::Device(cuda(0)),
    )?);
    assert!(matches!(
        error,
        Err(MemoryPolicyError::LocationOutOfMemory { .. })
    ));
    assert!(inventory.resources().is_empty());
    Ok(())
}

#[test]
fn source_device_queries_and_mps_cache_release_use_canonical_worker_owners()
-> Result<(), Box<dyn Error>> {
    let metal = DeviceId::new(DeviceKind::Metal, 0);
    let topology = MemoryTopology::new(
        vec![
            DeviceMemoryBudget {
                device: cuda(0),
                capacity_bytes: 1_000,
                durable_baseline_bytes: 0,
            },
            DeviceMemoryBudget {
                device: DeviceId::new(DeviceKind::Rocm, 1),
                capacity_bytes: 1_000,
                durable_baseline_bytes: 0,
            },
            DeviceMemoryBudget {
                device: metal,
                capacity_bytes: 1_000,
                durable_baseline_bytes: 0,
            },
        ],
        1_000,
        1_000,
        Vec::new(),
    )?;
    let cancellation = CancellationToken::default();
    assert_eq!(cuda_device_count_exact_native(&topology, &cancellation)?, 2);
    assert!(cuda_is_available_exact_native(&topology, &cancellation)?);

    let mut inventory = MemoryPlacementInventory::new(topology);
    inventory.register(MemoryResource::new(
        resource_id(900)?,
        MemoryResourceKind::Cache,
        100,
        MemoryLocation::Device(metal),
    )?)?;
    inventory.register(
        MemoryResource::new(
            resource_id(901)?,
            MemoryResourceKind::Cache,
            50,
            MemoryLocation::Device(metal),
        )?
        .with_in_flight(true),
    )?;
    inventory.register(MemoryResource::new(
        resource_id(902)?,
        MemoryResourceKind::Weights,
        200,
        MemoryLocation::Device(metal),
    )?)?;
    let backend = FixtureCacheOwner {
        device: metal,
        fail: false,
        released: AtomicBool::new(false),
    };
    let release = mps_empty_cache_exact_native(&backend, &mut inventory, metal, &cancellation)?;
    assert!(backend.released.load(Ordering::Acquire));
    assert_eq!(release.allocator_bytes_released, 128);
    assert_eq!(release.accounting_actions.len(), 1);
    assert_eq!(release.accounting_actions[0].resource_id, resource_id(900)?);
    assert_eq!(
        release.accounting_actions[0].action,
        EvictionActionKind::Dropped
    );
    assert_eq!(inventory.resources().len(), 2);

    let before = inventory.resources().to_vec();
    let cancelled = CancellationToken::default();
    cancelled.cancel();
    let cancelled_backend = FixtureCacheOwner {
        device: metal,
        fail: true,
        released: AtomicBool::new(false),
    };
    assert!(matches!(
        mps_empty_cache_exact_native(
            &cancelled_backend,
            &mut inventory,
            DeviceId::CPU,
            &cancelled,
        ),
        Err(NativeMemoryStatsError::Cancelled)
    ));
    assert!(!cancelled_backend.released.load(Ordering::Acquire));
    assert_eq!(inventory.resources(), before);
    assert!(
        mps_empty_cache_exact_native(&backend, &mut inventory, DeviceId::CPU, &cancellation)
            .is_err()
    );

    let failing_backend = FixtureCacheOwner {
        device: metal,
        fail: true,
        released: AtomicBool::new(false),
    };
    assert!(
        mps_empty_cache_exact_native(&failing_backend, &mut inventory, metal, &cancellation)
            .is_err()
    );
    assert_eq!(inventory.resources(), before);
    Ok(())
}
