use crate::{CancellationToken, DeviceId};
use serde::{Deserialize, Serialize};
use sim_types::DeviceKind;
use thiserror::Error;

#[path = "rng_profiles/compat.rs"]
mod compat;

pub use compat::*;

const MT_STATE_LEN: usize = 624;
const MT_MIDDLE_WORD: usize = 397;
const MT_MATRIX_A: u32 = 0x9908_b0df;
const MT_UPPER_MASK: u32 = 0x8000_0000;
const MT_LOWER_MASK: u32 = 0x7fff_ffff;
const PHILOX_MULTIPLIER_0: u32 = 0xd251_1f53;
const PHILOX_MULTIPLIER_1: u32 = 0xcd9e_8d57;
const PHILOX_WEYL_0: u32 = 0x9e37_79b9;
const PHILOX_WEYL_1: u32 = 0xbb67_ae85;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Mt19937 {
    state: [u32; MT_STATE_LEN],
    index: usize,
}

impl Mt19937 {
    pub fn from_seed(seed: u64) -> Self {
        let mut state = [0_u32; MT_STATE_LEN];
        state[0] = seed as u32;
        for index in 1..MT_STATE_LEN {
            let previous = state[index - 1];
            state[index] = 1_812_433_253_u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(index as u32);
        }
        Self {
            state,
            index: MT_STATE_LEN,
        }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= MT_STATE_LEN {
            self.twist();
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^ (value >> 18)
    }

    pub fn snapshot(&self) -> Mt19937Snapshot {
        Mt19937Snapshot {
            state: self.state.to_vec(),
            index: self.index,
        }
    }

    pub fn from_snapshot(snapshot: Mt19937Snapshot) -> Result<Self, RngError> {
        if snapshot.state.len() != MT_STATE_LEN || snapshot.index > MT_STATE_LEN {
            return Err(RngError::InvalidCheckpoint {
                reason: format!(
                    "MT19937 requires {MT_STATE_LEN} state words and index at most {MT_STATE_LEN}"
                ),
            });
        }
        let mut state = [0_u32; MT_STATE_LEN];
        state.copy_from_slice(&snapshot.state);
        Ok(Self {
            state,
            index: snapshot.index,
        })
    }

    fn twist(&mut self) {
        for index in 0..MT_STATE_LEN {
            let next_index = (index + 1) % MT_STATE_LEN;
            let middle_index = (index + MT_MIDDLE_WORD) % MT_STATE_LEN;
            let combined =
                (self.state[index] & MT_UPPER_MASK) | (self.state[next_index] & MT_LOWER_MASK);
            let mut value = self.state[middle_index] ^ (combined >> 1);
            if combined & 1 != 0 {
                value ^= MT_MATRIX_A;
            }
            self.state[index] = value;
        }
        self.index = 0;
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NumpyRandomState {
    generator: Mt19937,
}

impl NumpyRandomState {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            generator: Mt19937::from_seed(seed % u64::from(u32::MAX)),
        }
    }

    pub fn randint(
        &mut self,
        low: u32,
        high_exclusive: u32,
        cancellation: &CancellationToken,
    ) -> Result<u32, RngError> {
        let interval = high_exclusive
            .checked_sub(low)
            .ok_or(RngError::InvalidBound)?;
        let maximum = interval.checked_sub(1).ok_or(RngError::InvalidBound)?;
        let mut mask = maximum;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            if cancellation.is_cancelled() {
                return Err(RngError::Cancelled);
            }
            let value = self.generator.next_u32() & mask;
            if value <= maximum {
                return low.checked_add(value).ok_or(RngError::InvalidBound);
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Mt19937Snapshot {
    state: Vec<u32>,
    index: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Philox4x32;

impl Philox4x32 {
    pub fn generate(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
        for _ in 0..10 {
            let product_0 = u64::from(PHILOX_MULTIPLIER_0) * u64::from(counter[0]);
            let product_1 = u64::from(PHILOX_MULTIPLIER_1) * u64::from(counter[2]);
            let high_0 = (product_0 >> 32) as u32;
            let low_0 = product_0 as u32;
            let high_1 = (product_1 >> 32) as u32;
            let low_1 = product_1 as u32;
            counter = [
                high_1 ^ counter[1] ^ key[0],
                low_1,
                high_0 ^ counter[3] ^ key[1],
                low_0,
            ];
            key[0] = key[0].wrapping_add(PHILOX_WEYL_0);
            key[1] = key[1].wrapping_add(PHILOX_WEYL_1);
        }
        counter
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PhiloxCursor {
    counter: [u32; 4],
    key: [u32; 2],
    block: [u32; 4],
    block_index: usize,
    counter_exhausted: bool,
}

impl PhiloxCursor {
    fn new(counter: [u32; 4], key: [u32; 2]) -> Self {
        Self {
            counter,
            key,
            block: [0; 4],
            block_index: 4,
            counter_exhausted: false,
        }
    }

    fn next_u32(&mut self) -> Result<u32, RngError> {
        if self.block_index >= self.block.len() {
            if self.counter_exhausted {
                return Err(RngError::CounterOverflow);
            }
            let block = Philox4x32::generate(self.counter, self.key);
            let next_counter = increment_counter(self.counter);
            self.block = block;
            if let Some(next_counter) = next_counter {
                self.counter = next_counter;
            } else {
                self.counter_exhausted = true;
            }
            self.block_index = 0;
        }
        let value = self
            .block
            .get(self.block_index)
            .copied()
            .ok_or(RngError::CounterOverflow)?;
        self.block_index += 1;
        Ok(value)
    }

    fn snapshot(&self) -> PhiloxSnapshot {
        PhiloxSnapshot {
            counter: self.counter,
            key: self.key,
            block: self.block,
            block_index: self.block_index,
            counter_exhausted: self.counter_exhausted,
        }
    }

    fn from_snapshot(mut snapshot: PhiloxSnapshot) -> Result<Self, RngError> {
        if snapshot.block_index > 4 {
            return Err(RngError::InvalidCheckpoint {
                reason: "Philox block index exceeds four words".to_owned(),
            });
        }
        if snapshot.block_index == 0 {
            return Err(RngError::InvalidCheckpoint {
                reason: "Philox checkpoints cannot expose an unconsumed generated block".to_owned(),
            });
        }
        if !snapshot.counter_exhausted
            && snapshot.counter == [u32::MAX; 4]
            && snapshot.block == Philox4x32::generate(snapshot.counter, snapshot.key)
        {
            snapshot.counter_exhausted = true;
        }
        if snapshot.counter_exhausted
            && (snapshot.counter != [u32::MAX; 4]
                || snapshot.block != Philox4x32::generate(snapshot.counter, snapshot.key))
        {
            return Err(RngError::InvalidCheckpoint {
                reason: "exhausted Philox state must contain the final counter block".to_owned(),
            });
        }
        if !snapshot.counter_exhausted && snapshot.block_index < 4 {
            let generated_counter =
                decrement_counter(snapshot.counter).ok_or_else(|| RngError::InvalidCheckpoint {
                    reason: "partially consumed Philox block has no preceding counter".to_owned(),
                })?;
            if snapshot.block != Philox4x32::generate(generated_counter, snapshot.key) {
                return Err(RngError::InvalidCheckpoint {
                    reason: "partially consumed Philox block does not match its counter and key"
                        .to_owned(),
                });
            }
        }
        if !snapshot.counter_exhausted && snapshot.block_index == 4 && snapshot.block != [0; 4] {
            let generated_counter =
                decrement_counter(snapshot.counter).ok_or_else(|| RngError::InvalidCheckpoint {
                    reason: "consumed Philox block has no preceding counter".to_owned(),
                })?;
            if snapshot.block != Philox4x32::generate(generated_counter, snapshot.key) {
                return Err(RngError::InvalidCheckpoint {
                    reason: "consumed Philox block does not match its counter and key".to_owned(),
                });
            }
        }
        Ok(Self {
            counter: snapshot.counter,
            key: snapshot.key,
            block: snapshot.block,
            block_index: snapshot.block_index,
            counter_exhausted: snapshot.counter_exhausted,
        })
    }
}

fn increment_counter(counter: [u32; 4]) -> Option<[u32; 4]> {
    let mut incremented = counter;
    for word in &mut incremented {
        let (next, overflowed) = word.overflowing_add(1);
        *word = next;
        if !overflowed {
            return Some(incremented);
        }
    }
    None
}

fn decrement_counter(counter: [u32; 4]) -> Option<[u32; 4]> {
    let mut decremented = counter;
    for word in &mut decremented {
        let (previous, underflowed) = word.overflowing_sub(1);
        *word = previous;
        if !underflowed {
            return Some(decremented);
        }
    }
    None
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RngProfileVersion {
    V1,
    V2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RngAlgorithm {
    Mt19937,
    Philox4x32_10,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryRngPolicy {
    Replay,
    Advance,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RngStreamAddressWire", into = "RngStreamAddressWire")]
pub struct RngStreamAddress {
    workflow: String,
    attempt: String,
    node: String,
    output: u32,
    phase: String,
    batch: u64,
    retry: u32,
    retry_policy: RetryRngPolicy,
    device: DeviceId,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RngStreamAddressWire {
    workflow: String,
    attempt: String,
    node: String,
    output: u32,
    phase: String,
    batch: u64,
    retry: u32,
    retry_policy: RetryRngPolicy,
    #[serde(default)]
    device: Option<DeviceId>,
}

impl TryFrom<RngStreamAddressWire> for RngStreamAddress {
    type Error = RngError;

    fn try_from(value: RngStreamAddressWire) -> Result<Self, Self::Error> {
        Self::for_device(
            value.workflow,
            value.attempt,
            value.node,
            value.output,
            value.phase,
            value.batch,
            value.retry,
            value.retry_policy,
            value.device.unwrap_or(DeviceId::CPU),
        )
    }
}

impl From<RngStreamAddress> for RngStreamAddressWire {
    fn from(value: RngStreamAddress) -> Self {
        Self {
            workflow: value.workflow,
            attempt: value.attempt,
            node: value.node,
            output: value.output,
            phase: value.phase,
            batch: value.batch,
            retry: value.retry,
            retry_policy: value.retry_policy,
            device: Some(value.device),
        }
    }
}

impl RngStreamAddress {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        workflow: impl Into<String>,
        attempt: impl Into<String>,
        node: impl Into<String>,
        output: u32,
        phase: impl Into<String>,
        batch: u64,
        retry: u32,
        retry_policy: RetryRngPolicy,
    ) -> Result<Self, RngError> {
        Self::for_device(
            workflow,
            attempt,
            node,
            output,
            phase,
            batch,
            retry,
            retry_policy,
            DeviceId::CPU,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn for_device(
        workflow: impl Into<String>,
        attempt: impl Into<String>,
        node: impl Into<String>,
        output: u32,
        phase: impl Into<String>,
        batch: u64,
        retry: u32,
        retry_policy: RetryRngPolicy,
        device: DeviceId,
    ) -> Result<Self, RngError> {
        let value = Self {
            workflow: workflow.into(),
            attempt: attempt.into(),
            node: node.into(),
            output,
            phase: phase.into(),
            batch,
            retry,
            retry_policy,
            device,
        };
        for (field, identity) in [
            ("workflow", value.workflow.as_str()),
            ("attempt", value.attempt.as_str()),
            ("node", value.node.as_str()),
            ("phase", value.phase.as_str()),
        ] {
            if identity.trim().is_empty() {
                return Err(RngError::InvalidAddress {
                    field: field.to_owned(),
                });
            }
        }
        Ok(value)
    }

    pub fn device(&self) -> DeviceId {
        self.device
    }

    pub fn workflow(&self) -> &str {
        &self.workflow
    }

    pub fn attempt(&self) -> &str {
        &self.attempt
    }

    pub fn node(&self) -> &str {
        &self.node
    }

    pub fn output(&self) -> u32 {
        self.output
    }

    pub fn phase(&self) -> &str {
        &self.phase
    }

    pub fn batch(&self) -> u64 {
        self.batch
    }

    pub fn retry(&self) -> u32 {
        self.retry
    }

    pub fn retry_policy(&self) -> RetryRngPolicy {
        self.retry_policy
    }

    fn digest(&self, profile: RngProfileVersion, seed: u64) -> [u64; 2] {
        let mut first = StableHasher::new(0xcbf2_9ce4_8422_2325);
        let mut second = StableHasher::new(0x8422_2325_cbf2_9ce4);
        for hasher in [&mut first, &mut second] {
            hasher.write_u8(match profile {
                RngProfileVersion::V1 => 1,
                RngProfileVersion::V2 => 2,
            });
            hasher.write_u64(seed);
            hasher.write_text(&self.workflow);
            hasher.write_text(&self.attempt);
            hasher.write_text(&self.node);
            hasher.write_u32(self.output);
            hasher.write_text(&self.phase);
            hasher.write_u64(self.batch);
            hasher.write_u8(match self.retry_policy {
                RetryRngPolicy::Replay => 0,
                RetryRngPolicy::Advance => 1,
            });
            if self.retry_policy == RetryRngPolicy::Advance {
                hasher.write_u32(self.retry);
            }
            if profile == RngProfileVersion::V2 {
                hasher.write_u8(device_kind_tag(self.device.kind()));
                hasher.write_u32(self.device.ordinal());
            }
        }
        [mix64(first.finish()), mix64(second.finish())]
    }
}

const fn device_kind_tag(device: DeviceKind) -> u8 {
    match device {
        DeviceKind::Cpu => 0,
        DeviceKind::Cuda => 1,
        DeviceKind::Rocm => 2,
        DeviceKind::Metal => 3,
        DeviceKind::DirectMl => 4,
        DeviceKind::Xpu => 5,
        DeviceKind::Npu => 6,
        DeviceKind::Mlu => 7,
        DeviceKind::CoreX => 8,
    }
}

struct StableHasher(u64);

impl StableHasher {
    fn new(domain: u64) -> Self {
        Self(domain)
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn write_u8(&mut self, value: u8) {
        self.write(&[value]);
    }

    fn write_u32(&mut self, value: u32) {
        self.write(&value.to_le_bytes());
    }

    fn write_u64(&mut self, value: u64) {
        self.write(&value.to_le_bytes());
    }

    fn write_text(&mut self, value: &str) {
        self.write_u64(value.len() as u64);
        self.write(value.as_bytes());
    }

    fn finish(self) -> u64 {
        self.0
    }
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PhiloxSnapshot {
    counter: [u32; 4],
    key: [u32; 2],
    block: [u32; 4],
    block_index: usize,
    #[serde(default)]
    counter_exhausted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "algorithm", content = "state")]
pub enum RngGeneratorSnapshot {
    Mt19937(Mt19937Snapshot),
    Philox4x32_10(PhiloxSnapshot),
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "RngCheckpointWire", into = "RngCheckpointWire")]
pub struct RngCheckpoint {
    pub profile: RngProfileVersion,
    pub algorithm: RngAlgorithm,
    pub address_digest: [u64; 2],
    pub device: DeviceId,
    pub generator: RngGeneratorSnapshot,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RngCheckpointWire {
    profile: RngProfileVersion,
    algorithm: RngAlgorithm,
    address_digest: [u64; 2],
    #[serde(default)]
    device: Option<DeviceId>,
    generator: RngGeneratorSnapshot,
}

impl TryFrom<RngCheckpointWire> for RngCheckpoint {
    type Error = RngError;

    fn try_from(value: RngCheckpointWire) -> Result<Self, Self::Error> {
        let device = match (value.profile, value.device) {
            (RngProfileVersion::V1, None) => DeviceId::CPU,
            (RngProfileVersion::V2, None) => {
                return Err(RngError::InvalidCheckpoint {
                    reason: "RNG profile v2 checkpoints require explicit device identity"
                        .to_owned(),
                });
            }
            (_, Some(device)) => device,
        };
        let mut generator = value.generator;
        match (&value.algorithm, &mut generator) {
            (RngAlgorithm::Mt19937, RngGeneratorSnapshot::Mt19937(snapshot)) => {
                Mt19937::from_snapshot(snapshot.clone())?;
            }
            (RngAlgorithm::Philox4x32_10, RngGeneratorSnapshot::Philox4x32_10(snapshot)) => {
                *snapshot = PhiloxCursor::from_snapshot(snapshot.clone())?.snapshot();
            }
            _ => return Err(RngError::CheckpointMismatch),
        }
        Ok(Self {
            profile: value.profile,
            algorithm: value.algorithm,
            address_digest: value.address_digest,
            device,
            generator,
        })
    }
}

impl From<RngCheckpoint> for RngCheckpointWire {
    fn from(value: RngCheckpoint) -> Self {
        Self {
            profile: value.profile,
            algorithm: value.algorithm,
            address_digest: value.address_digest,
            device: Some(value.device),
            generator: value.generator,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RngStream {
    profile: RngProfileVersion,
    algorithm: RngAlgorithm,
    seed: u64,
    address: RngStreamAddress,
}

impl RngStream {
    pub fn new(
        profile: RngProfileVersion,
        algorithm: RngAlgorithm,
        seed: u64,
        address: RngStreamAddress,
    ) -> Result<Self, RngError> {
        if profile == RngProfileVersion::V1 && address.device() != DeviceId::CPU {
            return Err(RngError::LegacyProfileDeviceMismatch);
        }
        Ok(Self {
            profile,
            algorithm,
            seed,
            address,
        })
    }

    pub fn begin(&self, checkpoint: Option<RngCheckpoint>) -> Result<RngTransaction, RngError> {
        let digest = self.address.digest(self.profile, self.seed);
        let generator = if let Some(checkpoint) = checkpoint {
            if checkpoint.profile != self.profile
                || checkpoint.algorithm != self.algorithm
                || checkpoint.address_digest != digest
                || checkpoint.device != self.address.device()
            {
                return Err(RngError::CheckpointMismatch);
            }
            match checkpoint.generator {
                RngGeneratorSnapshot::Mt19937(snapshot)
                    if self.algorithm == RngAlgorithm::Mt19937 =>
                {
                    Generator::Mt19937(Mt19937::from_snapshot(snapshot)?)
                }
                RngGeneratorSnapshot::Philox4x32_10(snapshot)
                    if self.algorithm == RngAlgorithm::Philox4x32_10 =>
                {
                    Generator::Philox4x32_10(PhiloxCursor::from_snapshot(snapshot)?)
                }
                _ => return Err(RngError::CheckpointMismatch),
            }
        } else {
            match self.algorithm {
                RngAlgorithm::Mt19937 => {
                    Generator::Mt19937(Mt19937::from_seed(self.seed ^ digest[0]))
                }
                RngAlgorithm::Philox4x32_10 => {
                    let key = [
                        (self.seed ^ digest[0]) as u32,
                        ((self.seed >> 32) ^ digest[1]) as u32,
                    ];
                    let counter = [
                        digest[0] as u32,
                        (digest[0] >> 32) as u32,
                        digest[1] as u32,
                        (digest[1] >> 32) as u32,
                    ];
                    Generator::Philox4x32_10(PhiloxCursor::new(counter, key))
                }
            }
        };
        Ok(RngTransaction {
            profile: self.profile,
            algorithm: self.algorithm,
            address_digest: digest,
            address_device: self.address.device(),
            generator,
        })
    }

    pub fn reseed(&self, seed: u64) -> Result<Self, RngError> {
        Self::new(self.profile, self.algorithm, seed, self.address.clone())
    }

    pub fn profile(&self) -> RngProfileVersion {
        self.profile
    }

    pub fn algorithm(&self) -> RngAlgorithm {
        self.algorithm
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn address(&self) -> &RngStreamAddress {
        &self.address
    }
}

#[derive(Clone, Debug)]
enum Generator {
    Mt19937(Mt19937),
    Philox4x32_10(PhiloxCursor),
}

#[derive(Clone)]
pub struct RngTransaction {
    profile: RngProfileVersion,
    algorithm: RngAlgorithm,
    address_digest: [u64; 2],
    address_device: DeviceId,
    generator: Generator,
}

impl RngTransaction {
    pub fn device(&self) -> DeviceId {
        self.address_device
    }

    pub fn require_device(&self, expected: DeviceId) -> Result<(), RngError> {
        if self.address_device != expected {
            return Err(RngError::DeviceMismatch {
                expected,
                actual: self.address_device,
            });
        }
        Ok(())
    }

    pub fn next_u32(&mut self, cancellation: &CancellationToken) -> Result<u32, RngError> {
        next_generator_u32(&mut self.generator, cancellation)
    }

    pub fn next_unit_f32(&mut self, cancellation: &CancellationToken) -> Result<f32, RngError> {
        let significand = self.next_u32(cancellation)? >> 8;
        Ok(significand as f32 / (1_u32 << 24) as f32)
    }

    pub fn next_unit_f64(&mut self, cancellation: &CancellationToken) -> Result<f64, RngError> {
        let mut generator = self.generator.clone();
        let high = u64::from(next_generator_u32(&mut generator, cancellation)? >> 5);
        let low = u64::from(next_generator_u32(&mut generator, cancellation)? >> 6);
        if cancellation.is_cancelled() {
            return Err(RngError::Cancelled);
        }
        self.generator = generator;
        Ok(((high << 26) | low) as f64 / (1_u64 << 53) as f64)
    }

    pub fn next_standard_normal_pair(
        &mut self,
        cancellation: &CancellationToken,
    ) -> Result<[f64; 2], RngError> {
        let mut generator = self.generator.clone();
        let first =
            (f64::from(next_generator_u32(&mut generator, cancellation)?) + 0.5) / 4_294_967_296.0;
        let second =
            (f64::from(next_generator_u32(&mut generator, cancellation)?) + 0.5) / 4_294_967_296.0;
        if cancellation.is_cancelled() {
            return Err(RngError::Cancelled);
        }
        self.generator = generator;
        let radius = (-2.0 * first.ln()).sqrt();
        let angle = std::f64::consts::TAU * second;
        Ok([radius * angle.cos(), radius * angle.sin()])
    }

    pub fn next_bounded_u64(
        &mut self,
        upper_exclusive: u64,
        cancellation: &CancellationToken,
    ) -> Result<u64, RngError> {
        if upper_exclusive == 0 {
            return Err(RngError::InvalidBound);
        }
        let mut generator = self.generator.clone();
        let rejection_threshold = upper_exclusive.wrapping_neg() % upper_exclusive;
        loop {
            let candidate = (u64::from(next_generator_u32(&mut generator, cancellation)?) << 32)
                | u64::from(next_generator_u32(&mut generator, cancellation)?);
            if candidate >= rejection_threshold {
                if cancellation.is_cancelled() {
                    return Err(RngError::Cancelled);
                }
                self.generator = generator;
                return Ok(candidate % upper_exclusive);
            }
        }
    }

    pub fn fill_bytes(
        &mut self,
        output: &mut [u8],
        cancellation: &CancellationToken,
    ) -> Result<(), RngError> {
        let mut generator = self.generator.clone();
        let mut generated = Vec::new();
        generated
            .try_reserve_exact(output.len())
            .map_err(|_| RngError::AllocationFailed {
                bytes: output.len(),
            })?;
        generated.resize(output.len(), 0);
        for chunk in generated.chunks_mut(4) {
            if cancellation.is_cancelled() {
                return Err(RngError::Cancelled);
            }
            let word = match &mut generator {
                Generator::Mt19937(generator) => generator.next_u32(),
                Generator::Philox4x32_10(generator) => generator.next_u32()?,
            }
            .to_le_bytes();
            let source = word.get(..chunk.len()).ok_or(RngError::CounterOverflow)?;
            chunk.copy_from_slice(source);
        }
        if cancellation.is_cancelled() {
            return Err(RngError::Cancelled);
        }
        self.generator = generator;
        output.copy_from_slice(&generated);
        Ok(())
    }

    pub fn checkpoint(&self) -> RngCheckpoint {
        let generator = match &self.generator {
            Generator::Mt19937(generator) => RngGeneratorSnapshot::Mt19937(generator.snapshot()),
            Generator::Philox4x32_10(generator) => {
                RngGeneratorSnapshot::Philox4x32_10(generator.snapshot())
            }
        };
        RngCheckpoint {
            profile: self.profile,
            algorithm: self.algorithm,
            address_digest: self.address_digest,
            device: self.address_device,
            generator,
        }
    }

    pub fn commit(self) -> RngCheckpoint {
        self.checkpoint()
    }

    pub fn abort(self) {}
}

fn next_generator_u32(
    generator: &mut Generator,
    cancellation: &CancellationToken,
) -> Result<u32, RngError> {
    if cancellation.is_cancelled() {
        return Err(RngError::Cancelled);
    }
    match generator {
        Generator::Mt19937(generator) => Ok(generator.next_u32()),
        Generator::Philox4x32_10(generator) => generator.next_u32(),
    }
}

#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum RngError {
    #[error("RNG stream address field {field} is empty")]
    InvalidAddress { field: String },
    #[error("RNG checkpoint does not match the stream profile, algorithm, or address")]
    CheckpointMismatch,
    #[error("RNG checkpoint is invalid: {reason}")]
    InvalidCheckpoint { reason: String },
    #[error("RNG counter overflowed")]
    CounterOverflow,
    #[error("RNG byte transaction could not reserve {bytes} bytes")]
    AllocationFailed { bytes: usize },
    #[error("RNG profile v1 is restricted to the legacy CPU stream identity")]
    LegacyProfileDeviceMismatch,
    #[error("RNG transaction device {actual:?} does not match output device {expected:?}")]
    DeviceMismatch {
        expected: DeviceId,
        actual: DeviceId,
    },
    #[error("RNG generation was cancelled before commit")]
    Cancelled,
    #[error("RNG bounded integer sampling requires a nonzero upper bound")]
    InvalidBound,
    #[error(
        "Sobol dimension {dimension} is unsupported; native profile v1 certifies dimensions 1 through 3"
    )]
    UnsupportedSobolDimension { dimension: usize },
    #[error("Sobol draw count overflowed the native 30-bit sequence")]
    SobolSequenceExhausted,
    #[error("Brownian interval is invalid: {reason}")]
    InvalidBrownianInterval { reason: String },
}

const SOBOL_MAX_BITS: usize = 30;
const SOBOL_SCALE: f64 = (1_u64 << SOBOL_MAX_BITS) as f64;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SobolEngine {
    dimension: usize,
    directions: Vec<[u32; SOBOL_MAX_BITS]>,
    shift: Vec<u32>,
    quasi: Vec<u32>,
    generated: u64,
}

impl SobolEngine {
    pub fn new(dimension: usize, scramble: bool, seed: u64) -> Result<Self, RngError> {
        if !(1..=3).contains(&dimension) {
            return Err(RngError::UnsupportedSobolDimension { dimension });
        }
        let mut directions = Vec::new();
        directions
            .try_reserve_exact(dimension)
            .map_err(|_| RngError::AllocationFailed {
                bytes: dimension.saturating_mul(std::mem::size_of::<[u32; SOBOL_MAX_BITS]>()),
            })?;
        for dimension_index in 0..dimension {
            let mut row = sobol_direction_row(dimension_index)?;
            if scramble {
                scramble_sobol_row(&mut row, seed, dimension_index);
            }
            directions.push(row);
        }
        let mut shift = Vec::new();
        shift
            .try_reserve_exact(dimension)
            .map_err(|_| RngError::AllocationFailed {
                bytes: dimension.saturating_mul(std::mem::size_of::<u32>()),
            })?;
        for dimension_index in 0..dimension {
            shift.push(if scramble {
                keyed_u64(seed, 0x534f_424f_4c53_4846, dimension_index as u64) as u32
                    & ((1_u32 << SOBOL_MAX_BITS) - 1)
            } else {
                0
            });
        }
        Ok(Self {
            dimension,
            quasi: shift.clone(),
            directions,
            shift,
            generated: 0,
        })
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }

    pub fn generated(&self) -> u64 {
        self.generated
    }

    pub fn reset(&mut self) {
        self.quasi.clone_from(&self.shift);
        self.generated = 0;
    }

    pub fn draw(
        &mut self,
        count: usize,
        cancellation: &CancellationToken,
    ) -> Result<Vec<f32>, RngError> {
        let requested = u64::try_from(count).map_err(|_| RngError::SobolSequenceExhausted)?;
        if self
            .generated
            .checked_add(requested)
            .is_none_or(|total| total > (1_u64 << SOBOL_MAX_BITS))
        {
            return Err(RngError::SobolSequenceExhausted);
        }
        let output_len = count
            .checked_mul(self.dimension)
            .ok_or(RngError::AllocationFailed { bytes: usize::MAX })?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(output_len)
            .map_err(|_| RngError::AllocationFailed {
                bytes: output_len.saturating_mul(std::mem::size_of::<f32>()),
            })?;
        let mut quasi = self.quasi.clone();
        let mut generated = self.generated;
        for sample_index in 0..count {
            if sample_index % 1_024 == 0 && cancellation.is_cancelled() {
                return Err(RngError::Cancelled);
            }
            if generated != 0 {
                let direction_index = usize::try_from((generated - 1).trailing_ones())
                    .map_err(|_| RngError::SobolSequenceExhausted)?;
                for (value, row) in quasi.iter_mut().zip(&self.directions) {
                    let direction = row
                        .get(direction_index)
                        .copied()
                        .ok_or(RngError::SobolSequenceExhausted)?;
                    *value ^= direction;
                }
            }
            output.extend(
                quasi
                    .iter()
                    .map(|value| (*value as f64 / SOBOL_SCALE) as f32),
            );
            generated += 1;
        }
        if cancellation.is_cancelled() {
            return Err(RngError::Cancelled);
        }
        self.quasi = quasi;
        self.generated = generated;
        Ok(output)
    }
}

fn sobol_direction_row(dimension: usize) -> Result<[u32; SOBOL_MAX_BITS], RngError> {
    let (degree, polynomial, initial): (usize, u32, &[u32]) = match dimension {
        0 => (0, 0, &[]),
        1 => (1, 3, &[1]),
        2 => (2, 7, &[1, 3]),
        _ => {
            return Err(RngError::UnsupportedSobolDimension {
                dimension: dimension + 1,
            });
        }
    };
    let mut unscaled = [0_u32; SOBOL_MAX_BITS];
    if dimension == 0 {
        unscaled.fill(1);
    } else {
        for (destination, source) in unscaled.iter_mut().zip(initial) {
            *destination = *source;
        }
        for index in degree..SOBOL_MAX_BITS {
            let mut value = unscaled[index - degree];
            let mut power = 1_u32;
            for offset in 0..degree {
                power <<= 1;
                if (polynomial >> (degree - 1 - offset)) & 1 != 0 {
                    value ^= power * unscaled[index - offset - 1];
                }
            }
            unscaled[index] = value;
        }
    }
    let mut directions = [0_u32; SOBOL_MAX_BITS];
    for (index, value) in unscaled.into_iter().enumerate() {
        directions[index] = value << (SOBOL_MAX_BITS - 1 - index);
    }
    Ok(directions)
}

fn scramble_sobol_row(directions: &mut [u32; SOBOL_MAX_BITS], seed: u64, dimension: usize) {
    for direction in directions {
        let original = *direction;
        let mut scrambled = 0_u32;
        for output_bit in 0..SOBOL_MAX_BITS {
            let source_limit = SOBOL_MAX_BITS - output_bit;
            let mut bit = (original >> (SOBOL_MAX_BITS - 1 - output_bit)) & 1;
            let random = keyed_u64(
                seed,
                0x534f_424f_4c4c_544d ^ dimension as u64,
                output_bit as u64,
            );
            for source_offset in 1..source_limit {
                if (random.rotate_left(source_offset as u32) & 1) != 0 {
                    bit ^= (original >> (SOBOL_MAX_BITS - 1 - output_bit - source_offset)) & 1;
                }
            }
            scrambled |= bit << (SOBOL_MAX_BITS - 1 - output_bit);
        }
        *direction = scrambled;
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrownianTree {
    start: f64,
    end: f64,
    entropy: u64,
    values: Vec<(f64, Vec<f64>)>,
}

impl BrownianTree {
    pub fn new(start: f64, initial: Vec<f64>, end: f64, entropy: u64) -> Result<Self, RngError> {
        if !start.is_finite() || !end.is_finite() || start >= end {
            return Err(RngError::InvalidBrownianInterval {
                reason: "start and end must be finite and strictly increasing".to_owned(),
            });
        }
        if initial.is_empty() || initial.iter().any(|value| !value.is_finite()) {
            return Err(RngError::InvalidBrownianInterval {
                reason: "initial value must be a nonempty finite vector".to_owned(),
            });
        }
        let scale = (end - start).sqrt();
        let terminal = initial
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value + scale * keyed_standard_normal(entropy, end.to_bits(), index as u64)
            })
            .collect();
        Ok(Self {
            start,
            end,
            entropy,
            values: vec![(start, initial), (end, terminal)],
        })
    }

    pub fn dimension(&self) -> usize {
        self.values.first().map_or(0, |(_, values)| values.len())
    }

    pub fn increment(
        &mut self,
        start: f64,
        end: f64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<f64>, RngError> {
        if cancellation.is_cancelled() {
            return Err(RngError::Cancelled);
        }
        if !start.is_finite()
            || !end.is_finite()
            || start < self.start
            || end > self.end
            || start > end
        {
            return Err(RngError::InvalidBrownianInterval {
                reason: "query must be finite, ordered, and contained by the tree".to_owned(),
            });
        }
        if start == end {
            return Ok(vec![0.0; self.dimension()]);
        }
        let mut candidate = self.clone();
        let start_value = candidate.value_at(start, cancellation)?;
        let end_value = candidate.value_at(end, cancellation)?;
        if cancellation.is_cancelled() {
            return Err(RngError::Cancelled);
        }
        let increment = end_value
            .iter()
            .zip(start_value)
            .map(|(end, start)| end - start)
            .collect();
        *self = candidate;
        Ok(increment)
    }

    fn value_at(
        &mut self,
        time: f64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<f64>, RngError> {
        match self
            .values
            .binary_search_by(|(candidate, _)| candidate.total_cmp(&time))
        {
            Ok(index) => self
                .values
                .get(index)
                .map(|(_, values)| values.clone())
                .ok_or_else(|| RngError::InvalidBrownianInterval {
                    reason: "cached Brownian value is unavailable".to_owned(),
                }),
            Err(index) => {
                if cancellation.is_cancelled() || index == 0 || index >= self.values.len() {
                    return Err(if cancellation.is_cancelled() {
                        RngError::Cancelled
                    } else {
                        RngError::InvalidBrownianInterval {
                            reason: "query is outside the cached root interval".to_owned(),
                        }
                    });
                }
                let (left_time, left) = self.values.get(index - 1).cloned().ok_or_else(|| {
                    RngError::InvalidBrownianInterval {
                        reason: "left Brownian bridge endpoint is unavailable".to_owned(),
                    }
                })?;
                let (right_time, right) = self.values.get(index).cloned().ok_or_else(|| {
                    RngError::InvalidBrownianInterval {
                        reason: "right Brownian bridge endpoint is unavailable".to_owned(),
                    }
                })?;
                let width = right_time - left_time;
                let left_weight = (right_time - time) / width;
                let right_weight = (time - left_time) / width;
                let standard_deviation = ((time - left_time) * (right_time - time) / width).sqrt();
                let value = left
                    .iter()
                    .zip(right)
                    .enumerate()
                    .map(|(dimension, (left, right))| {
                        left_weight * left
                            + right_weight * right
                            + standard_deviation
                                * keyed_standard_normal(
                                    self.entropy,
                                    time.to_bits(),
                                    dimension as u64,
                                )
                    })
                    .collect::<Vec<_>>();
                self.values.insert(index, (time, value.clone()));
                Ok(value)
            }
        }
    }
}

fn keyed_standard_normal(seed: u64, domain: u64, index: u64) -> f64 {
    let first = keyed_u64(seed, domain, index);
    let second = keyed_u64(seed, domain ^ 0x9e37_79b9_7f4a_7c15, index);
    let unit_1 = (first as f64 + 0.5) / (u64::MAX as f64 + 1.0);
    let unit_2 = (second as f64 + 0.5) / (u64::MAX as f64 + 1.0);
    (-2.0 * unit_1.ln()).sqrt() * (std::f64::consts::TAU * unit_2).cos()
}

fn keyed_u64(seed: u64, domain: u64, index: u64) -> u64 {
    mix64(seed ^ mix64(domain) ^ mix64(index))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation_artifacts;
    use std::{
        collections::{BTreeMap, BTreeSet},
        error::Error,
    };

    fn address(retry: u32, retry_policy: RetryRngPolicy) -> RngStreamAddress {
        match RngStreamAddress::new(
            "workflow",
            "attempt",
            "node",
            0,
            "noise",
            0,
            retry,
            retry_policy,
        ) {
            Ok(value) => value,
            Err(error) => panic!("test address failed: {error}"),
        }
    }

    #[test]
    fn mt19937_matches_reference_vector() {
        let mut generator = Mt19937::from_seed(5489);
        assert_eq!(
            (0..5).map(|_| generator.next_u32()).collect::<Vec<_>>(),
            vec![
                3_499_211_612,
                581_869_302,
                3_890_346_734,
                3_586_334_585,
                545_404_204,
            ]
        );
    }

    #[test]
    fn numpy_random_state_randint_matches_legacy_interval_sampling() {
        let cancellation = CancellationToken::default();
        let mut state = NumpyRandomState::from_seed(0);
        let values = (0..10)
            .map(|_| state.randint(0, 10, &cancellation))
            .collect::<Result<Vec<_>, _>>()
            .expect("legacy NumPy bounded draws should succeed");
        assert_eq!(values, vec![5, 0, 3, 3, 7, 9, 3, 5, 2, 4]);

        let mut wrapped_seed = NumpyRandomState::from_seed(u64::from(u32::MAX) + 7);
        let mut canonical_seed = NumpyRandomState::from_seed(7);
        assert_eq!(
            wrapped_seed.randint(2, 9, &cancellation),
            canonical_seed.randint(2, 9, &cancellation)
        );
        assert!(matches!(
            state.randint(4, 4, &cancellation),
            Err(RngError::InvalidBound)
        ));
        cancellation.cancel();
        assert!(matches!(
            state.randint(0, 2, &cancellation),
            Err(RngError::Cancelled)
        ));
    }

    #[test]
    fn philox_matches_random123_reference_vector() {
        assert_eq!(
            Philox4x32::generate([0; 4], [0; 2]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
    }

    #[test]
    fn phase_and_retry_identity_are_explicit() {
        let replay_zero = address(0, RetryRngPolicy::Replay).digest(RngProfileVersion::V1, 7);
        let replay_one = address(1, RetryRngPolicy::Replay).digest(RngProfileVersion::V1, 7);
        let advance_zero = address(0, RetryRngPolicy::Advance).digest(RngProfileVersion::V1, 7);
        let advance_one = address(1, RetryRngPolicy::Advance).digest(RngProfileVersion::V1, 7);
        assert_eq!(replay_zero, replay_one);
        assert_ne!(advance_zero, advance_one);
        assert_ne!(replay_zero, advance_zero);
    }

    #[test]
    fn checkpoint_resumes_without_global_state() {
        let token = CancellationToken::default();
        let stream = RngStream::new(
            RngProfileVersion::V1,
            RngAlgorithm::Philox4x32_10,
            19,
            address(0, RetryRngPolicy::Replay),
        )
        .expect("legacy CPU stream is valid");
        let Ok(mut first) = stream.begin(None) else {
            panic!("stream should begin");
        };
        let Ok(_) = first.next_u32(&token) else {
            panic!("first word should generate");
        };
        let checkpoint = first.commit();
        let Ok(mut resumed) = stream.begin(Some(checkpoint.clone())) else {
            panic!("checkpoint should resume");
        };
        let Ok(resumed_word) = resumed.next_u32(&token) else {
            panic!("resumed word should generate");
        };

        let Ok(mut replay) = stream.begin(None) else {
            panic!("stream should replay");
        };
        let Ok(_) = replay.next_u32(&token) else {
            panic!("first replay word should generate");
        };
        let Ok(replayed_word) = replay.next_u32(&token) else {
            panic!("second replay word should generate");
        };
        assert_eq!(resumed_word, replayed_word);

        let other_stream = RngStream::new(
            RngProfileVersion::V1,
            RngAlgorithm::Philox4x32_10,
            19,
            address(1, RetryRngPolicy::Advance),
        )
        .expect("legacy CPU stream is valid");
        assert!(matches!(
            other_stream.begin(Some(checkpoint)),
            Err(RngError::CheckpointMismatch)
        ));
    }

    #[test]
    fn cancellation_prevents_advancement_from_being_committed() {
        let token = CancellationToken::default();
        let stream = RngStream::new(
            RngProfileVersion::V1,
            RngAlgorithm::Mt19937,
            23,
            address(0, RetryRngPolicy::Replay),
        )
        .expect("legacy CPU stream is valid");
        let Ok(mut transaction) = stream.begin(None) else {
            panic!("stream should begin");
        };
        token.cancel();
        assert_eq!(transaction.next_u32(&token), Err(RngError::Cancelled));
        transaction.abort();
        let fresh = stream.begin(None);
        assert!(fresh.is_ok());
    }

    #[test]
    fn address_wire_conversion_validates_and_migrates_legacy_cpu_addresses() {
        let legacy = RngStreamAddressWire {
            workflow: "workflow".to_owned(),
            attempt: "attempt".to_owned(),
            node: "node".to_owned(),
            output: 0,
            phase: "noise".to_owned(),
            batch: 0,
            retry: 0,
            retry_policy: RetryRngPolicy::Replay,
            device: None,
        };
        assert!(matches!(
            RngStreamAddress::try_from(legacy),
            Ok(address) if address.device() == DeviceId::CPU
        ));

        let invalid = RngStreamAddressWire {
            workflow: " ".to_owned(),
            attempt: "attempt".to_owned(),
            node: "node".to_owned(),
            output: 0,
            phase: "noise".to_owned(),
            batch: 0,
            retry: 0,
            retry_policy: RetryRngPolicy::Replay,
            device: Some(DeviceId::CPU),
        };
        assert!(matches!(
            RngStreamAddress::try_from(invalid),
            Err(RngError::InvalidAddress { .. })
        ));

        let generator = RngGeneratorSnapshot::Mt19937(Mt19937::from_seed(1).snapshot());
        let legacy_checkpoint = RngCheckpoint::try_from(RngCheckpointWire {
            profile: RngProfileVersion::V1,
            algorithm: RngAlgorithm::Mt19937,
            address_digest: [1, 2],
            device: None,
            generator: generator.clone(),
        });
        assert!(matches!(
            legacy_checkpoint,
            Ok(checkpoint) if checkpoint.device == DeviceId::CPU
        ));
        assert!(matches!(
            RngCheckpoint::try_from(RngCheckpointWire {
                profile: RngProfileVersion::V2,
                algorithm: RngAlgorithm::Mt19937,
                address_digest: [1, 2],
                device: None,
                generator,
            }),
            Err(RngError::InvalidCheckpoint { .. })
        ));

        let non_cpu_legacy_address = RngStreamAddress::for_device(
            "workflow",
            "attempt",
            "node",
            0,
            "noise",
            0,
            0,
            RetryRngPolicy::Replay,
            DeviceId::new(DeviceKind::Cuda, 0),
        )
        .expect("non-CPU address is valid for v2");
        assert!(matches!(
            RngStream::new(
                RngProfileVersion::V1,
                RngAlgorithm::Mt19937,
                1,
                non_cpu_legacy_address,
            ),
            Err(RngError::LegacyProfileDeviceMismatch)
        ));
    }

    #[test]
    fn fill_bytes_rolls_back_output_and_generator_on_error() {
        let stream = RngStream::new(
            RngProfileVersion::V2,
            RngAlgorithm::Philox4x32_10,
            31,
            address(0, RetryRngPolicy::Replay),
        )
        .expect("v2 CPU stream is valid");
        let checkpoint = RngCheckpoint {
            profile: RngProfileVersion::V2,
            algorithm: RngAlgorithm::Philox4x32_10,
            address_digest: stream.address.digest(RngProfileVersion::V2, 31),
            device: DeviceId::CPU,
            generator: RngGeneratorSnapshot::Philox4x32_10(PhiloxSnapshot {
                counter: [u32::MAX; 4],
                key: [0; 2],
                block: Philox4x32::generate([u32::MAX - 1, u32::MAX, u32::MAX, u32::MAX], [0; 2]),
                block_index: 3,
                counter_exhausted: false,
            }),
        };
        let mut transaction = stream
            .begin(Some(checkpoint.clone()))
            .expect("validation checkpoint begins");
        let mut output = [0xa5; 24];
        assert_eq!(
            transaction.fill_bytes(&mut output, &CancellationToken::default()),
            Err(RngError::CounterOverflow)
        );
        assert_eq!(output, [0xa5; 24]);
        assert_eq!(transaction.commit(), checkpoint);
    }

    #[test]
    fn philox_generates_the_last_counter_block_once_and_then_stays_exhausted() {
        let key = [0x1234_5678, 0x9abc_def0];
        let expected = Philox4x32::generate([u32::MAX; 4], key);
        let mut cursor = PhiloxCursor::new([u32::MAX; 4], key);

        for expected_word in expected {
            assert_eq!(cursor.next_u32(), Ok(expected_word));
        }

        let terminal = cursor.snapshot();
        assert_eq!(terminal.counter, [u32::MAX; 4]);
        assert_eq!(terminal.block_index, 4);
        assert!(terminal.counter_exhausted);
        assert_eq!(cursor.next_u32(), Err(RngError::CounterOverflow));
        assert_eq!(cursor.snapshot(), terminal);
        assert_eq!(cursor.next_u32(), Err(RngError::CounterOverflow));
        assert_eq!(cursor.snapshot(), terminal);
    }

    #[test]
    fn legacy_max_counter_checkpoint_deserializes_as_final_block_pending() {
        let stream = RngStream::new(
            RngProfileVersion::V2,
            RngAlgorithm::Philox4x32_10,
            31,
            address(0, RetryRngPolicy::Replay),
        )
        .expect("v2 CPU stream is valid");
        let key = [17, 29];
        let pending = RngCheckpoint {
            profile: RngProfileVersion::V2,
            algorithm: RngAlgorithm::Philox4x32_10,
            address_digest: stream.address.digest(RngProfileVersion::V2, 31),
            device: DeviceId::CPU,
            generator: RngGeneratorSnapshot::Philox4x32_10(PhiloxSnapshot {
                counter: [u32::MAX; 4],
                key,
                block: [0; 4],
                block_index: 4,
                counter_exhausted: false,
            }),
        };
        let mut serialized =
            serde_json::to_value(&pending).expect("checkpoint should serialize for migration test");
        let Some(generator) = serialized
            .pointer_mut("/generator/state")
            .and_then(serde_json::Value::as_object_mut)
        else {
            panic!("serialized checkpoint should contain Philox state");
        };
        assert!(generator.remove("counter_exhausted").is_some());
        let deserialized: RngCheckpoint =
            serde_json::from_value(serialized).expect("legacy checkpoint should deserialize");
        assert_eq!(deserialized, pending);

        let mut transaction = stream
            .begin(Some(deserialized))
            .expect("legacy checkpoint should resume");
        let expected = Philox4x32::generate([u32::MAX; 4], key);
        for expected_word in expected {
            assert_eq!(
                transaction.next_u32(&CancellationToken::default()),
                Ok(expected_word)
            );
        }
        assert_eq!(
            transaction.next_u32(&CancellationToken::default()),
            Err(RngError::CounterOverflow)
        );

        let final_block = Philox4x32::generate([u32::MAX; 4], key);
        let legacy_partial = RngCheckpoint {
            profile: RngProfileVersion::V2,
            algorithm: RngAlgorithm::Philox4x32_10,
            address_digest: stream.address.digest(RngProfileVersion::V2, 31),
            device: DeviceId::CPU,
            generator: RngGeneratorSnapshot::Philox4x32_10(PhiloxSnapshot {
                counter: [u32::MAX; 4],
                key,
                block: final_block,
                block_index: 2,
                counter_exhausted: false,
            }),
        };
        let mut serialized = serde_json::to_value(legacy_partial)
            .expect("partial legacy checkpoint should serialize");
        let Some(generator) = serialized
            .pointer_mut("/generator/state")
            .and_then(serde_json::Value::as_object_mut)
        else {
            panic!("partial legacy checkpoint should contain Philox state");
        };
        assert!(generator.remove("counter_exhausted").is_some());
        let normalized: RngCheckpoint = serde_json::from_value(serialized)
            .expect("partial legacy checkpoint should deserialize");
        assert!(matches!(
            &normalized.generator,
            RngGeneratorSnapshot::Philox4x32_10(snapshot) if snapshot.counter_exhausted
        ));
        let mut transaction = stream
            .begin(Some(normalized))
            .expect("partial legacy checkpoint should resume");
        assert_eq!(
            transaction.next_u32(&CancellationToken::default()),
            Ok(final_block[2])
        );
        assert_eq!(
            transaction.next_u32(&CancellationToken::default()),
            Ok(final_block[3])
        );
        assert_eq!(
            transaction.next_u32(&CancellationToken::default()),
            Err(RngError::CounterOverflow)
        );
    }

    #[test]
    fn philox_terminal_checkpoint_commit_and_faults_are_transactional() {
        let token = CancellationToken::default();
        let stream = RngStream::new(
            RngProfileVersion::V2,
            RngAlgorithm::Philox4x32_10,
            37,
            address(0, RetryRngPolicy::Replay),
        )
        .expect("v2 CPU stream is valid");
        let key = [41, 43];
        let pending = RngCheckpoint {
            profile: RngProfileVersion::V2,
            algorithm: RngAlgorithm::Philox4x32_10,
            address_digest: stream.address.digest(RngProfileVersion::V2, 37),
            device: DeviceId::CPU,
            generator: RngGeneratorSnapshot::Philox4x32_10(PhiloxSnapshot {
                counter: [u32::MAX; 4],
                key,
                block: [0; 4],
                block_index: 4,
                counter_exhausted: false,
            }),
        };

        let mut successful = stream
            .begin(Some(pending.clone()))
            .expect("pending final block should resume");
        let mut output = [0; 16];
        successful
            .fill_bytes(&mut output, &token)
            .expect("the final block should be available");
        let expected = Philox4x32::generate([u32::MAX; 4], key)
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(output.as_slice(), expected.as_slice());
        let terminal = successful.commit();
        assert!(matches!(
            &terminal.generator,
            RngGeneratorSnapshot::Philox4x32_10(snapshot)
                if snapshot.counter_exhausted && snapshot.block_index == 4
        ));

        let mut exhausted = stream
            .begin(Some(terminal.clone()))
            .expect("terminal checkpoint should resume");
        assert_eq!(exhausted.next_u32(&token), Err(RngError::CounterOverflow));
        assert_eq!(exhausted.next_u32(&token), Err(RngError::CounterOverflow));
        assert_eq!(exhausted.commit(), terminal);

        let mut faulted = stream
            .begin(Some(pending.clone()))
            .expect("pending final block should resume");
        let mut oversized_output = [0xa5; 20];
        assert_eq!(
            faulted.fill_bytes(&mut oversized_output, &token),
            Err(RngError::CounterOverflow)
        );
        assert_eq!(oversized_output, [0xa5; 20]);
        assert_eq!(faulted.commit(), pending);
    }

    #[test]
    fn philox_cancellation_preserves_pending_terminal_checkpoint() {
        let stream = RngStream::new(
            RngProfileVersion::V2,
            RngAlgorithm::Philox4x32_10,
            47,
            address(0, RetryRngPolicy::Replay),
        )
        .expect("v2 CPU stream is valid");
        let pending = RngCheckpoint {
            profile: RngProfileVersion::V2,
            algorithm: RngAlgorithm::Philox4x32_10,
            address_digest: stream.address.digest(RngProfileVersion::V2, 47),
            device: DeviceId::CPU,
            generator: RngGeneratorSnapshot::Philox4x32_10(PhiloxSnapshot {
                counter: [u32::MAX; 4],
                key: [53, 59],
                block: [0; 4],
                block_index: 4,
                counter_exhausted: false,
            }),
        };
        let cancellation = CancellationToken::default();
        cancellation.cancel();

        let mut transaction = stream
            .begin(Some(pending.clone()))
            .expect("pending final block should resume");
        assert_eq!(
            transaction.next_u32(&cancellation),
            Err(RngError::Cancelled)
        );
        let mut output = [0x5a; 4];
        assert_eq!(
            transaction.fill_bytes(&mut output, &cancellation),
            Err(RngError::Cancelled)
        );
        assert_eq!(output, [0x5a; 4]);
        assert_eq!(transaction.commit(), pending);
    }

    #[test]
    fn invalid_philox_terminal_snapshot_is_rejected() {
        assert!(matches!(
            PhiloxCursor::from_snapshot(PhiloxSnapshot {
                counter: [u32::MAX; 4],
                key: [61, 67],
                block: [0; 4],
                block_index: 4,
                counter_exhausted: true,
            }),
            Err(RngError::InvalidCheckpoint { .. })
        ));
    }

    #[test]
    fn val_rng_001() -> Result<(), Box<dyn Error>> {
        let mt_reference = {
            let mut generator = Mt19937::from_seed(5489);
            (0..5).map(|_| generator.next_u32()).collect::<Vec<_>>()
                == vec![
                    3_499_211_612,
                    581_869_302,
                    3_890_346_734,
                    3_586_334_585,
                    545_404_204,
                ]
        };
        let philox_reference = Philox4x32::generate([0; 4], [0; 2])
            == [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8];

        let cpu_address = RngStreamAddress::for_device(
            "workflow",
            "attempt",
            "node",
            0,
            "noise",
            0,
            0,
            RetryRngPolicy::Replay,
            DeviceId::CPU,
        )?;
        let cuda_device = DeviceId::new(DeviceKind::Cuda, 0);
        let cuda_address = RngStreamAddress::for_device(
            "workflow",
            "attempt",
            "node",
            0,
            "noise",
            0,
            0,
            RetryRngPolicy::Replay,
            cuda_device,
        )?;
        let device_identity_separates_v2 = cpu_address.digest(RngProfileVersion::V2, 7)
            != cuda_address.digest(RngProfileVersion::V2, 7);
        let legacy_non_cpu_rejected = matches!(
            RngStream::new(
                RngProfileVersion::V1,
                RngAlgorithm::Philox4x32_10,
                7,
                cuda_address.clone(),
            ),
            Err(RngError::LegacyProfileDeviceMismatch)
        );
        let missing_v2_device_rejected = matches!(
            RngCheckpoint::try_from(RngCheckpointWire {
                profile: RngProfileVersion::V2,
                algorithm: RngAlgorithm::Mt19937,
                address_digest: [0, 0],
                device: None,
                generator: RngGeneratorSnapshot::Mt19937(Mt19937::from_seed(7).snapshot()),
            }),
            Err(RngError::InvalidCheckpoint { .. })
        );

        let cpu_stream = RngStream::new(
            RngProfileVersion::V2,
            RngAlgorithm::Philox4x32_10,
            7,
            cpu_address,
        )?;
        let checkpoint = cpu_stream.begin(None)?.commit();
        let cuda_stream = RngStream::new(
            RngProfileVersion::V2,
            RngAlgorithm::Philox4x32_10,
            7,
            cuda_address,
        )?;
        let checkpoint_device_separation = matches!(
            cuda_stream.begin(Some(checkpoint)),
            Err(RngError::CheckpointMismatch)
        );
        let cuda_transaction = cuda_stream.begin(None)?;
        let transaction_device_validation_is_canonical = cuda_transaction.device() == cuda_device
            && cuda_transaction.require_device(cuda_device).is_ok()
            && matches!(
                cuda_transaction.require_device(DeviceId::CPU),
                Err(RngError::DeviceMismatch {
                    expected: DeviceId::CPU,
                    actual,
                }) if actual == cuda_device
            );

        let rollback_stream = RngStream::new(
            RngProfileVersion::V2,
            RngAlgorithm::Philox4x32_10,
            31,
            address(0, RetryRngPolicy::Replay),
        )?;
        let rollback_checkpoint = RngCheckpoint {
            profile: RngProfileVersion::V2,
            algorithm: RngAlgorithm::Philox4x32_10,
            address_digest: rollback_stream.address.digest(RngProfileVersion::V2, 31),
            device: DeviceId::CPU,
            generator: RngGeneratorSnapshot::Philox4x32_10(PhiloxSnapshot {
                counter: [u32::MAX; 4],
                key: [0; 2],
                block: Philox4x32::generate([u32::MAX - 1, u32::MAX, u32::MAX, u32::MAX], [0; 2]),
                block_index: 3,
                counter_exhausted: false,
            }),
        };
        let mut rollback = rollback_stream.begin(Some(rollback_checkpoint.clone()))?;
        let mut output = [0x5a; 24];
        let transactional_fill = matches!(
            rollback.fill_bytes(&mut output, &CancellationToken::default()),
            Err(RngError::CounterOverflow)
        ) && output == [0x5a; 24]
            && rollback.commit() == rollback_checkpoint;

        let forged_partial_block_rejected = matches!(
            PhiloxCursor::from_snapshot(PhiloxSnapshot {
                counter: [17, 0, 0, 0],
                key: [19, 23],
                block: [0; 4],
                block_index: 2,
                counter_exhausted: false,
            }),
            Err(RngError::InvalidCheckpoint { .. })
        );
        let terminal_checkpoint = RngCheckpoint {
            profile: RngProfileVersion::V2,
            algorithm: RngAlgorithm::Philox4x32_10,
            address_digest: rollback_stream.address.digest(RngProfileVersion::V2, 31),
            device: DeviceId::CPU,
            generator: RngGeneratorSnapshot::Philox4x32_10(PhiloxSnapshot {
                counter: [u32::MAX; 4],
                key: [29, 31],
                block: Philox4x32::generate([u32::MAX; 4], [29, 31]),
                block_index: 3,
                counter_exhausted: true,
            }),
        };
        let mut unit = rollback_stream.begin(Some(terminal_checkpoint.clone()))?;
        let unit_fault_is_transactional = matches!(
            unit.next_unit_f64(&CancellationToken::default()),
            Err(RngError::CounterOverflow)
        ) && unit.commit() == terminal_checkpoint;
        let mut normal = rollback_stream.begin(Some(terminal_checkpoint.clone()))?;
        let normal_fault_is_transactional = matches!(
            normal.next_standard_normal_pair(&CancellationToken::default()),
            Err(RngError::CounterOverflow)
        ) && normal.commit() == terminal_checkpoint;
        let mut bounded = rollback_stream.begin(Some(terminal_checkpoint.clone()))?;
        let bounded_fault_is_transactional = matches!(
            bounded.next_bounded_u64(17, &CancellationToken::default()),
            Err(RngError::CounterOverflow)
        ) && bounded.commit() == terminal_checkpoint;
        let multiword_faults_are_transactional = unit_fault_is_transactional
            && normal_fault_is_transactional
            && bounded_fault_is_transactional;

        let cancellation_stream = RngStream::new(
            RngProfileVersion::V2,
            RngAlgorithm::Mt19937,
            17,
            address(0, RetryRngPolicy::Replay),
        )?;
        let mut cancelled_transaction = cancellation_stream.begin(None)?;
        let cancellation = CancellationToken::default();
        cancellation.cancel();
        let mut cancelled_output = [0x3c; 8];
        let cancelled_fill_is_transactional = matches!(
            cancelled_transaction.fill_bytes(&mut cancelled_output, &cancellation),
            Err(RngError::Cancelled)
        ) && cancelled_output == [0x3c; 8]
            && cancelled_transaction.commit() == cancellation_stream.begin(None)?.commit();

        let replay_zero = address(0, RetryRngPolicy::Replay).digest(RngProfileVersion::V2, 7);
        let replay_one = address(1, RetryRngPolicy::Replay).digest(RngProfileVersion::V2, 7);
        let advance_zero = address(0, RetryRngPolicy::Advance).digest(RngProfileVersion::V2, 7);
        let advance_one = address(1, RetryRngPolicy::Advance).digest(RngProfileVersion::V2, 7);
        let retry_policy_is_explicit =
            replay_zero == replay_one && advance_zero != advance_one && replay_zero != advance_zero;
        let compatibility_ids = RNG_COMPATIBILITY_CONTRACTS
            .iter()
            .map(|contract| contract.rng_id())
            .collect::<BTreeSet<_>>();
        let compatibility_phases = RNG_COMPATIBILITY_CONTRACTS
            .iter()
            .map(|contract| contract.phase())
            .collect::<BTreeSet<_>>();
        let catalog_compatibility_is_closed = RNG_COMPATIBILITY_CONTRACTS.len() == 54
            && compatibility_ids.len() == 54
            && compatibility_phases.len() == 9
            && RNG_COMPATIBILITY_CONTRACTS.iter().all(|contract| {
                contract.rng_id().starts_with("COMFY-RNG-") && !contract.symbol().is_empty()
            });
        let execution_profiles_are_versioned = NativeRngExecutionProfile::CpuMt19937V1.algorithm()
            == RngAlgorithm::Mt19937
            && NativeRngExecutionProfile::CpuMt19937V1.stream_profile() == RngProfileVersion::V1
            && NativeRngExecutionProfile::DevicePhilox4x32_10V1.algorithm()
                == RngAlgorithm::Philox4x32_10
            && NativeRngExecutionProfile::DevicePhilox4x32_10V1.stream_profile()
                == RngProfileVersion::V2;

        let cases = BTreeMap::from([
            (
                "checkpoint_device_identity_is_separate",
                checkpoint_device_separation,
            ),
            (
                "cataloged_compatibility_contracts_are_closed",
                catalog_compatibility_is_closed,
            ),
            (
                "device_identity_separates_v2_streams",
                device_identity_separates_v2,
            ),
            (
                "transaction_device_validation_is_canonical",
                transaction_device_validation_is_canonical,
            ),
            ("fill_bytes_is_transactional", transactional_fill),
            (
                "forged_partial_philox_blocks_are_rejected",
                forged_partial_block_rejected,
            ),
            (
                "multiword_faults_do_not_publish_partial_advancement",
                multiword_faults_are_transactional,
            ),
            (
                "cancelled_fill_does_not_publish_or_advance",
                cancelled_fill_is_transactional,
            ),
            (
                "legacy_profile_rejects_non_cpu_identity",
                legacy_non_cpu_rejected,
            ),
            ("retry_policy_is_explicit", retry_policy_is_explicit),
            (
                "rng_execution_profiles_are_versioned",
                execution_profiles_are_versioned,
            ),
            (
                "v2_checkpoint_requires_device_identity",
                missing_v2_device_rejected,
            ),
            ("mt19937_matches_reference_vector", mt_reference),
            (
                "philox_matches_random123_reference_vector",
                philox_reference,
            ),
            (
                "wire_address_invariants_are_checked",
                RngStreamAddress::try_from(RngStreamAddressWire {
                    workflow: String::new(),
                    attempt: "attempt".to_owned(),
                    node: "node".to_owned(),
                    output: 0,
                    phase: "noise".to_owned(),
                    batch: 0,
                    retry: 0,
                    retry_policy: RetryRngPolicy::Replay,
                    device: None,
                })
                .is_err(),
            ),
        ]);
        let fixture_path = ".agents/specs/comfy-parity/catalogs/backend-rng.csv";
        let fixture_digest = validation_artifacts::workspace_fixture_digest(
            fixture_path,
            "d207ea66d8949eb73067828da6f2ed160ab8bdf641b4cf6ed1789faa0f65d06b",
        )?;
        let fixture_digests = BTreeMap::from([(fixture_path, fixture_digest.as_str())]);
        validation_artifacts::write(
            "val-rng-001.json",
            "VAL-RNG-001",
            "sim-parity-native-rng-breadth native RNG vectors, 54 compatibility contracts, stream identity, device validation, checkpoints, retry, and transactional execution",
            "sim-parity-native-rng-breadth-closure",
            &fixture_digests,
            &cases,
            &["comfy-parity-final-validation"],
        )
    }
}
