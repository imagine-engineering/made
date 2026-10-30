//! Versioned registry for the four native text-encoder architecture owners.
//!
//! This module owns only source-to-owner routing and the deterministic registry
//! identity. Execution remains in the four focused owner modules.

use crate::{
    BERT_SOURCE_PATH, BERT_SOURCE_SHA256, COMPOSITE_TEXT_ENCODER_CONTRACTS,
    DECODER_TEXT_ENCODER_CATALOG_SYMBOLS, GEMMA4_SOURCE_PATH, GEMMA4_SOURCE_SHA256,
    GPT_OSS_SOURCE_PATH, GPT_OSS_SOURCE_SHA256, IDEOGRAM4_SOURCE_PATH, IDEOGRAM4_SOURCE_SHA256,
    JINA_CLIP2_SOURCE_PATH, JINA_CLIP2_SOURCE_SHA256, LLAMA_SOURCE_PATH, LLAMA_SOURCE_SHA256,
    MULTIMODAL_TEXT_ENCODER_CATALOG_SYMBOLS, OVIS_SOURCE_PATH, OVIS_SOURCE_SHA256,
    QWEN_VL_SOURCE_PATH, QWEN_VL_SOURCE_SHA256, QWEN3VL_SOURCE_PATH, QWEN3VL_SOURCE_SHA256,
    QWEN35_SOURCE_PATH, QWEN35_SOURCE_SHA256, SAM3_CLIP_SOURCE_PATH, SAM3_CLIP_SOURCE_SHA256,
    SPIECE_TOKENIZER_SOURCE_PATH, SPIECE_TOKENIZER_SOURCE_SHA256, T5_BIDIRECTIONAL_CATALOG_SYMBOLS,
    T5_SOURCE_PATH, T5_SOURCE_SHA256,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub const TEXT_ENCODER_ARCHITECTURE_REGISTRY_VERSION: u16 = 1;
pub const TEXT_ENCODER_ARCHITECTURE_CONTRACT_COUNT: usize = 398;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum TextEncoderArchitectureOwner {
    BidirectionalT5,
    Decoder,
    Multimodal,
    Composite,
}

impl TextEncoderArchitectureOwner {
    pub const fn native_owner(self) -> &'static str {
        match self {
            Self::BidirectionalT5 => "sim_model::clip_text_encoder_t5",
            Self::Decoder => "sim_model::clip_text_encoder_decoder",
            Self::Multimodal => "sim_model::clip_text_encoder_multimodal",
            Self::Composite => "sim_model::clip_text_encoder_composite",
        }
    }

    pub const fn implementation_task(self) -> &'static str {
        match self {
            Self::BidirectionalT5 => "comfy-parity-clip-text-encoder-t5-foundation",
            Self::Decoder => "comfy-parity-clip-text-encoder-decoder-foundation",
            Self::Multimodal => "comfy-parity-clip-text-encoder-multimodal-foundation",
            Self::Composite => "comfy-parity-clip-text-encoder-composite-adapters",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextEncoderOwnerFact {
    pub owner: TextEncoderArchitectureOwner,
    pub native_owner: &'static str,
    pub implementation_task: &'static str,
    pub contract_count: usize,
}

pub const TEXT_ENCODER_OWNER_FACTS: [TextEncoderOwnerFact; 4] = [
    TextEncoderOwnerFact {
        owner: TextEncoderArchitectureOwner::BidirectionalT5,
        native_owner: "sim_model::clip_text_encoder_t5",
        implementation_task: "comfy-parity-clip-text-encoder-t5-foundation",
        contract_count: 19,
    },
    TextEncoderOwnerFact {
        owner: TextEncoderArchitectureOwner::Decoder,
        native_owner: "sim_model::clip_text_encoder_decoder",
        implementation_task: "comfy-parity-clip-text-encoder-decoder-foundation",
        contract_count: 127,
    },
    TextEncoderOwnerFact {
        owner: TextEncoderArchitectureOwner::Multimodal,
        native_owner: "sim_model::clip_text_encoder_multimodal",
        implementation_task: "comfy-parity-clip-text-encoder-multimodal-foundation",
        contract_count: 53,
    },
    TextEncoderOwnerFact {
        owner: TextEncoderArchitectureOwner::Composite,
        native_owner: "sim_model::clip_text_encoder_composite",
        implementation_task: "comfy-parity-clip-text-encoder-composite-adapters",
        contract_count: 199,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextEncoderSourceSegment {
    pub owner: TextEncoderArchitectureOwner,
    pub source_path: &'static str,
    pub source_sha256: &'static str,
    symbols: &'static [&'static str],
    symbol_start: usize,
    symbol_end: usize,
}

impl TextEncoderSourceSegment {
    pub fn symbols(self) -> &'static [&'static str] {
        &self.symbols[self.symbol_start..self.symbol_end]
    }
}

const BIDIRECTIONAL_SYMBOLS: &[&str] = &T5_BIDIRECTIONAL_CATALOG_SYMBOLS;
const DECODER_SYMBOLS: &[&str] = &DECODER_TEXT_ENCODER_CATALOG_SYMBOLS;
const MULTIMODAL_SYMBOLS: &[&str] = &MULTIMODAL_TEXT_ENCODER_CATALOG_SYMBOLS;

pub const TEXT_ENCODER_SOURCE_SEGMENTS: [TextEncoderSourceSegment; 13] = [
    TextEncoderSourceSegment {
        owner: TextEncoderArchitectureOwner::BidirectionalT5,
        source_path: BERT_SOURCE_PATH,
        source_sha256: BERT_SOURCE_SHA256,
        symbols: BIDIRECTIONAL_SYMBOLS,
        symbol_start: 0,
        symbol_end: 9,
    },
    TextEncoderSourceSegment {
        owner: TextEncoderArchitectureOwner::BidirectionalT5,
        source_path: SPIECE_TOKENIZER_SOURCE_PATH,
        source_sha256: SPIECE_TOKENIZER_SOURCE_SHA256,
        symbols: BIDIRECTIONAL_SYMBOLS,
        symbol_start: 9,
        symbol_end: 10,
    },
    TextEncoderSourceSegment {
        owner: TextEncoderArchitectureOwner::BidirectionalT5,
        source_path: T5_SOURCE_PATH,
        source_sha256: T5_SOURCE_SHA256,
        symbols: BIDIRECTIONAL_SYMBOLS,
        symbol_start: 10,
        symbol_end: 19,
    },
    TextEncoderSourceSegment {
        owner: TextEncoderArchitectureOwner::Decoder,
        source_path: GEMMA4_SOURCE_PATH,
        source_sha256: GEMMA4_SOURCE_SHA256,
        symbols: DECODER_SYMBOLS,
        symbol_start: 0,
        symbol_end: 35,
    },
    TextEncoderSourceSegment {
        owner: TextEncoderArchitectureOwner::Decoder,
        source_path: GPT_OSS_SOURCE_PATH,
        source_sha256: GPT_OSS_SOURCE_SHA256,
        symbols: DECODER_SYMBOLS,
        symbol_start: 35,
        symbol_end: 54,
    },
    TextEncoderSourceSegment {
        owner: TextEncoderArchitectureOwner::Decoder,
        source_path: LLAMA_SOURCE_PATH,
        source_sha256: LLAMA_SOURCE_SHA256,
        symbols: DECODER_SYMBOLS,
        symbol_start: 54,
        symbol_end: 101,
    },
    TextEncoderSourceSegment {
        owner: TextEncoderArchitectureOwner::Decoder,
        source_path: QWEN35_SOURCE_PATH,
        source_sha256: QWEN35_SOURCE_SHA256,
        symbols: DECODER_SYMBOLS,
        symbol_start: 101,
        symbol_end: 127,
    },
    TextEncoderSourceSegment {
        owner: TextEncoderArchitectureOwner::Multimodal,
        source_path: IDEOGRAM4_SOURCE_PATH,
        source_sha256: IDEOGRAM4_SOURCE_SHA256,
        symbols: MULTIMODAL_SYMBOLS,
        symbol_start: 0,
        symbol_end: 9,
    },
    TextEncoderSourceSegment {
        owner: TextEncoderArchitectureOwner::Multimodal,
        source_path: JINA_CLIP2_SOURCE_PATH,
        source_sha256: JINA_CLIP2_SOURCE_SHA256,
        symbols: MULTIMODAL_SYMBOLS,
        symbol_start: 9,
        symbol_end: 22,
    },
    TextEncoderSourceSegment {
        owner: TextEncoderArchitectureOwner::Multimodal,
        source_path: OVIS_SOURCE_PATH,
        source_sha256: OVIS_SOURCE_SHA256,
        symbols: MULTIMODAL_SYMBOLS,
        symbol_start: 22,
        symbol_end: 27,
    },
    TextEncoderSourceSegment {
        owner: TextEncoderArchitectureOwner::Multimodal,
        source_path: QWEN3VL_SOURCE_PATH,
        source_sha256: QWEN3VL_SOURCE_SHA256,
        symbols: MULTIMODAL_SYMBOLS,
        symbol_start: 27,
        symbol_end: 37,
    },
    TextEncoderSourceSegment {
        owner: TextEncoderArchitectureOwner::Multimodal,
        source_path: QWEN_VL_SOURCE_PATH,
        source_sha256: QWEN_VL_SOURCE_SHA256,
        symbols: MULTIMODAL_SYMBOLS,
        symbol_start: 37,
        symbol_end: 48,
    },
    TextEncoderSourceSegment {
        owner: TextEncoderArchitectureOwner::Multimodal,
        source_path: SAM3_CLIP_SOURCE_PATH,
        source_sha256: SAM3_CLIP_SOURCE_SHA256,
        symbols: MULTIMODAL_SYMBOLS,
        symbol_start: 48,
        symbol_end: 53,
    },
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TextEncoderArchitectureRegistry {
    version: u16,
}

impl TextEncoderArchitectureRegistry {
    pub fn checked() -> Result<Self, TextEncoderRegistryError> {
        let registry = Self {
            version: TEXT_ENCODER_ARCHITECTURE_REGISTRY_VERSION,
        };
        registry.validate()?;
        Ok(registry)
    }

    pub const fn version(self) -> u16 {
        self.version
    }

    pub const fn contract_count(self) -> usize {
        TEXT_ENCODER_ARCHITECTURE_CONTRACT_COUNT
    }

    pub fn owner_for(
        self,
        source_path: &str,
        source_symbol: &str,
    ) -> Result<TextEncoderArchitectureOwner, TextEncoderRegistryError> {
        let mut owner = None;
        for segment in TEXT_ENCODER_SOURCE_SEGMENTS {
            if segment.source_path == source_path && segment.symbols().contains(&source_symbol) {
                record_owner(&mut owner, segment.owner, source_path, source_symbol)?;
            }
        }
        if COMPOSITE_TEXT_ENCODER_CONTRACTS
            .iter()
            .any(|fact| fact.source_path == source_path && fact.symbol == source_symbol)
        {
            record_owner(
                &mut owner,
                TextEncoderArchitectureOwner::Composite,
                source_path,
                source_symbol,
            )?;
        }
        owner.ok_or_else(|| TextEncoderRegistryError::UnknownContract {
            source_path: source_path.to_owned(),
            source_symbol: source_symbol.to_owned(),
        })
    }

    pub fn identity_sha256(self) -> [u8; 32] {
        let mut digest = Sha256::new();
        digest.update(b"zed-native-text-encoder-architecture-registry\0");
        digest.update(self.version.to_be_bytes());
        for fact in TEXT_ENCODER_OWNER_FACTS {
            digest.update(fact.native_owner.as_bytes());
            digest.update([0]);
            digest.update(fact.implementation_task.as_bytes());
            digest.update([0]);
            digest.update(fact.contract_count.to_be_bytes());
        }
        for segment in TEXT_ENCODER_SOURCE_SEGMENTS {
            digest.update(segment.owner.native_owner().as_bytes());
            digest.update([0]);
            digest.update(segment.source_path.as_bytes());
            digest.update([0]);
            digest.update(segment.source_sha256.as_bytes());
            digest.update([0]);
            for symbol in segment.symbols() {
                digest.update(symbol.as_bytes());
                digest.update([0]);
            }
        }
        for fact in COMPOSITE_TEXT_ENCODER_CONTRACTS {
            digest.update(
                TextEncoderArchitectureOwner::Composite
                    .native_owner()
                    .as_bytes(),
            );
            digest.update([0]);
            digest.update(fact.source_path.as_bytes());
            digest.update([0]);
            digest.update(fact.source_sha256.as_bytes());
            digest.update([0]);
            digest.update(fact.symbol.as_bytes());
            digest.update([0]);
            digest.update(fact.symbol_sha256.as_bytes());
            digest.update([0]);
        }
        digest.finalize().into()
    }

    fn validate(self) -> Result<(), TextEncoderRegistryError> {
        if self.version != TEXT_ENCODER_ARCHITECTURE_REGISTRY_VERSION {
            return Err(TextEncoderRegistryError::InvalidRegistry(
                "version mismatch",
            ));
        }
        let mut keys = BTreeSet::new();
        let mut counts = BTreeMap::new();
        for segment in TEXT_ENCODER_SOURCE_SEGMENTS {
            if segment.symbol_start >= segment.symbol_end
                || segment.symbol_end > segment.symbols.len()
                || !valid_sha256(segment.source_sha256)
            {
                return Err(TextEncoderRegistryError::InvalidRegistry(
                    "source segment range or digest is invalid",
                ));
            }
            for symbol in segment.symbols() {
                if symbol.is_empty() || !keys.insert((segment.source_path, *symbol)) {
                    return Err(TextEncoderRegistryError::InvalidRegistry(
                        "source segment contains an empty or duplicate key",
                    ));
                }
                *counts.entry(segment.owner).or_insert(0_usize) += 1;
            }
        }
        for fact in COMPOSITE_TEXT_ENCODER_CONTRACTS {
            if fact.symbol.is_empty()
                || !valid_sha256(fact.source_sha256)
                || !valid_sha256(fact.symbol_sha256)
                || !keys.insert((fact.source_path, fact.symbol))
            {
                return Err(TextEncoderRegistryError::InvalidRegistry(
                    "composite contract identity or digest is invalid",
                ));
            }
            *counts
                .entry(TextEncoderArchitectureOwner::Composite)
                .or_insert(0_usize) += 1;
        }
        if keys.len() != TEXT_ENCODER_ARCHITECTURE_CONTRACT_COUNT
            || TEXT_ENCODER_OWNER_FACTS
                .iter()
                .map(|fact| fact.contract_count)
                .sum::<usize>()
                != TEXT_ENCODER_ARCHITECTURE_CONTRACT_COUNT
        {
            return Err(TextEncoderRegistryError::InvalidRegistry(
                "registry contract count is not exact",
            ));
        }
        for fact in TEXT_ENCODER_OWNER_FACTS {
            if fact.owner.native_owner() != fact.native_owner
                || fact.owner.implementation_task() != fact.implementation_task
                || counts.get(&fact.owner).copied() != Some(fact.contract_count)
            {
                return Err(TextEncoderRegistryError::InvalidRegistry(
                    "owner fact does not match registered routes",
                ));
            }
        }
        Ok(())
    }
}

fn record_owner(
    found: &mut Option<TextEncoderArchitectureOwner>,
    owner: TextEncoderArchitectureOwner,
    source_path: &str,
    source_symbol: &str,
) -> Result<(), TextEncoderRegistryError> {
    if found.replace(owner).is_some() {
        return Err(TextEncoderRegistryError::MultipleOwners {
            source_path: source_path.to_owned(),
            source_symbol: source_symbol.to_owned(),
        });
    }
    Ok(())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum TextEncoderRegistryError {
    #[error("native text-encoder registry is invalid: {0}")]
    InvalidRegistry(&'static str),
    #[error("text-encoder contract has multiple owners: {source_path}:{source_symbol}")]
    MultipleOwners {
        source_path: String,
        source_symbol: String,
    },
    #[error("text-encoder contract is unregistered: {source_path}:{source_symbol}")]
    UnknownContract {
        source_path: String,
        source_symbol: String,
    },
}
