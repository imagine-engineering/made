use crate::{
    BidirectionalTextOutput, BidirectionalTextRequest, ClipTextOutput, ClipTextRequest,
    ClipVisionIntermediate, ClipVisionOutput, DecoderTextOutput, DecoderTextRequest,
    MultimodalTextError, NativeClipText, NativeClipVision, NativeDecoderTextEncoder,
    NativeT5TextEncoder, run_bidirectional_text_owner, run_clip_text_owner, run_clip_vision_owner,
    run_decoder_text_owner,
};
use sim_tensor::{
    CpuBackend, CpuWorkspaceVec, DecodedScalar, DeviceId, ExecutionContext, RngError,
    RngTransaction, Tensor, TensorError,
    generated_indexing_masking_01::{IndexingMaskingPartOneError, narrow_method_exact_native},
    generated_native_diffusion::{NativeDiffusionTensorError, tensor_from_f32},
    generated_shape_layout_transform_02::{
        ShapeLayoutTransformPartTwoError, torch_cat_with_context_exact_native,
    },
    generated_shape_layout_transform_03::{
        FunctionalPadMode, ShapeLayoutTransformPartThreeError,
        functional_pad_with_context_exact_native,
    },
};
use thiserror::Error;

pub const COMPOSITE_TEXT_ENCODER_CONTRACT_COUNT: usize = 199;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositeSymbolBehavior {
    Profile,
    TokenizerAdapter,
    BidirectionalDelegation,
    DecoderDelegation,
    MultimodalDelegation,
    Cleaner,
    AudioTokenGeneration,
    Projection,
    CompositeOrdering,
    ModelAdapter,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositeContractFact {
    pub source_path: &'static str,
    pub source_sha256: &'static str,
    pub symbol: &'static str,
    pub symbol_sha256: &'static str,
    pub behavior: CompositeSymbolBehavior,
}

pub const COMPOSITE_TEXT_ENCODER_CONTRACTS: [CompositeContractFact;
    COMPOSITE_TEXT_ENCODER_CONTRACT_COUNT] = [
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace.py",
        source_sha256: "f200d0a8579b43e4e9994bdba7298063c5fc8aa5f06fe0c377f704341a5f4bb4",
        symbol: "VoiceBpeTokenizer",
        symbol_sha256: "88a31a7328b7bb1fa6ca7a7b46e60f9838488759ac7047d1d19347abcc7f6ccd",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace.py",
        source_sha256: "f200d0a8579b43e4e9994bdba7298063c5fc8aa5f06fe0c377f704341a5f4bb4",
        symbol: "UMT5BaseModel",
        symbol_sha256: "42c1d917359fb3ac9e8c6be2fe34df9cf1ade48afb571dc00af0d2f5aacf1c91",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace.py",
        source_sha256: "f200d0a8579b43e4e9994bdba7298063c5fc8aa5f06fe0c377f704341a5f4bb4",
        symbol: "UMT5BaseTokenizer",
        symbol_sha256: "a682aef87d55210025a105cf5ec3aa05aad2cb06c805d5a87b3aec47ee33a461",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace.py",
        source_sha256: "f200d0a8579b43e4e9994bdba7298063c5fc8aa5f06fe0c377f704341a5f4bb4",
        symbol: "LyricsTokenizer",
        symbol_sha256: "b4d94139cf6ffbe7f2ac63b81bbbb44747f69d164c9b28fb52aa483fd3a205f1",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace.py",
        source_sha256: "f200d0a8579b43e4e9994bdba7298063c5fc8aa5f06fe0c377f704341a5f4bb4",
        symbol: "AceT5Tokenizer",
        symbol_sha256: "04a54a85bcac98cbd142e462a47e1391526ee34c4cd903befaa398ad79639a7f",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace.py",
        source_sha256: "f200d0a8579b43e4e9994bdba7298063c5fc8aa5f06fe0c377f704341a5f4bb4",
        symbol: "AceT5Model",
        symbol_sha256: "f5ae30e7a68fcfdf0914f379fcb94833c0704e8d9cdf83cb15fc85758eb655b4",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace15.py",
        source_sha256: "21106ab99d7d543a2d157202531e60a80bfa55c88740f9708d2bf7415466f04e",
        symbol: "sample_manual_loop_no_classes",
        symbol_sha256: "f530cbf5ee377eab7637a776ddf7b5f76bce3dfb95309c5267f79686072b6815",
        behavior: CompositeSymbolBehavior::AudioTokenGeneration,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace15.py",
        source_sha256: "21106ab99d7d543a2d157202531e60a80bfa55c88740f9708d2bf7415466f04e",
        symbol: "generate_audio_codes",
        symbol_sha256: "1c4d78b47359b034ad936cc9463f5479ba972b09fcbafe3d01ba1984834c618a",
        behavior: CompositeSymbolBehavior::AudioTokenGeneration,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace15.py",
        source_sha256: "21106ab99d7d543a2d157202531e60a80bfa55c88740f9708d2bf7415466f04e",
        symbol: "ACE15Tokenizer",
        symbol_sha256: "8aa74f4b4161e5ea753ce34a9e28d9ad70e871aaca56b2a48df0586c446b183a",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace15.py",
        source_sha256: "21106ab99d7d543a2d157202531e60a80bfa55c88740f9708d2bf7415466f04e",
        symbol: "Qwen3_06BModel",
        symbol_sha256: "d2d6979a929489dcfaf5ee09ab8f321995365eb172715e0e51c568f0f563001e",
        behavior: CompositeSymbolBehavior::DecoderDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace15.py",
        source_sha256: "21106ab99d7d543a2d157202531e60a80bfa55c88740f9708d2bf7415466f04e",
        symbol: "Qwen3_2B_ACE15",
        symbol_sha256: "68832d33288a48fbfcb949427b4a63f91c263561bda1ab58c4957cb9de43ac1a",
        behavior: CompositeSymbolBehavior::DecoderDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace15.py",
        source_sha256: "21106ab99d7d543a2d157202531e60a80bfa55c88740f9708d2bf7415466f04e",
        symbol: "Qwen3_4B_ACE15",
        symbol_sha256: "899d36a6dd717f37f264ae1596a2f6398930680cc1f3230c0e6edaa795d9e65b",
        behavior: CompositeSymbolBehavior::DecoderDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace15.py",
        source_sha256: "21106ab99d7d543a2d157202531e60a80bfa55c88740f9708d2bf7415466f04e",
        symbol: "ACE15TEModel",
        symbol_sha256: "0b12af8f586c6ae1086439e6f548c110c6f1bcf78a4c977d2dc0132bffd2e852",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace15.py",
        source_sha256: "21106ab99d7d543a2d157202531e60a80bfa55c88740f9708d2bf7415466f04e",
        symbol: "te",
        symbol_sha256: "4af4456731cf0b36078def6753eaad439bd047ca6c8455d4d6ec9f419ea29603",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "japanese_to_romaji",
        symbol_sha256: "ec6bf332e8e8b8a492702a8381f5bb891d6f76166672b7a9299971d1d43fe8a1",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "number_to_text",
        symbol_sha256: "0a4680506fed1b24b2b6d6ba20d596aa0dfc9dc6bdb0cd6acb5c77c3a6fb3bcd",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "_int_to_text",
        symbol_sha256: "15cd8153ae1641fbb65f1b1a3bbd2d524244928921b23ab49f4cb8fcf396fc99",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "_digit_to_text",
        symbol_sha256: "908474a7b396bf87d2397adb340e00005ea8b8e0976d0648580e89805d25fda8",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "expand_abbreviations_multilingual",
        symbol_sha256: "688b37ebefb33a4d48b7097a38dfcdc426cef82eacd322647dcaa876811a0cb9",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "expand_symbols_multilingual",
        symbol_sha256: "25e27c7b78867a4a5ce1a12c073c05e9d1cdf50dfd9f7ad6a72061bec0181d68",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "_remove_commas",
        symbol_sha256: "f4e4d68b3cf0305a5dd4b9dca61cc6cfff81d52179f0c4dfbf234a75fc4e935a",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "_remove_dots",
        symbol_sha256: "73c3624c91f6c21354fc7cccc5ef9372c6346cc5464acfecbbe1384e3ed85d13",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "_expand_decimal_point",
        symbol_sha256: "01a36be256fc127ca353ee5af0f0eb836004f4006bfd3d90b3dd7c5239480f9e",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "_expand_currency",
        symbol_sha256: "29455a1bb9d70bcb83588a00d4774d52a9a0d30646b214dc7fdd7c6ce0bb44e3",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "_expand_ordinal",
        symbol_sha256: "9aeac33f6595fef5b1e29d3bd621d99eab97d0cadd41a96b8d855d6411978088",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "_expand_number",
        symbol_sha256: "ae1efe892d1d61c623551b64da0913a34c34cf5d70c6418cdf9f23bbe153c33f",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "expand_numbers_multilingual",
        symbol_sha256: "709643d86c7652d1197e7c12d58f783cf289144cc903672e5b62029446131e91",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "lowercase",
        symbol_sha256: "296274b940cb0b327c16a3cfa756fcb68373d762bec13aca1771a7e588667214",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "collapse_whitespace",
        symbol_sha256: "249be6041d9ffd9d6b051e6b62dea2d341bc6f98de3c4ce1ba50bb9cc159d31d",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "multilingual_cleaners",
        symbol_sha256: "c19b47e2106e37a11847c5fcf2cf0dfdd1b861724e8b05d218a7d0f127f5759a",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ace_text_cleaners.py",
        source_sha256: "798e1d7d2e29a0a98a1d0bac53a058d5c0c90346f660b3d1aee0622ade85a9c3",
        symbol: "basic_cleaners",
        symbol_sha256: "0ca8593064defb36ae2bc5a9dfc8ec1680fdf1a1f0ff496d8c218311379e5f1c",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/anima.py",
        source_sha256: "7eef5186d57c7f6d4b23ecdebb454ae9b64def8117d0e7a57b0b537de64c69c5",
        symbol: "Qwen3Tokenizer",
        symbol_sha256: "7cee37869b0f11f11ac106bfb0f17c76752141e5e6abe63adb5232090bfd8228",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/anima.py",
        source_sha256: "7eef5186d57c7f6d4b23ecdebb454ae9b64def8117d0e7a57b0b537de64c69c5",
        symbol: "T5XXLTokenizer",
        symbol_sha256: "528cb0764886679fa58663465667e1e0334f1411c68bd187bc3bd8cb640db67d",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/anima.py",
        source_sha256: "7eef5186d57c7f6d4b23ecdebb454ae9b64def8117d0e7a57b0b537de64c69c5",
        symbol: "AnimaTokenizer",
        symbol_sha256: "55a60c704a66edec04a03d4a00bc7c62a246f1a000d81f8be9b2490045d52864",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/anima.py",
        source_sha256: "7eef5186d57c7f6d4b23ecdebb454ae9b64def8117d0e7a57b0b537de64c69c5",
        symbol: "Qwen3_06BModel",
        symbol_sha256: "6b9f47d3053c2c7b94415f8260e9edddaa52022c844a634c72b2d81cf4f9be15",
        behavior: CompositeSymbolBehavior::DecoderDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/anima.py",
        source_sha256: "7eef5186d57c7f6d4b23ecdebb454ae9b64def8117d0e7a57b0b537de64c69c5",
        symbol: "AnimaTEModel",
        symbol_sha256: "71d4254a39d63e47195c8e87e0207099959b6fd18c260171b290b9ca3ad173b6",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/anima.py",
        source_sha256: "7eef5186d57c7f6d4b23ecdebb454ae9b64def8117d0e7a57b0b537de64c69c5",
        symbol: "te",
        symbol_sha256: "0f052036c881409b29911415cbceca0e7f8df4d3f3c4bb8c890dcb547664c19e",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/aura_t5.py",
        source_sha256: "d20edaa3fea38bb65359941882546c42ba1142ba70fe9933d766aeddac9c4ab3",
        symbol: "PT5XlModel",
        symbol_sha256: "a86d862127311d451bba1f17913c5f5c536e760d38b3928ad67a13badaa77550",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/aura_t5.py",
        source_sha256: "d20edaa3fea38bb65359941882546c42ba1142ba70fe9933d766aeddac9c4ab3",
        symbol: "PT5XlTokenizer",
        symbol_sha256: "fbb502a76c19d32a7c4417c39fb0f19d701bc9cd02e884e1987671e42aec2188",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/aura_t5.py",
        source_sha256: "d20edaa3fea38bb65359941882546c42ba1142ba70fe9933d766aeddac9c4ab3",
        symbol: "AuraT5Tokenizer",
        symbol_sha256: "061c9a179b3a0687e4ce4b01e9c43b5e3422cf8ee8485ef4aa023991a7e1b5d7",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/aura_t5.py",
        source_sha256: "d20edaa3fea38bb65359941882546c42ba1142ba70fe9933d766aeddac9c4ab3",
        symbol: "AuraT5Model",
        symbol_sha256: "87e77017e3329ba07dd299c6c059be37a8fa50e1d7722de4e4a752e88255e2b3",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/boogu.py",
        source_sha256: "b8ca823a27240a676c896a8116e1c6a5f2238e478e41fdbf2d0a6807bb5d3cd4",
        symbol: "BooguTokenizer",
        symbol_sha256: "4810af68ba04c9fedc70c3c2c01772fd6ccc23605890b29067443c35cc125279",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/boogu.py",
        source_sha256: "b8ca823a27240a676c896a8116e1c6a5f2238e478e41fdbf2d0a6807bb5d3cd4",
        symbol: "BooguQwen3VLClipModel",
        symbol_sha256: "9c8012f3507ff4fc2ef29a3b088a46c9e9cefdde22a4bdf4c5117e3bb9a0ff18",
        behavior: CompositeSymbolBehavior::MultimodalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/boogu.py",
        source_sha256: "b8ca823a27240a676c896a8116e1c6a5f2238e478e41fdbf2d0a6807bb5d3cd4",
        symbol: "BooguTEModel",
        symbol_sha256: "b7df32e44f94158ba5c84472f9a77893fb2316a452bc4b2f61288a9dd1969b68",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/boogu.py",
        source_sha256: "b8ca823a27240a676c896a8116e1c6a5f2238e478e41fdbf2d0a6807bb5d3cd4",
        symbol: "te",
        symbol_sha256: "939b47ac6430f59f32e66a14144cb443999f67586e7013084e38bfda1a2891ab",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/cogvideo.py",
        source_sha256: "60240724679701a21ad5756a2509bcc0e76436d6f9a7647b67f38ab6ed02fd35",
        symbol: "CogVideoXT5Tokenizer",
        symbol_sha256: "6e0f8441e86485bd3a7b6082ff1d0304175d167f07c848e01f44c0b794d9c9e3",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/cogvideo.py",
        source_sha256: "60240724679701a21ad5756a2509bcc0e76436d6f9a7647b67f38ab6ed02fd35",
        symbol: "CogVideoXTokenizer",
        symbol_sha256: "0223c3134aca0532bc49b1c9ceac700e7c0cc144f4d26d63ee62a666013d393c",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/cogvideo.py",
        source_sha256: "60240724679701a21ad5756a2509bcc0e76436d6f9a7647b67f38ab6ed02fd35",
        symbol: "CogVideoXT5XXL",
        symbol_sha256: "2652684717e8bdfd869ce6c3b708233cd824074b5519595b0ccf1c6b462b0174",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/cogvideo.py",
        source_sha256: "60240724679701a21ad5756a2509bcc0e76436d6f9a7647b67f38ab6ed02fd35",
        symbol: "cogvideo_te",
        symbol_sha256: "8c05d2536ff35789ccb6c3f38ec3e0d7f9ca84757c3d083af5e3921c72737824",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/cosmos.py",
        source_sha256: "98c09ef00e0b0953bfae891ca2df97c3d70fdbfbacb1165168fe74eaff230524",
        symbol: "T5XXLModel",
        symbol_sha256: "68d3b7c52c6d64befc9a01357d3e3344b6dd5fd15ae8499c7e060d6aff1e3ab3",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/cosmos.py",
        source_sha256: "98c09ef00e0b0953bfae891ca2df97c3d70fdbfbacb1165168fe74eaff230524",
        symbol: "CosmosT5XXL",
        symbol_sha256: "1e3509c220ae3a47072f7d5cc9fc52778cd64b0162d54aa557fdf7e6bc521ce2",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/cosmos.py",
        source_sha256: "98c09ef00e0b0953bfae891ca2df97c3d70fdbfbacb1165168fe74eaff230524",
        symbol: "T5XXLTokenizer",
        symbol_sha256: "446a13f620569acbbd5d6582566eb684aca3363d4df9c62a14fa4a49667dcfa5",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/cosmos.py",
        source_sha256: "98c09ef00e0b0953bfae891ca2df97c3d70fdbfbacb1165168fe74eaff230524",
        symbol: "CosmosT5Tokenizer",
        symbol_sha256: "4b2dfffed78306237cd425f52286231d32060cac5a0b7eb4caaf99a765d84ac6",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/cosmos.py",
        source_sha256: "98c09ef00e0b0953bfae891ca2df97c3d70fdbfbacb1165168fe74eaff230524",
        symbol: "te",
        symbol_sha256: "72899bea40acf870e11f2c7f1e289b4c90878c3723115ed5f95f22c1c1a60b83",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ernie.py",
        source_sha256: "6a9ead74fcd909b503013e3df76405b10843ddf074edaeabfad20aeb2399efdf",
        symbol: "Ministral3_3BTokenizer",
        symbol_sha256: "3c461b348aa0718a2b23e94bdf7069c0b9181371890820b75a1b0e04b55ca7f4",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ernie.py",
        source_sha256: "6a9ead74fcd909b503013e3df76405b10843ddf074edaeabfad20aeb2399efdf",
        symbol: "ErnieTokenizer",
        symbol_sha256: "58c5f38419045fb66912515e98897a76585cef87006a816cb78163aca3f3f741",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ernie.py",
        source_sha256: "6a9ead74fcd909b503013e3df76405b10843ddf074edaeabfad20aeb2399efdf",
        symbol: "Ministral3_3BModel",
        symbol_sha256: "5950bbf487cdc6eca268395dddc1284c8a48ac217bb6422d03bede5ee6b64ba8",
        behavior: CompositeSymbolBehavior::ModelAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ernie.py",
        source_sha256: "6a9ead74fcd909b503013e3df76405b10843ddf074edaeabfad20aeb2399efdf",
        symbol: "ErnieTEModel",
        symbol_sha256: "e04213401e171548c04446c4e534e621605786ce6223a59f70f2100bca6d4084",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/ernie.py",
        source_sha256: "6a9ead74fcd909b503013e3df76405b10843ddf074edaeabfad20aeb2399efdf",
        symbol: "te",
        symbol_sha256: "4e4ce067436a2902127d566feb9a08abb3a9a7ca1755b5fd71b19faa19f8dda1",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "T5XXLTokenizer",
        symbol_sha256: "6bf1a811156d55e4f0c165e32fc82090a01e860104fd0dd82d0dd03f1f68ed7b",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "FluxTokenizer",
        symbol_sha256: "e28d2fab104e5309a4f70bcf71ad932e7a4173ec44c3906215405e50e4ccbd74",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "FluxClipModel",
        symbol_sha256: "19851ddb41d6b06d042a6e5115d648db8947937f9d2ce4bdbfa9f90b2f9d0b61",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "flux_clip",
        symbol_sha256: "6547d17eed0b3cb24ca6d46e935607d8de846f1701c8aa82cdec064ce5c407c9",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "load_mistral_tokenizer",
        symbol_sha256: "6c1d65d906392be5940e32be9abf8667f2227c976f9ad601d43f20b1574d9ecb",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "MistralTokenizerClass",
        symbol_sha256: "cdee7cb2978d4b1962899ac9f22ecb4050de5ac7fc5fad0dd3f20f61058e82ce",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "Mistral3Tokenizer",
        symbol_sha256: "e79532deb09ce2fa50612deba66d99cc212402bda203eab39a66749a2e1513ba",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "Flux2Tokenizer",
        symbol_sha256: "0d7e6b1194471f916b6fa6869d36c45bdfe5d4329b90ebecaff9e5d6a06ac45f",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "Mistral3_24BModel",
        symbol_sha256: "0b3ce817f9f73af5dc68b72f2bbf0d092e5e6601505a867194f3805ff777deec",
        behavior: CompositeSymbolBehavior::DecoderDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "Flux2TEModel",
        symbol_sha256: "4e85a46625f6ea5dcfd34f0035a8f723b7cb007fcdcd26e79c131cc59f33341a",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "flux2_te",
        symbol_sha256: "68190ef029bc9cb255543eacfacd9e5a55879d97b5f27f6a3e1f8f98fee14971",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "Qwen3Tokenizer",
        symbol_sha256: "70044a0791f755e2523fc27832eb69cff06899a9216bed98fa88aaaf767cb4e0",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "Qwen3Tokenizer8B",
        symbol_sha256: "d87f2806a2739ac5009b72e7229b6cc3ac5a998de3d7917e09961bc02e5af020",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "KleinTokenizer",
        symbol_sha256: "4df873a14f7dc3c849343907c8f24fe6101dd2d15b3cb9c00db3f23f59952d44",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "KleinTokenizer8B",
        symbol_sha256: "26d71481315e0be3094f573be29a65b26825505179c9366ae138aa8b8c45a244",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "Qwen3_4BModel",
        symbol_sha256: "4b8eaac2f9fbf6d2a35b7e039b48a97ff37267baa6b7b6e461ddbc4ac53bba9b",
        behavior: CompositeSymbolBehavior::DecoderDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "Qwen3_8BModel",
        symbol_sha256: "0e46e945a10646941caa64d18656a3567ae8bbdfaf9b8e12b87ae71f8178a24b",
        behavior: CompositeSymbolBehavior::DecoderDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/flux.py",
        source_sha256: "c522ad47ccdad9878f0e8ae4c219dadfc4f12409adceaa7e663571766cbbe46b",
        symbol: "klein_te",
        symbol_sha256: "d55406847ba281bf96794da95f7ccd5cbb6df2f7fe4a2929a2746fbbf70fb0d3",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/genmo.py",
        source_sha256: "dfbf5c44111595ecfcb531a7b6e6d8e32a6c629d176fa90a932865a57dbbd84b",
        symbol: "T5XXLModel",
        symbol_sha256: "446ffbb06728ea18e0aa30ecc782c14b71d75a1dcc34f44227c2b5b5e9c369a8",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/genmo.py",
        source_sha256: "dfbf5c44111595ecfcb531a7b6e6d8e32a6c629d176fa90a932865a57dbbd84b",
        symbol: "MochiT5XXL",
        symbol_sha256: "f7eff2eb1d086be741a87e4e3710b768101e75b25128a04131041e625a064758",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/genmo.py",
        source_sha256: "dfbf5c44111595ecfcb531a7b6e6d8e32a6c629d176fa90a932865a57dbbd84b",
        symbol: "T5XXLTokenizer",
        symbol_sha256: "6bf1a811156d55e4f0c165e32fc82090a01e860104fd0dd82d0dd03f1f68ed7b",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/genmo.py",
        source_sha256: "dfbf5c44111595ecfcb531a7b6e6d8e32a6c629d176fa90a932865a57dbbd84b",
        symbol: "MochiT5Tokenizer",
        symbol_sha256: "8a87a6b63e9a996dbab1c3764c5327d66d84fa221b729bf41cfed1d1a61d9dc8",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/genmo.py",
        source_sha256: "dfbf5c44111595ecfcb531a7b6e6d8e32a6c629d176fa90a932865a57dbbd84b",
        symbol: "mochi_te",
        symbol_sha256: "c97ff990c4b94c0c3ad29351f4a66c1fc216cde394bfeed5dd11a20a18778c74",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hidream.py",
        source_sha256: "fa4c0e7c438254a12fe877d9e1db1b13723b473ddab8f43bf5e2cf2b56ad6d75",
        symbol: "HiDreamTokenizer",
        symbol_sha256: "dd42d6befdef74ac646474cfdb5d9ab8b680165dd67ca225c906da2cf6710bbe",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hidream.py",
        source_sha256: "fa4c0e7c438254a12fe877d9e1db1b13723b473ddab8f43bf5e2cf2b56ad6d75",
        symbol: "HiDreamTEModel",
        symbol_sha256: "d2be5fae3d34488552f552119c1e59b90c35048eb6145d5aa61b32514a0561d3",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hidream.py",
        source_sha256: "fa4c0e7c438254a12fe877d9e1db1b13723b473ddab8f43bf5e2cf2b56ad6d75",
        symbol: "hidream_clip",
        symbol_sha256: "4bd41a173aec8f74e5f340fdc75fcaebf8a8c87f3fe6b932cef30a3a82e43ddc",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hidream_o1.py",
        source_sha256: "d49e31c803d12133c54229788bb9775fb32a6046aea2724fd1d835a6188c5bae",
        symbol: "HiDreamO1QwenTokenizer",
        symbol_sha256: "9aaa1e780168c191db1042e4f9faeb6cab01ef6ffaa28536529731f4d2730310",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hidream_o1.py",
        source_sha256: "d49e31c803d12133c54229788bb9775fb32a6046aea2724fd1d835a6188c5bae",
        symbol: "HiDreamO1Tokenizer",
        symbol_sha256: "30c6439367f284184e2042acc3e67211cbb1f38d684acf2c91275401546d94ea",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hidream_o1.py",
        source_sha256: "d49e31c803d12133c54229788bb9775fb32a6046aea2724fd1d835a6188c5bae",
        symbol: "HiDreamO1TE",
        symbol_sha256: "94ffc4c92f83788c0237d24d6cde2d258b53c7c5ddf3884cb007309608642ec6",
        behavior: CompositeSymbolBehavior::ModelAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hunyuan_image.py",
        source_sha256: "ab462b226963fc4aa56ef8956195227a8f7e07622353353ef7e5a8ee20a84b9f",
        symbol: "ByT5SmallTokenizer",
        symbol_sha256: "ff3132a40e59667d68f8c77ae40e595e64423128c95f745d9446a9058d583967",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hunyuan_image.py",
        source_sha256: "ab462b226963fc4aa56ef8956195227a8f7e07622353353ef7e5a8ee20a84b9f",
        symbol: "HunyuanImageTokenizer",
        symbol_sha256: "8dbadb743aba70b7ff4b0e5e0ce83cc2f86dd2cc77172af40a692db3bfeed060",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hunyuan_image.py",
        source_sha256: "ab462b226963fc4aa56ef8956195227a8f7e07622353353ef7e5a8ee20a84b9f",
        symbol: "Qwen25_7BVLIModel",
        symbol_sha256: "2cc155e235c43e8601b95e99ce48a9fb0f5b97f6403717aefa3be745ac9daa4a",
        behavior: CompositeSymbolBehavior::MultimodalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hunyuan_image.py",
        source_sha256: "ab462b226963fc4aa56ef8956195227a8f7e07622353353ef7e5a8ee20a84b9f",
        symbol: "ByT5SmallModel",
        symbol_sha256: "c4c09846f74a95390eae124a998b4b184a582fa95c1046b861b499127dec9912",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hunyuan_image.py",
        source_sha256: "ab462b226963fc4aa56ef8956195227a8f7e07622353353ef7e5a8ee20a84b9f",
        symbol: "HunyuanImageTEModel",
        symbol_sha256: "9a85a403845cbd34f46ff8af72aa258ddcc6e0faf962532a9c2a40d6e09f15dc",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hunyuan_image.py",
        source_sha256: "ab462b226963fc4aa56ef8956195227a8f7e07622353353ef7e5a8ee20a84b9f",
        symbol: "te",
        symbol_sha256: "ac53044be06e0e5d7e073f82cd72208d77932d3791ca0186b50e65e237827bce",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hunyuan_video.py",
        source_sha256: "5e9001d75b3eaaed8bb145be1b1b2a39d012140c11a45078b21beed2ddabae4e",
        symbol: "llama_detect",
        symbol_sha256: "7d34180a250ef1bdb7b30cba14e790ccd8416cf8f4c56e82565eb9ad048071f8",
        behavior: CompositeSymbolBehavior::Profile,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hunyuan_video.py",
        source_sha256: "5e9001d75b3eaaed8bb145be1b1b2a39d012140c11a45078b21beed2ddabae4e",
        symbol: "LLAMA3Tokenizer",
        symbol_sha256: "7d290207e33bdf4588ac2f11ca3768f36cd0b60cc0653f23853be733bffb4a03",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hunyuan_video.py",
        source_sha256: "5e9001d75b3eaaed8bb145be1b1b2a39d012140c11a45078b21beed2ddabae4e",
        symbol: "LLAMAModel",
        symbol_sha256: "1bba313c7a9ab5a6c4f740c33a9c375cd29071698ae6dba30d5ea1a97314f6ad",
        behavior: CompositeSymbolBehavior::ModelAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hunyuan_video.py",
        source_sha256: "5e9001d75b3eaaed8bb145be1b1b2a39d012140c11a45078b21beed2ddabae4e",
        symbol: "HunyuanVideoTokenizer",
        symbol_sha256: "628ece7b389850c4cddc172508596757aba758816de9050e64ac6bc2ee47c3f9",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hunyuan_video.py",
        source_sha256: "5e9001d75b3eaaed8bb145be1b1b2a39d012140c11a45078b21beed2ddabae4e",
        symbol: "HunyuanVideo15Tokenizer",
        symbol_sha256: "dcad245fcb4e9585359c4246b2386e22ec5d6a49537ed815948c5825b28de552",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hunyuan_video.py",
        source_sha256: "5e9001d75b3eaaed8bb145be1b1b2a39d012140c11a45078b21beed2ddabae4e",
        symbol: "HunyuanVideoClipModel",
        symbol_sha256: "f31d27ceacbb31e57b7211de40e007e64d3f7ca1967e4de2f393e3d2ea3f1c29",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hunyuan_video.py",
        source_sha256: "5e9001d75b3eaaed8bb145be1b1b2a39d012140c11a45078b21beed2ddabae4e",
        symbol: "hunyuan_video_clip",
        symbol_sha256: "507f3a9f274e4fcf0fb8e72c00cdb3be5ae9f4089fa0e4dd5f419f5961e62922",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hydit.py",
        source_sha256: "3a56925c2ef403e888b2bd2187aaa5c0fdbf16ad16c88c282da095603873e748",
        symbol: "HyditBertModel",
        symbol_sha256: "66a5d4310466441948f432e37a051306e39fc0d2bb2bc013fd66c9dbb86b35a0",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hydit.py",
        source_sha256: "3a56925c2ef403e888b2bd2187aaa5c0fdbf16ad16c88c282da095603873e748",
        symbol: "HyditBertTokenizer",
        symbol_sha256: "66fb924f99d78648ebcfd853dc0bef5654aedc2dedb7da24f9d33aeebf99e113",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hydit.py",
        source_sha256: "3a56925c2ef403e888b2bd2187aaa5c0fdbf16ad16c88c282da095603873e748",
        symbol: "MT5XLModel",
        symbol_sha256: "6de93e005623631a0cd814739c71869badaa2909fde0f763f54855954ac5c8c9",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hydit.py",
        source_sha256: "3a56925c2ef403e888b2bd2187aaa5c0fdbf16ad16c88c282da095603873e748",
        symbol: "MT5XLTokenizer",
        symbol_sha256: "dc7141804ef11d8acd509e2ca335bfd3ea26655808c0c573945ebd7df57b5b9c",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hydit.py",
        source_sha256: "3a56925c2ef403e888b2bd2187aaa5c0fdbf16ad16c88c282da095603873e748",
        symbol: "HyditTokenizer",
        symbol_sha256: "aa7e9be697ea76b45d563085f8f25ad4fa4448d032209b84ba9b0e13da0790f9",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/hydit.py",
        source_sha256: "3a56925c2ef403e888b2bd2187aaa5c0fdbf16ad16c88c282da095603873e748",
        symbol: "HyditModel",
        symbol_sha256: "73b4f75846768f03f63c55d8f76163ade7cfd4bedb25a9acb54348ea88e88359",
        behavior: CompositeSymbolBehavior::ModelAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/kandinsky5.py",
        source_sha256: "1de8b4f8744946ce52914c03b8e10da1b897a720672401b91f8ddb5897e201f8",
        symbol: "Kandinsky5Tokenizer",
        symbol_sha256: "5a1ffcbaf40b7a8dfddeaff6131646948af35adcb95030517cc352f5d214dde1",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/kandinsky5.py",
        source_sha256: "1de8b4f8744946ce52914c03b8e10da1b897a720672401b91f8ddb5897e201f8",
        symbol: "Kandinsky5TokenizerImage",
        symbol_sha256: "52924a1d2cc97d4e3040efeddc102a8832167ad96e7d7603d95f549da63542f8",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/kandinsky5.py",
        source_sha256: "1de8b4f8744946ce52914c03b8e10da1b897a720672401b91f8ddb5897e201f8",
        symbol: "Qwen25_7BVLIModel",
        symbol_sha256: "64edff1d00445d2014f76607eaed3fd68d33e89465968e092fe3ed9f3cc40191",
        behavior: CompositeSymbolBehavior::MultimodalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/kandinsky5.py",
        source_sha256: "1de8b4f8744946ce52914c03b8e10da1b897a720672401b91f8ddb5897e201f8",
        symbol: "Kandinsky5TEModel",
        symbol_sha256: "05d423c03f5117e58120f15773f6ab7cd903e77be2e8825bc5041a049742efcf",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/kandinsky5.py",
        source_sha256: "1de8b4f8744946ce52914c03b8e10da1b897a720672401b91f8ddb5897e201f8",
        symbol: "te",
        symbol_sha256: "baa8db2305d0c8ce5c12488354e7208d0a4201d4469ffdbe4efadbf34d51b334",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/krea2.py",
        source_sha256: "be316682d4787ff0eb2bda9f2396b3769efa8f1f28bd1a688069f414de04c5e4",
        symbol: "Krea2Tokenizer",
        symbol_sha256: "54bf6bf5f881520667eb170638ac0e49326469230fdb559809358696f2146fca",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/krea2.py",
        source_sha256: "be316682d4787ff0eb2bda9f2396b3769efa8f1f28bd1a688069f414de04c5e4",
        symbol: "Krea2Qwen3VLClipModel",
        symbol_sha256: "5ad1b705ef91e95dd75c3188bf5cc05f02a499b8c189f8d6c3be46ad913d771d",
        behavior: CompositeSymbolBehavior::MultimodalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/krea2.py",
        source_sha256: "be316682d4787ff0eb2bda9f2396b3769efa8f1f28bd1a688069f414de04c5e4",
        symbol: "Krea2TEModel",
        symbol_sha256: "0244057884c554f7f50475393796c269f9fbd63fcbc39649da63792467cbbef4",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/krea2.py",
        source_sha256: "be316682d4787ff0eb2bda9f2396b3769efa8f1f28bd1a688069f414de04c5e4",
        symbol: "te",
        symbol_sha256: "5e61c89d4999bd6dd5b595de9ce3d47d8df8db318172a7c11cfd91cb2aa4c6c9",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/long_clipl.py",
        source_sha256: "ec23dfced81defe857d88d3cb4536c07ffe8f9c7ab03e9506589e87b73b81508",
        symbol: "model_options_long_clip",
        symbol_sha256: "0819c125d203385cb907a77202b77226746bc01ef1f492638317e16f0a7de980",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/longcat_image.py",
        source_sha256: "8c9a061f054948ae6be006677876d621bee6d5e370f18af6b7aa6b61049f11dd",
        symbol: "split_quotation",
        symbol_sha256: "2b7ed9b9b3fe05cee04ec296e0f94c6baf35b0b54e0676150c3d0356a93899c8",
        behavior: CompositeSymbolBehavior::Cleaner,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/longcat_image.py",
        source_sha256: "8c9a061f054948ae6be006677876d621bee6d5e370f18af6b7aa6b61049f11dd",
        symbol: "LongCatImageBaseTokenizer",
        symbol_sha256: "82701759419a9521a7d224764b5cf549266756a4abe55d18ef26650d4287cc21",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/longcat_image.py",
        source_sha256: "8c9a061f054948ae6be006677876d621bee6d5e370f18af6b7aa6b61049f11dd",
        symbol: "LongCatImageTokenizer",
        symbol_sha256: "205dab9deaa1ee59a505d2a0f4e0038168f6d54dbed120829362ecf7c0c64188",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/longcat_image.py",
        source_sha256: "8c9a061f054948ae6be006677876d621bee6d5e370f18af6b7aa6b61049f11dd",
        symbol: "LongCatImageTEModel",
        symbol_sha256: "b0452afc1fd4846e40283c9bb5d7d664085d9b3ef83a53e8d2aab38ab64a3248",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/longcat_image.py",
        source_sha256: "8c9a061f054948ae6be006677876d621bee6d5e370f18af6b7aa6b61049f11dd",
        symbol: "te",
        symbol_sha256: "19d78fde368cb40a13dcead08185455c8c1f949ed5b6ac3d1f8d79cf8e3b6fea",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lt.py",
        source_sha256: "9ddf9e68c4afd1cf848f881b7489abb49d37ac8ad6d5d2893eba4f98c9c37ca2",
        symbol: "T5XXLTokenizer",
        symbol_sha256: "e829119cdbce7e03a94b51a1f48315437c4d24cea9a858fe4c882d8ee89b612a",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lt.py",
        source_sha256: "9ddf9e68c4afd1cf848f881b7489abb49d37ac8ad6d5d2893eba4f98c9c37ca2",
        symbol: "LTXVT5Tokenizer",
        symbol_sha256: "c9c3bb688306b0120178b6c444b130279a607a0d60f4c7801aff00b7e5702aa6",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lt.py",
        source_sha256: "9ddf9e68c4afd1cf848f881b7489abb49d37ac8ad6d5d2893eba4f98c9c37ca2",
        symbol: "ltxv_te",
        symbol_sha256: "f6403fa7322f68c1a6a7283667c811c9ed4c53cf744aa39cfb4e2a121c2cda41",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lt.py",
        source_sha256: "9ddf9e68c4afd1cf848f881b7489abb49d37ac8ad6d5d2893eba4f98c9c37ca2",
        symbol: "Gemma3_Tokenizer",
        symbol_sha256: "cf486690f45eb9feb46ab2dfe460eeebf4d6771a2f470cf5554bb83de090bc55",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lt.py",
        source_sha256: "9ddf9e68c4afd1cf848f881b7489abb49d37ac8ad6d5d2893eba4f98c9c37ca2",
        symbol: "Gemma3_12BTokenizer",
        symbol_sha256: "e240ada6d51b6b1425ef7b7d5d6e3decd1433660143c9358582ddb826a6f2a36",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lt.py",
        source_sha256: "9ddf9e68c4afd1cf848f881b7489abb49d37ac8ad6d5d2893eba4f98c9c37ca2",
        symbol: "LTXAVGemmaTokenizer",
        symbol_sha256: "18a5b8283621ed8575e8f6ce97faf2613d81a1a72776a4a4cee1bb3ed235df98",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lt.py",
        source_sha256: "9ddf9e68c4afd1cf848f881b7489abb49d37ac8ad6d5d2893eba4f98c9c37ca2",
        symbol: "Gemma3_12BModel",
        symbol_sha256: "a7e4f8867f7eefc588c185d319a277d7aaf7825ce2cb3d2a2fc1bfd131e80e23",
        behavior: CompositeSymbolBehavior::DecoderDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lt.py",
        source_sha256: "9ddf9e68c4afd1cf848f881b7489abb49d37ac8ad6d5d2893eba4f98c9c37ca2",
        symbol: "DualLinearProjection",
        symbol_sha256: "92ea4bdcab7e2a2b9f78bb416aaa5f7d707c40105f34c255f9961784ea77dcd4",
        behavior: CompositeSymbolBehavior::Projection,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lt.py",
        source_sha256: "9ddf9e68c4afd1cf848f881b7489abb49d37ac8ad6d5d2893eba4f98c9c37ca2",
        symbol: "LTXAVTEModel",
        symbol_sha256: "e69b290653bbabb4e0c944e56f225eff3b582ae8c4848a8b76d3d78633f47711",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lt.py",
        source_sha256: "9ddf9e68c4afd1cf848f881b7489abb49d37ac8ad6d5d2893eba4f98c9c37ca2",
        symbol: "ltxav_te",
        symbol_sha256: "c8ced824a6778bf8dc278de187695501c8168aed7f404ffcf38b65f445450baf",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lt.py",
        source_sha256: "9ddf9e68c4afd1cf848f881b7489abb49d37ac8ad6d5d2893eba4f98c9c37ca2",
        symbol: "sd_detect",
        symbol_sha256: "62b12940f5f7538234c7adf22b6ef9c7d1cf48da8d07054fd399f10cbb47afe2",
        behavior: CompositeSymbolBehavior::Profile,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lt.py",
        source_sha256: "9ddf9e68c4afd1cf848f881b7489abb49d37ac8ad6d5d2893eba4f98c9c37ca2",
        symbol: "gemma3_te",
        symbol_sha256: "6672686da3693786b7f879a504c07fadf9728a9abff8538fbfd6e5547b079a6e",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lumina2.py",
        source_sha256: "f6af934b3d5014c6df37bb527167d1f94e44b1309079f57d4cb2f9460729da84",
        symbol: "Gemma2BTokenizer",
        symbol_sha256: "22db76900779ea297b2e75c14d8dcf164e012e39b48780ed2791913c02447f41",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lumina2.py",
        source_sha256: "f6af934b3d5014c6df37bb527167d1f94e44b1309079f57d4cb2f9460729da84",
        symbol: "Gemma3_4BTokenizer",
        symbol_sha256: "8e9ec5fe85412f06d1e00ffeb0f20870693c15f308f53a9148f04f05b1b3497d",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lumina2.py",
        source_sha256: "f6af934b3d5014c6df37bb527167d1f94e44b1309079f57d4cb2f9460729da84",
        symbol: "LuminaTokenizer",
        symbol_sha256: "15e0bd422188a28c72e5e435bf0952272ae03fbf62afcc2045a93a9c3f1d8703",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lumina2.py",
        source_sha256: "f6af934b3d5014c6df37bb527167d1f94e44b1309079f57d4cb2f9460729da84",
        symbol: "NTokenizer",
        symbol_sha256: "2a14da60ff05f3f0aa1d3acf8c44e10195851ea8faa06e339333dd8fc91d486a",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lumina2.py",
        source_sha256: "f6af934b3d5014c6df37bb527167d1f94e44b1309079f57d4cb2f9460729da84",
        symbol: "Gemma2_2BModel",
        symbol_sha256: "14c5441e0cb62713f85d8745a0188b10f42b6fec6b1eb754374ddadb3edf30d3",
        behavior: CompositeSymbolBehavior::DecoderDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lumina2.py",
        source_sha256: "f6af934b3d5014c6df37bb527167d1f94e44b1309079f57d4cb2f9460729da84",
        symbol: "Gemma3_4BModel",
        symbol_sha256: "e05fcc6bacdfad8efcf71ed585917bdef406acca06dc56b12c4dfefe6b0dcf41",
        behavior: CompositeSymbolBehavior::DecoderDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lumina2.py",
        source_sha256: "f6af934b3d5014c6df37bb527167d1f94e44b1309079f57d4cb2f9460729da84",
        symbol: "Gemma3_4B_Vision_Model",
        symbol_sha256: "5423aa127a402826fae215a66139a82c105db7d5ca9fe1b5835150136deda488",
        behavior: CompositeSymbolBehavior::MultimodalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lumina2.py",
        source_sha256: "f6af934b3d5014c6df37bb527167d1f94e44b1309079f57d4cb2f9460729da84",
        symbol: "LuminaModel",
        symbol_sha256: "05d11def74581b02d0d690e2918b256c7f52d81a62d360089a4cce48d6f2f4b6",
        behavior: CompositeSymbolBehavior::ModelAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/lumina2.py",
        source_sha256: "f6af934b3d5014c6df37bb527167d1f94e44b1309079f57d4cb2f9460729da84",
        symbol: "te",
        symbol_sha256: "78a69ac379ee78b4ac448329d51f1c1907c17e4a38964b996c3aabe4955de2e8",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/newbie.py",
        source_sha256: "444a1bed8e47d27a1b5672e488689f8adb92879198326b99fc1ca03180d52ac4",
        symbol: "NewBieTokenizer",
        symbol_sha256: "a479c78e74e6aeec8d0003fb2e9ea721e22b6e26d96d5943579bdbf0a16de38a",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/newbie.py",
        source_sha256: "444a1bed8e47d27a1b5672e488689f8adb92879198326b99fc1ca03180d52ac4",
        symbol: "NewBieTEModel",
        symbol_sha256: "bca93940940828c1b85f9e2a9558bce31b5553cbb2bad3be6bdfadd4109bc148",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/newbie.py",
        source_sha256: "444a1bed8e47d27a1b5672e488689f8adb92879198326b99fc1ca03180d52ac4",
        symbol: "te",
        symbol_sha256: "298ea40c0bd530a0ab5a8a8589cddb88701ecf8670bdd100ed6b96afde71d2b6",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/omnigen2.py",
        source_sha256: "ef76972277b8b2141172ea2e53685bdeb26f2ff383c679946f10ff3396067c92",
        symbol: "Qwen25_3BTokenizer",
        symbol_sha256: "87d2574112517fcafeca4165c2cdea26e90b7332df3694d516ec9041e188519b",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/omnigen2.py",
        source_sha256: "ef76972277b8b2141172ea2e53685bdeb26f2ff383c679946f10ff3396067c92",
        symbol: "Omnigen2Tokenizer",
        symbol_sha256: "a2e38235634e364fd7f71e400d87b3da4593a4ae1c4024f5a5478dd2f0298a3d",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/omnigen2.py",
        source_sha256: "ef76972277b8b2141172ea2e53685bdeb26f2ff383c679946f10ff3396067c92",
        symbol: "Qwen25_3BModel",
        symbol_sha256: "8bd759d680f3b7ee38cde8e87763ee45a8b4af4d5a07b7fef6a536dc02caf200",
        behavior: CompositeSymbolBehavior::DecoderDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/omnigen2.py",
        source_sha256: "ef76972277b8b2141172ea2e53685bdeb26f2ff383c679946f10ff3396067c92",
        symbol: "Omnigen2Model",
        symbol_sha256: "d69735a7620eb014a06ce4014c0ad113635e9f015135adcd3422d34cce883fee",
        behavior: CompositeSymbolBehavior::ModelAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/omnigen2.py",
        source_sha256: "ef76972277b8b2141172ea2e53685bdeb26f2ff383c679946f10ff3396067c92",
        symbol: "te",
        symbol_sha256: "e26c135047a9720e2be8311436a72355a5d0e0dccbef72f057c91951ad62390a",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/pixart_t5.py",
        source_sha256: "9645107e3fcaafb8adf74b74f7444bc7cf4c9e8b05cdcb7651d65b1798d3d8ac",
        symbol: "T5XXLModel",
        symbol_sha256: "d708233c9e17f36d0bf61bdd91b0b8f170cd999fe728de7399266fc76d6d65ce",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/pixart_t5.py",
        source_sha256: "9645107e3fcaafb8adf74b74f7444bc7cf4c9e8b05cdcb7651d65b1798d3d8ac",
        symbol: "PixArtT5XXL",
        symbol_sha256: "6340c7b592ee5590f7aa109f16d803f24e5777ba71c213484662daf001e41799",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/pixart_t5.py",
        source_sha256: "9645107e3fcaafb8adf74b74f7444bc7cf4c9e8b05cdcb7651d65b1798d3d8ac",
        symbol: "T5XXLTokenizer",
        symbol_sha256: "98212e46e8e2ff7837fb75fa8c4aef750820afc4ac35d903d893603845a81685",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/pixart_t5.py",
        source_sha256: "9645107e3fcaafb8adf74b74f7444bc7cf4c9e8b05cdcb7651d65b1798d3d8ac",
        symbol: "PixArtTokenizer",
        symbol_sha256: "0acd61c9ab7ba2b1b23810cc209c7794f51f705137045f2391b0fcd594097e5c",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/pixart_t5.py",
        source_sha256: "9645107e3fcaafb8adf74b74f7444bc7cf4c9e8b05cdcb7651d65b1798d3d8ac",
        symbol: "pixart_te",
        symbol_sha256: "10b2ded3cbec0036787692cb99eedc56f92f0e02ba6006109fe5b092f7ddb828",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/pixeldit.py",
        source_sha256: "316ea00a65f2a4df3bcb57dac1ffe51cb83024a04d9e43ca6aba4a8d8119e409",
        symbol: "PixelDiTGemma2_2BModel",
        symbol_sha256: "70baae9fd495f32a01e9402b4614dac1d62de0c8d7a8511e00215d0707179b8f",
        behavior: CompositeSymbolBehavior::DecoderDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/pixeldit.py",
        source_sha256: "316ea00a65f2a4df3bcb57dac1ffe51cb83024a04d9e43ca6aba4a8d8119e409",
        symbol: "PixelDiTGemma2Tokenizer",
        symbol_sha256: "e6e14f34aa5a055899c66af51cbb5bc5bfd8f3e3abd65e84a6cd29a31b4795c7",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/pixeldit.py",
        source_sha256: "316ea00a65f2a4df3bcb57dac1ffe51cb83024a04d9e43ca6aba4a8d8119e409",
        symbol: "PixelDiTGemma2TE",
        symbol_sha256: "d3af810e9cb8ac80622355558f2ec6f0846547c1fea7d60e1691912792451d34",
        behavior: CompositeSymbolBehavior::DecoderDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/pixeldit.py",
        source_sha256: "316ea00a65f2a4df3bcb57dac1ffe51cb83024a04d9e43ca6aba4a8d8119e409",
        symbol: "pixeldit_te",
        symbol_sha256: "a11f4cd133d7f3e3efdfdb577bb534e17dea76e4730f17b39aa9d141fcc3352d",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/qwen_image.py",
        source_sha256: "9eab4371988f5b5e55260aba00e3984996d8efd4851569cc7e54b0e39811c1a9",
        symbol: "Qwen25_7BVLITokenizer",
        symbol_sha256: "cc550848b03fab7e25a78fb57a7019ea4a709c8658cd5b6338f35c3c86c120d7",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/qwen_image.py",
        source_sha256: "9eab4371988f5b5e55260aba00e3984996d8efd4851569cc7e54b0e39811c1a9",
        symbol: "QwenImageTokenizer",
        symbol_sha256: "7c4f83dc9adbfe600881535c520cd8d04d374d44ac303542c484b2fddd0aa617",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/qwen_image.py",
        source_sha256: "9eab4371988f5b5e55260aba00e3984996d8efd4851569cc7e54b0e39811c1a9",
        symbol: "Qwen25_7BVLIModel",
        symbol_sha256: "d235e0d9f095bad439662dba571c47458fbf0123e77f51a7757f9e00138eb4f6",
        behavior: CompositeSymbolBehavior::MultimodalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/qwen_image.py",
        source_sha256: "9eab4371988f5b5e55260aba00e3984996d8efd4851569cc7e54b0e39811c1a9",
        symbol: "QwenImageTEModel",
        symbol_sha256: "92e08a7f1a14e3753c23b430c8552ffaf921c984c7bb21bf5b9dd19a2c26fbcf",
        behavior: CompositeSymbolBehavior::DecoderDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/qwen_image.py",
        source_sha256: "9eab4371988f5b5e55260aba00e3984996d8efd4851569cc7e54b0e39811c1a9",
        symbol: "te",
        symbol_sha256: "699ba05c72e9721d83d7e47ea741ebe7de5c023fdc2777154167fbe62e72054e",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sa3.py",
        source_sha256: "e6603d11cf57923518b1393d0fe915d590f23b348072f61c11aa2e83a770f994",
        symbol: "T5GemmaEncoderConfig",
        symbol_sha256: "73d8f940473b6d4dad69a5d24c2f31d449be27628ce70716bd0cb9b86c04d0b4",
        behavior: CompositeSymbolBehavior::Profile,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sa3.py",
        source_sha256: "e6603d11cf57923518b1393d0fe915d590f23b348072f61c11aa2e83a770f994",
        symbol: "T5GemmaAttention",
        symbol_sha256: "ee97563845f3dca0e1bcda84bb03b06acc714da97ba364b136002d2a4f5f876a",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sa3.py",
        source_sha256: "e6603d11cf57923518b1393d0fe915d590f23b348072f61c11aa2e83a770f994",
        symbol: "T5GemmaBlock",
        symbol_sha256: "6e216e123cbe44665de9eed997314d2bd2c3c10b2b8446b936c8870899ada22a",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sa3.py",
        source_sha256: "e6603d11cf57923518b1393d0fe915d590f23b348072f61c11aa2e83a770f994",
        symbol: "T5GemmaEncoder",
        symbol_sha256: "6baeb85b6bbd2c2a08ef980cf3b367d8fec93b5a8adcdea2ec187a0e14d784b5",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sa3.py",
        source_sha256: "e6603d11cf57923518b1393d0fe915d590f23b348072f61c11aa2e83a770f994",
        symbol: "T5GemmaBody",
        symbol_sha256: "5be213b0aa78186ff6c2180373d3ad6ebbbc13fe0c8dbd5446e606cb476cfc06",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sa3.py",
        source_sha256: "e6603d11cf57923518b1393d0fe915d590f23b348072f61c11aa2e83a770f994",
        symbol: "T5GemmaModel",
        symbol_sha256: "feea1dbc42a13f65cc1d123b5a5fc19434956bb91edc5bbd404632a8213964c3",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sa3.py",
        source_sha256: "e6603d11cf57923518b1393d0fe915d590f23b348072f61c11aa2e83a770f994",
        symbol: "T5GemmaSDClipModel",
        symbol_sha256: "13eb2b4a4e85b5c0761e39299dd9e4eac05baaea0c1e6024be67e37bd3c988e4",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sa3.py",
        source_sha256: "e6603d11cf57923518b1393d0fe915d590f23b348072f61c11aa2e83a770f994",
        symbol: "T5GemmaSDTokenizer",
        symbol_sha256: "8914e81f5cbfa117d4aecd2ee60fca88c1b7745dfc4eb78464477ba99856d365",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sa3.py",
        source_sha256: "e6603d11cf57923518b1393d0fe915d590f23b348072f61c11aa2e83a770f994",
        symbol: "SAT5GemmaTokenizer",
        symbol_sha256: "67355ff66668410bf6bcac68bc668b958b7f75eeb65477bdd6cd4a2f7972b6a8",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sa3.py",
        source_sha256: "e6603d11cf57923518b1393d0fe915d590f23b348072f61c11aa2e83a770f994",
        symbol: "SAT5GemmaModel",
        symbol_sha256: "68f36dd198b403929bff8fc4bb1d466cb8b102e71b09e3983b77c91abfc34a76",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sa_t5.py",
        source_sha256: "ec3913c5fd9a5d209d8f21ee031a61f30c30f9b1c9d6e7c96124429a699a7421",
        symbol: "T5BaseModel",
        symbol_sha256: "597a1baad260accac1b326369b7112a66d2472a82ff56575dbea9e4e3c35abba",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sa_t5.py",
        source_sha256: "ec3913c5fd9a5d209d8f21ee031a61f30c30f9b1c9d6e7c96124429a699a7421",
        symbol: "T5BaseTokenizer",
        symbol_sha256: "58e85d60fc84b97e91048c14a5f7322827bd4d5b9d2ae8f00ce54ec030f5eea5",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sa_t5.py",
        source_sha256: "ec3913c5fd9a5d209d8f21ee031a61f30c30f9b1c9d6e7c96124429a699a7421",
        symbol: "SAT5Tokenizer",
        symbol_sha256: "9ea95d9676630307f930499ea694bc920e3129dae9a1a1c931f1b89e16fcc696",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sa_t5.py",
        source_sha256: "ec3913c5fd9a5d209d8f21ee031a61f30c30f9b1c9d6e7c96124429a699a7421",
        symbol: "SAT5Model",
        symbol_sha256: "ee7275b12a1e492d92603610eb2d5dafe87196454d8d3e5988e3a01c20f3ff24",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sd2_clip.py",
        source_sha256: "f271a816164925da14895a742a323cd3149d651a8600c916d6ee71080fde4f6e",
        symbol: "SD2ClipHModel",
        symbol_sha256: "1520100610113b222e163f90e9d2123c770e61cbce856067f79f18d96251967d",
        behavior: CompositeSymbolBehavior::ModelAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sd2_clip.py",
        source_sha256: "f271a816164925da14895a742a323cd3149d651a8600c916d6ee71080fde4f6e",
        symbol: "SD2ClipHTokenizer",
        symbol_sha256: "9417de190dd6c22465f9d106aec50c7d0718cd24f9f2c0d153746dc8d87a45f2",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sd2_clip.py",
        source_sha256: "f271a816164925da14895a742a323cd3149d651a8600c916d6ee71080fde4f6e",
        symbol: "SD2Tokenizer",
        symbol_sha256: "f9a4cb126392f5b4fdcdb57e460db46d4def7787773052cdcdc0b59fccd015d2",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sd2_clip.py",
        source_sha256: "f271a816164925da14895a742a323cd3149d651a8600c916d6ee71080fde4f6e",
        symbol: "SD2ClipModel",
        symbol_sha256: "da054cb06c8803cf264e987c872b4e4a91555d48c3bbdb7fab594b20bb67b088",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sd3_clip.py",
        source_sha256: "61bde2b4a428779469a01f342a9195d115f2a207c7a3d2c0d6c0a5715fdc03ed",
        symbol: "T5XXLModel",
        symbol_sha256: "730522c345292d92a2a46e5a2f2fea14a9cdaabf5dc5cd77e541bd0a53d10689",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sd3_clip.py",
        source_sha256: "61bde2b4a428779469a01f342a9195d115f2a207c7a3d2c0d6c0a5715fdc03ed",
        symbol: "t5_xxl_detect",
        symbol_sha256: "f84ea16ad83bb659451f62c74c338d92cf8f0432e36a7d4f0d1dfa926bb5ac48",
        behavior: CompositeSymbolBehavior::Profile,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sd3_clip.py",
        source_sha256: "61bde2b4a428779469a01f342a9195d115f2a207c7a3d2c0d6c0a5715fdc03ed",
        symbol: "T5XXLTokenizer",
        symbol_sha256: "9d3475246520351c4a6c47100cfd29cd61bdb9fdd47cd1fbef1ea6713bcb548d",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sd3_clip.py",
        source_sha256: "61bde2b4a428779469a01f342a9195d115f2a207c7a3d2c0d6c0a5715fdc03ed",
        symbol: "SD3Tokenizer",
        symbol_sha256: "03db75d3fe567da0ab1d72165c286133b65343a97cbae212a73a64ed735a6192",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sd3_clip.py",
        source_sha256: "61bde2b4a428779469a01f342a9195d115f2a207c7a3d2c0d6c0a5715fdc03ed",
        symbol: "SD3ClipModel",
        symbol_sha256: "668c6c93e43c482de477afcdb00013dd76ba7b073e31b81779b26780ce867e96",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/sd3_clip.py",
        source_sha256: "61bde2b4a428779469a01f342a9195d115f2a207c7a3d2c0d6c0a5715fdc03ed",
        symbol: "sd3_clip",
        symbol_sha256: "30a7064b104fd861aa24916c2a3010667846cf5098dceed55acaf0780a94ed7b",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/wan.py",
        source_sha256: "e9a6ca356790c76f7a36550842fc8a93c8c584cd175d69615eaddb7b4cd276bc",
        symbol: "UMT5XXlModel",
        symbol_sha256: "6f2e766954436da03502cbfe2d53d7b82186cf9eb656921f6b04824c4958ab3e",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/wan.py",
        source_sha256: "e9a6ca356790c76f7a36550842fc8a93c8c584cd175d69615eaddb7b4cd276bc",
        symbol: "UMT5XXlTokenizer",
        symbol_sha256: "0da7c9643ac9f4c4cce3f56d58fa6e8e2e6df947d9f4577d628946f67fdfca7c",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/wan.py",
        source_sha256: "e9a6ca356790c76f7a36550842fc8a93c8c584cd175d69615eaddb7b4cd276bc",
        symbol: "WanT5Tokenizer",
        symbol_sha256: "b9bce69b0f7dc6e9d0ff89ab450eee02934a79e788454c3a434389d502875629",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/wan.py",
        source_sha256: "e9a6ca356790c76f7a36550842fc8a93c8c584cd175d69615eaddb7b4cd276bc",
        symbol: "WanT5Model",
        symbol_sha256: "4c8ecf4fa1d748a215a9c90402eb3c7e32a44471906fa223ab1b4fd89e3974e6",
        behavior: CompositeSymbolBehavior::BidirectionalDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/wan.py",
        source_sha256: "e9a6ca356790c76f7a36550842fc8a93c8c584cd175d69615eaddb7b4cd276bc",
        symbol: "te",
        symbol_sha256: "b3df9b2b44b870ebd0d1179fd93d1cb21eb8395eea49c3d26638905c5c7959c6",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/z_image.py",
        source_sha256: "c6d1d15054ec8f6b1b5cddb759f5e637aa64dd9ffeaa937e266f03e035550aed",
        symbol: "Qwen3Tokenizer",
        symbol_sha256: "ec1b653ce8c96257dc3191c44ac6cc15daa88faa4c06638b9f7ab66397c02850",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/z_image.py",
        source_sha256: "c6d1d15054ec8f6b1b5cddb759f5e637aa64dd9ffeaa937e266f03e035550aed",
        symbol: "ZImageTokenizer",
        symbol_sha256: "6f1c2164681f87beb005a550cedd1b6d3bd7bf69068ab123577b60a2971c9d24",
        behavior: CompositeSymbolBehavior::TokenizerAdapter,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/z_image.py",
        source_sha256: "c6d1d15054ec8f6b1b5cddb759f5e637aa64dd9ffeaa937e266f03e035550aed",
        symbol: "Qwen3_4BModel",
        symbol_sha256: "86eb836093c6854634964b64e08877d31eebd5208c211b2c59b8e0f7fa7a233e",
        behavior: CompositeSymbolBehavior::DecoderDelegation,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/z_image.py",
        source_sha256: "c6d1d15054ec8f6b1b5cddb759f5e637aa64dd9ffeaa937e266f03e035550aed",
        symbol: "ZImageTEModel",
        symbol_sha256: "bad318c87c1df149860c6997d5809db322390979705830b984b497009b6ba79b",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
    CompositeContractFact {
        source_path: "projects/comfy/ComfyUI/sim/text_encoders/z_image.py",
        source_sha256: "c6d1d15054ec8f6b1b5cddb759f5e637aa64dd9ffeaa937e266f03e035550aed",
        symbol: "te",
        symbol_sha256: "e09fed3b8410a7936d510b2f57763b3fcec431796aa542b214aafb5f96501e60",
        behavior: CompositeSymbolBehavior::CompositeOrdering,
    },
];

pub fn composite_contract_fact(
    source_path: &str,
    symbol: &str,
) -> Option<&'static CompositeContractFact> {
    COMPOSITE_TEXT_ENCODER_CONTRACTS
        .iter()
        .find(|fact| fact.source_path == source_path && fact.symbol == symbol)
}

pub fn composite_symbol_behavior(
    source_path: &str,
    symbol: &str,
) -> Option<CompositeSymbolBehavior> {
    composite_contract_fact(source_path, symbol).map(|fact| fact.behavior)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositeOwner {
    Bidirectional,
    Decoder,
    Multimodal,
    ClipText,
    ClipVision,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositeHiddenJoin {
    Sequence,
    Feature,
    Select(usize),
    Sd3,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CompositePooledPolicy {
    None,
    FirstAvailable,
    ConcatenateAvailable,
    ConcatenatePrefix(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CompositeExecutionPlan {
    pub owners: &'static [CompositeOwner],
    pub hidden_join: CompositeHiddenJoin,
    pub pooled_policy: CompositePooledPolicy,
}

const FLUX_OWNERS: [CompositeOwner; 2] = [CompositeOwner::ClipText, CompositeOwner::Bidirectional];
const SD3_OWNERS: [CompositeOwner; 3] = [
    CompositeOwner::ClipText,
    CompositeOwner::ClipText,
    CompositeOwner::Bidirectional,
];
const SINGLE_T5_OWNER: [CompositeOwner; 1] = [CompositeOwner::Bidirectional];
const SINGLE_DECODER_OWNER: [CompositeOwner; 1] = [CompositeOwner::Decoder];
const SINGLE_CLIP_OWNER: [CompositeOwner; 1] = [CompositeOwner::ClipText];
const MULTIMODAL_DECODER_OWNERS: [CompositeOwner; 2] =
    [CompositeOwner::Decoder, CompositeOwner::Multimodal];
const DECODER_T5_OWNERS: [CompositeOwner; 2] =
    [CompositeOwner::Decoder, CompositeOwner::Bidirectional];
const T5_PAIR_OWNERS: [CompositeOwner; 2] =
    [CompositeOwner::Bidirectional, CompositeOwner::Bidirectional];
const HIDREAM_OWNERS: [CompositeOwner; 4] = [
    CompositeOwner::ClipText,
    CompositeOwner::ClipText,
    CompositeOwner::Bidirectional,
    CompositeOwner::Decoder,
];

pub fn composite_execution_plan(name: &str) -> Option<CompositeExecutionPlan> {
    Some(match name {
        "flux" => CompositeExecutionPlan {
            owners: &FLUX_OWNERS,
            hidden_join: CompositeHiddenJoin::Select(1),
            pooled_policy: CompositePooledPolicy::FirstAvailable,
        },
        "flux2" | "klein" | "ace15" | "ernie" | "hunyuan_video" | "kandinsky5" | "ltxav"
        | "lumina2" | "pixeldit" | "z_image" => CompositeExecutionPlan {
            owners: &SINGLE_DECODER_OWNER,
            hidden_join: CompositeHiddenJoin::Select(0),
            pooled_policy: CompositePooledPolicy::FirstAvailable,
        },
        "sd3" => CompositeExecutionPlan {
            owners: &SD3_OWNERS,
            hidden_join: CompositeHiddenJoin::Sd3,
            pooled_policy: CompositePooledPolicy::ConcatenatePrefix(2),
        },
        "anima" => CompositeExecutionPlan {
            owners: &SINGLE_DECODER_OWNER,
            hidden_join: CompositeHiddenJoin::Select(0),
            pooled_policy: CompositePooledPolicy::FirstAvailable,
        },
        "boogu" | "hidream_o1" | "krea2" | "longcat_image" | "omnigen2" | "qwen_image" => {
            CompositeExecutionPlan {
                owners: &MULTIMODAL_DECODER_OWNERS,
                hidden_join: CompositeHiddenJoin::Select(0),
                pooled_policy: CompositePooledPolicy::FirstAvailable,
            }
        }
        "hidream" => CompositeExecutionPlan {
            owners: &HIDREAM_OWNERS,
            hidden_join: CompositeHiddenJoin::Select(2),
            pooled_policy: CompositePooledPolicy::ConcatenatePrefix(2),
        },
        "hunyuan_image" | "newbie" => CompositeExecutionPlan {
            owners: &DECODER_T5_OWNERS,
            hidden_join: CompositeHiddenJoin::Select(0),
            pooled_policy: CompositePooledPolicy::FirstAvailable,
        },
        "hydit" => CompositeExecutionPlan {
            owners: &T5_PAIR_OWNERS,
            hidden_join: CompositeHiddenJoin::Select(0),
            pooled_policy: CompositePooledPolicy::FirstAvailable,
        },
        "long_clipl" | "sd2" => CompositeExecutionPlan {
            owners: &SINGLE_CLIP_OWNER,
            hidden_join: CompositeHiddenJoin::Select(0),
            pooled_policy: CompositePooledPolicy::FirstAvailable,
        },
        "ace" | "aura_t5" | "cogvideo" | "cosmos" | "genmo" | "ltxv" | "pixart_t5" | "sa3"
        | "sa_t5" | "wan" => CompositeExecutionPlan {
            owners: &SINGLE_T5_OWNER,
            hidden_join: CompositeHiddenJoin::Select(0),
            pooled_policy: CompositePooledPolicy::FirstAvailable,
        },
        _ => return None,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct CompositeConditioningInput<'a> {
    pub owner: CompositeOwner,
    pub hidden: &'a Tensor,
    pub pooled: Option<&'a Tensor>,
}

#[derive(Clone, Debug)]
pub struct CompositeConditioningOutput {
    pub hidden: Tensor,
    pub pooled: Option<Tensor>,
}

#[derive(Clone, Debug)]
pub struct SourceClipCompositionOutput {
    pub hidden: Tensor,
    pub pooled: Tensor,
    pub conditioning_llama3: Option<Tensor>,
}

#[derive(Debug, Error)]
pub enum CompositeTextEncoderError {
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    NativeTensor(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Shape(#[from] ShapeLayoutTransformPartTwoError),
    #[error(transparent)]
    ShapeThree(#[from] ShapeLayoutTransformPartThreeError),
    #[error(transparent)]
    Indexing(#[from] IndexingMaskingPartOneError),
    #[error(transparent)]
    Delegate(Box<MultimodalTextError>),
    #[error(transparent)]
    Rng(#[from] RngError),
    #[error("composite text-encoder input is invalid: {0}")]
    InvalidInput(&'static str),
    #[error("composite text-encoder arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("composite text-encoder execution was cancelled")]
    Cancelled,
}

pub fn compose_source_sdxl(
    backend: &CpuBackend,
    clip_l: &Tensor,
    clip_g: &Tensor,
    projected_clip_g_pooled: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<SourceClipCompositionOutput, CompositeTextEncoderError> {
    let hidden = concatenate_source_clip_hidden(backend, clip_l, clip_g, context)?;
    require_pooled_width(projected_clip_g_pooled, None)?;
    Ok(SourceClipCompositionOutput {
        hidden,
        pooled: projected_clip_g_pooled.clone(),
        conditioning_llama3: None,
    })
}

pub fn compose_source_sd3(
    backend: &CpuBackend,
    clip_l: Option<(&Tensor, &Tensor)>,
    clip_g: Option<(&Tensor, &Tensor)>,
    t5: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<SourceClipCompositionOutput, CompositeTextEncoderError> {
    let lg = match (clip_l, clip_g) {
        (Some((left, _)), Some((right, _))) => Some(concatenate_source_clip_hidden(
            backend, left, right, context,
        )?),
        (Some((left, _)), None) => Some(pad_source_features(backend, left, 0, 1_280, context)?),
        (None, Some((right, _))) => Some(pad_source_features(backend, right, 768, 0, context)?),
        (None, None) => None,
    };
    let hidden = match (lg, t5) {
        (Some(lg), Some(t5)) => {
            let lg = pad_source_features(backend, &lg, 0, 2_048, context)?;
            require_hidden_width(t5, 4_096)?;
            torch_cat_with_context_exact_native(backend, &[lg, t5.clone()], -2, context)?
        }
        (Some(lg), None) => pad_source_features(backend, &lg, 0, 2_048, context)?,
        (None, Some(t5)) => {
            require_hidden_width(t5, 4_096)?;
            t5.clone()
        }
        (None, None) => zero_source_tensor(backend, &[1, 77, 4_096], context)?,
    };
    let pooled_l = match clip_l {
        Some((_, pooled)) => {
            require_pooled_width(pooled, Some(768))?;
            pooled.clone()
        }
        None => zero_source_tensor(backend, &[1, 768], context)?,
    };
    let pooled_g = match clip_g {
        Some((_, pooled)) => {
            require_pooled_width(pooled, Some(1_280))?;
            pooled.clone()
        }
        None => zero_source_tensor(backend, &[1, 1_280], context)?,
    };
    let pooled = torch_cat_with_context_exact_native(backend, &[pooled_l, pooled_g], -1, context)?;
    Ok(SourceClipCompositionOutput {
        hidden,
        pooled,
        conditioning_llama3: None,
    })
}

pub fn compose_source_hidream(
    backend: &CpuBackend,
    clip_l_pooled: Option<&Tensor>,
    clip_g_pooled: Option<&Tensor>,
    t5: Option<&Tensor>,
    llama_all_layers: Option<&Tensor>,
    context: &ExecutionContext<'_>,
) -> Result<SourceClipCompositionOutput, CompositeTextEncoderError> {
    let hidden = match t5 {
        Some(t5) => {
            require_hidden_width(t5, 4_096)?;
            t5.clone()
        }
        None => zero_source_tensor(backend, &[1, 128, 4_096], context)?,
    };
    let pooled_l = match clip_l_pooled {
        Some(pooled) => {
            require_pooled_width(pooled, Some(768))?;
            pooled.clone()
        }
        None => zero_source_tensor(backend, &[1, 768], context)?,
    };
    let pooled_g = match clip_g_pooled {
        Some(pooled) => {
            require_pooled_width(pooled, Some(1_280))?;
            pooled.clone()
        }
        None => zero_source_tensor(backend, &[1, 1_280], context)?,
    };
    let pooled = torch_cat_with_context_exact_native(backend, &[pooled_l, pooled_g], -1, context)?;
    let conditioning_llama3 = match llama_all_layers {
        Some(llama) => {
            let shape = llama.descriptor().shape();
            if shape.len() != 4 || shape[0] != 1 || shape[1] < 2 || shape[3] != 4_096 {
                return Err(CompositeTextEncoderError::InvalidInput(
                    "HiDream Llama all-layer output must be [1, layers+1, tokens, 4096]",
                ));
            }
            narrow_method_exact_native(llama, 1, 1, shape[1] - 1, context.cancellation)?
        }
        None => zero_source_tensor(backend, &[1, 32, 1, 4_096], context)?,
    };
    Ok(SourceClipCompositionOutput {
        hidden,
        pooled,
        conditioning_llama3: Some(conditioning_llama3),
    })
}

fn concatenate_source_clip_hidden(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, CompositeTextEncoderError> {
    let left_shape = require_rank_three_hidden(left)?;
    let right_shape = require_rank_three_hidden(right)?;
    if left_shape[0] != right_shape[0] {
        return Err(CompositeTextEncoderError::InvalidInput(
            "CLIP hidden batches differ",
        ));
    }
    let tokens = left_shape[1].min(right_shape[1]);
    let left = narrow_method_exact_native(left, 1, 0, tokens, context.cancellation)?;
    let right = narrow_method_exact_native(right, 1, 0, tokens, context.cancellation)?;
    Ok(torch_cat_with_context_exact_native(
        backend,
        &[left, right],
        -1,
        context,
    )?)
}

fn pad_source_features(
    backend: &CpuBackend,
    tensor: &Tensor,
    left: u64,
    right: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, CompositeTextEncoderError> {
    require_rank_three_hidden(tensor)?;
    if left == 0 && right == 0 {
        return Ok(tensor.clone());
    }
    Ok(functional_pad_with_context_exact_native(
        backend,
        tensor,
        &[
            i64::try_from(left)
                .map_err(|_| CompositeTextEncoderError::Overflow("CLIP left padding"))?,
            i64::try_from(right)
                .map_err(|_| CompositeTextEncoderError::Overflow("CLIP right padding"))?,
        ],
        FunctionalPadMode::Constant,
        Some(DecodedScalar::Signed(0)),
        context,
    )?)
}

fn zero_source_tensor(
    backend: &CpuBackend,
    shape: &[u64],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, CompositeTextEncoderError> {
    let elements = shape.iter().try_fold(1_usize, |elements, dimension| {
        elements
            .checked_mul(
                usize::try_from(*dimension)
                    .map_err(|_| CompositeTextEncoderError::Overflow("zero tensor shape"))?,
            )
            .ok_or(CompositeTextEncoderError::Overflow("zero tensor elements"))
    })?;
    context.check()?;
    let mut values: CpuWorkspaceVec<f32> = backend.workspace_vec(context, elements)?;
    for index in 0..elements {
        if index.is_multiple_of(4_096) {
            context.check()?;
        }
        values.try_push(0.0)?;
    }
    Ok(tensor_from_f32(backend, shape, &values, context)?)
}

fn require_rank_three_hidden(tensor: &Tensor) -> Result<&[u64], CompositeTextEncoderError> {
    let shape = tensor.descriptor().shape();
    if shape.len() != 3 || shape[0] == 0 || shape[1] == 0 || shape[2] == 0 {
        return Err(CompositeTextEncoderError::InvalidInput(
            "CLIP hidden tensor must be nonempty rank three",
        ));
    }
    Ok(shape)
}

fn require_hidden_width(tensor: &Tensor, width: u64) -> Result<(), CompositeTextEncoderError> {
    if require_rank_three_hidden(tensor)?[2] != width {
        return Err(CompositeTextEncoderError::InvalidInput(
            "CLIP hidden feature width is invalid",
        ));
    }
    Ok(())
}

fn require_pooled_width(
    tensor: &Tensor,
    width: Option<u64>,
) -> Result<(), CompositeTextEncoderError> {
    let shape = tensor.descriptor().shape();
    if shape.len() != 2
        || shape[0] != 1
        || shape[1] == 0
        || width.is_some_and(|width| shape[1] != width)
    {
        return Err(CompositeTextEncoderError::InvalidInput(
            "CLIP pooled tensor shape is invalid",
        ));
    }
    Ok(())
}

impl From<sim_types::CancellationError> for CompositeTextEncoderError {
    fn from(_: sim_types::CancellationError) -> Self {
        Self::Cancelled
    }
}

impl From<MultimodalTextError> for CompositeTextEncoderError {
    fn from(error: MultimodalTextError) -> Self {
        Self::Delegate(Box::new(error))
    }
}

pub fn compose_conditioning(
    backend: &CpuBackend,
    plan: CompositeExecutionPlan,
    inputs: &[CompositeConditioningInput<'_>],
    context: &ExecutionContext<'_>,
) -> Result<CompositeConditioningOutput, CompositeTextEncoderError> {
    context.cancellation.check()?;
    if inputs.is_empty() || inputs.len() != plan.owners.len() {
        return Err(CompositeTextEncoderError::InvalidInput(
            "the input count must match the execution plan",
        ));
    }
    if !inputs
        .iter()
        .zip(plan.owners)
        .all(|(input, owner)| input.owner == *owner)
    {
        return Err(CompositeTextEncoderError::InvalidInput(
            "owner order does not match the execution plan",
        ));
    }
    let hidden = match plan.hidden_join {
        CompositeHiddenJoin::Select(index) => inputs
            .get(index)
            .ok_or(CompositeTextEncoderError::InvalidInput(
                "selected hidden owner is outside the execution plan",
            ))?
            .hidden
            .clone(),
        CompositeHiddenJoin::Sd3 => compose_sd3_hidden(backend, inputs, context)?,
        CompositeHiddenJoin::Sequence | CompositeHiddenJoin::Feature => {
            let hidden_inputs = inputs
                .iter()
                .map(|input| input.hidden.clone())
                .collect::<Vec<_>>();
            if let [only] = hidden_inputs.as_slice() {
                only.clone()
            } else {
                let dimension = match plan.hidden_join {
                    CompositeHiddenJoin::Sequence => -2,
                    CompositeHiddenJoin::Feature => -1,
                    CompositeHiddenJoin::Select(_) | CompositeHiddenJoin::Sd3 => unreachable!(),
                };
                torch_cat_with_context_exact_native(backend, &hidden_inputs, dimension, context)?
            }
        }
    };
    let pooled_inputs = inputs
        .iter()
        .filter_map(|input| input.pooled.cloned())
        .collect::<Vec<_>>();
    let pooled = match plan.pooled_policy {
        CompositePooledPolicy::None => None,
        CompositePooledPolicy::FirstAvailable => pooled_inputs.first().cloned(),
        CompositePooledPolicy::ConcatenateAvailable => match pooled_inputs.as_slice() {
            [] => None,
            [only] => Some(only.clone()),
            _ => Some(torch_cat_with_context_exact_native(
                backend,
                &pooled_inputs,
                -1,
                context,
            )?),
        },
        CompositePooledPolicy::ConcatenatePrefix(count) => {
            let selected = inputs
                .iter()
                .take(count)
                .filter_map(|input| input.pooled.cloned())
                .collect::<Vec<_>>();
            match selected.as_slice() {
                [] => None,
                [only] => Some(only.clone()),
                _ => Some(torch_cat_with_context_exact_native(
                    backend, &selected, -1, context,
                )?),
            }
        }
    };
    context.cancellation.check()?;
    Ok(CompositeConditioningOutput { hidden, pooled })
}

fn compose_sd3_hidden(
    backend: &CpuBackend,
    inputs: &[CompositeConditioningInput<'_>],
    context: &ExecutionContext<'_>,
) -> Result<Tensor, CompositeTextEncoderError> {
    let [clip_l, clip_g, t5] = inputs else {
        return Err(CompositeTextEncoderError::InvalidInput(
            "SD3 requires CLIP-L, CLIP-G, and T5 inputs",
        ));
    };
    let l_shape = clip_l.hidden.descriptor().shape();
    let g_shape = clip_g.hidden.descriptor().shape();
    let t5_shape = t5.hidden.descriptor().shape();
    if l_shape.len() != 3
        || g_shape.len() != 3
        || t5_shape.len() != 3
        || l_shape[0] != g_shape[0]
        || l_shape[0] != t5_shape[0]
        || t5_shape[2] != 4096
    {
        return Err(CompositeTextEncoderError::InvalidInput(
            "SD3 hidden inputs require compatible rank-three batches and 4096 T5 features",
        ));
    }
    let sequence = l_shape[1].min(g_shape[1]);
    let l = narrow_method_exact_native(clip_l.hidden, 1, 0, sequence, context.cancellation)?;
    let g = narrow_method_exact_native(clip_g.hidden, 1, 0, sequence, context.cancellation)?;
    let lg = torch_cat_with_context_exact_native(backend, &[l, g], -1, context)?;
    let lg_features = lg.descriptor().shape()[2];
    if lg_features > 4096 {
        return Err(CompositeTextEncoderError::InvalidInput(
            "SD3 CLIP feature width exceeds 4096",
        ));
    }
    let right = i64::try_from(4096 - lg_features)
        .map_err(|_| CompositeTextEncoderError::Overflow("SD3 CLIP padding"))?;
    let lg = if right == 0 {
        lg
    } else {
        functional_pad_with_context_exact_native(
            backend,
            &lg,
            &[0, right],
            FunctionalPadMode::Constant,
            Some(DecodedScalar::Signed(0)),
            context,
        )?
    };
    Ok(torch_cat_with_context_exact_native(
        backend,
        &[lg, t5.hidden.clone()],
        -2,
        context,
    )?)
}

pub fn delegate_bidirectional_text(
    owner: &NativeT5TextEncoder,
    backend: &CpuBackend,
    request: BidirectionalTextRequest<'_>,
    context: &ExecutionContext<'_>,
) -> Result<BidirectionalTextOutput, CompositeTextEncoderError> {
    Ok(run_bidirectional_text_owner(
        owner, backend, request, context,
    )?)
}

pub fn delegate_decoder_text(
    owner: &NativeDecoderTextEncoder,
    backend: &CpuBackend,
    request: DecoderTextRequest<'_>,
    context: &ExecutionContext<'_>,
) -> Result<DecoderTextOutput, CompositeTextEncoderError> {
    Ok(run_decoder_text_owner(owner, backend, request, context)?)
}

pub fn delegate_clip_text(
    owner: &NativeClipText,
    backend: &CpuBackend,
    request: ClipTextRequest<'_>,
    context: &ExecutionContext<'_>,
) -> Result<ClipTextOutput, CompositeTextEncoderError> {
    Ok(run_clip_text_owner(owner, backend, request, context)?)
}

pub fn delegate_clip_vision(
    owner: &mut NativeClipVision,
    backend: &CpuBackend,
    image: &Tensor,
    crop: bool,
    intermediate: ClipVisionIntermediate,
    context: &ExecutionContext<'_>,
) -> Result<ClipVisionOutput, CompositeTextEncoderError> {
    Ok(run_clip_vision_owner(
        owner,
        backend,
        image,
        crop,
        intermediate,
        context,
    )?)
}

pub fn number_to_text_i64(number: i64) -> String {
    if number == 0 {
        return "zero".to_owned();
    }
    if number < 0 {
        let magnitude = number.unsigned_abs();
        return format!("negative {}", unsigned_number_to_text(magnitude));
    }
    unsigned_number_to_text(number as u64)
}

fn unsigned_number_to_text(number: u64) -> String {
    const ONES: [&str; 20] = [
        "",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 10] = [
        "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    if number < 20 {
        return ONES[number as usize].to_owned();
    }
    if number < 100 {
        let tail = number % 10;
        return if tail == 0 {
            TENS[(number / 10) as usize].to_owned()
        } else {
            format!("{} {}", TENS[(number / 10) as usize], ONES[tail as usize])
        };
    }
    for (scale, name) in [
        (1_000_000_000_000_000_000_u64, "quintillion"),
        (1_000_000_000_000_000_u64, "quadrillion"),
        (1_000_000_000_000_u64, "trillion"),
        (1_000_000_000_u64, "billion"),
        (1_000_000_u64, "million"),
        (1_000_u64, "thousand"),
        (100_u64, "hundred"),
    ] {
        if number >= scale {
            let head = unsigned_number_to_text(number / scale);
            let tail = number % scale;
            return if tail == 0 {
                format!("{head} {name}")
            } else {
                format!("{head} {name} {}", unsigned_number_to_text(tail))
            };
        }
    }
    unreachable!("numbers below one hundred returned above")
}

pub fn expand_symbols_multilingual(text: &str, language: &str) -> String {
    if language != "en" {
        return text.to_owned();
    }
    let mut output = text.to_owned();
    for (symbol, replacement) in [
        ("&", " and "),
        ("@", " at "),
        ("%", " percent "),
        ("#", " hash "),
        ("$", " dollar "),
        ("£", " pound "),
        ("°", " degree "),
    ] {
        output = output.replace(symbol, replacement);
        while output.contains("  ") {
            output = output.replace("  ", " ");
        }
    }
    output.trim().to_owned()
}

pub fn expand_abbreviations_multilingual(text: &str, language: &str) -> String {
    if language != "en" {
        return text.to_owned();
    }
    let replacements = [
        ("mrs.", "misess"),
        ("mr.", "mister"),
        ("dr.", "doctor"),
        ("st.", "saint"),
        ("co.", "company"),
        ("jr.", "junior"),
        ("maj.", "major"),
        ("gen.", "general"),
        ("drs.", "doctors"),
        ("rev.", "reverend"),
        ("lt.", "lieutenant"),
        ("hon.", "honorable"),
        ("sgt.", "sergeant"),
        ("capt.", "captain"),
        ("esq.", "esquire"),
        ("ltd.", "limited"),
        ("col.", "colonel"),
        ("ft.", "fort"),
    ];
    replace_ascii_words_case_insensitive(text, &replacements)
}

fn replace_ascii_words_case_insensitive(text: &str, replacements: &[(&str, &str)]) -> String {
    let mut output = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let mut replaced = false;
        for (needle, replacement) in replacements {
            let end = index.saturating_add(needle.len());
            let left_boundary = index == 0 || !bytes[index - 1].is_ascii_alphanumeric();
            let right_boundary =
                end == bytes.len() || (end < bytes.len() && !bytes[end].is_ascii_alphanumeric());
            if left_boundary
                && right_boundary
                && end <= bytes.len()
                && bytes[index..end].eq_ignore_ascii_case(needle.as_bytes())
            {
                output.push_str(replacement);
                index = end;
                replaced = true;
                break;
            }
        }
        if !replaced {
            let character = text[index..].chars().next().expect("valid UTF-8 boundary");
            output.push(character);
            index += character.len_utf8();
        }
    }
    output
}

pub fn expand_numbers_multilingual(text: &str, language: &str) -> String {
    if language != "en" && language != "ru" {
        return text.to_owned();
    }
    let characters = text.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(text.len());
    let mut index = 0;
    while index < characters.len() {
        if !characters[index].is_ascii_digit() {
            output.push(characters[index]);
            index += 1;
            continue;
        }
        let mut integer_digits = String::new();
        while index < characters.len() {
            if characters[index].is_ascii_digit() {
                integer_digits.push(characters[index]);
                index += 1;
                continue;
            }
            if characters[index] == ','
                && index + 3 < characters.len()
                && characters[index + 1..=index + 3]
                    .iter()
                    .all(char::is_ascii_digit)
                && characters
                    .get(index + 4)
                    .is_none_or(|next| !next.is_ascii_digit())
            {
                index += 1;
                continue;
            }
            break;
        }
        if matches!(output.chars().next_back(), Some('$' | '£' | '€')) {
            output.pop();
        }
        if let Ok(number) = integer_digits.parse::<u64>() {
            output.push_str(&unsigned_number_to_text(number));
        } else {
            output.push_str(&integer_digits);
        }
        if index + 1 < characters.len()
            && matches!(characters[index], '.' | ',')
            && characters[index + 1].is_ascii_digit()
        {
            index += 1;
            let mut fractional_digits = Vec::new();
            while index < characters.len() && characters[index].is_ascii_digit() {
                fractional_digits.push(characters[index]);
                index += 1;
            }
            while fractional_digits.len() > 1 && fractional_digits.last() == Some(&'0') {
                fractional_digits.pop();
            }
            output.push_str(" point");
            for digit in fractional_digits {
                output.push(' ');
                output.push_str(match digit {
                    '0' => "zero",
                    '1' => "one",
                    '2' => "two",
                    '3' => "three",
                    '4' => "four",
                    '5' => "five",
                    '6' => "six",
                    '7' => "seven",
                    '8' => "eight",
                    _ => "nine",
                });
            }
        }
        if index + 1 < characters.len()
            && matches!(
                (characters[index], characters[index + 1]),
                ('s', 't') | ('n', 'd') | ('r', 'd') | ('t', 'h')
            )
        {
            index += 2;
        }
        if matches!(characters.get(index), Some('$' | '£' | '€')) {
            index += 1;
        }
    }
    output
}

pub fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn basic_cleaners(text: &str) -> String {
    collapse_whitespace(&text.to_lowercase())
}

pub fn multilingual_cleaners(text: &str, language: &str) -> String {
    let mut normalized = text.replace('"', "");
    if language == "tr" {
        normalized = normalized
            .replace('İ', "i")
            .replace('Ö', "ö")
            .replace('Ü', "ü");
    }
    normalized = normalized.to_lowercase();
    normalized = expand_numbers_multilingual(&normalized, language);
    normalized = expand_abbreviations_multilingual(&normalized, language);
    normalized = expand_symbols_multilingual(&normalized, language);
    collapse_whitespace(&normalized)
}

pub fn japanese_to_romaji(text: &str) -> String {
    let characters = text.chars().collect::<Vec<_>>();
    let mut output = String::with_capacity(text.len());
    let mut index = 0;
    while index < characters.len() {
        if matches!(characters[index], 'っ' | 'ッ') {
            if let Some(next) = characters
                .get(index + 1)
                .and_then(|value| kana_romaji(*value))
            {
                if let Some(consonant) = next
                    .chars()
                    .next()
                    .filter(|value| !"aiueon".contains(*value))
                {
                    output.push(consonant);
                }
            }
            index += 1;
            continue;
        }
        if let Some(next) = characters.get(index + 1) {
            if let Some(pair) = kana_pair_romaji(characters[index], *next) {
                output.push_str(pair);
                index += 2;
                continue;
            }
        }
        if let Some(romanized) = kana_romaji(characters[index]) {
            output.push_str(romanized);
        } else {
            output.push(characters[index]);
        }
        index += 1;
    }
    output
}

fn kana_pair_romaji(first: char, second: char) -> Option<&'static str> {
    Some(match (first, second) {
        ('キ', 'ャ') | ('き', 'ゃ') => "kya",
        ('キ', 'ュ') | ('き', 'ゅ') => "kyu",
        ('キ', 'ョ') | ('き', 'ょ') => "kyo",
        ('シ', 'ャ') | ('し', 'ゃ') => "sha",
        ('シ', 'ュ') | ('し', 'ゅ') => "shu",
        ('シ', 'ョ') | ('し', 'ょ') => "sho",
        ('チ', 'ャ') | ('ち', 'ゃ') => "cha",
        ('チ', 'ュ') | ('ち', 'ゅ') => "chu",
        ('チ', 'ョ') | ('ち', 'ょ') => "cho",
        ('ニ', 'ャ') | ('に', 'ゃ') => "nya",
        ('ニ', 'ュ') | ('に', 'ゅ') => "nyu",
        ('ニ', 'ョ') | ('に', 'ょ') => "nyo",
        ('ヒ', 'ャ') | ('ひ', 'ゃ') => "hya",
        ('ヒ', 'ュ') | ('ひ', 'ゅ') => "hyu",
        ('ヒ', 'ョ') | ('ひ', 'ょ') => "hyo",
        ('ミ', 'ャ') | ('み', 'ゃ') => "mya",
        ('ミ', 'ュ') | ('み', 'ゅ') => "myu",
        ('ミ', 'ョ') | ('み', 'ょ') => "myo",
        ('リ', 'ャ') | ('り', 'ゃ') => "rya",
        ('リ', 'ュ') | ('り', 'ゅ') => "ryu",
        ('リ', 'ョ') | ('り', 'ょ') => "ryo",
        ('ギ', 'ャ') | ('ぎ', 'ゃ') => "gya",
        ('ギ', 'ュ') | ('ぎ', 'ゅ') => "gyu",
        ('ギ', 'ョ') | ('ぎ', 'ょ') => "gyo",
        ('ジ', 'ャ') | ('じ', 'ゃ') => "ja",
        ('ジ', 'ュ') | ('じ', 'ゅ') => "ju",
        ('ジ', 'ョ') | ('じ', 'ょ') => "jo",
        ('ビ', 'ャ') | ('び', 'ゃ') => "bya",
        ('ビ', 'ュ') | ('び', 'ゅ') => "byu",
        ('ビ', 'ョ') | ('び', 'ょ') => "byo",
        ('ピ', 'ャ') | ('ぴ', 'ゃ') => "pya",
        ('ピ', 'ュ') | ('ぴ', 'ゅ') => "pyu",
        ('ピ', 'ョ') | ('ぴ', 'ょ') => "pyo",
        ('フ', 'ァ') | ('ふ', 'ぁ') => "fa",
        ('フ', 'ィ') | ('ふ', 'ぃ') => "fi",
        ('フ', 'ェ') | ('ふ', 'ぇ') => "fe",
        ('フ', 'ォ') | ('ふ', 'ぉ') => "fo",
        ('ウ', 'ィ') | ('う', 'ぃ') => "wi",
        ('ウ', 'ェ') | ('う', 'ぇ') => "we",
        ('ウ', 'ォ') | ('う', 'ぉ') => "wo",
        _ => return None,
    })
}

fn kana_romaji(character: char) -> Option<&'static str> {
    Some(match character {
        'ア' | 'あ' => "a",
        'イ' | 'い' => "i",
        'ウ' | 'う' => "u",
        'エ' | 'え' => "e",
        'オ' | 'お' => "o",
        'カ' | 'か' => "ka",
        'キ' | 'き' => "ki",
        'ク' | 'く' => "ku",
        'ケ' | 'け' => "ke",
        'コ' | 'こ' => "ko",
        'サ' | 'さ' => "sa",
        'シ' | 'し' => "shi",
        'ス' | 'す' => "su",
        'セ' | 'せ' => "se",
        'ソ' | 'そ' => "so",
        'タ' | 'た' => "ta",
        'チ' | 'ち' => "chi",
        'ツ' | 'つ' => "tsu",
        'テ' | 'て' => "te",
        'ト' | 'と' => "to",
        'ナ' | 'な' => "na",
        'ニ' | 'に' => "ni",
        'ヌ' | 'ぬ' => "nu",
        'ネ' | 'ね' => "ne",
        'ノ' | 'の' => "no",
        'ハ' | 'は' => "ha",
        'ヒ' | 'ひ' => "hi",
        'フ' | 'ふ' => "fu",
        'ヘ' | 'へ' => "he",
        'ホ' | 'ほ' => "ho",
        'マ' | 'ま' => "ma",
        'ミ' | 'み' => "mi",
        'ム' | 'む' => "mu",
        'メ' | 'め' => "me",
        'モ' | 'も' => "mo",
        'ヤ' | 'や' => "ya",
        'ユ' | 'ゆ' => "yu",
        'ヨ' | 'よ' => "yo",
        'ラ' | 'ら' => "ra",
        'リ' | 'り' => "ri",
        'ル' | 'る' => "ru",
        'レ' | 'れ' => "re",
        'ロ' | 'ろ' => "ro",
        'ワ' | 'わ' => "wa",
        'ヲ' | 'を' => "wo",
        'ン' | 'ん' => "n",
        'ガ' | 'が' => "ga",
        'ギ' | 'ぎ' => "gi",
        'グ' | 'ぐ' => "gu",
        'ゲ' | 'げ' => "ge",
        'ゴ' | 'ご' => "go",
        'ザ' | 'ざ' => "za",
        'ジ' | 'じ' => "ji",
        'ズ' | 'ず' => "zu",
        'ゼ' | 'ぜ' => "ze",
        'ゾ' | 'ぞ' => "zo",
        'ダ' | 'だ' => "da",
        'ヂ' | 'ぢ' => "ji",
        'ヅ' | 'づ' => "zu",
        'デ' | 'で' => "de",
        'ド' | 'ど' => "do",
        'バ' | 'ば' => "ba",
        'ビ' | 'び' => "bi",
        'ブ' | 'ぶ' => "bu",
        'ベ' | 'べ' => "be",
        'ボ' | 'ぼ' => "bo",
        'パ' | 'ぱ' => "pa",
        'ピ' | 'ぴ' => "pi",
        'プ' | 'ぷ' => "pu",
        'ペ' | 'ぺ' => "pe",
        'ポ' | 'ぽ' => "po",
        'ャ' | 'ゃ' => "ya",
        'ュ' | 'ゅ' => "yu",
        'ョ' | 'ょ' => "yo",
        'ヴ' | 'ゔ' => "vu",
        '　' => " ",
        '、' => ", ",
        '。' => ". ",
        _ => return None,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct QuotedPromptPart {
    pub text: String,
    pub quoted: bool,
}

pub fn split_quotation(prompt: &str) -> Vec<QuotedPromptPart> {
    let characters = prompt.char_indices().collect::<Vec<_>>();
    let mut result = Vec::new();
    let mut cursor = 0;
    let mut index = 0;
    while index < characters.len() {
        let (byte, character) = characters[index];
        let closing = match character {
            '"' => Some('"'),
            '‘' => Some('’'),
            '“' => Some('”'),
            '\'' => {
                let internal = index > 0
                    && index + 1 < characters.len()
                    && characters[index - 1].1.is_ascii_alphabetic()
                    && characters[index + 1].1.is_ascii_alphabetic();
                (!internal).then_some('\'')
            }
            _ => None,
        };
        let Some(closing) = closing else {
            index += 1;
            continue;
        };
        if byte > cursor {
            result.push(QuotedPromptPart {
                text: prompt[cursor..byte].to_owned(),
                quoted: false,
            });
        }
        let mut close_index = index + 1;
        while close_index < characters.len() && characters[close_index].1 != closing {
            close_index += 1;
        }
        if close_index == characters.len() {
            if byte < prompt.len() {
                result.push(QuotedPromptPart {
                    text: prompt[byte..].to_owned(),
                    quoted: false,
                });
            }
            cursor = prompt.len();
            break;
        }
        let end = characters
            .get(close_index + 1)
            .map_or(prompt.len(), |(offset, _)| *offset);
        result.push(QuotedPromptPart {
            text: prompt[byte..end].to_owned(),
            quoted: true,
        });
        cursor = end;
        index = close_index + 1;
    }
    if cursor < prompt.len() {
        result.push(QuotedPromptPart {
            text: prompt[cursor..].to_owned(),
            quoted: false,
        });
    }
    result
}

#[derive(Clone, Copy, Debug)]
pub struct AudioSamplingOptions {
    pub token_offset: u32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub temperature: f32,
}

impl Default for AudioSamplingOptions {
    fn default() -> Self {
        Self {
            token_offset: 0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            temperature: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct AudioCandidate {
    index: usize,
    weight: f32,
}

pub fn sample_audio_token(
    backend: &CpuBackend,
    logits: &[f32],
    options: AudioSamplingOptions,
    transaction: &mut RngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<u32, CompositeTextEncoderError> {
    context.cancellation.check()?;
    transaction.require_device(DeviceId::CPU)?;
    if logits.is_empty()
        || !options.temperature.is_finite()
        || options.temperature <= 0.0
        || !options.top_p.is_finite()
        || !(0.0 < options.top_p && options.top_p <= 1.0)
        || !options.min_p.is_finite()
        || !(0.0..=1.0).contains(&options.min_p)
        || logits.iter().all(|value| !value.is_finite())
    {
        return Err(CompositeTextEncoderError::InvalidInput(
            "audio sampling parameters or logits are invalid",
        ));
    }
    let mut candidates = backend.workspace_vec::<AudioCandidate>(context, logits.len())?;
    for (index, logit) in logits.iter().copied().enumerate() {
        if logit.is_finite() {
            candidates.try_push(AudioCandidate {
                index,
                weight: logit,
            })?;
        }
    }
    candidates.sort_by(|left, right| right.weight.total_cmp(&left.weight));
    if options.top_k > 0 && candidates.len() > options.top_k {
        let keep = options.top_k;
        for candidate in &mut candidates[keep..] {
            candidate.weight = 0.0;
        }
    }
    let maximum = candidates[0].weight;
    let mut total = 0.0_f32;
    let mut maximum_weight = 0.0_f32;
    for candidate in &mut *candidates {
        if candidate.weight == 0.0 && options.top_k > 0 {
            continue;
        }
        candidate.weight = ((candidate.weight - maximum) / options.temperature).exp();
        maximum_weight = maximum_weight.max(candidate.weight);
        total += candidate.weight;
    }
    let minimum = maximum_weight * options.min_p;
    let mut retained_total = 0.0_f32;
    for candidate in &mut *candidates {
        if candidate.weight < minimum {
            candidate.weight = 0.0;
            continue;
        }
        if retained_total > 0.0 && retained_total / total >= options.top_p {
            candidate.weight = 0.0;
            continue;
        }
        retained_total += candidate.weight;
    }
    if !retained_total.is_finite() || retained_total <= 0.0 {
        return Err(CompositeTextEncoderError::InvalidInput(
            "audio sampling retained no finite probability mass",
        ));
    }
    let mut working = transaction.clone();
    let threshold = working.next_unit_f32(context.cancellation)? * retained_total;
    let mut cumulative = 0.0_f32;
    let mut selected = candidates[0].index;
    for candidate in &*candidates {
        cumulative += candidate.weight;
        if threshold <= cumulative {
            selected = candidate.index;
            break;
        }
    }
    let selected = u32::try_from(selected)
        .ok()
        .and_then(|index| options.token_offset.checked_add(index))
        .ok_or(CompositeTextEncoderError::Overflow(
            "audio token identifier",
        ))?;
    context.cancellation.check()?;
    *transaction = working;
    Ok(selected)
}

pub fn generate_audio_codes(
    backend: &CpuBackend,
    logits_by_step: &[&[f32]],
    options: AudioSamplingOptions,
    transaction: &mut RngTransaction,
    context: &ExecutionContext<'_>,
) -> Result<Vec<u32>, CompositeTextEncoderError> {
    context.cancellation.check()?;
    let mut working = transaction.clone();
    let mut output = Vec::new();
    output
        .try_reserve_exact(logits_by_step.len())
        .map_err(|_| CompositeTextEncoderError::Overflow("audio code publication"))?;
    for logits in logits_by_step {
        output.push(sample_audio_token(
            backend,
            logits,
            options,
            &mut working,
            context,
        )?);
    }
    context.cancellation.check()?;
    *transaction = working;
    Ok(output)
}

#[cfg(test)]
mod source_clip_composition_tests {
    use super::*;
    use sim_tensor::generated_native_diffusion::tensor_to_f32;
    use sim_tensor::{CancellationToken, CpuWorkspaceAuthority, StreamId};

    fn tensor(
        backend: &CpuBackend,
        context: &ExecutionContext<'_>,
        shape: &[u64],
        value: f32,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let elements = shape.iter().try_fold(1_usize, |total, dimension| {
            total.checked_mul(usize::try_from(*dimension).ok()?)
        });
        let elements = elements.ok_or("test tensor shape overflowed")?;
        Ok(tensor_from_f32(
            backend,
            shape,
            &vec![value; elements],
            context,
        )?)
    }

    #[test]
    fn source_sdxl_truncates_tokens_and_concatenates_l_before_g()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(8 * 1024 * 1024)?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(8 * 1024 * 1024)?,
            &cancellation,
        );
        let clip_l = tensor(&backend, &context, &[1, 3, 2], 1.0)?;
        let clip_g = tensor(&backend, &context, &[1, 2, 3], 2.0)?;
        let pooled = tensor(&backend, &context, &[1, 3], 4.0)?;
        let output = compose_source_sdxl(&backend, &clip_l, &clip_g, &pooled, &context)?;
        assert_eq!(output.hidden.descriptor().shape(), &[1, 2, 5]);
        let hidden = tensor_to_f32(&backend, &output.hidden, &context)?;
        assert_eq!(
            &*hidden,
            &[1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        );
        assert_eq!(output.pooled.descriptor().shape(), &[1, 3]);
        Ok(())
    }

    #[test]
    fn source_sd3_preserves_missing_role_offsets_and_zero_fallbacks()
    -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(16 * 1024 * 1024)?,
            &cancellation,
        );
        let clip_g = tensor(&backend, &context, &[1, 2, 1_280], 2.0)?;
        let pooled_g = tensor(&backend, &context, &[1, 1_280], 3.0)?;
        let g_only =
            compose_source_sd3(&backend, None, Some((&clip_g, &pooled_g)), None, &context)?;
        assert_eq!(g_only.hidden.descriptor().shape(), &[1, 2, 4_096]);
        let hidden = tensor_to_f32(&backend, &g_only.hidden, &context)?;
        assert!(hidden[..768].iter().all(|value| *value == 0.0));
        assert!(hidden[768..2_048].iter().all(|value| *value == 2.0));
        assert_eq!(g_only.pooled.descriptor().shape(), &[1, 2_048]);

        let t5 = tensor(&backend, &context, &[1, 4, 4_096], 5.0)?;
        let t5_only = compose_source_sd3(&backend, None, None, Some(&t5), &context)?;
        assert_eq!(t5_only.hidden.descriptor().shape(), &[1, 4, 4_096]);
        assert!(
            tensor_to_f32(&backend, &t5_only.pooled, &context)?
                .iter()
                .all(|value| *value == 0.0)
        );

        let empty = compose_source_sd3(&backend, None, None, None, &context)?;
        assert_eq!(empty.hidden.descriptor().shape(), &[1, 77, 4_096]);
        assert_eq!(empty.pooled.descriptor().shape(), &[1, 2_048]);
        Ok(())
    }

    #[test]
    fn source_hidream_trims_the_pre_layer_capture_axis() -> Result<(), Box<dyn std::error::Error>> {
        let cancellation = CancellationToken::default();
        let (backend, authority) = CpuWorkspaceAuthority::create_backend(16 * 1024 * 1024)?;
        let context = backend.execution_context(
            StreamId::DEFAULT,
            authority.authorize_workspace(16 * 1024 * 1024)?,
            &cancellation,
        );
        let llama = tensor(&backend, &context, &[1, 3, 1, 4_096], 7.0)?;
        let output = compose_source_hidream(&backend, None, None, None, Some(&llama), &context)?;
        assert_eq!(output.hidden.descriptor().shape(), &[1, 128, 4_096]);
        assert_eq!(output.pooled.descriptor().shape(), &[1, 2_048]);
        assert_eq!(
            output
                .conditioning_llama3
                .as_ref()
                .ok_or("missing Llama conditioning")?
                .descriptor()
                .shape(),
            &[1, 2, 1, 4_096]
        );
        Ok(())
    }
}
