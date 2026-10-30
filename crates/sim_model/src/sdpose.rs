use crate::{
    MappedModelWeights, NativeModule,
    attention::{
        AttentionBackend, AttentionError, AttentionFallbackPolicy, AttentionRequest,
        scaled_dot_product_attention_tensor_with_context,
    },
    generated_lotusd_sim_model_0106,
    native_ops::{
        GeluApproximation, NativeOpsError, tensor_from_f32 as tensor_from_values,
        tensor_to_f32 as tensor_to_values,
    },
};
use sha2::{Digest, Sha256};
use sim_media::{NativePoseKeypoint, NativePosePerson};
use sim_tensor::{
    CpuBackend, DType, DecodedScalar, DeviceId, ExecutionContext, StorageId, StreamId, Tensor,
    TensorError,
    generated_native_diffusion::NativeDiffusionTensorError,
    generated_shape_layout_transform_02::{
        ShapeLayoutTransformPartTwoError, tensor_reshape_with_context_exact_native,
    },
    generated_sim_operator_indirection_01::{ConvolutionGeometry, OperatorIndirectionError},
};
use sim_types::{CancellationError, CancellationToken};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use thiserror::Error;

pub const SDPOSE_HEATMAP_CHANNELS: usize = 133;
pub const SDPOSE_HEATMAP_HEIGHT: usize = 256;
pub const SDPOSE_HEATMAP_WIDTH: usize = 192;
pub const SDPOSE_INPUT_HEIGHT: f32 = 1024.0;
pub const SDPOSE_INPUT_WIDTH: f32 = 768.0;
pub const LOTUS_CONDITIONING_SOURCE_SHA256: &str =
    "a1db3c22c71719b11d470655ddd408daa06b8e13f24e25556f77ce0ba825ee08";
pub const LOTUS_CONDITIONING_F32_SHA256: &str =
    "a7ba15b03c89931a7764a7d4b398201b322e00524d03efaac172b48dc637397d";
pub const LOTUS_CONDITIONING_F16_SHA256: &str =
    "168b6113df66f4f1006e36bcd143aa46215a28ff60d0e769cc8a412c16c0b951";

const GAUSSIAN_RADIUS: isize = 5;
const GAUSSIAN_SIGMA: f32 = 2.0;
const OPENPOSE_KEYPOINTS: usize = 134;
const MMPOSE_INDICES: [usize; 15] = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3];
const OPENPOSE_INDICES: [usize; 15] = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17];

const LOTUS_CONDITIONING_F16_BITS: [u16; 2048] = [
    0xb504, 0xb729, 0xa038, 0xb34c, 0x31b7, 0xb37f, 0xb301, 0xa090, 0xb506, 0x324a, 0xab27, 0xb620,
    0xae2e, 0x1f84, 0xb48d, 0x2cc8, 0xad84, 0xb2c2, 0x2bac, 0x2ed8, 0x127f, 0xba3f, 0xb61f, 0xb1ed,
    0xb080, 0xadfe, 0xafde, 0xb0de, 0xb286, 0xb385, 0xb3cd, 0xaef3, 0xadc1, 0x2db1, 0xb3a7, 0xb128,
    0xb6d9, 0xb047, 0xb46e, 0xb1b3, 0xb62c, 0x3408, 0xab2b, 0xac90, 0x9558, 0x2849, 0xb43a, 0x2c97,
    0xa0a6, 0xb27b, 0xb46c, 0xae51, 0xb752, 0xb364, 0x1ca3, 0xb5e8, 0xb5df, 0xb292, 0xb0d1, 0xb73a,
    0xb297, 0x2451, 0xb4e3, 0xb0d2, 0x26e0, 0xb8e0, 0xa4de, 0x2a75, 0xa052, 0xb4f6, 0xb1e6, 0xb0dc,
    0x2c4f, 0x30d0, 0xaefd, 0xada8, 0xb80b, 0xb63d, 0xa789, 0xaeb7, 0xb1fe, 0xb6ea, 0xb540, 0xbc3e,
    0x2214, 0x2a0e, 0xae18, 0xacc5, 0xb4b0, 0xaefc, 0xb401, 0xb620, 0xb180, 0xa85a, 0xb0f4, 0xb1f5,
    0xb3ad, 0x2ebe, 0xb324, 0xb22f, 0xb204, 0xb6e3, 0xaade, 0xb798, 0xb42f, 0xaee6, 0xb707, 0x2dbd,
    0xb1ee, 0xb569, 0xb4c1, 0xb4c7, 0xb19e, 0xb5f9, 0xb079, 0xb42b, 0xb623, 0xae63, 0xb51d, 0xac68,
    0xb6fc, 0xb1e2, 0xb64b, 0x98aa, 0x2f31, 0xb42b, 0xb32d, 0xb234, 0xb817, 0x281e, 0xaa25, 0xb477,
    0xb4c4, 0xb417, 0x2f64, 0xc417, 0xb157, 0xb473, 0xb892, 0xb06f, 0xb3d2, 0xb6a2, 0xb3d0, 0xac56,
    0xb086, 0xaf6b, 0xb09b, 0xa4bf, 0xc0fb, 0xb0f5, 0xb361, 0xb620, 0xaf82, 0xad49, 0xb181, 0xb535,
    0xb4af, 0xad38, 0xb699, 0xb4fc, 0xa043, 0x1d02, 0xb328, 0xb0d6, 0xb751, 0xb052, 0xb13a, 0xad3e,
    0x2c48, 0x255b, 0xaf9f, 0xaea3, 0xb4cc, 0xb7a9, 0xad15, 0xa3c5, 0xb4d4, 0xb581, 0xb423, 0xb72a,
    0xb649, 0xa68b, 0xb4cc, 0x2eef, 0xb459, 0xb5ca, 0xb418, 0xb295, 0xab19, 0x2bbf, 0xb752, 0xb36f,
    0xb18f, 0xb67e, 0xb351, 0xaacb, 0xa743, 0xb213, 0x82d0, 0xb946, 0xb17f, 0xb7ad, 0xb07b, 0xb686,
    0x30f0, 0x1a5a, 0x1ec6, 0xb2a3, 0xb7e5, 0x4058, 0xb742, 0xb48b, 0xb4d5, 0xb406, 0xb2a2, 0xb0e1,
    0xb9d7, 0xad40, 0x2acf, 0xb44c, 0x31df, 0xb453, 0xaddc, 0xad53, 0xb027, 0xb743, 0x2c66, 0x26dd,
    0xb14c, 0xb6bc, 0xb0f8, 0xab58, 0xb4cf, 0xa9d0, 0xb7c7, 0xb30f, 0xb63e, 0x1fab, 0x9d75, 0xae3f,
    0xb0a3, 0xb304, 0xb6df, 0xb66a, 0xb234, 0xac20, 0xa7d7, 0xa9dd, 0xab45, 0x2fa4, 0xb2a3, 0xb5b3,
    0xb162, 0xb4ab, 0xb4a3, 0xb292, 0xb6b7, 0xb7a7, 0xb285, 0xb5e5, 0xb1f6, 0xb6d8, 0xb19f, 0x2ccd,
    0xb3dd, 0xa8ba, 0x1c18, 0xc5c2, 0xa163, 0xb361, 0xb335, 0xb252, 0xb103, 0xb157, 0xb621, 0xaaee,
    0x2fcd, 0xb5eb, 0xa4b9, 0xb0ce, 0xb817, 0xb75a, 0xb0ff, 0x2bbd, 0x2991, 0xb063, 0xac6c, 0xb493,
    0xb71e, 0xb399, 0xb303, 0xb6f5, 0xa89c, 0x3028, 0x2593, 0xb819, 0xb054, 0xae01, 0xb46d, 0xb180,
    0xb17c, 0xb71c, 0xb2f9, 0xb451, 0xb62e, 0xb50e, 0xb6c6, 0xb47b, 0xb417, 0xb174, 0x2c03, 0xb203,
    0x9ef0, 0xb38c, 0xada1, 0xb325, 0xb181, 0xb7bf, 0xb543, 0x29fa, 0xa960, 0xaf1d, 0x25ec, 0x9ac9,
    0xb8ce, 0x2dcb, 0xb208, 0xb435, 0xb603, 0xb834, 0xaf7e, 0xb60c, 0xb3c6, 0xb6b8, 0x2c42, 0xb57d,
    0x2c43, 0xb58b, 0xb05d, 0xb267, 0xb624, 0x2736, 0x299d, 0xa461, 0xb7a3, 0xaee0, 0xb022, 0xb51d,
    0xb526, 0xb7ec, 0xa16c, 0xb1b1, 0xb670, 0xad30, 0x283e, 0xb064, 0xb50e, 0xaa2d, 0xb465, 0xac6a,
    0xa6f2, 0xae1a, 0x30b2, 0xb630, 0xa680, 0xb6c6, 0xb6ae, 0xb613, 0xb255, 0xb844, 0xb55e, 0x2c72,
    0xb52b, 0xb201, 0x20d6, 0x31e3, 0xb411, 0xb4d0, 0xade8, 0xadaa, 0x2feb, 0x30cd, 0xafc7, 0xb4c6,
    0x2906, 0xb0b5, 0xb127, 0xb28f, 0xa460, 0xb3e8, 0xb0df, 0xb422, 0x2ef5, 0xb4d7, 0xb39b, 0xb5c3,
    0xb718, 0xad06, 0xb917, 0xae46, 0xb551, 0xb199, 0xb4f0, 0xb24d, 0xb401, 0xb727, 0xb1e4, 0x9f13,
    0xb28b, 0xb414, 0xb5e8, 0xaf40, 0xaf10, 0xb472, 0xb535, 0x2313, 0x28b5, 0xb0ab, 0xb58a, 0xb36e,
    0xab34, 0xb72c, 0xb300, 0xad7a, 0xb46b, 0xb0f1, 0xb326, 0xb427, 0x2d8a, 0x9bc9, 0xb571, 0xace0,
    0xb40e, 0x922f, 0xabbd, 0xb5a3, 0x2893, 0xb2fb, 0xb36b, 0xb177, 0x24c6, 0x2dfc, 0xb24e, 0x2849,
    0xa883, 0x29a4, 0xb295, 0xa93d, 0xab82, 0x2b48, 0xada3, 0xaef1, 0x2e1f, 0xaef8, 0xb551, 0xaf2a,
    0xaf64, 0x4a83, 0x1a27, 0xb84b, 0xb1f1, 0xac5c, 0xb09a, 0xb1b2, 0xb53a, 0xb5a8, 0xb101, 0xb520,
    0x2066, 0xaeae, 0xb5d9, 0xb4db, 0xb57e, 0xb693, 0xa8da, 0xaf4f, 0xb4b4, 0x2cc4, 0xb2fc, 0x2637,
    0xb33e, 0xb5a8, 0xb1b0, 0xb16a, 0xb3c1, 0xafc1, 0xad05, 0xb575, 0xb2c7, 0xb80c, 0xb85c, 0xb64c,
    0xaee5, 0xb293, 0xb05e, 0xb1c0, 0xb657, 0xb24f, 0xb979, 0xa82f, 0xa8f2, 0xac8a, 0xae70, 0xb0fe,
    0xb624, 0xb51d, 0x2f1b, 0xb64f, 0x2d1e, 0xb807, 0xb108, 0x1c3b, 0xafdd, 0xb383, 0xb4aa, 0xae9b,
    0xb495, 0x24d0, 0xb0de, 0xb523, 0x24e8, 0xaf48, 0xb861, 0x3114, 0xb41f, 0xb3f8, 0xb4fc, 0xae40,
    0xb59d, 0xb5d2, 0xacf0, 0xb35b, 0xb4e6, 0xaaad, 0xafbd, 0xb672, 0xb147, 0x2549, 0xb501, 0xae99,
    0xb807, 0xb19d, 0xb269, 0xb18e, 0xb28e, 0xb201, 0xafcb, 0xabf0, 0xa876, 0xb26c, 0xb2e0, 0xb3e2,
    0x9ebf, 0xae31, 0xb774, 0xb664, 0xb313, 0xae23, 0xb00a, 0xade7, 0xb0aa, 0xb2f6, 0xb4a1, 0xb486,
    0xb1b8, 0xb16a, 0xb5c0, 0xb2fc, 0xa99b, 0xb292, 0xb4a4, 0xab18, 0xb0c5, 0xb573, 0xa9e1, 0xb4e2,
    0xb817, 0xaeb2, 0xb599, 0xaef7, 0xb15c, 0xb447, 0xafd4, 0xae4e, 0xb4ec, 0xb1a7, 0xb407, 0x3249,
    0xb31b, 0xb385, 0xa89b, 0xb199, 0xb396, 0xb7af, 0xb3b0, 0xae1b, 0xb3f1, 0xb63b, 0xb198, 0xb5ef,
    0xa174, 0xb0ba, 0x9b43, 0xb536, 0xb4ca, 0xb077, 0xb482, 0xae49, 0xa483, 0xab96, 0xacd6, 0xb197,
    0xb488, 0xb12d, 0xb70c, 0x29eb, 0xb1e2, 0xaaa9, 0xb3e4, 0xaf2f, 0xae74, 0xb136, 0xb4c2, 0xb658,
    0x2f8d, 0x9ea7, 0xb536, 0xb591, 0xa9e2, 0xae2b, 0xb585, 0xa1f9, 0xb58b, 0xb494, 0xb1e8, 0xb46d,
    0xa2b8, 0xb653, 0xb813, 0xaf22, 0xb1f5, 0xb2e8, 0xb1cd, 0x2b18, 0xb4b5, 0x2c29, 0xb09f, 0xac13,
    0xb1e4, 0xa0b2, 0xb140, 0xb1f1, 0x2dc4, 0x264a, 0xa4ff, 0xb870, 0xb4d1, 0xb703, 0xac9c, 0x27d0,
    0xb0c5, 0x2a42, 0xb394, 0xb0f9, 0xa4a2, 0xb35a, 0xb4ac, 0xb5b1, 0xb491, 0xb19f, 0xad59, 0xb6a9,
    0x25ce, 0xb6ad, 0x3691, 0xb3fe, 0x2400, 0xa84d, 0xa929, 0xb823, 0xaaa4, 0xada9, 0xb279, 0xb5fc,
    0xb493, 0xb1a2, 0xb4af, 0xb5e8, 0xad5b, 0xb5b5, 0xb62e, 0x2c31, 0x3158, 0xb2b8, 0x9f9d, 0x3020,
    0x9c2c, 0xb2f5, 0xa66e, 0xb391, 0xb16d, 0xaa04, 0xa906, 0xb5ca, 0xaa15, 0xb50c, 0xb469, 0xb41a,
    0xabed, 0xafc9, 0xb792, 0xaee7, 0xb9fb, 0xb03c, 0xb530, 0xaa66, 0xb495, 0x2e9f, 0xb179, 0xb639,
    0x3014, 0xb560, 0xb4d7, 0xb3a2, 0xb0a8, 0xb114, 0xb3c5, 0xa88c, 0xafa5, 0xb502, 0x2f55, 0xb49b,
    0xb85a, 0xb4ec, 0xad88, 0xb3b5, 0xb489, 0xb6db, 0xad7c, 0xb163, 0x3415, 0x2ce3, 0xb62e, 0xb27b,
    0xb043, 0xaeed, 0xb7fe, 0x2dd8, 0xb24c, 0xa4e0, 0xb0d6, 0xb583, 0xb511, 0xa8fe, 0x3079, 0x3045,
    0xa8c3, 0xa4c4, 0xb053, 0xac81, 0xb617, 0xb572, 0xac6b, 0x330a, 0x22ca, 0xb114, 0xb91a, 0x3116,
    0xb83a, 0xb36d, 0xb1f2, 0xb2d3, 0xb6fd, 0xb6e5, 0xb1a5, 0x9249, 0xb6c9, 0xac4d, 0xb591, 0xb0a0,
    0xb4e4, 0xb5d6, 0xb5c2, 0xb766, 0xb723, 0xb219, 0xaf22, 0xb6de, 0xac1e, 0xb5dc, 0xb6e0, 0xaf05,
    0xb4cc, 0xb19f, 0xb67c, 0xad62, 0xb409, 0xaa11, 0x1c72, 0xae6d, 0xb458, 0xb1ee, 0xb21d, 0x2e91,
    0x2323, 0xa84a, 0xb56e, 0xb545, 0x2b5c, 0xb20c, 0xb1ac, 0xb6de, 0xb446, 0xb6b4, 0xb0c6, 0xb440,
    0xb25e, 0xb183, 0x3101, 0xb2b4, 0xb449, 0xac11, 0xb5d9, 0xb571, 0xb4f4, 0xb51f, 0xada2, 0xb5a9,
    0x90c3, 0xb12d, 0xb1f9, 0xb272, 0xb1ce, 0xb385, 0xb628, 0xb832, 0x2249, 0xb2d4, 0xb4ee, 0xb1a0,
    0xb4dc, 0xb83d, 0xb0ff, 0xb4bd, 0x2b57, 0xb5e9, 0xaee4, 0xb56b, 0xb298, 0xa579, 0xb651, 0xb306,
    0xb006, 0xa91e, 0xb6ca, 0xb04e, 0xb3bd, 0xaeb9, 0x30f8, 0x2f36, 0xb091, 0xb046, 0xaf1c, 0xb603,
    0xb474, 0xa8b7, 0xaf13, 0x33e9, 0xb40c, 0xb885, 0xb5ff, 0xb539, 0x32d7, 0xae0d, 0xb565, 0xb560,
    0xb580, 0xae66, 0xb638, 0x214a, 0x30cc, 0xb288, 0xb652, 0xb63e, 0xb007, 0x30a7, 0xb403, 0xac4f,
    0xaecf, 0xa825, 0xb5be, 0xae54, 0xb186, 0x2a90, 0xb243, 0xb1e2, 0xb5ee, 0x30fb, 0xb580, 0xb496,
    0xac78, 0xb4d7, 0xb47f, 0xb140, 0xb83e, 0xb061, 0xb00f, 0xb5a6, 0xb0b7, 0xb74a, 0xb5c2, 0xb461,
    0xb333, 0xb20f, 0x2dfd, 0xb3f2, 0xb256, 0xb5a4, 0xb4d2, 0xb34e, 0xb3c0, 0xadce, 0x334a, 0xb4c0,
    0xa104, 0xb4b4, 0xae8a, 0xb2e6, 0xb449, 0xaabd, 0x3282, 0xb089, 0x31ec, 0x2fe7, 0xae3a, 0xb853,
    0xb6e3, 0xadce, 0xb856, 0xb354, 0xac00, 0xafc7, 0xb7ef, 0xb29e, 0xa69e, 0x28ff, 0xb743, 0xb518,
    0xb6d1, 0xb5ec, 0xb2f1, 0x2d6a, 0xa9d8, 0x2f19, 0x151a, 0x329d, 0xa9ef, 0xb37b, 0xb4bb, 0x2b69,
    0xb13f, 0xaadc, 0xb42c, 0xb0c3, 0xb120, 0xb859, 0xb361, 0xb867, 0xb1ba, 0xb31e, 0xac8e, 0xafdf,
    0xafa0, 0xb40f, 0xb570, 0xaf3f, 0xaedd, 0xb290, 0xb5c5, 0x2d74, 0xb409, 0xb5db, 0xb450, 0x3267,
    0xb80d, 0x2b4e, 0xb4fc, 0x2638, 0xb168, 0xb3af, 0xb604, 0xb2cd, 0x2ce4, 0xb7d3, 0xaf85, 0xb879,
    0xb503, 0xb894, 0xb071, 0xa8bf, 0xb62b, 0x2824, 0xb64e, 0xaa7e, 0xaead, 0xaefd, 0xb49a, 0xb5fe,
    0xa456, 0xb30b, 0xb4c0, 0xa757, 0xb2d2, 0x2aa8, 0xaab3, 0xb40f, 0x2ca7, 0xb3ff, 0xb0ce, 0xae5b,
    0xb57f, 0x3411, 0xa827, 0xb4bd, 0x3da5, 0x1fc2, 0xb6dd, 0x2dbc, 0x25d5, 0xa87c, 0xbc65, 0x3236,
    0x403b, 0xba09, 0x344f, 0xb9e1, 0xbc69, 0xaa6f, 0xb8aa, 0x26e1, 0xb8d4, 0xb8d2, 0xbcad, 0x3402,
    0xb6b4, 0xb86a, 0xb4da, 0x3a2e, 0x3b74, 0xb08d, 0x408e, 0xb887, 0x3c0e, 0xbbb6, 0xb958, 0xba80,
    0x3d7f, 0xbc3e, 0xbc1e, 0xbeb8, 0x3908, 0x37e6, 0x3960, 0x3bfb, 0xbc13, 0xaa05, 0xb5fd, 0x3389,
    0x3dfd, 0xbe45, 0x372f, 0x32a7, 0xbc3e, 0xb180, 0xb8f6, 0xb125, 0x3a0a, 0xb8b7, 0xb291, 0x2bc8,
    0x326a, 0x424f, 0xc3c2, 0x4012, 0x3c82, 0x3140, 0xbf61, 0x195e, 0x3a66, 0xb565, 0x378e, 0xb8ae,
    0x3bfa, 0x364c, 0x3ce4, 0xbe05, 0xb941, 0x3c97, 0xbeec, 0xbf7b, 0x3e1e, 0xb96e, 0xbdc8, 0x3e3a,
    0xb53d, 0x41f9, 0xb0ae, 0xc1ae, 0x34db, 0xb9bd, 0xae35, 0xb3c1, 0x396a, 0xc065, 0xbae7, 0xb84f,
    0xbce5, 0x3eae, 0xbcad, 0xb4a9, 0xb861, 0xa8a7, 0x3c2a, 0xbee8, 0xb8da, 0xb13b, 0x339f, 0x3935,
    0xbcae, 0x3faf, 0x3882, 0x360e, 0x3a64, 0xbc16, 0xb68c, 0x408d, 0xb4d8, 0xbf1d, 0x3c2f, 0x3dea,
    0x39ac, 0xbe2a, 0x90f3, 0x3cc1, 0x4015, 0x387c, 0xaf3e, 0x3c23, 0x2bd6, 0x415c, 0xb845, 0xbcff,
    0xb9de, 0xbc12, 0xbc08, 0x3d49, 0xb79e, 0xbdad, 0x3853, 0xba60, 0xb9d9, 0xbda6, 0xbb97, 0xc08c,
    0xbc7d, 0xbd89, 0xb812, 0xbc27, 0xbb7d, 0xb8ab, 0xbc48, 0xbc6c, 0xc136, 0x2bf7, 0x361e, 0xb695,
    0xb5e8, 0xbbde, 0xb9d6, 0xb1ce, 0xb78e, 0xb88d, 0xaf11, 0xc159, 0x3472, 0xbadc, 0x3b70, 0x3fd5,
    0x306f, 0xbfac, 0xbe1e, 0x384f, 0x3814, 0xbb3b, 0xae10, 0xc02b, 0x3b62, 0xb492, 0xba78, 0xadfc,
    0xb8d0, 0xbd06, 0x393a, 0x353f, 0xacdd, 0xb344, 0xb4d0, 0xb826, 0xbe6b, 0x3e5f, 0x3daa, 0xbb58,
    0xb46d, 0xba11, 0xbcad, 0xbef8, 0x3c38, 0xb43e, 0xa83e, 0x3473, 0xbd5a, 0x2b0b, 0x2ec0, 0xbc42,
    0x3c32, 0xbd9f, 0xbc7f, 0xb81d, 0xbc3c, 0xbee8, 0xbc88, 0x36ee, 0xc127, 0xc048, 0xba59, 0x3803,
    0x3fdc, 0x3be3, 0x3567, 0xba3d, 0x3ad7, 0xb937, 0xb7c0, 0x416b, 0x3877, 0xb0b6, 0xb86e, 0xba09,
    0xbef2, 0x3c81, 0xbc62, 0xb424, 0x41f5, 0x355d, 0x35b9, 0xadfd, 0x34bb, 0xb39c, 0x3e4d, 0x2ea8,
    0x3ecd, 0xb9d5, 0xbdde, 0xb271, 0x3914, 0xb0fb, 0xc006, 0x353a, 0x2a13, 0xb159, 0xba45, 0xbbf4,
    0x3a3e, 0xb093, 0xaf60, 0x3843, 0xbbb1, 0xac1c, 0x3d96, 0x3e9e, 0xbe68, 0xba31, 0x346b, 0xb6c2,
    0x3aea, 0x3844, 0x365b, 0xbcdc, 0x3648, 0xb804, 0xb36a, 0x3de6, 0x3eab, 0x3f77, 0xbd33, 0x332a,
    0xb81b, 0xb701, 0xbcac, 0xba2e, 0x3ebb, 0x38fb, 0x4061, 0xb8cc, 0xb93f, 0xbc67, 0xb73e, 0xb4ad,
    0x412a, 0xbd83, 0x3384, 0xbed6, 0xbda5, 0xb810, 0xbad2, 0xbb43, 0x3ae4, 0xc072, 0xbd86, 0x39d2,
    0x3ad9, 0xba3d, 0x4017, 0xb6e5, 0x3a4f, 0xbcdc, 0xadd8, 0x381e, 0xbfb4, 0xb078, 0x34a1, 0x38d7,
    0x3b83, 0xadf3, 0xb752, 0xbbe4, 0xb94d, 0x3dec, 0x3463, 0xb447, 0xbd35, 0xbd73, 0xbfee, 0xbae3,
    0xb69a, 0xb4b4, 0xbfa4, 0x3ebb, 0x36ff, 0x3c03, 0x3605, 0xa15c, 0xbaa1, 0xb81c, 0xc245, 0x3c15,
    0xbd0c, 0xbc0a, 0x3c59, 0xbba8, 0x2150, 0x3e92, 0x3a49, 0x3c1e, 0x37f3, 0x3d27, 0x3830, 0x2ad0,
    0xb29e, 0xba5b, 0xbcab, 0x2a57, 0xb47a, 0x2617, 0xb9ef, 0xbc46, 0x3ae3, 0x3cb7, 0x3b49, 0xb743,
    0xbf4a, 0xbb62, 0xb801, 0x35bf, 0x3a6c, 0xbd6d, 0xb529, 0xbc73, 0xbc28, 0xb837, 0xbd2a, 0xb57d,
    0xbcd8, 0xb21c, 0x404b, 0xa986, 0xb5ec, 0xc017, 0xb6e1, 0xb87d, 0xb150, 0xb774, 0xc096, 0xbc50,
    0xbc5f, 0xae9e, 0xb87d, 0xab5a, 0xbff2, 0xb81c, 0x39a1, 0xc0f8, 0xbd37, 0xbea9, 0x37b4, 0xbacb,
    0xc165, 0xb256, 0x2d90, 0xb186, 0xbd4e, 0xbcca, 0x3cf4, 0xb632, 0xacdc, 0x2694, 0xbcd9, 0x2dc7,
    0xc042, 0xbe90, 0xb0e1, 0x3398, 0x3d7c, 0x407f, 0x3dd9, 0x3504, 0x3805, 0xb06d, 0xbd15, 0x39e2,
    0x3943, 0xb6e7, 0xbd5e, 0x2e75, 0x32b0, 0xa11e, 0x3aa8, 0x3d42, 0xb6c3, 0xbe5c, 0xb8d6, 0x3394,
    0xbe6c, 0xb8e3, 0xbe3f, 0x3e84, 0xb962, 0xb975, 0xb82e, 0xbb52, 0xb781, 0x344b, 0x3499, 0xa68a,
    0x3719, 0x3e7c, 0xbc5c, 0xb872, 0x3c24, 0xb502, 0xa959, 0x342d, 0x358e, 0xc0a7, 0x334e, 0xbacf,
    0xc083, 0x3594, 0xb422, 0xb5e0, 0xba60, 0xbc76, 0x3f59, 0xb254, 0xbcf2, 0xa9bc, 0x3998, 0x3d84,
    0x3c74, 0x2df2, 0x390e, 0xb8fd, 0x3878, 0xb497, 0x3e25, 0xb643, 0xaaaf, 0xbc84, 0x3871, 0xb490,
    0xb5c4, 0x3983, 0x3969, 0x1a4a, 0x3ce5, 0x311d, 0xbd3b, 0x3bff, 0xc104, 0xbcd9, 0x3199, 0xbcbf,
    0xbcf0, 0xaf8e, 0xbf02, 0x283b, 0x337e, 0xc237, 0xbc27, 0x38e5, 0xbb01, 0x350c, 0xbc6a, 0x342f,
    0xbc86, 0xb9fa, 0xb6f2, 0x3d88, 0xb69a, 0xbf09, 0x3a20, 0x1cf8, 0x3056, 0xba28, 0xb035, 0x3668,
    0x3dbc, 0x2c91, 0x2c22, 0xb8b2, 0xb8a6, 0xb4a4, 0xbd37, 0x301f, 0xb466, 0x38bd, 0x40ab, 0xb85d,
    0xb874, 0x3728, 0xba74, 0x2ddb, 0xb9a7, 0xb5b7, 0xbc66, 0xb876, 0xb712, 0x3668, 0xb954, 0xbdd9,
    0x39d4, 0xbe65, 0x30e6, 0xb00e, 0xc1ae, 0xbf60, 0xb3e7, 0x396c, 0x36fe, 0x4299, 0x3ca3, 0xb9ba,
    0xba9a, 0x3738, 0xc0c1, 0xb8b1, 0xbaf1, 0x2d0e, 0x3053, 0xb535, 0x37c0, 0x3caa, 0xb127, 0xb0e0,
    0xb85f, 0xba74, 0x3491, 0x3151, 0xb803, 0xbc31, 0x3e4d, 0x2804, 0x33ab, 0xb942, 0xb007, 0xbc3e,
    0x3d3b, 0xb4d3, 0xbcf7, 0x391a, 0xbe47, 0x35ab, 0x33c5, 0xb647, 0x38f5, 0xb8b5, 0xbedc, 0xb2b7,
    0xbaa0, 0x354b, 0x313c, 0xb33b, 0x2e3a, 0xb739, 0x3f2c, 0xbeb2, 0xbe54, 0xc033, 0xb7cd, 0xbaf7,
    0x3903, 0xbde2, 0xb0fd, 0x3760, 0x3d59, 0x336f, 0x3139, 0x3947, 0x3c53, 0x2eaa, 0x390f, 0xb530,
    0xbc2c, 0xbc30, 0xbd7d, 0xbaca, 0x3021, 0x306a, 0x30f5, 0xbc5d, 0x244b, 0xb512, 0x3d90, 0x2ca7,
    0x3c28, 0x3c77, 0xb74b, 0xac48, 0xaae1, 0x380a, 0xbaba, 0xbccc, 0x3b07, 0x39f0, 0x327f, 0x2f04,
    0xb8cb, 0xbd5a, 0xb9ef, 0xb928, 0xbd4e, 0x32bc, 0xbd6f, 0xbebe, 0xb791, 0xb453, 0x3bf3, 0xbbb3,
    0x3c83, 0xb75a, 0xb74e, 0xbbef, 0xc308, 0xbff1, 0x39e5, 0x3b63, 0xb87f, 0xbdc8, 0xc028, 0x3b89,
    0xbd56, 0x2e4a, 0x38db, 0x35e1, 0x4208, 0xbc8b, 0xbe41, 0x3b38, 0xb1b5, 0x31c5, 0xb513, 0xb81c,
    0x3cf1, 0xa840, 0x3dcb, 0x3c09, 0x3b66, 0x2eab, 0x3a1a, 0xbd85, 0x346b, 0x386b, 0x3c67, 0xb8f9,
    0xba74, 0x3935, 0x3453, 0xb56d, 0xbb82, 0xae01, 0xb9c6, 0x3d84, 0xb65f, 0xb4f4, 0xbdf7, 0x3bbb,
    0x3702, 0x38fb, 0x2577, 0x338a, 0xbd2f, 0x2407, 0xba7c, 0x3fe5, 0xb3e1, 0x39c4, 0x3a94, 0xba0c,
    0xb944, 0x3583, 0xc1c9, 0xb485, 0x3d03, 0xb4b5, 0x3966, 0xba78, 0x3b6e, 0x3d1c, 0xbee0, 0xb150,
    0x3796, 0xb69a, 0xbd79, 0xbbd4, 0xba3d, 0xbc2e, 0xbb3b, 0xad3d, 0xbae5, 0xaee5, 0xb4cb, 0xb6e4,
    0xb78a, 0x3ba8, 0x3c6e, 0x3b58, 0x3a15, 0x39ea, 0xaddd, 0xb9ea, 0x3bc6, 0xb655, 0xba08, 0xb142,
    0xad9c, 0x2750, 0xbb5b, 0xb958, 0x40f9, 0x36d8, 0x3904, 0x3799, 0x2dee, 0x3aa4, 0xb98a, 0x3a46,
    0xbb23, 0xc0f7, 0x2878, 0xbd61, 0xb79b, 0xb495, 0xb9dc, 0x321a, 0xb7dd, 0xc22f, 0xbd1d, 0xb948,
    0xb0d3, 0x2968, 0xbc1a, 0x3014, 0xbff0, 0xb64d, 0xbd00, 0xbc90, 0x3f1c, 0x33dd, 0xba6a, 0x3a36,
    0xb635, 0xb5dc, 0x3ca1, 0x2fe4, 0xac91, 0x3da6, 0xb97c, 0xc228, 0x2809, 0xb9a3, 0x3a4d, 0x36f1,
    0xbf72, 0xbd54, 0x403d, 0xbd6b, 0xb946, 0xbdfd, 0x3680, 0xc028, 0xbd3c, 0x2a6f, 0xb8cf, 0x3c08,
    0x2b3e, 0xbbbf, 0x26c8, 0x311e, 0x3ab9, 0xbc99, 0xa98c, 0xb283, 0x3c64, 0xbc92, 0xb488, 0xb6d9,
    0x3c5f, 0xb967, 0xb8ed, 0x4109, 0x3abb, 0x3f77, 0x3198, 0xb687, 0xa78a, 0xb46a, 0xbbac, 0xae6b,
    0x3cfe, 0xad12, 0xb74a, 0x3a26, 0x3e39, 0xbf81, 0xb036, 0xbd6f, 0x3a50, 0x3a77, 0xbc21, 0xb83f,
    0xb0c0, 0xbf23, 0xbcfd, 0x317c, 0xaa3a, 0xba6d, 0xadfa, 0x3be0, 0x399a, 0x384e, 0x3279, 0xbcbf,
    0x385a, 0x3c87, 0x3a65, 0x353a, 0xbd0c, 0x3753, 0x3ca3, 0xb582, 0x3943, 0x32ee, 0x3cc4, 0xb464,
    0xa53a, 0xbae9, 0x3b71, 0x3c39, 0x3997, 0xb0d4, 0xb54a, 0x3ace, 0x2c69, 0x35c0, 0xb435, 0x2fff,
    0x245e, 0xbbeb, 0x3816, 0x33b5, 0x38ec, 0xb908, 0x3af5, 0xacc1, 0x3ad9, 0x34cf, 0xbc4a, 0xbe6d,
    0xb5a8, 0xb7bf, 0xb8d6, 0xb74b, 0x3bab, 0x3d78, 0x385c, 0x387c, 0x3586, 0xb936, 0xb691, 0xb278,
    0xb14d, 0x2cb4, 0x3ac7, 0xbf33, 0xbc0d, 0xb03d, 0x3bab, 0xb95c, 0xba53, 0xc063, 0xb6ce, 0xbee9,
    0xbc2c, 0x39bc, 0xb714, 0xbcc6, 0x38ed, 0xbe17, 0x3908, 0xb9a4, 0x396c, 0xb2d2, 0xbeb3, 0xbc5a,
    0xb937, 0xbc89, 0x40a1, 0xb5a5, 0xb452, 0xc00a, 0xbd13, 0x34f1, 0xb47d, 0xbd85, 0xbe62, 0x3924,
    0x3ec6, 0xba99, 0xbd06, 0x3897, 0x3e58, 0x3c46, 0xbb04, 0x35ef, 0x3ce2, 0x38c8, 0x1fc9, 0x2df8,
    0xbb9e, 0xa662, 0xbb95, 0xb4cb, 0x9932, 0x3dac, 0xac6b, 0xbe5f, 0x35a8, 0x4346, 0xba20, 0xbc75,
    0x36dc, 0xbaf3, 0xbb6a, 0xb2fc, 0xbc6c, 0x34b0, 0xb33a, 0x3852, 0xb74a, 0xb97f, 0xba55, 0xbc53,
    0x3412, 0xbd33, 0xb7e8, 0xb0d5, 0x3e3f, 0xb6cc, 0xb7c9, 0x3746, 0x2a19, 0xaf73, 0x2b59, 0x302a,
    0xae7b, 0xb87f, 0x3de8, 0xbefb, 0x3d64, 0xb750, 0x3afc, 0xab2f, 0xbb1a, 0xbbe4, 0x3e43, 0x37fa,
    0x3878, 0xb309, 0x3901, 0x3460, 0xb918, 0xb69a, 0xbd15, 0x2c42, 0x3d59, 0x3b39, 0xba37, 0xbe00,
    0xbf69, 0xbd42, 0xbaf1, 0xb54f, 0x3117, 0x3b95, 0xb1ab, 0x3e8e, 0xc05a, 0x38b3, 0x36cb, 0xb58c,
    0x3924, 0x3793, 0x3bb7, 0x3d8c, 0xbb4f, 0x3c25, 0xb963, 0x3e01,
];

const SDPOSE_SD2_SOURCE_DOMAIN: &[u8] = b"zed.sim.sdpose-sd2-capture.v1\0";
const SDPOSE_MODEL_SOURCE_DOMAIN: &[u8] = b"zed.sim.sdpose-model-resource.v1\0";
const SDPOSE_HEATMAP_HEAD_SOURCE_DOMAIN: &[u8] = b"zed.sim.sdpose-heatmap-head.v1\0";
pub const SDPOSE_HEAD_SOURCE_SHA256: &str =
    "19a55d1ecf16796226ed204241852b9b237a563addf636ff738167d9273cf97a";
pub const SDPOSE_MODEL_DETECTION_SOURCE_SHA256: &str =
    "f13b11988fccf9fa4d878ef5f63313c23c5f1400ec8cde04a502584e157c5072";
const OPENAI_MODEL_SOURCE_SHA256: &str =
    "9d27fb036cab8a262ef3d866a643f7fdc40994022616f1b8be14b7d919f57f96";
const ATTENTION_SOURCE_SHA256: &str =
    "436e1d91f8d5d84c5667e051cdf3ab2f91d8db25b66d88a084c89a202de0579e";
const MODEL_BASE_SOURCE_SHA256: &str =
    "99dc53baee665eca1a6aea70cfb9ab071d55784dff339b5e919dc14ae4fde8bd";
const SUPPORTED_MODELS_SOURCE_SHA256: &str =
    "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69";
const SD2_CHANNEL_MULTIPLIERS: [usize; 4] = [1, 2, 4, 4];
const SD2_RESIDUAL_BLOCKS: [usize; 4] = [2, 2, 2, 2];
const SD2_INPUT_TRANSFORMER_DEPTHS: [usize; 8] = [1, 1, 1, 1, 1, 1, 0, 0];
const SD2_OUTPUT_TRANSFORMER_DEPTHS: [usize; 12] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SdPoseSd2Configuration {
    source_exact_profile: bool,
    model_channels: usize,
    context_dimension: usize,
    attention_head_channels: usize,
    normalization_groups: usize,
    latent_height: usize,
    latent_width: usize,
}

impl SdPoseSd2Configuration {
    pub const fn source() -> Self {
        Self {
            source_exact_profile: true,
            model_channels: 320,
            context_dimension: 1_024,
            attention_head_channels: 64,
            normalization_groups: 32,
            latent_height: 128,
            latent_width: 96,
        }
    }

    pub fn reduced_fixture(
        model_channels: usize,
        context_dimension: usize,
        attention_head_channels: usize,
        normalization_groups: usize,
        latent_height: usize,
        latent_width: usize,
    ) -> Result<Self, SdPoseSd2Error> {
        let configuration = Self {
            source_exact_profile: false,
            model_channels,
            context_dimension,
            attention_head_channels,
            normalization_groups,
            latent_height,
            latent_width,
        };
        configuration.validate()?;
        Ok(configuration)
    }

    pub const fn is_source_exact(&self) -> bool {
        self.source_exact_profile
    }

    pub const fn model_channels(&self) -> usize {
        self.model_channels
    }

    pub const fn context_dimension(&self) -> usize {
        self.context_dimension
    }

    pub const fn latent_height(&self) -> usize {
        self.latent_height
    }

    pub const fn latent_width(&self) -> usize {
        self.latent_width
    }

    pub const fn capture_channels(&self) -> usize {
        self.model_channels * 2
    }

    fn validate(&self) -> Result<(), SdPoseSd2Error> {
        if self.model_channels == 0
            || self.context_dimension == 0
            || self.attention_head_channels == 0
            || self.normalization_groups == 0
            || self.latent_height == 0
            || self.latent_width == 0
            || !self.latent_height.is_multiple_of(8)
            || !self.latent_width.is_multiple_of(8)
        {
            return Err(SdPoseSd2Error::InvalidConfiguration);
        }
        for multiplier in SD2_CHANNEL_MULTIPLIERS {
            let channels = self
                .model_channels
                .checked_mul(multiplier)
                .ok_or(SdPoseSd2Error::Overflow("SD2 channels"))?;
            if !channels.is_multiple_of(self.attention_head_channels)
                || !channels.is_multiple_of(self.normalization_groups)
            {
                return Err(SdPoseSd2Error::InvalidConfiguration);
            }
        }
        if self.source_exact_profile && self != &Self::source() {
            return Err(SdPoseSd2Error::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SdPoseSd2WeightSpec {
    key: String,
    shape: Vec<u64>,
}

impl SdPoseSd2WeightSpec {
    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[derive(Debug, Error)]
pub enum SdPoseSd2Error {
    #[error("SDPose SD2 configuration is invalid")]
    InvalidConfiguration,
    #[error("SDPose SD2 production admission requires the exact LotusD family binding")]
    WrongFamily,
    #[error(
        "SDPose SD2 weights differ from the complete source topology; missing={missing:?}, unexpected={unexpected:?}"
    )]
    WeightKeys {
        missing: Vec<String>,
        unexpected: Vec<String>,
    },
    #[error(
        "SDPose SD2 weight {key} expected a supported dense CPU dtype and shape {expected:?}, got {dtype:?} {device:?} {actual:?}"
    )]
    WeightShape {
        key: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
        dtype: DType,
        device: DeviceId,
    },
    #[error("SDPose SD2 tensor {name} has invalid shape {actual:?}")]
    InputShape {
        name: &'static str,
        actual: Vec<u64>,
    },
    #[error("SDPose SD2 tensor stream differs from the retained model stream")]
    StreamMismatch,
    #[error("SDPose SD2 forward did not produce the required last pre-output-block capture")]
    MissingCapture,
    #[error("SDPose SD2 capture has invalid shape {0:?}")]
    InvalidCapture(Vec<u64>),
    #[error("SDPose SD2 arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("SDPose SD2 retained storage {0:?} has inconsistent byte lengths")]
    InconsistentStorage(StorageId),
    #[error("SDPose SD2 input or retained weight contains a non-finite value")]
    NonFinite,
    #[error(transparent)]
    Cancellation(#[from] CancellationError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    TensorOperation(#[from] NativeDiffusionTensorError),
    #[error(transparent)]
    Attention(#[from] AttentionError),
    #[error(transparent)]
    Shape(#[from] ShapeLayoutTransformPartTwoError),
    #[error(transparent)]
    Module(#[from] NativeOpsError),
    #[error(transparent)]
    Operator(#[from] OperatorIndirectionError),
}

#[derive(Clone, Debug)]
pub struct SdPoseSd2ForwardOutput {
    denoised: Tensor,
    feature_640: Tensor,
    capture_output_block: usize,
}

impl SdPoseSd2ForwardOutput {
    pub fn denoised(&self) -> &Tensor {
        &self.denoised
    }

    pub fn feature_640(&self) -> &Tensor {
        &self.feature_640
    }

    pub const fn capture_output_block(&self) -> usize {
        self.capture_output_block
    }
}

#[derive(Clone, Debug)]
pub struct NativeSdPoseSd2Denoiser {
    configuration: SdPoseSd2Configuration,
    weights: BTreeMap<String, Tensor>,
    dtype: DType,
    stream: StreamId,
    semantic_state_digest_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SdPoseHeatmapHeadConfiguration {
    source_exact_profile: bool,
    input_channels: usize,
    hidden_channels: usize,
    output_channels: usize,
}

impl SdPoseHeatmapHeadConfiguration {
    pub const fn source() -> Self {
        Self {
            source_exact_profile: true,
            input_channels: 640,
            hidden_channels: 640,
            output_channels: SDPOSE_HEATMAP_CHANNELS,
        }
    }

    pub fn reduced_fixture(
        input_channels: usize,
        hidden_channels: usize,
        output_channels: usize,
    ) -> Result<Self, SdPoseModelError> {
        let configuration = Self {
            source_exact_profile: false,
            input_channels,
            hidden_channels,
            output_channels,
        };
        configuration.validate()?;
        Ok(configuration)
    }

    pub const fn is_source_exact(&self) -> bool {
        self.source_exact_profile
    }

    pub const fn input_channels(&self) -> usize {
        self.input_channels
    }

    pub const fn output_channels(&self) -> usize {
        self.output_channels
    }

    fn validate(&self) -> Result<(), SdPoseModelError> {
        if self.input_channels == 0 || self.hidden_channels == 0 || self.output_channels == 0 {
            return Err(SdPoseModelError::InvalidConfiguration);
        }
        if self.source_exact_profile && self != &Self::source() {
            return Err(SdPoseModelError::InvalidConfiguration);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SdPoseHeatmapHeadWeightSpec {
    key: String,
    shape: Vec<u64>,
}

impl SdPoseHeatmapHeadWeightSpec {
    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }
}

#[derive(Clone, Debug)]
pub struct NativeSdPoseHeatmapHead {
    configuration: SdPoseHeatmapHeadConfiguration,
    weights: BTreeMap<String, Tensor>,
    dtype: DType,
    stream: StreamId,
    semantic_state_digest_sha256: String,
}

#[derive(Clone, Debug)]
pub struct NativeSdPoseModel {
    artifact_sha256: String,
    denoiser: NativeSdPoseSd2Denoiser,
    heatmap_head: NativeSdPoseHeatmapHead,
    semantic_state_digest_sha256: String,
}

#[derive(Debug, Error)]
pub enum SdPoseModelError {
    #[error("SDPose model or heatmap-head configuration is invalid")]
    InvalidConfiguration,
    #[error("SDPose model production admission requires the exact source profile")]
    ReducedProductionResource,
    #[error("SDPose model artifact identity is invalid")]
    InvalidArtifactIdentity,
    #[error(
        "SDPose heatmap-head weights differ from the complete source topology; missing={missing:?}, unexpected={unexpected:?}"
    )]
    WeightKeys {
        missing: Vec<String>,
        unexpected: Vec<String>,
    },
    #[error("SDPose heatmap-head weight {key} expected shape {expected:?}, got {actual:?}")]
    WeightShape {
        key: String,
        expected: Vec<u64>,
        actual: Vec<u64>,
    },
    #[error("SDPose denoiser and heatmap head target different dtype, device, stream, or channels")]
    ComponentMismatch,
    #[error("SDPose retained storage {0:?} has inconsistent byte lengths")]
    InconsistentStorage(StorageId),
    #[error("SDPose model arithmetic overflowed while computing {0}")]
    Overflow(&'static str),
    #[error("SDPose heatmap-head weight contains a non-finite value")]
    NonFinite,
    #[error(transparent)]
    Denoiser(#[from] SdPoseSd2Error),
    #[error(transparent)]
    Cancellation(#[from] CancellationError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Module(#[from] NativeOpsError),
    #[error(transparent)]
    Operator(#[from] OperatorIndirectionError),
}

pub fn prepare_lotus_sdpose_conditioning(
    backend: &CpuBackend,
    batch_size: usize,
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<(Tensor, Tensor), SdPoseModelError> {
    context.check()?;
    if batch_size == 0 || !matches!(dtype, DType::F16 | DType::Bf16 | DType::F32) {
        return Err(SdPoseModelError::InvalidConfiguration);
    }
    let conditioning_count = batch_size
        .checked_mul(LOTUS_CONDITIONING_F16_BITS.len())
        .ok_or(SdPoseModelError::Overflow("Lotus conditioning elements"))?;
    let mut conditioning_values = backend.workspace_vec(context, conditioning_count)?;
    for batch_index in 0..batch_size {
        context.check()?;
        for (value_index, bits) in LOTUS_CONDITIONING_F16_BITS.iter().copied().enumerate() {
            if value_index % 64 == 0 {
                context.check()?;
            }
            let value = match DType::F16.decode_scalar(&bits.to_le_bytes())? {
                DecodedScalar::Real(value) => value as f32,
                _ => return Err(SdPoseModelError::InvalidConfiguration),
            };
            conditioning_values.try_push(value)?;
        }
        debug_assert_eq!(
            conditioning_values.len(),
            (batch_index + 1) * LOTUS_CONDITIONING_F16_BITS.len()
        );
    }
    let batch_dimension = u64::try_from(batch_size)
        .map_err(|_| SdPoseModelError::Overflow("Lotus conditioning batch"))?;
    let conditioning = tensor_from_values(
        backend,
        &[batch_dimension, 2, 1024],
        &conditioning_values,
        dtype,
        DeviceId::CPU,
        context,
    )?;

    let adm_count = batch_size
        .checked_mul(4)
        .ok_or(SdPoseModelError::Overflow("Lotus ADM elements"))?;
    let mut adm_values = backend.workspace_vec(context, adm_count)?;
    let source_adm = [1.0_f32.sin(), 0.0, 1.0_f32.cos(), 1.0];
    for _ in 0..batch_size {
        context.check()?;
        for value in source_adm {
            adm_values.try_push(value)?;
        }
    }
    let adm = tensor_from_values(
        backend,
        &[batch_dimension, 4],
        &adm_values,
        dtype,
        DeviceId::CPU,
        context,
    )?;
    context.check()?;
    Ok((conditioning, adm))
}

pub fn sdpose_heatmap_head_weight_manifest(
    configuration: &SdPoseHeatmapHeadConfiguration,
) -> Result<Vec<SdPoseHeatmapHeadWeightSpec>, SdPoseModelError> {
    configuration.validate()?;
    let input = u64::try_from(configuration.input_channels)
        .map_err(|_| SdPoseModelError::Overflow("heatmap input channels"))?;
    let hidden = u64::try_from(configuration.hidden_channels)
        .map_err(|_| SdPoseModelError::Overflow("heatmap hidden channels"))?;
    let output = u64::try_from(configuration.output_channels)
        .map_err(|_| SdPoseModelError::Overflow("heatmap output channels"))?;
    Ok(vec![
        SdPoseHeatmapHeadWeightSpec {
            key: "native.heatmap_head.deconv_layers.0.weight".to_owned(),
            shape: vec![input, hidden, 4, 4],
        },
        SdPoseHeatmapHeadWeightSpec {
            key: "native.heatmap_head.conv_layers.0.weight".to_owned(),
            shape: vec![hidden, hidden, 1, 1],
        },
        SdPoseHeatmapHeadWeightSpec {
            key: "native.heatmap_head.conv_layers.0.bias".to_owned(),
            shape: vec![hidden],
        },
        SdPoseHeatmapHeadWeightSpec {
            key: "native.heatmap_head.final_layer.weight".to_owned(),
            shape: vec![output, hidden, 1, 1],
        },
        SdPoseHeatmapHeadWeightSpec {
            key: "native.heatmap_head.final_layer.bias".to_owned(),
            shape: vec![output],
        },
    ])
}

pub fn sdpose_sd2_weight_manifest(
    configuration: &SdPoseSd2Configuration,
) -> Result<Vec<SdPoseSd2WeightSpec>, SdPoseSd2Error> {
    configuration.validate()?;
    let mut specifications = Vec::new();
    let model_channels = configuration.model_channels;
    let embedding_channels = model_channels
        .checked_mul(4)
        .ok_or(SdPoseSd2Error::Overflow("time embedding channels"))?;
    add_convolution_specifications(
        &mut specifications,
        "native.input_blocks.0.0",
        model_channels,
        4,
        3,
    )?;
    add_linear_specifications(
        &mut specifications,
        "native.time_embed.0",
        embedding_channels,
        model_channels,
        true,
    )?;
    add_linear_specifications(
        &mut specifications,
        "native.time_embed.2",
        embedding_channels,
        embedding_channels,
        true,
    )?;
    add_linear_specifications(
        &mut specifications,
        "native.label_emb.0.0",
        embedding_channels,
        4,
        true,
    )?;
    add_linear_specifications(
        &mut specifications,
        "native.label_emb.0.2",
        embedding_channels,
        embedding_channels,
        true,
    )?;

    let mut channels = model_channels;
    let mut input_block_channels = vec![channels];
    let mut input_block = 1usize;
    let mut transformer_index = 0usize;
    for (level, multiplier) in SD2_CHANNEL_MULTIPLIERS.iter().copied().enumerate() {
        for _ in 0..SD2_RESIDUAL_BLOCKS[level] {
            let output_channels = model_channels
                .checked_mul(multiplier)
                .ok_or(SdPoseSd2Error::Overflow("input block channels"))?;
            add_residual_block_specifications(
                &mut specifications,
                &format!("native.input_blocks.{input_block}.0"),
                channels,
                output_channels,
                embedding_channels,
            )?;
            channels = output_channels;
            if SD2_INPUT_TRANSFORMER_DEPTHS[transformer_index] != 0 {
                add_spatial_transformer_specifications(
                    &mut specifications,
                    &format!("native.input_blocks.{input_block}.1"),
                    channels,
                    configuration.context_dimension,
                )?;
            }
            transformer_index += 1;
            input_block_channels.push(channels);
            input_block += 1;
        }
        if level + 1 < SD2_CHANNEL_MULTIPLIERS.len() {
            add_convolution_specifications(
                &mut specifications,
                &format!("native.input_blocks.{input_block}.0.op"),
                channels,
                channels,
                3,
            )?;
            input_block_channels.push(channels);
            input_block += 1;
        }
    }

    add_residual_block_specifications(
        &mut specifications,
        "native.middle_block.0",
        channels,
        channels,
        embedding_channels,
    )?;
    add_spatial_transformer_specifications(
        &mut specifications,
        "native.middle_block.1",
        channels,
        configuration.context_dimension,
    )?;
    add_residual_block_specifications(
        &mut specifications,
        "native.middle_block.2",
        channels,
        channels,
        embedding_channels,
    )?;

    let mut output_depths = SD2_OUTPUT_TRANSFORMER_DEPTHS.to_vec();
    let mut output_block = 0usize;
    for (level, multiplier) in SD2_CHANNEL_MULTIPLIERS.iter().copied().enumerate().rev() {
        for residual_index in 0..=SD2_RESIDUAL_BLOCKS[level] {
            let skip_channels = input_block_channels
                .pop()
                .ok_or(SdPoseSd2Error::InvalidConfiguration)?;
            let input_channels = channels
                .checked_add(skip_channels)
                .ok_or(SdPoseSd2Error::Overflow("output block channels"))?;
            let output_channels = model_channels
                .checked_mul(multiplier)
                .ok_or(SdPoseSd2Error::Overflow("output block channels"))?;
            add_residual_block_specifications(
                &mut specifications,
                &format!("native.output_blocks.{output_block}.0"),
                input_channels,
                output_channels,
                embedding_channels,
            )?;
            channels = output_channels;
            let transformer_depth = output_depths
                .pop()
                .ok_or(SdPoseSd2Error::InvalidConfiguration)?;
            let mut next_layer = 1usize;
            if transformer_depth != 0 {
                add_spatial_transformer_specifications(
                    &mut specifications,
                    &format!("native.output_blocks.{output_block}.1"),
                    channels,
                    configuration.context_dimension,
                )?;
                next_layer += 1;
            }
            if level != 0 && residual_index == SD2_RESIDUAL_BLOCKS[level] {
                add_convolution_specifications(
                    &mut specifications,
                    &format!("native.output_blocks.{output_block}.{next_layer}.conv"),
                    channels,
                    channels,
                    3,
                )?;
            }
            output_block += 1;
        }
    }
    if !input_block_channels.is_empty() || !output_depths.is_empty() || output_block != 12 {
        return Err(SdPoseSd2Error::InvalidConfiguration);
    }
    add_normalization_specifications(&mut specifications, "native.out.0", model_channels)?;
    add_convolution_specifications(&mut specifications, "native.out.2", 4, model_channels, 3)?;

    let mut keys = BTreeSet::new();
    if specifications
        .iter()
        .any(|specification| !keys.insert(specification.key.clone()))
    {
        return Err(SdPoseSd2Error::InvalidConfiguration);
    }
    Ok(specifications)
}

fn push_weight_specification(
    specifications: &mut Vec<SdPoseSd2WeightSpec>,
    key: String,
    shape: &[usize],
) -> Result<(), SdPoseSd2Error> {
    let shape = shape
        .iter()
        .map(|dimension| {
            u64::try_from(*dimension).map_err(|_| SdPoseSd2Error::Overflow("weight shape"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    specifications.push(SdPoseSd2WeightSpec { key, shape });
    Ok(())
}

fn add_linear_specifications(
    specifications: &mut Vec<SdPoseSd2WeightSpec>,
    prefix: &str,
    output: usize,
    input: usize,
    bias: bool,
) -> Result<(), SdPoseSd2Error> {
    push_weight_specification(specifications, format!("{prefix}.weight"), &[output, input])?;
    if bias {
        push_weight_specification(specifications, format!("{prefix}.bias"), &[output])?;
    }
    Ok(())
}

fn add_convolution_specifications(
    specifications: &mut Vec<SdPoseSd2WeightSpec>,
    prefix: &str,
    output: usize,
    input: usize,
    kernel: usize,
) -> Result<(), SdPoseSd2Error> {
    push_weight_specification(
        specifications,
        format!("{prefix}.weight"),
        &[output, input, kernel, kernel],
    )?;
    push_weight_specification(specifications, format!("{prefix}.bias"), &[output])
}

fn add_normalization_specifications(
    specifications: &mut Vec<SdPoseSd2WeightSpec>,
    prefix: &str,
    channels: usize,
) -> Result<(), SdPoseSd2Error> {
    push_weight_specification(specifications, format!("{prefix}.weight"), &[channels])?;
    push_weight_specification(specifications, format!("{prefix}.bias"), &[channels])
}

fn add_residual_block_specifications(
    specifications: &mut Vec<SdPoseSd2WeightSpec>,
    prefix: &str,
    input: usize,
    output: usize,
    embedding: usize,
) -> Result<(), SdPoseSd2Error> {
    add_normalization_specifications(specifications, &format!("{prefix}.in_layers.0"), input)?;
    add_convolution_specifications(
        specifications,
        &format!("{prefix}.in_layers.2"),
        output,
        input,
        3,
    )?;
    add_linear_specifications(
        specifications,
        &format!("{prefix}.emb_layers.1"),
        output,
        embedding,
        true,
    )?;
    add_normalization_specifications(specifications, &format!("{prefix}.out_layers.0"), output)?;
    add_convolution_specifications(
        specifications,
        &format!("{prefix}.out_layers.3"),
        output,
        output,
        3,
    )?;
    if input != output {
        add_convolution_specifications(
            specifications,
            &format!("{prefix}.skip_connection"),
            output,
            input,
            1,
        )?;
    }
    Ok(())
}

fn add_spatial_transformer_specifications(
    specifications: &mut Vec<SdPoseSd2WeightSpec>,
    prefix: &str,
    channels: usize,
    context: usize,
) -> Result<(), SdPoseSd2Error> {
    add_normalization_specifications(specifications, &format!("{prefix}.norm"), channels)?;
    add_linear_specifications(
        specifications,
        &format!("{prefix}.proj_in"),
        channels,
        channels,
        true,
    )?;
    let block = format!("{prefix}.transformer_blocks.0");
    for normalization in ["norm1", "norm2", "norm3"] {
        add_normalization_specifications(
            specifications,
            &format!("{block}.{normalization}"),
            channels,
        )?;
    }
    for attention in ["attn1", "attn2"] {
        let attention_prefix = format!("{block}.{attention}");
        add_linear_specifications(
            specifications,
            &format!("{attention_prefix}.to_q"),
            channels,
            channels,
            false,
        )?;
        let key_value_input = if attention == "attn1" {
            channels
        } else {
            context
        };
        for projection in ["to_k", "to_v"] {
            add_linear_specifications(
                specifications,
                &format!("{attention_prefix}.{projection}"),
                channels,
                key_value_input,
                false,
            )?;
        }
        add_linear_specifications(
            specifications,
            &format!("{attention_prefix}.to_out.0"),
            channels,
            channels,
            true,
        )?;
    }
    let feed_forward_width = channels
        .checked_mul(4)
        .ok_or(SdPoseSd2Error::Overflow("feed-forward width"))?;
    add_linear_specifications(
        specifications,
        &format!("{block}.ff.net.0.proj"),
        feed_forward_width
            .checked_mul(2)
            .ok_or(SdPoseSd2Error::Overflow("GEGLU width"))?,
        channels,
        true,
    )?;
    add_linear_specifications(
        specifications,
        &format!("{block}.ff.net.2"),
        channels,
        feed_forward_width,
        true,
    )?;
    add_linear_specifications(
        specifications,
        &format!("{prefix}.proj_out"),
        channels,
        channels,
        true,
    )
}

impl NativeSdPoseSd2Denoiser {
    pub fn from_mapped_weights(
        mapped: &MappedModelWeights,
        cancellation: &CancellationToken,
    ) -> Result<Self, SdPoseSd2Error> {
        cancellation.check()?;
        let binding = mapped.binding().ok_or(SdPoseSd2Error::WrongFamily)?;
        if binding.family().feature_id() != generated_lotusd_sim_model_0106::MODEL_FAMILY_FEATURE_ID
            || binding.family().identifier()
                != generated_lotusd_sim_model_0106::MODEL_FAMILY_IDENTIFIER
        {
            return Err(SdPoseSd2Error::WrongFamily);
        }
        let candidate_weights = mapped
            .tensors()
            .iter()
            .filter(|(key, _)| is_sd2_unet_key(key) || key.starts_with("native.heatmap_head."))
            .map(|(key, tensor)| (key.clone(), tensor.clone()))
            .collect::<BTreeMap<_, _>>();
        Self::checked(
            SdPoseSd2Configuration::source(),
            &candidate_weights,
            true,
            cancellation,
        )
    }

    pub fn from_reduced_fixture(
        configuration: SdPoseSd2Configuration,
        weights: BTreeMap<String, Tensor>,
        cancellation: &CancellationToken,
    ) -> Result<Self, SdPoseSd2Error> {
        if configuration.is_source_exact() {
            return Err(SdPoseSd2Error::InvalidConfiguration);
        }
        Self::checked(configuration, &weights, false, cancellation)
    }

    fn checked(
        configuration: SdPoseSd2Configuration,
        candidate_weights: &BTreeMap<String, Tensor>,
        allow_heatmap_head: bool,
        cancellation: &CancellationToken,
    ) -> Result<Self, SdPoseSd2Error> {
        cancellation.check()?;
        configuration.validate()?;
        let manifest = sdpose_sd2_weight_manifest(&configuration)?;
        let expected = manifest
            .iter()
            .map(|specification| specification.key.as_str())
            .collect::<BTreeSet<_>>();
        let actual = candidate_weights
            .keys()
            .filter(|key| !(allow_heatmap_head && key.starts_with("native.heatmap_head.")))
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if expected != actual {
            return Err(SdPoseSd2Error::WeightKeys {
                missing: expected
                    .difference(&actual)
                    .map(|key| (*key).to_owned())
                    .collect(),
                unexpected: actual
                    .difference(&expected)
                    .map(|key| (*key).to_owned())
                    .collect(),
            });
        }
        let first = manifest
            .first()
            .and_then(|specification| candidate_weights.get(&specification.key))
            .ok_or(SdPoseSd2Error::InvalidConfiguration)?;
        let dtype = first.descriptor().dtype();
        let stream = first.descriptor().stream();
        let mut weights = BTreeMap::new();
        for specification in manifest {
            cancellation.check()?;
            let tensor = candidate_weights.get(&specification.key).ok_or_else(|| {
                SdPoseSd2Error::WeightKeys {
                    missing: vec![specification.key.clone()],
                    unexpected: Vec::new(),
                }
            })?;
            let descriptor = tensor.descriptor();
            if descriptor.shape() != specification.shape
                || descriptor.dtype() != dtype
                || !matches!(dtype, DType::F32 | DType::F16 | DType::Bf16)
                || descriptor.device() != DeviceId::CPU
                || descriptor.stream() != stream
            {
                return Err(SdPoseSd2Error::WeightShape {
                    key: specification.key,
                    expected: specification.shape,
                    actual: descriptor.shape().to_vec(),
                    dtype: descriptor.dtype(),
                    device: descriptor.device(),
                });
            }
            require_finite_tensor(tensor, cancellation)?;
            weights.insert(specification.key, tensor.clone());
        }
        let semantic_state_digest_sha256 =
            sdpose_sd2_semantic_digest(&configuration, &weights, cancellation)?;
        Ok(Self {
            configuration,
            weights,
            dtype,
            stream,
            semantic_state_digest_sha256,
        })
    }

    pub fn configuration(&self) -> &SdPoseSd2Configuration {
        &self.configuration
    }

    pub const fn execution_stream(&self) -> StreamId {
        self.stream
    }

    pub const fn execution_dtype(&self) -> DType {
        self.dtype
    }

    pub fn semantic_state_digest_sha256(&self) -> &str {
        &self.semantic_state_digest_sha256
    }

    pub fn validate(&self, cancellation: &CancellationToken) -> Result<(), SdPoseSd2Error> {
        cancellation.check()?;
        self.configuration.validate()?;
        let digest = sdpose_sd2_semantic_digest(&self.configuration, &self.weights, cancellation)?;
        if digest != self.semantic_state_digest_sha256 {
            return Err(SdPoseSd2Error::InvalidConfiguration);
        }
        let manifest = sdpose_sd2_weight_manifest(&self.configuration)?;
        if manifest.len() != self.weights.len() {
            return Err(SdPoseSd2Error::InvalidConfiguration);
        }
        for specification in manifest {
            cancellation.check()?;
            let tensor = self
                .weights
                .get(&specification.key)
                .ok_or(SdPoseSd2Error::InvalidConfiguration)?;
            if tensor.descriptor().shape() != specification.shape
                || tensor.descriptor().dtype() != self.dtype
                || tensor.descriptor().device() != DeviceId::CPU
                || tensor.descriptor().stream() != self.stream
            {
                return Err(SdPoseSd2Error::InvalidConfiguration);
            }
        }
        cancellation.check()?;
        Ok(())
    }

    pub fn resident_tensor_allocations(&self) -> Result<Vec<(StorageId, u64)>, SdPoseSd2Error> {
        let mut allocations = HashMap::new();
        for tensor in self.weights.values() {
            let storage = tensor.storage_id();
            let bytes = tensor.storage_byte_len();
            if let Some(existing) = allocations.insert(storage, bytes)
                && existing != bytes
            {
                return Err(SdPoseSd2Error::InconsistentStorage(storage));
            }
        }
        Ok(allocations.into_iter().collect())
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, SdPoseSd2Error> {
        let entries = self
            .weights
            .len()
            .checked_mul(std::mem::size_of::<(String, Tensor)>())
            .ok_or(SdPoseSd2Error::Overflow("retained weight entries"))?;
        let keys = self.weights.keys().try_fold(0usize, |total, key| {
            total
                .checked_add(key.capacity())
                .ok_or(SdPoseSd2Error::Overflow("retained weight keys"))
        })?;
        let owned = std::mem::size_of::<Self>()
            .checked_add(entries)
            .and_then(|bytes| bytes.checked_add(keys))
            .and_then(|bytes| bytes.checked_add(self.semantic_state_digest_sha256.capacity()))
            .ok_or(SdPoseSd2Error::Overflow("retained owner bytes"))?;
        u64::try_from(owned).map_err(|_| SdPoseSd2Error::Overflow("retained owner bytes"))
    }

    pub fn resident_bytes(&self) -> Result<u64, SdPoseSd2Error> {
        self.resident_tensor_allocations()?.into_iter().try_fold(
            self.resident_owned_bytes()?,
            |total, (_, bytes)| {
                total
                    .checked_add(bytes)
                    .ok_or(SdPoseSd2Error::Overflow("retained total bytes"))
            },
        )
    }

    fn weight(&self, key: &str) -> Result<&Tensor, SdPoseSd2Error> {
        self.weights
            .get(key)
            .ok_or(SdPoseSd2Error::InvalidConfiguration)
    }

    pub fn forward(
        &self,
        backend: &CpuBackend,
        latent: &Tensor,
        timesteps: &[f32],
        conditioning: &Tensor,
        adm: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<SdPoseSd2ForwardOutput, SdPoseSd2Error> {
        context.cancellation.check()?;
        context.check()?;
        self.validate(context.cancellation)?;
        if context.stream != self.stream {
            return Err(SdPoseSd2Error::StreamMismatch);
        }
        let batch = require_forward_inputs(
            &self.configuration,
            self.dtype,
            self.stream,
            latent,
            timesteps,
            conditioning,
            adm,
        )?;
        require_finite_tensor(latent, context.cancellation)?;
        require_finite_tensor(conditioning, context.cancellation)?;
        require_finite_tensor(adm, context.cancellation)?;

        let timestep_embedding = build_timestep_embedding(
            backend,
            timesteps,
            self.configuration.model_channels,
            self.dtype,
            context,
        )?;
        let mut embedding = self.linear(
            backend,
            &timestep_embedding,
            "native.time_embed.0",
            true,
            context,
        )?;
        embedding = immutable_silu(backend, &embedding, context)?;
        embedding = self.linear(backend, &embedding, "native.time_embed.2", true, context)?;
        let mut label = self.linear(backend, adm, "native.label_emb.0.0", true, context)?;
        label = immutable_silu(backend, &label, context)?;
        label = self.linear(backend, &label, "native.label_emb.0.2", true, context)?;
        embedding = add_tensors(backend, &embedding, &label, context)?;

        let mut hidden =
            self.convolution(backend, latent, "native.input_blocks.0.0", 1, 1, context)?;
        let mut skips = Vec::new();
        skips
            .try_reserve_exact(12)
            .map_err(|_| SdPoseSd2Error::Overflow("skip allocation"))?;
        skips.push(hidden.clone());
        let mut input_block = 1usize;
        let mut transformer_index = 0usize;
        for (level, _) in SD2_CHANNEL_MULTIPLIERS.iter().enumerate() {
            for _ in 0..SD2_RESIDUAL_BLOCKS[level] {
                context.check()?;
                hidden = self.residual_block(
                    backend,
                    &hidden,
                    &embedding,
                    &format!("native.input_blocks.{input_block}.0"),
                    context,
                )?;
                if SD2_INPUT_TRANSFORMER_DEPTHS[transformer_index] != 0 {
                    hidden = self.spatial_transformer(
                        backend,
                        &hidden,
                        conditioning,
                        &format!("native.input_blocks.{input_block}.1"),
                        context,
                    )?;
                }
                transformer_index += 1;
                skips.push(hidden.clone());
                input_block += 1;
            }
            if level + 1 < SD2_CHANNEL_MULTIPLIERS.len() {
                hidden = self.convolution(
                    backend,
                    &hidden,
                    &format!("native.input_blocks.{input_block}.0.op"),
                    2,
                    1,
                    context,
                )?;
                skips.push(hidden.clone());
                input_block += 1;
            }
        }

        hidden = self.residual_block(
            backend,
            &hidden,
            &embedding,
            "native.middle_block.0",
            context,
        )?;
        hidden = self.spatial_transformer(
            backend,
            &hidden,
            conditioning,
            "native.middle_block.1",
            context,
        )?;
        hidden = self.residual_block(
            backend,
            &hidden,
            &embedding,
            "native.middle_block.2",
            context,
        )?;

        let mut capture = None;
        let mut capture_output_block = None;
        let mut output_depths = SD2_OUTPUT_TRANSFORMER_DEPTHS.to_vec();
        let mut output_block = 0usize;
        for (level, _) in SD2_CHANNEL_MULTIPLIERS.iter().enumerate().rev() {
            for residual_index in 0..=SD2_RESIDUAL_BLOCKS[level] {
                context.check()?;
                let hidden_shape = hidden.descriptor().shape();
                if hidden_shape.get(1).copied()
                    == Some(
                        u64::try_from(self.configuration.capture_channels())
                            .map_err(|_| SdPoseSd2Error::Overflow("capture channels"))?,
                    )
                {
                    capture = Some(copy_tensor(backend, &hidden, context)?);
                    capture_output_block = Some(output_block);
                }
                let skip = skips.pop().ok_or(SdPoseSd2Error::MissingCapture)?;
                hidden = concat_channel_tensors(backend, &hidden, &skip, context)?;
                hidden = self.residual_block(
                    backend,
                    &hidden,
                    &embedding,
                    &format!("native.output_blocks.{output_block}.0"),
                    context,
                )?;
                let transformer_depth = output_depths
                    .pop()
                    .ok_or(SdPoseSd2Error::InvalidConfiguration)?;
                let mut next_layer = 1usize;
                if transformer_depth != 0 {
                    hidden = self.spatial_transformer(
                        backend,
                        &hidden,
                        conditioning,
                        &format!("native.output_blocks.{output_block}.1"),
                        context,
                    )?;
                    next_layer += 1;
                }
                if level != 0 && residual_index == SD2_RESIDUAL_BLOCKS[level] {
                    hidden = nearest_upsample_tensor_2x(backend, &hidden, context)?;
                    hidden = self.convolution(
                        backend,
                        &hidden,
                        &format!("native.output_blocks.{output_block}.{next_layer}.conv"),
                        1,
                        1,
                        context,
                    )?;
                }
                output_block += 1;
            }
        }
        if !skips.is_empty() || !output_depths.is_empty() || output_block != 12 {
            return Err(SdPoseSd2Error::InvalidConfiguration);
        }
        hidden = self.normalization(backend, &hidden, "native.out.0", 1.0e-5, context)?;
        hidden = immutable_silu(backend, &hidden, context)?;
        let denoised = self.convolution(backend, &hidden, "native.out.2", 1, 1, context)?;
        let feature_640 = capture.ok_or(SdPoseSd2Error::MissingCapture)?;
        let expected_capture = [
            u64::try_from(batch).map_err(|_| SdPoseSd2Error::Overflow("capture batch"))?,
            u64::try_from(self.configuration.capture_channels())
                .map_err(|_| SdPoseSd2Error::Overflow("capture channels"))?,
            u64::try_from(self.configuration.latent_height)
                .map_err(|_| SdPoseSd2Error::Overflow("capture height"))?,
            u64::try_from(self.configuration.latent_width)
                .map_err(|_| SdPoseSd2Error::Overflow("capture width"))?,
        ];
        if feature_640.descriptor().shape() != expected_capture {
            return Err(SdPoseSd2Error::InvalidCapture(
                feature_640.descriptor().shape().to_vec(),
            ));
        }
        if self.configuration.is_source_exact() && capture_output_block != Some(9) {
            return Err(SdPoseSd2Error::InvalidCapture(
                feature_640.descriptor().shape().to_vec(),
            ));
        }
        context.check()?;
        Ok(SdPoseSd2ForwardOutput {
            denoised,
            feature_640,
            capture_output_block: capture_output_block.ok_or(SdPoseSd2Error::MissingCapture)?,
        })
    }

    fn linear(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        prefix: &str,
        bias: bool,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, SdPoseSd2Error> {
        let weight = self.weight(&format!("{prefix}.weight"))?;
        let shape = weight.descriptor().shape();
        let [output_features, input_features]: [u64; 2] =
            shape.try_into().map_err(|_| SdPoseSd2Error::InputShape {
                name: "linear weight",
                actual: shape.to_vec(),
            })?;
        let mut module = NativeModule::linear(
            prefix,
            usize::try_from(input_features)
                .map_err(|_| SdPoseSd2Error::Overflow("linear input features"))?,
            usize::try_from(output_features)
                .map_err(|_| SdPoseSd2Error::Overflow("linear output features"))?,
            bias,
            false,
        )?;
        module.load_dense_parameters(
            weight.clone(),
            bias.then(|| self.weight(&format!("{prefix}.bias")).cloned())
                .transpose()?,
        )?;
        Ok(module.forward_dense_inference_with_context(backend, input, context)?)
    }

    fn convolution(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        prefix: &str,
        stride: usize,
        padding: usize,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, SdPoseSd2Error> {
        let weight = self.weight(&format!("{prefix}.weight"))?;
        let shape = weight.descriptor().shape();
        let [output_channels, input_channels, kernel_height, kernel_width]: [u64; 4] =
            shape.try_into().map_err(|_| SdPoseSd2Error::InputShape {
                name: "convolution weight",
                actual: shape.to_vec(),
            })?;
        let geometry = ConvolutionGeometry::new(
            2,
            vec![stride; 2],
            vec![padding; 2],
            vec![1; 2],
            1,
            false,
            vec![0; 2],
        )?;
        let mut module = NativeModule::convolution(
            prefix,
            usize::try_from(input_channels)
                .map_err(|_| SdPoseSd2Error::Overflow("convolution input channels"))?,
            usize::try_from(output_channels)
                .map_err(|_| SdPoseSd2Error::Overflow("convolution output channels"))?,
            vec![
                usize::try_from(kernel_height)
                    .map_err(|_| SdPoseSd2Error::Overflow("convolution kernel height"))?,
                usize::try_from(kernel_width)
                    .map_err(|_| SdPoseSd2Error::Overflow("convolution kernel width"))?,
            ],
            true,
            geometry,
            false,
        )?;
        module.load_dense_parameters(
            weight.clone(),
            Some(self.weight(&format!("{prefix}.bias"))?.clone()),
        )?;
        Ok(module.forward_dense_inference_with_context(backend, input, context)?)
    }

    fn normalization(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        prefix: &str,
        epsilon: f32,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, SdPoseSd2Error> {
        let channels = usize::try_from(*input.descriptor().shape().get(1).ok_or(
            SdPoseSd2Error::InputShape {
                name: "group norm input",
                actual: input.descriptor().shape().to_vec(),
            },
        )?)
        .map_err(|_| SdPoseSd2Error::Overflow("group norm channels"))?;
        let mut module = NativeModule::group_norm(
            prefix,
            self.configuration.normalization_groups,
            channels,
            epsilon,
            true,
            false,
        )?;
        module.load_dense_parameters(
            self.weight(&format!("{prefix}.weight"))?.clone(),
            Some(self.weight(&format!("{prefix}.bias"))?.clone()),
        )?;
        Ok(module.forward_dense_inference_with_context(backend, input, context)?)
    }

    fn residual_block(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        embedding: &Tensor,
        prefix: &str,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, SdPoseSd2Error> {
        context.check()?;
        let mut hidden = self.normalization(
            backend,
            input,
            &format!("{prefix}.in_layers.0"),
            1.0e-5,
            context,
        )?;
        hidden = immutable_silu(backend, &hidden, context)?;
        hidden = self.convolution(
            backend,
            &hidden,
            &format!("{prefix}.in_layers.2"),
            1,
            1,
            context,
        )?;
        let embedding = immutable_silu(backend, embedding, context)?;
        let embedding = self.linear(
            backend,
            &embedding,
            &format!("{prefix}.emb_layers.1"),
            true,
            context,
        )?;
        hidden = add_embedding_bias(backend, &hidden, &embedding, context)?;
        hidden = self.normalization(
            backend,
            &hidden,
            &format!("{prefix}.out_layers.0"),
            1.0e-5,
            context,
        )?;
        hidden = immutable_silu(backend, &hidden, context)?;
        hidden = self.convolution(
            backend,
            &hidden,
            &format!("{prefix}.out_layers.3"),
            1,
            1,
            context,
        )?;
        let residual = if input.descriptor().shape().get(1) == hidden.descriptor().shape().get(1) {
            input.clone()
        } else {
            self.convolution(
                backend,
                input,
                &format!("{prefix}.skip_connection"),
                1,
                0,
                context,
            )?
        };
        add_tensors(backend, &residual, &hidden, context)
    }

    fn spatial_transformer(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        conditioning: &Tensor,
        prefix: &str,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, SdPoseSd2Error> {
        context.check()?;
        let shape = require_rank_four(input, "spatial transformer")?;
        let mut hidden =
            self.normalization(backend, input, &format!("{prefix}.norm"), 1.0e-6, context)?;
        hidden = nchw_to_tokens(backend, &hidden, context)?;
        hidden = self.linear(
            backend,
            &hidden,
            &format!("{prefix}.proj_in"),
            true,
            context,
        )?;
        let block = format!("{prefix}.transformer_blocks.0");

        let normalized =
            self.layer_normalization(backend, &hidden, &format!("{block}.norm1"), context)?;
        let attended = self.cross_attention(
            backend,
            &normalized,
            &normalized,
            &format!("{block}.attn1"),
            context,
        )?;
        hidden = add_tensors(backend, &hidden, &attended, context)?;

        let normalized =
            self.layer_normalization(backend, &hidden, &format!("{block}.norm2"), context)?;
        let attended = self.cross_attention(
            backend,
            &normalized,
            conditioning,
            &format!("{block}.attn2"),
            context,
        )?;
        hidden = add_tensors(backend, &hidden, &attended, context)?;

        let normalized =
            self.layer_normalization(backend, &hidden, &format!("{block}.norm3"), context)?;
        let projected = self.linear(
            backend,
            &normalized,
            &format!("{block}.ff.net.0.proj"),
            true,
            context,
        )?;
        let gated = geglu(backend, &projected, context)?;
        let feed_forward =
            self.linear(backend, &gated, &format!("{block}.ff.net.2"), true, context)?;
        hidden = add_tensors(backend, &hidden, &feed_forward, context)?;
        hidden = self.linear(
            backend,
            &hidden,
            &format!("{prefix}.proj_out"),
            true,
            context,
        )?;
        hidden = tokens_to_nchw(backend, &hidden, shape[2], shape[3], context)?;
        add_tensors(backend, input, &hidden, context)
    }

    fn layer_normalization(
        &self,
        backend: &CpuBackend,
        input: &Tensor,
        prefix: &str,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, SdPoseSd2Error> {
        let width = usize::try_from(*input.descriptor().shape().last().ok_or(
            SdPoseSd2Error::InputShape {
                name: "layer norm input",
                actual: input.descriptor().shape().to_vec(),
            },
        )?)
        .map_err(|_| SdPoseSd2Error::Overflow("layer norm width"))?;
        let mut module = NativeModule::layer_norm(prefix, vec![width], 1.0e-5, true, true, false)?;
        module.load_dense_parameters(
            self.weight(&format!("{prefix}.weight"))?.clone(),
            Some(self.weight(&format!("{prefix}.bias"))?.clone()),
        )?;
        Ok(module.forward_dense_inference_with_context(backend, input, context)?)
    }

    fn cross_attention(
        &self,
        backend: &CpuBackend,
        query_input: &Tensor,
        key_value_input: &Tensor,
        prefix: &str,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, SdPoseSd2Error> {
        let query = self.linear(
            backend,
            query_input,
            &format!("{prefix}.to_q"),
            false,
            context,
        )?;
        let key = self.linear(
            backend,
            key_value_input,
            &format!("{prefix}.to_k"),
            false,
            context,
        )?;
        let value = self.linear(
            backend,
            key_value_input,
            &format!("{prefix}.to_v"),
            false,
            context,
        )?;
        let query_shape = require_rank_three(&query, "attention query")?;
        let key_shape = require_rank_three(&key, "attention key")?;
        let value_shape = require_rank_three(&value, "attention value")?;
        if query_shape[0] != key_shape[0]
            || key_shape != value_shape
            || query_shape[2] != key_shape[2]
        {
            return Err(SdPoseSd2Error::InputShape {
                name: "attention",
                actual: query.descriptor().shape().to_vec(),
            });
        }
        let channels = usize::try_from(query_shape[2])
            .map_err(|_| SdPoseSd2Error::Overflow("attention channels"))?;
        let heads = channels
            .checked_div(self.configuration.attention_head_channels)
            .ok_or(SdPoseSd2Error::Overflow("attention heads"))?;
        let batch = usize::try_from(query_shape[0])
            .map_err(|_| SdPoseSd2Error::Overflow("attention batch"))?;
        let query_tokens = usize::try_from(query_shape[1])
            .map_err(|_| SdPoseSd2Error::Overflow("attention queries"))?;
        let key_tokens = usize::try_from(key_shape[1])
            .map_err(|_| SdPoseSd2Error::Overflow("attention keys"))?;
        let batch_i64 =
            i64::try_from(batch).map_err(|_| SdPoseSd2Error::Overflow("attention batch"))?;
        let query_tokens_i64 = i64::try_from(query_tokens)
            .map_err(|_| SdPoseSd2Error::Overflow("attention queries"))?;
        let key_tokens_i64 =
            i64::try_from(key_tokens).map_err(|_| SdPoseSd2Error::Overflow("attention keys"))?;
        let heads_i64 =
            i64::try_from(heads).map_err(|_| SdPoseSd2Error::Overflow("attention heads"))?;
        let head_dimension_i64 = i64::try_from(self.configuration.attention_head_channels)
            .map_err(|_| SdPoseSd2Error::Overflow("attention head dimension"))?;
        let query = tensor_reshape_with_context_exact_native(
            backend,
            &query,
            &[batch_i64, query_tokens_i64, heads_i64, head_dimension_i64],
            context,
        )?;
        let key = tensor_reshape_with_context_exact_native(
            backend,
            &key,
            &[batch_i64, key_tokens_i64, heads_i64, head_dimension_i64],
            context,
        )?;
        let value = tensor_reshape_with_context_exact_native(
            backend,
            &value,
            &[batch_i64, key_tokens_i64, heads_i64, head_dimension_i64],
            context,
        )?;
        let attended = scaled_dot_product_attention_tensor_with_context(
            backend,
            AttentionRequest {
                backend: AttentionBackend::SplitOrSubQuadratic,
                fallback: AttentionFallbackPolicy::AllowExactNative,
                batch,
                query_tokens,
                key_tokens,
                heads,
                head_dimension: self.configuration.attention_head_channels,
                value_dimension: self.configuration.attention_head_channels,
                scale: None,
                workspace_limit_bytes: key_tokens
                    .checked_mul(std::mem::size_of::<f32>())
                    .ok_or(SdPoseSd2Error::Overflow("attention workspace"))?,
            },
            &query,
            &key,
            &value,
            None,
            context,
        )?;
        let attended = tensor_reshape_with_context_exact_native(
            backend,
            &attended,
            &[
                batch_i64,
                query_tokens_i64,
                i64::try_from(channels)
                    .map_err(|_| SdPoseSd2Error::Overflow("attention channels"))?,
            ],
            context,
        )?;
        self.linear(
            backend,
            &attended,
            &format!("{prefix}.to_out.0"),
            true,
            context,
        )
    }
}

impl NativeSdPoseHeatmapHead {
    fn from_mapped_weights(
        mapped: &MappedModelWeights,
        cancellation: &CancellationToken,
    ) -> Result<Self, SdPoseModelError> {
        let weights = mapped
            .tensors()
            .iter()
            .filter(|(key, _)| key.starts_with("native.heatmap_head."))
            .map(|(key, tensor)| (key.clone(), tensor.clone()))
            .collect();
        Self::checked(
            SdPoseHeatmapHeadConfiguration::source(),
            weights,
            cancellation,
        )
    }

    pub fn from_reduced_fixture(
        configuration: SdPoseHeatmapHeadConfiguration,
        weights: BTreeMap<String, Tensor>,
        cancellation: &CancellationToken,
    ) -> Result<Self, SdPoseModelError> {
        if configuration.is_source_exact() {
            return Err(SdPoseModelError::InvalidConfiguration);
        }
        Self::checked(configuration, weights, cancellation)
    }

    fn checked(
        configuration: SdPoseHeatmapHeadConfiguration,
        candidate_weights: BTreeMap<String, Tensor>,
        cancellation: &CancellationToken,
    ) -> Result<Self, SdPoseModelError> {
        cancellation.check()?;
        configuration.validate()?;
        let manifest = sdpose_heatmap_head_weight_manifest(&configuration)?;
        let expected = manifest
            .iter()
            .map(|specification| specification.key.as_str())
            .collect::<BTreeSet<_>>();
        let actual = candidate_weights
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if expected != actual {
            return Err(SdPoseModelError::WeightKeys {
                missing: expected
                    .difference(&actual)
                    .map(|key| (*key).to_owned())
                    .collect(),
                unexpected: actual
                    .difference(&expected)
                    .map(|key| (*key).to_owned())
                    .collect(),
            });
        }
        let first = manifest
            .first()
            .and_then(|specification| candidate_weights.get(&specification.key))
            .ok_or(SdPoseModelError::InvalidConfiguration)?;
        let dtype = first.descriptor().dtype();
        let stream = first.descriptor().stream();
        if !matches!(dtype, DType::F32 | DType::F16 | DType::Bf16) {
            return Err(SdPoseModelError::InvalidConfiguration);
        }
        for specification in &manifest {
            cancellation.check()?;
            let tensor = candidate_weights
                .get(&specification.key)
                .ok_or(SdPoseModelError::InvalidConfiguration)?;
            let descriptor = tensor.descriptor();
            if descriptor.shape() != specification.shape
                || descriptor.dtype() != dtype
                || descriptor.device() != DeviceId::CPU
                || descriptor.stream() != stream
            {
                return Err(SdPoseModelError::WeightShape {
                    key: specification.key.clone(),
                    expected: specification.shape.clone(),
                    actual: descriptor.shape().to_vec(),
                });
            }
            require_finite_heatmap_tensor(tensor, cancellation)?;
        }
        let semantic_state_digest_sha256 =
            sdpose_heatmap_head_semantic_digest(&configuration, &candidate_weights, cancellation)?;
        Ok(Self {
            configuration,
            weights: candidate_weights,
            dtype,
            stream,
            semantic_state_digest_sha256,
        })
    }

    pub fn configuration(&self) -> &SdPoseHeatmapHeadConfiguration {
        &self.configuration
    }

    pub const fn execution_dtype(&self) -> DType {
        self.dtype
    }

    pub const fn execution_stream(&self) -> StreamId {
        self.stream
    }

    pub fn semantic_state_digest_sha256(&self) -> &str {
        &self.semantic_state_digest_sha256
    }

    pub fn validate(&self, cancellation: &CancellationToken) -> Result<(), SdPoseModelError> {
        cancellation.check()?;
        self.configuration.validate()?;
        let expected = sdpose_heatmap_head_weight_manifest(&self.configuration)?;
        if expected.len() != self.weights.len() {
            return Err(SdPoseModelError::InvalidConfiguration);
        }
        for specification in expected {
            cancellation.check()?;
            let tensor = self
                .weights
                .get(&specification.key)
                .ok_or(SdPoseModelError::InvalidConfiguration)?;
            if tensor.descriptor().shape() != specification.shape
                || tensor.descriptor().dtype() != self.dtype
                || tensor.descriptor().device() != DeviceId::CPU
                || tensor.descriptor().stream() != self.stream
            {
                return Err(SdPoseModelError::InvalidConfiguration);
            }
        }
        if self.semantic_state_digest_sha256
            != sdpose_heatmap_head_semantic_digest(
                &self.configuration,
                &self.weights,
                cancellation,
            )?
        {
            return Err(SdPoseModelError::InvalidConfiguration);
        }
        self.resident_tensor_allocations()?;
        cancellation.check()?;
        Ok(())
    }

    pub fn forward(
        &self,
        backend: &CpuBackend,
        captured_feature: &Tensor,
        context: &ExecutionContext<'_>,
    ) -> Result<Tensor, SdPoseModelError> {
        context.check()?;
        let descriptor = captured_feature.descriptor();
        let expected_channels = u64::try_from(self.configuration.input_channels)
            .map_err(|_| SdPoseModelError::Overflow("heatmap input channels"))?;
        if descriptor.rank() != 4
            || descriptor.shape().first() == Some(&0)
            || descriptor.shape().get(1) != Some(&expected_channels)
            || descriptor.shape().get(2).is_none_or(|value| *value <= 1)
            || descriptor.shape().get(3).is_none_or(|value| *value <= 1)
            || descriptor.dtype() != self.dtype
            || descriptor.device() != DeviceId::CPU
            || descriptor.stream() != self.stream
            || descriptor.stream() != context.stream
            || !descriptor.is_contiguous()?
        {
            return Err(SdPoseModelError::ComponentMismatch);
        }

        let deconvolution = self.convolution_module(
            "sdpose.heatmap-head.deconvolution",
            "native.heatmap_head.deconv_layers.0.weight",
            None,
            self.configuration.input_channels,
            self.configuration.hidden_channels,
            [4, 4],
            [2, 2],
            [1, 1],
            true,
        )?;
        let hidden = deconvolution.forward_dense_inference_with_context(
            backend,
            captured_feature,
            context,
        )?;
        let normalization = NativeModule::instance_norm_2d(
            "sdpose.heatmap-head.deconvolution-normalization",
            self.configuration.hidden_channels,
            1.0e-5,
            false,
            false,
        )?;
        let hidden =
            normalization.forward_dense_inference_with_context(backend, &hidden, context)?;
        let activation = NativeModule::silu("sdpose.heatmap-head.deconvolution-activation")?;
        let hidden = activation.forward_dense_inference_with_context(backend, &hidden, context)?;

        let convolution = self.convolution_module(
            "sdpose.heatmap-head.convolution",
            "native.heatmap_head.conv_layers.0.weight",
            Some("native.heatmap_head.conv_layers.0.bias"),
            self.configuration.hidden_channels,
            self.configuration.hidden_channels,
            [1, 1],
            [1, 1],
            [0, 0],
            false,
        )?;
        let hidden = convolution.forward_dense_inference_with_context(backend, &hidden, context)?;
        let normalization = NativeModule::instance_norm_2d(
            "sdpose.heatmap-head.convolution-normalization",
            self.configuration.hidden_channels,
            1.0e-5,
            false,
            false,
        )?;
        let hidden =
            normalization.forward_dense_inference_with_context(backend, &hidden, context)?;
        let activation = NativeModule::silu("sdpose.heatmap-head.convolution-activation")?;
        let hidden = activation.forward_dense_inference_with_context(backend, &hidden, context)?;

        let output = self.convolution_module(
            "sdpose.heatmap-head.final",
            "native.heatmap_head.final_layer.weight",
            Some("native.heatmap_head.final_layer.bias"),
            self.configuration.hidden_channels,
            self.configuration.output_channels,
            [1, 1],
            [1, 1],
            [0, 0],
            false,
        )?;
        let heatmaps = output.forward_dense_inference_with_context(backend, &hidden, context)?;
        context.check()?;
        Ok(heatmaps)
    }

    #[allow(clippy::too_many_arguments)]
    fn convolution_module(
        &self,
        name: &str,
        weight_key: &str,
        bias_key: Option<&str>,
        input_channels: usize,
        output_channels: usize,
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        transposed: bool,
    ) -> Result<NativeModule, SdPoseModelError> {
        let geometry = ConvolutionGeometry::new(
            2,
            stride.to_vec(),
            padding.to_vec(),
            vec![1, 1],
            1,
            transposed,
            vec![0, 0],
        )?;
        let mut module = NativeModule::convolution(
            name,
            input_channels,
            output_channels,
            kernel.to_vec(),
            bias_key.is_some(),
            geometry,
            false,
        )?;
        let weight = self
            .weights
            .get(weight_key)
            .ok_or(SdPoseModelError::InvalidConfiguration)?
            .clone();
        let bias = bias_key
            .map(|key| {
                self.weights
                    .get(key)
                    .cloned()
                    .ok_or(SdPoseModelError::InvalidConfiguration)
            })
            .transpose()?;
        module.load_dense_parameters(weight, bias)?;
        Ok(module)
    }

    pub fn resident_tensor_allocations(&self) -> Result<Vec<(StorageId, u64)>, SdPoseModelError> {
        checked_sdpose_storage_union(
            self.weights
                .values()
                .map(|tensor| (tensor.storage_id(), tensor.storage_byte_len())),
        )
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, SdPoseModelError> {
        let entries = self
            .weights
            .len()
            .checked_mul(std::mem::size_of::<(String, Tensor)>())
            .ok_or(SdPoseModelError::Overflow("heatmap retained entries"))?;
        let keys = self.weights.keys().try_fold(0usize, |total, key| {
            total
                .checked_add(key.capacity())
                .ok_or(SdPoseModelError::Overflow("heatmap retained keys"))
        })?;
        let bytes = std::mem::size_of::<Self>()
            .checked_add(entries)
            .and_then(|bytes| bytes.checked_add(keys))
            .and_then(|bytes| bytes.checked_add(self.semantic_state_digest_sha256.capacity()))
            .ok_or(SdPoseModelError::Overflow("heatmap owner residency"))?;
        u64::try_from(bytes).map_err(|_| SdPoseModelError::Overflow("heatmap owner residency"))
    }

    pub fn resident_bytes(&self) -> Result<u64, SdPoseModelError> {
        self.resident_tensor_allocations()?.into_iter().try_fold(
            self.resident_owned_bytes()?,
            |total, (_, bytes)| {
                total
                    .checked_add(bytes)
                    .ok_or(SdPoseModelError::Overflow("heatmap total residency"))
            },
        )
    }
}

impl NativeSdPoseModel {
    pub fn from_mapped_weights(
        mapped: &MappedModelWeights,
        cancellation: &CancellationToken,
    ) -> Result<Self, SdPoseModelError> {
        cancellation.check()?;
        let denoiser = NativeSdPoseSd2Denoiser::from_mapped_weights(mapped, cancellation)?;
        let heatmap_head = NativeSdPoseHeatmapHead::from_mapped_weights(mapped, cancellation)?;
        Self::checked(
            mapped.base_artifact_digest().to_owned(),
            denoiser,
            heatmap_head,
            true,
            cancellation,
        )
    }

    #[doc(hidden)]
    pub fn from_reduced_fixture(
        artifact_sha256: String,
        denoiser: NativeSdPoseSd2Denoiser,
        heatmap_head: NativeSdPoseHeatmapHead,
        cancellation: &CancellationToken,
    ) -> Result<Self, SdPoseModelError> {
        Self::checked(artifact_sha256, denoiser, heatmap_head, false, cancellation)
    }

    fn checked(
        artifact_sha256: String,
        denoiser: NativeSdPoseSd2Denoiser,
        heatmap_head: NativeSdPoseHeatmapHead,
        require_source_exact: bool,
        cancellation: &CancellationToken,
    ) -> Result<Self, SdPoseModelError> {
        cancellation.check()?;
        if !valid_sdpose_sha256(&artifact_sha256) {
            return Err(SdPoseModelError::InvalidArtifactIdentity);
        }
        denoiser.validate(cancellation)?;
        heatmap_head.validate(cancellation)?;
        let source_exact = denoiser.configuration().is_source_exact()
            && heatmap_head.configuration().is_source_exact();
        if require_source_exact && !source_exact {
            return Err(SdPoseModelError::ReducedProductionResource);
        }
        if denoiser.configuration().capture_channels()
            != heatmap_head.configuration().input_channels()
            || denoiser.execution_dtype() != heatmap_head.execution_dtype()
            || denoiser.execution_stream() != heatmap_head.execution_stream()
        {
            return Err(SdPoseModelError::ComponentMismatch);
        }
        let semantic_state_digest_sha256 =
            sdpose_model_semantic_digest(&artifact_sha256, &denoiser, &heatmap_head, cancellation)?;
        let model = Self {
            artifact_sha256,
            denoiser,
            heatmap_head,
            semantic_state_digest_sha256,
        };
        model.resident_tensor_allocations()?;
        cancellation.check()?;
        Ok(model)
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn denoiser(&self) -> &NativeSdPoseSd2Denoiser {
        &self.denoiser
    }

    pub fn heatmap_head(&self) -> &NativeSdPoseHeatmapHead {
        &self.heatmap_head
    }

    pub fn is_source_exact_profile(&self) -> bool {
        self.denoiser.configuration().is_source_exact()
            && self.heatmap_head.configuration().is_source_exact()
    }

    pub fn semantic_state_digest_sha256(&self) -> &str {
        &self.semantic_state_digest_sha256
    }

    pub fn validate(&self, cancellation: &CancellationToken) -> Result<(), SdPoseModelError> {
        cancellation.check()?;
        if !valid_sdpose_sha256(&self.artifact_sha256) {
            return Err(SdPoseModelError::InvalidArtifactIdentity);
        }
        self.denoiser.validate(cancellation)?;
        self.heatmap_head.validate(cancellation)?;
        if self.denoiser.configuration().capture_channels()
            != self.heatmap_head.configuration().input_channels()
            || self.denoiser.execution_dtype() != self.heatmap_head.execution_dtype()
            || self.denoiser.execution_stream() != self.heatmap_head.execution_stream()
        {
            return Err(SdPoseModelError::ComponentMismatch);
        }
        if self.semantic_state_digest_sha256
            != sdpose_model_semantic_digest(
                &self.artifact_sha256,
                &self.denoiser,
                &self.heatmap_head,
                cancellation,
            )?
        {
            return Err(SdPoseModelError::InvalidConfiguration);
        }
        self.resident_tensor_allocations()?;
        cancellation.check()?;
        Ok(())
    }

    pub fn resident_tensor_allocations(&self) -> Result<Vec<(StorageId, u64)>, SdPoseModelError> {
        checked_sdpose_storage_union(
            self.denoiser
                .resident_tensor_allocations()?
                .into_iter()
                .chain(self.heatmap_head.resident_tensor_allocations()?),
        )
    }

    pub fn resident_owned_bytes(&self) -> Result<u64, SdPoseModelError> {
        let denoiser_inline = u64::try_from(std::mem::size_of::<NativeSdPoseSd2Denoiser>())
            .map_err(|_| SdPoseModelError::Overflow("SDPose denoiser inline residency"))?;
        let head_inline = u64::try_from(std::mem::size_of::<NativeSdPoseHeatmapHead>())
            .map_err(|_| SdPoseModelError::Overflow("SDPose head inline residency"))?;
        let denoiser_owned = self
            .denoiser
            .resident_owned_bytes()?
            .checked_sub(denoiser_inline)
            .ok_or(SdPoseModelError::Overflow(
                "SDPose denoiser owner residency",
            ))?;
        let head_owned = self
            .heatmap_head
            .resident_owned_bytes()?
            .checked_sub(head_inline)
            .ok_or(SdPoseModelError::Overflow("SDPose head owner residency"))?;
        let artifact_capacity = u64::try_from(self.artifact_sha256.capacity())
            .map_err(|_| SdPoseModelError::Overflow("SDPose artifact residency"))?;
        let digest_capacity = u64::try_from(self.semantic_state_digest_sha256.capacity())
            .map_err(|_| SdPoseModelError::Overflow("SDPose digest residency"))?;
        let owned = u64::try_from(std::mem::size_of::<Self>())
            .map_err(|_| SdPoseModelError::Overflow("SDPose owner residency"))?
            .checked_add(denoiser_owned)
            .and_then(|bytes| bytes.checked_add(head_owned))
            .and_then(|bytes| bytes.checked_add(artifact_capacity))
            .and_then(|bytes| bytes.checked_add(digest_capacity))
            .ok_or(SdPoseModelError::Overflow("SDPose owner residency"))?;
        Ok(owned)
    }

    pub fn resident_bytes(&self) -> Result<u64, SdPoseModelError> {
        self.resident_tensor_allocations()?.into_iter().try_fold(
            self.resident_owned_bytes()?,
            |total, (_, bytes)| {
                total
                    .checked_add(bytes)
                    .ok_or(SdPoseModelError::Overflow("SDPose total residency"))
            },
        )
    }
}

fn valid_sdpose_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn checked_sdpose_storage_union(
    allocations: impl IntoIterator<Item = (StorageId, u64)>,
) -> Result<Vec<(StorageId, u64)>, SdPoseModelError> {
    let mut unique = BTreeMap::<u64, (StorageId, u64)>::new();
    for (storage_id, bytes) in allocations {
        if let Some((_, existing)) = unique.get(&storage_id.get()) {
            if *existing != bytes {
                return Err(SdPoseModelError::InconsistentStorage(storage_id));
            }
        } else {
            unique.insert(storage_id.get(), (storage_id, bytes));
        }
    }
    Ok(unique.into_values().collect())
}

fn sdpose_heatmap_head_semantic_digest(
    configuration: &SdPoseHeatmapHeadConfiguration,
    weights: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<String, SdPoseModelError> {
    cancellation.check()?;
    let mut digest = Sha256::new();
    digest.update(SDPOSE_HEATMAP_HEAD_SOURCE_DOMAIN);
    digest.update(SDPOSE_HEAD_SOURCE_SHA256.as_bytes());
    digest.update(SDPOSE_MODEL_DETECTION_SOURCE_SHA256.as_bytes());
    digest.update([u8::from(configuration.source_exact_profile)]);
    for value in [
        configuration.input_channels,
        configuration.hidden_channels,
        configuration.output_channels,
    ] {
        digest.update(
            u64::try_from(value)
                .map_err(|_| SdPoseModelError::Overflow("heatmap configuration digest"))?
                .to_le_bytes(),
        );
    }
    hash_sdpose_weight_map(&mut digest, weights, cancellation)?;
    Ok(format!("{:x}", digest.finalize()))
}

fn sdpose_model_semantic_digest(
    artifact_sha256: &str,
    denoiser: &NativeSdPoseSd2Denoiser,
    heatmap_head: &NativeSdPoseHeatmapHead,
    cancellation: &CancellationToken,
) -> Result<String, SdPoseModelError> {
    cancellation.check()?;
    let mut digest = Sha256::new();
    digest.update(SDPOSE_MODEL_SOURCE_DOMAIN);
    for field in [
        artifact_sha256,
        generated_lotusd_sim_model_0106::MODEL_FAMILY_FEATURE_ID,
        generated_lotusd_sim_model_0106::MODEL_FAMILY_IDENTIFIER,
        denoiser.semantic_state_digest_sha256(),
        heatmap_head.semantic_state_digest_sha256(),
        SDPOSE_HEAD_SOURCE_SHA256,
        SDPOSE_MODEL_DETECTION_SOURCE_SHA256,
    ] {
        digest.update(
            u64::try_from(field.len())
                .map_err(|_| SdPoseModelError::Overflow("SDPose model digest"))?
                .to_le_bytes(),
        );
        digest.update(field.as_bytes());
    }
    cancellation.check()?;
    Ok(format!("{:x}", digest.finalize()))
}

fn hash_sdpose_weight_map(
    digest: &mut Sha256,
    weights: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<(), SdPoseModelError> {
    for (index, (key, tensor)) in weights.iter().enumerate() {
        if index.is_multiple_of(8) {
            cancellation.check()?;
        }
        digest.update(
            u64::try_from(key.len())
                .map_err(|_| SdPoseModelError::Overflow("heatmap key digest"))?
                .to_le_bytes(),
        );
        digest.update(key.as_bytes());
        digest.update(
            u64::try_from(tensor.descriptor().shape().len())
                .map_err(|_| SdPoseModelError::Overflow("heatmap shape digest"))?
                .to_le_bytes(),
        );
        for dimension in tensor.descriptor().shape() {
            digest.update(dimension.to_le_bytes());
        }
        digest.update([sdpose_sd2_dtype_tag(tensor.descriptor().dtype())?]);
        let bytes = tensor.contiguous_bytes()?;
        digest.update(
            u64::try_from(bytes.len())
                .map_err(|_| SdPoseModelError::Overflow("heatmap bytes digest"))?
                .to_le_bytes(),
        );
        digest.update(bytes);
    }
    cancellation.check()?;
    Ok(())
}

fn require_finite_heatmap_tensor(
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<(), SdPoseModelError> {
    let count = tensor.descriptor().element_count()?;
    for index in 0..count {
        if index.is_multiple_of(1_024) {
            cancellation.check()?;
        }
        let value = tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.linear_element_bytes(index)?)?;
        if matches!(value, DecodedScalar::Real(value) if !value.is_finite())
            || matches!(value, DecodedScalar::Complex { real, imaginary } if !real.is_finite() || !imaginary.is_finite())
        {
            return Err(SdPoseModelError::NonFinite);
        }
    }
    Ok(())
}

fn sdpose_sd2_semantic_digest(
    configuration: &SdPoseSd2Configuration,
    weights: &BTreeMap<String, Tensor>,
    cancellation: &CancellationToken,
) -> Result<String, SdPoseSd2Error> {
    cancellation.check()?;
    let mut digest = Sha256::new();
    digest.update(SDPOSE_SD2_SOURCE_DOMAIN);
    for source in [
        OPENAI_MODEL_SOURCE_SHA256,
        ATTENTION_SOURCE_SHA256,
        MODEL_BASE_SOURCE_SHA256,
        SUPPORTED_MODELS_SOURCE_SHA256,
    ] {
        digest.update(
            u64::try_from(source.len())
                .map_err(|_| SdPoseSd2Error::Overflow("source digest length"))?
                .to_le_bytes(),
        );
        digest.update(source.as_bytes());
    }
    digest.update([u8::from(configuration.source_exact_profile)]);
    for value in [
        configuration.model_channels,
        configuration.context_dimension,
        configuration.attention_head_channels,
        configuration.normalization_groups,
        configuration.latent_height,
        configuration.latent_width,
    ] {
        digest.update(
            u64::try_from(value)
                .map_err(|_| SdPoseSd2Error::Overflow("configuration digest"))?
                .to_le_bytes(),
        );
    }
    for (index, (key, tensor)) in weights.iter().enumerate() {
        if index.is_multiple_of(8) {
            cancellation.check()?;
        }
        digest.update(
            u64::try_from(key.len())
                .map_err(|_| SdPoseSd2Error::Overflow("weight key digest"))?
                .to_le_bytes(),
        );
        digest.update(key.as_bytes());
        digest.update(
            u64::try_from(tensor.descriptor().shape().len())
                .map_err(|_| SdPoseSd2Error::Overflow("weight shape digest"))?
                .to_le_bytes(),
        );
        for dimension in tensor.descriptor().shape() {
            digest.update(dimension.to_le_bytes());
        }
        digest.update([sdpose_sd2_dtype_tag(tensor.descriptor().dtype())?]);
        let bytes = tensor.contiguous_bytes()?;
        digest.update(
            u64::try_from(bytes.len())
                .map_err(|_| SdPoseSd2Error::Overflow("weight bytes digest"))?
                .to_le_bytes(),
        );
        digest.update(bytes);
    }
    cancellation.check()?;
    Ok(format!("{:x}", digest.finalize()))
}

fn is_sd2_unet_key(key: &str) -> bool {
    [
        "native.input_blocks.",
        "native.time_embed.",
        "native.label_emb.",
        "native.middle_block.",
        "native.output_blocks.",
        "native.out.",
    ]
    .iter()
    .any(|prefix| key.starts_with(prefix))
}

fn sdpose_sd2_dtype_tag(dtype: DType) -> Result<u8, SdPoseSd2Error> {
    match dtype {
        DType::F32 => Ok(1),
        DType::F16 => Ok(2),
        DType::Bf16 => Ok(3),
        _ => Err(SdPoseSd2Error::InvalidConfiguration),
    }
}

fn require_finite_tensor(
    tensor: &Tensor,
    cancellation: &CancellationToken,
) -> Result<(), SdPoseSd2Error> {
    let count = tensor.descriptor().element_count()?;
    for index in 0..count {
        if index.is_multiple_of(1_024) {
            cancellation.check()?;
        }
        let value = tensor
            .descriptor()
            .dtype()
            .decode_scalar(tensor.linear_element_bytes(index)?)?;
        let finite = match value {
            DecodedScalar::Real(value) => value.is_finite(),
            DecodedScalar::Signed(_) | DecodedScalar::Unsigned(_) | DecodedScalar::Boolean(_) => {
                true
            }
            DecodedScalar::Complex { real, imaginary } => real.is_finite() && imaginary.is_finite(),
        };
        if !finite {
            return Err(SdPoseSd2Error::NonFinite);
        }
    }
    Ok(())
}

fn require_forward_inputs(
    configuration: &SdPoseSd2Configuration,
    dtype: DType,
    stream: StreamId,
    latent: &Tensor,
    timesteps: &[f32],
    conditioning: &Tensor,
    adm: &Tensor,
) -> Result<usize, SdPoseSd2Error> {
    for tensor in [latent, conditioning, adm] {
        if tensor.descriptor().dtype() != dtype
            || tensor.descriptor().device() != DeviceId::CPU
            || tensor.descriptor().stream() != stream
        {
            return Err(SdPoseSd2Error::StreamMismatch);
        }
    }
    let latent_shape = require_rank_four(latent, "latent")?;
    let conditioning_shape = require_rank_three(conditioning, "conditioning")?;
    let adm_shape = adm.descriptor().shape();
    let batch =
        usize::try_from(latent_shape[0]).map_err(|_| SdPoseSd2Error::Overflow("input batch"))?;
    let expected_height = u64::try_from(configuration.latent_height)
        .map_err(|_| SdPoseSd2Error::Overflow("latent height"))?;
    let expected_width = u64::try_from(configuration.latent_width)
        .map_err(|_| SdPoseSd2Error::Overflow("latent width"))?;
    let expected_context = u64::try_from(configuration.context_dimension)
        .map_err(|_| SdPoseSd2Error::Overflow("context dimension"))?;
    if batch == 0
        || latent_shape[1..] != [4, expected_height, expected_width]
        || timesteps.len() != batch
        || timesteps.iter().any(|value| !value.is_finite())
        || conditioning_shape[0] != latent_shape[0]
        || conditioning_shape[1] == 0
        || conditioning_shape[2] != expected_context
        || adm_shape != [latent_shape[0], 4]
    {
        return Err(SdPoseSd2Error::InputShape {
            name: "forward request",
            actual: latent.descriptor().shape().to_vec(),
        });
    }
    Ok(batch)
}

fn require_rank_four(tensor: &Tensor, name: &'static str) -> Result<[u64; 4], SdPoseSd2Error> {
    tensor
        .descriptor()
        .shape()
        .try_into()
        .map_err(|_| SdPoseSd2Error::InputShape {
            name,
            actual: tensor.descriptor().shape().to_vec(),
        })
}

fn require_rank_three(tensor: &Tensor, name: &'static str) -> Result<[u64; 3], SdPoseSd2Error> {
    tensor
        .descriptor()
        .shape()
        .try_into()
        .map_err(|_| SdPoseSd2Error::InputShape {
            name,
            actual: tensor.descriptor().shape().to_vec(),
        })
}

fn require_same_target(left: &Tensor, right: &Tensor) -> Result<(), SdPoseSd2Error> {
    if left.descriptor().dtype() != right.descriptor().dtype()
        || left.descriptor().device() != right.descriptor().device()
        || left.descriptor().stream() != right.descriptor().stream()
    {
        return Err(SdPoseSd2Error::StreamMismatch);
    }
    Ok(())
}

fn add_tensors(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SdPoseSd2Error> {
    require_same_target(left, right)?;
    if left.descriptor().shape() != right.descriptor().shape() {
        return Err(SdPoseSd2Error::InputShape {
            name: "tensor add",
            actual: right.descriptor().shape().to_vec(),
        });
    }
    let left_values = tensor_to_values(backend, left, context)?;
    let right_values = tensor_to_values(backend, right, context)?;
    let mut output = backend.workspace_vec(context, left_values.len())?;
    for (index, (left_value, right_value)) in
        left_values.iter().zip(right_values.iter()).enumerate()
    {
        if index.is_multiple_of(1_024) {
            context.cancellation.check()?;
        }
        output.try_push(left_value + right_value)?;
    }
    Ok(tensor_from_values(
        backend,
        left.descriptor().shape(),
        &output,
        left.descriptor().dtype(),
        left.descriptor().device(),
        context,
    )?)
}

fn concat_channel_tensors(
    backend: &CpuBackend,
    left: &Tensor,
    right: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SdPoseSd2Error> {
    require_same_target(left, right)?;
    let left_shape = require_rank_four(left, "concat left")?;
    let right_shape = require_rank_four(right, "concat right")?;
    if left_shape[0] != right_shape[0] || left_shape[2..] != right_shape[2..] {
        return Err(SdPoseSd2Error::InputShape {
            name: "concat right",
            actual: right_shape.to_vec(),
        });
    }
    let batch =
        usize::try_from(left_shape[0]).map_err(|_| SdPoseSd2Error::Overflow("concat batch"))?;
    let left_channels = usize::try_from(left_shape[1])
        .map_err(|_| SdPoseSd2Error::Overflow("concat left channels"))?;
    let right_channels = usize::try_from(right_shape[1])
        .map_err(|_| SdPoseSd2Error::Overflow("concat right channels"))?;
    let spatial = usize::try_from(left_shape[2])
        .map_err(|_| SdPoseSd2Error::Overflow("concat height"))?
        .checked_mul(
            usize::try_from(left_shape[3]).map_err(|_| SdPoseSd2Error::Overflow("concat width"))?,
        )
        .ok_or(SdPoseSd2Error::Overflow("concat spatial size"))?;
    let left_values = tensor_to_values(backend, left, context)?;
    let right_values = tensor_to_values(backend, right, context)?;
    let output_channels = left_channels
        .checked_add(right_channels)
        .ok_or(SdPoseSd2Error::Overflow("concat output channels"))?;
    let output_count = batch
        .checked_mul(output_channels)
        .and_then(|value| value.checked_mul(spatial))
        .ok_or(SdPoseSd2Error::Overflow("concat output"))?;
    let mut output = backend.workspace_vec(context, output_count)?;
    for batch_index in 0..batch {
        context.cancellation.check()?;
        let left_start = batch_index * left_channels * spatial;
        for value in &left_values[left_start..left_start + left_channels * spatial] {
            output.try_push(*value)?;
        }
        let right_start = batch_index * right_channels * spatial;
        for value in &right_values[right_start..right_start + right_channels * spatial] {
            output.try_push(*value)?;
        }
    }
    Ok(tensor_from_values(
        backend,
        &[
            left_shape[0],
            u64::try_from(output_channels)
                .map_err(|_| SdPoseSd2Error::Overflow("concat output channels"))?,
            left_shape[2],
            left_shape[3],
        ],
        &output,
        left.descriptor().dtype(),
        left.descriptor().device(),
        context,
    )?)
}

fn nearest_upsample_tensor_2x(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SdPoseSd2Error> {
    let [batch, channels, height, width] = require_rank_four(input, "nearest upsample")?;
    let batch = usize::try_from(batch).map_err(|_| SdPoseSd2Error::Overflow("upsample batch"))?;
    let channels =
        usize::try_from(channels).map_err(|_| SdPoseSd2Error::Overflow("upsample channels"))?;
    let height =
        usize::try_from(height).map_err(|_| SdPoseSd2Error::Overflow("upsample height"))?;
    let width = usize::try_from(width).map_err(|_| SdPoseSd2Error::Overflow("upsample width"))?;
    let output_height = height
        .checked_mul(2)
        .ok_or(SdPoseSd2Error::Overflow("upsample height"))?;
    let output_width = width
        .checked_mul(2)
        .ok_or(SdPoseSd2Error::Overflow("upsample width"))?;
    let source = tensor_to_values(backend, input, context)?;
    let output_count = batch
        .checked_mul(channels)
        .and_then(|value| value.checked_mul(output_height))
        .and_then(|value| value.checked_mul(output_width))
        .ok_or(SdPoseSd2Error::Overflow("upsample output"))?;
    let mut output = backend.workspace_vec(context, output_count)?;
    for batch_index in 0..batch {
        for channel in 0..channels {
            context.cancellation.check()?;
            let source_offset = (batch_index * channels + channel) * height * width;
            for output_y in 0..output_height {
                let source_y = output_y / 2;
                for output_x in 0..output_width {
                    output.try_push(source[source_offset + source_y * width + output_x / 2])?;
                }
            }
        }
    }
    Ok(tensor_from_values(
        backend,
        &[
            u64::try_from(batch).map_err(|_| SdPoseSd2Error::Overflow("upsample batch"))?,
            u64::try_from(channels).map_err(|_| SdPoseSd2Error::Overflow("upsample channels"))?,
            u64::try_from(output_height)
                .map_err(|_| SdPoseSd2Error::Overflow("upsample height"))?,
            u64::try_from(output_width).map_err(|_| SdPoseSd2Error::Overflow("upsample width"))?,
        ],
        &output,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context,
    )?)
}

fn build_timestep_embedding(
    backend: &CpuBackend,
    timesteps: &[f32],
    width: usize,
    dtype: DType,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SdPoseSd2Error> {
    let half = width / 2;
    if half == 0 || width != half * 2 {
        return Err(SdPoseSd2Error::InvalidConfiguration);
    }
    let count = timesteps
        .len()
        .checked_mul(width)
        .ok_or(SdPoseSd2Error::Overflow("timestep embedding"))?;
    let mut values = backend.workspace_vec(context, count)?;
    for timestep in timesteps {
        context.check()?;
        for index in 0..half {
            let frequency = (-10_000_f32.ln() * index as f32 / half as f32).exp();
            values.try_push((timestep * frequency).cos())?;
        }
        for index in 0..half {
            let frequency = (-10_000_f32.ln() * index as f32 / half as f32).exp();
            values.try_push((timestep * frequency).sin())?;
        }
    }
    Ok(tensor_from_values(
        backend,
        &[
            u64::try_from(timesteps.len())
                .map_err(|_| SdPoseSd2Error::Overflow("timestep batch"))?,
            u64::try_from(width).map_err(|_| SdPoseSd2Error::Overflow("timestep width"))?,
        ],
        &values,
        dtype,
        DeviceId::CPU,
        context,
    )?)
}

fn copy_tensor(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SdPoseSd2Error> {
    context.check()?;
    let values = tensor_to_values(backend, input, context)?;
    let output = tensor_from_values(
        backend,
        input.descriptor().shape(),
        &values,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context,
    )?;
    context.check()?;
    Ok(output)
}

fn immutable_silu(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SdPoseSd2Error> {
    let module = NativeModule::silu("sdpose.silu")?;
    Ok(module.forward_dense_inference_with_context(backend, input, context)?)
}

fn add_embedding_bias(
    backend: &CpuBackend,
    input: &Tensor,
    embedding: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SdPoseSd2Error> {
    let [batch, channels, height, width] = require_rank_four(input, "embedding input")?;
    if embedding.descriptor().shape() != [batch, channels] {
        return Err(SdPoseSd2Error::InputShape {
            name: "embedding bias",
            actual: embedding.descriptor().shape().to_vec(),
        });
    }
    let batch = usize::try_from(batch).map_err(|_| SdPoseSd2Error::Overflow("embedding batch"))?;
    let channels =
        usize::try_from(channels).map_err(|_| SdPoseSd2Error::Overflow("embedding channels"))?;
    let spatial = usize::try_from(height)
        .map_err(|_| SdPoseSd2Error::Overflow("embedding height"))?
        .checked_mul(
            usize::try_from(width).map_err(|_| SdPoseSd2Error::Overflow("embedding width"))?,
        )
        .ok_or(SdPoseSd2Error::Overflow("embedding spatial size"))?;
    let input_values = tensor_to_values(backend, input, context)?;
    let embedding_values = tensor_to_values(backend, embedding, context)?;
    let mut output = backend.workspace_vec(context, input_values.len())?;
    for batch_index in 0..batch {
        context.check()?;
        for channel in 0..channels {
            let bias = embedding_values[batch_index * channels + channel];
            let offset = (batch_index * channels + channel) * spatial;
            for value in &input_values[offset..offset + spatial] {
                output.try_push(*value + bias)?;
            }
        }
    }
    Ok(tensor_from_values(
        backend,
        input.descriptor().shape(),
        &output,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context,
    )?)
}

fn nchw_to_tokens(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SdPoseSd2Error> {
    let [batch, channels, height, width] = require_rank_four(input, "NCHW tokens")?;
    let batch = usize::try_from(batch).map_err(|_| SdPoseSd2Error::Overflow("token batch"))?;
    let channels =
        usize::try_from(channels).map_err(|_| SdPoseSd2Error::Overflow("token channels"))?;
    let height = usize::try_from(height).map_err(|_| SdPoseSd2Error::Overflow("token height"))?;
    let width = usize::try_from(width).map_err(|_| SdPoseSd2Error::Overflow("token width"))?;
    let source = tensor_to_values(backend, input, context)?;
    let mut output = backend.workspace_vec(context, source.len())?;
    for _ in 0..source.len() {
        output.try_push(0.0)?;
    }
    for batch_index in 0..batch {
        context.check()?;
        for y in 0..height {
            for x in 0..width {
                for channel in 0..channels {
                    let source_index =
                        ((batch_index * channels + channel) * height + y) * width + x;
                    let output_index =
                        ((batch_index * height * width + y * width + x) * channels) + channel;
                    output[output_index] = source[source_index];
                }
            }
        }
    }
    Ok(tensor_from_values(
        backend,
        &[
            u64::try_from(batch).map_err(|_| SdPoseSd2Error::Overflow("token batch"))?,
            u64::try_from(
                height
                    .checked_mul(width)
                    .ok_or(SdPoseSd2Error::Overflow("token count"))?,
            )
            .map_err(|_| SdPoseSd2Error::Overflow("token count"))?,
            u64::try_from(channels).map_err(|_| SdPoseSd2Error::Overflow("token channels"))?,
        ],
        &output,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context,
    )?)
}

fn tokens_to_nchw(
    backend: &CpuBackend,
    input: &Tensor,
    height: u64,
    width: u64,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SdPoseSd2Error> {
    let [batch, tokens, channels] = require_rank_three(input, "tokens NCHW")?;
    if tokens
        != height
            .checked_mul(width)
            .ok_or(SdPoseSd2Error::Overflow("token geometry"))?
    {
        return Err(SdPoseSd2Error::InputShape {
            name: "tokens NCHW",
            actual: input.descriptor().shape().to_vec(),
        });
    }
    let batch = usize::try_from(batch).map_err(|_| SdPoseSd2Error::Overflow("token batch"))?;
    let channels =
        usize::try_from(channels).map_err(|_| SdPoseSd2Error::Overflow("token channels"))?;
    let height = usize::try_from(height).map_err(|_| SdPoseSd2Error::Overflow("token height"))?;
    let width = usize::try_from(width).map_err(|_| SdPoseSd2Error::Overflow("token width"))?;
    let source = tensor_to_values(backend, input, context)?;
    let mut output = backend.workspace_vec(context, source.len())?;
    for _ in 0..source.len() {
        output.try_push(0.0)?;
    }
    for batch_index in 0..batch {
        context.check()?;
        for y in 0..height {
            for x in 0..width {
                for channel in 0..channels {
                    let source_index =
                        ((batch_index * height * width + y * width + x) * channels) + channel;
                    let output_index =
                        ((batch_index * channels + channel) * height + y) * width + x;
                    output[output_index] = source[source_index];
                }
            }
        }
    }
    Ok(tensor_from_values(
        backend,
        &[
            u64::try_from(batch).map_err(|_| SdPoseSd2Error::Overflow("token batch"))?,
            u64::try_from(channels).map_err(|_| SdPoseSd2Error::Overflow("token channels"))?,
            u64::try_from(height).map_err(|_| SdPoseSd2Error::Overflow("token height"))?,
            u64::try_from(width).map_err(|_| SdPoseSd2Error::Overflow("token width"))?,
        ],
        &output,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context,
    )?)
}

fn geglu(
    backend: &CpuBackend,
    input: &Tensor,
    context: &ExecutionContext<'_>,
) -> Result<Tensor, SdPoseSd2Error> {
    let shape = input.descriptor().shape();
    let width = usize::try_from(*shape.last().ok_or(SdPoseSd2Error::InputShape {
        name: "GEGLU",
        actual: shape.to_vec(),
    })?)
    .map_err(|_| SdPoseSd2Error::Overflow("GEGLU width"))?;
    if width == 0 || !width.is_multiple_of(2) {
        return Err(SdPoseSd2Error::InputShape {
            name: "GEGLU",
            actual: shape.to_vec(),
        });
    }
    let half = width / 2;
    let source = tensor_to_values(backend, input, context)?;
    let rows = source.len() / width;
    let output_count = rows
        .checked_mul(half)
        .ok_or(SdPoseSd2Error::Overflow("GEGLU output"))?;
    let mut left = backend.workspace_vec(context, output_count)?;
    let mut gate = backend.workspace_vec(context, output_count)?;
    for row in source.chunks_exact(width) {
        context.check()?;
        for value in &row[..half] {
            left.try_push(*value)?;
        }
        for value in &row[half..] {
            gate.try_push(*value)?;
        }
    }
    let mut output_shape = shape.to_vec();
    *output_shape.last_mut().ok_or(SdPoseSd2Error::InputShape {
        name: "GEGLU",
        actual: shape.to_vec(),
    })? = u64::try_from(half).map_err(|_| SdPoseSd2Error::Overflow("GEGLU width"))?;
    let gate = tensor_from_values(
        backend,
        &output_shape,
        &gate,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context,
    )?;
    let gelu = NativeModule::gelu("sdpose.geglu", GeluApproximation::None)?;
    let gate = gelu.forward_dense_inference_with_context(backend, &gate, context)?;
    let gate = tensor_to_values(backend, &gate, context)?;
    let mut output = backend.workspace_vec(context, output_count)?;
    for (index, (left, gate)) in left.iter().zip(gate.iter()).enumerate() {
        if index.is_multiple_of(1_024) {
            context.check()?;
        }
        output.try_push(left * gate)?;
    }
    Ok(tensor_from_values(
        backend,
        &output_shape,
        &output,
        input.descriptor().dtype(),
        input.descriptor().device(),
        context,
    )?)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SdPoseRawKeypoint {
    x: f32,
    y: f32,
    score: f32,
}

impl SdPoseRawKeypoint {
    pub fn checked(x: f32, y: f32, score: f32) -> Result<Self, SdPoseProjectionError> {
        if !x.is_finite() || !y.is_finite() || !score.is_finite() {
            return Err(SdPoseProjectionError::NonFiniteInput);
        }
        Ok(Self { x, y, score })
    }

    pub const fn x(self) -> f32 {
        self.x
    }

    pub const fn y(self) -> f32 {
        self.y
    }

    pub const fn score(self) -> f32 {
        self.score
    }
}

#[derive(Debug, Error)]
pub enum SdPoseProjectionError {
    #[error("SDPose heatmaps must have shape [batch, 133, 256, 192]")]
    InvalidHeatmapShape,
    #[error("SDPose projection received a non-finite value")]
    NonFiniteInput,
    #[error("SDPose DARK refinement encountered a singular Hessian")]
    SingularHessian,
    #[error("SDPose projection allocation failed: {0}")]
    AllocationFailed(String),
    #[error(transparent)]
    Cancellation(#[from] CancellationError),
    #[error(transparent)]
    Tensor(#[from] TensorError),
    #[error(transparent)]
    Media(#[from] sim_media::NativeMediaPayloadError),
}

pub fn decode_sdpose_heatmaps(
    heatmaps: &[f32],
    batch_size: usize,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Vec<SdPoseRawKeypoint>>, SdPoseProjectionError> {
    context.check()?;
    let plane_length = SDPOSE_HEATMAP_HEIGHT
        .checked_mul(SDPOSE_HEATMAP_WIDTH)
        .ok_or(SdPoseProjectionError::InvalidHeatmapShape)?;
    let expected = batch_size
        .checked_mul(SDPOSE_HEATMAP_CHANNELS)
        .and_then(|value| value.checked_mul(plane_length))
        .ok_or(SdPoseProjectionError::InvalidHeatmapShape)?;
    if batch_size == 0 || heatmaps.len() != expected {
        return Err(SdPoseProjectionError::InvalidHeatmapShape);
    }
    if heatmaps.iter().any(|value| !value.is_finite()) {
        return Err(SdPoseProjectionError::NonFiniteInput);
    }

    let mut batches = Vec::new();
    batches
        .try_reserve_exact(batch_size)
        .map_err(|error| SdPoseProjectionError::AllocationFailed(error.to_string()))?;
    for batch_index in 0..batch_size {
        context.check()?;
        let mut points = Vec::new();
        points
            .try_reserve_exact(SDPOSE_HEATMAP_CHANNELS)
            .map_err(|error| SdPoseProjectionError::AllocationFailed(error.to_string()))?;
        for channel in 0..SDPOSE_HEATMAP_CHANNELS {
            context.check()?;
            let plane_index = batch_index
                .checked_mul(SDPOSE_HEATMAP_CHANNELS)
                .and_then(|value| value.checked_add(channel))
                .and_then(|value| value.checked_mul(plane_length))
                .ok_or(SdPoseProjectionError::InvalidHeatmapShape)?;
            let plane_end = plane_index
                .checked_add(plane_length)
                .ok_or(SdPoseProjectionError::InvalidHeatmapShape)?;
            let plane = heatmaps
                .get(plane_index..plane_end)
                .ok_or(SdPoseProjectionError::InvalidHeatmapShape)?;
            points.push(decode_plane(plane, backend, context)?);
        }
        batches.push(points);
    }
    context.check()?;
    Ok(batches)
}

pub fn decode_sdpose_heatmap_tensor(
    heatmaps: &Tensor,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<Vec<Vec<SdPoseRawKeypoint>>, SdPoseProjectionError> {
    context.check()?;
    let descriptor = heatmaps.descriptor();
    let shape = descriptor.shape();
    let expected_channels = u64::try_from(SDPOSE_HEATMAP_CHANNELS)
        .map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?;
    let expected_height = u64::try_from(SDPOSE_HEATMAP_HEIGHT)
        .map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?;
    let expected_width = u64::try_from(SDPOSE_HEATMAP_WIDTH)
        .map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?;
    if shape.len() != 4
        || shape.first() == Some(&0)
        || shape.get(1) != Some(&expected_channels)
        || shape.get(2) != Some(&expected_height)
        || shape.get(3) != Some(&expected_width)
        || !matches!(descriptor.dtype(), DType::F16 | DType::Bf16 | DType::F32)
        || descriptor.device() != DeviceId::CPU
        || descriptor.stream() != context.stream
        || !descriptor.is_contiguous()?
    {
        return Err(SdPoseProjectionError::InvalidHeatmapShape);
    }
    let batch_size =
        usize::try_from(shape[0]).map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?;
    let plane_length = SDPOSE_HEATMAP_HEIGHT
        .checked_mul(SDPOSE_HEATMAP_WIDTH)
        .ok_or(SdPoseProjectionError::InvalidHeatmapShape)?;
    let mut batches = Vec::new();
    batches
        .try_reserve_exact(batch_size)
        .map_err(|error| SdPoseProjectionError::AllocationFailed(error.to_string()))?;
    for batch_index in 0..batch_size {
        context.check()?;
        let mut points = Vec::new();
        points
            .try_reserve_exact(SDPOSE_HEATMAP_CHANNELS)
            .map_err(|error| SdPoseProjectionError::AllocationFailed(error.to_string()))?;
        for channel in 0..SDPOSE_HEATMAP_CHANNELS {
            context.check()?;
            let plane_index = batch_index
                .checked_mul(SDPOSE_HEATMAP_CHANNELS)
                .and_then(|value| value.checked_add(channel))
                .and_then(|value| value.checked_mul(plane_length))
                .ok_or(SdPoseProjectionError::InvalidHeatmapShape)?;
            let point = {
                let mut plane = backend.workspace_vec(context, plane_length)?;
                for offset in 0..plane_length {
                    if offset % SDPOSE_HEATMAP_WIDTH == 0 {
                        context.check()?;
                    }
                    let linear_index = plane_index
                        .checked_add(offset)
                        .ok_or(SdPoseProjectionError::InvalidHeatmapShape)?;
                    let linear_index = u64::try_from(linear_index)
                        .map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?;
                    let value = match descriptor
                        .dtype()
                        .decode_scalar(heatmaps.linear_element_bytes(linear_index)?)?
                    {
                        DecodedScalar::Boolean(value) => f32::from(u8::from(value)),
                        DecodedScalar::Signed(value) => value as f32,
                        DecodedScalar::Unsigned(value) => value as f32,
                        DecodedScalar::Real(value) => value as f32,
                        DecodedScalar::Complex { .. } => {
                            return Err(SdPoseProjectionError::NonFiniteInput);
                        }
                    };
                    if !value.is_finite() {
                        return Err(SdPoseProjectionError::NonFiniteInput);
                    }
                    plane.try_push(value)?;
                }
                decode_plane(&plane, backend, context)?
            };
            points.push(point);
        }
        batches.push(points);
    }
    context.check()?;
    Ok(batches)
}

pub fn project_sdpose_heatmap_tensor(
    heatmaps: &Tensor,
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<Vec<NativePosePerson>, SdPoseProjectionError> {
    let decoded = decode_sdpose_heatmap_tensor(heatmaps, backend, context)?;
    let mut people = Vec::new();
    people
        .try_reserve_exact(decoded.len())
        .map_err(|error| SdPoseProjectionError::AllocationFailed(error.to_string()))?;
    for raw in decoded {
        context.check()?;
        people.push(project_sdpose_openpose_person(&raw)?);
    }
    context.check()?;
    Ok(people)
}

fn decode_plane(
    plane: &[f32],
    backend: &CpuBackend,
    context: &ExecutionContext<'_>,
) -> Result<SdPoseRawKeypoint, SdPoseProjectionError> {
    let (maximum_index, score) = plane.iter().copied().enumerate().fold(
        (0usize, f32::NEG_INFINITY),
        |current, candidate| {
            if candidate.1 > current.1 {
                candidate
            } else {
                current
            }
        },
    );
    let invalid = score <= 0.0;

    let maximum_y = maximum_index / SDPOSE_HEATMAP_WIDTH;
    let maximum_x = maximum_index % SDPOSE_HEATMAP_WIDTH;
    let radius =
        usize::try_from(GAUSSIAN_RADIUS).map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?;
    let padded_height = SDPOSE_HEATMAP_HEIGHT + 2 * radius;
    let padded_width = SDPOSE_HEATMAP_WIDTH + 2 * radius;
    let padded_length = padded_height
        .checked_mul(padded_width)
        .ok_or(SdPoseProjectionError::InvalidHeatmapShape)?;
    let mut horizontal = backend.workspace_vec::<f32>(context, padded_length)?;
    let mut blurred = backend.workspace_vec::<f32>(context, padded_length)?;
    for _ in 0..padded_length {
        horizontal.try_push(0.0)?;
        blurred.try_push(0.0)?;
    }

    for y in 0..SDPOSE_HEATMAP_HEIGHT {
        context.check()?;
        for x in 0..SDPOSE_HEATMAP_WIDTH {
            let source = y * SDPOSE_HEATMAP_WIDTH + x;
            let destination = (y + radius) * padded_width + x + radius;
            blurred[destination] = plane[source];
        }
    }
    let kernel = gaussian_kernel()?;
    for y in 0..padded_height {
        context.check()?;
        for x in 0..padded_width {
            let mut value = 0.0f32;
            for (kernel_index, weight) in kernel.iter().copied().enumerate() {
                let delta = isize::try_from(kernel_index)
                    .map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?
                    - GAUSSIAN_RADIUS;
                if let Some(source_x) = x
                    .checked_add_signed(delta)
                    .filter(|value| *value < padded_width)
                {
                    value += blurred[y * padded_width + source_x] * weight;
                }
            }
            horizontal[y * padded_width + x] = value;
        }
    }
    for y in 0..padded_height {
        context.check()?;
        for x in 0..padded_width {
            let mut value = 0.0f32;
            for (kernel_index, weight) in kernel.iter().copied().enumerate() {
                let delta = isize::try_from(kernel_index)
                    .map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?
                    - GAUSSIAN_RADIUS;
                if let Some(source_y) = y
                    .checked_add_signed(delta)
                    .filter(|value| *value < padded_height)
                {
                    value += horizontal[source_y * padded_width + x] * weight;
                }
            }
            blurred[y * padded_width + x] = value;
        }
    }

    let mut current_maximum = f32::NEG_INFINITY;
    for y in 0..SDPOSE_HEATMAP_HEIGHT {
        for x in 0..SDPOSE_HEATMAP_WIDTH {
            current_maximum =
                current_maximum.max(blurred[(y + radius) * padded_width + x + radius]);
        }
    }
    if current_maximum > 0.0 {
        let scale = score / current_maximum;
        for y in 0..SDPOSE_HEATMAP_HEIGHT {
            context.check()?;
            for x in 0..SDPOSE_HEATMAP_WIDTH {
                let index = (y + radius) * padded_width + x + radius;
                blurred[index] *= scale;
            }
        }
    }
    for y in 0..SDPOSE_HEATMAP_HEIGHT {
        context.check()?;
        for x in 0..SDPOSE_HEATMAP_WIDTH {
            let index = (y + radius) * padded_width + x + radius;
            blurred[index] = blurred[index].clamp(1.0e-3, 50.0).ln();
        }
    }

    let sample = |x: isize, y: isize| -> Result<f32, SdPoseProjectionError> {
        let clamped_x = x.clamp(
            0,
            isize::try_from(SDPOSE_HEATMAP_WIDTH - 1)
                .map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?,
        );
        let clamped_y = y.clamp(
            0,
            isize::try_from(SDPOSE_HEATMAP_HEIGHT - 1)
                .map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?,
        );
        let source_x = usize::try_from(clamped_x)
            .map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?
            + radius;
        let source_y = usize::try_from(clamped_y)
            .map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?
            + radius;
        Ok(blurred[source_y * padded_width + source_x])
    };
    let x = isize::try_from(maximum_x).map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?;
    let y = isize::try_from(maximum_y).map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?;
    let center = sample(x, y)?;
    let right = sample(x + 1, y)?;
    let left = sample(x - 1, y)?;
    let down = sample(x, y + 1)?;
    let up = sample(x, y - 1)?;
    let down_right = sample(x + 1, y + 1)?;
    let up_left = sample(x - 1, y - 1)?;
    let derivative_x = 0.5 * (right - left);
    let derivative_y = 0.5 * (down - up);
    let hessian_xx = right - 2.0 * center + left + f32::EPSILON;
    let hessian_yy = down - 2.0 * center + up + f32::EPSILON;
    let hessian_xy = 0.5 * (down_right - right - down + 2.0 * center - left - up + up_left);
    let correction = checked_hessian_correction(
        hessian_xx,
        hessian_xy,
        hessian_yy,
        derivative_x,
        derivative_y,
    )?;
    let maximum_x = f32::from(
        u16::try_from(maximum_x).map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?,
    );
    let maximum_y = f32::from(
        u16::try_from(maximum_y).map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?,
    );
    let refined_x = maximum_x - correction[0];
    let refined_y = maximum_y - correction[1];
    let heatmap_width = f32::from(
        u16::try_from(SDPOSE_HEATMAP_WIDTH)
            .map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?,
    );
    let heatmap_height = f32::from(
        u16::try_from(SDPOSE_HEATMAP_HEIGHT)
            .map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?,
    );
    let scale_x = (SDPOSE_INPUT_WIDTH - 1.0) / (heatmap_width - 1.0);
    let scale_y = (SDPOSE_INPUT_HEIGHT - 1.0) / (heatmap_height - 1.0);
    if invalid {
        SdPoseRawKeypoint::checked(-1.0, -1.0, score)
    } else {
        SdPoseRawKeypoint::checked(refined_x * scale_x, refined_y * scale_y, score)
    }
}

fn checked_hessian_correction(
    hessian_xx: f32,
    hessian_xy: f32,
    hessian_yy: f32,
    derivative_x: f32,
    derivative_y: f32,
) -> Result<[f32; 2], SdPoseProjectionError> {
    let determinant = hessian_xx * hessian_yy - hessian_xy * hessian_xy;
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(SdPoseProjectionError::SingularHessian);
    }
    let inverse_xx = hessian_yy / determinant;
    let inverse_xy = -hessian_xy / determinant;
    let inverse_yy = hessian_xx / determinant;
    Ok([
        inverse_xx * derivative_x + inverse_xy * derivative_y,
        inverse_xy * derivative_x + inverse_yy * derivative_y,
    ])
}

fn gaussian_kernel() -> Result<[f32; 11], SdPoseProjectionError> {
    let mut kernel = [0.0; 11];
    let mut total = 0.0f32;
    let radius = f32::from(
        i16::try_from(GAUSSIAN_RADIUS).map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?,
    );
    for (index, value) in kernel.iter_mut().enumerate() {
        let index = f32::from(
            u16::try_from(index).map_err(|_| SdPoseProjectionError::InvalidHeatmapShape)?,
        );
        let distance = index - radius;
        let weight = (-0.5 * (distance / GAUSSIAN_SIGMA).powi(2)).exp();
        *value = weight;
        total += weight;
    }
    for value in &mut kernel {
        *value /= total;
    }
    Ok(kernel)
}

pub fn project_sdpose_openpose_person(
    raw: &[SdPoseRawKeypoint],
) -> Result<NativePosePerson, SdPoseProjectionError> {
    if raw.len() != SDPOSE_HEATMAP_CHANNELS {
        return Err(SdPoseProjectionError::InvalidHeatmapShape);
    }
    let mut points = Vec::new();
    points
        .try_reserve_exact(OPENPOSE_KEYPOINTS)
        .map_err(|error| SdPoseProjectionError::AllocationFailed(error.to_string()))?;
    points.extend_from_slice(&raw[..17]);
    let left_shoulder = raw[5];
    let right_shoulder = raw[6];
    points.push(SdPoseRawKeypoint::checked(
        (left_shoulder.x + right_shoulder.x) * 0.5,
        (left_shoulder.y + right_shoulder.y) * 0.5,
        if left_shoulder.score > 0.3 && right_shoulder.score > 0.3 {
            left_shoulder.score.min(right_shoulder.score)
        } else {
            0.0
        },
    )?);
    points.extend_from_slice(&raw[17..]);
    let original = points.clone();
    for (&source, &destination) in MMPOSE_INDICES.iter().zip(OPENPOSE_INDICES.iter()) {
        points[destination] = original[source];
    }

    let convert = |point: SdPoseRawKeypoint| {
        NativePoseKeypoint::checked(point.x.into(), point.y.into(), point.score.into())
    };
    let collect = |slice: &[SdPoseRawKeypoint]| {
        slice
            .iter()
            .copied()
            .map(convert)
            .collect::<Result<Vec<_>, _>>()
    };
    let mut face = collect(&points[24..92])?;
    face.try_reserve_exact(2)
        .map_err(|error| SdPoseProjectionError::AllocationFailed(error.to_string()))?;
    face.push(convert(points[14])?);
    face.push(convert(points[15])?);
    Ok(NativePosePerson::checked(
        collect(&points[0..18])?,
        collect(&points[18..24])?,
        face,
        collect(&points[92..113])?,
        collect(&points[113..134])?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singular_hessian_is_typed() {
        assert!(matches!(
            checked_hessian_correction(f32::EPSILON, f32::EPSILON, f32::EPSILON, 1.0, 1.0),
            Err(SdPoseProjectionError::SingularHessian)
        ));
    }
}
