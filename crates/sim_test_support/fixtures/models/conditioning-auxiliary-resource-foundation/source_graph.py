import hashlib
import math
import struct
from fractions import Fraction


STYLE_WIDTH = 8
STYLE_CONTEXT = 6
STYLE_HEADS = 2
STYLE_TOKENS = 2
STYLE_LAYERS = 3
REDUX_INPUT = 4
REDUX_HIDDEN = 12
REDUX_OUTPUT = 6
INPUT_TOKENS = 2
PHOTO_HIDDEN = 4
PHOTO_INTERMEDIATE = 8
PHOTO_HEADS = 2
PHOTO_LAYERS = 24
PHOTO_IMAGE = 4
PHOTO_PATCH = 2
PHOTO_PROJECTION = 3
PHOTO_EXTRA_PROJECTION = 5
PHOTO_PROMPT = PHOTO_PROJECTION + PHOTO_EXTRA_PROJECTION
PHOTO_SEQUENCE = 4
PHOTO_IMAGES = 2
GLIGEN_KEY = 4
GLIGEN_QUERY = 64
GLIGEN_VISUAL_TOKENS = 2
GLIGEN_MAX_OBJECTS = 30
GLIGEN_POSITION_WIDTH = 64
GLIGEN_POSITION_HIDDEN = 512
GLIGEN_FUSERS = [
    "input_blocks.2.transformer_blocks.0",
    "output_blocks.7.transformer_blocks.0",
]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value):
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def from_bits(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def fadd(left, right):
    return f32(f32(left) + f32(right))


def fmul(left, right):
    return f32(f32(left) * f32(right))


def fdiv(left, right):
    return f32(f32(left) / f32(right))


def round_ratio_ties_even(numerator, denominator):
    quotient, remainder = divmod(numerator, denominator)
    doubled_remainder = remainder * 2
    if doubled_remainder > denominator or (
        doubled_remainder == denominator and quotient & 1
    ):
        quotient += 1
    return quotient


def fraction_to_f32(value):
    if value == 0:
        return 0.0
    sign = 0x80000000 if value < 0 else 0
    value = abs(value)
    numerator = value.numerator
    denominator = value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1
    if exponent > 127:
        return from_bits(sign | 0x7F800000)
    if exponent >= -126:
        shift = 23 - exponent
        if shift >= 0:
            significand = round_ratio_ties_even(numerator << shift, denominator)
        else:
            significand = round_ratio_ties_even(numerator, denominator << -shift)
        if significand == 1 << 24:
            significand >>= 1
            exponent += 1
            if exponent > 127:
                return from_bits(sign | 0x7F800000)
        return from_bits(sign | ((exponent + 127) << 23) | (significand - (1 << 23)))
    significand = round_ratio_ties_even(numerator << 149, denominator)
    if significand >= 1 << 23:
        return from_bits(sign | (1 << 23))
    return from_bits(sign | significand)


def fmadd(left, right, addend):
    exact = Fraction(f32(left)) * Fraction(f32(right)) + Fraction(f32(addend))
    return fraction_to_f32(exact)


def sigmoid(value):
    exponential = f32(math.exp(f32(-value)))
    return fdiv(1.0, fadd(1.0, exponential))


def silu(value):
    exponential = f32(math.exp(f32(-value)))
    return fdiv(value, fadd(1.0, exponential))


def storage_bits(value, dtype):
    value = f32(value)
    if dtype == "float32":
        return bits(value)
    if dtype == "float16":
        return struct.unpack("<H", struct.pack("<e", value))[0]
    raw = bits(value)
    return ((raw + 0x7FFF + ((raw >> 16) & 1)) >> 16) & 0xFFFF


def project_storage(raw, dtype):
    if dtype == "float32":
        return from_bits(raw)
    if dtype == "float16":
        return f32(struct.unpack("<e", struct.pack("<H", raw))[0])
    return from_bits(raw << 16)


def product(shape):
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def style_manifest():
    definitions = [
        ("style_embedding", [1, STYLE_TOKENS, STYLE_WIDTH]),
        ("proj", [STYLE_WIDTH, STYLE_CONTEXT]),
    ]
    for layer in range(STYLE_LAYERS):
        prefix = f"transformer_layes.{layer}"
        definitions.extend([
            (f"{prefix}.attn.in_proj_weight", [STYLE_WIDTH * 3, STYLE_WIDTH]),
            (f"{prefix}.attn.in_proj_bias", [STYLE_WIDTH * 3]),
            (f"{prefix}.attn.out_proj.weight", [STYLE_WIDTH, STYLE_WIDTH]),
            (f"{prefix}.attn.out_proj.bias", [STYLE_WIDTH]),
            (f"{prefix}.ln_1.weight", [STYLE_WIDTH]),
            (f"{prefix}.ln_1.bias", [STYLE_WIDTH]),
            (f"{prefix}.mlp.c_fc.weight", [STYLE_WIDTH * 4, STYLE_WIDTH]),
            (f"{prefix}.mlp.c_fc.bias", [STYLE_WIDTH * 4]),
            (f"{prefix}.mlp.c_proj.weight", [STYLE_WIDTH, STYLE_WIDTH * 4]),
            (f"{prefix}.mlp.c_proj.bias", [STYLE_WIDTH]),
            (f"{prefix}.ln_2.weight", [STYLE_WIDTH]),
            (f"{prefix}.ln_2.bias", [STYLE_WIDTH]),
        ])
    definitions.extend([
        ("ln_post.weight", [STYLE_WIDTH]),
        ("ln_post.bias", [STYLE_WIDTH]),
        ("ln_pre.weight", [STYLE_WIDTH]),
        ("ln_pre.bias", [STYLE_WIDTH]),
    ])
    return definitions


def redux_manifest():
    return [
        ("redux_up.weight", [REDUX_HIDDEN, REDUX_INPUT]),
        ("redux_up.bias", [REDUX_HIDDEN]),
        ("redux_down.weight", [REDUX_OUTPUT, REDUX_HIDDEN]),
        ("redux_down.bias", [REDUX_OUTPUT]),
    ]


def photomaker_manifest():
    positions = (PHOTO_IMAGE // PHOTO_PATCH) ** 2 + 1
    definitions = [
        ("vision_model.embeddings.class_embedding", [PHOTO_HIDDEN]),
        ("vision_model.embeddings.patch_embedding.weight",
         [PHOTO_HIDDEN, 3, PHOTO_PATCH, PHOTO_PATCH]),
        ("vision_model.embeddings.position_embedding.weight", [positions, PHOTO_HIDDEN]),
        ("vision_model.pre_layrnorm.weight", [PHOTO_HIDDEN]),
        ("vision_model.pre_layrnorm.bias", [PHOTO_HIDDEN]),
    ]
    for layer in range(PHOTO_LAYERS):
        prefix = f"vision_model.encoder.layers.{layer}"
        definitions.extend([
            (f"{prefix}.layer_norm1.weight", [PHOTO_HIDDEN]),
            (f"{prefix}.layer_norm1.bias", [PHOTO_HIDDEN]),
            (f"{prefix}.self_attn.q_proj.weight", [PHOTO_HIDDEN, PHOTO_HIDDEN]),
            (f"{prefix}.self_attn.q_proj.bias", [PHOTO_HIDDEN]),
            (f"{prefix}.self_attn.k_proj.weight", [PHOTO_HIDDEN, PHOTO_HIDDEN]),
            (f"{prefix}.self_attn.k_proj.bias", [PHOTO_HIDDEN]),
            (f"{prefix}.self_attn.v_proj.weight", [PHOTO_HIDDEN, PHOTO_HIDDEN]),
            (f"{prefix}.self_attn.v_proj.bias", [PHOTO_HIDDEN]),
            (f"{prefix}.self_attn.out_proj.weight", [PHOTO_HIDDEN, PHOTO_HIDDEN]),
            (f"{prefix}.self_attn.out_proj.bias", [PHOTO_HIDDEN]),
            (f"{prefix}.layer_norm2.weight", [PHOTO_HIDDEN]),
            (f"{prefix}.layer_norm2.bias", [PHOTO_HIDDEN]),
            (f"{prefix}.mlp.fc1.weight", [PHOTO_INTERMEDIATE, PHOTO_HIDDEN]),
            (f"{prefix}.mlp.fc1.bias", [PHOTO_INTERMEDIATE]),
            (f"{prefix}.mlp.fc2.weight", [PHOTO_HIDDEN, PHOTO_INTERMEDIATE]),
            (f"{prefix}.mlp.fc2.bias", [PHOTO_HIDDEN]),
        ])
    definitions.extend([
        ("vision_model.post_layernorm.weight", [PHOTO_HIDDEN]),
        ("vision_model.post_layernorm.bias", [PHOTO_HIDDEN]),
        ("visual_projection.weight", [PHOTO_PROJECTION, PHOTO_HIDDEN]),
        ("visual_projection_2.weight", [PHOTO_EXTRA_PROJECTION, PHOTO_HIDDEN]),
    ])
    for prefix, width in [
        ("fuse_module.mlp1", PHOTO_PROMPT * 2),
        ("fuse_module.mlp2", PHOTO_PROMPT),
    ]:
        definitions.extend([
            (f"{prefix}.layernorm.weight", [width]),
            (f"{prefix}.layernorm.bias", [width]),
            (f"{prefix}.fc1.weight", [PHOTO_PROMPT, width]),
            (f"{prefix}.fc1.bias", [PHOTO_PROMPT]),
            (f"{prefix}.fc2.weight", [PHOTO_PROMPT, PHOTO_PROMPT]),
            (f"{prefix}.fc2.bias", [PHOTO_PROMPT]),
        ])
    definitions.extend([
        ("fuse_module.layer_norm.weight", [PHOTO_PROMPT]),
        ("fuse_module.layer_norm.bias", [PHOTO_PROMPT]),
    ])
    assert len(definitions) == 407
    return definitions


def gligen_manifest(key_dimension=GLIGEN_KEY, query_dimension=GLIGEN_QUERY, fusers=GLIGEN_FUSERS):
    position_input = key_dimension + GLIGEN_POSITION_WIDTH
    definitions = [
        ("position_net.null_positive_feature", [key_dimension]),
        ("position_net.null_position_feature", [GLIGEN_POSITION_WIDTH]),
        ("position_net.linears.0.weight", [GLIGEN_POSITION_HIDDEN, position_input]),
        ("position_net.linears.0.bias", [GLIGEN_POSITION_HIDDEN]),
        ("position_net.linears.2.weight", [GLIGEN_POSITION_HIDDEN, GLIGEN_POSITION_HIDDEN]),
        ("position_net.linears.2.bias", [GLIGEN_POSITION_HIDDEN]),
        ("position_net.linears.4.weight", [key_dimension, GLIGEN_POSITION_HIDDEN]),
        ("position_net.linears.4.bias", [key_dimension]),
    ]
    for namespace in fusers:
        prefix = f"{namespace}.fuser"
        definitions.extend([
            (f"{prefix}.alpha_attn", []),
            (f"{prefix}.alpha_dense", []),
            (f"{prefix}.linear.weight", [query_dimension, key_dimension]),
            (f"{prefix}.linear.bias", [query_dimension]),
            (f"{prefix}.attn.to_q.weight", [query_dimension, query_dimension]),
            (f"{prefix}.attn.to_k.weight", [query_dimension, query_dimension]),
            (f"{prefix}.attn.to_v.weight", [query_dimension, query_dimension]),
            (f"{prefix}.attn.to_out.0.weight", [query_dimension, query_dimension]),
            (f"{prefix}.attn.to_out.0.bias", [query_dimension]),
            (f"{prefix}.ff.net.0.proj.weight", [query_dimension * 2, query_dimension]),
            (f"{prefix}.ff.net.0.proj.bias", [query_dimension * 2]),
            (f"{prefix}.ff.net.2.weight", [query_dimension, query_dimension]),
            (f"{prefix}.ff.net.2.bias", [query_dimension]),
            (f"{prefix}.norm1.weight", [query_dimension]),
            (f"{prefix}.norm1.bias", [query_dimension]),
            (f"{prefix}.norm2.weight", [query_dimension]),
            (f"{prefix}.norm2.bias", [query_dimension]),
        ])
    return definitions


def gligen_value(key, index, shape):
    output = index // shape[-1] if len(shape) == 2 else index
    component = index % shape[-1] if len(shape) == 2 else index
    variant = 1 if "output_blocks" in key else 0
    if key == "position_net.null_positive_feature":
        return f32((index + 1) * 0.03125)
    if key == "position_net.null_position_feature":
        return f32(((index % 9) - 4) * 0.0078125)
    if key == "position_net.linears.0.weight":
        return f32(0.125 + (output % 3) * 0.015625) if component == output % shape[1] else 0.0
    if key == "position_net.linears.0.bias":
        return f32(((index % 7) - 3) * 0.00390625)
    if key == "position_net.linears.2.weight":
        return f32(0.5) if component == output else 0.0
    if key == "position_net.linears.2.bias":
        return f32(((index % 5) - 2) * 0.001953125)
    if key == "position_net.linears.4.weight":
        return f32(0.25 + output * 0.03125) if component == (output * 17 + 3) % shape[1] else 0.0
    if key == "position_net.linears.4.bias":
        return f32((index - 1) * 0.015625)
    suffix = key.split(".fuser.", 1)[1]
    if suffix == "alpha_attn":
        return f32(0.375 + variant * 0.0625)
    if suffix == "alpha_dense":
        return f32(-0.3125 + variant * 0.03125)
    if suffix == "linear.weight":
        return f32(0.1875 + (output % 4) * 0.015625) if component == output % shape[1] else 0.0
    if suffix == "linear.bias":
        return f32(((index % 5) - 2) * 0.0078125)
    if suffix == "attn.to_q.weight":
        return f32(0.125 + variant * 0.015625) if component == output else 0.0
    if suffix == "attn.to_k.weight":
        return f32(0.09375) if component == (output + 1) % shape[1] else 0.0
    if suffix == "attn.to_v.weight":
        return f32(0.25) if component == output else 0.0
    if suffix == "attn.to_out.0.weight":
        return f32(0.3125) if component == output else 0.0
    if suffix == "attn.to_out.0.bias":
        return f32(((index % 3) - 1) * 0.00390625)
    if suffix == "ff.net.0.proj.weight":
        return f32(0.21875 if output < shape[1] else 0.15625) if component == output % shape[1] else 0.0
    if suffix == "ff.net.0.proj.bias":
        return f32(((index % 7) - 3) * 0.002)
    if suffix == "ff.net.2.weight":
        return f32(0.28125) if component == output else 0.0
    if suffix == "ff.net.2.bias":
        return f32(((index % 4) - 1) * 0.0025)
    if suffix in ["norm1.weight", "norm2.weight"]:
        return f32(1.0 + (index % 4) * 0.015625)
    if suffix in ["norm1.bias", "norm2.bias"]:
        return f32(((index % 5) - 2) * 0.001)
    raise RuntimeError(f"unknown GLIGEN key {key}")


def gligen_projected_value(key, index, shape, dtype):
    raw = storage_bits(gligen_value(key, index, shape), dtype)
    return project_storage(raw, dtype)


def gligen_linear(rows, input_width, output_width, values, key, dtype):
    shape = [output_width, input_width]
    output = []
    bias_key = key[:-6] + "bias" if key.endswith("weight") else None
    if ".attn.to_q." in key or ".attn.to_k." in key or ".attn.to_v." in key:
        bias_key = None
    def live_component(output_channel):
        if key == "position_net.linears.0.weight":
            return output_channel % input_width
        if key == "position_net.linears.2.weight":
            return output_channel
        if key == "position_net.linears.4.weight":
            return (output_channel * 17 + 3) % input_width
        suffix = key.split(".fuser.", 1)[1]
        if suffix == "linear.weight":
            return output_channel % input_width
        if suffix == "attn.to_k.weight":
            return (output_channel + 1) % input_width
        return output_channel % input_width
    for row in range(rows):
        for output_channel in range(output_width):
            total = gligen_projected_value(bias_key, output_channel, [output_width], dtype) if bias_key else 0.0
            input_channel = live_component(output_channel)
            total = fmadd(
                values[row * input_width + input_channel],
                gligen_projected_value(key, output_channel * input_width + input_channel, shape, dtype),
                total,
            )
            output.append(total)
    return output


def gligen_layer_norm(values, rows, width, prefix, dtype):
    output = []
    for row in range(rows):
        source = values[row * width:(row + 1) * width]
        mean_total = 0.0
        for value in source:
            mean_total += float(value)
        mean = f32(mean_total / width)
        variance_total = 0.0
        for value in source:
            difference = float(value) - float(mean)
            variance_total += difference * difference
        variance = variance_total / width
        inverse = f32(1.0 / math.sqrt(variance + float(f32(1.0e-5))))
        for component, value in enumerate(source):
            normalized = fmul(fadd(value, -mean), inverse)
            scale = gligen_projected_value(f"{prefix}.weight", component, [width], dtype)
            bias = gligen_projected_value(f"{prefix}.bias", component, [width], dtype)
            output.append(fadd(fmul(normalized, scale), bias))
    return output


def canonical_erf(value):
    sign = -1.0 if value < 0.0 else 1.0
    absolute = f32(abs(value))
    t = fdiv(1.0, fadd(1.0, fmul(0.3275911, absolute)))
    polynomial = fmul(fadd(fmul(fadd(fmul(fadd(fmul(fadd(fmul(1.0614054, t), -1.4531521), t), 1.4214138), t), -0.28449672), t), 0.2548296), t)
    return fmul(sign, fadd(1.0, -fmul(polynomial, f32(math.exp(fmul(-absolute, absolute))))))


def canonical_gelu(value):
    scaled = fmul(value, from_bits(0x3f3504f3))
    return fmul(fmul(0.5, value), fadd(1.0, canonical_erf(scaled)))


def gligen_attention(query, key, value, tokens, width):
    scale = f32(1.0 / math.sqrt(width))
    output = [0.0] * (tokens * width)
    for query_token in range(tokens):
        scores = []
        for key_token in range(tokens):
            score = 0.0
            for component in range(width):
                score = fadd(score, fmul(query[query_token * width + component], key[key_token * width + component]))
            scores.append(fmul(score, scale))
        maximum = max(scores)
        probabilities = [f32(math.exp(fadd(score, -maximum))) for score in scores]
        denominator = 0.0
        for probability in probabilities:
            denominator = fadd(denominator, probability)
        probabilities = [fdiv(probability, denominator) for probability in probabilities]
        for component in range(width):
            total = 0.0
            for key_token, probability in enumerate(probabilities):
                total = fadd(total, fmul(probability, value[key_token * width + component]))
            output[query_token * width + component] = total
    return output


def gligen_prepare(dtype):
    positions = [
        ([0.125, -0.25, 0.375, -0.5], 3.0, 5.0, 1.0, 2.0),
        ([-0.2, 0.15, 0.4, 0.05], 2.0, 4.0, -1.0, 6.0),
    ]
    rows = []
    for object_index in range(GLIGEN_MAX_OBJECTS):
        if object_index < len(positions):
            embedding, height, width, y, x = positions[object_index]
            box_values = [x / 8.0, y / 6.0, (x + width) / 8.0, (y + height) / 6.0]
            fourier = []
            for frequency_index in range(8):
                frequency = f32(100.0 ** (frequency_index / 8.0))
                fourier.extend(f32(math.sin(fmul(frequency, value))) for value in box_values)
                fourier.extend(f32(math.cos(fmul(frequency, value))) for value in box_values)
            rows.extend(f32(value) for value in embedding)
            rows.extend(fourier)
        else:
            rows.extend(gligen_projected_value("position_net.null_positive_feature", index, [GLIGEN_KEY], dtype) for index in range(GLIGEN_KEY))
            rows.extend(gligen_projected_value("position_net.null_position_feature", index, [GLIGEN_POSITION_WIDTH], dtype) for index in range(GLIGEN_POSITION_WIDTH))
    hidden = gligen_linear(GLIGEN_MAX_OBJECTS, GLIGEN_KEY + GLIGEN_POSITION_WIDTH, GLIGEN_POSITION_HIDDEN, rows, "position_net.linears.0.weight", dtype)
    hidden = [silu(value) for value in hidden]
    hidden = gligen_linear(GLIGEN_MAX_OBJECTS, GLIGEN_POSITION_HIDDEN, GLIGEN_POSITION_HIDDEN, hidden, "position_net.linears.2.weight", dtype)
    hidden = [silu(value) for value in hidden]
    return gligen_linear(GLIGEN_MAX_OBJECTS, GLIGEN_POSITION_HIDDEN, GLIGEN_KEY, hidden, "position_net.linears.4.weight", dtype)


def gligen_apply(dtype, namespace, objects):
    visual = [f32(((index % 11) - 5) * 0.03125) for index in range(GLIGEN_VISUAL_TOKENS * GLIGEN_QUERY)]
    projected = gligen_linear(GLIGEN_MAX_OBJECTS, GLIGEN_KEY, GLIGEN_QUERY, objects, f"{namespace}.fuser.linear.weight", dtype)
    combined = visual + projected
    tokens = GLIGEN_VISUAL_TOKENS + GLIGEN_MAX_OBJECTS
    normalized = gligen_layer_norm(combined, tokens, GLIGEN_QUERY, f"{namespace}.fuser.norm1", dtype)
    query = gligen_linear(tokens, GLIGEN_QUERY, GLIGEN_QUERY, normalized, f"{namespace}.fuser.attn.to_q.weight", dtype)
    key = gligen_linear(tokens, GLIGEN_QUERY, GLIGEN_QUERY, normalized, f"{namespace}.fuser.attn.to_k.weight", dtype)
    value = gligen_linear(tokens, GLIGEN_QUERY, GLIGEN_QUERY, normalized, f"{namespace}.fuser.attn.to_v.weight", dtype)
    attention = gligen_attention(query, key, value, tokens, GLIGEN_QUERY)
    attention = gligen_linear(tokens, GLIGEN_QUERY, GLIGEN_QUERY, attention, f"{namespace}.fuser.attn.to_out.0.weight", dtype)[:GLIGEN_VISUAL_TOKENS * GLIGEN_QUERY]
    alpha_attention = f32(math.tanh(gligen_projected_value(f"{namespace}.fuser.alpha_attn", 0, [], dtype)))
    output = [fadd(left, fmul(alpha_attention, right)) for left, right in zip(visual, attention)]
    normalized = gligen_layer_norm(output, GLIGEN_VISUAL_TOKENS, GLIGEN_QUERY, f"{namespace}.fuser.norm2", dtype)
    projected = gligen_linear(GLIGEN_VISUAL_TOKENS, GLIGEN_QUERY, GLIGEN_QUERY * 2, normalized, f"{namespace}.fuser.ff.net.0.proj.weight", dtype)
    dense = []
    for row in range(GLIGEN_VISUAL_TOKENS):
        start = row * GLIGEN_QUERY * 2
        for component in range(GLIGEN_QUERY):
            dense.append(fmul(projected[start + component], canonical_gelu(projected[start + GLIGEN_QUERY + component])))
    dense = gligen_linear(GLIGEN_VISUAL_TOKENS, GLIGEN_QUERY, GLIGEN_QUERY, dense, f"{namespace}.fuser.ff.net.2.weight", dtype)
    alpha_dense = f32(math.tanh(gligen_projected_value(f"{namespace}.fuser.alpha_dense", 0, [], dtype)))
    return visual, [fadd(left, fmul(alpha_dense, right)) for left, right in zip(output, dense)]


def gligen_oracle():
    cases = {}
    for dtype in ["float32", "float16", "bfloat16"]:
        objects = gligen_prepare(dtype)
        visual, output = gligen_apply(dtype, GLIGEN_FUSERS[0], objects)
        cases[dtype] = {
            "prepared_shape": [1, GLIGEN_MAX_OBJECTS, GLIGEN_KEY],
            "prepared_bits": [bits(value) for value in objects],
            "visual_shape": [1, GLIGEN_VISUAL_TOKENS, GLIGEN_QUERY],
            "visual_bits": [bits(value) for value in visual],
            "output_bits": [bits(value) for value in output],
        }
    return {
        "key_dimension": GLIGEN_KEY,
        "query_dimension": GLIGEN_QUERY,
        "state": [{"key": key, "shape": shape} for key, shape in gligen_manifest()],
        "fusers": [
            {"namespace": GLIGEN_FUSERS[0], "region": "input_blocks", "block_index": 2, "transformer_index": 0},
            {"namespace": GLIGEN_FUSERS[1], "region": "output_blocks", "block_index": 7, "transformer_index": 1},
        ],
        "positions": [
            {"embedding_bits": [bits(value) for value in [0.125, -0.25, 0.375, -0.5]], "height": 3.0, "width": 5.0, "y": 1.0, "x": 2.0},
            {"embedding_bits": [bits(value) for value in [-0.2, 0.15, 0.4, 0.05]], "height": 2.0, "width": 4.0, "y": -1.0, "x": 6.0},
        ],
        "latent_shape": [1, 4, 6, 8],
        "dtypes": cases,
        "head_rule_cases": [
            {"key_dimension": 4, "query_dimension": 64, "heads": 1, "head_dimension": 64},
            {"key_dimension": 768, "query_dimension": 8, "heads": 8, "head_dimension": 1},
        ],
    }


def photomaker_value(state_index, value_index, key, shape):
    if key == "vision_model.embeddings.class_embedding":
        return fmul(value_index - 1.5, 0.25)
    if key == "vision_model.embeddings.patch_embedding.weight" or key == \
            "vision_model.embeddings.position_embedding.weight":
        return f32(0.0)
    if any(marker in key for marker in ["layer_norm", "layrnorm", "layernorm"]):
        return f32(1.0 if key.endswith(".weight") else 0.0)
    if ".self_attn." in key or (".mlp." in key and key.startswith("vision_model")):
        return f32(0.0)
    lane = ((state_index * 17 + value_index * 11) % 23) - 11
    return fmul(lane, 0.015625 if key.endswith(".weight") else 0.0078125)


def photomaker_state(dtype):
    state = []
    for state_index, (key, shape) in enumerate(photomaker_manifest()):
        raw = [
            storage_bits(photomaker_value(state_index, index, key, shape), dtype)
            for index in range(product(shape))
        ]
        state.append({"key": key, "shape": shape, "storage_bits": raw})
    return state


def fixture_value(profile, state_index, value_index, key, shape):
    if key == "style_embedding" and value_index == 0:
        return from_bits(0x80000000)
    if ".attn.in_proj_weight" in key:
        output_row = value_index // shape[1]
        if output_row < STYLE_WIDTH * 2:
            return f32(0.0)
    if ".attn.in_proj_bias" in key and value_index < STYLE_WIDTH * 2:
        return f32(0.0)
    if key.endswith(".weight") and (".ln_" in key or key.startswith("ln_")):
        lane = ((state_index + value_index * 3) % 7) - 3
        return fadd(1.0, fmul(lane, 0.015625))
    if key.endswith(".bias") and (".ln_" in key or key.startswith("ln_")):
        lane = ((state_index * 5 + value_index * 7) % 9) - 4
        return fmul(lane, 0.0078125)
    lane = ((state_index * 19 + value_index * 13 + (3 if profile == "redux" else 0)) % 31) - 15
    scale = 0.00625 if key.endswith(".bias") else 0.0125
    return fmul(lane, scale)


def source_state(profile, dtype):
    definitions = style_manifest() if profile == "style" else redux_manifest()
    state = []
    for state_index, (key, shape) in enumerate(definitions):
        raw = [
            storage_bits(fixture_value(profile, state_index, index, key, shape), dtype)
            for index in range(product(shape))
        ]
        state.append({"key": key, "shape": shape, "storage_bits": raw})
    return state


def projected_state(state, dtype):
    return {entry["key"]: [project_storage(value, dtype) for value in entry["storage_bits"]] for entry in state}


def input_values(width, batch=1):
    values = []
    for index in range(batch * INPUT_TOKENS * width):
        if index == 0:
            values.append(from_bits(0x80000000))
        else:
            values.append(fmul(((index * 11) % 17) - 8, 0.03125))
    return values


def linear(values, rows, input_width, weight, output_width, bias):
    output = []
    for row in range(rows):
        for output_channel in range(output_width):
            total = f32(0.0 if bias is None else bias[output_channel])
            for input_channel in range(input_width):
                total = fmadd(
                    values[row * input_width + input_channel],
                    weight[output_channel * input_width + input_channel],
                    total,
                )
            output.append(total)
    return output


def layer_norm(values, width, weight, bias):
    output = []
    for start in range(0, len(values), width):
        row = values[start:start + width]
        total = 0.0
        for value in row:
            total += float(value)
        mean = f32(total / width)
        variance_total = 0.0
        for value in row:
            difference = float(value) - float(mean)
            variance_total += difference * difference
        variance = variance_total / width
        inverse = f32(1.0 / math.sqrt(variance + float(f32(1.0e-5))))
        for index, value in enumerate(row):
            normalized = fmul(fadd(value, -mean), inverse)
            output.append(fadd(fmul(normalized, weight[index]), bias[index]))
    return output


def add(left, right):
    return [fadd(a, b) for a, b in zip(left, right)]


def attention(values, state, layer, batch, tokens):
    prefix = f"transformer_layes.{layer}.attn"
    fused_weight = state[f"{prefix}.in_proj_weight"]
    fused_bias = state[f"{prefix}.in_proj_bias"]
    width = STYLE_WIDTH
    rows = batch * tokens
    query = linear(values, rows, width, fused_weight[:width * width], width, fused_bias[:width])
    key = linear(values, rows, width, fused_weight[width * width:2 * width * width], width,
                 fused_bias[width:2 * width])
    value = linear(values, rows, width, fused_weight[2 * width * width:], width,
                   fused_bias[2 * width:])
    head_width = width // STYLE_HEADS
    scale = f32(1.0 / math.sqrt(head_width))
    output = [f32(0.0)] * (rows * width)
    for batch_index in range(batch):
        batch_start = batch_index * tokens
        for head in range(STYLE_HEADS):
            for query_token in range(tokens):
                query_row = batch_start + query_token
                scores = []
                for key_token in range(tokens):
                    key_row = batch_start + key_token
                    score = f32(0.0)
                    for component in range(head_width):
                        q = query[query_row * width + head * head_width + component]
                        k = key[key_row * width + head * head_width + component]
                        score = fadd(score, fmul(q, k))
                    scores.append(fmul(score, scale))
                maximum = max(scores)
                probabilities = [f32(math.exp(fadd(score, -maximum))) for score in scores]
                denominator = f32(0.0)
                for probability in probabilities:
                    denominator = fadd(denominator, probability)
                probabilities = [fdiv(probability, denominator) for probability in probabilities]
                for component in range(head_width):
                    result = f32(0.0)
                    for key_token, probability in enumerate(probabilities):
                        key_row = batch_start + key_token
                        lane = value[key_row * width + head * head_width + component]
                        result = fadd(result, fmul(probability, lane))
                    output[query_row * width + head * head_width + component] = result
    return output


def matmul_f64(left, rows, inner, right, columns):
    output = []
    for row in range(rows):
        for column in range(columns):
            total = 0.0
            for lane in range(inner):
                total += float(left[row * inner + lane]) * float(right[lane * columns + column])
            output.append(f32(total))
    return output


def execute_style(state, input_data, batch=1):
    sequence = INPUT_TOKENS + STYLE_TOKENS
    rows = batch * sequence
    embedding = state["style_embedding"]
    embedding = [fadd(value, 0.0) for value in embedding]
    values = []
    input_batch_values = INPUT_TOKENS * STYLE_WIDTH
    for batch_index in range(batch):
        start = batch_index * input_batch_values
        values.extend(input_data[start:start + input_batch_values])
        values.extend(embedding)
    values = layer_norm(values, STYLE_WIDTH, state["ln_pre.weight"], state["ln_pre.bias"])
    for layer in range(STYLE_LAYERS):
        prefix = f"transformer_layes.{layer}"
        normalized = layer_norm(values, STYLE_WIDTH, state[f"{prefix}.ln_1.weight"],
                                state[f"{prefix}.ln_1.bias"])
        attended = attention(normalized, state, layer, batch, sequence)
        projected = linear(attended, rows, STYLE_WIDTH, state[f"{prefix}.attn.out_proj.weight"],
                           STYLE_WIDTH, state[f"{prefix}.attn.out_proj.bias"])
        values = add(values, projected)
        normalized = layer_norm(values, STYLE_WIDTH, state[f"{prefix}.ln_2.weight"],
                                state[f"{prefix}.ln_2.bias"])
        expanded = linear(normalized, rows, STYLE_WIDTH, state[f"{prefix}.mlp.c_fc.weight"],
                          STYLE_WIDTH * 4, state[f"{prefix}.mlp.c_fc.bias"])
        activated = [fmul(value, sigmoid(fmul(1.702, value))) for value in expanded]
        projected = linear(activated, rows, STYLE_WIDTH * 4,
                           state[f"{prefix}.mlp.c_proj.weight"], STYLE_WIDTH,
                           state[f"{prefix}.mlp.c_proj.bias"])
        values = add(values, projected)
    selected = []
    for batch_index in range(batch):
        start = (batch_index * sequence + INPUT_TOKENS) * STYLE_WIDTH
        selected.extend(values[start:start + STYLE_TOKENS * STYLE_WIDTH])
    selected = layer_norm(selected, STYLE_WIDTH, state["ln_post.weight"], state["ln_post.bias"])
    return matmul_f64(
        selected, batch * STYLE_TOKENS, STYLE_WIDTH, state["proj"], STYLE_CONTEXT
    )


def execute_redux(state, input_data):
    expanded = linear(input_data, INPUT_TOKENS, REDUX_INPUT, state["redux_up.weight"],
                      REDUX_HIDDEN, state["redux_up.bias"])
    activated = [silu(value) for value in expanded]
    return linear(activated, INPUT_TOKENS, REDUX_HIDDEN, state["redux_down.weight"],
                  REDUX_OUTPUT, state["redux_down.bias"])


def gelu(value):
    return fmul(fmul(0.5, value), fadd(1.0, f32(math.erf(f32(value) / math.sqrt(2.0)))))


def canonical_erf_approximation(value):
    sign = -1.0 if value < 0.0 else 1.0
    absolute = abs(f32(value))
    t = fdiv(1.0, fadd(1.0, fmul(0.3275911, absolute)))
    polynomial = fadd(fmul(1.0614054, t), -1.4531521)
    polynomial = fadd(fmul(polynomial, t), 1.4214138)
    polynomial = fadd(fmul(polynomial, t), -0.28449672)
    polynomial = fadd(fmul(polynomial, t), 0.2548296)
    polynomial = fmul(polynomial, t)
    exponential = f32(math.exp(fmul(-absolute, absolute)))
    return fmul(sign, fadd(1.0, -fmul(polynomial, exponential)))


def canonical_gelu(value):
    scaled = fmul(value, from_bits(0x3F3504F3))
    return fmul(fmul(0.5, value), fadd(1.0, canonical_erf_approximation(scaled)))


def ordered_float_bits(value):
    raw = bits(value)
    return (~raw & 0xFFFFFFFF) if raw & 0x80000000 else raw | 0x80000000


def max_ulp_distance(left, right):
    return max(
        abs(ordered_float_bits(a) - ordered_float_bits(b))
        for a, b in zip(left, right)
    )


def photomaker_inputs():
    image = [
        fmul(((index * 7) % 19) - 9, 0.03125)
        for index in range(PHOTO_IMAGES * 3 * PHOTO_IMAGE * PHOTO_IMAGE)
    ]
    prompt = []
    for index in range(PHOTO_SEQUENCE * PHOTO_PROMPT):
        prompt.append(from_bits(0x80000000) if index == 0 else
                      fmul(((index * 13) % 29) - 14, 0.015625))
    return image, prompt, [False, True, False, True]


def execute_photomaker(state, activation=gelu):
    _, prompt, mask = photomaker_inputs()
    pooled = state["vision_model.embeddings.class_embedding"][:]
    pooled = layer_norm(
        pooled,
        PHOTO_HIDDEN,
        state["vision_model.pre_layrnorm.weight"],
        state["vision_model.pre_layrnorm.bias"],
    )
    pooled = layer_norm(
        pooled,
        PHOTO_HIDDEN,
        state["vision_model.post_layernorm.weight"],
        state["vision_model.post_layernorm.bias"],
    )
    pooled = pooled * PHOTO_IMAGES
    first_projection = linear(
        pooled,
        PHOTO_IMAGES,
        PHOTO_HIDDEN,
        state["visual_projection.weight"],
        PHOTO_PROJECTION,
        None,
    )
    second_projection = linear(
        pooled,
        PHOTO_IMAGES,
        PHOTO_HIDDEN,
        state["visual_projection_2.weight"],
        PHOTO_EXTRA_PROJECTION,
        None,
    )
    identity = []
    for row in range(PHOTO_IMAGES):
        identity.extend(first_projection[row * PHOTO_PROJECTION:(row + 1) * PHOTO_PROJECTION])
        identity.extend(second_projection[
            row * PHOTO_EXTRA_PROJECTION:(row + 1) * PHOTO_EXTRA_PROJECTION
        ])
    positions = [index for index, selected in enumerate(mask) if selected]
    prompt_rows = []
    for position in positions:
        prompt_rows.extend(prompt[position * PHOTO_PROMPT:(position + 1) * PHOTO_PROMPT])
    stacked = []
    for row in range(PHOTO_IMAGES):
        stacked.extend(prompt_rows[row * PHOTO_PROMPT:(row + 1) * PHOTO_PROMPT])
        stacked.extend(identity[row * PHOTO_PROMPT:(row + 1) * PHOTO_PROMPT])
    normalized = layer_norm(
        stacked,
        PHOTO_PROMPT * 2,
        state["fuse_module.mlp1.layernorm.weight"],
        state["fuse_module.mlp1.layernorm.bias"],
    )
    expanded = linear(
        normalized,
        PHOTO_IMAGES,
        PHOTO_PROMPT * 2,
        state["fuse_module.mlp1.fc1.weight"],
        PHOTO_PROMPT,
        state["fuse_module.mlp1.fc1.bias"],
    )
    activated = [activation(value) for value in expanded]
    fused = linear(
        activated,
        PHOTO_IMAGES,
        PHOTO_PROMPT,
        state["fuse_module.mlp1.fc2.weight"],
        PHOTO_PROMPT,
        state["fuse_module.mlp1.fc2.bias"],
    )
    fused = add(fused, prompt_rows)
    normalized = layer_norm(
        fused,
        PHOTO_PROMPT,
        state["fuse_module.mlp2.layernorm.weight"],
        state["fuse_module.mlp2.layernorm.bias"],
    )
    expanded = linear(
        normalized,
        PHOTO_IMAGES,
        PHOTO_PROMPT,
        state["fuse_module.mlp2.fc1.weight"],
        PHOTO_PROMPT,
        state["fuse_module.mlp2.fc1.bias"],
    )
    activated = [activation(value) for value in expanded]
    residual = linear(
        activated,
        PHOTO_IMAGES,
        PHOTO_PROMPT,
        state["fuse_module.mlp2.fc2.weight"],
        PHOTO_PROMPT,
        state["fuse_module.mlp2.fc2.bias"],
    )
    fused = add(fused, residual)
    fused = layer_norm(
        fused,
        PHOTO_PROMPT,
        state["fuse_module.layer_norm.weight"],
        state["fuse_module.layer_norm.bias"],
    )
    output = prompt[:]
    for row, position in enumerate(positions):
        output[position * PHOTO_PROMPT:(position + 1) * PHOTO_PROMPT] = \
            fused[row * PHOTO_PROMPT:(row + 1) * PHOTO_PROMPT]
    return {
        "pooled": pooled,
        "first_projection": first_projection,
        "second_projection": second_projection,
        "identity": identity,
        "output": output,
    }


def photomaker_profile_oracle():
    image, prompt, mask = photomaker_inputs()
    cases = {}
    for dtype in ["float32", "float16", "bfloat16"]:
        state = photomaker_state(dtype)
        projected = projected_state(state, dtype)
        phases = execute_photomaker(projected)
        canonical_phases = execute_photomaker(projected, canonical_gelu)
        output_ulp_bound = max_ulp_distance(
            phases["output"], canonical_phases["output"]
        )
        cases[dtype] = {
            "state": state,
            "source_identity_sha256": state_identity(state, dtype),
            "projected_identity_sha256": state_identity(state, dtype, projected=True),
            "pooled_bits": [bits(value) for value in phases["pooled"]],
            "first_projection_bits": [bits(value) for value in phases["first_projection"]],
            "second_projection_bits": [bits(value) for value in phases["second_projection"]],
            "identity_bits": [bits(value) for value in phases["identity"]],
            "output_shape": [1, PHOTO_SEQUENCE, PHOTO_PROMPT],
            "output_bits": [bits(value) for value in phases["output"]],
            "canonical_output_bits": [bits(value) for value in canonical_phases["output"]],
            "output_ulp_bound": output_ulp_bound,
            "output_ulp_rejected_distance": output_ulp_bound + 1,
        }
    return {
        "image_shape": [1, PHOTO_IMAGES, 3, PHOTO_IMAGE, PHOTO_IMAGE],
        "image_bits": [bits(value) for value in image],
        "prompt_shape": [1, PHOTO_SEQUENCE, PHOTO_PROMPT],
        "prompt_bits": [bits(value) for value in prompt],
        "mask_shape": [1, PHOTO_SEQUENCE],
        "mask": mask,
        "mask_positions": [index for index, selected in enumerate(mask) if selected],
        "dtypes": cases,
    }


def photomaker_mutation(name, key, index, delta):
    state = photomaker_state("float32")
    entry = next(candidate for candidate in state if candidate["key"] == key)
    entry["storage_bits"][index] = bits(fadd(from_bits(entry["storage_bits"][index]), delta))
    phases = execute_photomaker(projected_state(state, "float32"))
    canonical_phases = execute_photomaker(projected_state(state, "float32"), canonical_gelu)
    output_ulp_bound = max_ulp_distance(phases["output"], canonical_phases["output"])
    return {
        "profile": "photomaker",
        "key": key,
        "index": index,
        "delta_bits": bits(delta),
        "source_identity_sha256": state_identity(state, "float32"),
        "output_bits": [bits(value) for value in phases["output"]],
        "canonical_output_bits": [bits(value) for value in canonical_phases["output"]],
        "output_ulp_bound": output_ulp_bound,
        "output_ulp_rejected_distance": output_ulp_bound + 1,
    }


def state_identity(state, dtype, projected=False):
    digest = hashlib.sha256()
    digest.update(b"conditioning-auxiliary-state-v1\0")
    digest.update(("float32" if projected else dtype).encode())
    for entry in state:
        key = entry["key"].encode()
        digest.update(len(key).to_bytes(8, "little"))
        digest.update(key)
        for dimension in entry["shape"]:
            digest.update(dimension.to_bytes(8, "little"))
        for raw in entry["storage_bits"]:
            value = bits(project_storage(raw, dtype)) if projected else raw
            width = 4 if projected or dtype == "float32" else 2
            digest.update(value.to_bytes(width, "little"))
    return digest.hexdigest()


def profile_oracle(profile):
    cases = {}
    input_data = input_values(STYLE_WIDTH if profile == "style" else REDUX_INPUT)
    for dtype in ["float32", "float16", "bfloat16"]:
        state = source_state(profile, dtype)
        projected = projected_state(state, dtype)
        output = execute_style(projected, input_data) if profile == "style" else execute_redux(projected, input_data)
        cases[dtype] = {
            "state": state,
            "source_identity_sha256": state_identity(state, dtype),
            "projected_identity_sha256": state_identity(state, dtype, projected=True),
            "output_shape": [1, STYLE_TOKENS, STYLE_CONTEXT] if profile == "style" else [1, INPUT_TOKENS, REDUX_OUTPUT],
            "output_bits": [bits(value) for value in output],
        }
    return {
        "input_shape": [1, INPUT_TOKENS, STYLE_WIDTH if profile == "style" else REDUX_INPUT],
        "input_bits": [bits(value) for value in input_data],
        "dtypes": cases,
    }


def mutation_oracle(profile, name, key, index, delta):
    state = source_state(profile, "float32")
    for entry in state:
        if entry["key"] == key:
            value = project_storage(entry["storage_bits"][index], "float32")
            entry["storage_bits"][index] = bits(fadd(value, delta))
            break
    projected = projected_state(state, "float32")
    input_data = input_values(STYLE_WIDTH if profile == "style" else REDUX_INPUT)
    output = execute_style(projected, input_data) if profile == "style" else execute_redux(projected, input_data)
    return {
        "profile": profile,
        "key": key,
        "index": index,
        "delta_bits": bits(delta),
        "source_identity_sha256": state_identity(state, "float32"),
        "output_bits": [bits(value) for value in output],
    }


def attention_discriminator_oracle():
    state = source_state("style", "float32")
    modifications = []
    for layer in range(STYLE_LAYERS):
        weight_key = f"transformer_layes.{layer}.attn.in_proj_weight"
        bias_key = f"transformer_layes.{layer}.attn.in_proj_bias"
        layer_scale = f32((layer + 1) * 0.125)
        weight_values = [
            (0 * STYLE_WIDTH + 0, fadd(0.5, layer_scale)),
            (0 * STYLE_WIDTH + 1, -0.25),
            (4 * STYLE_WIDTH + 2, fadd(-0.375, layer_scale)),
            (4 * STYLE_WIDTH + 3, 0.625),
            ((STYLE_WIDTH + 0) * STYLE_WIDTH + 1, fadd(-0.75, layer_scale)),
            ((STYLE_WIDTH + 0) * STYLE_WIDTH + 2, 0.375),
            ((STYLE_WIDTH + 4) * STYLE_WIDTH + 3, fadd(0.875, -layer_scale)),
            ((STYLE_WIDTH + 4) * STYLE_WIDTH + 4, -0.5),
        ]
        bias_values = [
            (0, fadd(0.0625, layer_scale)),
            (4, fadd(-0.09375, layer_scale)),
            (STYLE_WIDTH + 0, fadd(0.15625, -layer_scale)),
            (STYLE_WIDTH + 4, fadd(-0.21875, layer_scale)),
        ]
        for key, values in [(weight_key, weight_values), (bias_key, bias_values)]:
            entry = next(candidate for candidate in state if candidate["key"] == key)
            for index, value in values:
                value_bits = bits(value)
                entry["storage_bits"][index] = value_bits
                modifications.append({"key": key, "index": index, "value_bits": value_bits})
    projected = projected_state(state, "float32")
    batch = 2
    input_data = input_values(STYLE_WIDTH, batch)
    output = execute_style(projected, input_data, batch)
    batch_output_values = STYLE_TOKENS * STYLE_CONTEXT
    return {
        "state_modifications": modifications,
        "source_identity_sha256": state_identity(state, "float32"),
        "input_shape": [batch, INPUT_TOKENS, STYLE_WIDTH],
        "input_bits": [bits(value) for value in input_data],
        "output_shape": [batch, STYLE_TOKENS, STYLE_CONTEXT],
        "output_bits": [bits(value) for value in output],
        "batch_outputs_differ": output[:batch_output_values] != output[batch_output_values:],
        "query_key_are_asymmetric": any(
            modification["index"] < STYLE_WIDTH * STYLE_WIDTH
            for modification in modifications
        ) and any(
            STYLE_WIDTH * STYLE_WIDTH
            <= modification["index"]
            < STYLE_WIDTH * STYLE_WIDTH * 2
            for modification in modifications
        ),
    }


def build_oracle():
    quick_input = f32(0.375)
    quick_scaled = fmul(1.702, quick_input)
    quick_sigmoid = sigmoid(quick_scaled)
    mutations = {
        "style_embedding": mutation_oracle("style", "style_embedding", "style_embedding", 1, 0.125),
        "style_value_projection": mutation_oracle(
            "style", "style_value_projection", "transformer_layes.0.attn.in_proj_weight",
            STYLE_WIDTH * STYLE_WIDTH * 2 + 3, 0.125),
        "style_quick_gelu": mutation_oracle(
            "style", "style_quick_gelu", "transformer_layes.1.mlp.c_fc.bias", 5, 0.25),
        "style_projection": mutation_oracle("style", "style_projection", "proj", 7, 0.125),
        "redux_up": mutation_oracle("redux", "redux_up", "redux_up.bias", 2, 0.25),
        "redux_down": mutation_oracle("redux", "redux_down", "redux_down.weight", 9, 0.125),
    }
    photomaker_mutations = {
        "photomaker_extra_projection": photomaker_mutation(
            "photomaker_extra_projection", "visual_projection_2.weight", 3, 0.125),
        "photomaker_mlp1": photomaker_mutation(
            "photomaker_mlp1", "fuse_module.mlp1.fc1.weight", 11, 0.125),
        "photomaker_mlp2": photomaker_mutation(
            "photomaker_mlp2", "fuse_module.mlp2.fc2.bias", 2, 0.125),
    }
    return {
        "format": "conditioning-auxiliary-resource-foundation-v1",
        "reduced_profiles_are_source_exact": False,
        "source_dimensions": {
            "style": {"width": 1024, "context": 768, "heads": 8, "layers": 3, "tokens": 8, "state_count": 42},
            "redux": {"input": 1152, "hidden": 12288, "output": 4096, "state_count": 4},
            "photomaker": {"hidden": 1024, "intermediate": 4096, "heads": 16,
                           "layers": 24, "image": 224, "patch": 14,
                           "projection": 768, "extra_projection": 1280,
                           "prompt": 2048, "state_count": 407},
        },
        "reduced_dimensions": {
            "style": {"width": STYLE_WIDTH, "context": STYLE_CONTEXT, "heads": STYLE_HEADS,
                      "layers": STYLE_LAYERS, "tokens": STYLE_TOKENS, "state_count": 42},
            "redux": {"input": REDUX_INPUT, "hidden": REDUX_HIDDEN, "output": REDUX_OUTPUT,
                      "state_count": 4},
            "photomaker": {"hidden": PHOTO_HIDDEN, "intermediate": PHOTO_INTERMEDIATE,
                           "heads": PHOTO_HEADS, "layers": PHOTO_LAYERS,
                           "image": PHOTO_IMAGE, "patch": PHOTO_PATCH,
                           "projection": PHOTO_PROJECTION,
                           "extra_projection": PHOTO_EXTRA_PROJECTION,
                           "prompt": PHOTO_PROMPT, "state_count": 407},
        },
        "style": profile_oracle("style"),
        "redux": profile_oracle("redux"),
        "photomaker": photomaker_profile_oracle(),
        "gligen": gligen_oracle(),
        "attention_discriminator": attention_discriminator_oracle(),
        "mutations": mutations,
        "photomaker_mutations": photomaker_mutations,
        "discriminators": {
            "signed_zero_input_bits": 0x80000000,
            "signed_zero_after_add_bits": bits(fadd(from_bits(0x80000000), 0.0)),
            "quick_gelu": {
                "coefficient_bits": bits(1.702),
                "input_bits": bits(quick_input),
                "scaled_bits": bits(quick_scaled),
                "sigmoid_bits": bits(quick_sigmoid),
                "output_bits": bits(fmul(quick_input, quick_sigmoid)),
            },
            "detection_precedence": ["style_embedding", "redux_down.weight"],
        },
    }
