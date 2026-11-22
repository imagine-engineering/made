#!/usr/bin/env python3
import hashlib
import json
import math
import platform
import struct
from pathlib import Path


PINNED_SOURCES = {
    "projects/comfy/ComfyUI/comfy/ldm/aura/mmdit.py": "0104396eda01a9f78e8aa5b9d15470fc551aa8b0e05137d264f3515fd1739db1",
    "projects/comfy/ComfyUI/comfy/ldm/qwen_image/model.py": "14c805af8da13d31094c2c704e413cc282ef43f4afe17f77de9070bdca301f28",
    "projects/comfy/ComfyUI/comfy/ldm/modules/attention.py": "436e1d91f8d5d84c5667e051cdf3ab2f91d8db25b66d88a084c89a202de0579e",
    "projects/comfy/ComfyUI/comfy/ldm/lightricks/model.py": "edcdab6083e4e2b4c7c1ff51323174ba0308b7626895e5874238ca09c0bd5f43",
    "projects/comfy/ComfyUI/comfy/ldm/flux/layers.py": "35f2dfbadb8b59de79c306f365f26c2baa0a7d54836414737cd781fc24e6a2bc",
    "projects/comfy/ComfyUI/comfy/ldm/flux/math.py": "ee3473e262894884b12eb3af4caa22d069273eb87db3549fb46c42c37f910ece",
}
EXPECTED_AURA_SHA256 = "d4b51dc633cbc296d70d5eba222ccac3a6e4a29133e8caca50ae886e2a6208bc"
EXPECTED_QWEN_SHA256 = "1fd6d5c61affabe7e2dd7ce573eb9d60eb77037d194586eb433ac916fe69609b"


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def add(left, right):
    return f32(f32(left) + f32(right))


def mul(left, right):
    return f32(f32(left) * f32(right))


def patterned(key, count):
    normalization = "norm_" in key or key.endswith("txt_norm.weight")
    bias = key.endswith(".bias")
    aura_mlp = ".mlpC." in key or ".mlpX." in key or "single_layers.0.mlp." in key
    aura_attention = "native.double_layers.0.attn." in key
    values = []
    for index in range(count):
        if normalization:
            value = 0.95 + (index % 7) * 0.01
        elif bias:
            value = ((index % 11) - 5.0) * 0.002
        elif aura_mlp:
            value = ((index % 17) - 8.0) * 0.5
        elif aura_attention:
            value = ((index % 17) - 8.0) * 0.1
        else:
            value = ((index % 17) - 8.0) * 0.00075
        values.append(f32(value))
    return values


def linear(values, inside, outside, key, bias_key=None):
    weight = patterned(key, inside * outside)
    bias = patterned(bias_key, outside) if bias_key else None
    output = []
    for row in range(len(values) // inside):
        for outer in range(outside):
            value = bias[outer] if bias else 0.0
            for inner in range(inside):
                value = add(value, mul(values[row * inside + inner], weight[outer * inside + inner]))
            output.append(value)
    return output


def silu(value):
    return f32(value / f32(1.0 + f32(math.exp(-value))))


def gelu(value):
    cube = mul(mul(value, value), value)
    inside = mul(f32(0.7978846), add(value, mul(f32(0.044715), cube)))
    return mul(mul(f32(0.5), value), add(1.0, f32(math.tanh(inside))))


def layer_norm(values, width, epsilon):
    output = []
    for start in range(0, len(values), width):
        row = values[start:start + width]
        total = 0.0
        for value in row:
            total = add(total, value)
        mean = f32(total / width)
        variance = 0.0
        for value in row:
            delta = add(value, -mean)
            variance = add(variance, mul(delta, delta))
        variance = f32(variance / width)
        inverse = f32(1.0 / f32(math.sqrt(add(variance, epsilon))))
        output.extend(mul(add(value, -mean), inverse) for value in row)
    return output


def rms_norm(values, width, key, epsilon):
    weight = patterned(key, width)
    output = []
    for start in range(0, len(values), width):
        row = values[start:start + width]
        square = 0.0
        for value in row:
            square = add(square, mul(value, value))
        inverse = f32(1.0 / f32(math.sqrt(add(f32(square / width), epsilon))))
        output.extend(mul(mul(value, inverse), weight[channel]) for channel, value in enumerate(row))
    return output


def modulate(values, batch, tokens, width, parameters, shift_offset, scale_offset):
    parameter_width = len(parameters) // batch
    output = [0.0] * len(values)
    for batch_index in range(batch):
        for token in range(tokens):
            for channel in range(width):
                index = (batch_index * tokens + token) * width + channel
                shift = parameters[batch_index * parameter_width + shift_offset + channel]
                scale = parameters[batch_index * parameter_width + scale_offset + channel]
                output[index] = add(shift, mul(values[index], add(1.0, scale)))
    return output


def gated(residual, update, batch, tokens, width, parameters, offset):
    parameter_width = len(parameters) // batch
    output = [0.0] * len(residual)
    for batch_index in range(batch):
        for token in range(tokens):
            for channel in range(width):
                index = (batch_index * tokens + token) * width + channel
                gate = parameters[batch_index * parameter_width + offset + channel]
                output[index] = add(residual[index], mul(gate, update[index]))
    return output


def patchify(values, channels, temporal, height, width):
    patch_height = (height + 1) // 2
    patch_width = (width + 1) // 2
    output = [0.0] * (temporal * patch_height * patch_width * channels * 4)
    for time in range(temporal):
        for patch_y in range(patch_height):
            for patch_x in range(patch_width):
                token = (time * patch_height + patch_y) * patch_width + patch_x
                for channel in range(channels):
                    for local_y in range(2):
                        for local_x in range(2):
                            source_y = patch_y * 2 + local_y
                            source_x = patch_x * 2 + local_x
                            if source_y < height and source_x < width:
                                source = (((channel * temporal + time) * height + source_y) * width) + source_x
                                feature = (channel * 2 + local_y) * 2 + local_x
                                output[token * channels * 4 + feature] = values[source]
    return output


def unpatchify(values, channels, temporal, height, width):
    patch_height = (height + 1) // 2
    patch_width = (width + 1) // 2
    output = [0.0] * (channels * temporal * height * width)
    for time in range(temporal):
        for patch_y in range(patch_height):
            for patch_x in range(patch_width):
                token = (time * patch_height + patch_y) * patch_width + patch_x
                for channel in range(channels):
                    for local_y in range(2):
                        for local_x in range(2):
                            target_y = patch_y * 2 + local_y
                            target_x = patch_x * 2 + local_x
                            if target_y < height and target_x < width:
                                target = (((channel * temporal + time) * height + target_y) * width) + target_x
                                feature = (channel * 2 + local_y) * 2 + local_x
                                output[target] = values[token * channels * 4 + feature]
    return output


def attention(query, key, value, tokens, width, mask=None):
    scale = f32(1.0 / math.sqrt(width))
    output = [0.0] * len(query)
    for query_token in range(tokens):
        scores = []
        for key_token in range(tokens):
            score = 0.0
            for channel in range(width):
                score = add(score, mul(query[query_token * width + channel], key[key_token * width + channel]))
            score = mul(score, scale)
            if mask is not None:
                score = add(score, mask[query_token * tokens + key_token])
            scores.append(score)
        maximum = max(scores)
        exponentials = [f32(math.exp(add(score, -maximum))) for score in scores]
        denominator = 0.0
        for exponential in exponentials:
            denominator = add(denominator, exponential)
        probabilities = [f32(exponential / denominator) for exponential in exponentials]
        for channel in range(width):
            result = 0.0
            for key_token in range(tokens):
                result = add(result, mul(probabilities[key_token], value[key_token * width + channel]))
            output[query_token * width + channel] = result
    return output


def aura_mlp(values, prefix):
    first = linear(values, 2, 256, prefix + ".c_fc1.weight")
    second = linear(values, 2, 256, prefix + ".c_fc2.weight")
    activation = [mul(silu(left), right) for left, right in zip(first, second)]
    return linear(activation, 256, 2, prefix + ".c_proj.weight")


def aura_time(time):
    basis = []
    for frequency in range(128):
        omega = mul(1000.0, f32(math.exp(f32(mul(-f32(math.log(10000.0)), f32(frequency / 128.0))))))
        basis.append(f32(math.cos(mul(time, omega))))
    for frequency in range(128):
        omega = mul(1000.0, f32(math.exp(f32(mul(-f32(math.log(10000.0)), f32(frequency / 128.0))))))
        basis.append(f32(math.sin(mul(time, omega))))
    first = linear(basis, 256, 2, "native.t_embedder.mlp.0.weight", "native.t_embedder.mlp.0.bias")
    second = linear([silu(value) for value in first], 2, 2, "native.t_embedder.mlp.2.weight", "native.t_embedder.mlp.2.bias")
    return second


def aura_oracle():
    height = width = 3
    image_tokens = 4
    latent = [f32((index - 18.0) * 0.025) for index in range(36)]
    image = linear(patchify(latent, 4, 1, height, width), 16, 2, "native.init_x_linear.weight", "native.init_x_linear.bias")
    positions = patterned("native.positional_encoding", 32)
    # Fixed 4x4 positional state, centered 2x2 crop.
    cropped = []
    for row in (1, 2):
        cropped.extend(positions[(row * 4 + 1) * 2:(row * 4 + 3) * 2])
    image = [add(value, cropped[index % 8]) for index, value in enumerate(image)]
    condition = patterned("aura.conditioning", 2 * 2048)
    projected = linear(condition, 2048, 2, "native.cond_seq_linear.weight")
    text = patterned("native.register_tokens", 16) + projected
    text_tokens = 10
    time = aura_time(f32(0.31415927))
    time_silu = [silu(value) for value in time]
    text_mod = linear(time_silu, 2, 12, "native.double_layers.0.modC.1.weight")
    image_mod = linear(time_silu, 2, 12, "native.double_layers.0.modX.1.weight")
    text_residual = list(text)
    image_residual = list(image)
    text_input = modulate(layer_norm(text, 2, f32(1e-5)), 1, text_tokens, 2, text_mod, 0, 2)
    image_input = modulate(layer_norm(image, 2, f32(1e-5)), 1, image_tokens, 2, image_mod, 0, 2)
    text_q = linear(text_input, 2, 2, "native.double_layers.0.attn.w1q.weight")
    text_k = linear(text_input, 2, 2, "native.double_layers.0.attn.w1k.weight")
    text_v = linear(text_input, 2, 2, "native.double_layers.0.attn.w1v.weight")
    image_q = linear(image_input, 2, 2, "native.double_layers.0.attn.w2q.weight")
    image_k = linear(image_input, 2, 2, "native.double_layers.0.attn.w2k.weight")
    image_v = linear(image_input, 2, 2, "native.double_layers.0.attn.w2v.weight")
    joint = text_tokens + image_tokens
    attended = attention(layer_norm(text_q + image_q, 2, f32(1e-5)), layer_norm(text_k + image_k, 2, f32(1e-5)), text_v + image_v, joint, 2)
    text_attention = linear(attended[:text_tokens * 2], 2, 2, "native.double_layers.0.attn.w1o.weight")
    image_attention = linear(attended[text_tokens * 2:], 2, 2, "native.double_layers.0.attn.w2o.weight")
    text = gated(text_residual, text_attention, 1, text_tokens, 2, text_mod, 4)
    image = gated(image_residual, image_attention, 1, image_tokens, 2, image_mod, 4)
    text_mlp = aura_mlp(modulate(layer_norm(text, 2, f32(1e-5)), 1, text_tokens, 2, text_mod, 6, 8), "native.double_layers.0.mlpC")
    image_mlp = aura_mlp(modulate(layer_norm(image, 2, f32(1e-5)), 1, image_tokens, 2, image_mod, 6, 8), "native.double_layers.0.mlpX")
    text = gated(text_residual, text_mlp, 1, text_tokens, 2, text_mod, 10)
    image = gated(image_residual, image_mlp, 1, image_tokens, 2, image_mod, 10)
    combined = text + image
    residual = list(combined)
    single_mod = linear(time_silu, 2, 12, "native.single_layers.0.modCX.1.weight")
    single_input = modulate(layer_norm(combined, 2, f32(1e-5)), 1, joint, 2, single_mod, 0, 2)
    query = linear(single_input, 2, 2, "native.single_layers.0.attn.w1q.weight")
    key = linear(single_input, 2, 2, "native.single_layers.0.attn.w1k.weight")
    value = linear(single_input, 2, 2, "native.single_layers.0.attn.w1v.weight")
    attended = linear(attention(query, key, value, joint, 2), 2, 2, "native.single_layers.0.attn.w1o.weight")
    combined = gated(residual, attended, 1, joint, 2, single_mod, 4)
    single_mlp = aura_mlp(modulate(layer_norm(combined, 2, f32(1e-5)), 1, joint, 2, single_mod, 6, 8), "native.single_layers.0.mlp")
    combined = gated(residual, single_mlp, 1, joint, 2, single_mod, 10)
    image = combined[text_tokens * 2:]
    final_mod = linear(time_silu, 2, 4, "native.modF.1.weight")
    image = modulate(image, 1, image_tokens, 2, final_mod, 0, 2)
    patches = linear(image, 2, 16, "native.final_linear.weight")
    return unpatchify(patches, 4, 1, height, width)


def qwen_time(time):
    basis = []
    for frequency in range(128):
        omega = f32(math.exp(f32(mul(-f32(math.log(10000.0)), f32(frequency / 128.0)))))
        angle = mul(mul(time, omega), 1000.0)
        basis.append(f32(math.cos(angle)))
    for frequency in range(128):
        omega = f32(math.exp(f32(mul(-f32(math.log(10000.0)), f32(frequency / 128.0)))))
        angle = mul(mul(time, omega), 1000.0)
        basis.append(f32(math.sin(angle)))
    first = linear(basis, 256, 128, "native.time_text_embed.timestep_embedder.linear_1.weight", "native.time_text_embed.timestep_embedder.linear_1.bias")
    return linear([silu(value) for value in first], 128, 128, "native.time_text_embed.timestep_embedder.linear_2.weight", "native.time_text_embed.timestep_embedder.linear_2.bias")


def qwen_mlp(values, stream):
    prefix = "native.transformer_blocks.0." + stream + "_mlp"
    first = linear(values, 128, 512, prefix + ".net.0.proj.weight", prefix + ".net.0.proj.bias")
    return linear([gelu(value) for value in first], 512, 128, prefix + ".net.2.weight", prefix + ".net.2.bias")


def qwen_rope(values, axes):
    output = list(values)
    dimensions = (16, 56, 56)
    for token in range(len(axes[0])):
        pair = 0
        for axis, dimension in enumerate(dimensions):
            pairs = dimension // 2
            for local_pair in range(pairs):
                angle = float(axes[axis][token]) * math.pow(10000.0, -local_pair / pairs)
                cosine = f32(math.cos(angle))
                sine = f32(math.sin(angle))
                left = output[token * 128 + pair * 2]
                right = output[token * 128 + pair * 2 + 1]
                output[token * 128 + pair * 2] = add(mul(left, cosine), -mul(right, sine))
                output[token * 128 + pair * 2 + 1] = add(mul(right, cosine), mul(left, sine))
                pair += 1
    return output


def qwen_oracle():
    height = width = 3
    image_tokens = 4
    latent = [f32((index - 72.0) * 0.005) for index in range(144)]
    image = linear(patchify(latent, 16, 1, height, width), 64, 128, "native.img_in.weight", "native.img_in.bias")
    conditioning = patterned("qwen.conditioning", 2 * 3584)
    conditioning = rms_norm(conditioning, 3584, "native.txt_norm.weight", f32(1e-6))
    text = linear(conditioning, 3584, 128, "native.txt_in.weight", "native.txt_in.bias")
    time = qwen_time(f32(0.27182818))
    time_silu = [silu(value) for value in time]
    image_mod = linear(time_silu, 128, 768, "native.transformer_blocks.0.img_mod.1.weight", "native.transformer_blocks.0.img_mod.1.bias")
    text_mod = linear(time_silu, 128, 768, "native.transformer_blocks.0.txt_mod.1.weight", "native.transformer_blocks.0.txt_mod.1.bias")
    image_residual = list(image)
    text_residual = list(text)
    image_input = modulate(layer_norm(image, 128, f32(1e-6)), 1, image_tokens, 128, image_mod, 0, 128)
    text_input = modulate(layer_norm(text, 128, f32(1e-6)), 1, 2, 128, text_mod, 0, 128)
    def projection(values, name, norm_name=None):
        result = linear(values, 128, 128, "native.transformer_blocks.0.attn." + name + ".weight", "native.transformer_blocks.0.attn." + name + ".bias")
        return rms_norm(result, 128, "native.transformer_blocks.0.attn." + norm_name + ".weight", f32(1e-6)) if norm_name else result
    image_q = projection(image_input, "to_q", "norm_q")
    image_k = projection(image_input, "to_k", "norm_k")
    image_v = projection(image_input, "to_v")
    text_q = projection(text_input, "add_q_proj", "norm_added_q")
    text_k = projection(text_input, "add_k_proj", "norm_added_k")
    text_v = projection(text_input, "add_v_proj")
    total = 6
    axes = [[0.0] * total for _ in range(3)]
    for token in range(2):
        position = f32(1 + token)
        for axis in axes:
            axis[token] = position
    for row in range(2):
        for column in range(2):
            token = 2 + row * 2 + column
            axes[0][token] = 0.0
            axes[1][token] = f32(row - 1)
            axes[2][token] = f32(column - 1)
    query = qwen_rope(text_q + image_q, axes)
    key = qwen_rope(text_k + image_k, axes)
    value = text_v + image_v
    mask = [0.0] * (total * total)
    for query_token in range(total):
        mask[query_token * total] = 0.0
        mask[query_token * total + 1] = f32(-0.75)
    attended = attention(query, key, value, total, 128, mask)
    text_attention = linear(attended[:256], 128, 128, "native.transformer_blocks.0.attn.to_add_out.weight", "native.transformer_blocks.0.attn.to_add_out.bias")
    image_attention = linear(attended[256:], 128, 128, "native.transformer_blocks.0.attn.to_out.0.weight", "native.transformer_blocks.0.attn.to_out.0.bias")
    image = gated(image_residual, image_attention, 1, image_tokens, 128, image_mod, 256)
    text = gated(text_residual, text_attention, 1, 2, 128, text_mod, 256)
    image_second = modulate(layer_norm(image, 128, f32(1e-6)), 1, image_tokens, 128, image_mod, 384, 512)
    text_second = modulate(layer_norm(text, 128, f32(1e-6)), 1, 2, 128, text_mod, 384, 512)
    image = gated(image, qwen_mlp(image_second, "img"), 1, image_tokens, 128, image_mod, 640)
    text = gated(text, qwen_mlp(text_second, "txt"), 1, 2, 128, text_mod, 640)
    final_mod = linear(time_silu, 128, 256, "native.norm_out.linear.weight", "native.norm_out.linear.bias")
    image = modulate(layer_norm(image, 128, f32(1e-6)), 1, image_tokens, 128, final_mod, 128, 0)
    patches = linear(image, 128, 64, "native.proj_out.weight", "native.proj_out.bias")
    return unpatchify(patches, 16, 1, height, width)


def digest(values):
    encoded = b"".join(struct.pack("<f", value) for value in values)
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[5]
    for relative_path, expected in PINNED_SOURCES.items():
        actual = hashlib.sha256((repository / relative_path).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"pinned source drift: {relative_path}: {actual} != {expected}")
    aura = aura_oracle()
    qwen = qwen_oracle()
    aura_sha256 = digest(aura)
    qwen_sha256 = digest(qwen)
    if aura_sha256 != EXPECTED_AURA_SHA256 or qwen_sha256 != EXPECTED_QWEN_SHA256:
        raise SystemExit(
            f"oracle drift: aura={aura_sha256}, qwen={qwen_sha256}"
        )
    print(json.dumps({
        "python": platform.python_version(),
        "platform": platform.platform(),
        "aura": aura,
        "aura_bits": [struct.unpack("<I", struct.pack("<f", value))[0] for value in aura],
        "aura_sha256": aura_sha256,
        "qwen": qwen,
        "qwen_bits": [struct.unpack("<I", struct.pack("<f", value))[0] for value in qwen],
        "qwen_sha256": qwen_sha256,
    }, separators=(",", ":")))
