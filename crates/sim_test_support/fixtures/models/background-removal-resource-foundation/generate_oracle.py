#!/usr/bin/env python3
import hashlib
import json
import math
import platform
import struct
import sys
from pathlib import Path

PINNED_SOURCES = {
    "projects/sim/ComfyUI/sim_extras/nodes_bg_removal.py": "c2cf4b42f10cfb1bb057b60a8745fb96a2462e1f1a2bd275e00795bb3f758cce",
    "projects/sim/ComfyUI/sim/bg_removal_model.py": "c4f6f7beea512c759849efa07f03f09044ea76fe5c71fb7afff31e4886e4daa7",
    "projects/sim/ComfyUI/sim/background_removal/birefnet.py": "00a083bd9a619943a7fdd1d8f827dae7734a5031ced3c37893f25ee925c670b1",
    "projects/sim/ComfyUI/sim/background_removal/birefnet.json": "50dd9639fa207a823437370b46d32a56b3f00eb1bef3bd225fe87eeeb8f255d2",
    "projects/sim/ComfyUI/sim/clip_model.py": "08be993d86c3b494b58305fb868638b4b525bbe40abead89e9c94da021716845",
    "projects/sim/ComfyUI/sim/ops.py": "9d8a4ec8357a9bfcd98dddbf06fcc2a0244643a392aacbe0970d945462c86a42",
    "projects/sim/ComfyUI/sim/model_management.py": "c2ca243c80a5262ecafe19feb15cec22d4003c16e523b5376f543f0f75acabaa",
}

INPUT_RGBA = [
    0.0, 0.1, 0.2, 0.3, 0.25, 0.5, 0.75, 0.0, 1.0, 0.0, 0.5, 1.0,
    0.5019608, 0.5, 0.49803922, 0.75, 0.003921569, 0.99607843, 0.2, 0.25,
    0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.8, 0.6, 0.4, 0.2,
    0.2, 0.4, 0.6, 0.8, 0.9, 0.7, 0.5, 0.3, 0.05, 0.15, 0.25, 0.35,
    0.35, 0.45, 0.55, 0.65, 0.65, 0.55, 0.45, 0.35, 0.95, 0.85, 0.75, 0.65,
    0.125, 0.375, 0.625, 0.875,
]
ORACLE_PLATFORM = "macOS-26.6.1-arm64-arm-64bit-Mach-O"
ORACLE_PYTHON = "3.14.5"
STATE_MUTATION = None
RELATIVE_INDEX_MUTATION = None


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def add32(left, right):
    return f32(f32(left) + f32(right))


def mul32(left, right):
    return f32(f32(left) * f32(right))


def fma32(left, right, addend):
    return f32(math.fma(f32(left), f32(right), f32(addend)))


def raw_f32_sha256(values):
    return hashlib.sha256(b"".join(struct.pack("<f", value) for value in values)).hexdigest()


class OracleTensor:
    def __init__(self, shape, values):
        self.shape = tuple(shape)
        count = math.prod(shape)
        if count != len(values):
            raise ValueError(f"shape {shape} needs {count} values, got {len(values)}")
        self.values = [f32(value) for value in values]


def state_values(key, count):
    if key.endswith("running_var") or (
        key.endswith(".weight") and (".norm" in key or ".bn" in key)
    ):
        output = [f32(1.0)] * count
    elif key == "decoder.decoder_block1.dec_att.bn1.bias":
        output = [f32(0.1)] * count
    elif key.endswith(".bias") or key.endswith("running_mean"):
        output = [f32(0.0)] * count
    else:
        seed = 2166136261
        for byte in key.encode():
            seed = ((seed * 16777619) & 0xFFFFFFFF) ^ byte
        output = []
        for index in range(count):
            lane = (seed + index * 2654435761) & 0xFFFFFFFF
            scale = (
                0.1
                if key.startswith("decoder.decoder_block1.")
                or key.startswith("bb.layers.0.blocks.0.attn.")
                else 0.001
            )
            output.append(mul32(f32((lane % 17) - 8), f32(scale)))
    if STATE_MUTATION is not None and key == STATE_MUTATION[0]:
        output[STATE_MUTATION[1]] = add32(output[STATE_MUTATION[1]], STATE_MUTATION[2])
    return output


def weight(key, shape):
    return state_values(key, math.prod(shape))


def tensor_index(shape, indices):
    index = 0
    for extent, coordinate in zip(shape, indices):
        index = index * extent + coordinate
    return index


def conv2d(tensor, prefix, output_channels, kernel, stride=1, padding=0, bias=True):
    batch, input_channels, height, width = tensor.shape
    output_height = (height + 2 * padding - kernel) // stride + 1
    output_width = (width + 2 * padding - kernel) // stride + 1
    matrix = weight(prefix + ".weight", [output_channels, input_channels, kernel, kernel])
    offsets = weight(prefix + ".bias", [output_channels]) if bias else [0.0] * output_channels
    output = [0.0] * (batch * output_channels * output_height * output_width)
    for batch_index in range(batch):
        for output_channel in range(output_channels):
            for output_y in range(output_height):
                for output_x in range(output_width):
                    value = offsets[output_channel]
                    for input_channel in range(input_channels):
                        for kernel_y in range(kernel):
                            source_y = output_y * stride + kernel_y - padding
                            if source_y < 0 or source_y >= height:
                                continue
                            for kernel_x in range(kernel):
                                source_x = output_x * stride + kernel_x - padding
                                if source_x < 0 or source_x >= width:
                                    continue
                                source = tensor.values[tensor_index(tensor.shape, [batch_index, input_channel, source_y, source_x])]
                                matrix_index = (((output_channel * input_channels + input_channel) * kernel + kernel_y) * kernel + kernel_x)
                                value = fma32(source, matrix[matrix_index], value)
                    output[tensor_index([batch, output_channels, output_height, output_width], [batch_index, output_channel, output_y, output_x])] = value
    return OracleTensor([batch, output_channels, output_height, output_width], output)


def linear(values, rows, input_width, prefix, output_width, bias=True):
    matrix = weight(prefix + ".weight", [output_width, input_width])
    offsets = weight(prefix + ".bias", [output_width]) if bias else [0.0] * output_width
    output = []
    for row in range(rows):
        for destination in range(output_width):
            value = offsets[destination]
            for source in range(input_width):
                value = fma32(values[row * input_width + source], matrix[destination * input_width + source], value)
            output.append(value)
    return output


def layer_norm(values, width, prefix):
    scales = weight(prefix + ".weight", [width])
    offsets = weight(prefix + ".bias", [width])
    output = []
    for start in range(0, len(values), width):
        row = values[start:start + width]
        mean = f32(sum(float(value) for value in row) / width)
        variance = sum((float(value) - float(mean)) ** 2 for value in row) / width
        inverse = f32(1.0 / math.sqrt(variance + float(f32(1.0e-5))))
        for component, value in enumerate(row):
            normalized = mul32(add32(value, -mean), inverse)
            output.append(add32(mul32(normalized, scales[component]), offsets[component]))
    return output


def erf_approximation(value):
    sign = -1.0 if value < 0.0 else (1.0 if value > 0.0 else 0.0)
    absolute = f32(abs(value))
    t = f32(1.0 / add32(1.0, mul32(0.3275911, absolute)))
    polynomial = add32(mul32(1.0614054, t), -1.4531521)
    polynomial = add32(mul32(polynomial, t), 1.4214138)
    polynomial = add32(mul32(polynomial, t), -0.28449672)
    polynomial = add32(mul32(polynomial, t), 0.2548296)
    polynomial = mul32(polynomial, t)
    exponential = f32(math.exp(mul32(-absolute, absolute)))
    return mul32(sign, add32(1.0, -mul32(polynomial, exponential)))


def gelu(values):
    output = []
    inverse_sqrt_two = f32(0.7071067811865476)
    for value in values:
        cdf = add32(1.0, erf_approximation(mul32(value, inverse_sqrt_two)))
        output.append(mul32(mul32(0.5, value), cdf))
    return output


def sigmoid_scalar(value):
    return f32(1.0 / add32(1.0, f32(math.exp(f32(-value)))))


def batch_norm(tensor, prefix):
    batch, channels, height, width = tensor.shape
    scales = weight(prefix + ".weight", [channels])
    offsets = weight(prefix + ".bias", [channels])
    means = weight(prefix + ".running_mean", [channels])
    variances = weight(prefix + ".running_var", [channels])
    output = [0.0] * len(tensor.values)
    for channel in range(channels):
        inverse = f32(1.0 / math.sqrt(add32(variances[channel], f32(1.0e-5))))
        for batch_index in range(batch):
            for y in range(height):
                for x in range(width):
                    index = tensor_index(tensor.shape, [batch_index, channel, y, x])
                    normalized = mul32(add32(tensor.values[index], -means[channel]), inverse)
                    output[index] = add32(mul32(normalized, scales[channel]), offsets[channel])
    return OracleTensor(tensor.shape, output)


def relu(tensor):
    return OracleTensor(tensor.shape, [max(value, 0.0) for value in tensor.values])


def sigmoid(tensor, scale=1.0):
    return OracleTensor(tensor.shape, [mul32(sigmoid_scalar(value), scale) for value in tensor.values])


def add_tensors(left, right):
    assert left.shape == right.shape
    return OracleTensor(left.shape, [add32(a, b) for a, b in zip(left.values, right.values)])


def multiply_tensors(left, right):
    if left.shape == right.shape:
        values = [mul32(a, b) for a, b in zip(left.values, right.values)]
    elif right.shape[0] == left.shape[0] and right.shape[1] == 1 and right.shape[2:] == left.shape[2:]:
        batch, channels, height, width = left.shape
        values = []
        for batch_index in range(batch):
            for channel in range(channels):
                for y in range(height):
                    for x in range(width):
                        left_index = tensor_index(left.shape, [batch_index, channel, y, x])
                        right_index = tensor_index(right.shape, [batch_index, 0, y, x])
                        values.append(mul32(left.values[left_index], right.values[right_index]))
    else:
        raise ValueError("unsupported broadcast")
    return OracleTensor(left.shape, values)


def concat_channels(tensors):
    batch, _, height, width = tensors[0].shape
    output_channels = sum(tensor.shape[1] for tensor in tensors)
    output = []
    for batch_index in range(batch):
        for tensor in tensors:
            for channel in range(tensor.shape[1]):
                for y in range(height):
                    for x in range(width):
                        output.append(tensor.values[tensor_index(tensor.shape, [batch_index, channel, y, x])])
    return OracleTensor([batch, output_channels, height, width], output)


def cubic_weight(distance):
    distance = f32(abs(distance))
    if distance <= 1.0:
        return add32(mul32(mul32(add32(mul32(1.25, distance), -2.25), distance), distance), 1.0)
    if distance < 2.0:
        return add32(mul32(add32(mul32(add32(mul32(-0.75, distance), 3.75), distance), -6.0), distance), 3.0)
    return 0.0


def axis_weights(input_extent, output_extent, output_coordinate, mode, align_corners, antialias):
    inverse_scale = f32(f32(input_extent) / f32(output_extent))
    if align_corners:
        coordinate = 0.0 if output_extent <= 1 else f32(f32(output_coordinate) * f32(input_extent - 1) / f32(output_extent - 1))
    else:
        coordinate = fma32(add32(f32(output_coordinate), 0.5), inverse_scale, -0.5)
    if antialias and output_extent < input_extent:
        filter_scale = max(inverse_scale, 1.0)
        radius = mul32(2.0 if mode == "bicubic" else 1.0, filter_scale)
        first = math.floor(add32(coordinate, -radius))
        last = math.floor(add32(coordinate, radius)) + 1
        combined = {}
        for source in range(first, last + 1):
            mapped = min(max(source, 0), input_extent - 1)
            distance = f32(add32(coordinate, -f32(source)) / filter_scale)
            value = cubic_weight(distance) if mode == "bicubic" else max(add32(1.0, -abs(distance)), 0.0)
            combined[mapped] = add32(combined.get(mapped, 0.0), value)
        total = 0.0
        for value in combined.values():
            total = add32(total, value)
        return [(source, f32(value / total)) for source, value in sorted(combined.items())]
    if mode == "bilinear":
        coordinate = min(max(coordinate, 0.0), f32(input_extent - 1))
        low = math.floor(coordinate)
        high = low + 1
        fraction = add32(coordinate, -f32(low))
        output = [(low, add32(1.0, -fraction))]
        if high < input_extent:
            output.append((high, fraction))
        return output
    low = math.floor(coordinate)
    combined = {}
    for source in range(low - 1, low + 3):
        mapped = min(max(source, 0), input_extent - 1)
        value = cubic_weight(add32(coordinate, -f32(source)))
        combined[mapped] = add32(combined.get(mapped, 0.0), value)
    return sorted(combined.items())


def resize(tensor, output_height, output_width, mode, align_corners=False, antialias=False):
    batch, channels, input_height, input_width = tensor.shape
    y_weights = [axis_weights(input_height, output_height, y, mode, align_corners, antialias) for y in range(output_height)]
    x_weights = [axis_weights(input_width, output_width, x, mode, align_corners, antialias) for x in range(output_width)]
    output = [0.0] * (batch * channels * output_height * output_width)
    for batch_index in range(batch):
        for channel in range(channels):
            for y in range(output_height):
                for x in range(output_width):
                    value = 0.0
                    for source_y, y_weight in y_weights[y]:
                        for source_x, x_weight in x_weights[x]:
                            source = tensor.values[tensor_index(tensor.shape, [batch_index, channel, source_y, source_x])]
                            value = fma32(source, mul32(y_weight, x_weight), value)
                    output[tensor_index([batch, channels, output_height, output_width], [batch_index, channel, y, x])] = value
    return OracleTensor([batch, channels, output_height, output_width], output)


def preprocess(rgba):
    rgb = []
    for pixel in range(15):
        rgb.extend(rgba[pixel * 4:pixel * 4 + 3])
    nchw = []
    for channel in range(3):
        for pixel in range(15):
            nchw.append(rgb[pixel * 3 + channel])
    tensor = resize(OracleTensor([1, 3, 3, 5], nchw), 8, 8, "bicubic", False, True)
    values = []
    for value in tensor.values:
        scaled = mul32(value, 255.0)
        clipped = min(max(scaled, 0.0), 255.0)
        rounded = f32(round(clipped))
        values.append(f32(rounded / f32(255.0)))
    return OracleTensor(tensor.shape, values)


def nchw_to_tokens(tensor):
    batch, channels, height, width = tensor.shape
    return [tensor.values[tensor_index(tensor.shape, [b, c, y, x])] for b in range(batch) for y in range(height) for x in range(width) for c in range(channels)]


def tokens_to_nchw(values, batch, height, width, channels):
    output = [0.0] * len(values)
    shape = [batch, channels, height, width]
    for b in range(batch):
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    output[tensor_index(shape, [b, c, y, x])] = values[((b * height + y) * width + x) * channels + c]
    return OracleTensor(shape, output)


def roll(tensor, shift):
    batch, channels, height, width = tensor.shape
    output = [0.0] * len(tensor.values)
    for b in range(batch):
        for c in range(channels):
            for y in range(height):
                for x in range(width):
                    destination = tensor_index(tensor.shape, [b, c, y, x])
                    source = tensor_index(tensor.shape, [b, c, (y - shift) % height, (x - shift) % width])
                    output[destination] = tensor.values[source]
    return OracleTensor(tensor.shape, output)


def relative_indices(window):
    tokens = window * window
    width = 2 * window - 1
    return [
        ((query // window) - (key // window) + window - 1) * width
        + ((query % window) - (key % window) + window - 1)
        for query in range(tokens) for key in range(tokens)
    ]


def shift_region(position, extent, window, shift):
    if shift == 0 or position < max(extent - window, 0):
        return 0
    if position < max(extent - shift, 0):
        return 1
    return 2


def attention(values, height, width, layer, block, channels, heads):
    window = 2
    shift = 0 if block % 2 == 0 else 1
    prefix = f"bb.layers.{layer}.blocks.{block}"
    normalized = layer_norm(values, channels, prefix + ".norm1")
    tensor = tokens_to_nchw(normalized, 1, height, width, channels)
    padded_height = ((height + window - 1) // window) * window
    padded_width = ((width + window - 1) // window) * window
    if (padded_height, padded_width) != (height, width):
        padded = [0.0] * (channels * padded_height * padded_width)
        for c in range(channels):
            for y in range(height):
                for x in range(width):
                    padded[tensor_index([1, channels, padded_height, padded_width], [0, c, y, x])] = tensor.values[tensor_index(tensor.shape, [0, c, y, x])]
        tensor = OracleTensor([1, channels, padded_height, padded_width], padded)
    if shift:
        tensor = roll(tensor, -shift)
    padded = nchw_to_tokens(tensor)
    windows_y = padded_height // window
    windows_x = padded_width // window
    window_count = windows_y * windows_x
    tokens = window * window
    window_values = [0.0] * (window_count * tokens * channels)
    for window_y in range(windows_y):
        for window_x in range(windows_x):
            window_index = window_y * windows_x + window_x
            for local_y in range(window):
                for local_x in range(window):
                    token = local_y * window + local_x
                    y = window_y * window + local_y
                    x = window_x * window + local_x
                    for channel in range(channels):
                        window_values[(window_index * tokens + token) * channels + channel] = padded[(y * padded_width + x) * channels + channel]
    qkv = linear(window_values, window_count * tokens, channels, prefix + ".attn.qkv", 3 * channels)
    head_dimension = channels // heads
    scale = f32(float(head_dimension) ** -0.5)
    query = [0.0] * (window_count * tokens * heads * head_dimension)
    key = [0.0] * len(query)
    projected_value = [0.0] * len(query)
    for batch in range(window_count):
        for token in range(tokens):
            for head in range(heads):
                for dimension in range(head_dimension):
                    destination = ((batch * tokens + token) * heads + head) * head_dimension + dimension
                    base = (batch * tokens + token) * (3 * channels) + head * head_dimension + dimension
                    query[destination] = mul32(qkv[base], scale)
                    key[destination] = qkv[base + channels]
                    projected_value[destination] = qkv[base + 2 * channels]
    bias = weight(prefix + ".attn.relative_position_bias_table", [(2 * window - 1) ** 2, heads])
    indices = relative_indices(window)
    if RELATIVE_INDEX_MUTATION is not None and RELATIVE_INDEX_MUTATION[:2] == (layer, block):
        index, delta = RELATIVE_INDEX_MUTATION[2:]
        indices[index] = (indices[index] + delta) % ((2 * window - 1) ** 2)
    attended = [0.0] * len(query)
    for batch in range(window_count):
        window_y, window_x = divmod(batch, windows_x)
        for head in range(heads):
            for query_token in range(tokens):
                scores = []
                query_y = window_y * window + query_token // window
                query_x = window_x * window + query_token % window
                query_region = shift_region(query_y, padded_height, window, shift) * 3 + shift_region(query_x, padded_width, window, shift)
                for key_token in range(tokens):
                    score = 0.0
                    for component in range(head_dimension):
                        q_index = ((batch * tokens + query_token) * heads + head) * head_dimension + component
                        k_index = ((batch * tokens + key_token) * heads + head) * head_dimension + component
                        score = add32(score, mul32(query[q_index], key[k_index]))
                    score = add32(score, bias[indices[query_token * tokens + key_token] * heads + head])
                    key_y = window_y * window + key_token // window
                    key_x = window_x * window + key_token % window
                    key_region = shift_region(key_y, padded_height, window, shift) * 3 + shift_region(key_x, padded_width, window, shift)
                    score = add32(score, -100.0 if shift and query_region != key_region else 0.0)
                    scores.append(score)
                maximum = max(scores)
                probabilities = [f32(math.exp(add32(score, -maximum))) for score in scores]
                denominator = 0.0
                for probability in probabilities:
                    denominator = add32(denominator, probability)
                probabilities = [f32(probability / denominator) for probability in probabilities]
                for component in range(head_dimension):
                    result = 0.0
                    for key_token, probability in enumerate(probabilities):
                        v_index = ((batch * tokens + key_token) * heads + head) * head_dimension + component
                        result = add32(result, mul32(probability, projected_value[v_index]))
                    destination = ((batch * tokens + query_token) * heads + head) * head_dimension + component
                    attended[destination] = result
    projected = linear(attended, window_count * tokens, channels, prefix + ".attn.proj", channels)
    reversed_values = [0.0] * (padded_height * padded_width * channels)
    for window_y in range(windows_y):
        for window_x in range(windows_x):
            window_index = window_y * windows_x + window_x
            for local_y in range(window):
                for local_x in range(window):
                    token = local_y * window + local_x
                    y, x = window_y * window + local_y, window_x * window + local_x
                    for channel in range(channels):
                        reversed_values[(y * padded_width + x) * channels + channel] = projected[(window_index * tokens + token) * channels + channel]
    reversed_tensor = tokens_to_nchw(reversed_values, 1, padded_height, padded_width, channels)
    if shift:
        reversed_tensor = roll(reversed_tensor, shift)
    cropped = []
    for c in range(channels):
        for y in range(height):
            for x in range(width):
                cropped.append(reversed_tensor.values[tensor_index(reversed_tensor.shape, [0, c, y, x])])
    residual = add_tensors(tokens_to_nchw(values, 1, height, width, channels), OracleTensor([1, channels, height, width], cropped))
    residual_values = nchw_to_tokens(residual)
    hidden = layer_norm(residual_values, channels, prefix + ".norm2")
    hidden = linear(hidden, height * width, channels, prefix + ".mlp.fc1", channels * 4)
    hidden = gelu(hidden)
    hidden = linear(hidden, height * width, channels * 4, prefix + ".mlp.fc2", channels)
    return nchw_to_tokens(add_tensors(residual, tokens_to_nchw(hidden, 1, height, width, channels)))


def patch_merge(values, height, width, channels, layer):
    output_height, output_width = (height + 1) // 2, (width + 1) // 2
    merged = [0.0] * (output_height * output_width * channels * 4)
    for output_y in range(output_height):
        for output_x in range(output_width):
            for part, (delta_y, delta_x) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
                source_y, source_x = output_y * 2 + delta_y, output_x * 2 + delta_x
                if source_y < height and source_x < width:
                    for channel in range(channels):
                        merged[((output_y * output_width + output_x) * channels * 4) + part * channels + channel] = values[(source_y * width + source_x) * channels + channel]
    prefix = f"bb.layers.{layer}.downsample"
    merged = layer_norm(merged, channels * 4, prefix + ".norm")
    return linear(merged, output_height * output_width, channels * 4, prefix + ".reduction", channels * 2, False), output_height, output_width


def backbone(tensor):
    projected = conv2d(tensor, "bb.patch_embed.proj", 4, 4, 4, 0, True)
    _, channels, height, width = projected.shape
    values = layer_norm(nchw_to_tokens(projected), channels, "bb.patch_embed.norm")
    outputs = []
    depths, heads = [2, 1, 1, 1], [1, 1, 2, 4]
    for layer in range(4):
        channels = 4 << layer
        for block in range(depths[layer]):
            values = attention(values, height, width, layer, block, channels, heads[layer])
        outputs.append(tokens_to_nchw(layer_norm(values, channels, f"bb.norm{layer}"), 1, height, width, channels))
        if layer < 3:
            values, height, width = patch_merge(values, height, width, channels, layer)
    return outputs


def adaptive_average(tensor):
    batch, channels, height, width = tensor.shape
    output = []
    for b in range(batch):
        for c in range(channels):
            total = 0.0
            for y in range(height):
                for x in range(width):
                    total = add32(total, tensor.values[tensor_index(tensor.shape, [b, c, y, x])])
            output.append(f32(total / f32(height * width)))
    return OracleTensor([batch, channels, 1, 1], output)


def bilinear_samples(height, width, y, x):
    if y < -1.0 or y > f32(height) or x < -1.0 or x > f32(width):
        return []
    y_low, x_low = math.floor(y), math.floor(x)
    y_fraction, x_fraction = add32(y, -f32(y_low)), add32(x, -f32(x_low))
    candidates = [
        (y_low, x_low, add32(1.0, -y_fraction), add32(1.0, -x_fraction)),
        (y_low, x_low + 1, add32(1.0, -y_fraction), x_fraction),
        (y_low + 1, x_low, y_fraction, add32(1.0, -x_fraction)),
        (y_low + 1, x_low + 1, y_fraction, x_fraction),
    ]
    return [(source_y, source_x, mul32(y_weight, x_weight)) for source_y, source_x, y_weight, x_weight in candidates if 0 <= source_y < height and 0 <= source_x < width]


def deform_conv(tensor, offset, mask, prefix, output_channels, kernel, padding):
    batch, input_channels, height, width = tensor.shape
    matrix = weight(prefix + ".weight", [output_channels, input_channels, kernel, kernel])
    output = [0.0] * (batch * output_channels * height * width)
    for b in range(batch):
        for output_channel in range(output_channels):
            for output_y in range(height):
                for output_x in range(width):
                    value = 0.0
                    for input_channel in range(input_channels):
                        for kernel_y in range(kernel):
                            for kernel_x in range(kernel):
                                kernel_index = kernel_y * kernel + kernel_x
                                offset_channel = 2 * kernel_index
                                offset_y = offset.values[tensor_index(offset.shape, [b, offset_channel, output_y, output_x])]
                                offset_x = offset.values[tensor_index(offset.shape, [b, offset_channel + 1, output_y, output_x])]
                                sample_y = add32(add32(f32(output_y - padding + kernel_y), offset_y), 0.0)
                                sample_x = add32(add32(f32(output_x - padding + kernel_x), offset_x), 0.0)
                                sample = 0.0
                                for source_y, source_x, sample_weight in bilinear_samples(height, width, sample_y, sample_x):
                                    source = tensor.values[tensor_index(tensor.shape, [b, input_channel, source_y, source_x])]
                                    sample = fma32(source, sample_weight, sample)
                                modulation = mask.values[tensor_index(mask.shape, [b, kernel_index, output_y, output_x])]
                                matrix_index = (((output_channel * input_channels + input_channel) * kernel + kernel_y) * kernel + kernel_x)
                                value = fma32(mul32(sample, modulation), matrix[matrix_index], value)
                    output[tensor_index([batch, output_channels, height, width], [b, output_channel, output_y, output_x])] = value
    return OracleTensor([batch, output_channels, height, width], output)


def deformable_branch(tensor, prefix, kernel):
    padding = kernel // 2
    offset = conv2d(tensor, prefix + ".atrous_conv.offset_conv", kernel * kernel * 2, kernel, 1, padding, True)
    mask = sigmoid(conv2d(tensor, prefix + ".atrous_conv.modulator_conv", kernel * kernel, kernel, 1, padding, True), 2.0)
    hidden = deform_conv(tensor, offset, mask, prefix + ".atrous_conv.regular_conv", 2, kernel, padding)
    return relu(batch_norm(hidden, prefix + ".bn"))


def basic_decoder(tensor, prefix, output_channels):
    hidden = relu(batch_norm(conv2d(tensor, prefix + ".conv_in", 2, 3, 1, 1, True), prefix + ".bn_in"))
    branches = [deformable_branch(hidden, prefix + ".dec_att.aspp1", 1)]
    for index, kernel in enumerate([1, 3, 7]):
        branches.append(deformable_branch(hidden, f"{prefix}.dec_att.aspp_deforms.{index}", kernel))
    pooled = adaptive_average(hidden)
    pooled = relu(batch_norm(conv2d(pooled, prefix + ".dec_att.global_avg_pool.1", 2, 1, 1, 0, False), prefix + ".dec_att.global_avg_pool.2"))
    pooled = resize(pooled, hidden.shape[2], hidden.shape[3], "bilinear", True, False)
    branches.append(pooled)
    hidden = concat_channels(branches)
    hidden = relu(batch_norm(conv2d(hidden, prefix + ".dec_att.conv1", min(output_channels, 2), 1, 1, 0, False), prefix + ".dec_att.bn1"))
    return batch_norm(conv2d(hidden, prefix + ".conv_out", output_channels, 3, 1, 1, True), prefix + ".bn_out")


def split_patches(image, target_height, target_width):
    _, channels, height, width = image.shape
    rows, columns = height // target_height, width // target_width
    patches = []
    for column in range(columns):
        for row in range(rows):
            values = []
            for channel in range(channels):
                for y in range(target_height):
                    for x in range(target_width):
                        values.append(image.values[tensor_index(image.shape, [0, channel, row * target_height + y, column * target_width + x])])
            patches.append(OracleTensor([1, channels, target_height, target_width], values))
    return concat_channels(patches)


def simple_convs(tensor, prefix, output_channels):
    return conv2d(conv2d(tensor, prefix + ".conv1", 2, 3, 1, 1, True), prefix + ".conv_out", output_channels, 3, 1, 1, True)


def decode(image, x1, x2, x3, x4):
    current = x4
    for stage, lateral, lateral_prefix, patch_output, block_output in [
        (4, x3, "decoder.lateral_block4.conv", 8, 32),
        (3, x2, "decoder.lateral_block3.conv", 8, 16),
        (2, x1, "decoder.lateral_block2.conv", 4, 8),
    ]:
        height, width = current.shape[2:]
        patches = simple_convs(resize(split_patches(image, height, width), height, width, "bilinear", True, False), f"decoder.ipt_blk{stage + 1}", patch_output)
        current = basic_decoder(concat_channels([current, patches]), f"decoder.decoder_block{stage}", block_output)
        gdt = relu(batch_norm(conv2d(current, f"decoder.gdt_convs_{stage}.0", 2, 3, 1, 1, True), f"decoder.gdt_convs_{stage}.1"))
        current = multiply_tensors(current, sigmoid(conv2d(gdt, f"decoder.gdt_convs_attn_{stage}.0", 1, 1, 1, 0, True)))
        current = resize(current, lateral.shape[2], lateral.shape[3], "bilinear", True, False)
        current = add_tensors(current, conv2d(lateral, lateral_prefix, lateral.shape[1], 1, 1, 0, True))
    height, width = current.shape[2:]
    patches = simple_convs(resize(split_patches(image, height, width), height, width, "bilinear", True, False), "decoder.ipt_blk2", 2)
    current = basic_decoder(concat_channels([current, patches]), "decoder.decoder_block1", 4)
    current = resize(current, image.shape[2], image.shape[3], "bilinear", True, False)
    patches = simple_convs(split_patches(image, image.shape[2], image.shape[3]), "decoder.ipt_blk1", 1)
    return conv2d(concat_channels([current, patches]), "decoder.conv_out1.0", 1, 1, 1, 0, True)


def birefnet(image):
    full = backbone(image)
    half = backbone(resize(image, image.shape[2] // 2, image.shape[3] // 2, "bilinear", True, False))
    features = []
    for full_feature, half_feature in zip(full, half):
        features.append(concat_channels([full_feature, resize(half_feature, full_feature.shape[2], full_feature.shape[3], "bilinear", True, False)]))
    x1, x2, x3, x4 = features
    squeezed_input = concat_channels([resize(x1, x4.shape[2], x4.shape[3], "bilinear", True, False), resize(x2, x4.shape[2], x4.shape[3], "bilinear", True, False), resize(x3, x4.shape[2], x4.shape[3], "bilinear", True, False), x4])
    squeezed = basic_decoder(squeezed_input, "squeeze_module.0", 64)
    return decode(image, x1, x2, x3, squeezed)


def main():
    global RELATIVE_INDEX_MUTATION, STATE_MUTATION
    if platform.platform() != ORACLE_PLATFORM or sys.version.split()[0] != ORACLE_PYTHON:
        raise SystemExit(
            "background-removal bit oracle requires "
            f"{ORACLE_PLATFORM} with CPython {ORACLE_PYTHON}"
        )
    repository = Path(__file__).resolve().parents[5]
    for relative, expected in PINNED_SOURCES.items():
        actual = hashlib.sha256((repository / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"pinned source drift: {relative}: {actual}")
    preprocessed = preprocess(INPUT_RGBA)

    def projected_output():
        logits = birefnet(preprocessed)
        projected = resize(logits, 3, 5, "bicubic", False, False)
        return [sigmoid_scalar(value) for value in projected.values]

    output = projected_output()
    mutation_specifications = {
        "aspp-dilated-branch": (
            "decoder.decoder_block1.dec_att.aspp_deforms.2.atrous_conv.regular_conv.weight",
            73,
            10.0,
        ),
        "aspp-global-pool": (
            "decoder.decoder_block1.dec_att.global_avg_pool.1.weight",
            1,
            0.125,
        ),
        "deform-mask": (
            "decoder.decoder_block1.dec_att.aspp_deforms.1.atrous_conv.modulator_conv.weight",
            85,
            100.0,
        ),
        "deform-offset": (
            "decoder.decoder_block1.dec_att.aspp_deforms.1.atrous_conv.offset_conv.weight",
            139,
            -100.0,
        ),
        "shifted-window-block": (
            "bb.layers.0.blocks.1.attn.proj.bias",
            0,
            1.0,
        ),
        "unused-decoder-head": ("decoder.conv_ms_spvn_4.weight", 0, 0.125),
    }
    mutations = {}
    for name, (state_key, state_index, delta) in mutation_specifications.items():
        STATE_MUTATION = (state_key, state_index, delta)
        mutated_output = projected_output()
        mutations[name] = {
            "delta": delta,
            "output_bits": [
                struct.unpack("<I", struct.pack("<f", value))[0]
                for value in mutated_output
            ],
            "raw_f32_sha256": raw_f32_sha256(mutated_output),
            "state_index": state_index,
            "state_key": state_key,
        }
    STATE_MUTATION = None
    RELATIVE_INDEX_MUTATION = (0, 0, 0, 3)
    relative_index_output = projected_output()
    RELATIVE_INDEX_MUTATION = None
    mutations["relative-position-index"] = {
        "buffer_index": 0,
        "delta": 3,
        "output_bits": [
            struct.unpack("<I", struct.pack("<f", value))[0]
            for value in relative_index_output
        ],
        "raw_f32_sha256": raw_f32_sha256(relative_index_output),
        "state_key": "bb.layers.0.blocks.0.attn.relative_position_index",
    }
    generator_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    document = {
        "format": "zed.sim.background-removal-reduced-oracle.v1",
        "generator_sha256": generator_hash,
        "generator_command": "PYTHONDONTWRITEBYTECODE=1 python3 crates/sim_test_support/fixtures/models/background-removal-resource-foundation/generate_oracle.py",
        "pinned_sources": dict(sorted(PINNED_SOURCES.items())),
        "python": ORACLE_PYTHON,
        "platform": ORACLE_PLATFORM,
        "f32_rule": "Every source f32 primitive is rounded with IEEE-754 little-endian pack/unpack; convolution, linear, interpolation, and deform accumulation use fused f32 multiply-add where the canonical source owner does. Transcendental results are bit-pinned to the recorded CPython/libm platform.",
        "input_shape": [1, 3, 5, 4],
        "preprocessed_shape": list(preprocessed.shape),
        "output_shape": [1, 3, 5],
        "raw_f32_sha256": raw_f32_sha256(output),
        "output_bits": [struct.unpack("<I", struct.pack("<f", value))[0] for value in output],
        "mutations": mutations,
    }
    output_path = Path(__file__).with_name("oracle.json")
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if len(sys.argv) == 2 and sys.argv[1] == "--check":
        if output_path.read_text() != rendered:
            raise SystemExit("background-removal oracle is stale")
        print(f"checked {output_path}")
    elif len(sys.argv) == 1:
        output_path.write_text(rendered)
        print(output_path)
    else:
        raise SystemExit("usage: generate_oracle.py [--check]")


if __name__ == "__main__":
    main()
