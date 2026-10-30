import ctypes
import math
import struct


LIBC = ctypes.CDLL(None)
LIBC.atanf.argtypes = [ctypes.c_float]
LIBC.atanf.restype = ctypes.c_float
LIBC.sinf.argtypes = [ctypes.c_float]
LIBC.sinf.restype = ctypes.c_float
LIBC.cosf.argtypes = [ctypes.c_float]
LIBC.cosf.restype = ctypes.c_float
LIBC.powf.argtypes = [ctypes.c_float, ctypes.c_float]
LIBC.powf.restype = ctypes.c_float
LIBC.expf.argtypes = [ctypes.c_float]
LIBC.expf.restype = ctypes.c_float

MASK64 = (1 << 64) - 1
RAY_POSE_RNG_ADDRESS = {
    "workflow": "task390",
    "attempt": "fixture",
    "node": "depth-anything-3-ray-pose",
    "output": 0,
    "phase": "reduced-ray-pose",
    "batch": 0,
    "retry": 0,
    "retry_policy": "replay",
    "device_kind": "cpu",
    "device_ordinal": 0,
}


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fadd(left, right):
    return f32(f32(left) + f32(right))


def fsub(left, right):
    return f32(f32(left) - f32(right))


def fmul(left, right):
    return f32(f32(left) * f32(right))


def fmadd(left, right, addend):
    return f32(math.fma(f32(left), f32(right), f32(addend)))


def fdiv(left, right):
    return f32(f32(left) / f32(right))


def fatan(value):
    return f32(LIBC.atanf(ctypes.c_float(f32(value))))


def fsin(value):
    return f32(LIBC.sinf(ctypes.c_float(f32(value))))


def fcos(value):
    return f32(LIBC.cosf(ctypes.c_float(f32(value))))


def fpow(left, right):
    return f32(LIBC.powf(ctypes.c_float(f32(left)), ctypes.c_float(f32(right))))


def fexp(value):
    return f32(LIBC.expf(ctypes.c_float(f32(value))))


def project_storage(value, source_dtype):
    if source_dtype == "f16":
        return f32(struct.unpack("<e", struct.pack("<e", f32(value)))[0])
    if source_dtype == "bf16":
        raw = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
        rounded = ((raw + 0x7FFF + ((raw >> 16) & 1)) >> 16) & 0xFFFF
        return struct.unpack("<f", struct.pack("<I", rounded << 16))[0]
    return f32(value)


def fsum(values):
    output = f32(0.0)
    for value in values:
        output = fadd(output, value)
    return output


def mix_u64(value):
    value = (value + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def rng_address_digest(seed, address=RAY_POSE_RNG_ADDRESS):
    value = 0xCBF29CE484222325

    def write(raw):
        nonlocal value
        for byte in raw:
            value ^= byte
            value = (value * 0x00000100000001B3) & MASK64

    def write_u8(item):
        write(bytes([item]))

    def write_u32(item):
        write(int(item).to_bytes(4, "little"))

    def write_u64(item):
        write(int(item).to_bytes(8, "little"))

    def write_text(item):
        encoded = item.encode()
        write_u64(len(encoded))
        write(encoded)

    write_u8(2)
    write_u64(seed)
    write_text(address["workflow"])
    write_text(address["attempt"])
    write_text(address["node"])
    write_u32(address["output"])
    write_text(address["phase"])
    write_u64(address["batch"])
    if address["retry_policy"] == "replay":
        write_u8(0)
    elif address["retry_policy"] == "advance":
        write_u8(1)
        write_u32(address["retry"])
    else:
        raise ValueError(address["retry_policy"])
    if address["device_kind"] != "cpu":
        raise ValueError(address["device_kind"])
    write_u8(0)
    write_u32(address["device_ordinal"])
    return mix_u64(value)


class Mt19937:
    def __init__(self, seed):
        self.state = [0] * 624
        self.state[0] = seed & 0xFFFFFFFF
        for index in range(1, 624):
            previous = self.state[index - 1]
            self.state[index] = (
                1812433253 * (previous ^ (previous >> 30)) + index
            ) & 0xFFFFFFFF
        self.index = 624

    def twist(self):
        for index in range(624):
            combined = (self.state[index] & 0x80000000) | (
                self.state[(index + 1) % 624] & 0x7FFFFFFF
            )
            value = self.state[(index + 397) % 624] ^ (combined >> 1)
            if combined & 1:
                value ^= 0x9908B0DF
            self.state[index] = value & 0xFFFFFFFF
        self.index = 0

    def next_u32(self):
        if self.index >= 624:
            self.twist()
        value = self.state[self.index]
        self.index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C5680
        value ^= (value << 15) & 0xEFC60000
        return (value ^ (value >> 18)) & 0xFFFFFFFF

    def next_bounded_u64(self, upper_exclusive):
        if upper_exclusive <= 0:
            raise ValueError(upper_exclusive)
        rejection_threshold = ((-upper_exclusive) & MASK64) % upper_exclusive
        while True:
            candidate = (self.next_u32() << 32) | self.next_u32()
            if candidate >= rejection_threshold:
                return candidate % upper_exclusive

    def randperm(self, count):
        values = list(range(count))
        for upper in range(count - 1, 0, -1):
            selected = self.next_bounded_u64(upper + 1)
            values[upper], values[selected] = values[selected], values[upper]
        return values


def ransac_generator(seed=17, address=RAY_POSE_RNG_ADDRESS):
    return Mt19937(seed ^ rng_address_digest(seed, address))


def ransac_samples(candidate_count, seed=17, address=RAY_POSE_RNG_ADDRESS):
    generator = ransac_generator(seed, address)
    return [generator.randperm(candidate_count)[:8] for _ in range(100)]


def product(values):
    output = 1
    for value in values:
        output *= value
    return output


def round_to_patch(value, patch):
    down = value // patch * patch
    up = down + patch
    return up if abs(up - value) <= abs(value - down) else down


class Tensor:
    def __init__(self, shape, values):
        self.shape = tuple(shape)
        self.values = [f32(value) for value in values]
        if product(self.shape) != len(self.values):
            raise ValueError((self.shape, len(self.values)))

    def offset(self, indices):
        if len(indices) != len(self.shape):
            raise ValueError((indices, self.shape))
        output = 0
        for index, dimension in zip(indices, self.shape):
            if index < 0 or index >= dimension:
                raise IndexError((indices, self.shape))
            output = output * dimension + index
        return output

    def get(self, *indices):
        return self.values[self.offset(indices)]

    def set(self, value, *indices):
        self.values[self.offset(indices)] = f32(value)

    def clone(self):
        return Tensor(self.shape, self.values)


def state_values(state_index, count, key):
    is_norm_weight = key.endswith(".weight") and (".norm" in key or "layernorm" in key)
    is_layer_scale = key.endswith(".lambda1") or key.endswith(".gamma")
    values = []
    for value_index in range(count):
        if is_norm_weight:
            value = 1.0
        elif is_layer_scale:
            value = 0.125
        elif key.endswith(".bias"):
            value = 0.0
        else:
            lane = ((state_index * 17 + value_index * 13) % 29) - 14
            value = fmul(f32(lane), f32(0.0025))
        values.append(f32(value))
    if key == "native.cam_dec.fc_qvec.bias":
        values[3] = f32(1.0)
    if key == "native.cam_dec.fc_fov.0.bias":
        values[:] = [f32(0.75), f32(1.0)]
    if key == "native.head.scratch.output_conv2_aux.3.5.weight":
        values[194] = fadd(values[194], 1.0)
    if key == "native.head.scratch.output_conv2_aux.3.5.bias":
        values[2] = f32(1.0e-4)
    return values


def manifest(profile):
    dual = profile == "dualdpt"
    states = []

    def add(key, shape):
        states.append((key, tuple(shape)))

    def affine(prefix, size):
        add(prefix + ".weight", [size])
        add(prefix + ".bias", [size])

    def linear(prefix, output, input_size, bias=True):
        add(prefix + ".weight", [output, input_size])
        if bias:
            add(prefix + ".bias", [output])

    def convolution(prefix, output, input_size, kernel, bias=True):
        add(prefix + ".weight", [output, input_size, kernel, kernel])
        if bias:
            add(prefix + ".bias", [output])

    def transposed(prefix, input_size, output, kernel, bias=True):
        add(prefix + ".weight", [input_size, output, kernel, kernel])
        if bias:
            add(prefix + ".bias", [output])

    def refine(prefix, features, residual):
        if residual:
            for name in ["conv1", "conv2"]:
                convolution(prefix + ".resConfUnit1." + name, features, features, 3)
        for name in ["conv1", "conv2"]:
            convolution(prefix + ".resConfUnit2." + name, features, features, 3)
        convolution(prefix + ".out_conv", features, features, 1)

    hidden = 4
    convolution("native.backbone.embeddings.patch_embeddings.projection", hidden, 3, 2)
    add("native.backbone.embeddings.position_embeddings", [1, 5, hidden])
    add("native.backbone.embeddings.cls_token", [1, 1, hidden])
    if dual:
        add("native.backbone.embeddings.camera_token", [1, 2, hidden])
    for layer in range(4):
        prefix = f"native.backbone.encoder.layer.{layer}"
        affine(prefix + ".norm1", hidden)
        for name in ["query", "key", "value"]:
            linear(prefix + ".attention.attention." + name, hidden, hidden)
        linear(prefix + ".attention.output.dense", hidden, hidden)
        if layer >= 1:
            affine(prefix + ".attention.q_norm", hidden)
            affine(prefix + ".attention.k_norm", hidden)
        add(prefix + ".layer_scale1.lambda1", [hidden])
        add(prefix + ".layer_scale2.lambda1", [hidden])
        affine(prefix + ".norm2", hidden)
        linear(prefix + ".mlp.fc1", hidden * 4, hidden)
        linear(prefix + ".mlp.fc2", hidden, hidden * 4)
    affine("native.backbone.layernorm", hidden)
    input_size = 8 if dual else 4
    features = 8 if dual else 4
    if dual:
        affine("native.head.norm", input_size)
    projected = 4 if dual else 2
    for index in range(4):
        convolution(f"native.head.projects.{index}", projected, input_size, 1)
    transposed("native.head.resize_layers.0", projected, projected, 4)
    transposed("native.head.resize_layers.1", projected, projected, 2)
    convolution("native.head.resize_layers.3", projected, projected, 3)
    for index in range(4):
        convolution(f"native.head.scratch.layer{index + 1}_rn", features, projected, 3, False)
    for index in range(1, 5):
        refine(f"native.head.scratch.refinenet{index}", features, index != 4)
    convolution("native.head.scratch.output_conv1", features // 2, features, 3)
    convolution("native.head.scratch.output_conv2.0", 32, features // 2, 3)
    convolution("native.head.scratch.output_conv2.2", 2 if dual else 1, 32, 1)
    if not dual:
        convolution("native.head.scratch.sky_output_conv2.0", 32, features // 2, 3)
        convolution("native.head.scratch.sky_output_conv2.2", 1, 32, 1)
    if dual:
        for index in range(1, 5):
            refine(f"native.head.scratch.refinenet{index}_aux", features, index != 4)
        for level in range(4):
            prefix = f"native.head.scratch.output_conv1_aux.{level}"
            for index, (output, input_size) in enumerate([(features // 2, features), (features, features // 2), (features // 2, features), (features, features // 2), (features // 2, features)]):
                convolution(f"{prefix}.{index}", output, input_size, 3)
            prefix = f"native.head.scratch.output_conv2_aux.{level}"
            convolution(prefix + ".0", 32, features // 2, 3)
            affine(prefix + ".2", 32)
            convolution(prefix + ".5", 7, 32, 1)
        affine("native.cam_enc.token_norm", hidden)
        affine("native.cam_enc.trunk_norm", hidden)
        linear("native.cam_enc.pose_branch.fc1", hidden // 2, 9)
        linear("native.cam_enc.pose_branch.fc2", hidden, hidden // 2)
        for block in range(4):
            prefix = f"native.cam_enc.trunk.{block}"
            affine(prefix + ".norm1", hidden)
            linear(prefix + ".attn.qkv", hidden * 3, hidden)
            linear(prefix + ".attn.proj", hidden, hidden)
            add(prefix + ".ls1.gamma", [hidden])
            affine(prefix + ".norm2", hidden)
            linear(prefix + ".mlp.fc1", hidden * 4, hidden)
            linear(prefix + ".mlp.fc2", hidden, hidden * 4)
            add(prefix + ".ls2.gamma", [hidden])
        dimension = 8
        linear("native.cam_dec.backbone.0", dimension, dimension)
        linear("native.cam_dec.backbone.2", dimension, dimension)
        linear("native.cam_dec.fc_t", 3, dimension)
        linear("native.cam_dec.fc_qvec", 4, dimension)
        linear("native.cam_dec.fc_fov.0", 2, dimension)
    return states


def make_state(profile, mutation=None, source_dtype="f32"):
    output = {}
    for index, (key, shape) in enumerate(manifest(profile)):
        output[key] = Tensor(shape, [project_storage(value, source_dtype) for value in state_values(index, product(shape), key)])
    if mutation is not None:
        key, lane, delta = mutation
        output[key].values[lane] = fadd(output[key].values[lane], delta)
    return output


def linear(input_tensor, weight, bias):
    rows = product(input_tensor.shape[:-1])
    input_size = input_tensor.shape[-1]
    output_size = weight.shape[0]
    output = Tensor(input_tensor.shape[:-1] + (output_size,), [0.0] * (rows * output_size))
    for row in range(rows):
        for out_lane in range(output_size):
            value = bias.get(out_lane) if bias is not None else f32(0.0)
            for lane in range(input_size):
                value = fmadd(
                    input_tensor.values[row * input_size + lane],
                    weight.get(out_lane, lane),
                    value,
                )
            output.values[row * output_size + out_lane] = value
    return output


def layer_norm(input_tensor, weight, bias, epsilon):
    width = input_tensor.shape[-1]
    rows = product(input_tensor.shape[:-1])
    output = input_tensor.clone()
    for row in range(rows):
        values = input_tensor.values[row * width:(row + 1) * width]
        mean = f32(sum(float(value) for value in values) / width)
        variance = (
            sum((float(value) - float(mean)) ** 2 for value in values) / width
        )
        inverse = f32(1.0 / math.sqrt(variance + float(epsilon)))
        for lane, value in enumerate(values):
            normalized = fmul(fsub(value, mean), inverse)
            output.values[row * width + lane] = fadd(
                fmul(normalized, weight.get(lane)), bias.get(lane)
            )
    return output


def erf_approximation(value):
    value = f32(value)
    if math.isnan(value):
        return value
    if math.isinf(value):
        return f32(math.copysign(1.0, value))
    sign = f32(math.copysign(1.0, value))
    absolute = abs(value)
    t = fdiv(1.0, fadd(1.0, fmul(f32(0.327_591_1), absolute)))
    polynomial = fsub(fmul(f32(1.061_405_4), t), f32(1.453_152_1))
    polynomial = fadd(fmul(polynomial, t), f32(1.421_413_8))
    polynomial = fsub(fmul(polynomial, t), f32(0.284_496_72))
    polynomial = fadd(fmul(polynomial, t), f32(0.254_829_6))
    polynomial = fmul(polynomial, t)
    exponential = fexp(fmul(fmul(-1.0, absolute), absolute))
    return fmul(sign, fsub(1.0, fmul(polynomial, exponential)))


def gelu(input_tensor):
    coefficient = f32(2.0 ** -0.5)
    output = []
    for value in input_tensor.values:
        inner = erf_approximation(fmul(value, coefficient))
        output.append(fmul(fmul(0.5, value), fadd(1.0, inner)))
    return Tensor(input_tensor.shape, output)


def relu(input_tensor):
    return Tensor(input_tensor.shape, [value if value > 0.0 else 0.0 for value in input_tensor.values])


def add(left, right):
    return Tensor(left.shape, [fadd(a, b) for a, b in zip(left.values, right.values)])


def multiply_channels(input_tensor, weight):
    width = input_tensor.shape[-1]
    return Tensor(input_tensor.shape, [fmul(value, weight.get(index % width)) for index, value in enumerate(input_tensor.values)])


def conv2d(input_tensor, weight, bias=None, stride=1, padding=0):
    batch, input_channels, input_height, input_width = input_tensor.shape
    output_channels, _, kernel_height, kernel_width = weight.shape
    output_height = (input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1
    output = Tensor((batch, output_channels, output_height, output_width), [0.0] * (batch * output_channels * output_height * output_width))
    for batch_index in range(batch):
        for output_channel in range(output_channels):
            for output_y in range(output_height):
                for output_x in range(output_width):
                    value = bias.get(output_channel) if bias is not None else f32(0.0)
                    for input_channel in range(input_channels):
                        for kernel_y in range(kernel_height):
                            input_y = output_y * stride + kernel_y - padding
                            if input_y < 0 or input_y >= input_height:
                                continue
                            for kernel_x in range(kernel_width):
                                input_x = output_x * stride + kernel_x - padding
                                if input_x < 0 or input_x >= input_width:
                                    continue
                                value = fmadd(
                                    input_tensor.get(
                                        batch_index, input_channel, input_y, input_x
                                    ),
                                    weight.get(
                                        output_channel,
                                        input_channel,
                                        kernel_y,
                                        kernel_x,
                                    ),
                                    value,
                                )
                    output.set(value, batch_index, output_channel, output_y, output_x)
    return output


def conv_transpose2d(input_tensor, weight, bias=None, stride=1, padding=0):
    batch, input_channels, input_height, input_width = input_tensor.shape
    _, output_channels, kernel_height, kernel_width = weight.shape
    output_height = (input_height - 1) * stride - 2 * padding + kernel_height
    output_width = (input_width - 1) * stride - 2 * padding + kernel_width
    output = Tensor((batch, output_channels, output_height, output_width), [0.0] * (batch * output_channels * output_height * output_width))
    if bias is not None:
        for batch_index in range(batch):
            for channel in range(output_channels):
                for y in range(output_height):
                    for x in range(output_width):
                        output.set(bias.get(channel), batch_index, channel, y, x)
    for batch_index in range(batch):
        for input_channel in range(input_channels):
            for input_y in range(input_height):
                for input_x in range(input_width):
                    source = input_tensor.get(batch_index, input_channel, input_y, input_x)
                    for output_channel in range(output_channels):
                        for kernel_y in range(kernel_height):
                            output_y = input_y * stride + kernel_y - padding
                            if output_y < 0 or output_y >= output_height:
                                continue
                            for kernel_x in range(kernel_width):
                                output_x = input_x * stride + kernel_x - padding
                                if output_x < 0 or output_x >= output_width:
                                    continue
                                output.set(
                                    fmadd(
                                        source,
                                        weight.get(
                                            input_channel,
                                            output_channel,
                                            kernel_y,
                                            kernel_x,
                                        ),
                                        output.get(
                                            batch_index,
                                            output_channel,
                                            output_y,
                                            output_x,
                                        ),
                                    ),
                                    batch_index,
                                    output_channel,
                                    output_y,
                                    output_x,
                                )
    return output


def bilinear(input_tensor, output_height, output_width, align_corners):
    batch, channels, input_height, input_width = input_tensor.shape
    output = Tensor((batch, channels, output_height, output_width), [0.0] * (batch * channels * output_height * output_width))

    def axis_weights(input_extent, output_extent, output_coordinate):
        inverse_scale = f32(input_extent / output_extent)
        if align_corners and output_extent > 1:
            coordinate = f32(
                f32(output_coordinate)
                * f32(input_extent - 1)
                / f32(output_extent - 1)
            )
        else:
            coordinate = fmadd(fadd(f32(output_coordinate), 0.5), inverse_scale, -0.5)
        coordinate = max(0.0, min(input_extent - 1, coordinate))
        low = int(math.floor(coordinate))
        high = min(low + 1, input_extent - 1)
        if low == high:
            return [(low, f32(1.0))]
        high_weight = f32(coordinate - low)
        return [(low, fsub(1.0, high_weight)), (high, high_weight)]

    for batch_index in range(batch):
        for channel in range(channels):
            for output_y in range(output_height):
                y_weights = axis_weights(input_height, output_height, output_y)
                for output_x in range(output_width):
                    value = f32(0.0)
                    for source_y, weight_y in y_weights:
                        for source_x, weight_x in axis_weights(
                            input_width, output_width, output_x
                        ):
                            value = fmadd(
                                input_tensor.get(
                                    batch_index, channel, source_y, source_x
                                ),
                                fmul(weight_y, weight_x),
                                value,
                            )
                    output.set(value, batch_index, channel, output_y, output_x)
    return output


def convolution(state, input_tensor, prefix, stride=1, padding=0, transposed=False):
    weight = state[prefix + ".weight"]
    bias = state.get(prefix + ".bias")
    return conv_transpose2d(input_tensor, weight, bias, stride, padding) if transposed else conv2d(input_tensor, weight, bias, stride, padding)


def linear_state(state, input_tensor, prefix):
    return linear(input_tensor, state[prefix + ".weight"], state.get(prefix + ".bias"))


def norm_state(state, input_tensor, prefix, epsilon):
    return layer_norm(input_tensor, state[prefix + ".weight"], state[prefix + ".bias"], epsilon)


def attention(query, key, value, batch, tokens, heads, head_dimension):
    output = Tensor((batch, tokens, heads * head_dimension), [0.0] * (batch * tokens * heads * head_dimension))
    scale = f32((head_dimension ** -0.5))
    for batch_index in range(batch):
        for head in range(heads):
            for query_index in range(tokens):
                scores = []
                for key_index in range(tokens):
                    score = f32(0.0)
                    for lane in range(head_dimension):
                        offset = head * head_dimension + lane
                        score = fadd(score, fmul(query.get(batch_index, query_index, offset), key.get(batch_index, key_index, offset)))
                    scores.append(fmul(score, scale))
                maximum = max(scores)
                exponentials = [fexp(fsub(score, maximum)) for score in scores]
                denominator = fsum(exponentials)
                probabilities = [fdiv(item, denominator) for item in exponentials]
                for lane in range(head_dimension):
                    result = f32(0.0)
                    for key_index in range(tokens):
                        result = fadd(result, fmul(probabilities[key_index], value.get(batch_index, key_index, head * head_dimension + lane)))
                    output.set(result, batch_index, query_index, head * head_dimension + lane)
    return output


def rotary(values, batch, tokens, positions_y, positions_x):
    output = values.clone()
    for batch_index in range(batch):
        for token in range(tokens):
            row = batch_index * tokens + token
            for offset, position in [(0, positions_y[token]), (2, positions_x[token])]:
                cosine = f32(math.cos(f32(position)))
                sine = f32(math.sin(f32(position)))
                first = values.values[row * 4 + offset]
                second = values.values[row * 4 + offset + 1]
                output.values[row * 4 + offset] = fsub(fmul(first, cosine), fmul(second, sine))
                output.values[row * 4 + offset + 1] = fadd(fmul(second, cosine), fmul(first, sine))
    return output


def block(state, input_tensor, layer, patch_height, patch_width, view_groups=1, global_positions=False):
    prefix = f"native.backbone.encoder.layer.{layer}"
    normalized = norm_state(state, input_tensor, prefix + ".norm1", f32(1.0e-6))
    query = linear_state(state, normalized, prefix + ".attention.attention.query")
    key = linear_state(state, normalized, prefix + ".attention.attention.key")
    value = linear_state(state, normalized, prefix + ".attention.attention.value")
    if layer >= 1:
        query = norm_state(state, query, prefix + ".attention.q_norm", f32(1.0e-6))
        key = norm_state(state, key, prefix + ".attention.k_norm", f32(1.0e-6))
        positions_y = []
        positions_x = []
        for _ in range(view_groups):
            positions_y.append(0)
            positions_x.append(0)
            for index in range(patch_height * patch_width):
                positions_y.append(1 if global_positions else index // patch_width + 1)
                positions_x.append(1 if global_positions else index % patch_width + 1)
        query = rotary(query, input_tensor.shape[0], input_tensor.shape[1], positions_y, positions_x)
        key = rotary(key, input_tensor.shape[0], input_tensor.shape[1], positions_y, positions_x)
    attended = attention(query, key, value, input_tensor.shape[0], input_tensor.shape[1], 1, 4)
    attended = linear_state(state, attended, prefix + ".attention.output.dense")
    residual = add(input_tensor, multiply_channels(attended, state[prefix + ".layer_scale1.lambda1"]))
    normalized = norm_state(state, residual, prefix + ".norm2", f32(1.0e-6))
    hidden = gelu(linear_state(state, normalized, prefix + ".mlp.fc1"))
    hidden = linear_state(state, hidden, prefix + ".mlp.fc2")
    return add(residual, multiply_channels(hidden, state[prefix + ".layer_scale2.lambda1"]))


def pillow_weight(value):
    value = abs(value)
    if value == 0.0:
        return 1.0
    if value >= 3.0:
        return 0.0
    return math.sin(math.pi * value) * math.sin(math.pi * value / 3.0) / (math.pi * math.pi * value * value / 3.0)


def pillow_coefficients(output, input_size, output_size):
    scale = input_size / output_size
    filter_scale = max(scale, 1.0)
    support = 3.0 * filter_scale
    center = (output + 0.5) * scale
    start = max(0, int(center - support + 0.5))
    end = min(input_size, int(center + support + 0.5))
    normalization = sum(pillow_weight((source - center + 0.5) / filter_scale) for source in range(start, end))
    weights = []
    for source in range(start, end):
        normalized = pillow_weight((source - center + 0.5) / filter_scale) / normalization
        weights.append((source, int(normalized * (1 << 22) + 0.5)))
    return weights


def pillow_lanczos(values, batch, input_height, input_width, output_height, output_width):
    horizontal = [pillow_coefficients(x, input_width, output_width) for x in range(output_width)]
    vertical = [pillow_coefficients(y, input_height, output_height) for y in range(output_height)]
    output = [0.0] * (batch * output_height * output_width * 3)
    bias = 1 << 21
    for batch_index in range(batch):
        for y in range(output_height):
            for x in range(output_width):
                for channel in range(3):
                    vertical_sum = bias
                    for source_y, y_weight in vertical[y]:
                        horizontal_sum = bias
                        for source_x, x_weight in horizontal[x]:
                            value = values[((batch_index * input_height + source_y) * input_width + source_x) * 3 + channel]
                            byte = max(0, min(255, int(fmul(value, 255.0))))
                            horizontal_sum += byte * x_weight
                        horizontal_byte = max(0, min(255, horizontal_sum >> 22))
                        vertical_sum += horizontal_byte * y_weight
                    output[((batch_index * output_height + y) * output_width + x) * 3 + channel] = fdiv(max(0, min(255, vertical_sum >> 22)), 255.0)
    return output


def preprocess(values, batch=1, height=4, width=4, target_height=None, target_width=None):
    target_height = height if target_height is None else target_height
    target_width = width if target_width is None else target_width
    if (height, width) != (target_height, target_width):
        values = pillow_lanczos(values, batch, height, width, target_height, target_width)
    output = Tensor((batch, 3, target_height, target_width), [0.0] * (batch * 3 * target_height * target_width))
    means = [0.485, 0.456, 0.406]
    deviations = [0.229, 0.224, 0.225]
    for batch_index in range(batch):
        for y in range(target_height):
            for x in range(target_width):
                for channel in range(3):
                    source = values[((batch_index * target_height + y) * target_width + x) * 3 + channel]
                    output.set(fdiv(fsub(min(1.0, max(0.0, source)), means[channel]), deviations[channel]), batch_index, channel, y, x)
    return output


def concatenate_channels(left, right):
    rows = product(left.shape[:-1])
    channels = left.shape[-1]
    values = []
    for row in range(rows):
        values.extend(left.values[row * channels:(row + 1) * channels])
        values.extend(right.values[row * channels:(row + 1) * channels])
    return Tensor(left.shape[:-1] + (channels * 2,), values)


def select_reference(values, views, strategy):
    if strategy == "first":
        return 0
    if strategy == "middle":
        return views // 2
    channels = values.shape[-1]
    tokens = values.shape[1]
    classes = [values.values[(view * tokens) * channels:(view * tokens + 1) * channels] for view in range(views)]
    normalized = []
    norms = []
    variances = []
    for row in classes:
        norm = f32(math.sqrt(fsum(fmul(value, value) for value in row)))
        norms.append(norm)
        normalized_row = [fdiv(value, norm) for value in row]
        normalized.append(normalized_row)
        mean = fdiv(fsum(normalized_row), channels)
        variances.append(fdiv(fsum(fmul(fsub(value, mean), fsub(value, mean)) for value in normalized_row), max(1, channels - 1)))
    similarities = []
    ranges = []
    for view in range(views):
        pairwise = []
        for other in range(views):
            dot = fsum(fmul(normalized[view][lane], normalized[other][lane]) for lane in range(channels))
            pairwise.append(fsub(dot, 1.0 if view == other else 0.0))
        similarities.append(fdiv(fsum(pairwise), max(1, views - 1)))
        ranges.append(fsub(max(pairwise), min(pairwise)))
    if strategy == "saddle_sim_range":
        return max(range(views), key=lambda index: (ranges[index], -index))

    def normalized_metric(items):
        minimum = min(items)
        denominator = fadd(fsub(max(items), minimum), 1.0e-8)
        return [fdiv(fsub(value, minimum), denominator) for value in items]

    similarities = normalized_metric(similarities)
    norms = normalized_metric(norms)
    variances = normalized_metric(variances)
    scores = [fadd(fadd(abs(fsub(similarities[index], 0.5)), abs(fsub(norms[index], 0.5))), abs(fsub(variances[index], 0.5))) for index in range(views)]
    return min(range(views), key=lambda index: (scores[index], index))


def reorder_views(values, views, reference, restore):
    tokens, channels = values.shape[1:]
    output = Tensor(values.shape, [0.0] * len(values.values))
    stride = tokens * channels
    for destination in range(views):
        if restore:
            source = destination + 1 if destination < reference else (0 if destination == reference else destination)
        else:
            source = reference if destination == 0 else (destination - 1 if destination <= reference else destination)
        output.values[destination * stride:(destination + 1) * stride] = values.values[source * stride:(source + 1) * stride]
    return output


def final_norm(state, values, hidden):
    if values.shape[-1] == hidden:
        return norm_state(state, values, "native.backbone.layernorm", f32(1.0e-6))
    left_values = []
    right_values = []
    for row in range(product(values.shape[:-1])):
        left_values.extend(values.values[row * hidden * 2:row * hidden * 2 + hidden])
        right_values.extend(values.values[row * hidden * 2 + hidden:(row + 1) * hidden * 2])
    left = Tensor(values.shape[:-1] + (hidden,), left_values)
    right = norm_state(state, Tensor(values.shape[:-1] + (hidden,), right_values), "native.backbone.layernorm", f32(1.0e-6))
    return concatenate_channels(left, right)


def cubic_weight(distance):
    distance = abs(f32(distance))
    if distance <= 1.0:
        return fadd(fmul(fmul(fsub(fmul(1.25, distance), 2.25), distance), distance), 1.0)
    if distance < 2.0:
        return fadd(fmul(fsub(fmul(fadd(fmul(-0.75, distance), 3.75), distance), 6.0), distance), 3.0)
    return f32(0.0)


def cubic_axis(input_extent, output_coordinate, inverse_scale):
    coordinate = fmadd(fadd(output_coordinate, 0.5), inverse_scale, -0.5)
    low = math.floor(coordinate)
    combined = {}
    for source in range(low - 1, low + 3):
        mapped = max(0, min(input_extent - 1, source))
        combined[mapped] = fadd(combined.get(mapped, 0.0), cubic_weight(fsub(coordinate, source)))
    return sorted(combined.items())


def interpolate_positions(positions, target_height, target_width):
    hidden = positions.shape[-1]
    if (target_height, target_width) == (2, 2):
        return positions.clone()
    output = Tensor((1, target_height * target_width + 1, hidden), [0.0] * ((target_height * target_width + 1) * hidden))
    for channel in range(hidden):
        output.set(positions.get(0, 0, channel), 0, 0, channel)
    inverse_y = f32(1.0 / ((target_height + 0.1) / 2.0))
    inverse_x = f32(1.0 / ((target_width + 0.1) / 2.0))
    for y in range(target_height):
        y_weights = cubic_axis(2, y, inverse_y)
        for x in range(target_width):
            x_weights = cubic_axis(2, x, inverse_x)
            for channel in range(hidden):
                value = f32(0.0)
                for source_y, y_weight in y_weights:
                    for source_x, x_weight in x_weights:
                        weight = fmul(y_weight, x_weight)
                        source_token = source_y * 2 + source_x + 1
                        value = fmadd(positions.get(0, source_token, channel), weight, value)
                output.set(value, 0, y * target_width + x + 1, channel)
    return output


def backbone(state, image, profile="dpt", views=1, reference_strategy="first", camera_token=None):
    patch = convolution(state, image, "native.backbone.embeddings.patch_embeddings.projection", 2, 0)
    batch, hidden, patch_height, patch_width = patch.shape
    patches = patch_height * patch_width
    positions = interpolate_positions(state["native.backbone.embeddings.position_embeddings"], patch_height, patch_width)
    cls = state["native.backbone.embeddings.cls_token"]
    values = Tensor((batch, patches + 1, hidden), [0.0] * (batch * (patches + 1) * hidden))
    for batch_index in range(batch):
        for channel in range(hidden):
            values.set(fadd(cls.get(0, 0, channel), positions.get(0, 0, channel)), batch_index, 0, channel)
        for y in range(patch_height):
            for x in range(patch_width):
                token = y * patch_width + x + 1
                for channel in range(hidden):
                    values.set(fadd(patch.get(batch_index, channel, y, x), positions.get(0, token, channel)), batch_index, token, channel)
    outputs = []
    local_values = values.clone()
    reference = None
    for layer in range(4):
        if profile == "dualdpt" and layer + 1 == 2 and views >= 3 and camera_token is None:
            reference = select_reference(values, views, reference_strategy)
            values = reorder_views(values, views, reference, False)
            local_values = reorder_views(local_values, views, reference, False)
        if profile == "dualdpt" and layer == 2:
            learned = state["native.backbone.embeddings.camera_token"]
            for view in range(views):
                source = camera_token.values[view * hidden:(view + 1) * hidden] if camera_token is not None else learned.values[(0 if view == 0 else hidden):(hidden if view == 0 else hidden * 2)]
                values.values[(view * (patches + 1)) * hidden:(view * (patches + 1) + 1) * hidden] = source
        global_block = profile == "dualdpt" and layer >= 2 and layer % 2 == 1
        if global_block:
            flattened = Tensor((batch // views, views * (patches + 1), hidden), values.values)
            transformed = block(state, flattened, layer, patch_height, patch_width, views, True)
            values = Tensor((batch, patches + 1, hidden), transformed.values)
        else:
            values = block(state, values, layer, patch_height, patch_width)
            local_values = values.clone()
        output_values = concatenate_channels(local_values, values) if profile == "dualdpt" else values.clone()
        if reference is not None:
            output_values = reorder_views(output_values, views, reference, True)
        camera_values = []
        for view in range(batch):
            start = view * (patches + 1) * output_values.shape[-1]
            camera_values.extend(output_values.values[start:start + output_values.shape[-1]])
        normalized = final_norm(state, output_values, hidden)
        patches_only = []
        output_channels = normalized.shape[-1]
        for batch_index in range(batch):
            for token in range(1, patches + 1):
                for channel in range(output_channels):
                    patches_only.append(normalized.get(batch_index, token, channel))
        outputs.append((Tensor((batch, patches, normalized.shape[-1]), patches_only), Tensor((batch, normalized.shape[-1]), camera_values)))
    return outputs, patch_height, patch_width


def tokens_to_nchw(tokens, height, width):
    batch, _, channels = tokens.shape
    output = Tensor((batch, channels, height, width), [0.0] * (batch * channels * height * width))
    for batch_index in range(batch):
        for y in range(height):
            for x in range(width):
                token = y * width + x
                for channel in range(channels):
                    output.set(tokens.get(batch_index, token, channel), batch_index, channel, y, x)
    return output


def residual_unit(state, input_tensor, prefix):
    output = convolution(state, relu(input_tensor), prefix + ".conv1", 1, 1)
    output = convolution(state, relu(output), prefix + ".conv2", 1, 1)
    return add(output, input_tensor)


def fusion(state, input_tensor, residual, prefix, target=None):
    output = input_tensor
    if residual is not None:
        output = add(output, residual_unit(state, residual, prefix + ".resConfUnit1"))
    output = residual_unit(state, output, prefix + ".resConfUnit2")
    if target is None:
        target = (output.shape[2] * 2, output.shape[3] * 2)
    output = bilinear(output, target[0], target[1], True)
    return convolution(state, output, prefix + ".out_conv")


def position_embedding(input_tensor, source_width, source_height, lane_trace=None):
    batch, channels, height, width = input_tensor.shape
    aspect = source_width / source_height
    diagonal = math.sqrt(aspect * aspect + 1.0)
    span_x = aspect / diagonal
    span_y = 1.0 / diagonal
    left = -span_x * (width - 1) / width
    right = span_x * (width - 1) / width
    top = -span_y * (height - 1) / height
    bottom = span_y * (height - 1) / height
    xs = [f32(left + (right - left) * index / (width - 1)) if width > 1 else f32(left) for index in range(width)]
    ys = [f32(top + (bottom - top) * index / (height - 1)) if height > 1 else f32(top) for index in range(height)]
    output = input_tensor.clone()
    axis_channels = channels // 2
    half = axis_channels // 2
    for batch_index in range(batch):
        for channel in range(channels):
            local = channel if channel < axis_channels else channel - axis_channels
            exponent_ratio = fdiv(f32(local % half), f32(half))
            positive_power = fpow(100.0, exponent_ratio)
            reciprocal = fdiv(1.0, positive_power)
            negative_power = fpow(100.0, f32(-exponent_ratio))
            frequency = reciprocal
            for y in range(height):
                for x in range(width):
                    position = xs[x] if channel < axis_channels else ys[y]
                    phase = fmul(position, frequency)
                    python_double_sine = f32(math.sin(phase))
                    python_double_cosine = f32(math.cos(phase))
                    sine = fsin(phase)
                    cosine = fcos(phase)
                    selected = sine if local < half else cosine
                    embedding = fmul(selected, 0.1)
                    input_value = input_tensor.get(batch_index, channel, y, x)
                    output_value = fadd(input_value, embedding)
                    output.set(output_value, batch_index, channel, y, x)
                    if lane_trace is not None and batch_index == 0 and channel == 0 and y == 0 and x == 0:
                        discriminator_ratio = fdiv(1.0, 3.0)
                        discriminator_positive_power = fpow(100.0, discriminator_ratio)
                        lane_trace.update(
                            {
                                "x_coordinate_bits": [struct.unpack("<I", struct.pack("<f", value))[0] for value in xs],
                                "y_coordinate_bits": [struct.unpack("<I", struct.pack("<f", value))[0] for value in ys],
                                "batch": batch_index,
                                "channel": channel,
                                "y": y,
                                "x": x,
                                "exponent_ratio_bits": struct.unpack("<I", struct.pack("<f", exponent_ratio))[0],
                                "positive_power_bits": struct.unpack("<I", struct.pack("<f", positive_power))[0],
                                "reciprocal_bits": struct.unpack("<I", struct.pack("<f", reciprocal))[0],
                                "negative_power_bits": struct.unpack("<I", struct.pack("<f", negative_power))[0],
                                "frequency_bits": struct.unpack("<I", struct.pack("<f", frequency))[0],
                                "discriminator_ratio_bits": struct.unpack("<I", struct.pack("<f", discriminator_ratio))[0],
                                "discriminator_positive_power_bits": struct.unpack("<I", struct.pack("<f", discriminator_positive_power))[0],
                                "discriminator_reciprocal_bits": struct.unpack("<I", struct.pack("<f", fdiv(1.0, discriminator_positive_power)))[0],
                                "discriminator_negative_power_bits": struct.unpack("<I", struct.pack("<f", fpow(100.0, f32(-discriminator_ratio))))[0],
                                "position_bits": struct.unpack("<I", struct.pack("<f", position))[0],
                                "phase_bits": struct.unpack("<I", struct.pack("<f", phase))[0],
                                "python_double_sine_bits": struct.unpack("<I", struct.pack("<f", python_double_sine))[0],
                                "python_double_cosine_bits": struct.unpack("<I", struct.pack("<f", python_double_cosine))[0],
                                "libc_sine_bits": struct.unpack("<I", struct.pack("<f", sine))[0],
                                "libc_cosine_bits": struct.unpack("<I", struct.pack("<f", cosine))[0],
                                "selected_trig": "sin" if local < half else "cos",
                                "selected_trig_bits": struct.unpack("<I", struct.pack("<f", selected))[0],
                                "ratio_bits": struct.unpack("<I", struct.pack("<f", f32(0.1)))[0],
                                "embedding_bits": struct.unpack("<I", struct.pack("<f", embedding))[0],
                                "input_bits": struct.unpack("<I", struct.pack("<f", input_value))[0],
                                "output_bits": struct.unpack("<I", struct.pack("<f", output_value))[0],
                            }
                        )
    return output


def depth_head(state, features, height=4, width=4, profile="dpt", use_ray=False, head_trace=None):
    resized = []
    patch_height = height // 2
    patch_width = width // 2
    for index, feature_pair in enumerate(features):
        feature = feature_pair[0]
        if profile == "dualdpt":
            feature = norm_state(state, feature, "native.head.norm", f32(1.0e-5))
        tensor = tokens_to_nchw(feature, patch_height, patch_width)
        tensor = convolution(state, tensor, f"native.head.projects.{index}")
        if profile == "dualdpt":
            tensor = position_embedding(tensor, width, height)
        if index == 0:
            tensor = convolution(state, tensor, "native.head.resize_layers.0", 4, 0, True)
        elif index == 1:
            tensor = convolution(state, tensor, "native.head.resize_layers.1", 2, 0, True)
        elif index == 3:
            tensor = convolution(state, tensor, "native.head.resize_layers.3", 2, 1)
        tensor = convolution(state, tensor, f"native.head.scratch.layer{index + 1}_rn", 1, 1)
        resized.append(tensor)
    if head_trace is not None:
        for index, tensor in enumerate(resized):
            head_trace[f"resized_{index}"] = tensor.clone()
    main = fusion(state, resized[3], None, "native.head.scratch.refinenet4", resized[2].shape[2:])
    auxiliary = [fusion(state, resized[3], None, "native.head.scratch.refinenet4_aux", resized[2].shape[2:])] if use_ray else None
    if head_trace is not None and auxiliary is not None:
        head_trace["refinenet4_aux"] = auxiliary[-1].clone()
    main = fusion(state, main, resized[2], "native.head.scratch.refinenet3", resized[1].shape[2:])
    if auxiliary is not None:
        auxiliary.append(fusion(state, auxiliary[-1], resized[2], "native.head.scratch.refinenet3_aux", resized[1].shape[2:]))
        if head_trace is not None:
            head_trace["refinenet3_aux"] = auxiliary[-1].clone()
    main = fusion(state, main, resized[1], "native.head.scratch.refinenet2", resized[0].shape[2:])
    if auxiliary is not None:
        auxiliary.append(fusion(state, auxiliary[-1], resized[1], "native.head.scratch.refinenet2_aux", resized[0].shape[2:]))
        if head_trace is not None:
            head_trace["refinenet2_aux"] = auxiliary[-1].clone()
    main = fusion(state, main, resized[0], "native.head.scratch.refinenet1")
    if auxiliary is not None:
        auxiliary.append(fusion(state, auxiliary[-1], resized[0], "native.head.scratch.refinenet1_aux"))
        if head_trace is not None:
            head_trace["refinenet1_aux"] = auxiliary[-1].clone()
    fused = convolution(state, main, "native.head.scratch.output_conv1", 1, 1)
    fused = bilinear(fused, height, width, True)
    if profile == "dualdpt":
        fused = position_embedding(fused, width, height)
    main = relu(convolution(state, fused, "native.head.scratch.output_conv2.0", 1, 1))
    logits = convolution(state, main, "native.head.scratch.output_conv2.2")
    depth_values = []
    confidence_values = []
    for batch_index in range(logits.shape[0]):
        for y in range(height):
            for x in range(width):
                depth_values.append(fexp(logits.get(batch_index, 0, y, x)))
                if logits.shape[1] > 1:
                    confidence_values.append(fadd(fexp(logits.get(batch_index, logits.shape[1] - 1, y, x)), 1.0))
    depth = Tensor((logits.shape[0], 1, height, width), depth_values)
    confidence = Tensor((logits.shape[0], 1, height, width), confidence_values) if confidence_values else None
    sky = None
    if profile == "dpt":
        sky = relu(convolution(state, relu(convolution(state, fused, "native.head.scratch.sky_output_conv2.0", 1, 1)), "native.head.scratch.sky_output_conv2.2"))
    ray = None
    ray_confidence = None
    if auxiliary is not None:
        processed = []
        for level, tensor in enumerate(auxiliary):
            for index in range(5):
                tensor = convolution(state, tensor, f"native.head.scratch.output_conv1_aux.{level}.{index}", 1, 1)
                if head_trace is not None and level == 3:
                    head_trace[f"output_conv1_aux_3_conv_{index}"] = tensor.clone()
            processed.append(tensor)
            if head_trace is not None:
                head_trace[f"output_conv1_aux_{level}"] = tensor.clone()
        position_lane_trace = {} if head_trace is not None else None
        last = position_embedding(
            processed[-1],
            width,
            height,
            position_lane_trace,
        )
        if head_trace is not None:
            head_trace["position_lane"] = position_lane_trace
            head_trace["positioned"] = last.clone()
        last = convolution(state, last, "native.head.scratch.output_conv2_aux.3.0", 1, 1)
        if head_trace is not None:
            head_trace["convolution_3_0"] = last.clone()
        channels_last = Tensor((last.shape[0], last.shape[2], last.shape[3], last.shape[1]), [last.get(batch_index, channel, y, x) for batch_index in range(last.shape[0]) for y in range(last.shape[2]) for x in range(last.shape[3]) for channel in range(last.shape[1])])
        channels_last = norm_state(state, channels_last, "native.head.scratch.output_conv2_aux.3.2", f32(1.0e-5))
        last = Tensor(last.shape, [channels_last.get(batch_index, y, x, channel) for batch_index in range(last.shape[0]) for channel in range(last.shape[1]) for y in range(last.shape[2]) for x in range(last.shape[3])])
        if head_trace is not None:
            head_trace["normalized"] = last.clone()
        last = relu(last)
        if head_trace is not None:
            head_trace["relu"] = last.clone()
        last = convolution(state, last, "native.head.scratch.output_conv2_aux.3.5")
        if head_trace is not None:
            head_trace["logits"] = last.clone()
        ray_values = []
        ray_confidence_values = []
        for flat in range(last.shape[0]):
            for y in range(last.shape[2]):
                for x in range(last.shape[3]):
                    ray_values.extend(last.get(flat, channel, y, x) for channel in range(6))
                    ray_confidence_values.append(fadd(fexp(last.get(flat, 6, y, x)), 1.0))
        ray = Tensor((last.shape[0], last.shape[2], last.shape[3], 6), ray_values)
        ray_confidence = Tensor((last.shape[0], last.shape[2], last.shape[3]), ray_confidence_values)
        if head_trace is not None:
            head_trace["ray"] = ray.clone()
            head_trace["confidence"] = ray_confidence.clone()
    return depth, confidence, sky, ray, ray_confidence


def execute_dpt(input_values, mutation=None, source_dtype="f32"):
    state = make_state("dpt", mutation, source_dtype)
    image = preprocess(input_values)
    features, _, _ = backbone(state, image)
    return depth_head(state, features)


def execute_dpt_resized(input_values, input_height, input_width, process_resolution, resize_method):
    reference = max(input_height, input_width) if resize_method == "upper_bound" else min(input_height, input_width)
    scale = process_resolution / reference
    target_height = max(1, round_to_patch(round(input_height * scale), 2))
    target_width = max(1, round_to_patch(round(input_width * scale), 2))
    state = make_state("dpt")
    image = preprocess(input_values, 1, input_height, input_width, target_height, target_width)
    features, _, _ = backbone(state, image)
    depth, confidence, sky, _, _ = depth_head(state, features, target_height, target_width)
    depth = bilinear(depth, input_height, input_width, False)
    sky = bilinear(sky, input_height, input_width, False)
    return depth, confidence, sky, None, None


def camera_block(state, values, block_index):
    prefix = f"native.cam_enc.trunk.{block_index}"
    normalized = norm_state(state, values, prefix + ".norm1", f32(1.0e-5))
    projected = linear_state(state, normalized, prefix + ".attn.qkv")
    batch, views, hidden3 = projected.shape
    hidden = hidden3 // 3
    query = Tensor((batch, views, hidden), [projected.get(batch_index, view, lane) for batch_index in range(batch) for view in range(views) for lane in range(hidden)])
    key = Tensor((batch, views, hidden), [projected.get(batch_index, view, hidden + lane) for batch_index in range(batch) for view in range(views) for lane in range(hidden)])
    value = Tensor((batch, views, hidden), [projected.get(batch_index, view, hidden * 2 + lane) for batch_index in range(batch) for view in range(views) for lane in range(hidden)])
    attended = attention(query, key, value, batch, views, 1, hidden)
    attended = linear_state(state, attended, prefix + ".attn.proj")
    residual = add(values, multiply_channels(attended, state[prefix + ".ls1.gamma"]))
    normalized = norm_state(state, residual, prefix + ".norm2", f32(1.0e-5))
    projected = linear_state(state, gelu(linear_state(state, normalized, prefix + ".mlp.fc1")), prefix + ".mlp.fc2")
    return add(residual, multiply_channels(projected, state[prefix + ".ls2.gamma"]))


def affine_inverse(extrinsics):
    batch, views, _, _ = extrinsics.shape
    output = Tensor(extrinsics.shape, [0.0] * len(extrinsics.values))
    for batch_index in range(batch):
        for view in range(views):
            for row in range(3):
                for column in range(3):
                    output.set(extrinsics.get(batch_index, view, column, row), batch_index, view, row, column)
                translation = f32(0.0)
                for column in range(3):
                    translation = fadd(translation, fmul(output.get(batch_index, view, row, column), extrinsics.get(batch_index, view, column, 3)))
                output.set(fmul(-1.0, translation), batch_index, view, row, 3)
    return output


def rotation_to_quaternion(rotation):
    candidates = [
        f32(math.sqrt(max(0.0, fadd(fadd(fadd(1.0, rotation[0]), rotation[4]), rotation[8])))),
        f32(math.sqrt(max(0.0, fsub(fsub(fadd(1.0, rotation[0]), rotation[4]), rotation[8])))),
        f32(math.sqrt(max(0.0, fsub(fadd(fsub(1.0, rotation[0]), rotation[4]), rotation[8])))),
        f32(math.sqrt(max(0.0, fadd(fsub(fsub(1.0, rotation[0]), rotation[4]), rotation[8])))),
    ]
    selected = 0
    for index in range(1, 4):
        if candidates[index] > candidates[selected]:
            selected = index
    denominator = fmul(2.0, max(candidates[selected], 0.1))
    choices = [
        [fmul(candidates[0], candidates[0]), fsub(rotation[7], rotation[5]), fsub(rotation[2], rotation[6]), fsub(rotation[3], rotation[1])],
        [fsub(rotation[7], rotation[5]), fmul(candidates[1], candidates[1]), fadd(rotation[3], rotation[1]), fadd(rotation[2], rotation[6])],
        [fsub(rotation[2], rotation[6]), fadd(rotation[3], rotation[1]), fmul(candidates[2], candidates[2]), fadd(rotation[5], rotation[7])],
        [fsub(rotation[3], rotation[1]), fadd(rotation[6], rotation[2]), fadd(rotation[7], rotation[5]), fmul(candidates[3], candidates[3])],
    ]
    values = choices[selected]
    quaternion = [fdiv(values[1], denominator), fdiv(values[2], denominator), fdiv(values[3], denominator), fdiv(values[0], denominator)]
    if quaternion[3] < 0.0:
        quaternion = [fmul(-1.0, value) for value in quaternion]
    return quaternion


def encode_camera(state, extrinsics, intrinsics, height, width):
    inverse = affine_inverse(extrinsics)
    batch, views = extrinsics.shape[:2]
    pose_values = []
    for batch_index in range(batch):
        for view in range(views):
            rotation = [inverse.get(batch_index, view, row, column) for row in range(3) for column in range(3)]
            pose_values.extend(inverse.get(batch_index, view, row, 3) for row in range(3))
            pose_values.extend(rotation_to_quaternion(rotation))
            pose_values.append(fmul(2.0, fatan(fdiv(height / 2.0, intrinsics.get(batch_index, view, 1, 1)))))
            pose_values.append(fmul(2.0, fatan(fdiv(width / 2.0, intrinsics.get(batch_index, view, 0, 0)))))
    values = Tensor((batch, views, 9), pose_values)
    values = linear_state(state, values, "native.cam_enc.pose_branch.fc1")
    values = gelu(values)
    values = linear_state(state, values, "native.cam_enc.pose_branch.fc2")
    values = norm_state(state, values, "native.cam_enc.token_norm", f32(1.0e-5))
    for block_index in range(4):
        values = camera_block(state, values, block_index)
    return norm_state(state, values, "native.cam_enc.trunk_norm", f32(1.0e-5))


def pose_geometry(translation, quaternion, field_of_view, height, width):
    count = translation.shape[0]
    extrinsics = Tensor((1, count, 3, 4), [0.0] * (count * 12))
    intrinsics = Tensor((1, count, 3, 3), [0.0] * (count * 9))
    for index in range(count):
        q = [quaternion.get(index, lane) for lane in range(4)]
        norm = max(f32(math.sqrt(fsum(fmul(value, value) for value in q))), f32(1.0e-6))
        x, y, z, real = [fdiv(value, norm) for value in q]
        two = fdiv(2.0, fsum([fmul(x, x), fmul(y, y), fmul(z, z), fmul(real, real)]))
        rotation = [
            fsub(1.0, fmul(two, fadd(fmul(y, y), fmul(z, z)))), fmul(two, fsub(fmul(x, y), fmul(z, real))), fmul(two, fadd(fmul(x, z), fmul(y, real))),
            fmul(two, fadd(fmul(x, y), fmul(z, real))), fsub(1.0, fmul(two, fadd(fmul(x, x), fmul(z, z)))), fmul(two, fsub(fmul(y, z), fmul(x, real))),
            fmul(two, fsub(fmul(x, z), fmul(y, real))), fmul(two, fadd(fmul(y, z), fmul(x, real))), fsub(1.0, fmul(two, fadd(fmul(x, x), fmul(y, y)))),
        ]
        for row in range(3):
            for column in range(3):
                extrinsics.set(rotation[column * 3 + row], 0, index, row, column)
            translated = fsum(rotation[column * 3 + row] * translation.get(index, column) for column in range(3))
            extrinsics.set(fmul(-1.0, translated), 0, index, row, 3)
        fov_height = field_of_view.get(index, 0)
        fov_width = field_of_view.get(index, 1)
        intrinsics.set(fdiv(width / 2.0, max(f32(math.tan(fdiv(fov_width, 2.0))), f32(1.0e-6))), 0, index, 0, 0)
        intrinsics.set(fdiv(height / 2.0, max(f32(math.tan(fdiv(fov_height, 2.0))), f32(1.0e-6))), 0, index, 1, 1)
        intrinsics.set(f32(width / 2.0), 0, index, 0, 2)
        intrinsics.set(f32(height / 2.0), 0, index, 1, 2)
        intrinsics.set(1.0, 0, index, 2, 2)
    return extrinsics, intrinsics


def decode_camera(state, camera_values, height, width):
    hidden = linear_state(state, camera_values, "native.cam_dec.backbone.0")
    hidden = relu(hidden)
    hidden = relu(linear_state(state, hidden, "native.cam_dec.backbone.2"))
    translation = linear_state(state, hidden, "native.cam_dec.fc_t")
    quaternion = linear_state(state, hidden, "native.cam_dec.fc_qvec")
    field_of_view = relu(linear_state(state, hidden, "native.cam_dec.fc_fov.0"))
    count = product(camera_values.shape[:-1])
    return pose_geometry(Tensor((count, 3), translation.values), Tensor((count, 4), quaternion.values), Tensor((count, 2), field_of_view.values), height, width)


def smallest_symmetric_eigenvector(matrix):
    size = len(matrix)
    values = [[float(matrix[row][column]) for column in range(size)] for row in range(size)]
    vectors = [[1.0 if row == column else 0.0 for column in range(size)] for row in range(size)]
    for _ in range(256):
        row, column = max(((row, column) for row in range(size) for column in range(row + 1, size)), key=lambda pair: abs(values[pair[0]][pair[1]]))
        if abs(values[row][column]) < 1.0e-12:
            break
        angle = 0.5 * math.atan2(2.0 * values[row][column], values[column][column] - values[row][row])
        cosine = math.cos(angle)
        sine = math.sin(angle)
        for index in range(size):
            left = values[index][row]
            right = values[index][column]
            values[index][row] = cosine * left - sine * right
            values[index][column] = sine * left + cosine * right
        for index in range(size):
            left = values[row][index]
            right = values[column][index]
            values[row][index] = cosine * left - sine * right
            values[column][index] = sine * left + cosine * right
        for index in range(size):
            left = vectors[index][row]
            right = vectors[index][column]
            vectors[index][row] = cosine * left - sine * right
            vectors[index][column] = sine * left + cosine * right
    selected = min(range(size), key=lambda index: values[index][index])
    return [f32(vectors[row][selected]) for row in range(size)]


def weighted_homography(source, destination, weights, indices):
    first_rows = []
    second_rows = []
    for index in indices:
        weight = f32(math.sqrt(weights[index]))
        x, y = source[index]
        u, v = destination[index]
        first_rows.append([fmul(-x, weight), fmul(-y, weight), fmul(-1.0, weight), 0.0, 0.0, 0.0, fmul(fmul(x, u), weight), fmul(fmul(y, u), weight), fmul(u, weight)])
        second_rows.append([0.0, 0.0, 0.0, fmul(-x, weight), fmul(-y, weight), fmul(-1.0, weight), fmul(fmul(x, v), weight), fmul(fmul(y, v), weight), fmul(v, weight)])
    gram = [[f32(0.0) for _ in range(9)] for _ in range(9)]
    for row in first_rows + second_rows:
        for left in range(9):
            for right in range(9):
                gram[left][right] = fadd(gram[left][right], fmul(row[left], row[right]))
    homography = smallest_symmetric_eigenvector(gram)
    divisor = homography[8]
    return [fdiv(value, divisor) for value in homography]


def determinant3(matrix):
    return fadd(fsub(fmul(matrix[0], fsub(fmul(matrix[4], matrix[8]), fmul(matrix[5], matrix[7]))), fmul(matrix[1], fsub(fmul(matrix[3], matrix[8]), fmul(matrix[5], matrix[6])))), fmul(matrix[2], fsub(fmul(matrix[3], matrix[7]), fmul(matrix[4], matrix[6]))))


def matrix3_multiply(left, right):
    return [fsum(fmul(left[row * 3 + index], right[index * 3 + column]) for index in range(3)) for row in range(3) for column in range(3)]


def qr3(matrix):
    columns = [[matrix[row * 3 + column] for row in range(3)] for column in range(3)]
    orthogonal = []
    upper = [[f32(0.0) for _ in range(3)] for _ in range(3)]
    for column in range(3):
        vector = list(columns[column])
        for previous in range(column):
            coefficient = fsum(fmul(orthogonal[previous][row], columns[column][row]) for row in range(3))
            upper[previous][column] = coefficient
            vector = [fsub(vector[row], fmul(coefficient, orthogonal[previous][row])) for row in range(3)]
        norm = f32(math.sqrt(fsum(fmul(value, value) for value in vector)))
        upper[column][column] = norm
        orthogonal.append([fdiv(value, norm) for value in vector])
    q = [orthogonal[column][row] for row in range(3) for column in range(3)]
    return q, [value for row in upper for value in row]


def ql3(matrix):
    permutation = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    q_tilde, r_tilde = qr3(matrix3_multiply(matrix, permutation))
    q = matrix3_multiply(q_tilde, permutation)
    lower = matrix3_multiply(matrix3_multiply(permutation, r_tilde), permutation)
    for diagonal in range(3):
        value = lower[diagonal * 3 + diagonal]
        sign = 1.0 if value > 0.0 else (-1.0 if value < 0.0 else 0.0)
        for row in range(3):
            q[row * 3 + diagonal] = fmul(q[row * 3 + diagonal], sign)
        for column in range(3):
            lower[diagonal * 3 + column] = fmul(lower[diagonal * 3 + column], sign)
    return q, lower


def ray_pose_with_trace(ray, confidence, views, seed=17, address=RAY_POSE_RNG_ADDRESS):
    _, height, width, _ = ray.shape
    points = height * width
    dx = 1.0 / width
    dy = 1.0 / height
    horizontal = [f32(-(1.0 - dx) + 2.0 * (1.0 - dx) * index / (width - 1)) for index in range(width)]
    vertical = [f32(-(1.0 - dy) + 2.0 * (1.0 - dy) * index / (height - 1)) for index in range(height)]
    c2w = Tensor((1, views, 3, 4), [0.0] * (views * 12))
    intrinsics = Tensor((1, views, 3, 3), [0.0] * (views * 9))
    candidate_count = max(8, int(points * 0.3))
    generator = ransac_generator(seed, address)
    samples = [generator.randperm(candidate_count)[:8] for _ in range(100)]
    view_traces = []
    for view in range(views):
        source = []
        destination = []
        weights = []
        raw_confidence = []
        for y in range(height):
            for x in range(width):
                source.append((horizontal[x], vertical[y]))
                target_z = ray.get(view, y, x, 2)
                if abs(target_z) > 1.0e-4:
                    destination.append((fdiv(ray.get(view, y, x, 0), target_z), fdiv(ray.get(view, y, x, 1), target_z)))
                    weights.append(confidence.get(view, y, x))
                else:
                    destination.append((0.0, 0.0))
                    weights.append(0.0)
                raw_confidence.append(confidence.get(view, y, x))
        sorted_indices = sorted(range(points), key=lambda index: (-weights[index], index))
        candidate_indices = sorted_indices[:candidate_count]
        best_score = f32(float("-inf"))
        best_iteration = 0
        best_inliers = []
        for iteration, sample in enumerate(samples):
            selected = [candidate_indices[index] for index in sample]
            candidate_homography = weighted_homography(
                source, destination, weights, selected
            )
            score = f32(0.0)
            inliers = []
            for index in range(points):
                x, y = source[index]
                denominator = fsum([fmul(x, candidate_homography[6]), fmul(y, candidate_homography[7]), candidate_homography[8]])
                projected_x = fdiv(fsum([fmul(x, candidate_homography[0]), fmul(y, candidate_homography[1]), candidate_homography[2]]), denominator)
                projected_y = fdiv(fsum([fmul(x, candidate_homography[3]), fmul(y, candidate_homography[4]), candidate_homography[5]]), denominator)
                error = f32(math.sqrt(fadd(fmul(fsub(projected_x, destination[index][0]), fsub(projected_x, destination[index][0])), fmul(fsub(projected_y, destination[index][1]), fsub(projected_y, destination[index][1])))))
                if error < 0.2:
                    score = fadd(score, weights[index])
                    inliers.append(index)
            if score > best_score:
                best_score = score
                best_iteration = iteration
                best_inliers = inliers
        fallback = len(best_inliers) < 4
        if not fallback:
            best_inliers.sort(key=lambda index: (-weights[index], index))
            if len(best_inliers) > 8000:
                keep = max(int(len(best_inliers) * 0.95), 8000)
                refit_sample = generator.randperm(keep)[:8000]
                best_inliers = [best_inliers[index] for index in refit_sample]
            homography = weighted_homography(source, destination, weights, best_inliers)
        else:
            homography = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        normalized_homography = list(homography)
        if determinant3(homography) < 0.0:
            homography = [fmul(-1.0, value) for value in homography]
        rotation, lower = ql3(homography)
        scale = lower[8]
        total = fsum(raw_confidence)
        for row in range(3):
            for column in range(3):
                c2w.set(rotation[row * 3 + column], 0, view, row, column)
            translation = fdiv(fsum(fmul(ray.get(view, point // width, point % width, 3 + row), raw_confidence[point]) for point in range(points)), total)
            c2w.set(translation, 0, view, row, 3)
        view_traces.append(
            {
                "candidate_indices": candidate_indices,
                "best_iteration": best_iteration,
                "best_inliers": best_inliers,
                "best_score_bits": struct.unpack("<I", struct.pack("<f", best_score))[0],
                "fallback": fallback,
                "normalized_homography_bits": [struct.unpack("<I", struct.pack("<f", value))[0] for value in normalized_homography],
                "homography_post_sign_bits": [struct.unpack("<I", struct.pack("<f", value))[0] for value in homography],
                "rotation_bits": [struct.unpack("<I", struct.pack("<f", value))[0] for value in rotation],
                "lower_bits": [struct.unpack("<I", struct.pack("<f", value))[0] for value in lower],
                "c2w_pre_inverse_bits": [
                    struct.unpack("<I", struct.pack("<f", c2w.get(0, view, row, column)))[0]
                    for row in range(3)
                    for column in range(4)
                ],
            }
        )
        focal_x = fdiv(1.0, fdiv(lower[0], scale))
        focal_y = fdiv(1.0, fdiv(lower[4], scale))
        principal_x = fadd(fdiv(lower[6], scale), 1.0)
        principal_y = fadd(fdiv(lower[7], scale), 1.0)
        intrinsics.set(fmul(fdiv(focal_x, 2.0), width), 0, view, 0, 0)
        intrinsics.set(fmul(fdiv(focal_y, 2.0), height), 0, view, 1, 1)
        intrinsics.set(fmul(fmul(principal_x, width), 0.5), 0, view, 0, 2)
        intrinsics.set(fmul(fmul(principal_y, height), 0.5), 0, view, 1, 2)
        intrinsics.set(1.0, 0, view, 2, 2)
    return (affine_inverse(c2w), intrinsics), {
        "profile_version": 2,
        "algorithm": "mt19937",
        "seed": seed,
        "address": dict(address),
        "address_digest": rng_address_digest(seed, address),
        "samples": samples,
        "views": view_traces,
    }


def ray_pose(ray, confidence, views):
    return ray_pose_with_trace(ray, confidence, views)[0]


def execute_dualdpt(input_values, views=3, mutation=None, camera_inputs=None, use_ray=False, reference_strategy="saddle_sim_range", source_dtype="f32", head_trace=None):
    state = make_state("dualdpt", mutation, source_dtype)
    image = preprocess(input_values, batch=views)
    camera_token = None
    if camera_inputs is not None:
        camera_token = encode_camera(state, camera_inputs[0], camera_inputs[1], 4, 4)
    features, _, _ = backbone(state, image, "dualdpt", views, reference_strategy, camera_token)
    depth, confidence, _, ray, ray_confidence = depth_head(state, features, profile="dualdpt", use_ray=use_ray, head_trace=head_trace)
    camera_geometry = ray_pose(ray, ray_confidence, views) if use_ray else decode_camera(state, features[-1][1], 4, 4)
    return depth, confidence, ray, ray_confidence, camera_geometry
