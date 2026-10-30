#!/usr/bin/env python3
import hashlib
import json
import math
import platform
import struct
import sys
from pathlib import Path

PINNED_SOURCES = {
    "projects/sim/ComfyUI/sim_extras/nodes_audio_encoder.py": "fbc6f4d8ca0e099dc2f35f9420dce9aced3763325b9804b031a1641db2bb4a8a",
    "projects/sim/ComfyUI/sim/audio_encoders/audio_encoders.py": "c8d3260799ea0222b6bf9e1bde8d16f105a46aaf1944213bf66fdaa05433dec8",
    "projects/sim/ComfyUI/sim/audio_encoders/wav2vec2.py": "32494297021e54e42845276255a026ce8ab62be5d54f6d40cecfb042c8b238e7",
    "projects/sim/ComfyUI/sim/audio_encoders/whisper.py": "f0e214e79fdfa9926fbc863038fe3e0455caa61f486a1b37a3039ca7253dea22",
}


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def state_values(key, count, mutation=0):
    seed = 0
    for byte in key.encode():
        seed = ((seed * 16777619) + byte) & 0xFFFFFFFF
    normalization = key.endswith(".weight") and "norm" in key
    position_scale = key.endswith("parametrizations.weight.original0")
    values = []
    for element in range(count):
        if normalization or position_scale:
            value = 1.0 + ((seed + element) & 0xFFFFFFFF) % 5 * 0.01
        elif key.endswith(".bias"):
            value = (((seed + element) & 0xFFFFFFFF) % 7 - 3) * 0.002
        else:
            value = (((seed + element * 17) & 0xFFFFFFFF) % 13 - 6) * 0.01
        values.append(f32(value))
    if mutation and key == "encoder.layer_norm.bias":
        values[0] = f32(values[0] + mutation * 0.125)
    return values


def product(shape):
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def weight(key, shape, mutation=0):
    return state_values(key, product(shape), mutation)


def input_audio(samples):
    values = []
    for batch in range(2):
        for channel in range(2):
            for sample in range(samples):
                phase = f32(f32(sample) * f32(0.013))
                phase = f32(phase + f32(f32(channel) * f32(0.31)))
                amplitude = f32(1.0 + f32(f32(batch) * f32(1.7)))
                value = f32(f32(math.sin(phase)) * amplitude)
                values.append(f32(value + f32(f32(batch) * f32(0.2))))
    return values


def channel_mean(values, samples):
    output = []
    for batch in range(2):
        for sample in range(samples):
            total = 0.0
            for channel in range(2):
                total += values[(batch * 2 + channel) * samples + sample]
            output.append(f32(total / 2.0))
    return output


def global_sample_normalize(values):
    mean = f32(sum(float(value) for value in values) / len(values))
    running_mean = 0.0
    squared_deviation = 0.0
    count = 0
    for value in values:
        count += 1
        delta = float(value) - running_mean
        running_mean += delta / count
        squared_deviation += delta * (float(value) - running_mean)
    variance = f32(squared_deviation / (count - 1))
    denominator = f32(math.sqrt(f32(variance + f32(1e-7))))
    return [f32(f32(value - mean) / denominator) for value in values]


def conv1d(values, batch, input_channels, length, key, output_channels, kernel, stride, padding, groups, bias, mutation=0):
    output_length = (length + 2 * padding - kernel) // stride + 1
    input_per_group = input_channels // groups
    kernel_values = weight(key + ".weight", [output_channels, input_per_group, kernel], mutation)
    bias_values = weight(key + ".bias", [output_channels], mutation) if bias else [0.0] * output_channels
    output = [0.0] * (batch * output_channels * output_length)
    output_per_group = output_channels // groups
    for b in range(batch):
        for oc in range(output_channels):
            group = oc // output_per_group
            for destination in range(output_length):
                result = bias_values[oc]
                for ic_local in range(input_per_group):
                    ic = group * input_per_group + ic_local
                    for tap in range(kernel):
                        source = destination * stride + tap - padding
                        if 0 <= source < length:
                            input_index = (b * input_channels + ic) * length + source
                            kernel_index = (oc * input_per_group + ic_local) * kernel + tap
                            result = f32(values[input_index] * kernel_values[kernel_index] + result)
                output[(b * output_channels + oc) * output_length + destination] = result
    return output, output_length


def transpose_ncl_nlc(values, batch, channels, length):
    return [values[(b * channels + c) * length + token] for b in range(batch) for token in range(length) for c in range(channels)]


def transpose_nlc_ncl(values, batch, tokens, width):
    return [values[(b * tokens + token) * width + component] for b in range(batch) for component in range(width) for token in range(tokens)]


def affine_norm_rows(values, rows, width, key, epsilon, mutation=0):
    scales = weight(key + ".weight", [width], mutation)
    biases = weight(key + ".bias", [width], mutation)
    output = []
    for row in range(rows):
        source = values[row * width:(row + 1) * width]
        mean = f32(sum(float(value) for value in source) / width)
        variance = sum((float(value) - float(mean)) ** 2 for value in source) / width
        inverse = f32(1.0 / math.sqrt(variance + float(f32(epsilon))))
        for component in range(width):
            normalized = f32(f32(source[component] - mean) * inverse)
            output.append(f32(f32(normalized * scales[component]) + biases[component]))
    return output


def group_norm_channels(values, batch, channels, length, key, mutation=0):
    scales = weight(key + ".weight", [channels], mutation)
    biases = weight(key + ".bias", [channels], mutation)
    output = [0.0] * len(values)
    for b in range(batch):
        for channel in range(channels):
            source = values[(b * channels + channel) * length:(b * channels + channel + 1) * length]
            mean = f32(sum(float(value) for value in source) / length)
            variance = sum((float(value) - float(mean)) ** 2 for value in source) / length
            inverse = f32(1.0 / math.sqrt(variance + float(f32(1e-5))))
            for index, value in enumerate(source):
                normalized = f32(f32(value - mean) * inverse)
                output[(b * channels + channel) * length + index] = f32(f32(normalized * scales[channel]) + biases[channel])
    return output


def linear(values, rows, input_width, key, output_width, bias=True, mutation=0):
    matrix = weight(key + ".weight", [output_width, input_width], mutation)
    offsets = weight(key + ".bias", [output_width], mutation) if bias else [0.0] * output_width
    output = []
    for row in range(rows):
        for destination in range(output_width):
            value = offsets[destination]
            for source in range(input_width):
                value = f32(values[row * input_width + source] * matrix[destination * input_width + source] + value)
            output.append(value)
    return output


def erf_approximation(value):
    sign = -1.0 if value < 0.0 else (1.0 if value > 0.0 else 0.0)
    absolute = abs(value)
    t = f32(1.0 / f32(1.0 + f32(f32(0.3275911) * absolute)))
    polynomial = f32(f32(f32(f32(f32(f32(1.0614054) * t - f32(1.4531521)) * t + f32(1.4214138)) * t - f32(0.28449672)) * t + f32(0.2548296)) * t)
    return f32(sign * f32(1.0 - f32(polynomial * f32(math.exp(f32(-absolute * absolute))))))


def gelu(values):
    output = []
    for value in values:
        argument = f32(value * f32(1.0 / math.sqrt(2.0)))
        output.append(f32(f32(f32(0.5) * value) * f32(1.0 + erf_approximation(argument))))
    return output


def add(left, right):
    return [f32(a + b) for a, b in zip(left, right)]


def softmax(values):
    maximum = max(values)
    exponentials = [f32(math.exp(f32(value - maximum))) for value in values]
    denominator = sum(float(value) for value in exponentials)
    return [f32(value / denominator) for value in exponentials]


def attention(values, batch, tokens, width, key, key_bias, mutation=0):
    rows = batch * tokens
    query = linear(values, rows, width, key + ".q_proj", width, True, mutation)
    keys = linear(values, rows, width, key + ".k_proj", width, key_bias, mutation)
    projected = linear(values, rows, width, key + ".v_proj", width, True, mutation)
    attended = [0.0] * len(values)
    scale = f32(1.0 / math.sqrt(width))
    for b in range(batch):
        for query_token in range(tokens):
            scores = []
            for key_token in range(tokens):
                score = 0.0
                for component in range(width):
                    score = f32(query[(b * tokens + query_token) * width + component] * keys[(b * tokens + key_token) * width + component] + score)
                scores.append(f32(score * scale))
            probabilities = softmax(scores)
            for component in range(width):
                value = 0.0
                for key_token in range(tokens):
                    value = f32(probabilities[key_token] * projected[(b * tokens + key_token) * width + component] + value)
                attended[(b * tokens + query_token) * width + component] = value
    return linear(attended, rows, width, key + ".out_proj", width, True, mutation)


def feed_forward(values, rows, width, hidden, first, second, mutation=0):
    return linear(gelu(linear(values, rows, width, first, hidden, True, mutation)), rows, hidden, second, width, True, mutation)


def weight_norm_position(values, batch, channels, length, kernel, mutation=0):
    key = "encoder.pos_conv_embed.conv"
    magnitude = weight(key + ".parametrizations.weight.original0", [1, 1, kernel], mutation)
    direction = weight(key + ".parametrizations.weight.original1", [channels, channels, kernel], mutation)
    materialized = [0.0] * len(direction)
    for tap in range(kernel):
        norm = math.sqrt(sum(float(direction[(output * channels + source) * kernel + tap]) ** 2 for output in range(channels) for source in range(channels)))
        for output in range(channels):
            for source in range(channels):
                index = (output * channels + source) * kernel + tap
                materialized[index] = f32(float(direction[index]) * float(magnitude[tap]) / norm)
    bias = weight(key + ".bias", [channels], mutation)
    padding = kernel // 2
    output_length = length + 2 * padding - kernel + 1
    output = [0.0] * (batch * channels * output_length)
    for b in range(batch):
        for destination_channel in range(channels):
            for destination in range(output_length):
                value = bias[destination_channel]
                for source_channel in range(channels):
                    for tap in range(kernel):
                        source = destination + tap - padding
                        if 0 <= source < length:
                            value = f32(values[(b * channels + source_channel) * length + source] * materialized[(destination_channel * channels + source_channel) * kernel + tap] + value)
                output[(b * channels + destination_channel) * output_length + destination] = value
    return output[:], output_length


def wav2vec2(large):
    batch = 2
    samples = 1600
    convolution_width = 2
    width = 2
    layers = 24 if large else 12
    values = channel_mean(input_audio(samples), samples)
    if large:
        values = global_sample_normalize(values)
    values = [value for b in range(batch) for value in values[b * samples:(b + 1) * samples]]
    length = samples
    input_channels = 1
    for index, (kernel, stride) in enumerate([(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]):
        key = f"feature_extractor.conv_layers.{index}"
        values, length = conv1d(values, batch, input_channels, length, key + ".conv", convolution_width, kernel, stride, 0, 1, large)
        if large:
            values = transpose_ncl_nlc(values, batch, convolution_width, length)
            values = affine_norm_rows(values, batch * length, convolution_width, key + ".layer_norm", 1e-5)
            values = transpose_nlc_ncl(values, batch, length, convolution_width)
        elif index == 0:
            values = group_norm_channels(values, batch, convolution_width, length, key + ".layer_norm")
        values = gelu(values)
        input_channels = convolution_width
    values = transpose_ncl_nlc(values, batch, convolution_width, length)
    values = affine_norm_rows(values, batch * length, convolution_width, "feature_projection.layer_norm", 1e-5)
    values = linear(values, batch * length, convolution_width, "feature_projection.projection", width, True)
    position_input = transpose_nlc_ncl(values, batch, length, width)
    position, position_length = weight_norm_position(position_input, batch, width, length, 4)
    assert position_length == length + 1
    position = [position[(b * width + c) * position_length + token] for b in range(batch) for c in range(width) for token in range(length)]
    position = gelu(position)
    position = transpose_ncl_nlc(position, batch, width, length)
    values = add(values, position)
    if not large:
        values = affine_norm_rows(values, batch * length, width, "encoder.layer_norm", 1e-5)
    for layer in range(layers):
        prefix = f"encoder.layers.{layer}"
        residual = values[:]
        if large:
            values = affine_norm_rows(values, batch * length, width, prefix + ".layer_norm", 1e-5)
        values = attention(values, batch, length, width, prefix + ".attention", True)
        values = add(residual, values)
        if not large:
            values = affine_norm_rows(values, batch * length, width, prefix + ".layer_norm", 1e-5)
        feed_input = affine_norm_rows(values, batch * length, width, prefix + ".final_layer_norm", 1e-5) if large else values[:]
        feed = feed_forward(feed_input, batch * length, width, 4, prefix + ".feed_forward.intermediate_dense", prefix + ".feed_forward.output_dense")
        values = add(values, feed)
        if not large:
            values = affine_norm_rows(values, batch * length, width, prefix + ".final_layer_norm", 1e-5)
    if large:
        values = affine_norm_rows(values, batch * length, width, "encoder.layer_norm", 1e-5)
    return [batch, length, width], values


def periodic_hann(length):
    return [f32(0.5 * (1.0 - math.cos(2.0 * math.pi * index / length))) for index in range(length)]


def slaney_mel_filters(n_fft, sample_rate, mel_bins):
    frequency_count = n_fft // 2 + 1
    maximum = sample_rate / 2.0
    minimum_mel = 0.0
    maximum_mel = 15.0 + math.log(maximum / 1000.0) / (math.log(6.4) / 27.0)
    points = []
    for index in range(mel_bins + 2):
        mel = minimum_mel + (maximum_mel - minimum_mel) * index / (mel_bins + 1)
        frequency = mel * (200.0 / 3.0) if mel < 15.0 else 1000.0 * math.exp((mel - 15.0) * (math.log(6.4) / 27.0))
        points.append(frequency)
    filters = []
    for mel in range(mel_bins):
        left, center, right = points[mel:mel + 3]
        normalization = 2.0 / (right - left)
        for frequency in range(frequency_count):
            hertz = frequency * maximum / (frequency_count - 1)
            value = max(0.0, min((hertz - left) / (center - left), (right - hertz) / (right - center)))
            filters.append(f32(value * normalization))
    return filters


def whisper_mel(values, batch, samples, target=32, n_fft=16, hop=4, mel_bins=4):
    mono = channel_mean(values, samples)
    padded = []
    for b in range(batch):
        row = mono[b * samples:(b + 1) * samples]
        padded.extend((row + [0.0] * target)[:target])
    window = periodic_hann(n_fft)
    frames = 1 + ((target + n_fft) - n_fft) // hop
    frequencies = n_fft // 2 + 1
    filters = slaney_mel_filters(n_fft, 16000, mel_bins)
    spectrum = [[[0.0 for _ in range(frames)] for _ in range(frequencies)] for _ in range(batch)]
    center = n_fft // 2
    for b in range(batch):
        row = padded[b * target:(b + 1) * target]
        for frame in range(frames):
            frame_values = []
            for sample in range(n_fft):
                index = frame * hop + sample
                if index < center:
                    source = center - index
                else:
                    shifted = index - center
                    source = shifted if shifted < target else 2 * target - 2 - shifted
                frame_values.append(f32(row[source] * window[sample]))
            for frequency in range(frequencies):
                real = 0.0
                imaginary = 0.0
                for sample, value in enumerate(frame_values):
                    angle = -2.0 * math.pi * frequency * sample / n_fft
                    real += float(value) * math.cos(angle)
                    imaginary += float(value) * math.sin(angle)
                spectrum[b][frequency][frame] = f32(f32(math.hypot(f32(real), f32(imaginary))) ** f32(2.0))
    mel = []
    for b in range(batch):
        for band in range(mel_bins):
            for frame in range(frames - 1):
                value = 0.0
                for frequency in range(frequencies):
                    value = f32(filters[band * frequencies + frequency] * spectrum[b][frequency][frame] + value)
                mel.append(value)
    logs = [f32(math.log10(max(value, f32(1e-10)))) for value in mel]
    maximum = max(logs)
    floor = f32(maximum - f32(8.0))
    return [f32(f32(max(value, floor) + f32(4.0)) / f32(4.0)) for value in logs], frames - 1


def whisper():
    batch = 2
    samples = 29
    width = 2
    mel_bins = 4
    values, mel_frames = whisper_mel(input_audio(samples), batch, samples)
    values, length = conv1d(values, batch, mel_bins, mel_frames, "encoder.conv1", width, 3, 1, 1, 1, True)
    values = gelu(values)
    values, length = conv1d(values, batch, width, length, "encoder.conv2", width, 3, 2, 1, 1, True)
    values = gelu(values)
    values = transpose_ncl_nlc(values, batch, width, length)
    positions = weight("encoder.embed_positions.weight", [4, width])
    values = [f32(value + positions[index % (length * width)]) for index, value in enumerate(values)]
    for layer in range(32):
        prefix = f"encoder.layers.{layer}"
        normalized = affine_norm_rows(values, batch * length, width, prefix + ".self_attn_layer_norm", 1e-5)
        attended = attention(normalized, batch, length, width, prefix + ".self_attn", False)
        values = add(values, attended)
        normalized = affine_norm_rows(values, batch * length, width, prefix + ".final_layer_norm", 1e-5)
        values = add(values, feed_forward(normalized, batch * length, width, 4, prefix + ".fc1", prefix + ".fc2"))
    values = affine_norm_rows(values, batch * length, width, "encoder.layer_norm", 1e-5)
    return [batch, length, width], values


def sha_f32(values):
    return hashlib.sha256(b"".join(struct.pack("<f", value) for value in values)).hexdigest()


def main():
    root = Path(__file__).resolve().parents[5]
    generator_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for relative, expected in PINNED_SOURCES.items():
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"source drift for {relative}: {actual} != {expected}")
    results = {}
    for name, function in [
        ("wav2vec2-base", lambda: wav2vec2(False)),
        ("wav2vec2-large", lambda: wav2vec2(True)),
        ("whisper-large-v3", whisper),
    ]:
        shape, values = function()
        results[name] = {"shape": shape, "values": values, "raw_f32_sha256": sha_f32(values)}
    document = {
        "format": "zed.sim.audio-encoder-reduced-oracle.v1",
        "generator_sha256": generator_sha256,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pinned_sources": PINNED_SOURCES,
        "tolerance": {"absolute": 0.0002, "relative": 0.0002},
        "results": results,
    }
    output = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if sys.argv[1:] == ["--check"]:
        tracked = Path(__file__).with_name("oracle.json").read_text()
        if tracked != output:
            raise SystemExit("tracked audio-encoder oracle differs from regenerated output")
        print("audio-encoder oracle is reproducible")
        return
    if sys.argv[1:]:
        raise SystemExit("usage: generate_oracle.py [--check]")
    print(output, end="")
    print("document_sha256=" + hashlib.sha256(output.encode()).hexdigest(), file=sys.stderr)


if __name__ == "__main__":
    main()
