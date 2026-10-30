#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import math
import platform
import struct
from pathlib import Path


PINNED_SOURCES = {
    "projects/sim/ComfyUI/sim/model_sampling.py": "8559ac2f700b788babf30cc16f410c93890e1f71485ac8f6a299ea36e2cb8717",
    "projects/sim/ComfyUI/sim/samplers.py": "d882256ae9baa1d23f1367ab2ec3b021fdc15fe39ce4cb49ea2c1ee10026a649",
    "projects/sim/ComfyUI/sim/supported_models.py": "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69",
}
BASE_GENERATOR = "crates/sim_test_support/fixtures/models/native-family-denoiser-invocation-foundation/generate_oracle.py"
BASE_GENERATOR_SHA256 = "a3f204112a0b5847f0265c98421ed72c3c3f8b6f2487081e52bfe5dcc1ba104f"
SIGMA = 0.5
CFG = 2.0
STEPS = 4
EXPECTED = {
    "aura": {
        "positive_raw_sha256": "861d33e4e562094bff8ea959aec7ef1f9fbe2c94709e6aa0106c887b096030cd",
        "positive_raw_tolerance_sha256": "713e81ad34e5bdf063cee82d0020b43dc21c286daf5f6146ca002ec563463e46",
        "negative_raw_sha256": "2b5aab48dd1ba80a6fd06873e9146e747ce7de5de59bd237642c8367d5f9cfe6",
        "negative_raw_tolerance_sha256": "713e81ad34e5bdf063cee82d0020b43dc21c286daf5f6146ca002ec563463e46",
        "positive_interpreted_sha256": "da71e383b337baec904f87797c6237ac4cfcf87aaf9a52f1678b1b630942ffcb",
        "positive_interpreted_tolerance_sha256": "69adc941bd0b0076c88bc1394ab1567a1adb45b0481e6a1552bc2ee7caae55ae",
        "negative_interpreted_sha256": "1cefdc2e35dc8665cb32988a2a04692efe5b88ac9d00c35d2f38fa2705774051",
        "negative_interpreted_tolerance_sha256": "69adc941bd0b0076c88bc1394ab1567a1adb45b0481e6a1552bc2ee7caae55ae",
        "cfg_sha256": "02f365d267a1134e7f91a9798899b434d3b287a32d728700e1e1393d6b929834",
        "cfg_tolerance_sha256": "69adc941bd0b0076c88bc1394ab1567a1adb45b0481e6a1552bc2ee7caae55ae",
        "normal_sigmas_sha256": "908c6f4facbad5eef0787c63e163c7c40b1dfe36df10009922d907124ab524d3",
    },
    "qwen": {
        "positive_raw_sha256": "a54623926304017249ea3940c18c193ecb7835216c68497cf897484105cf3a99",
        "positive_raw_tolerance_sha256": "0e0bb02fb085aa45287625d17c18b3a4337ff5056c5f3072779abd1c5ebbc6de",
        "negative_raw_sha256": "d1f284a2469edb714831ce70d4b833b51bc929ea9ad8611002532fd094f7cf89",
        "negative_raw_tolerance_sha256": "33b5428fe91fdb5bcb1915a3893a85e64187906e9bc701666adf5952a9eec605",
        "positive_interpreted_sha256": "1d787532f7d8cd027e05856eb3861cab8f6bae6721a9dbc510097d04dfb94b51",
        "positive_interpreted_tolerance_sha256": "945d2013401682723180c962e8e0c9594d8f7a622dac9a711a83e40740b6d1e9",
        "negative_interpreted_sha256": "1d07522cffa86421b0756b28731e83e2283bbe6461282b920d3fb72344548d24",
        "negative_interpreted_tolerance_sha256": "5f2ca5d6dbdb2e85d3dbcbc40ad5d6a4349e86067a82a9b3d9cff62f0915e854",
        "cfg_sha256": "1e762d8e8f5bb92831b1eef70e2a13d965c960f7746f3a62cc354a214f184559",
        "cfg_tolerance_sha256": "9ad6d7a02b17fbe123a332f8ef7f8bf0b9cdf311a8f92789c0dce3aff0c6de3d",
        "normal_sigmas_sha256": "9d8928caed588103f8e6437af49f34eb0a998e6d63436168b35cb03a4f93ad5d",
    },
}


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def add(left, right):
    return f32(f32(left) + f32(right))


def mul(left, right):
    return f32(f32(left) * f32(right))


def fused_multiply_add(left, right, addend):
    return f32(math.fma(f32(left), f32(right), f32(addend)))


def digest(values):
    return hashlib.sha256(b"".join(struct.pack("<f", value) for value in values)).hexdigest()


def encoded(values):
    return b"".join(struct.pack("<f", value) for value in values).hex()


def tolerance_digest(values):
    quantized = [math.floor(value * 1_000_000.0 + 0.5) for value in values]
    return hashlib.sha256(b"".join(struct.pack("<q", value) for value in quantized)).hexdigest()


def bits(values):
    return [struct.unpack("<I", struct.pack("<f", value))[0] for value in values]


def load_base_generator(repository):
    path = repository / BASE_GENERATOR
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != BASE_GENERATOR_SHA256:
        raise SystemExit(f"base oracle generator drift: {actual} != {BASE_GENERATOR_SHA256}")
    specification = importlib.util.spec_from_file_location("native_family_denoiser_oracle", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def raw_branch(base, family, branch):
    original_patterned = base.patterned
    original_aura_time = base.aura_time
    original_qwen_time = base.qwen_time
    original_attention = base.attention

    def branch_patterned(key, count):
        if key == f"{family}.conditioning":
            return original_patterned("positive" if branch == "positive" else "negative.bias", count)
        return original_patterned(key, count)

    def no_qwen_encoder_mask(query, key, value, tokens, width, mask=None):
        if tokens == 6 and width == 128:
            mask = None
        return original_attention(query, key, value, tokens, width, mask)

    base.patterned = branch_patterned
    base.aura_time = lambda _time: original_aura_time(f32(SIGMA))
    base.qwen_time = lambda _time: original_qwen_time(f32(SIGMA))
    base.attention = no_qwen_encoder_mask
    try:
        return base.aura_oracle() if family == "aura" else base.qwen_oracle()
    finally:
        base.patterned = original_patterned
        base.aura_time = original_aura_time
        base.qwen_time = original_qwen_time
        base.attention = original_attention


def latent(family):
    if family == "aura":
        return [f32((index - 18.0) * 0.025) for index in range(36)]
    return [f32((index - 72.0) * 0.005) for index in range(144)]


def interpret(raw, model_input):
    return [fused_multiply_add(output, -SIGMA, value) for output, value in zip(raw, model_input)]


def cfg(positive, negative):
    return [add(unconditional, mul(add(conditional, -unconditional), CFG)) for conditional, unconditional in zip(positive, negative)]


def aura_shift(time):
    shift = f32(1.73)
    time = f32(time)
    return f32(mul(shift, time) / add(1.0, mul(add(shift, -1.0), time)))


def qwen_shift(time):
    time = f32(time)
    exponential = f32(math.exp(f32(1.15)))
    return f32(exponential / add(exponential, add(f32(1.0 / time), -1.0)))


def normal_sigmas(shift, minimum_time):
    start = f32(1.0)
    end = shift(f32(minimum_time))
    result = []
    for index in range(STEPS):
        fraction = f32(index / (STEPS - 1))
        model_time = fused_multiply_add(add(end, -start), fraction, start)
        result.append(shift(model_time))
    result.append(f32(0.0))
    return result


def family_result(base, family, shift, minimum_time):
    positive_raw = raw_branch(base, family, "positive")
    negative_raw = raw_branch(base, family, "negative")
    model_input = latent(family)
    positive = interpret(positive_raw, model_input)
    negative = interpret(negative_raw, model_input)
    guided = cfg(positive, negative)
    sigmas = normal_sigmas(shift, minimum_time)
    return {
        "positive_raw_sha256": digest(positive_raw),
        "positive_raw_f32_le_hex": encoded(positive_raw),
        "positive_raw_tolerance_sha256": tolerance_digest(positive_raw),
        "negative_raw_sha256": digest(negative_raw),
        "negative_raw_f32_le_hex": encoded(negative_raw),
        "negative_raw_tolerance_sha256": tolerance_digest(negative_raw),
        "positive_interpreted_sha256": digest(positive),
        "positive_interpreted_tolerance_sha256": tolerance_digest(positive),
        "negative_interpreted_sha256": digest(negative),
        "negative_interpreted_tolerance_sha256": tolerance_digest(negative),
        "cfg_sha256": digest(guided),
        "cfg_tolerance_sha256": tolerance_digest(guided),
        "normal_sigmas_bits": bits(sigmas),
        "normal_sigmas_sha256": digest(sigmas),
    }


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[5]
    for relative_path, expected in PINNED_SOURCES.items():
        actual = hashlib.sha256((repository / relative_path).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"pinned source drift: {relative_path}: {actual} != {expected}")
    base = load_base_generator(repository)
    result = {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sigma": SIGMA,
        "cfg": CFG,
        "steps": STEPS,
        "aura": family_result(base, "aura", aura_shift, 0.001),
        "qwen": family_result(base, "qwen", qwen_shift, 0.0001),
    }
    for family, expected in EXPECTED.items():
        actual = result[family]
        for field, expected_value in expected.items():
            if actual[field] != expected_value:
                raise SystemExit(
                    f"resource oracle drift: {family}.{field}: {actual[field]} != {expected_value}"
                )
    print(json.dumps(result, separators=(",", ":"), sort_keys=True))
