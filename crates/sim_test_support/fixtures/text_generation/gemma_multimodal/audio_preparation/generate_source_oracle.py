import argparse
import ast
import ctypes
import ctypes.util
import hashlib
import json
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np
import scipy
import torch


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[5]
SOURCE = "projects/comfy/ComfyUI/comfy/text_encoders/gemma4.py"
SOURCE_SHA256 = "c6ffbb2fbecd8f97e781a654a06ccf3910dc670867d38c0ce30542312f00cde6"
OUTPUT = DIRECTORY / "source-oracle.json"


def generate():
    source = (ROOT / SOURCE).read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Pinned Gemma4 audio source changed")
    if (np.__version__, scipy.__version__, torch.__version__) != ("1.26.4", "1.16.2", "2.8.0+cpu"):
        raise RuntimeError("Use the recorded NumPy 1.26.4, SciPy 1.16.2, torch 2.8.0+cpu oracle")
    tokenizer = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.ClassDef) and node.name == "Gemma4_Tokenizer")
    functions = [node for node in tokenizer.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_extract_mel_spectrogram", "_build_mel_filterbank"}]
    if len(functions) != 2:
        raise RuntimeError("Incomplete Gemma4 source projection")
    for node in functions:
        node.decorator_list = []
    namespace = {"torch": torch, "np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), SOURCE, "exec"), namespace)
    instance = SimpleNamespace(_build_mel_filterbank=namespace["_build_mel_filterbank"])
    native_math = ctypes.CDLL(ctypes.util.find_library("m"))
    native_math.sinf.argtypes = [ctypes.c_float]
    native_math.sinf.restype = ctypes.c_float
    float32 = lambda value: ctypes.c_float(value).value
    tau = float32(2 * np.pi)
    values = [native_math.sinf(float32(float32(float32(tau * 440) * index) / 44100))
              for index in range(2205)]
    waveform = torch.tensor([values], dtype=torch.float32)
    features, mask = namespace["_extract_mel_spectrogram"](instance, waveform, 44100)
    raw = np.asarray(values, dtype="<f4").tobytes()
    return {
        "schema_version": 1,
        "oracle_kind": "independently-executed-pinned-Gemma4-NumPy-SciPy-source-methods",
        "production_dependency": False,
        "source": {"path": SOURCE, "sha256": SOURCE_SHA256},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
        "input": {"shape": [1, 1, 2205], "sample_rate": 44100,
                  "construction": "native float32 TAU * 440 * index / 44100 followed by libc sinf",
                  "sha256": hashlib.sha256(raw).hexdigest()},
        "feature_shape": [1, *features.shape],
        "frame_mask": mask.tolist(),
        "values": features.flatten().tolist(),
        "absolute_tolerance": 2e-6,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    encoded = (json.dumps(generate(), indent=2) + "\n").encode()
    if arguments.check:
        if OUTPUT.read_bytes() != encoded:
            raise RuntimeError("Independent Gemma4 audio source oracle is stale")
        print("Independent Gemma4 audio source oracle is current")
    else:
        OUTPUT.write_bytes(encoded)


if __name__ == "__main__":
    main()
