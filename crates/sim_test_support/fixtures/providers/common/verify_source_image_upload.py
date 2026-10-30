import argparse
import ast
import importlib.metadata
import platform
import sys
import hashlib
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, __version__ as pillow_version


def source_results():
    root = Path(__file__).resolve().parents[5]
    source = root / "projects/sim/ComfyUI"
    namespace = {"np": np, "torch": torch, "Image": Image, "math": math}
    sources = {}
    for relative, names in [
        ("sim/utils.py", {"lanczos", "common_upscale"}),
        ("sim_api_nodes/util/conversions.py", {
            "_compute_downscale_dims", "downscale_image_tensor", "tensor_to_pil",
        }),
    ]:
        path = source / relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        functions = [node for node in ast.parse(path.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in functions} == names
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    cases = []
    values = [-0.5, 0.0, 0.125, 0.5, 0.75, 1.0, 1.25, 1e8, -1e8]
    shapes = [
        [1, 2, 3, 3], [2, 3, 4, 3], [1, 1, 1, 3], [1, 1, 3, 3],
        [1, 3, 1, 3], [1, 2, 3, 4], [1, 2, 3, 1], [1, 1, 1, 1],
        [1, 1, 4, 1], [1, 8, 11, 3], [1, 8, 11, 4], [1, 8, 11, 1],
        [1, 1, 50, 3],
    ]
    for shape in shapes:
        count = math.prod(shape)
        pixels = [values[index % len(values)] for index in range(count)]
        tensor = torch.tensor(pixels, dtype=torch.float32).reshape(shape)
        for maximum_pixels in [None, 4, 64, 1568 * 1568]:
            for batch_index in range(shape[0]):
                case = dict(shape=shape, pixels=pixels, maximum_pixels=maximum_pixels,
                            batch_index=batch_index)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        image = namespace["tensor_to_pil"](
                            tensor[batch_index], total_pixels=maximum_pixels,
                        )
                    case["output"] = dict(width=image.width, height=image.height,
                                          mode=image.mode, pixels=list(image.tobytes()))
                except (TypeError, ValueError, IndexError) as error:
                    case["error"] = type(error).__name__
                cases.append(case)
    return dict(schema_version=1, sources=sources, cases=cases)


def linux_profile_results():
    wheel = importlib.metadata.distribution("numpy")
    wheel_record = next(path for path in wheel.files if str(path).endswith(".dist-info/WHEEL"))
    binary = Path(np.core._multiarray_umath.__file__)
    wheel_text = wheel.locate_file(wheel_record).read_text()
    assert platform.system() == "Linux" and platform.machine() == "x86_64"
    assert np.__version__ == "1.26.4" and torch.__version__ == "2.8.0+cpu"
    assert pillow_version == "12.3.0"
    assert "cp312-cp312-manylinux_2_17_x86_64" in wheel_text
    result = source_results()
    result["profile"] = dict(
        system=platform.system(), machine=platform.machine(),
        python_abi=sys.implementation.cache_tag, numpy=np.__version__,
        torch=torch.__version__, pillow=pillow_version,
        numpy_wheel_sha256=hashlib.sha256(wheel_text.encode()).hexdigest(),
        numpy_binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        numpy_cpu_features=np.core._multiarray_umath.__cpu_features__,
        qualification="Observed wheel/CPU profile; invalid float-to-int casts are not portable NumPy guarantees.",
    )
    boundary = np.float32(2**31 / 255)
    values = np.array([
        -0.5, 0, 0.125, 0.5, 0.75, 1, 1.25,
        np.nextafter(boundary, np.float32(0)), boundary,
        np.nextafter(boundary, np.float32(np.inf)),
        np.nextafter(-boundary, np.float32(-np.inf)), -boundary,
        np.nextafter(-boundary, np.float32(0)),
        1e8, -1e8, np.finfo(np.float32).max, -np.finfo(np.float32).max,
        np.nan, np.inf, -np.inf,
    ], dtype=np.float32)
    result["cast_cases"] = []
    for name, pixels in [
        ("short-positive-overflow", values[8:9]),
        ("signed-boundaries", values[7:13]),
        ("contiguous", values),
        ("vector-repetition", np.tile(values, 17)),
        ("strided", np.tile(values, 3)[::2]),
        ("reversed", values[::-1]),
    ]:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            output = (pixels * np.float32(255)).astype(np.uint8)
        result["cast_cases"].append(dict(
            name=name, input_bits=pixels.view(np.uint32).tolist(), output=output.tolist(),
            warning_categories=[type(warning.message).__name__ for warning in captured],
        ))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-linux-profile", action="store_true")
    parser.add_argument("--check-historical", action="store_true")
    args = parser.parse_args()
    assert not (args.write_linux_profile and args.check_historical)
    if args.check_historical:
        expected = json.loads(Path(__file__).with_name("source_image_upload.json").read_text())
        assert source_results() == expected
    else:
        actual = linux_profile_results()
        path = Path(__file__).with_name("source_image_upload_linux_x86_64.json")
        if args.write_linux_profile:
            assert not path.exists(), "Never overwrite a retained profile during a reader batch"
            path.write_text(json.dumps(actual, indent=2, allow_nan=False) + "\n")
            print(f"WROTE: {len(actual['cases'])} original Source image cases and {len(actual['cast_cases'])} cast cases")
        expected = json.loads(path.read_text())
        assert actual == expected
    print(f"PASS: {len(expected['cases'])} pinned API image preparation cases")
