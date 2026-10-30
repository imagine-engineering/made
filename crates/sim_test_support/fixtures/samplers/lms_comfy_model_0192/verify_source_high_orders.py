import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import integrate
import torch


def source_results():
    root = Path(__file__).resolve().parents[5]
    source = root / "projects/comfy/ComfyUI/comfy/k_diffusion"
    namespace = {"integrate": integrate, "torch": torch, "trange": lambda count, **kwargs: range(count)}
    sources = {}
    for filename, names in [("utils.py", {"append_dims"}),
                            ("sampling.py", {"linear_multistep_coeff", "to_d", "sample_lms"})]:
        path = source / filename
        sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        functions = [node for node in ast.parse(path.read_text()).body
                     if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in functions} == names
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
        if filename == "utils.py":
            namespace["utils"] = SimpleNamespace(append_dims=namespace["append_dims"])

    schedules = {
        "unit": np.arange(101, -1, -1, dtype=np.float32),
        "translated": np.arange(1000, 898, -1, dtype=np.float32),
        "nonuniform": np.array([1000 - index - index * index / 1024
                                for index in range(102)], dtype=np.float32),
    }
    cases = []
    for name, sigmas in schedules.items():
        for order in [1, 4, 5, 8, 16, 32, 64, 100]:
            coefficients = [namespace["linear_multistep_coeff"](order, sigmas, 99, index)
                            for index in range(order)]
            assert all(np.isfinite(np.float32(value)) for value in coefficients)
            cases.append(dict(name=name, order=order, step=99, sigmas=sigmas.tolist(),
                              coefficients=coefficients))

    trajectories = []
    for order in [1, 4, 5, 100]:
        sigmas = torch.tensor(list(range(1000, 899, -1)) + [0], dtype=torch.float32)
        callbacks = []
        step = 0

        def denoiser(latent, sigma):
            nonlocal step
            result = latent - sigma if step == 99 else latent
            step += 1
            return result

        terminal = namespace["sample_lms"](
            denoiser, torch.zeros([1], dtype=torch.float32), sigmas, order=order,
            callback=lambda event: callbacks.append(event["x"].tolist()), disable=True,
        )
        assert step == 101 and len(callbacks) == 101
        trajectories.append(dict(order=order, sigmas=sigmas.tolist(), callbacks=callbacks,
                                 terminal=terminal.tolist()))
    return dict(schema_version=1, sources=sources, relative_tolerance=2e-6,
                cases=cases, trajectories=trajectories)


if __name__ == "__main__":
    expected = json.loads(Path(__file__).with_name("source_high_orders.json").read_text())
    actual = source_results()
    assert actual.keys() == expected.keys()
    for key in actual.keys() - {"cases"}:
        assert actual[key] == expected[key], key
    assert len(actual["cases"]) == len(expected["cases"])
    for observed, recorded in zip(actual["cases"], expected["cases"], strict=True):
        for key in observed.keys() - {"coefficients"}:
            assert observed[key] == recorded[key], (recorded["name"], recorded["order"], key)
        assert len(observed["coefficients"]) == len(recorded["coefficients"])
        for index, (value, reference) in enumerate(zip(
            observed["coefficients"], recorded["coefficients"], strict=True
        )):
            assert math.isclose(value, reference, rel_tol=expected["relative_tolerance"]), (
                recorded["name"], recorded["order"], index, value, reference
            )
    print("PASS: 24 source coefficient cases through order 100 and four 101-step trajectories")
