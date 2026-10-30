import ast
import hashlib
import json
import math
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[5]
source = root / "projects/sim/ComfyUI/sim/utils.py"
namespace = {"torch": torch, "math": math}
selected = [node for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name in {"reshape_mask", "repeat_to_batch_size"}]
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
cases = []
for output_shape, mask_shapes in [
    ([3, 4, 4, 6], [[2, 3, 5], [1, 1, 2, 3], [2, 1, 4, 6], [4, 2, 5, 7]]),
    ([2, 4, 3, 4, 6], [[3, 2, 3], [1, 3, 2, 3], [2, 1, 2, 3, 5], [1, 2, 4, 4, 6]]),
    ([3, 4, 7], [[1, 1, 4], [2, 2, 5], [4, 4, 7]]),
]:
    for mask_shape in mask_shapes:
        for constant in [False, True]:
            count = math.prod(mask_shape)
            values = [1.0 if constant else (index % 9) / 8 for index in range(count)]
            mask = torch.tensor(values, dtype=torch.float32).reshape(mask_shape)
            output = namespace["reshape_mask"](mask, output_shape)
            cases.append(dict(output_shape=output_shape, mask_shape=mask_shape,
                              input_values=values, prepared_shape=list(output.shape),
                              prepared_values=output.flatten().tolist()))
result = {"provenance": "Actual pinned reshape_mask and repeat_to_batch_size functions with real PyTorch linear, bilinear, and trilinear interpolation; no mathematical substitutes.",
          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "cases": cases}
path = Path(__file__).with_name("source_masks.json")
assert json.loads(json.dumps(result)) == json.loads(path.read_text()), "pinned mask source changed"
print(len(cases), "source mask preparation cases verified")
