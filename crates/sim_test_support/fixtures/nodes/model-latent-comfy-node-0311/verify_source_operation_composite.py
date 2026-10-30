"""Offline source-pin and arithmetic oracle for Task 629 operation/composite cases."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
SOURCE = ROOT / "projects/sim/ComfyUI"
FIXTURE = json.loads(Path(__file__).with_name("operation_composite.json").read_text())
for relative, expected in FIXTURE["source_sha256"].items():
    actual = hashlib.sha256((SOURCE / relative).read_bytes()).hexdigest()
    assert actual == expected, (relative, actual)

latent_source = (SOURCE / "sim_extras/nodes_latent.py").read_text()
mask_source = (SOURCE / "sim_extras/nodes_mask.py").read_text()
plain_source = (SOURCE / "nodes.py").read_text()
kernel_source = (SOURCE / "sim_extras/nodes_post_processing.py").read_text()
for marker in (
    "class LatentApplyOperation(io.ComfyNode)",
    "class LatentApplyOperationCFG(io.ComfyNode)",
    "class LatentOperationTonemapReinhard(io.ComfyNode)",
    "class LatentOperationSharpen(io.ComfyNode)",
    "torch.std(latent_vector_magnitude, dim=dims, keepdim=True)",
    "conds_out[0] = operation(latent=(conds_out[0] - conds_out[1])) + conds_out[1]",
):
    assert marker in latent_source, marker
assert "class LatentComposite:" in plain_source
assert "for t in range(feather):" in plain_source
assert "class LatentCompositeMasked(IO.ComfyNode):" in mask_source
assert "source = sim.utils.repeat_to_batch_size(source, destination.shape[0])" in mask_source
assert "def gaussian_kernel(kernel_size: int, sigma: float" in kernel_source

values = FIXTURE["tonemap"]["input"]
mean = sum(values) / len(values)
standard_deviation = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
top = (standard_deviation * 5 + mean) * FIXTURE["tonemap"]["multiplier"]
for value, expected in zip(values, FIXTURE["tonemap"]["expected"]):
    magnitude = abs(value) + 1e-10
    scaled = magnitude / top
    actual = value / magnitude * scaled / (scaled + 1) * top
    assert abs(actual - expected) < 1e-8, (actual, expected)

feather = FIXTURE["composite"]["feather"] // 8
for expected in FIXTURE["composite"]["expected_overlay"]:
    actual = ((1 / feather) * 1) ** 2
    assert abs(actual - expected) < 1e-8
oversized = FIXTURE["composite"]["oversized_feather"] // 8
rows = [1.0, 1.0]
for step in range(oversized):
    rows[step:step + 1] = [value * ((step + 1) / oversized) for value in rows[step:step + 1]]
    rows[1 - step:2 - step] = [value * ((step + 1) / oversized) for value in rows[1 - step:2 - step]]
for actual, expected in zip([rows[0] ** 2, rows[0] * rows[1], rows[1] * rows[0], rows[1] ** 2], FIXTURE["composite"]["oversized_expected_overlay"]):
    assert abs(actual - expected) < 1e-8, (actual, expected)
assert FIXTURE["masked"]["mask"] == FIXTURE["masked"]["expected_overlay"]
print("Task 629 operation/composite source pin and 4 arithmetic cases passed")
