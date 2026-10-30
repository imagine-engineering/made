#!/usr/bin/env python3
import argparse
import hashlib
import json
import math
import platform
import struct
import sys
from pathlib import Path

from source_graph import (
    RAY_POSE_RNG_ADDRESS,
    Tensor,
    execute_dpt,
    execute_dpt_resized,
    execute_dualdpt,
    fatan,
    fexp,
    make_state,
    manifest,
    preprocess,
    ray_pose_with_trace,
    ransac_samples,
)

ROOT = Path(__file__).resolve().parents[5]
LINUX_GNU_X86_64 = (
    sys.platform == "linux"
    and platform.machine() == "x86_64"
    and platform.libc_ver()[0] == "glibc"
)
OUTPUT = Path(__file__).with_name(
    "oracle-linux-gnu-x86_64.json" if LINUX_GNU_X86_64 else "oracle.json"
)
DOMAIN = "zed.sim.depth-anything-3-reduced-oracle.v1"
AUXILIARY_HEAD_PHASE_DOMAIN = "zed.sim.depth-anything-3.auxiliary-head-phase.v1"
SOURCES = {
    "projects/sim/ComfyUI/sim_extras/nodes_depth_anything_3.py": "adfce28637b6904a08596aa23e22502d20089bc28fff6bcdaabe0b3c35fb7f02",
    "projects/sim/ComfyUI/sim/ldm/depth_anything_3/model.py": "6f05ba0c22a34304f6bd6cde7e6dd26ceef474a99ad51d6632940f8d2decf6b0",
    "projects/sim/ComfyUI/sim/ldm/depth_anything_3/preprocess.py": "6bf00e9929451c39763a0661aa1430dbc78917bc028517a4c8dc290897601845",
    "projects/sim/ComfyUI/sim/ldm/depth_anything_3/dpt.py": "756fa18408e161cb2ddf8adde82902d9fe3aa555be8252b60d045cbc76513ee5",
    "projects/sim/ComfyUI/sim/ldm/depth_anything_3/camera.py": "b9c1bc79862c8f2b59a6058da1bf47c1aaef84ca75ec5131805fbdf2f81dca9a",
    "projects/sim/ComfyUI/sim/ldm/depth_anything_3/ray_pose.py": "a5ed28c0acc2daaeea57754dec4020fe91af60fd2af6548dfa68134713b36694",
    "projects/sim/ComfyUI/sim/ldm/depth_anything_3/reference_view_selector.py": "24e9428a820b5287d622bc865d4fd6520486294c4337a28de71fca6ec62e0c29",
    "projects/sim/ComfyUI/sim/ldm/depth_anything_3/transform.py": "30291a7f8d3d83cc6a911daf603342444ed30e7141abb3ba8ccc7f41273ac763",
    "projects/sim/ComfyUI/sim/image_encoders/dino2.py": "1dec8c1d6104c268e593cea20302d925f637266edce2a6e4dfa142af8a00d579",
    "projects/sim/ComfyUI/sim/model_detection.py": "f13b11988fccf9fa4d878ef5f63313c23c5f1400ec8cde04a502584e157c5072",
    "projects/sim/ComfyUI/sim/text_encoders/bert.py": "3f1f32353da95790285a10f452959a871aa949aab15a89b646a95abc6165955c",
    "projects/sim/ComfyUI/sim/ldm/modules/attention.py": "436e1d91f8d5d84c5667e051cdf3ab2f91d8db25b66d88a084c89a202de0579e",
    "projects/sim/ComfyUI/sim/utils.py": "8b8805ca837e20c922a846854156d10e214654f69df96be90969522f9def2bdb",
    "projects/sim/ComfyUI/sim/ops.py": "9d8a4ec8357a9bfcd98dddbf06fcc2a0244643a392aacbe0970d945462c86a42",
    "projects/sim/ComfyUI/sim/model_management.py": "c2ca243c80a5262ecafe19feb15cec22d4003c16e523b5376f543f0f75acabaa",
    "projects/sim/ComfyUI/sim/supported_models.py": "3801a60d15fe0abf8573cfa60f90e796d773450370f80784f2e0603cda3ffd69",
    "projects/sim/ComfyUI/sim/model_base.py": "99dc53baee665eca1a6aea70cfb9ab071d55784dff339b5e919dc14ae4fde8bd",
}


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value):
    return struct.unpack("<I", struct.pack("<f", f32(value)))[0]


def fadd(left, right):
    return f32(f32(left) + f32(right))


def fsub(left, right):
    return f32(f32(left) - f32(right))


def fmul(left, right):
    return f32(f32(left) * f32(right))


def fdiv(left, right):
    return f32(f32(left) / f32(right))


def f16_bits(value):
    return struct.unpack("<H", struct.pack("<e", f32(value)))[0]


def f16_to_f32(value):
    return struct.unpack("<e", struct.pack("<H", value))[0]


def bf16_bits(value):
    raw = bits(value)
    rounding = 0x7FFF + ((raw >> 16) & 1)
    return ((raw + rounding) >> 16) & 0xFFFF


def bf16_to_f32(value):
    return struct.unpack("<f", struct.pack("<I", value << 16))[0]


def round_to_patch(value, patch):
    down = value // patch * patch
    up = down + patch
    return up if abs(up - value) <= abs(value - down) else down


def target_size(height, width, process_resolution, method, patch):
    reference = max(height, width) if method == "upper_bound" else min(height, width)
    scale = process_resolution / float(reference)
    new_height = max(1, round_to_patch(round(height * scale), patch))
    new_width = max(1, round_to_patch(round(width * scale), patch))
    return [new_height, new_width]


def preprocess_same_size(values):
    output = []
    means = [0.485, 0.456, 0.406]
    deviations = [0.229, 0.224, 0.225]
    for channel in range(3):
        for pixel in range(16):
            value = min(1.0, max(0.0, values[pixel * 3 + channel]))
            output.append(fdiv(fsub(value, means[channel]), deviations[channel]))
    return output


def normalize(vector):
    norm = f32(math.sqrt(sum(fmul(value, value) for value in vector)))
    return [fdiv(value, norm) for value in vector], norm


def reference_indices(tokens):
    normalized = []
    norms = []
    variances = []
    for token in tokens:
        unit, norm = normalize(token)
        normalized.append(unit)
        norms.append(norm)
        mean = fdiv(sum(unit), len(unit))
        variances.append(fdiv(sum(fmul(fsub(value, mean), fsub(value, mean)) for value in unit), len(unit) - 1))
    similarity = []
    ranges = []
    for view, left in enumerate(normalized):
        row = []
        for other, right in enumerate(normalized):
            dot = sum(fmul(a, b) for a, b in zip(left, right))
            row.append(fsub(dot, 1.0 if view == other else 0.0))
        similarity.append(fdiv(sum(row), len(row) - 1))
        ranges.append(fsub(max(row), min(row)))

    def metric(values):
        minimum = min(values)
        span = fadd(fsub(max(values), minimum), 1.0e-8)
        return [fdiv(fsub(value, minimum), span) for value in values]

    sim_n, norm_n, var_n = metric(similarity), metric(norms), metric(variances)
    balanced = [
        fadd(fadd(abs(fsub(sim_n[index], 0.5)), abs(fsub(norm_n[index], 0.5))), abs(fsub(var_n[index], 0.5)))
        for index in range(len(tokens))
    ]
    return {
        "first": 0,
        "middle": len(tokens) // 2,
        "saddle_balanced": min(range(len(tokens)), key=lambda index: (balanced[index], index)),
        "saddle_sim_range": max(range(len(tokens)), key=lambda index: (ranges[index], -index)),
        "balanced_scores_bits": [bits(value) for value in balanced],
        "range_scores_bits": [bits(value) for value in ranges],
    }


def matrix_to_quaternion(matrix):
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix
    candidates = [
        f32(math.sqrt(max(0.0, f32(1.0 + m00 + m11 + m22)))),
        f32(math.sqrt(max(0.0, f32(1.0 + m00 - m11 - m22)))),
        f32(math.sqrt(max(0.0, f32(1.0 - m00 + m11 - m22)))),
        f32(math.sqrt(max(0.0, f32(1.0 - m00 - m11 + m22)))),
    ]
    rows = [
        [candidates[0] ** 2, m21 - m12, m02 - m20, m10 - m01],
        [m21 - m12, candidates[1] ** 2, m10 + m01, m02 + m20],
        [m02 - m20, m10 + m01, candidates[2] ** 2, m12 + m21],
        [m10 - m01, m20 + m02, m21 + m12, candidates[3] ** 2],
    ]
    selected = max(range(4), key=lambda index: (candidates[index], -index))
    divisor = fmul(2.0, max(candidates[selected], f32(0.1)))
    row = [fdiv(value, divisor) for value in rows[selected]]
    output = [row[1], row[2], row[3], row[0]]
    return [-value for value in output] if output[3] < 0.0 else output


def rotary_axis(values, position):
    cosine = f32(math.cos(f32(position)))
    sine = f32(math.sin(f32(position)))
    return [
        fsub(fmul(values[0], cosine), fmul(values[1], sine)),
        fadd(fmul(values[1], cosine), fmul(values[0], sine)),
    ]


def linspace(start, end, steps):
    if steps <= 1:
        return [f32(start)] * steps
    output = []
    for index in range(steps):
        if index + 1 == steps:
            value = end
        else:
            weight = index / float(steps - 1)
            value = start * (1.0 - weight) + end * weight
        output.append(f32(value))
    return output


def position_grid(source_width, source_height, width, height):
    aspect = source_width / float(source_height)
    diagonal = math.sqrt(aspect * aspect + 1.0)
    span_x = aspect / diagonal
    span_y = 1.0 / diagonal
    x_values = linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width)
    y_values = linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height)
    return [[bits(x_values[x]), bits(y_values[y])] for y in range(height) for x in range(width)]


def source_hashes_are_current():
    for relative, expected in SOURCES.items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise SystemExit(f"pinned source changed: {relative}: {actual}")


def tensor_document(tensor):
    if tensor is None:
        return None
    raw = b"".join(struct.pack("<f", value) for value in tensor.values)
    return {
        "shape": list(tensor.shape),
        "bits": [bits(value) for value in tensor.values],
        "raw_f32_sha256": hashlib.sha256(raw).hexdigest(),
    }


def tensor_phase_summary(phase, tensor):
    raw = b"".join(struct.pack("<f", value) for value in tensor.values)
    digest = hashlib.sha256()
    encoded_domain = AUXILIARY_HEAD_PHASE_DOMAIN.encode("ascii")
    encoded_phase = phase.encode("ascii")
    digest.update(len(encoded_domain).to_bytes(8, "little"))
    digest.update(encoded_domain)
    digest.update(len(encoded_phase).to_bytes(8, "little"))
    digest.update(encoded_phase)
    digest.update(len(tensor.shape).to_bytes(8, "little"))
    for dimension in tensor.shape:
        digest.update(dimension.to_bytes(8, "little"))
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    return {
        "phase": phase,
        "shape": list(tensor.shape),
        "raw_f32_sha256_domain": AUXILIARY_HEAD_PHASE_DOMAIN,
        "raw_f32_sha256": digest.hexdigest(),
        "first_bits": [bits(value) for value in tensor.values[:8]],
        "last_bits": [bits(value) for value in tensor.values[-8:]],
    }


def index_rows_sha256(domain, rows):
    digest = hashlib.sha256()
    encoded_domain = domain.encode("ascii")
    digest.update(len(encoded_domain).to_bytes(8, "little"))
    digest.update(encoded_domain)
    digest.update(len(rows).to_bytes(8, "little"))
    for row in rows:
        digest.update(len(row).to_bytes(8, "little"))
        for value in row:
            digest.update(value.to_bytes(8, "little"))
    return digest.hexdigest()


def ransac_trace_document(ray, confidence, views):
    geometry, trace = ray_pose_with_trace(ray, confidence, views)
    sample_domain = "zed.sim.depth-anything-3.ransac-samples.v1"
    inlier_domain = "zed.sim.depth-anything-3.ransac-inliers.v1"
    trace["samples_sha256_domain"] = sample_domain
    trace["samples_sha256"] = index_rows_sha256(sample_domain, trace["samples"])
    for view, view_trace in enumerate(trace["views"]):
        view_trace["best_inliers_sha256_domain"] = inlier_domain
        view_trace["best_inliers_sha256"] = index_rows_sha256(
            inlier_domain, [view_trace["best_inliers"]]
        )
        view_trace["best_inlier_count"] = len(view_trace.pop("best_inliers"))
        view_trace["view"] = view
    trace["pre_geometry_ray"] = tensor_document(ray)
    trace["pre_geometry_confidence"] = tensor_document(confidence)
    values_per_view = len(confidence.values) // views
    ray_values_per_view = len(ray.values) // views
    admission_views = []
    for view in range(views):
        confidence_values = confidence.values[
            view * values_per_view : (view + 1) * values_per_view
        ]
        if (
            len(confidence_values) != 256
            or any(not math.isfinite(value) or value <= 0.0 for value in confidence_values)
            or len({bits(value) for value in confidence_values}) != len(confidence_values)
        ):
            raise ValueError("reduced ray confidence must contain 256 finite positive bit-distinct values per view")
        sorted_values = sorted(confidence_values)
        adjacent_ulp_gaps = [
            bits(right) - bits(left)
            for left, right in zip(sorted_values, sorted_values[1:])
        ]
        adjacent_value_gaps = [
            fsub(right, left)
            for left, right in zip(sorted_values, sorted_values[1:])
        ]
        ray_values = ray.values[
            view * ray_values_per_view : (view + 1) * ray_values_per_view
        ]
        z_values = ray_values[2::6]
        if len(z_values) != 256 or any(
            not math.isfinite(value) or abs(value) <= 1.0e-4 for value in z_values
        ):
            raise ValueError("reduced ray z lanes must all be finite and source-valid")
        admission_views.append(
            {
                "view": view,
                "confidence_count": len(confidence_values),
                "confidence_all_finite_positive": True,
                "confidence_all_bit_distinct": True,
                "minimum_adjacent_ulp_gap": min(adjacent_ulp_gaps),
                "minimum_adjacent_value_gap_bits": bits(min(adjacent_value_gaps)),
                "ray_z_all_valid": True,
                "minimum_ray_z_abs_bits": bits(min(abs(value) for value in z_values)),
            }
        )
    trace["admission"] = {
        "candidate_count": len(trace["views"][0]["candidate_indices"]),
        "confidence_ordering": {
            "source": "torch.argsort(descending=True, stable=False-default)",
            "native_owner": "argsort_with_context_exact_native(descending=true, stable=false)",
            "tied_order_pinned": False,
        },
        "views": admission_views,
    }
    trace["geometry"] = geometry_document(geometry)
    changed_seed_samples = ransac_samples(len(trace["views"][0]["candidate_indices"]), 18)
    changed_address = dict(RAY_POSE_RNG_ADDRESS)
    changed_address["phase"] = "reduced-ray-pose-mutated"
    changed_address_samples = ransac_samples(
        len(trace["views"][0]["candidate_indices"]), 17, changed_address
    )
    trace["mutation_discriminators"] = {
        "seed_18_samples_sha256": index_rows_sha256(
            sample_domain, changed_seed_samples
        ),
        "changed_phase": changed_address["phase"],
        "changed_phase_samples_sha256": index_rows_sha256(
            sample_domain, changed_address_samples
        ),
    }
    return trace


def geometry_document(geometry):
    if geometry is None:
        return None
    return {
        "extrinsics": tensor_document(geometry[0]),
        "intrinsics": tensor_document(geometry[1]),
    }


def output_identity(outputs):
    tensors = [output for output in outputs[:3] if output is not None]
    if len(outputs) > 4 and outputs[4] is not None:
        tensors.extend(outputs[4])
    raw = bytearray()
    for tensor in tensors:
        for value in tensor.values:
            raw.extend(struct.pack("<f", value))
    return hashlib.sha256(raw).hexdigest()


def state_bytes(tensor, source_dtype):
    if source_dtype == "f16":
        return b"".join(struct.pack("<e", value) for value in tensor.values)
    if source_dtype == "bf16":
        return b"".join(struct.pack("<H", bits(value) >> 16) for value in tensor.values)
    return b"".join(struct.pack("<f", value) for value in tensor.values)


def state_identity(domain, key, dtype, shape, raw):
    digest = hashlib.sha256()
    digest.update(domain)
    encoded_key = key.encode("utf-8")
    encoded_dtype = dtype.encode("ascii")
    digest.update(len(encoded_key).to_bytes(8, "little"))
    digest.update(encoded_key)
    digest.update(len(encoded_dtype).to_bytes(8, "little"))
    digest.update(encoded_dtype)
    digest.update(len(shape).to_bytes(8, "little"))
    for dimension in shape:
        digest.update(dimension.to_bytes(8, "little"))
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)
    return digest.hexdigest()


def checkpoint_projection(profile, source_dtype):
    state = make_state(profile, source_dtype=source_dtype)
    catalog_dtype = {
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
    }[source_dtype]
    entries = []
    source_aggregate = hashlib.sha256()
    source_aggregate.update(b"zed.sim.depth-anything-3.checkpoint-source-aggregate.v1\0")
    projected_aggregate = hashlib.sha256()
    projected_aggregate.update(
        b"zed.sim.depth-anything-3.checkpoint-projected-aggregate.v1\0"
    )
    specifications = sorted(manifest(profile), key=lambda specification: specification[0])
    source_aggregate.update(len(specifications).to_bytes(8, "little"))
    projected_aggregate.update(len(specifications).to_bytes(8, "little"))
    for key, shape in specifications:
        tensor = state[key]
        source_raw = state_bytes(tensor, source_dtype)
        projected_raw = b"".join(struct.pack("<f", value) for value in tensor.values)
        source_sha256 = state_identity(
            b"zed.sim.depth-anything-3.checkpoint-source-state.v1\0",
            key,
            catalog_dtype,
            shape,
            source_raw,
        )
        projected_sha256 = state_identity(
            b"zed.sim.depth-anything-3.checkpoint-projected-state.v1\0",
            key,
            "float32",
            shape,
            projected_raw,
        )
        source_aggregate.update(source_sha256.encode("ascii"))
        projected_aggregate.update(projected_sha256.encode("ascii"))
        entries.append(
            {
                "key": key,
                "shape": list(shape),
                "source_sha256": source_sha256,
                "projected_f32_sha256": projected_sha256,
            }
        )
    return {
        "ordering": "utf8-key-ascending",
        "key_count": len(entries),
        "source_sha256": source_aggregate.hexdigest(),
        "projected_f32_sha256": projected_aggregate.hexdigest(),
        "states": entries,
    }


def document():
    source_hashes_are_current()
    values = [f32((index + 1) / 64.0) for index in range(48)]
    conversion_values = [0.0, -0.0, 1.0, -2.0, 0.333251953125, 65504.0]
    tokens = [
        [-0.70466894, -1.3966033, 0.6037379, -1.7102549],
        [0.14352801, -0.5372443, -1.7680043, 0.029742932],
        [-1.8500174, -0.26541728, -1.7205783, -1.6371479],
        [-0.30192325, 1.3074085, -1.5047922, -1.1070441],
    ]
    quaternion = matrix_to_quaternion([0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    vertical = rotary_axis([0.75, -0.25], 2)
    horizontal = rotary_axis([0.5, 1.25], 1)
    fov_height = f32(0.75)
    fov_width = f32(1.0)
    height, width = 4, 8
    fx = fdiv(width / 2.0, max(f32(math.tan(fdiv(fov_width, 2.0))), f32(1.0e-6)))
    fy = fdiv(height / 2.0, max(f32(math.tan(fdiv(fov_height, 2.0))), f32(1.0e-6)))
    atanf_input = fdiv(2.0, 2.25)
    dpt_depth, _, dpt_sky, _, _ = execute_dpt(values)
    nonsquare_values = [f32(((index * 11 + 3) % 29) / 28.0) for index in range(2 * 4 * 3)]
    nonsquare_cases = {}
    for method in ["upper_bound", "lower_bound"]:
        target = target_size(2, 4, 6, method, 2)
        outputs = execute_dpt_resized(nonsquare_values, 2, 4, 6, method)
        nonsquare_cases[method] = {
            "target_size": target,
            "preprocessed": tensor_document(preprocess(nonsquare_values, 1, 2, 4, target[0], target[1])),
            "depth": tensor_document(outputs[0]),
            "sky": tensor_document(outputs[2]),
            "output_identity_sha256": output_identity(outputs),
        }
    low_precision_dpt = {
        source_dtype: execute_dpt(values, source_dtype=source_dtype)
        for source_dtype in ["f16", "bf16"]
    }
    multiview_values = []
    for view in range(3):
        multiview_values.extend(f32(min(1.0, max(0.0, value + view * 0.03125))) for value in values)
    camera_extrinsics = Tensor(
        (1, 3, 3, 4),
        [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            1.0, 0.0, 0.0, 0.125,
            0.0, 1.0, 0.0, -0.0625,
            0.0, 0.0, 1.0, 0.03125,
            1.0, 0.0, 0.0, -0.25,
            0.0, 1.0, 0.0, 0.125,
            0.0, 0.0, 1.0, -0.0625,
        ],
    )
    camera_intrinsics = Tensor(
        (1, 3, 3, 3),
        [
            2.0, 0.0, 2.0, 0.0, 2.5, 2.0, 0.0, 0.0, 1.0,
            2.25, 0.0, 2.0, 0.0, 2.75, 2.0, 0.0, 0.0, 1.0,
            2.5, 0.0, 2.0, 0.0, 3.0, 2.0, 0.0, 0.0, 1.0,
        ],
    )
    dual = execute_dualdpt(multiview_values)
    strategy_outputs = {
        strategy: execute_dualdpt(multiview_values, reference_strategy=strategy)
        for strategy in ["first", "middle", "saddle_balanced", "saddle_sim_range"]
    }
    dual_camera = execute_dualdpt(multiview_values, camera_inputs=(camera_extrinsics, camera_intrinsics))
    auxiliary_head_trace = {}
    dual_ray = execute_dualdpt(
        multiview_values,
        use_ray=True,
        head_trace=auxiliary_head_trace,
    )
    ray_trace = ransac_trace_document(dual_ray[2], dual_ray[3], 3)
    prefix_phase_order = [
        "resized_0",
        "resized_1",
        "resized_2",
        "resized_3",
        "refinenet4_aux",
        "refinenet3_aux",
        "refinenet2_aux",
        "refinenet1_aux",
        "output_conv1_aux_0",
        "output_conv1_aux_1",
        "output_conv1_aux_2",
        "output_conv1_aux_3_conv_0",
        "output_conv1_aux_3_conv_1",
        "output_conv1_aux_3_conv_2",
        "output_conv1_aux_3_conv_3",
        "output_conv1_aux_3_conv_4",
        "output_conv1_aux_3",
    ]
    ray_trace["auxiliary_head_prefix_phases"] = [
        tensor_phase_summary(phase, auxiliary_head_trace[phase])
        for phase in prefix_phase_order
    ]
    ray_trace["auxiliary_position_lane"] = auxiliary_head_trace["position_lane"]
    confidence_index = 33
    confidence_logit_index = 6 * 16 * 16 + confidence_index
    confidence_logit = auxiliary_head_trace["logits"].values[confidence_logit_index]
    python_double_exponential = f32(math.exp(confidence_logit))
    libc_exponential = fexp(confidence_logit)
    ray_trace["auxiliary_confidence_activation_lane"] = {
        "confidence_index": confidence_index,
        "logit_index": confidence_logit_index,
        "logit_bits": bits(confidence_logit),
        "python_double_exp_bits": bits(python_double_exponential),
        "libc_expf_bits": bits(libc_exponential),
        "python_double_expp1_bits": bits(fadd(python_double_exponential, 1.0)),
        "libc_expp1_bits": bits(fadd(libc_exponential, 1.0)),
    }
    ray_trace["auxiliary_head_phases"] = {
        phase: tensor_document(auxiliary_head_trace[phase])
        for phase in [
            "positioned",
            "convolution_3_0",
            "normalized",
            "relu",
            "logits",
            "ray",
            "confidence",
        ]
    }
    ray_fixture_state = make_state("dualdpt")
    ray_trace["fixture_state"] = [
        {
            "key": "native.head.scratch.output_conv2_aux.3.5.weight",
            "index": 194,
            "bits": bits(
                ray_fixture_state[
                    "native.head.scratch.output_conv2_aux.3.5.weight"
                ].values[194]
            ),
        },
        {
            "key": "native.head.scratch.output_conv2_aux.3.5.bias",
            "index": 2,
            "bits": bits(
                ray_fixture_state[
                    "native.head.scratch.output_conv2_aux.3.5.bias"
                ].values[2]
            ),
        },
    ]
    mutations = {
        "dpt_local_attention": ("dpt", ("native.backbone.encoder.layer.0.attention.attention.query.weight", 0, 0.25)),
        "dpt_head": ("dpt", ("native.head.scratch.output_conv2.2.bias", 0, 0.25)),
        "dpt_sky": ("dpt", ("native.head.scratch.sky_output_conv2.2.weight", 0, 0.25)),
        "dual_global_attention": ("dual", ("native.backbone.encoder.layer.3.attention.attention.query.weight", 0, 0.25)),
        "dual_learned_camera_token": ("dual", ("native.backbone.embeddings.camera_token", 0, 0.25)),
        "dual_position_head": ("dual", ("native.head.projects.0.bias", 0, 0.25)),
        "dual_camera_decoder": ("dual", ("native.cam_dec.fc_t.weight", 0, 0.25)),
        "dual_camera_encoder": ("camera", ("native.cam_enc.pose_branch.fc1.weight", 0, 0.25)),
        "dual_auxiliary_ray": ("ray", ("native.head.scratch.output_conv2_aux.3.5.weight", 0, 0.25)),
        "dual_unused_retained": ("dual", ("native.head.scratch.output_conv2_aux.0.5.weight", 0, 0.25)),
    }
    mutation_documents = {}
    for name, (profile, mutation) in mutations.items():
        mutation_ray_trace = None
        if profile == "dpt":
            outputs = execute_dpt(values, mutation)
        elif profile == "camera":
            outputs = execute_dualdpt(multiview_values, mutation=mutation, camera_inputs=(camera_extrinsics, camera_intrinsics))
        elif profile == "ray":
            mutation_head_trace = {}
            outputs = execute_dualdpt(
                multiview_values,
                mutation=mutation,
                use_ray=True,
                head_trace=mutation_head_trace,
            )
            mutation_ray_trace = ransac_trace_document(outputs[2], outputs[3], 3)
        else:
            outputs = execute_dualdpt(multiview_values, mutation=mutation)
        mutation_documents[name] = {
            "execution": profile,
            "state_key": mutation[0],
            "lane": mutation[1],
            "delta_bits": bits(mutation[2]),
            "changes_output": name != "dual_unused_retained",
        }
        if mutation_ray_trace is not None:
            mutation_documents[name].update(
                {
                    "depth": tensor_document(outputs[0]),
                    "confidence": tensor_document(outputs[1]),
                    "raw_ray": tensor_document(outputs[2]),
                    "raw_ray_confidence": tensor_document(outputs[3]),
                    "ransac_trace": mutation_ray_trace,
                    "geometry": geometry_document(outputs[4]),
                    "source_raw_ray_changed": outputs[2].values != dual_ray[2].values,
                    "source_geometry_changed": geometry_document(outputs[4])
                    != geometry_document(dual_ray[4]),
                }
            )
        else:
            mutation_documents[name]["output_identity_sha256"] = output_identity(outputs)
    return {
        "format": DOMAIN,
        "generator_command": "PYTHONDONTWRITEBYTECODE=1 python3 crates/sim_test_support/fixtures/models/depth-anything-3-resource-foundation/generate_oracle.py",
        "source_graph_sha256": hashlib.sha256(Path(__file__).with_name("source_graph.py").read_bytes()).hexdigest(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "f32_rule": "round every scalar primitive through IEEE-754 little-endian binary32; libm transcendental bits are scoped to this recorded platform",
        "pinned_sources": SOURCES,
        "input_shape": [1, 4, 4, 3],
        "input_bits": [bits(value) for value in values],
        "multiview_input_shape": [3, 4, 4, 3],
        "multiview_input_bits": [bits(value) for value in multiview_values],
        "same_size_preprocess_bits": [bits(value) for value in preprocess_same_size(values)],
        "nonsquare_resize_projection": {
            "input_shape": [1, 2, 4, 3],
            "input_bits": [bits(value) for value in nonsquare_values],
            "cases": nonsquare_cases,
            "final_projection_mode": "bilinear-align-corners-false",
        },
        "target_cases": [
            {"input": [4, 4, 4, "upper_bound", 2], "output": target_size(4, 4, 4, "upper_bound", 2)},
            {"input": [3, 7, 6, "upper_bound", 2], "output": target_size(3, 7, 6, "upper_bound", 2)},
            {"input": [3, 7, 6, "lower_bound", 2], "output": target_size(3, 7, 6, "lower_bound", 2)},
            {"input": [1, 10000, 14, "upper_bound", 14], "output": target_size(1, 10000, 14, "upper_bound", 14)},
        ],
        "storage_projection": {
            "input_bits": [bits(value) for value in conversion_values],
            "f16_storage_bits": [f16_bits(value) for value in conversion_values],
            "f16_projected_bits": [bits(f16_to_f32(f16_bits(value))) for value in conversion_values],
            "bf16_storage_bits": [bf16_bits(value) for value in conversion_values],
            "bf16_projected_bits": [bits(bf16_to_f32(bf16_bits(value))) for value in conversion_values],
        },
        "checkpoint_projection": {
            profile: {
                source_dtype: checkpoint_projection(profile, source_dtype)
                for source_dtype in ["f32", "f16", "bf16"]
            }
            for profile in ["dpt", "dualdpt"]
        },
        "reference_fixture": {
            "token_bits": [[bits(value) for value in token] for token in tokens],
            **reference_indices(tokens),
        },
        "quaternion_tie": {
            "matrix_bits": [bits(value) for value in [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0]],
            "output_bits": [bits(value) for value in quaternion],
            "selected_candidate": 1,
        },
        "asymmetric_rope": {
            "input_bits": [bits(value) for value in [0.75, -0.25, 0.5, 1.25]],
            "positions": [2, 1],
            "output_bits": [bits(value) for value in vertical + horizontal],
        },
        "position_grid": {
            "source_size": [5, 3],
            "output_size": [3, 2],
            "xy_bits": position_grid(5, 3, 3, 2),
        },
        "ray_identity_grid": {
            "output_size": [5, 3],
            "x_bits": [bits(value) for value in linspace(-(1.0 - 1.0 / 5.0), 1.0 - 1.0 / 5.0, 5)],
            "y_bits": [bits(value) for value in linspace(-(1.0 - 1.0 / 3.0), 1.0 - 1.0 / 3.0, 3)],
        },
        "camera_projection": {
            "height": height,
            "width": width,
            "fov_bits": [bits(fov_height), bits(fov_width)],
            "intrinsics_diagonal_bits": [bits(fx), bits(fy), bits(1.0)],
            "atanf_discriminator": {
                "input_bits": bits(atanf_input),
                "output_bits": bits(fatan(atanf_input)),
                "python_double_cast_bits": bits(f32(math.atan(atanf_input))),
            },
        },
        "reduced_dpt": {
            "depth": tensor_document(dpt_depth),
            "sky": tensor_document(dpt_sky),
            "output_identity_sha256": output_identity(execute_dpt(values)),
            "low_precision": {
                source_dtype: {
                    "depth": tensor_document(outputs[0]),
                    "sky": tensor_document(outputs[2]),
                    "output_identity_sha256": output_identity(outputs),
                }
                for source_dtype, outputs in low_precision_dpt.items()
            },
        },
        "reduced_dualdpt": {
            "depth": tensor_document(dual[0]),
            "confidence": tensor_document(dual[1]),
            "camera": geometry_document(dual[4]),
            "output_identity_sha256": output_identity(dual),
            "supplied_camera_depth": tensor_document(dual_camera[0]),
            "supplied_camera_confidence": tensor_document(dual_camera[1]),
            "supplied_camera": geometry_document(dual_camera[4]),
            "supplied_camera_output_identity_sha256": output_identity(dual_camera),
            "ray_depth": tensor_document(dual_ray[0]),
            "ray_confidence": tensor_document(dual_ray[1]),
            "ray_pose_trace": ray_trace,
            "ray_extrinsics": tensor_document(dual_ray[4][0]),
            "ray_intrinsics": tensor_document(dual_ray[4][1]),
            "ray_output_identity_sha256": output_identity(dual_ray),
            "reference_strategies": {
                strategy: {
                    "depth": tensor_document(outputs[0]),
                    "confidence": tensor_document(outputs[1]),
                    "camera": geometry_document(outputs[4]),
                    "output_identity_sha256": output_identity(outputs),
                }
                for strategy, outputs in strategy_outputs.items()
            },
        },
        "camera_inputs": {
            "extrinsics": tensor_document(camera_extrinsics),
            "intrinsics": tensor_document(camera_intrinsics),
        },
        "full_path_mutations": mutation_documents,
        "ransac": {
            "profile_version": 2,
            "algorithm": "mt19937",
            "sample_ratio_f64_bits": struct.unpack("<Q", struct.pack("<d", 0.3))[0],
            "refit_ratio_f64_bits": struct.unpack("<Q", struct.pack("<d", 0.95))[0],
            "iterations": 100,
            "sample_count": 8,
            "maximum_refit_inliers": 8000,
            "translation_uses_raw_confidence": True,
        },
    }


def encoded():
    result = document()
    if LINUX_GNU_X86_64:
        probes = []
        for index in range(-512, 513):
            value = f32(index / 127.0)
            for operation, native, double in [("atan", fatan, math.atan), ("exp", fexp, math.exp)]:
                native_bits = bits(native(value))
                double_bits = bits(double(value))
                if native_bits != double_bits:
                    probes.append({
                        "operation": operation,
                        "input_bits": bits(value),
                        "output_bits": native_bits,
                        "double_cast_bits": double_bits,
                    })
        if not probes:
            raise RuntimeError("host libm probes do not discriminate F32 from double-cast math")
        result["native_libm_discriminators"] = probes
    return (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()


def check_cross_links(expected):
    expected_hashes = {
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_graph_sha256": hashlib.sha256(Path(__file__).with_name("source_graph.py").read_bytes()).hexdigest(),
    }
    for name in ["manifest.json", "provenance.json"]:
        document = json.loads(Path(__file__).with_name(name).read_text())
        for key, expected_hash in expected_hashes.items():
            if document.get(key) != expected_hash:
                raise SystemExit(f"{name} has stale {key}")
        oracle_hash = (
            document.get("oracle_platform_sha256", {}).get(OUTPUT.name)
            if LINUX_GNU_X86_64 else document.get("oracle_sha256")
        )
        if oracle_hash != hashlib.sha256(expected).hexdigest():
            raise SystemExit(f"{name} has stale digest for {OUTPUT.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    expected = encoded()
    if arguments.check:
        if not OUTPUT.exists() or OUTPUT.read_bytes() != expected:
            raise SystemExit("depth-anything-3 oracle is stale")
        check_cross_links(expected)
        return
    OUTPUT.write_bytes(expected)


if __name__ == "__main__":
    main()
