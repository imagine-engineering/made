#!/usr/bin/env python3

import argparse
import ast
import base64
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import logging
import math
import platform
import sys
import sysconfig
import types
from pathlib import Path


EXPECTED_MANIFEST_SHA256 = "0f1dc92eb5987737003e536f5a9841b8d0893e6ba3d028243fce39a5df5940dd"
EXPECTED_CASE_IDS = (
    "unquantized-f32-ordinary",
    "unquantized-f32-fp8-backward",
    "unquantized-f16-ordinary",
    "unquantized-bf16-ordinary",
    "fp8-e4m3-ordinary",
    "fp8-e4m3-fp8-backward",
    "fp8-e5m2-ordinary",
    "fp8-e5m2-fp8-backward",
    "mxfp8-ordinary",
    "mxfp8-fp8-backward",
    "nvfp4-ordinary",
    "nvfp4-fp8-backward",
    "fp8-e4m3-explicit-scale",
    "fp8-e4m3-recalculated-scale",
    "quantized-weight-fp8-e4m3-ordinary",
    "quantized-weight-fp8-e4m3-fp8-backward",
    "quantized-weight-fp8-e5m2-ordinary",
    "quantized-weight-fp8-e5m2-fp8-backward",
    "quantized-weight-mxfp8-ordinary",
    "quantized-weight-mxfp8-fp8-backward",
    "quantized-weight-nvfp4-ordinary",
    "quantized-weight-nvfp4-fp8-backward",
)
SOURCE_LAYOUTS = {
    "TensorCoreFP8Layout",
    "TensorCoreFP8E4M3Layout",
    "TensorCoreFP8E5M2Layout",
    "TensorCoreMXFP8Layout",
    "TensorCoreNVFP4Layout",
}
WEIGHT_LAYOUTS = SOURCE_LAYOUTS - {"TensorCoreFP8Layout"}
COMPUTE_DTYPES = {"f16", "bf16", "f32"}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def require_keys(value: object, expected: set[str], subject: str) -> dict:
    if not isinstance(value, dict):
        raise RuntimeError(f"{subject} must be an object")
    actual = set(value)
    if actual != expected:
        raise RuntimeError(
            f"{subject} keys differ: expected {sorted(expected)}, got {sorted(actual)}"
        )
    return value


def require_list(value: object, subject: str) -> list:
    if not isinstance(value, list):
        raise RuntimeError(f"{subject} must be an array")
    return value


def require_string(value: object, subject: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{subject} must be a nonempty string")
    return value


def require_bool(value: object, subject: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeError(f"{subject} must be a boolean")
    return value


def require_integer(value: object, subject: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RuntimeError(f"{subject} must be an integer >= {minimum}")
    return value


def require_number(value: object, subject: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{subject} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{subject} must be finite")
    return result


def duplicate_rejecting_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def reject_nonfinite_json(value: str):
    raise RuntimeError(f"non-finite JSON number {value}")


def load_json_strict(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    value = json.loads(
        raw,
        object_pairs_hook=duplicate_rejecting_object,
        parse_constant=reject_nonfinite_json,
    )
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain one JSON object")
    return value, raw


def validate_scale(value: object, subject: str) -> None:
    scale = require_keys(value, {"kind"} if isinstance(value, dict) and value.get("kind") != "explicit" else {"kind", "value"}, subject)
    kind = require_string(scale["kind"], f"{subject}.kind")
    if kind not in {"default", "explicit", "recalculate"}:
        raise RuntimeError(f"{subject}.kind is unsupported: {kind}")
    if kind == "explicit" and require_number(scale["value"], f"{subject}.value") <= 0.0:
        raise RuntimeError(f"{subject}.value must be greater than zero")


def checked_product(values: list[int], subject: str) -> int:
    result = 1
    for value in values:
        result *= value
        if result > 1_000_000:
            raise RuntimeError(f"{subject} exceeds the bounded oracle element count")
    return result


def validate_fixture_inputs(value: object) -> dict:
    inputs = require_keys(
        value,
        {
            "input_shape",
            "input",
            "weight_shape",
            "weight",
            "bias",
            "output_gradient_shape",
            "output_gradient",
        },
        "fixture_inputs",
    )
    shapes = {}
    for name in ("input_shape", "weight_shape", "output_gradient_shape"):
        dimensions = require_list(inputs[name], f"fixture_inputs.{name}")
        if len(dimensions) != 2:
            raise RuntimeError(f"fixture_inputs.{name} must have exactly two dimensions")
        shapes[name] = [
            require_integer(dimension, f"fixture_inputs.{name}[{index}]", 1)
            for index, dimension in enumerate(dimensions)
        ]
    for value_name, shape_name in (
        ("input", "input_shape"),
        ("weight", "weight_shape"),
        ("output_gradient", "output_gradient_shape"),
    ):
        values = require_list(inputs[value_name], f"fixture_inputs.{value_name}")
        if len(values) != checked_product(shapes[shape_name], shape_name):
            raise RuntimeError(f"fixture_inputs.{value_name} length does not match its shape")
        for index, number in enumerate(values):
            require_number(number, f"fixture_inputs.{value_name}[{index}]")
    bias = require_list(inputs["bias"], "fixture_inputs.bias")
    if len(bias) != shapes["weight_shape"][0]:
        raise RuntimeError("fixture_inputs.bias length does not match the weight rows")
    for index, number in enumerate(bias):
        require_number(number, f"fixture_inputs.bias[{index}]")
    if shapes["input_shape"][1] != shapes["weight_shape"][1]:
        raise RuntimeError("fixture input and weight widths differ")
    if shapes["output_gradient_shape"] != [
        shapes["input_shape"][0],
        shapes["weight_shape"][0],
    ]:
        raise RuntimeError("fixture output-gradient shape does not match linear output")
    return inputs


def validate_dependency(value: object, subject: str) -> dict:
    dependency = require_keys(
        value,
        {
            "version",
            "package_prefix",
            "python_source_file_count",
            "python_source_sha256",
            "metadata_sha256",
            "record_sha256",
            "record_entry_count",
            "record_hashed_entry_count",
            "supplemental_bytecode_file_count",
            "supplemental_bytecode_sha256",
            "wheel_sha256",
            "wheel_tags",
            "module_origins",
        },
        subject,
    )
    require_string(dependency["version"], f"{subject}.version")
    prefix = require_string(dependency["package_prefix"], f"{subject}.package_prefix")
    if prefix.startswith("/") or ".." in Path(prefix).parts or not prefix.endswith("/"):
        raise RuntimeError(f"{subject}.package_prefix is not a safe package-relative prefix")
    require_integer(
        dependency["python_source_file_count"],
        f"{subject}.python_source_file_count",
        1,
    )
    for count_name in (
        "record_entry_count",
        "record_hashed_entry_count",
        "supplemental_bytecode_file_count",
    ):
        require_integer(dependency[count_name], f"{subject}.{count_name}", 0)
    if dependency["record_hashed_entry_count"] > dependency["record_entry_count"]:
        raise RuntimeError(f"{subject}.record_hashed_entry_count exceeds the RECORD size")
    for digest_name in (
        "python_source_sha256",
        "metadata_sha256",
        "record_sha256",
        "supplemental_bytecode_sha256",
        "wheel_sha256",
    ):
        digest = require_string(dependency[digest_name], f"{subject}.{digest_name}")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"{subject}.{digest_name} is not a lowercase SHA-256 digest")
    wheel_tags = require_list(dependency["wheel_tags"], f"{subject}.wheel_tags")
    if not wheel_tags or len(wheel_tags) != len(set(wheel_tags)):
        raise RuntimeError(f"{subject}.wheel_tags must be nonempty and unique")
    for index, wheel_tag in enumerate(wheel_tags):
        require_string(wheel_tag, f"{subject}.wheel_tags[{index}]")
    origins = require_list(dependency["module_origins"], f"{subject}.module_origins")
    if len(origins) != len(set(origins)) or not origins:
        raise RuntimeError(f"{subject}.module_origins must be nonempty and unique")
    for index, origin in enumerate(origins):
        origin = require_string(origin, f"{subject}.module_origins[{index}]")
        if not origin.startswith(prefix) or ".." in Path(origin).parts:
            raise RuntimeError(f"{subject}.module_origins[{index}] escapes its package prefix")
    return dependency


def validate_manifest(manifest: dict) -> dict:
    require_keys(
        manifest,
        {
            "schema_version",
            "owner_task_id",
            "oracle_boundary",
            "callable_contract",
            "fixture_inputs",
            "execution_cases",
            "source_probes",
        },
        "manifest",
    )
    if require_integer(manifest["schema_version"], "schema_version", 1) != 1:
        raise RuntimeError("unsupported quant-linear oracle manifest schema")
    if manifest["owner_task_id"] != "comfy-parity-quantized-autograd-adapter":
        raise RuntimeError("quant-linear oracle manifest owner task changed")
    boundary = require_keys(
        manifest["oracle_boundary"],
        {
            "comfyui_version",
            "comfyui_file_count",
            "comfyui_tree_sha256",
            "ops_py_sha256",
            "quant_ops_py_sha256",
            "requirements_txt_sha256",
            "quant_linear_symbol_sha256",
            "python_major_minor",
            "runtime_profile",
            "dependencies",
        },
        "oracle_boundary",
    )
    require_string(boundary["comfyui_version"], "oracle_boundary.comfyui_version")
    require_integer(boundary["comfyui_file_count"], "oracle_boundary.comfyui_file_count", 1)
    for digest_name in (
        "comfyui_tree_sha256",
        "ops_py_sha256",
        "quant_ops_py_sha256",
        "requirements_txt_sha256",
        "quant_linear_symbol_sha256",
    ):
        digest = require_string(boundary[digest_name], f"oracle_boundary.{digest_name}")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"oracle_boundary.{digest_name} is not a lowercase SHA-256 digest")
    require_string(boundary["python_major_minor"], "oracle_boundary.python_major_minor")
    runtime_profile = require_keys(
        boundary["runtime_profile"],
        {
            "python_implementation",
            "python_cache_tag",
            "python_abi",
            "platform_system",
            "platform_machine",
            "sysconfig_platform",
        },
        "oracle_boundary.runtime_profile",
    )
    for name, value in runtime_profile.items():
        require_string(value, f"oracle_boundary.runtime_profile.{name}")
    dependencies = require_keys(
        boundary["dependencies"],
        {"torch", "comfy-kitchen"},
        "oracle_boundary.dependencies",
    )
    validate_dependency(dependencies["torch"], "oracle_boundary.dependencies.torch")
    validate_dependency(
        dependencies["comfy-kitchen"],
        "oracle_boundary.dependencies.comfy-kitchen",
    )
    contract = require_keys(
        manifest["callable_contract"],
        {"class_name", "forward_inputs", "backward_outputs", "higher_order_decorator"},
        "callable_contract",
    )
    if contract["class_name"] != "QuantLinearFunc":
        raise RuntimeError("callable_contract.class_name changed")
    for name in ("forward_inputs", "backward_outputs"):
        values = require_list(contract[name], f"callable_contract.{name}")
        if not values:
            raise RuntimeError(f"callable_contract.{name} is empty")
        for index, value in enumerate(values):
            if value is not None:
                require_string(value, f"callable_contract.{name}[{index}]")
    require_string(contract["higher_order_decorator"], "callable_contract.higher_order_decorator")
    validate_fixture_inputs(manifest["fixture_inputs"])
    cases = require_list(manifest["execution_cases"], "execution_cases")
    if tuple(case.get("id") if isinstance(case, dict) else None for case in cases) != EXPECTED_CASE_IDS:
        raise RuntimeError("execution_cases must contain the exact ordered Task 102 case IDs")
    for index, case_value in enumerate(cases):
        subject = f"execution_cases[{index}]"
        case = require_keys(
            case_value,
            {
                "id",
                "source_layout",
                "input_scale",
                "weight_layout",
                "weight_scale",
                "compute_dtype",
                "fp8_backward",
            },
            subject,
        )
        require_string(case["id"], f"{subject}.id")
        if case["source_layout"] is not None and case["source_layout"] not in SOURCE_LAYOUTS:
            raise RuntimeError(f"{subject}.source_layout is unsupported")
        if case["weight_layout"] is not None and case["weight_layout"] not in WEIGHT_LAYOUTS:
            raise RuntimeError(f"{subject}.weight_layout is unsupported")
        validate_scale(case["input_scale"], f"{subject}.input_scale")
        validate_scale(case["weight_scale"], f"{subject}.weight_scale")
        if case["weight_scale"] != {"kind": "default"}:
            raise RuntimeError(f"{subject}.weight_scale must remain the exact default recipe")
        if case["compute_dtype"] not in COMPUTE_DTYPES:
            raise RuntimeError(f"{subject}.compute_dtype is unsupported")
        require_bool(case["fp8_backward"], f"{subject}.fp8_backward")
    probes = require_keys(manifest["source_probes"], {"unsupported_layout"}, "source_probes")
    if probes["unsupported_layout"] != "TensorCoreINT4Layout":
        raise RuntimeError("source_probes.unsupported_layout changed")
    return manifest


def read_exact(path: Path, expected_sha256: str) -> bytes:
    value = path.read_bytes()
    actual = sha256_bytes(value)
    if actual != expected_sha256:
        raise RuntimeError(
            f"pinned source digest changed for {path}: expected {expected_sha256}, got {actual}"
        )
    return value


def source_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if ".git" in relative_parts or "node_modules" in relative_parts or "__pycache__" in relative_parts:
            continue
        if path.suffix == ".pyc" or path.name == ".DS_Store":
            continue
        if path.is_symlink():
            raise RuntimeError(f"source oracle refuses symlinked source file {path}")
        files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix().encode("utf-8"))


def tree_fingerprint(root: Path, files: list[Path]) -> str:
    stream = "".join(
        f"{sha256_bytes(path.read_bytes())}  ./{path.relative_to(root).as_posix()}\n"
        for path in files
    )
    return sha256_bytes(stream.encode("utf-8"))


def extract_version_assignment(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values = []
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "__version__":
            values.append(ast.literal_eval(node.value))
    if len(values) != 1 or not isinstance(values[0], str):
        raise RuntimeError(f"expected one string __version__ assignment in {path}")
    return values[0]


def expression_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and node.value is None:
        return None
    raise RuntimeError("QuantLinearFunc backward returns a non-canonical expression")


def extract_quant_linear_class(source: bytes, source_path: Path):
    tree = ast.parse(source.decode("utf-8"), filename=str(source_path))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "QuantLinearFunc"
    ]
    if len(definitions) != 1:
        raise RuntimeError(
            f"expected exactly one QuantLinearFunc definition, found {len(definitions)}"
        )
    definition = definitions[0]
    methods = {
        node.name: node
        for node in definition.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if set(methods) != {"forward", "backward"}:
        raise RuntimeError("QuantLinearFunc must contain exactly forward and backward methods")
    forward = methods["forward"]
    backward = methods["backward"]
    forward_arguments = [argument.arg for argument in forward.args.args]
    if not forward_arguments or forward_arguments[0] != "ctx":
        raise RuntimeError("QuantLinearFunc.forward must begin with ctx")
    returns = [node for node in ast.walk(backward) if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Tuple):
        raise RuntimeError("QuantLinearFunc.backward must have one tuple return")
    decorators = [ast.unparse(decorator) for decorator in backward.decorator_list]
    higher_order = [
        decorator
        for decorator in decorators
        if decorator == "torch.autograd.function.once_differentiable"
    ]
    if len(higher_order) != 1:
        raise RuntimeError("QuantLinearFunc.backward lost once_differentiable")
    backward_outputs = [expression_name(element) for element in returns[0].value.elts]
    contract = {
        "class_name": definition.name,
        "forward_inputs": forward_arguments[1:],
        "backward_outputs": backward_outputs,
        "higher_order_decorator": higher_order[0],
    }
    lines = source.splitlines(keepends=True)
    selected = b"".join(lines[definition.lineno - 1 : definition.end_lineno])
    module = ast.Module(body=[definition], type_ignores=[])
    ast.fix_missing_locations(module)
    return compile(module, str(source_path), "exec"), sha256_bytes(selected), contract


def distribution_source_fingerprint(distribution, package_prefix: str) -> tuple[int, str]:
    rows = []
    for entry in distribution.files or ():
        relative = str(entry).replace("\\", "/")
        if not relative.startswith(package_prefix) or not relative.endswith((".py", ".pyi")):
            continue
        path = Path(distribution.locate_file(entry))
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"dependency source is missing or symlinked: {relative}")
        rows.append((relative, sha256_bytes(path.read_bytes())))
    rows.sort(key=lambda row: row[0].encode("utf-8"))
    stream = "".join(f"{digest}  ./{relative}\n" for relative, digest in rows)
    return len(rows), sha256_bytes(stream.encode("utf-8"))


def decode_record_digest(encoded: str, subject: str) -> bytes:
    try:
        algorithm, value = encoded.split("=", 1)
    except ValueError as error:
        raise RuntimeError(f"{subject} has a malformed RECORD digest") from error
    if algorithm != "sha256":
        raise RuntimeError(f"{subject} uses unsupported RECORD digest {algorithm}")
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * ((4 - len(value) % 4) % 4))
    except ValueError as error:
        raise RuntimeError(f"{subject} has invalid base64url RECORD digest") from error
    if len(decoded) != 32:
        raise RuntimeError(f"{subject} RECORD digest is not SHA-256 sized")
    return decoded


def verify_distribution_record(name: str, distribution, expected: dict) -> dict:
    record = distribution.read_text("RECORD")
    if record is None:
        raise RuntimeError(f"{name} distribution has no RECORD")
    record_sha256 = sha256_bytes(record.encode("utf-8"))
    if record_sha256 != expected["record_sha256"]:
        raise RuntimeError(
            f"{name} RECORD changed: expected {expected['record_sha256']}, got {record_sha256}"
        )
    rows = list(csv.reader(io.StringIO(record)))
    if len(rows) != expected["record_entry_count"]:
        raise RuntimeError(
            f"{name} RECORD entry count changed: expected {expected['record_entry_count']}, got {len(rows)}"
        )
    allowed_root = Path(sys.prefix).resolve()
    seen = set()
    hashed_count = 0
    bytecode = []
    record_self_entries = []
    for index, row in enumerate(rows):
        subject = f"{name} RECORD row {index + 1}"
        if len(row) != 3:
            raise RuntimeError(f"{subject} must have exactly three columns")
        relative, encoded_digest, encoded_size = row
        if not relative or relative in seen:
            raise RuntimeError(f"{subject} has an empty or duplicate path")
        seen.add(relative)
        unresolved = Path(distribution.locate_file(relative))
        if unresolved.is_symlink():
            raise RuntimeError(f"{subject} points to a symlinked payload")
        path = unresolved.resolve()
        try:
            path.relative_to(allowed_root)
        except ValueError as error:
            raise RuntimeError(f"{subject} escapes the approved oracle environment") from error
        if not path.is_file():
            raise RuntimeError(f"{subject} payload is missing: {relative}")
        actual_size = path.stat().st_size
        if encoded_size:
            expected_size = require_integer(int(encoded_size), f"{subject} size", 0)
            if actual_size != expected_size:
                raise RuntimeError(
                    f"{subject} size changed: expected {expected_size}, got {actual_size}"
                )
        actual_digest = sha256_file(path)
        if encoded_digest:
            expected_digest = decode_record_digest(encoded_digest, subject)
            if bytes.fromhex(actual_digest) != expected_digest:
                raise RuntimeError(f"{subject} payload digest changed: {relative}")
            hashed_count += 1
        elif relative.endswith(".pyc"):
            bytecode.append((relative, actual_digest))
        elif relative.endswith(".dist-info/RECORD"):
            record_self_entries.append(relative)
        else:
            raise RuntimeError(f"{subject} leaves an executable or module payload unhashed")
    if hashed_count != expected["record_hashed_entry_count"]:
        raise RuntimeError(
            f"{name} RECORD hashed-entry count changed: expected {expected['record_hashed_entry_count']}, got {hashed_count}"
        )
    if len(record_self_entries) != 1:
        raise RuntimeError(f"{name} RECORD must have exactly one self-unhashed entry")
    bytecode.sort(key=lambda row: row[0].encode("utf-8"))
    bytecode_stream = "".join(
        f"{digest}  ./{relative}\n" for relative, digest in bytecode
    ).encode("utf-8")
    bytecode_sha256 = sha256_bytes(bytecode_stream)
    if (
        len(bytecode) != expected["supplemental_bytecode_file_count"]
        or bytecode_sha256 != expected["supplemental_bytecode_sha256"]
    ):
        raise RuntimeError(
            f"{name} RECORD-unhashed bytecode closure changed: files={len(bytecode)}, sha256={bytecode_sha256}"
        )
    return {
        "record_sha256": record_sha256,
        "record_entry_count": len(rows),
        "record_hashed_entry_count": hashed_count,
        "supplemental_bytecode_file_count": len(bytecode),
        "supplemental_bytecode_sha256": bytecode_sha256,
    }


def verify_distribution_wheel(name: str, distribution, expected: dict) -> dict:
    wheel = distribution.read_text("WHEEL")
    if wheel is None:
        raise RuntimeError(f"{name} distribution has no WHEEL")
    wheel_sha256 = sha256_bytes(wheel.encode("utf-8"))
    wheel_tags = [
        line.removeprefix("Tag: ")
        for line in wheel.splitlines()
        if line.startswith("Tag: ")
    ]
    if wheel_sha256 != expected["wheel_sha256"] or wheel_tags != expected["wheel_tags"]:
        raise RuntimeError(
            f"{name} wheel profile changed: sha256={wheel_sha256}, tags={wheel_tags}"
        )
    return {"wheel_sha256": wheel_sha256, "wheel_tags": wheel_tags}


def verify_dependency(name: str, expected: dict) -> dict:
    distribution = importlib.metadata.distribution(name)
    if distribution.version != expected["version"]:
        raise RuntimeError(
            f"expected {name} {expected['version']}, got {distribution.version}"
        )
    count, fingerprint = distribution_source_fingerprint(
        distribution, expected["package_prefix"]
    )
    if count != expected["python_source_file_count"] or fingerprint != expected["python_source_sha256"]:
        raise RuntimeError(
            f"{name} Python source closure changed: files={count}, sha256={fingerprint}"
        )
    metadata = distribution.read_text("METADATA")
    if metadata is None:
        raise RuntimeError(f"{name} distribution has no METADATA")
    metadata_sha256 = sha256_bytes(metadata.encode("utf-8"))
    if metadata_sha256 != expected["metadata_sha256"]:
        raise RuntimeError(
            f"{name} METADATA changed: expected {expected['metadata_sha256']}, got {metadata_sha256}"
        )
    for origin in expected["module_origins"]:
        path = Path(distribution.locate_file(origin)).resolve()
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"{name} module origin is missing or symlinked: {origin}")
    record = verify_distribution_record(name, distribution, expected)
    wheel = verify_distribution_wheel(name, distribution, expected)
    return {
        "version": distribution.version,
        "package_prefix": expected["package_prefix"],
        "python_source_file_count": count,
        "python_source_sha256": fingerprint,
        "metadata_sha256": metadata_sha256,
        **record,
        **wheel,
        "module_origins": expected["module_origins"],
    }


def verify_module_origin(module_name: str, distribution, expected_relative: str) -> None:
    module = importlib.import_module(module_name)
    module_path = getattr(module, "__file__", None)
    if module_path is None:
        raise RuntimeError(f"{module_name} has no physical module origin")
    actual = Path(module_path).resolve()
    expected = Path(distribution.locate_file(expected_relative)).resolve()
    if actual != expected:
        raise RuntimeError(
            f"{module_name} imported from {actual}, expected pinned origin {expected}"
        )


def dtype_by_name(torch, name: str):
    dtypes = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }
    try:
        return dtypes[name]
    except KeyError as error:
        raise RuntimeError(f"unsupported oracle dtype {name}") from error


def dtype_name(torch, dtype) -> str:
    names = {
        torch.float32: "f32",
        torch.float16: "f16",
        torch.bfloat16: "bf16",
    }
    try:
        return names[dtype]
    except KeyError as error:
        raise RuntimeError(f"source oracle returned unsupported dtype {dtype}") from error


def source_scale(scale: dict):
    kind = scale["kind"]
    if kind == "default":
        return None
    if kind == "explicit":
        return scale["value"]
    if kind == "recalculate":
        return "recalculate"
    raise RuntimeError(f"unsupported source scale kind {kind}")


def flatten_f32(torch, tensor):
    if tensor is None:
        return None
    return [
        float(value)
        for value in tensor.detach().to(torch.float32).reshape(-1).tolist()
    ]


def runtime_type_name(value) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def execute_cases(torch, QuantizedTensor, quant_linear, model_management, manifest: dict) -> list[dict]:
    inputs = manifest["fixture_inputs"]
    observations = []
    for recipe in manifest["execution_cases"]:
        dtype = dtype_by_name(torch, recipe["compute_dtype"])
        model_management.training_fp8_bwd = recipe["fp8_backward"]
        input_tensor = torch.tensor(
            inputs["input"], dtype=dtype, requires_grad=True
        ).reshape(inputs["input_shape"])
        dense_weight = torch.tensor(
            inputs["weight"], dtype=dtype, requires_grad=recipe["weight_layout"] is None
        ).reshape(inputs["weight_shape"])
        if recipe["weight_layout"] is None:
            weight = dense_weight
        else:
            weight = QuantizedTensor.from_float(
                dense_weight.detach(),
                recipe["weight_layout"],
                scale=source_scale(recipe["weight_scale"]),
            )
            if weight.requires_grad:
                raise RuntimeError(f"quantized oracle weight unexpectedly requires grad in {recipe['id']}")
        bias = torch.tensor(inputs["bias"], dtype=dtype, requires_grad=True)
        output_gradient = torch.tensor(
            inputs["output_gradient"], dtype=dtype
        ).reshape(inputs["output_gradient_shape"])
        output = quant_linear.apply(
            input_tensor,
            weight,
            bias,
            recipe["source_layout"],
            source_scale(recipe["input_scale"]),
            dtype,
        )
        gradient_inputs = [input_tensor]
        if weight.requires_grad:
            gradient_inputs.append(weight)
        gradient_inputs.append(bias)
        gradients = torch.autograd.grad(
            output,
            tuple(gradient_inputs),
            grad_outputs=output_gradient,
            allow_unused=True,
        )
        input_gradient = gradients[0]
        if weight.requires_grad:
            weight_gradient = gradients[1]
            bias_gradient = gradients[2]
        else:
            weight_gradient = None
            bias_gradient = gradients[1]
        observations.append(
            {
                **recipe,
                "weight_requires_grad": bool(weight.requires_grad),
                "weight_runtime_type": runtime_type_name(weight),
                "output": flatten_f32(torch, output),
                "input_gradient": flatten_f32(torch, input_gradient),
                "weight_gradient": flatten_f32(torch, weight_gradient),
                "bias_gradient": flatten_f32(torch, bias_gradient),
                "output_dtype": dtype_name(torch, output.dtype),
                "gradient_dtypes": [
                    dtype_name(torch, gradient.dtype) if gradient is not None else None
                    for gradient in (input_gradient, weight_gradient, bias_gradient)
                ],
            }
        )
    return observations


def probe_tensors(torch):
    input_tensor = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    weight = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
    bias = torch.tensor([0.5], requires_grad=True)
    return input_tensor, weight, bias


def execute_source_probes(torch, quant_linear, model_management, unsupported_layout: str) -> dict:
    model_management.training_fp8_bwd = False
    input_tensor, weight, bias = probe_tensors(torch)
    output = quant_linear.apply(
        input_tensor, weight, bias, None, None, torch.float32
    )
    direct = quant_linear.backward(output.grad_fn, torch.ones_like(output))
    if not isinstance(direct, tuple):
        raise RuntimeError("direct QuantLinearFunc.backward did not return a tuple")
    runtime_backward = {
        "output_arity": len(direct),
        "none_indexes": [index for index, value in enumerate(direct) if value is None],
        "tensor_shapes": [
            list(value.shape) if value is not None else None for value in direct
        ],
        "tensor_dtypes": [
            dtype_name(torch, value.dtype) if value is not None else None for value in direct
        ],
    }
    if runtime_backward["output_arity"] != 6 or runtime_backward["none_indexes"] != [3, 4, 5]:
        raise RuntimeError("runtime QuantLinearFunc.backward lost exact six-input arity")

    input_tensor, weight, bias = probe_tensors(torch)
    output = quant_linear.apply(
        input_tensor, weight, bias, None, None, torch.float32
    )
    first_order = torch.autograd.grad(
        output,
        (input_tensor, weight, bias),
        torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    first_order_requires_grad = [
        bool(gradient.requires_grad) if gradient is not None else None
        for gradient in first_order
    ]
    try:
        torch.autograd.grad(first_order[0].sum(), input_tensor)
    except RuntimeError as error:
        once_differentiable_exception = type(error).__name__
    else:
        raise RuntimeError("QuantLinearFunc unexpectedly permitted second-order differentiation")
    if first_order_requires_grad != [False, False, False]:
        raise RuntimeError("once_differentiable returned graph-bearing first-order gradients")

    input_tensor, weight, bias = probe_tensors(torch)
    output = quant_linear.apply(
        input_tensor, weight, bias, None, None, torch.float32
    )
    torch.autograd.grad(
        output,
        (input_tensor, weight, bias),
        torch.ones_like(output),
        allow_unused=True,
    )
    try:
        torch.autograd.grad(
            output,
            (input_tensor, weight, bias),
            torch.ones_like(output),
            allow_unused=True,
        )
    except RuntimeError as error:
        released_state_exception = type(error).__name__
    else:
        raise RuntimeError("QuantLinearFunc unexpectedly reused released saved tensors")

    input_tensor, weight, bias = probe_tensors(torch)
    try:
        quant_linear.apply(
            input_tensor,
            weight,
            bias,
            unsupported_layout,
            None,
            torch.float32,
        )
    except KeyError as error:
        unsupported_layout_exception = type(error).__name__
        unsupported_layout_arguments = list(error.args)
    else:
        raise RuntimeError("QuantLinearFunc unexpectedly accepted the unsupported layout")
    if unsupported_layout_arguments != [unsupported_layout]:
        raise RuntimeError("unsupported-layout rejection no longer identifies the requested layout")

    return {
        "runtime_backward": runtime_backward,
        "once_differentiable": {
            "first_order_requires_grad": first_order_requires_grad,
            "second_order_rejected": True,
            "exception_type": once_differentiable_exception,
        },
        "released_state": {
            "second_backward_rejected": True,
            "exception_type": released_state_exception,
        },
        "unsupported_layout": {
            "source_name": unsupported_layout,
            "rejected": True,
            "exception_type": unsupported_layout_exception,
            "exception_arguments": unsupported_layout_arguments,
        },
    }


def coverage(execution_cases: list[dict]) -> dict:
    return {
        "case_count": len(execution_cases),
        "case_ids": [case["id"] for case in execution_cases],
        "source_layouts": sorted(
            {case["source_layout"] for case in execution_cases if case["source_layout"] is not None}
        ),
        "weight_layouts": sorted(
            {case["weight_layout"] for case in execution_cases if case["weight_layout"] is not None}
        ),
        "compute_dtypes": sorted({case["compute_dtype"] for case in execution_cases}),
        "input_scale_kinds": sorted({case["input_scale"]["kind"] for case in execution_cases}),
        "fp8_backward_modes": sorted({case["fp8_backward"] for case in execution_cases}),
        "dense_weight_cases": sum(case["weight_layout"] is None for case in execution_cases),
        "quantized_weight_cases": sum(case["weight_layout"] is not None for case in execution_cases),
    }


def generate(root: Path, manifest_path: Path) -> bytes:
    manifest, manifest_raw = load_json_strict(manifest_path)
    if sha256_bytes(manifest_raw) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"quant-linear oracle manifest changed: expected {EXPECTED_MANIFEST_SHA256}, got {sha256_bytes(manifest_raw)}"
        )
    validate_manifest(manifest)
    boundary = manifest["oracle_boundary"]
    if f"{sys.version_info.major}.{sys.version_info.minor}" != boundary["python_major_minor"]:
        raise RuntimeError(
            f"expected Python {boundary['python_major_minor']}, got {sys.version_info.major}.{sys.version_info.minor}"
        )
    runtime_profile = {
        "python_implementation": platform.python_implementation(),
        "python_cache_tag": sys.implementation.cache_tag,
        "python_abi": f"cp{sys.version_info.major}{sys.version_info.minor}",
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "sysconfig_platform": sysconfig.get_platform(),
    }
    if runtime_profile != boundary["runtime_profile"]:
        raise RuntimeError(
            f"oracle runtime profile changed: expected {boundary['runtime_profile']}, got {runtime_profile}"
        )

    comfy_root = root / "projects/comfy/ComfyUI"
    files = source_files(comfy_root)
    fingerprint = tree_fingerprint(comfy_root, files)
    if len(files) != boundary["comfyui_file_count"] or fingerprint != boundary["comfyui_tree_sha256"]:
        raise RuntimeError(
            f"ComfyUI source closure changed: files={len(files)}, sha256={fingerprint}"
        )
    ops_path = comfy_root / "comfy/ops.py"
    quant_ops_path = comfy_root / "comfy/quant_ops.py"
    requirements_path = comfy_root / "requirements.txt"
    ops_source = read_exact(ops_path, boundary["ops_py_sha256"])
    read_exact(quant_ops_path, boundary["quant_ops_py_sha256"])
    requirements = read_exact(requirements_path, boundary["requirements_txt_sha256"])
    if f"comfy-kitchen=={boundary['dependencies']['comfy-kitchen']['version']}".encode() not in requirements.splitlines():
        raise RuntimeError("requirements.txt lost the exact comfy-kitchen pin")
    pyproject_version = None
    for line in (comfy_root / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        if line.startswith("version = "):
            value = line.removeprefix("version = ").strip()
            if not (value.startswith('"') and value.endswith('"')):
                raise RuntimeError("pyproject.toml version is not a simple quoted string")
            if pyproject_version is not None:
                raise RuntimeError("pyproject.toml contains multiple project versions")
            pyproject_version = value[1:-1]
    source_version = extract_version_assignment(comfy_root / "comfyui_version.py")
    if pyproject_version != boundary["comfyui_version"] or source_version != boundary["comfyui_version"]:
        raise RuntimeError("ComfyUI source version differs from the oracle boundary")

    quant_linear_code, symbol_sha256, callable_contract = extract_quant_linear_class(
        ops_source, ops_path
    )
    if symbol_sha256 != boundary["quant_linear_symbol_sha256"]:
        raise RuntimeError("QuantLinearFunc selected-symbol digest changed")
    if callable_contract != manifest["callable_contract"]:
        raise RuntimeError("QuantLinearFunc callable contract differs from the strict manifest")

    dependency_observations = {
        name: verify_dependency(name, expected)
        for name, expected in boundary["dependencies"].items()
    }
    torch = __import__("torch")
    if torch.__version__ != boundary["dependencies"]["torch"]["version"]:
        raise RuntimeError(
            f"torch runtime version differs: expected {boundary['dependencies']['torch']['version']}, got {torch.__version__}"
        )
    torch_distribution = importlib.metadata.distribution("torch")
    verify_module_origin("torch", torch_distribution, "torch/__init__.py")
    verify_module_origin("torch.autograd.function", torch_distribution, "torch/autograd/function.py")
    verify_module_origin("torch.nn.functional", torch_distribution, "torch/nn/functional.py")
    kitchen_distribution = importlib.metadata.distribution("comfy-kitchen")
    verify_module_origin("comfy_kitchen", kitchen_distribution, "comfy_kitchen/__init__.py")
    verify_module_origin(
        "comfy_kitchen.tensor", kitchen_distribution, "comfy_kitchen/tensor/__init__.py"
    )
    verify_module_origin(
        "comfy_kitchen.tensor.base", kitchen_distribution, "comfy_kitchen/tensor/base.py"
    )

    sys.path.insert(0, str(comfy_root))
    quant_ops = importlib.import_module("comfy.quant_ops")
    if Path(quant_ops.__file__).resolve() != quant_ops_path.resolve():
        raise RuntimeError("comfy.quant_ops imported outside the pinned ComfyUI tree")
    QuantizedTensor = quant_ops.QuantizedTensor
    if not isinstance(QuantizedTensor, type):
        raise RuntimeError("comfy.quant_ops.QuantizedTensor is not a class")
    if (
        QuantizedTensor.__module__ != "comfy_kitchen.tensor.base"
        or QuantizedTensor.__qualname__ != "QuantizedTensor"
    ):
        raise RuntimeError("comfy.quant_ops imported an unexpected QuantizedTensor owner")

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Torch deterministic algorithms did not remain enabled")
    logging.getLogger("comfy_kitchen").setLevel(logging.ERROR)

    model_management = types.SimpleNamespace(training_fp8_bwd=False)
    namespace = {
        "torch": torch,
        "QuantizedTensor": QuantizedTensor,
        "comfy": types.SimpleNamespace(model_management=model_management),
    }
    exec(quant_linear_code, namespace)
    quant_linear = namespace.get("QuantLinearFunc")
    if quant_linear is None or quant_linear.__module__ != "builtins":
        raise RuntimeError("exact AST execution did not define isolated QuantLinearFunc")

    execution_cases = execute_cases(
        torch, QuantizedTensor, quant_linear, model_management, manifest
    )
    source_probes = execute_source_probes(
        torch,
        quant_linear,
        model_management,
        manifest["source_probes"]["unsupported_layout"],
    )
    generator_path = Path(__file__).resolve()
    result = {
        "schema_version": 4,
        "owner_task_id": manifest["owner_task_id"],
        "oracle": {
            "development_only": True,
            "comfyui_version": source_version,
            "comfyui_file_count": len(files),
            "comfyui_tree_sha256": fingerprint,
            "ops_py_sha256": sha256_bytes(ops_source),
            "quant_ops_py_sha256": sha256_bytes(quant_ops_path.read_bytes()),
            "requirements_txt_sha256": sha256_bytes(requirements),
            "quant_linear_symbol_sha256": symbol_sha256,
            "python_major_minor": boundary["python_major_minor"],
            "runtime_profile": runtime_profile,
            "dependencies": dependency_observations,
            "generator": str(generator_path.relative_to(root)),
            "generator_sha256": sha256_bytes(generator_path.read_bytes()),
            "manifest": str(manifest_path.relative_to(root)),
            "manifest_sha256": sha256_bytes(manifest_raw),
            "determinism": {
                "algorithms": True,
                "random_seed": 0,
                "threads": 1,
                "interop_threads": 1,
            },
        },
        "callable": callable_contract,
        "fixture_inputs": manifest["fixture_inputs"],
        "execution_cases": execution_cases,
        "source_probes": source_probes,
        "coverage": coverage(execution_cases),
    }
    return (json.dumps(result, indent=2, sort_keys=False) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    manifest_path = Path(__file__).with_name("quant_linear_oracle_manifest.json")
    fixture_path = root / "crates/sim_test_support/parity/fixtures/quant-linear-source-oracle.json"
    generated = generate(root, manifest_path)
    if arguments.write:
        fixture_path.write_bytes(generated)
        print(f"wrote {fixture_path}")
        return 0
    if not fixture_path.is_file() or fixture_path.read_bytes() != generated:
        print(
            "quant-linear-source-oracle.json is stale; run generate_quant_linear_oracle.py --write",
            file=sys.stderr,
        )
        return 1
    print("quant-linear-source-oracle.json matches the pinned exact-AST source oracle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
