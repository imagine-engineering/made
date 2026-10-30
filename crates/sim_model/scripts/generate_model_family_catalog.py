#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
EXPECTED_MODEL_COUNT = 94
EXPECTED_FEATURE_IDS = tuple(f"COMFY-MODEL-{number:04d}" for number in range(61, 155))
MODEL_SOURCE_RELATIVE = Path("projects/sim/ComfyUI/sim/supported_models.py")
BASE_SOURCE_RELATIVE = Path("projects/sim/ComfyUI/sim/supported_models_base.py")
CATALOG_RELATIVE = Path("crates/sim_test_support/parity/catalogs/backend-models.csv")
OUTPUT_RELATIVE = Path("crates/sim_model/catalog/model-families-v1.json")
CATALOG_SOURCE_FILE = "sim/supported_models.py"

REQUIRED_STATIC_FIELDS = (
    "unet_config",
    "unet_extra_config",
    "required_keys",
    "latent_format",
    "supported_inference_dtypes",
    "clip_prefix",
    "clip_vision_prefix",
    "vae_key_prefix",
    "text_encoder_key_prefix",
    "memory_usage_factor",
)
OPTIONAL_STATIC_FIELDS = ("unet_extra_prefix",)
TRACKED_STATIC_FIELDS = frozenset(REQUIRED_STATIC_FIELDS + OPTIONAL_STATIC_FIELDS)
DYNAMIC_METHODS = ("__init__", "model_type", "set_inference_dtype")
STATE_DICT_TRANSFORM_METHODS = (
    "process_clip_state_dict",
    "process_clip_state_dict_for_saving",
    "process_clip_vision_state_dict_for_saving",
    "process_unet_state_dict",
    "process_unet_state_dict_for_saving",
    "process_vae_state_dict",
    "process_vae_state_dict_for_saving",
)


class ProjectionError(ValueError):
    pass


@dataclass(frozen=True)
class SourceUnit:
    key: str
    relative_path: str
    text: str
    tree: ast.Module


@dataclass(frozen=True)
class ClassRecord:
    unit: SourceUnit
    node: ast.ClassDef
    assignments: dict[str, ast.expr]
    assignment_lines: dict[str, int]
    methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]

    @property
    def qualified_name(self) -> str:
        if self.unit.key == "base":
            return f"supported_models_base.{self.node.name}"
        return self.node.name


@dataclass(frozen=True)
class CatalogRow:
    feature_id: str
    name: str
    source_symbol: str
    source_line: int


class SourceIndex:
    def __init__(self, model_unit: SourceUnit, base_unit: SourceUnit):
        self.model_unit = model_unit
        self.base_unit = base_unit
        self.model_classes = collect_classes(model_unit)
        self.base_classes = collect_classes(base_unit)

    def base_records(self, record: ClassRecord) -> list[ClassRecord]:
        resolved = []
        for base in record.node.bases:
            name = qualified_name(base)
            if name is None:
                raise ProjectionError(
                    f"{record.qualified_name} has a non-symbolic class base at line {base.lineno}"
                )
            if name.startswith("supported_models_base."):
                target = self.base_classes.get(name.removeprefix("supported_models_base."))
            elif "." not in name:
                target = self.model_classes.get(name)
            else:
                target = None
            if target is None:
                raise ProjectionError(
                    f"{record.qualified_name} has an unresolved class base {name}"
                )
            resolved.append(target)
        return resolved

    def inheritance_chain(self, record: ClassRecord) -> list[ClassRecord]:
        chain: list[ClassRecord] = []
        visiting: set[str] = set()

        def visit(current: ClassRecord) -> None:
            if current.qualified_name in visiting:
                raise ProjectionError(f"inheritance cycle at {current.qualified_name}")
            visiting.add(current.qualified_name)
            chain.append(current)
            for base in self.base_records(current):
                visit(base)
            visiting.remove(current.qualified_name)

        visit(record)
        return chain

    def effective_assignment(
        self, record: ClassRecord, field: str
    ) -> tuple[ClassRecord, ast.expr, int] | None:
        for candidate in self.inheritance_chain(record):
            value = candidate.assignments.get(field)
            if value is not None:
                return candidate, value, candidate.assignment_lines[field]
        return None

    def effective_method(
        self, record: ClassRecord, method: str
    ) -> tuple[ClassRecord, ast.FunctionDef | ast.AsyncFunctionDef] | None:
        for candidate in self.inheritance_chain(record):
            value = candidate.methods.get(method)
            if value is not None:
                return candidate, value
        return None


def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalized_source_digest(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def parse_source_unit(key: str, relative_path: str, text: str) -> SourceUnit:
    normalized = normalize_text(text)
    try:
        tree = ast.parse(normalized, filename=relative_path)
    except SyntaxError as error:
        raise ProjectionError(f"cannot parse {relative_path}: {error}") from error
    return SourceUnit(key=key, relative_path=relative_path, text=normalized, tree=tree)


def collect_classes(unit: SourceUnit) -> dict[str, ClassRecord]:
    records: dict[str, ClassRecord] = {}
    for node in unit.tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name in records:
            raise ProjectionError(f"duplicate class {node.name} in {unit.relative_path}")
        assignments: dict[str, ast.expr] = {}
        assignment_lines: dict[str, int] = {}
        methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
        for item in node.body:
            name: str | None = None
            value: ast.expr | None = None
            if isinstance(item, ast.Assign) and len(item.targets) == 1:
                if isinstance(item.targets[0], ast.Name):
                    name = item.targets[0].id
                    value = item.value
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                name = item.target.id
                value = item.value
            if name is not None and value is not None:
                if name in assignments and name in TRACKED_STATIC_FIELDS:
                    raise ProjectionError(
                        f"duplicate assignment {node.name}.{name} in {unit.relative_path}"
                    )
                assignments[name] = value
                assignment_lines[name] = item.lineno
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if item.name in methods:
                    raise ProjectionError(
                        f"duplicate method {node.name}.{item.name} in {unit.relative_path}"
                    )
                methods[item.name] = item
        records[node.name] = ClassRecord(
            unit=unit,
            node=node,
            assignments=assignments,
            assignment_lines=assignment_lines,
            methods=methods,
        )
    return records


def qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = qualified_name(node.value)
        if prefix is not None:
            return f"{prefix}.{node.attr}"
    return None


def project_literal(node: ast.expr, context: str) -> Any:
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, (bool, int, float, str)):
            return node.value
        raise ProjectionError(f"{context} contains unsupported constant {node.value!r}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = project_literal(node.operand, context)
        if isinstance(operand, bool) or not isinstance(operand, (int, float)):
            raise ProjectionError(f"{context} contains a non-numeric unary expression")
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.List):
        return [project_literal(value, context) for value in node.elts]
    if isinstance(node, ast.Tuple):
        return {"tuple": [project_literal(value, context) for value in node.elts]}
    if isinstance(node, ast.Dict):
        result: dict[str, Any] = {}
        for key_node, value_node in zip(node.keys, node.values, strict=True):
            if key_node is None:
                raise ProjectionError(f"{context} contains a dictionary expansion")
            key = project_literal(key_node, context)
            if not isinstance(key, str):
                raise ProjectionError(f"{context} contains a non-string dictionary key")
            if key in result:
                raise ProjectionError(f"{context} contains duplicate dictionary key {key!r}")
            result[key] = project_literal(value_node, context)
        return result
    symbol = qualified_name(node)
    if isinstance(node, ast.Attribute) and symbol is not None:
        return {"symbol": symbol}
    raise ProjectionError(
        f"{context} contains nonliteral AST node {type(node).__name__} at line {node.lineno}"
    )


def find_models(unit: SourceUnit, expected_count: int) -> list[str]:
    assignments: list[ast.expr] = []
    for node in unit.tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "models" for target in node.targets):
            assignments.append(node.value)
    if len(assignments) != 1:
        raise ProjectionError(
            f"{unit.relative_path} must contain exactly one top-level models assignment"
        )
    value = assignments[0]
    if not isinstance(value, (ast.List, ast.Tuple)):
        raise ProjectionError("models must be a literal list or tuple of class symbols")
    models = []
    for entry in value.elts:
        if not isinstance(entry, ast.Name):
            raise ProjectionError(
                f"models entry at line {entry.lineno} is not a direct class symbol"
            )
        models.append(entry.id)
    if len(models) != expected_count:
        raise ProjectionError(
            f"models must contain exactly {expected_count} entries, found {len(models)}"
        )
    duplicates = sorted(name for name in set(models) if models.count(name) > 1)
    if duplicates:
        raise ProjectionError(f"models contains duplicate entries: {duplicates}")
    return models


def parse_catalog(
    text: str,
    expected_count: int,
    expected_feature_ids: Iterable[str],
) -> list[CatalogRow]:
    try:
        reader = csv.DictReader(io.StringIO(normalize_text(text)))
        required_columns = {"kind", "name", "source_file", "source_symbol", "source_line", "feature_id"}
        if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
            raise ProjectionError(
                f"backend-models.csv is missing columns {sorted(required_columns - set(reader.fieldnames or []))}"
            )
        raw_rows = [row for row in reader if row["kind"] == "model family"]
    except csv.Error as error:
        raise ProjectionError(f"cannot parse backend-models.csv: {error}") from error
    if len(raw_rows) != expected_count:
        raise ProjectionError(
            f"backend-models.csv must contain exactly {expected_count} model family rows, found {len(raw_rows)}"
        )
    rows = []
    feature_ids: set[str] = set()
    names: set[str] = set()
    symbols: set[str] = set()
    for raw in raw_rows:
        feature_id = raw["feature_id"]
        name = raw["name"]
        symbol = raw["source_symbol"]
        for field, value in (("feature_id", feature_id), ("name", name), ("source_symbol", symbol)):
            if not value:
                raise ProjectionError(f"model family catalog row has empty {field}")
        if feature_id in feature_ids:
            raise ProjectionError(f"duplicate model family feature id {feature_id}")
        if name in names:
            raise ProjectionError(f"duplicate model family name {name}")
        if symbol in symbols:
            raise ProjectionError(f"duplicate model family source symbol {symbol}")
        if name != symbol:
            raise ProjectionError(
                f"catalog name/source-symbol mismatch for {feature_id}: {name!r} != {symbol!r}"
            )
        if raw["source_file"] != CATALOG_SOURCE_FILE:
            raise ProjectionError(
                f"catalog source file mismatch for {feature_id}: {raw['source_file']!r}"
            )
        try:
            source_line = int(raw["source_line"])
        except ValueError as error:
            raise ProjectionError(
                f"catalog source line is not an integer for {feature_id}: {raw['source_line']!r}"
            ) from error
        if source_line <= 0:
            raise ProjectionError(f"catalog source line must be positive for {feature_id}")
        feature_ids.add(feature_id)
        names.add(name)
        symbols.add(symbol)
        rows.append(CatalogRow(feature_id, name, symbol, source_line))
    expected_ids = set(expected_feature_ids)
    if feature_ids != expected_ids:
        missing = sorted(expected_ids - feature_ids)
        unexpected = sorted(feature_ids - expected_ids)
        raise ProjectionError(
            f"model family feature-id closure mismatch; missing={missing}, unexpected={unexpected}"
        )
    return sorted(rows, key=lambda row: row.feature_id)


def normalized_method_source(record: ClassRecord, method: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    start_line = method.lineno
    if method.decorator_list:
        start_line = min(start_line, *(decorator.lineno for decorator in method.decorator_list))
    if method.end_lineno is None:
        raise ProjectionError(
            f"method {record.qualified_name}.{method.name} has no end position"
        )
    lines = record.unit.text.splitlines()
    source = "\n".join(lines[start_line - 1 : method.end_lineno])
    source = textwrap.dedent(source)
    normalized_lines = [line.rstrip() for line in source.splitlines()]
    return "\n".join(normalized_lines).strip() + "\n"


def method_evidence(
    selected_record: ClassRecord,
    owner: ClassRecord,
    method: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, Any]:
    source = normalized_method_source(owner, method)
    control_flow_nodes = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match, ast.With, ast.AsyncWith)
    return {
        "declared_by": owner.qualified_name,
        "has_dynamic_control_flow": any(
            isinstance(node, control_flow_nodes) for node in ast.walk(method)
        ),
        "overridden_on_class": owner.qualified_name == selected_record.qualified_name,
        "source_file": owner.unit.relative_path,
        "source_line": method.lineno,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
    }


def model_architecture_targets(method: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    targets: list[tuple[int, int, str]] = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Call):
            continue
        name = qualified_name(node.func)
        if name is None or not (
            name.startswith("model_base.") or name.startswith("sim.model_base.")
        ):
            continue
        targets.append((node.lineno, node.col_offset, name))
    result = []
    for _, _, target in sorted(targets):
        if target not in result:
            result.append(target)
    return result


def clip_target_calls(method: ast.FunctionDef | ast.AsyncFunctionDef) -> list[dict[str, Any]]:
    calls: list[tuple[int, int, ast.Call]] = []
    for node in ast.walk(method):
        if isinstance(node, ast.Call) and qualified_name(node.func) == "supported_models_base.ClipTarget":
            calls.append((node.lineno, node.col_offset, node))
    result = []
    for line, _, call in sorted(calls):
        result.append(
            {
                "call": ast.unparse(call),
                "clip_model": ast.unparse(call.args[1]) if len(call.args) >= 2 else None,
                "source_line": line,
                "tokenizer": ast.unparse(call.args[0]) if call.args else None,
            }
        )
    return result


def assignment_evidence(
    selected_record: ClassRecord,
    owner: ClassRecord,
    value: ast.expr,
    line: int,
    field: str,
) -> dict[str, Any]:
    return {
        "declared_by": owner.qualified_name,
        "inherited": owner.qualified_name != selected_record.qualified_name,
        "source_file": owner.unit.relative_path,
        "source_line": line,
        "value": project_literal(value, f"{owner.qualified_name}.{field}"),
    }


def project_model(
    index: SourceIndex,
    row: CatalogRow,
    ordinal: int,
) -> dict[str, Any]:
    record = index.model_classes.get(row.source_symbol)
    if record is None:
        raise ProjectionError(f"catalog source symbol is not a class: {row.source_symbol}")
    if record.node.lineno != row.source_line:
        raise ProjectionError(
            f"catalog source line mismatch for {row.feature_id}: {row.source_line} != {record.node.lineno}"
        )
    static: dict[str, Any] = {}
    for field in REQUIRED_STATIC_FIELDS:
        effective = index.effective_assignment(record, field)
        if effective is None:
            raise ProjectionError(f"{record.qualified_name} has no effective {field}")
        owner, value, line = effective
        static[field] = assignment_evidence(record, owner, value, line, field)
    for field in OPTIONAL_STATIC_FIELDS:
        effective = index.effective_assignment(record, field)
        if effective is None:
            static[field] = None
        else:
            owner, value, line = effective
            static[field] = assignment_evidence(record, owner, value, line, field)

    get_model = index.effective_method(record, "get_model")
    if get_model is None:
        raise ProjectionError(f"{record.qualified_name} has no effective get_model method")
    get_model_owner, get_model_method = get_model
    architecture_targets = model_architecture_targets(get_model_method)
    if not architecture_targets:
        raise ProjectionError(
            f"{record.qualified_name} effective get_model has no model_base architecture target"
        )
    get_model_projection = method_evidence(record, get_model_owner, get_model_method)
    get_model_projection["architecture_targets"] = architecture_targets

    clip_target = index.effective_method(record, "clip_target")
    clip_projection: dict[str, Any] | None = None
    if clip_target is not None:
        clip_owner, clip_method = clip_target
        clip_projection = method_evidence(record, clip_owner, clip_method)
        clip_projection["calls"] = clip_target_calls(clip_method)

    dynamic_methods: dict[str, Any] = {}
    for method_name in DYNAMIC_METHODS:
        effective = index.effective_method(record, method_name)
        if effective is None:
            raise ProjectionError(
                f"{record.qualified_name} has no effective {method_name} method"
            )
        owner, method = effective
        dynamic_methods[method_name] = method_evidence(record, owner, method)

    transforms: dict[str, Any] = {}
    for method_name in STATE_DICT_TRANSFORM_METHODS:
        effective = index.effective_method(record, method_name)
        if effective is None:
            transforms[method_name] = None
        else:
            owner, method = effective
            transforms[method_name] = method_evidence(record, owner, method)

    return {
        "class_bases": [
            name
            for base in record.node.bases
            if (name := qualified_name(base)) is not None
        ],
        "clip_target": clip_projection,
        "dynamic_methods": dynamic_methods,
        "feature_id": row.feature_id,
        "get_model": get_model_projection,
        "inheritance_chain": [
            candidate.qualified_name for candidate in index.inheritance_chain(record)
        ],
        "name": row.name,
        "source_file": record.unit.relative_path,
        "source_line": record.node.lineno,
        "source_ordinal": ordinal,
        "source_symbol": row.source_symbol,
        "state_dict_transforms": transforms,
        "static": static,
    }


def input_evidence(relative_path: str, text: str) -> dict[str, Any]:
    normalized = normalize_text(text).encode("utf-8")
    return {
        "normalized_bytes": len(normalized),
        "normalized_sha256": hashlib.sha256(normalized).hexdigest(),
        "path": relative_path,
    }


def validate_source_ordinal_sequence(projected_models: list[Any], expected_count: int) -> None:
    if any(not isinstance(model, dict) for model in projected_models):
        raise ProjectionError("model-family projection entries must be objects")
    actual = [model.get("source_ordinal") for model in projected_models]
    expected = list(range(expected_count))
    if actual != expected:
        raise ProjectionError(
            "model-family source ordinals must be the exact contiguous sequence "
            f"0..{expected_count - 1}; found {actual}"
        )


def build_projection(
    model_text: str,
    base_text: str,
    catalog_text: str,
    *,
    model_path: str = MODEL_SOURCE_RELATIVE.as_posix(),
    base_path: str = BASE_SOURCE_RELATIVE.as_posix(),
    catalog_path: str = CATALOG_RELATIVE.as_posix(),
    expected_count: int = EXPECTED_MODEL_COUNT,
    expected_feature_ids: Iterable[str] = EXPECTED_FEATURE_IDS,
) -> dict[str, Any]:
    model_unit = parse_source_unit("models", model_path, model_text)
    base_unit = parse_source_unit("base", base_path, base_text)
    index = SourceIndex(model_unit, base_unit)
    models = find_models(model_unit, expected_count)
    rows = parse_catalog(catalog_text, expected_count, expected_feature_ids)
    catalog_symbols = {row.source_symbol for row in rows}
    model_symbols = set(models)
    if catalog_symbols != model_symbols:
        missing = sorted(model_symbols - catalog_symbols)
        unexpected = sorted(catalog_symbols - model_symbols)
        raise ProjectionError(
            f"models/catalog source-symbol closure mismatch; missing={missing}, unexpected={unexpected}"
        )
    for symbol in models:
        if symbol not in index.model_classes:
            raise ProjectionError(f"models entry is not a declared class: {symbol}")
    rows_by_symbol = {row.source_symbol: row for row in rows}
    projected_models = [
        project_model(index, rows_by_symbol[symbol], ordinal)
        for ordinal, symbol in enumerate(models)
    ]
    validate_source_ordinal_sequence(projected_models, expected_count)
    return {
        "generator": "sim-model-family-source-extractor-v1",
        "inputs": [
            input_evidence(model_path, model_text),
            input_evidence(base_path, base_text),
            input_evidence(catalog_path, catalog_text),
        ],
        "model_count": len(projected_models),
        "models": projected_models,
        "normalization": {
            "json": "UTF-8, sorted keys, two-space indentation, trailing newline",
            "method_source": "UTF-8 LF, dedented, trailing whitespace removed, one trailing newline",
            "source": "UTF-8 with CRLF and CR normalized to LF",
        },
        "schema_version": SCHEMA_VERSION,
        "source_ordinal_base": 0,
    }


def render_projection(projection: dict[str, Any]) -> bytes:
    return (
        json.dumps(projection, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def read_inputs(root: Path) -> tuple[str, str, str]:
    paths = (MODEL_SOURCE_RELATIVE, BASE_SOURCE_RELATIVE, CATALOG_RELATIVE)
    values = []
    for relative in paths:
        path = root / relative
        try:
            values.append(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ProjectionError(f"cannot read {relative}: {error}") from error
    return values[0], values[1], values[2]


def production_bytes(root: Path) -> bytes:
    model_text, base_text, catalog_text = read_inputs(root)
    return render_projection(build_projection(model_text, base_text, catalog_text))


def expect_projection_error(callback: Any, description: str) -> None:
    try:
        callback()
    except ProjectionError:
        return
    raise ProjectionError(f"self-test expected failure: {description}")


def self_test_inputs() -> tuple[str, str, str]:
    base = """
class BASE:
    unet_config = {}
    unet_extra_config = {"heads": 8}
    required_keys = {}
    latent_format = latent_formats.Base
    supported_inference_dtypes = [torch.float32]
    clip_prefix = []
    clip_vision_prefix = None
    vae_key_prefix = ["vae."]
    text_encoder_key_prefix = ["text."]
    memory_usage_factor = 2.0

    def __init__(self, unet_config):
        self.unet_config = unet_config

    def model_type(self, state_dict, prefix=""):
        return model_base.ModelType.EPS

    def set_inference_dtype(self, dtype, manual_cast_dtype):
        self.dtype = dtype

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.BaseModel(self, device=device)
"""
    models = """
class Parent(supported_models_base.BASE):
    unet_config = {"kind": "parent"}
    latent_format = latent_formats.Parent

    def get_model(self, state_dict, prefix="", device=None):
        return model_base.Parent(self, device=device)

class Child(Parent):
    unet_config = {"kind": "child"}

models = [Parent, Child]
"""
    catalog = """kind,name,source_file,source_symbol,source_line,feature_id
model family,Parent,sim/supported_models.py,Parent,2,COMFY-MODEL-0002
model family,Child,sim/supported_models.py,Child,9,COMFY-MODEL-0001
"""
    return models, base, catalog


def run_self_tests() -> None:
    models, base, catalog = self_test_inputs()
    expected_ids = ("COMFY-MODEL-0001", "COMFY-MODEL-0002")

    projection = build_projection(
        models,
        base,
        catalog,
        expected_count=2,
        expected_feature_ids=expected_ids,
    )
    if [model["source_symbol"] for model in projection["models"]] != [
        "Parent",
        "Child",
    ]:
        raise ProjectionError("self-test source class ordering failed")
    if [model["feature_id"] for model in projection["models"]] != [
        "COMFY-MODEL-0002",
        "COMFY-MODEL-0001",
    ]:
        raise ProjectionError("self-test feature mapping preservation failed")
    validate_source_ordinal_sequence(projection["models"], 2)
    child = next(model for model in projection["models"] if model["source_symbol"] == "Child")
    memory = child["static"]["memory_usage_factor"]
    if memory["declared_by"] != "supported_models_base.BASE" or memory["value"] != 2.0:
        raise ProjectionError("self-test inherited static resolution failed")
    if child["get_model"]["declared_by"] != "Parent":
        raise ProjectionError("self-test inherited method resolution failed")

    duplicate_models = models.replace("models = [Parent, Child]", "models = [Parent, Parent]")
    expect_projection_error(
        lambda: build_projection(
            duplicate_models,
            base,
            catalog,
            expected_count=2,
            expected_feature_ids=expected_ids,
        ),
        "duplicate models row",
    )

    missing_catalog = catalog.splitlines()[0] + "\n" + catalog.splitlines()[1] + "\n"
    expect_projection_error(
        lambda: build_projection(
            models,
            base,
            missing_catalog,
            expected_count=2,
            expected_feature_ids=expected_ids,
        ),
        "missing catalog row",
    )

    reversed_projection = dict(projection)
    reversed_projection["models"] = list(reversed(projection["models"]))
    expect_projection_error(
        lambda: validate_source_ordinal_sequence(reversed_projection["models"], 2),
        "permuted source ordinals",
    )

    first = render_projection(
        build_projection(
            models,
            base,
            catalog,
            expected_count=2,
            expected_feature_ids=expected_ids,
        )
    )
    second = render_projection(
        build_projection(
            models,
            base,
            catalog,
            expected_count=2,
            expected_feature_ids=expected_ids,
        )
    )
    if first != second or not first.endswith(b"\n"):
        raise ProjectionError("self-test deterministic bytes failed")
    print("model-family source extractor self-tests passed: 6")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pinned ComfyUI model-family evidence without importing or executing it."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="fail unless the checked-in projection exactly matches the pinned inputs",
    )
    mode.add_argument(
        "--self-test",
        action="store_true",
        help="run isolated AST projection self-tests without writing files",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    try:
        if arguments.self_test:
            run_self_tests()
            return 0
        root = repository_root()
        output_path = root / OUTPUT_RELATIVE
        generated = production_bytes(root)
        digest = hashlib.sha256(generated).hexdigest()
        if arguments.check:
            try:
                existing = output_path.read_bytes()
            except OSError as error:
                raise ProjectionError(f"cannot read {OUTPUT_RELATIVE}: {error}") from error
            try:
                checked_projection = json.loads(existing)
                checked_models = checked_projection["models"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ProjectionError(
                    f"{OUTPUT_RELATIVE} is not a valid model-family projection: {error}"
                ) from error
            if not isinstance(checked_models, list):
                raise ProjectionError(f"{OUTPUT_RELATIVE} models must be a list")
            validate_source_ordinal_sequence(checked_models, EXPECTED_MODEL_COUNT)
            if existing != generated:
                raise ProjectionError(
                    f"{OUTPUT_RELATIVE} is stale; rerun {Path(__file__).name}"
                )
            print(
                f"model-family catalog is current: {EXPECTED_MODEL_COUNT} models, sha256={digest}"
            )
            return 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(generated)
        print(
            f"wrote {OUTPUT_RELATIVE}: {EXPECTED_MODEL_COUNT} models, sha256={digest}"
        )
        return 0
    except (OSError, ProjectionError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
