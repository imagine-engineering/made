#!/usr/bin/env python3
import argparse
import ast
import hashlib
import json
import platform
from pathlib import Path
from types import SimpleNamespace
import unicodedata

SOURCE_SHA = "bb01963178f28efc6e3a9578aecb89f53f8265e89c0518da92ec21c4955e85f1"
BASELINE_SHA = "2e288b0974d0a811f55505031360e7f5c41cb95493d2326bafddcd2e832d4af9"


def generate(source_path):
    source = source_path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA:
        raise RuntimeError("Pinned original source changed")
    tree = ast.parse(source)
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "JsonExtractString")
    namespace = {
        "json": json,
        "io": SimpleNamespace(ComfyNode=object, Schema=object, NodeOutput=lambda *values: values),
    }
    exec(compile(ast.Module(body=[original], type_ignores=[]), str(source_path), "exec"), namespace)
    execute = namespace["JsonExtractString"].execute
    cases = []
    gaps = []

    def capture(identifier, json_string, key="value", gap=None):
        case = {"id": identifier, "json_string": json_string, "key": key}
        try:
            output = execute(json_string=json_string, key=key)
            if len(output) != 1 or not isinstance(output[0], str):
                raise RuntimeError("Unexpected original NodeOutput")
            case["output"] = output[0]
        except Exception as error:
            case["error"] = {"type": type(error).__name__, "message": str(error)}
        if gap:
            case["native_gap"] = gap
            gaps.append(case)
        else:
            if "error" in case:
                raise RuntimeError(f"Unplanned original source error: {case}")
            cases.append(case)

    def value(identifier, selected, gap=None):
        capture(identifier, json.dumps({"value": selected}, ensure_ascii=False), gap=gap)

    for identifier, selected in [
        ("null", None), ("true", True), ("false", False),
        ("signed-integer", -987), ("unsigned-integer", 18446744073709551615),
        ("fraction", 1.5), ("integral-float", 1.0), ("negative-zero", -0.0),
        ("direct-empty", ""), ("direct-control", "line\nbreak\t\x00"),
        ("direct-quotes", "don't say \"hello\""), ("direct-printable-unicode", "café 中文 🎵"),
        ("empty-array", []), ("empty-object", {}),
        ("nested-values", [True, False, None, 3, 1.5, {"x": ["café", "中文", "🎵"]}]),
        ("quote-selection", ["don't", 'say "hello"', "don't say \"hello\"", "\\\\path\\file"]),
        ("insertion-order", {"z": "first", "a": "second", "m": ["don't", "line\nbreak"]}),
        ("control-keys", {"line\nbreak": "tab\tvalue", "don't": {"\x00": "\x1f"}}),
    ]:
        gap = "Canonical STRING output rejects source-valid NUL" if identifier == "direct-control" else None
        value(identifier, selected, gap)
    for number in range(32):
        value(f"ascii-control-{number:02x}", [f"before{chr(number)}after"])
    for number in [127, 128, 133, 159]:
        selected = {"x": f"before{chr(number)}after"}
        value(f"c1-control-{number:02x}", selected, "Canonical workflow STRING input rejects literal DEL/C1 controls")
        capture(f"escaped-del-c1-{number:02x}", json.dumps({"value": selected}, ensure_ascii=True))
    for identifier, text, key in [
        ("malformed", "{broken", "value"), ("truncated", '{"value":', "value"),
        ("invalid-escape", r'{"value":"\q"}', "value"),
        ("nonobject-array", '["value"]', "value"), ("nonobject-null", "null", "value"),
        ("nonobject-string", '"value"', "value"), ("nonobject-number", "7", "value"),
        ("missing-key", '{"other":"value"}', "value"),
        ("duplicate-key", '{"value":"old","value":["new\\nline"]}', "value"),
        ("empty-key", '{"":"empty key"}', ""),
    ]:
        capture(identifier, text, key)

    for number in [0x200b, 0x2028, 0xe000]:
        value(f"nonprintable-unicode-{number:x}", [chr(number)], "Canonical repr does not escape all Python nonprintable Unicode")
    value("beyond-u64-integer", 18446744073709551617, "Canonical serde_json representation cannot retain Python arbitrary integer exactly")
    for identifier, text in [("nan", '{"value":NaN}'), ("infinity", '{"value":Infinity}'), ("negative-infinity", '{"value":-Infinity}')]:
        capture(identifier, text, gap="Python json accepts nonfinite extensions; canonical JSON rejects them")
    for identifier, selected in [("large-exponent", 1e20), ("small-exponent", 1e-8)]:
        value(identifier, selected, "Canonical JSON number display does not prove exact Python exponent formatting")
    capture("deep-source-container", '{"value":' + '[' * 130 + '0' + ']' * 130 + '}', gap="Canonical serde_json nesting bound differs from original source")
    capture("integer-digit-limit", '{"value":' + '1' * 4301 + '}', gap="Original integer conversion raises ValueError; canonical parse failure currently returns empty string")
    return {
        "schema_version": 1,
        "task": 676,
        "feature_id": "COMFY-NODE-0276",
        "source": {"file": "sim_extras/nodes_string.py", "sha256": SOURCE_SHA, "class_line": original.lineno, "unchanged_class_ast_sha256": hashlib.sha256(ast.dump(original, include_attributes=True).encode()).hexdigest()},
        "source_execution": "Unchanged original class AST; real stdlib json/str; only ComfyNode and NodeOutput descriptor capture",
        "environment": {"python": platform.python_version(), "unicode_data": unicodedata.unidata_version},
        "preserved_fixture_sha256": BASELINE_SHA,
        "ordinary_cases": cases,
        "capability_cases": gaps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("/home/asvegah/repos/sim/projects/sim/ComfyUI/sim_extras/nodes_string.py"))
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    folder = Path(__file__).resolve().parent
    if hashlib.sha256((folder / "fixture.json").read_bytes()).hexdigest() != BASELINE_SHA:
        raise RuntimeError("Existing source fixture was modified")
    result = generate(arguments.source)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    output = folder / "json-extract-source-contract.json"
    if arguments.check:
        if output.read_text() != encoded:
            raise RuntimeError("Immutable original-source oracle does not reproduce")
    else:
        output.write_text(encoded)
    print(json.dumps({"ordinary": len(result["ordinary_cases"]), "capability": len(result["capability_cases"]), "source_errors": sum("error" in case for case in result["capability_cases"]), "sha256": hashlib.sha256(encoded.encode()).hexdigest(), "mode": "check" if arguments.check else "generate"}))


if __name__ == "__main__":
    main()
