#!/usr/bin/env python3
import argparse
import ast
import hashlib
import json
import platform
import re
import struct
from pathlib import Path
from types import SimpleNamespace
import unicodedata

SOURCE_SHA = "bb01963178f28efc6e3a9578aecb89f53f8265e89c0518da92ec21c4955e85f1"
SOURCE_PATH = Path("/home/asvegah/repos/sim/projects/sim/ComfyUI/sim_extras/nodes_string.py")
TABLE_NAME = "PYTHON_UNICODE_15_NONPRINTABLE_RANGES"


def generate():
    if platform.python_version() != "3.12.14" or unicodedata.unidata_version != "15.0.0":
        raise RuntimeError("This source projection requires pinned Python 3.12.14 / Unicode 15.0.0")
    source = SOURCE_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA:
        raise RuntimeError("Pinned original source changed")
    original = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "StringFormat")
    namespace = {"io": SimpleNamespace(ComfyNode=object, Schema=object, Autogrow=SimpleNamespace(Type=dict), NodeOutput=lambda *values: values)}
    exec(compile(ast.Module(body=[original], type_ignores=[]), str(SOURCE_PATH), "exec"), namespace)
    execute = namespace["StringFormat"].execute

    def render(template, value):
        output = execute(values={"a": value}, f_string=template)
        if len(output) != 1 or not isinstance(output[0], str):
            raise RuntimeError("Unexpected original StringFormat output")
        return output[0]

    ranges = []
    range_start = None
    last_nonprintable = None
    bitmap = hashlib.sha256()
    representations = {"repr": hashlib.sha256(), "ascii": hashlib.sha256()}
    count = 0
    for codepoint in range(0x110000):
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        character = chr(codepoint)
        printable = character.isprintable()
        bitmap.update(bytes([printable]))
        count += 1
        if not printable:
            if range_start is None:
                range_start = codepoint
            elif last_nonprintable + 1 != codepoint:
                ranges.append([range_start, last_nonprintable])
                range_start = codepoint
            last_nonprintable = codepoint
        elif range_start is not None:
            ranges.append([range_start, last_nonprintable])
            range_start = None
        for name, template in [("repr", "{a!r}"), ("ascii", "{a!a}")]:
            encoded = render(template, character).encode("utf-8")
            representations[name].update(struct.pack("<II", codepoint, len(encoded)))
            representations[name].update(encoded)
    if range_start is not None:
        ranges.append([range_start, last_nonprintable])

    rows = []
    def capture(identifier, template, value):
        rows.append({"id": identifier, "template": template, "value": value, "output": render(template, value)})
    for codepoint in [0, 7, 8, 9, 10, 12, 13, 31, 32, 39, 34, 92, 127, 128, 160, 173, 0x378, 0x200B, 0x2028, 0x2029, 0xE000, 0x1C89, 0x1F600, 0x10FFFF]:
        capture(f"scalar-{codepoint:06x}", "{a!r}|{a!a}", chr(codepoint))
    for identifier, value in [("apostrophe", "don't\u200b"), ("double-quote", 'say "hello"\u2028'), ("both-quotes", 'don\'t say "hello"\ue000'), ("backslash", "\\path\\\u00a0")]:
        capture(identifier, "{a!r}|{a!a}", value)
    capture("nested-dictionary", "{a}|{a!r}|{a!a}", {"z": ["\u200b", "\u2028", "\ue000", "😀"], "a": "don't\u00a0"})
    capture("nested-list", "{a}|{a!r}|{a!a}", ["\u0378", {"\u2029": "\u200b"}, "café", " "])
    capture("converted-width", "{a!r:>20}|{a!a:^24}|{a!r:.10}", "café\u200b")
    capture("converted-astral-width", "{a!r:<12}|{a!a:>16}", "😀")
    float_bits = {0, 1, 0x8000000000000000, 0x8000000000000001, 0x0010000000000000, 0x000fffffffffffff, 0x7fefffffffffffff, 0xffefffffffffffff}
    for value in [1e-5, 1e-4, 1e-3, 1.0, 0.1, 0.3, 1e15, 1e16, 1e17, 1e20, 1.2345678901234567, 2.2250738585072014e-308]:
        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        for offset in [-2, -1, 0, 1, 2]:
            float_bits.add(bits + offset)
            float_bits.add((bits + offset) | 0x8000000000000000)
    state = 0x6f726967696e616c
    for _ in range(2048):
        state ^= (state << 13) & 0xffffffffffffffff
        state ^= state >> 7
        state ^= (state << 17) & 0xffffffffffffffff
        state &= 0xffffffffffffffff
        if (state >> 52) & 0x7ff != 0x7ff:
            float_bits.add(state)
    float_rows = []
    for bits in sorted(float_bits):
        value = struct.unpack("<d", struct.pack("<Q", bits))[0]
        float_rows.append({"id": f"f64-{bits:016x}", "bits_hex": f"{bits:016x}", "template": "{a}|{a!s}|{a!r}|{a!a}", "output": render("{a}|{a!s}|{a!r}|{a!a}", value)})
    container_rows = []
    for name, bits in [("negative-zero", 0x8000000000000000), ("tiny", 0x3ee4f8b588e368f1), ("large", 0x4341c37937e08000), ("subnormal", 1)]:
        value = struct.unpack("<d", struct.pack("<Q", bits))[0]
        container_rows.append({"id": name, "float_bits_hex": f"{bits:016x}", "template": "{a}|{a!r}|{a!a}", "output": render("{a}|{a!r}|{a!a}", {"a": [value, 7, -3, 18446744073709551615], "b": {"c": value}})})
    return {
        "schema_version": 1,
        "task": 413,
        "source": {"file": "sim_extras/nodes_string.py", "sha256": SOURCE_SHA, "class": "StringFormat", "class_line": original.lineno, "unchanged_class_ast_sha256": hashlib.sha256(ast.dump(original, include_attributes=True).encode()).hexdigest()},
        "source_execution": "Unchanged original StringFormat class AST, real Python str.format/repr/ascii/isprintable; descriptor annotations and NodeOutput wrapper captured only",
        "environment": {"python": platform.python_version(), "unicode_data": unicodedata.unidata_version},
        "scalar_scope": {"count": count, "excluded_surrogate_range": [0xD800, 0xDFFF], "printability_bitmap_sha256": bitmap.hexdigest(), "repr_framed_sha256": representations["repr"].hexdigest(), "ascii_framed_sha256": representations["ascii"].hexdigest(), "framing": "For each ascending valid scalar: little-endian u32 codepoint, little-endian u32 UTF-8 output byte count, UTF-8 original output"},
        "nonprintable_ranges": ranges,
        "cases": rows,
        "finite_float_cases": float_rows,
        "finite_float_container_cases": container_rows,
        "remaining_gaps": ["Native strings cannot represent lone UTF-16 surrogates", "Native workflow admission remains unchanged", "Default finite binary64 spelling is source-executed; numeric precision modes and decimal parsing remain separate", "Arbitrary integers, nonfinite JSON extensions and source digit/depth errors remain unproved"],
    }


def table(result):
    lines = [f"const {TABLE_NAME}: &[(u32, u32)] = &["]
    lines.extend(f"    (0x{start:x}, 0x{end:x})," for start, end in result["nonprintable_ranges"])
    return "\n".join(lines) + "\n];"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write-table", action="store_true")
    arguments = parser.parse_args()
    if arguments.check and arguments.write_table:
        raise RuntimeError("Immutable check must not write the production table")
    folder = Path(__file__).resolve().parent
    result = generate()
    encoded = json.dumps(result, ensure_ascii=True, indent=2) + "\n"
    output = folder / "source-contract.json"
    owner = folder.parents[4] / "crates/sim_nodes/src/text_format.rs"
    declaration = table(result)
    pattern = rf"const {TABLE_NAME}: &\[\(u32, u32\)\] = &\[.*?\n\];"
    if arguments.write_table:
        contents = owner.read_text()
        if not re.search(pattern, contents, flags=re.DOTALL):
            raise RuntimeError("Canonical source table insertion point missing")
        owner.write_text(re.sub(pattern, lambda match: declaration, contents, flags=re.DOTALL))
    elif arguments.check:
        found = re.search(pattern, owner.read_text(), flags=re.DOTALL)
        if not found or found.group() != declaration:
            raise RuntimeError("Canonical Unicode table differs from unchanged source predicate")
    if arguments.check:
        if output.read_text() != encoded:
            raise RuntimeError("Immutable original source oracle does not reproduce")
    else:
        output.write_text(encoded)
    print(json.dumps({"mode": "check" if arguments.check else "generate", "scalars": result["scalar_scope"]["count"], "ranges": len(result["nonprintable_ranges"]), "cases": len(result["cases"]), "sha256": hashlib.sha256(encoded.encode()).hexdigest(), "scalar_scope": result["scalar_scope"]}))


if __name__ == "__main__":
    main()
