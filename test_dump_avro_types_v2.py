#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_dump_avro_types.py

Reads a VALID Avro JSON schema (e.g., TCCDMDatum.json) and prints the "type sections"
for ALL (or filtered) records:
  - record name
  - doc (if present)
  - each field: name, resolved type (handles unions & named-type references)
  - notes on referenced enums/fixed (symbol count / size)

Filtering:
  --filter can be given multiple times; records whose NAME matches ANY regex are kept.
  Add --filter-fields to also keep a record if ANY of its FIELD NAMES match.

Usage:
  # dump all records
  python test_dump_avro_types.py --schema TCCDMDatum.json

  # only records with 'Event' or 'Subject' in the name (case-insensitive)
  python test_dump_avro_types.py --schema TCCDMDatum.json --filter Event --filter Subject

  # also match on field names (e.g., records that have a field containing 'uuid')
  python test_dump_avro_types.py --schema TCCDMDatum.json --filter uuid --filter-fields

  # save to markdown
  python test_dump_avro_types.py --schema TCCDMDatum.json --filter Event --out types_event.md
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

Json = Union[Dict[str, Any], List[Any], str, int, float, bool, None]

# ----------------------------- Helpers -----------------------------

def load_schema(path: Path) -> Json:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse JSON schema: {e}")

def iter_named_defs(node: Json, out: Dict[str, Dict[str, Any]]) -> None:
    """
    Collect all named types in the schema: records, enums, fixed.
    Stores by 'name' with the entire node for later reference.
    """
    if isinstance(node, dict):
        t = node.get("type")
        name = node.get("name")
        if name and t in ("record", "enum", "fixed"):
            out[name] = node
        for v in node.values():
            iter_named_defs(v, out)
    elif isinstance(node, list):
        for v in node:
            iter_named_defs(v, out)

def is_primitive(avro_type: Any) -> bool:
    return avro_type in {
        "null","boolean","int","long","float","double","bytes","string"
    }

def resolve_type(typ: Any, named: Dict[str, Dict[str, Any]]) -> str:
    """
    Convert an Avro field 'type' entry to a readable string.

    Supports:
      - primitives (string, int, ...)
      - unions (list of types)
      - named references (record/enum/fixed by name)
      - inline records/enums/fixed
      - arrays, maps, logicalType annotation
    """
    # Union
    if isinstance(typ, list):
        return " | ".join(resolve_type(t, named) for t in typ)

    # Inline dict (complex)
    if isinstance(typ, dict):
        t = typ.get("type")

        # Logical type on primitive
        if is_primitive(t):
            lt = typ.get("logicalType")
            return f"{t}<{lt}>" if lt else t

        if t == "array":
            items = typ.get("items", "any")
            return f"array<{resolve_type(items, named)}>"
        if t == "map":
            values = typ.get("values", "any")
            return f"map<string,{resolve_type(values, named)}>"
        if t == "enum":
            nm = typ.get("name", "enum")
            syms = typ.get("symbols", [])
            return f"{nm} (enum, {len(syms)} symbols)"
        if t == "fixed":
            nm = typ.get("name", "fixed")
            size = typ.get("size", "?")
            return f"{nm} (fixed, size={size})"
        if t == "record":
            nm = typ.get("name", "record")
            return f"{nm} (record)"

        # Fallback
        return resolve_type(t, named)

    # Named or primitive (string)
    if isinstance(typ, str):
        if is_primitive(typ):
            return typ
        if typ in named:
            kind = named[typ].get("type")
            if kind == "enum":
                return f"{typ} (enum, {len(named[typ].get('symbols', []))} symbols)"
            if kind == "fixed":
                return f"{typ} (fixed, size={named[typ].get('size','?')})"
            if kind == "record":
                return f"{typ} (record)"
            return typ
        return f"{typ} (?)"

    return str(typ)

def collect_records(named: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [v for v in named.values() if v.get("type") == "record"]

def format_record(rec: Dict[str, Any], named: Dict[str, Dict[str, Any]]) -> str:
    """Produce a readable 'type section' for a single record."""
    name = rec.get("name", "<unnamed_record>")
    doc = rec.get("doc") or rec.get("documentation") or ""
    fields = rec.get("fields", [])
    lines: List[str] = []
    lines.append(f"Record: {name}")
    if doc:
        lines.append(f"  Doc: {doc.strip()}")
    if not fields:
        lines.append("  (no fields)")
        return "\n".join(lines)
    lines.append("  Fields:")
    for f in fields:
        fname = f.get("name", "<unnamed_field>")
        ftype = resolve_type(f.get("type"), named)
        fdoc = f.get("doc") or ""
        default = f.get("default", None)
        extra = []
        if fdoc:
            extra.append(f"doc='{fdoc.strip()}'")
        if default is not None:
            extra.append(f"default={default!r}")
        suffix = f"  # {', '.join(extra)}" if extra else ""
        lines.append(f"    - {fname}: {ftype}{suffix}")
    return "\n".join(lines)

# ----------------------------- Filtering -----------------------------

def compile_filters(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(pat, re.IGNORECASE) for pat in patterns]

def record_matches_filters(rec: Dict[str, Any], pats: List[re.Pattern], include_fields: bool) -> bool:
    if not pats:
        return True
    name = rec.get("name", "")
    if any(p.search(name) for p in pats):
        return True
    if include_fields:
        for f in rec.get("fields", []) or []:
            fname = f.get("name", "")
            if any(p.search(fname) for p in pats):
                return True
    return False

# ----------------------------- Report -------------------------------

def make_report(schema: Path, filters: List[str], filter_fields: bool) -> str:
    root = load_schema(schema)
    named: Dict[str, Dict[str, Any]] = {}
    iter_named_defs(root, named)

    pats = compile_filters(filters)
    records_all = sorted(collect_records(named), key=lambda r: r.get("name",""))
    records = [r for r in records_all if record_matches_filters(r, pats, filter_fields)]

    out_lines: List[str] = []
    out_lines.append(f"Schema file: {schema}")
    out_lines.append(f"Named types collected: {len(named)} (records/enums/fixed)")
    if pats:
        out_lines.append(f"Filters: {filters}  (fields_included={filter_fields})")
    out_lines.append(f"Records found: {len(records)}\n")

    for i, rec in enumerate(records, 1):
        out_lines.append(f"[{i}/{len(records)}]")
        out_lines.append(format_record(rec, named))
        out_lines.append("")

    # Summary of enums/fixed (always list counts; optionally you could also filter these)
    enums = [(n, d) for n, d in named.items() if d.get("type") == "enum"]
    fixeds = [(n, d) for n, d in named.items() if d.get("type") == "fixed"]

    out_lines.append("----- Summary of other named types -----")
    out_lines.append(f"Enums: {len(enums)}")
    for n, d in sorted(enums, key=lambda x: x[0]):
        out_lines.append(f"  - {n}: {len(d.get('symbols', []))} symbols")
    out_lines.append(f"Fixed: {len(fixeds)}")
    for n, d in sorted(fixeds, key=lambda x: x[0]):
        out_lines.append(f"  - {n}: size={d.get('size','?')}")

    return "\n".join(out_lines)

# ----------------------------- CLI ------------------------------

def main():
    ap = argparse.ArgumentParser(description="Dump type sections for all/filtered Avro records in a JSON schema.")
    ap.add_argument("--schema", type=str, required=True,
                    help="Path to TCCDMDatum.json (valid Avro JSON schema).")
    ap.add_argument("--filter", action="append", default=[],
                    help="Regex to keep records by NAME (case-insensitive). Can be passed multiple times.")
    ap.add_argument("--filter-fields", action="store_true",
                    help="Also keep a record if ANY field NAME matches the filter regexes.")
    ap.add_argument("--out", type=str, default=None,
                    help="Optional path to write Markdown/PlainText output.")
    args = ap.parse_args()

    schema_path = Path(args.schema)
    if not schema_path.exists():
        raise SystemExit(f"Schema file not found: {schema_path}")

    report = make_report(schema_path, args.filter, args.filter_fields)
    print(report)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(report, encoding="utf-8")
        print(f"\nSaved report to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
