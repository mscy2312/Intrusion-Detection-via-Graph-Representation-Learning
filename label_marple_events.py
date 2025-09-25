#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Relabel MARPLE events in Parquet files by inclusive attack windows.

Rules:
- Attack windows are taken from marple_attack_window.json
- Intervals are treated as [start_ns, end_ns] with the END INCLUDED
- Timestamp column must be specified (e.g., --time-col timestampNanos)
- Overwrites the existing 'malicious' column in each Parquet file
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


# --------------------------- helpers ---------------------------

def load_inclusive_windows(json_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read marple_attack_window.json and return merged intervals (inclusive end)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    intervals: List[Tuple[int, int]] = []
    for host, items in data.get("attacks_by_host", {}).items():
        for it in items:
            s, e = int(it["start_ns"]), int(it["end_ns"])
            if e < s:
                s, e = e, s
            intervals.append((s, e))

    if not intervals:
        raise ValueError(f"No intervals found in {json_path}")

    # Merge overlapping/adjacent intervals with inclusive ends
    intervals.sort()
    merged: List[Tuple[int, int]] = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce + 1:  # overlap or touching
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))

    starts = np.array([s for s, _ in merged], dtype=np.int64)
    ends   = np.array([e for _, e in merged], dtype=np.int64)
    return starts, ends


def label_by_inclusive_windows(ts_ns: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """
    Vectorized membership for inclusive windows.
    For each t: malicious=1 if in any [start, end].
    """
    ts = np.asarray(ts_ns, dtype=np.int64)
    left_counts = np.searchsorted(starts, ts, side="right")  # intervals started
    ends_lt_t = np.searchsorted(ends, ts - 1, side="right")  # intervals ended strictly before t
    active = (left_counts - ends_lt_t) > 0
    return active.astype(np.int8)


def meaningful_outname(infile: Path, out_dir: Path) -> Path:
    """Output name with suffix to distinguish relabeled files."""
    return out_dir / f"{infile.stem}__marple_relabelled.parquet"


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Relabel MARPLE events in Parquet files (overwrite malicious column).")
    ap.add_argument("--input-dir", required=True, help="Folder containing .parquet files")
    ap.add_argument("--json", required=True, help="Path to marple_attack_window.json")
    ap.add_argument("--out-dir", required=True, help="Output folder for relabelled Parquet files")
    ap.add_argument("--time-col", required=True, help="Timestamp column name (e.g., timestampNanos)")
    ap.add_argument("--recursive", action="store_true", help="Scan input-dir recursively for .parquet files")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output files if they exist")
    args = ap.parse_args()

    in_dir  = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    json_path = Path(args.json)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load merged inclusive windows
    starts, ends = load_inclusive_windows(json_path)
    print(f"[INFO] Loaded {len(starts)} merged MARPLE window(s)")

    pattern = "**/*.parquet" if args.recursive else "*.parquet"
    files = sorted(in_dir.glob(pattern))
    if not files:
        print(f"[WARN] No parquet files found in {in_dir}")
        return

    for fp in files:
        try:
            df = pd.read_parquet(fp)
        except Exception as e:
            print(f"[ERROR] Could not read {fp.name}: {e}")
            continue

        if args.time_col not in df.columns:
            print(f"[ERROR] {fp.name} missing column {args.time_col}")
            continue
        if "malicious" not in df.columns:
            print(f"[ERROR] {fp.name} missing column 'malicious'")
            continue

        ts = pd.to_numeric(df[args.time_col], errors="coerce").astype("Int64")
        df = df.loc[ts.notna()].copy()
        ts = ts.astype(np.int64).to_numpy()

        # Overwrite malicious column
        df["malicious"] = label_by_inclusive_windows(ts, starts, ends)

        out_path = meaningful_outname(fp, out_dir)
        if out_path.exists() and not args.overwrite:
            print(f"[SKIP] {out_path.name} exists (use --overwrite).")
            continue

        try:
            df.to_parquet(out_path, index=False)
        except Exception as e:
            print(f"[ERROR] Failed to write {out_path.name}: {e}")
            continue

        print(f"[OK] {fp.name} → {out_path.name} | rows={len(df)} | label=1={int(df['malicious'].sum())} | label=0={int((df['malicious']==0).sum())}")

    print("[DONE]")


if __name__ == "__main__":
    main()
