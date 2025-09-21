#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_benign_windows_v2.py

Memory-safe, logged, progress-bar version to pick *benign* time windows by time, not by edge ratios.
- Scans only **non_malicious*.parquet** shards under events/ (so chosen windows are benign by construction).
- Streams **row groups** (not the whole dataset) to avoid OOM.
- Counts event density per (host, window_start_ns), where window size = WINDOW_MIN minutes.
- Selects up to K windows **per host-hour** (default K=1) with the highest benign activity.
- Writes CSV with column: window_start_ns (plus host for reference).

Usage (Windows 10 PowerShell):
  python select_benign_windows_v2.py --events-dir ".\\out_parquet_time_only\\events" --window-min 10 --per-hour 1 --out-csv ".\\benign_windows_10m.csv"

Install extra deps (optional for nicer progress):
  pip install tqdm
"""

from __future__ import annotations
import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Tuple, List

import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # fallback if tqdm not installed
    def tqdm(x, **kwargs):
        # fallback tqdm that just returns the iterable
        return x

NS_PER_SEC = 1_000_000_000
NS_PER_MIN = 60 * NS_PER_SEC

# Columns we try to read
TS_COL_CANDIDATES = ["timestampNanos", "tsNanos", "eventTimeNanos", "eventTime", "timestamp", "time"]
HOST_COL_CANDIDATES = ["hostId", "hostName", "source"]

def find_first(names: List[str], candidates: List[str]) -> str | None:
    for c in candidates:
        if c in names:
            return c
    return None

def hour_start_ns(ts_ns: int) -> int:
    return (ts_ns // (3600 * NS_PER_SEC)) * (3600 * NS_PER_SEC)

def choose_benign_windows(events_dir: Path, window_min: int, per_hour: int) -> List[Tuple[str, int]]:
    """Return list of (host, window_start_ns) chosen from non_malicious shards.
    Strategy: for each (host, hour) keep up to `per_hour` windows with highest benign counts.
    """
    win_ns = window_min * NS_PER_MIN

    # counters[(host, ws)] = count
    counters: Dict[Tuple[str, int], int] = {}

    files = sorted([p for p in events_dir.glob("*.parquet") if p.name.lower().startswith("non_malicious")])
    if not files:
        logging.warning("No non_malicious*.parquet files found under %s", events_dir)

    logging.info("Scanning %d non_malicious parquet shards ...", len(files))

    for fpath in tqdm(files, desc="files", unit="file"):
        try:
            pf = pq.ParquetFile(fpath)
        except Exception as e:
            logging.exception("Failed to open %s: %s", fpath, e)
            continue

        schema = getattr(pf, "schema_arrow", None) or pf.schema
        schema_names = list(schema.names)
        ts_col = find_first(schema_names, TS_COL_CANDIDATES)
        if ts_col is None:
            logging.warning("No timestamp column found in %s; skipping", fpath)
            continue
        host_col = find_first(schema_names, HOST_COL_CANDIDATES)

        num_rgs = pf.num_row_groups
        for rg_idx in tqdm(range(num_rgs), desc=f"row_groups({fpath.name})", unit="rg", leave=False):
            try:
                cols = [ts_col] + ([host_col] if host_col else [])
                tbl = pf.read_row_group(rg_idx, columns=cols)
            except Exception as e:
                logging.exception("Failed to read row group %d from %s: %s", rg_idx, fpath, e)
                continue

            # Convert to pandas just for vectorized ops
            def type_mapper(arrow_type):
                if arrow_type == pa.int64():
                    return pd.Int64Dtype()
                elif arrow_type == pa.int32():
                    return pd.Int32Dtype()
                elif arrow_type == pa.float64():
                    return float
                return None  # Let PyArrow handle other types

            df = tbl.to_pandas(types_mapper=type_mapper)
            if df.empty:
                continue

            # Drop NA timestamps
            if ts_col not in df.columns:
                continue
            df = df[df[ts_col].notna()]
            if df.empty:
                continue

            # Compute window start ns for each row (vectorized)
            ts = df[ts_col].astype("int64", errors="ignore")
            ws = (ts // win_ns) * win_ns

            # Host
            if host_col and host_col in df.columns:
                hosts = df[host_col].fillna("UNKNOWN").astype(str)
            else:
                hosts = "UNKNOWN"

            # Aggregate counts per (host, window_start)
            if isinstance(hosts, str):
                # single host string for whole RG
                vc = ws.value_counts()
                for wstart, cnt in vc.items():
                    counters[(hosts, int(wstart))] = counters.get((hosts, int(wstart)), 0) + int(cnt)
            else:
                grp = ws.groupby(hosts).value_counts()
                # grp is a Series indexed by (host, window_start) -> count
                for (host, wstart), cnt in grp.items():
                    counters[(str(host), int(wstart))] = counters.get((str(host), int(wstart)), 0) + int(cnt)

    logging.info("Counted %d distinct benign windows across all hosts", len(counters))

    # Select up to per_hour windows per host-hour, by highest counts
    buckets: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}
    for (host, ws), cnt in counters.items():
        hstart = hour_start_ns(ws)
        buckets.setdefault((host, hstart), []).append((ws, cnt))

    chosen: List[Tuple[str, int]] = []
    for (host, hstart), lst in buckets.items():
        lst.sort(key=lambda x: (-x[1], x[0]))  # by count desc, then earliest
        take = lst[:per_hour]
        for ws, _ in take:
            chosen.append((host, ws))

    chosen.sort(key=lambda x: (x[0], x[1]))
    logging.info("Selected %d benign windows (per_hour=%d)", len(chosen), per_hour)
    return chosen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-dir", required=True, help="Path to .../out_parquet_time_only/events")
    ap.add_argument("--window-min", type=int, default=10)
    ap.add_argument("--per-hour", type=int, default=1, help="Benign windows per host-hour")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"]) 
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    events_dir = Path(args.events_dir)
    if not events_dir.exists():
        logging.error("Events dir not found: %s", events_dir)
        return

    chosen = choose_benign_windows(events_dir, args.window_min, args.per_hour)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["host", "window_start_ns"])  # header
        for host, ws in chosen:
            w.writerow([host, ws])

    logging.info("Saved %d windows to %s", len(chosen), out_csv)

if __name__ == "__main__":
    main()