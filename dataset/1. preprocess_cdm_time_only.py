#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_cdm_time_only.py  —  time-only labeling + detailed logging

Label CDM JSON purely by time windows (ignore host names and hostIds).
- Reads one or more *.json / *.json.gz CDM files (newline-delimited JSON).
- Loads attack windows JSON and unions *all* intervals (half-open [start,end)).
- Streams records once, writes Parquet shards per record type.
- Adds `malicious` only to EVENTS (based on timestamp in any interval).

Logging:
- Per-file START/END with elapsed time
- Per-file counts (lines/events/malicious)
- Running totals after each file
- Optional heartbeat every N lines (default 1,000,000)
- Final total runtime

Requires: pip install pyarrow
"""

import argparse, gzip, json, os, time
from glob import glob
from bisect import bisect_right
from typing import Any, Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

# ----------------------- IO helpers -----------------------

def open_maybe_gzip(path: str):
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    yield line
    else:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    yield line

def get_inner_payload(datum: Dict[str, Any]):
    # datum is usually {"com.bbn.tc.schema.avro.cdm20.X": {...}}
    if not isinstance(datum, dict) or not datum:
        return {}
    return next(iter(datum.values()))

def _unwrap(v):
    # Unwrap {"long": 123}, {"string":"x"}, {"com...UUID":"..."} etc.
    if isinstance(v, dict) and len(v) == 1:
        return next(iter(v.values()))
    return v

# ----------------------- attack windows -----------------------

def coalesce(intervals: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    """Merge overlapping/adjacent half-open intervals [s,e)."""
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le:  # overlap or touch
            if e > le:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]

def load_attack_intervals(attacks_path: str, debug: bool=False) -> List[Tuple[int, int]]:
    """
    Robust loader that supports several schemas:

    A) { "attacks_by_host": { "<name>": [ {"start_ns":..,"end_ns":..}, ... ] } }
    B) { "attacks": [ {"host": "...", "window": {"start_unix_ns":..,"end_unix_ns":..}}, ... ] }
    C) { "intervals": [ [s,e], ... ] }
    D) [ {"start_ns":..,"end_ns":..}, ... ]           # flat list
    E) [ {"window":{"start_unix_ns":..,"end_unix_ns":..}}, ... ]
    """
    with open(attacks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_ivs: List[Tuple[int,int]] = []

    def add_pair(s, e):
        try:
            s = int(s); e = int(e)
        except Exception:
            return
        if e > s:
            all_ivs.append((s, e))

    if isinstance(data, dict):
        # A
        abh = data.get('attacks_by_host')
        if isinstance(abh, dict):
            for _host, arr in abh.items():
                if isinstance(arr, list):
                    for a in arr:
                        if isinstance(a, dict):
                            add_pair(a.get('start_ns'), a.get('end_ns'))

        # B
        if not all_ivs and isinstance(data.get('attacks'), list):
            for a in data['attacks']:
                if not isinstance(a, dict): continue
                w = a.get('window', {})
                add_pair(w.get('start_unix_ns'), w.get('end_unix_ns'))

        # C
        if not all_ivs and isinstance(data.get('intervals'), list):
            for pair in data['intervals']:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    add_pair(pair[0], pair[1])

    elif isinstance(data, list):
        # D/E
        for a in data:
            if isinstance(a, dict):
                if 'start_ns' in a or 'end_ns' in a:
                    add_pair(a.get('start_ns'), a.get('end_ns'))
                elif 'window' in a and isinstance(a['window'], dict):
                    w = a['window']
                    add_pair(w.get('start_unix_ns'), w.get('end_unix_ns'))

    if debug:
        print(f"[DEBUG] Loaded {len(all_ivs)} attack windows before merge.")
    return coalesce(all_ivs)

class TimeOnlyLabeler:
    def __init__(self, attacks_path: str, debug: bool=False):
        self.debug = debug
        self.intervals = load_attack_intervals(attacks_path, debug=debug)
        self.starts = [s for s, _ in self.intervals]
        if self.debug:
            if self.intervals:
                print(f"[DEBUG] Merged into {len(self.intervals)} intervals.")
            else:
                print("[DEBUG] No attack windows loaded! All events will be benign.")

    def ts_in_attack(self, t_ns: int) -> bool:
        # Binary search over starts; half-open [s,e)
        if not self.starts:
            return False
        i = bisect_right(self.starts, t_ns) - 1
        if i >= 0:
            s, e = self.intervals[i]
            return s <= t_ns < e
        return False

# ----------------------- parsers -----------------------

def parse_event(rec: Dict[str, Any]):
    d = rec.get('datum', {})
    if "com.bbn.tc.schema.avro.cdm20.Event" not in d:
        return None
    p = d["com.bbn.tc.schema.avro.cdm20.Event"]

    # Try multiple timestamp keys across performers
    ts = None
    for ts_key in ("timestampNanos","tsNanos","timestamp","eventTimeNanos","eventTime","time"):
        cand = p.get(ts_key)
        if cand is not None:
            cand = _unwrap(cand)
            try:
                ts = int(cand)
                break
            except Exception:
                ts = None
    if ts is None:
        return None

    return {
        'uuid': p.get('uuid'),
        'event_type': p.get('type'),
        'timestampNanos': ts,
        'subject_uuid': _unwrap(p.get('subject')),
        'object1_uuid': _unwrap(p.get('predicateObject')),
        'object2_uuid': _unwrap(p.get('predicateObject2')),
        'size': _unwrap(p.get('size')),
        'hostId': rec.get('hostId'),
        'source': rec.get('source'),
        'sessionNumber': rec.get('sessionNumber'),
    }

def parse_subject(rec: Dict[str, Any]):
    d = rec.get('datum', {})
    if "com.bbn.tc.schema.avro.cdm20.Subject" not in d:
        return None
    p = d["com.bbn.tc.schema.avro.cdm20.Subject"]
    st = _unwrap(p.get('startTimestampNanos'))
    try:
        st = int(st) if st is not None else None
    except Exception:
        st = None
    return {
        'uuid': p.get('uuid'),
        'subject_type': p.get('type'),
        'cid': p.get('cid'),
        'parentSubject': p.get('parentSubject'),
        'localPrincipal': p.get('localPrincipal'),
        'startTimestampNanos': st,
        'cmdLine': p.get('cmdLine'),
        'privilegeLevel': p.get('privilegeLevel'),
        'hostId': rec.get('hostId'),
        'source': rec.get('source'),
        'sessionNumber': rec.get('sessionNumber'),
    }

def parse_fileobject(rec: Dict[str, Any]):
    d = rec.get('datum', {})
    if "com.bbn.tc.schema.avro.cdm20.FileObject" not in d:
        return None
    p = d["com.bbn.tc.schema.avro.cdm20.FileObject"]
    sz = _unwrap(p.get('size'))
    try:
        sz = int(sz)
    except Exception:
        pass
    return {
        'uuid': p.get('uuid'),
        'file_type': p.get('type'),
        'fileDescriptor': p.get('fileDescriptor'),
        'size': sz,
        'hostId': rec.get('hostId'),
        'source': rec.get('source'),
        'sessionNumber': rec.get('sessionNumber'),
    }

def parse_netflow(rec: Dict[str, Any]):
    d = rec.get('datum', {})
    if "com.bbn.tc.schema.avro.cdm20.NetFlowObject" not in d:
        return None
    p = d["com.bbn.tc.schema.avro.cdm20.NetFlowObject"]
    def v(x):
        x = _unwrap(x)
        if isinstance(x, dict) and 'string' in x:
            return x['string']
        if isinstance(x, dict) and 'int' in x:
            return x['int']
        return x
    return {
        'uuid': p.get('uuid'),
        'localAddress': v(p.get('localAddress')),
        'localPort': v(p.get('localPort')),
        'remoteAddress': v(p.get('remoteAddress')),
        'remotePort': v(p.get('remotePort')),
        'ipProtocol': v(p.get('ipProtocol')),
        'fileDescriptor': p.get('fileDescriptor'),
        'hostId': rec.get('hostId'),
        'source': rec.get('source'),
        'sessionNumber': rec.get('sessionNumber'),
    }

def parse_registry(rec: Dict[str, Any]):
    d = rec.get('datum', {})
    if "com.bbn.tc.schema.avro.cdm20.RegistryKeyObject" not in d:
        return None
    p = d["com.bbn.tc.schema.avro.cdm20.RegistryKeyObject"]
    return {
        'uuid': p.get('uuid'),
        'key': p.get('key'),
        'hostId': rec.get('hostId'),
        'source': rec.get('source'),
        'sessionNumber': rec.get('sessionNumber'),
    }

def parse_ipc(rec: Dict[str, Any]):
    d = rec.get('datum', {})
    if "com.bbn.tc.schema.avro.cdm20.IpcObject" not in d:
        return None
    p = d["com.bbn.tc.schema.avro.cdm20.IpcObject"]
    return {
        'uuid': p.get('uuid'),
        'ipc_type': p.get('type'),
        'fd1': _unwrap(p.get('fd1')),
        'fd2': _unwrap(p.get('fd2')),
        'hostId': rec.get('hostId'),
        'source': rec.get('source'),
        'sessionNumber': rec.get('sessionNumber'),
    }

def parse_memory(rec: Dict[str, Any]):
    d = rec.get('datum', {})
    if "com.bbn.tc.schema.avro.cdm20.MemoryObject" not in d:
        return None
    p = d["com.bbn.tc.schema.avro.cdm20.MemoryObject"]
    sz = _unwrap(p.get('size'))
    try:
        sz = int(sz)
    except Exception:
        pass
    return {
        'uuid': p.get('uuid'),
        'memoryAddress': p.get('memoryAddress'),
        'size': sz,
        'hostId': rec.get('hostId'),
        'source': rec.get('source'),
        'sessionNumber': rec.get('sessionNumber'),
    }

# ----------------------- parquet writing -----------------------

def flush_parquet(out_dir: str, name: str, rows: List[Dict[str, Any]], compression: str, parts: Dict[str,int]):
    if not rows:
        return
    table = pa.Table.from_pylist(rows)
    os.makedirs(os.path.join(out_dir, name), exist_ok=True)
    parts[name] = parts.get(name, 0) + 1
    out_path = os.path.join(out_dir, name, f"part-{parts[name]:06d}.parquet")
    pq.write_table(table, out_path, compression=None if compression=='none' else compression)

# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser(description="CDM -> Parquet with time-only malicious labeling (ignore host)")
    ap.add_argument('--input', nargs='+', required=True, help='Paths/globs to CDM .json or .json.gz files')
    ap.add_argument('--attacks', required=True, help='Path to attack windows JSON (various schemas supported)')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--compression', default='zstd', choices=['zstd','snappy','gzip','brotli','none'])
    ap.add_argument('--batch-size', type=int, default=200000)
    ap.add_argument('--progress-every', type=int, default=1_000_000,
                    help='Log a heartbeat every N lines per file (0=disable)')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    # Expand file globs
    files: List[str] = []
    for pat in args.input:
        expanded = glob(pat)
        files.extend(expanded if expanded else [pat])

    t0 = time.perf_counter()
    labeler = TimeOnlyLabeler(args.attacks, debug=args.debug)

    buffers = {k: [] for k in ['events','subjects','fileobjects','netflows','registrykeys','ipc','memory']}
    parts: Dict[str,int] = {}
    grand_lines = grand_events = grand_mal = 0

    for idx, path in enumerate(files, 1):
        print(f"\n[START] ({idx}/{len(files)}) {path}")
        t_file = time.perf_counter()
        file_lines = file_events = file_mal = 0

        try:
            for line_no, line in enumerate(open_maybe_gzip(path), 1):
                if args.progress_every and (line_no % args.progress_every == 0):
                    print(f"  [..] {os.path.basename(path)} lines={line_no:,} events={file_events:,} mal={file_mal:,}")

                file_lines += 1
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                datum = rec.get('datum', {})

                # --- Event detection by inner key ---
                if "com.bbn.tc.schema.avro.cdm20.Event" in datum:
                    row = parse_event(rec)
                    if row:
                        file_events += 1
                        ts = row['timestampNanos']
                        row['malicious'] = 1 if labeler.ts_in_attack(ts) else 0
                        if row['malicious']:
                            file_mal += 1
                        buffers['events'].append(row)
                        if len(buffers['events']) >= args.batch_size:
                            flush_parquet(args.output_dir, 'events', buffers['events'], args.compression, parts)
                            buffers['events'].clear()
                    continue

                # --- Other types by inner key ---
                if "com.bbn.tc.schema.avro.cdm20.Subject" in datum:
                    row = parse_subject(rec)
                    if row:
                        buffers['subjects'].append(row)
                        if len(buffers['subjects']) >= args.batch_size:
                            flush_parquet(args.output_dir, 'subjects', buffers['subjects'], args.compression, parts)
                            buffers['subjects'].clear()
                    continue

                if "com.bbn.tc.schema.avro.cdm20.FileObject" in datum:
                    row = parse_fileobject(rec)
                    if row:
                        buffers['fileobjects'].append(row)
                        if len(buffers['fileobjects']) >= args.batch_size:
                            flush_parquet(args.output_dir, 'fileobjects', buffers['fileobjects'], args.compression, parts)
                            buffers['fileobjects'].clear()
                    continue

                if "com.bbn.tc.schema.avro.cdm20.NetFlowObject" in datum:
                    row = parse_netflow(rec)
                    if row:
                        buffers['netflows'].append(row)
                        if len(buffers['netflows']) >= args.batch_size:
                            flush_parquet(args.output_dir, 'netflows', buffers['netflows'], args.compression, parts)
                            buffers['netflows'].clear()
                    continue

                if "com.bbn.tc.schema.avro.cdm20.RegistryKeyObject" in datum:
                    row = parse_registry(rec)
                    if row:
                        buffers['registrykeys'].append(row)
                        if len(buffers['registrykeys']) >= args.batch_size:
                            flush_parquet(args.output_dir, 'registrykeys', buffers['registrykeys'], args.compression, parts)
                            buffers['registrykeys'].clear()
                    continue

                if "com.bbn.tc.schema.avro.cdm20.IpcObject" in datum:
                    row = parse_ipc(rec)
                    if row:
                        buffers['ipc'].append(row)
                        if len(buffers['ipc']) >= args.batch_size:
                            flush_parquet(args.output_dir, 'ipc', buffers['ipc'], args.compression, parts)
                            buffers['ipc'].clear()
                    continue

                if "com.bbn.tc.schema.avro.cdm20.MemoryObject" in datum:
                    row = parse_memory(rec)
                    if row:
                        buffers['memory'].append(row)
                        if len(buffers['memory']) >= args.batch_size:
                            flush_parquet(args.output_dir, 'memory', buffers['memory'], args.compression, parts)
                            buffers['memory'].clear()
                    continue

        finally:
            # Flush any remainder after each file to keep parts bounded
            for name, rows in buffers.items():
                if rows:
                    flush_parquet(args.output_dir, name, rows, args.compression, parts)
                    buffers[name].clear()

            dt_file = time.perf_counter() - t_file
            grand_lines += file_lines
            grand_events += file_events
            grand_mal += file_mal

            print(f"[DONE ] ({idx}/{len(files)}) {path}")
            print(f"        lines={file_lines:,} events={file_events:,} malicious={file_mal:,} elapsed={dt_file:,.2f}s")
            print(f"[TOTAL] so far: lines={grand_lines:,} events={grand_events:,} malicious={grand_mal:,}")

    dt_all = time.perf_counter() - t0
    print("\n========== SUMMARY ==========")
    print(f"Files processed : {len(files)}")
    print(f"Total lines     : {grand_lines:,}")
    print(f"Total events    : {grand_events:,}")
    print(f"Total malicious : {grand_mal:,}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Compression     : {args.compression}")
    print(f"Batch size      : {args.batch_size:,}")
    print(f"Total time      : {dt_all:,.2f}s")
    print("================================")

if __name__ == '__main__':
    main()
