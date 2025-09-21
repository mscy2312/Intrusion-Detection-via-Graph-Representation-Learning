#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess_cdm_time_only_fast_safe.py

Parallel, time-only labeling; SAFE writes (no collisions):
- Each input JSON writes into out_dir/<table>/<file_stem>/part-*.parquet
  so workers never overwrite each other.
"""

import argparse, gzip, json, os, time, multiprocessing as mp, hashlib, re
from glob import glob
from bisect import bisect_right
from typing import Any, Dict, List, Tuple, Optional

try:
    import orjson as fastjson
    def jloads(s: str): return fastjson.loads(s)
except Exception:
    fastjson = None
    def jloads(s: str): return json.loads(s)

import pyarrow as pa
import pyarrow.parquet as pq

# ----------------- helpers -----------------

def open_maybe_gzip(path: str):
    if path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line and line[0] == '{':
                    yield line
    else:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line and line[0] == '{':
                    yield line

def _unwrap(v):
    if isinstance(v, dict) and len(v) == 1:
        return next(iter(v.values()))
    return v

def sanitize_stem(path: str) -> str:
    stem = os.path.basename(path)
    # drop typical .json[.N] tails
    stem = re.sub(r'\.json(\.\d+)?$', '', stem, flags=re.IGNORECASE)
    return re.sub(r'[^A-Za-z0-9._-]+','_', stem)

# ----------------- intervals -----------------

def coalesce(intervals: List[Tuple[int,int]]) -> List[Tuple[int,int]]:
    if not intervals: return []
    intervals.sort(key=lambda x: x[0])
    out=[list(intervals[0])]
    for s,e in intervals[1:]:
        ls,le=out[-1]
        if s<=le:
            if e>le: out[-1][1]=e
        else:
            out.append([s,e])
    return [tuple(x) for x in out]

def load_attack_intervals(attacks_path: str) -> List[Tuple[int,int]]:
    with open(attacks_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    all_ivs=[]
    def add_pair(s,e):
        try:
            s=int(s); e=int(e)
            if e> s: all_ivs.append((s,e))
        except: pass
    if isinstance(data, dict):
        abh=data.get('attacks_by_host')
        if isinstance(abh, dict):
            for arr in abh.values():
                if isinstance(arr, list):
                    for a in arr:
                        if isinstance(a, dict):
                            add_pair(a.get('start_ns'), a.get('end_ns'))
        if not all_ivs and isinstance(data.get('attacks'), list):
            for a in data['attacks']:
                if isinstance(a, dict) and isinstance(a.get('window'), dict):
                    w=a['window']; add_pair(w.get('start_unix_ns'), w.get('end_unix_ns'))
        if not all_ivs and isinstance(data.get('intervals'), list):
            for pair in data['intervals']:
                if isinstance(pair,(list,tuple)) and len(pair)==2: add_pair(pair[0], pair[1])
    elif isinstance(data, list):
        for a in data:
            if isinstance(a, dict):
                if 'start_ns' in a or 'end_ns' in a: add_pair(a.get('start_ns'), a.get('end_ns'))
                elif isinstance(a.get('window'), dict):
                    w=a['window']; add_pair(w.get('start_unix_ns'), w.get('end_unix_ns'))
    return coalesce(all_ivs)

class TimeOnlyLabeler:
    def __init__(self, intervals: List[Tuple[int,int]]):
        self.intervals=intervals
        self.starts=[s for s,_ in intervals]
    def ts_in_attack(self,t_ns:int)->bool:
        if not self.starts: return False
        from bisect import bisect_right
        i=bisect_right(self.starts,t_ns)-1
        if i>=0:
            s,e=self.intervals[i]
            return s<=t_ns<e
        return False

# ----------------- parsers -----------------

def parse_event(datum: Dict[str,Any], rec: Dict[str,Any]) -> Optional[Dict[str,Any]]:
    p=datum.get("com.bbn.tc.schema.avro.cdm20.Event")
    if not p: return None
    ts=None
    for k in ("timestampNanos","tsNanos","timestamp","eventTimeNanos","eventTime","time"):
        v=p.get(k)
        if v is not None:
            try: ts=int(_unwrap(v)); break
            except: pass
    if ts is None: return None
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

def parse_subject(datum, rec):
    p=datum.get("com.bbn.tc.schema.avro.cdm20.Subject")
    if not p: return None
    st=_unwrap(p.get('startTimestampNanos'))
    try: st=int(st) if st is not None else None
    except: st=None
    return {'uuid':p.get('uuid'),'subject_type':p.get('type'),'cid':p.get('cid'),
            'parentSubject':p.get('parentSubject'),'localPrincipal':p.get('localPrincipal'),
            'startTimestampNanos':st,'cmdLine':p.get('cmdLine'),'privilegeLevel':p.get('privilegeLevel'),
            'hostId':rec.get('hostId'),'source':rec.get('source'),'sessionNumber':rec.get('sessionNumber')}

def parse_fileobject(datum, rec):
    p=datum.get("com.bbn.tc.schema.avro.cdm20.FileObject")
    if not p: return None
    sz=_unwrap(p.get('size'))
    try: sz=int(sz)
    except: pass
    return {'uuid':p.get('uuid'),'file_type':p.get('type'),'fileDescriptor':p.get('fileDescriptor'),
            'size':sz,'hostId':rec.get('hostId'),'source':rec.get('source'),'sessionNumber':rec.get('sessionNumber')}

def parse_netflow(datum, rec):
    p=datum.get("com.bbn.tc.schema.avro.cdm20.NetFlowObject")
    if not p: return None
    def v(x):
        x=_unwrap(x)
        if isinstance(x,dict):
            if 'string' in x: return x['string']
            if 'int' in x: return x['int']
        return x
    return {'uuid':p.get('uuid'),'localAddress':v(p.get('localAddress')),'localPort':v(p.get('localPort')),
            'remoteAddress':v(p.get('remoteAddress')),'remotePort':v(p.get('remotePort')),
            'ipProtocol':v(p.get('ipProtocol')),'fileDescriptor':p.get('fileDescriptor'),
            'hostId':rec.get('hostId'),'source':rec.get('source'),'sessionNumber':rec.get('sessionNumber')}

def parse_registry(datum, rec):
    p=datum.get("com.bbn.tc.schema.avro.cdm20.RegistryKeyObject")
    if not p: return None
    return {'uuid':p.get('uuid'),'key':p.get('key'),
            'hostId':rec.get('hostId'),'source':rec.get('source'),'sessionNumber':rec.get('sessionNumber')}

def parse_ipc(datum, rec):
    p=datum.get("com.bbn.tc.schema.avro.cdm20.IpcObject")
    if not p: return None
    return {'uuid':p.get('uuid'),'ipc_type':p.get('type'),'fd1':_unwrap(p.get('fd1')),'fd2':_unwrap(p.get('fd2')),
            'hostId':rec.get('hostId'),'source':rec.get('source'),'sessionNumber':rec.get('sessionNumber')}

def parse_memory(datum, rec):
    p=datum.get("com.bbn.tc.schema.avro.cdm20.MemoryObject")
    if not p: return None
    sz=_unwrap(p.get('size'))
    try: sz=int(sz)
    except: pass
    return {'uuid':p.get('uuid'),'memoryAddress':p.get('memoryAddress'),'size':sz,
            'hostId':rec.get('hostId'),'source':rec.get('source'),'sessionNumber':rec.get('sessionNumber')}

# ----------------- writer (safe) -----------------

def flush_table(base_out: str, table_name: str, file_stem: str, part_idx: int,
                rows: List[Dict[str,Any]], compression: str, row_group_size: int):
    if not rows: return
    table = pa.Table.from_pylist(rows)
    subdir = os.path.join(base_out, table_name, file_stem)  # SAFE: per-file subdir
    os.makedirs(subdir, exist_ok=True)
    out_path = os.path.join(subdir, f"part-{part_idx:06d}.parquet")
    pq.write_table(
        table, out_path,
        compression=None if compression=='none' else compression,
        row_group_size=row_group_size
    )

# ----------------- worker -----------------

def process_one_file(args_tuple):
    (path, out_dir, compression, batch_size, row_group_size, intervals, progress_every) = args_tuple
    t0 = time.perf_counter()
    file_stem = sanitize_stem(path)
    labeler = TimeOnlyLabeler(intervals)

    parts = {k:0 for k in ('events','subjects','fileobjects','netflows','registrykeys','ipc','memory')}
    bufs  = {k:[] for k in parts}
    lines = events = mal = 0

    for line_no, line in enumerate(open_maybe_gzip(path), 1):
        if progress_every and line_no % progress_every == 0:
            print(f"  [..] {os.path.basename(path)} lines={line_no:,} events={events:,} mal={mal:,}")

        lines += 1
        try:
            rec = jloads(line)
        except Exception:
            continue

        datum = rec.get('datum', {})

        if "com.bbn.tc.schema.avro.cdm20.Event" in datum:
            row = parse_event(datum, rec)
            if row:
                events += 1
                row['malicious'] = 1 if labeler.ts_in_attack(row['timestampNanos']) else 0
                if row['malicious']: mal += 1
                bufs['events'].append(row)
                if len(bufs['events']) >= batch_size:
                    parts['events'] += 1
                    flush_table(out_dir, 'events', file_stem, parts['events'], bufs['events'], compression, row_group_size)
                    bufs['events'].clear()
            continue

        r = parse_subject(datum, rec)
        if r:
            bufs['subjects'].append(r)
            if len(bufs['subjects']) >= batch_size:
                parts['subjects'] += 1
                flush_table(out_dir, 'subjects', file_stem, parts['subjects'], bufs['subjects'], compression, row_group_size)
                bufs['subjects'].clear()
            continue

        r = parse_fileobject(datum, rec)
        if r:
            bufs['fileobjects'].append(r)
            if len(bufs['fileobjects']) >= batch_size:
                parts['fileobjects'] += 1
                flush_table(out_dir, 'fileobjects', file_stem, parts['fileobjects'], bufs['fileobjects'], compression, row_group_size)
                bufs['fileobjects'].clear()
            continue

        r = parse_netflow(datum, rec)
        if r:
            bufs['netflows'].append(r)
            if len(bufs['netflows']) >= batch_size:
                parts['netflows'] += 1
                flush_table(out_dir, 'netflows', file_stem, parts['netflows'], bufs['netflows'], compression, row_group_size)
                bufs['netflows'].clear()
            continue

        r = parse_registry(datum, rec)
        if r:
            bufs['registrykeys'].append(r)
            if len(bufs['registrykeys']) >= batch_size:
                parts['registrykeys'] += 1
                flush_table(out_dir, 'registrykeys', file_stem, parts['registrykeys'], bufs['registrykeys'], compression, row_group_size)
                bufs['registrykeys'].clear()
            continue

        r = parse_ipc(datum, rec)
        if r:
            bufs['ipc'].append(r)
            if len(bufs['ipc']) >= batch_size:
                parts['ipc'] += 1
                flush_table(out_dir, 'ipc', file_stem, parts['ipc'], bufs['ipc'], compression, row_group_size)
                bufs['ipc'].clear()
            continue

        r = parse_memory(datum, rec)
        if r:
            bufs['memory'].append(r)
            if len(bufs['memory']) >= batch_size:
                parts['memory'] += 1
                flush_table(out_dir, 'memory', file_stem, parts['memory'], bufs['memory'], compression, row_group_size)
                bufs['memory'].clear()
            continue

    # final flush
    for name, rows in bufs.items():
        if rows:
            parts[name] += 1
            flush_table(out_dir, name, file_stem, parts[name], rows, compression, row_group_size)
            rows.clear()

    dt = time.perf_counter() - t0
    return {'path': path, 'lines': lines, 'events': events, 'malicious': mal, 'elapsed_sec': dt, 'parts': parts}

# ----------------- driver -----------------

def main():
    ap = argparse.ArgumentParser(description="Fast + safe CDM → Parquet (time-only labeling, parallel)")
    ap.add_argument('--input', nargs='+', required=True)
    ap.add_argument('--attacks', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--compression', default='snappy', choices=['snappy','zstd','gzip','brotli','none'])
    ap.add_argument('--batch-size', type=int, default=800000)
    ap.add_argument('--row-group-size', type=int, default=800000)
    ap.add_argument('--workers', type=int, default=max(1, mp.cpu_count()-1))
    ap.add_argument('--progress-every', type=int, default=1_000_000)
    args = ap.parse_args()

    files=[]
    for pat in args.input:
        expanded = glob(pat)
        files.extend(expanded if expanded else [pat])

    os.makedirs(args.output_dir, exist_ok=True)
    intervals = load_attack_intervals(args.attacks)

    print(f"[INFO] Files: {len(files)} | Workers: {args.workers} | Parser: {'orjson' if fastjson else 'json'}")
    t0 = time.perf_counter()

    worker_args = [(p, args.output_dir, args.compression, args.batch_size, args.row_group_size,
                    intervals, args.progress_every) for p in files]

    if args.workers == 1:
        results = [process_one_file(a) for a in worker_args]
    else:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = list(pool.imap_unordered(process_one_file, worker_args))

    total_lines = total_events = total_mal = 0
    for r in sorted(results, key=lambda x: x['path']):
        print(f"[DONE] {r['path']}  lines={r['lines']:,}  events={r['events']:,}  malicious={r['malicious']:,}  time={r['elapsed_sec']:.2f}s  parts={r['parts']}")
        total_lines += r['lines']; total_events += r['events']; total_mal += r['malicious']

    dt = time.perf_counter() - t0
    print("\n========== SUMMARY ==========")
    print(f"Files processed : {len(results)}")
    print(f"Total lines     : {total_lines:,}")
    print(f"Total events    : {total_events:,}")
    print(f"Total malicious : {total_mal:,}")
    print(f"Output dir      : {args.output_dir}")
    print(f"Compression     : {args.compression}")
    print(f"Batch size      : {args.batch_size:,} | Row group: {args.row_group_size:,}")
    print(f"Workers         : {args.workers} | Parser: {'orjson' if fastjson else 'json'}")
    print(f"Total time      : {dt:,.2f}s")
    print("================================")

if __name__ == '__main__':
    main()
