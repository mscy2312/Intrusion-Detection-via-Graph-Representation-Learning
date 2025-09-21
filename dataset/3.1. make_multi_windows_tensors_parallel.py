#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
make_multi_windows_tensors_parallel.py

Parallel 10-minute window tensor builder (NO external DB):
- TARGET: 100 malicious + 200 benign windows (hardcoded)
- Multiple windows per file (mal:2, ben:1), stride 10 min (hardcoded)
- UUID->type mapping: on-the-fly from parquet node folders per window (subjects/fileobjects/netflows)
- Vectorized Arrow batching (reduces Python per-row overhead)
- Profiling per window: read_parse_sec, uuid_typing_sec, tensor_build_sec, save_sec, total_window_sec
- Windows-safe filenames
- Terminal: only progress bars + final summary
- Detailed logs: <OUTPUT_DIR>/win_build.log and <OUTPUT_DIR>/profile.csv

Run:
  python make_multi_windows_tensors_parallel.py
"""

import os, sys, json, csv, time, logging, bisect, random
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable, Optional, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc
from tqdm import tqdm

# ============================ CONFIG ============================
INPUT_DIR        = Path("./out_parquet_time_only")
OUTPUT_DIR       = Path("./win_tensors_balanced")
GT_JSON          = Path("./attack_windows_e5_iso_and_ns_cleaned.json")

TARGET_MAL       = 100
TARGET_BENIGN    = 200
WINDOW_MIN       = 10
STRIDE_MIN       = 10
MAL_WINS_PER_FILE = 2
BEN_WINS_PER_FILE = 1
EDGES_PER_WINDOW = 100_000
SHUFFLE_FILES    = True
MAX_WORKERS      = max(1, (os.cpu_count() or 2) - 1)

SUBMIT_SLACK     = 0.25    # submit ~25% more files than theoretical minimum

# ======================== CONST / HELPERS =======================
BATCH_SIZE = 200_000
EVENT_COLS = ["timestampNanos","event_type","subject_uuid","object1_uuid","object2_uuid"]
NS_PER_SEC = 1_000_000_000
NS_PER_MIN = 60 * NS_PER_SEC

def ns_to_iso(ns:int)->str:
    return datetime.fromtimestamp(ns/1e9, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def ns_to_iso_safe(ns:int)->str:
    # Windows-safe (no colons)
    return datetime.fromtimestamp(ns/1e9, tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

def setup_logging(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "win_build.log"
    logger = logging.getLogger("win-builder")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)  # keep terminal clean
    ch.setFormatter(logging.Formatter("[%(levelname)s %(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(fh); logger.addHandler(ch)
    logger.info(f"Logging to {log_path}")
    return logger

# ---------------------- GT intervals -------------------
def load_attack_intervals(path: Path, log: logging.Logger):
    if not path or not path.exists():
        log.warning("No GT json provided; all windows labeled 0")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ivs = []
    def add_pair(s,e):
        try:
            s=int(s); e=int(e)
            if e>s: ivs.append((s,e))
        except: pass
    if isinstance(data, dict):
        if isinstance(data.get("attacks_by_host"), dict):
            for arr in data["attacks_by_host"].values():
                if isinstance(arr, list):
                    for a in arr: add_pair(a.get("start_ns"), a.get("end_ns"))
        if not ivs and isinstance(data.get("attacks"), list):
            for a in data["attacks"]:
                w=a.get("window",{}); add_pair(w.get("start_unix_ns"), w.get("end_unix_ns"))
        if not ivs and isinstance(data.get("intervals"), list):
            for s,e in data["intervals"]: add_pair(s,e)
    elif isinstance(data, list):
        for a in data:
            if isinstance(a, dict):
                if "start_ns" in a or "end_ns" in a:
                    add_pair(a.get("start_ns"), a.get("end_ns"))
                elif isinstance(a.get("window"), dict):
                    w=a["window"]; add_pair(w.get("start_unix_ns"), w.get("end_unix_ns"))
    ivs.sort()
    # coalesce
    out=[]
    for s,e in ivs:
        if not out or s>out[-1][1]:
            out.append([s,e])
        else:
            out[-1][1]=max(out[-1][1],e)
    log.info(f"Loaded {len(out)} GT intervals (coalesced)")
    return [(s,e) for s,e in out]

def overlaps(ivs, s, e):
    if not ivs: return False
    starts=[x[0] for x in ivs]
    i=bisect.bisect_right(starts, s)-1
    for j in (i,i+1):
        if 0<=j<len(ivs):
            ss,ee=ivs[j]
            if not (e<=ss or s>=ee): return True
    return False

# --------- UUID -> node_type using parquet (per window) ---------
def node_types_from_parquet(base: Path, node_uuids:set) -> Dict[str,str]:
    """
    Query subjects/, fileobjects/, netflows/ datasets with an IN filter for the given UUIDs.
    Anything not found is labeled 'unknown'. No external DB, done per window.
    """
    if not node_uuids:
        return {}
    uuid2type={u:None for u in node_uuids}
    remaining=set(node_uuids)
    for folder in ["subjects","fileobjects","netflows"]:
        p=base/folder
        if not p.exists() or not remaining:
            continue
        dataset=ds.dataset(p, format="parquet")
        # Filter only the UUIDs we still need
        try:
            table=dataset.to_table(filter=pc.field("uuid").isin(list(remaining)), columns=["uuid"])
        except Exception:
            continue
        found=set()
        for r in table.to_pylist():
            u=r.get("uuid")
            if u in remaining:
                uuid2type[u]=folder
                found.add(u)
        remaining-=found
        if not remaining:
            break
    for u in node_uuids:
        if uuid2type[u] is None:
            uuid2type[u]="unknown"
    return uuid2type

# ---------------------- tensors ------------------------
def build_tensors(edges, node_types_map, min_ts, max_ts):
    nodelist=sorted(node_types_map.keys())
    nid={u:i for i,u in enumerate(nodelist)}
    # event types
    etypes=sorted({et for (_,_,et,_) in edges if et is not None})
    event_type2idx={t:i for i,t in enumerate(etypes)}
    # node types
    order=["subjects","fileobjects","netflows","memory","ipc","registrykeys","unknown"]
    present={node_types_map[u] for u in nodelist}
    node_type2idx={}
    for t in order:
        if t in present: node_type2idx[t]=len(node_type2idx)
    for t in sorted(present):
        if t not in node_type2idx: node_type2idx[t]=len(node_type2idx)
    num_nt=max(1,len(node_type2idx)); num_et=max(1,len(event_type2idx))
    # degrees
    deg_in={u:0 for u in nodelist}; deg_out={u:0 for u in nodelist}
    for u,v,et,ts in edges:
        deg_out[u]+=1; deg_in[v]+=1
    deg_tot={u:deg_in[u]+deg_out[u] for u in nodelist}
    max_in=max(deg_in.values()) or 1
    max_out=max(deg_out.values()) or 1
    max_tot=max(deg_tot.values()) or 1
    # node features
    x_rows=[]
    for u in nodelist:
        t=node_types_map.get(u,"unknown")
        t_idx=node_type2idx.get(t, node_type2idx.get("unknown",0))
        onehot=[0.0]*num_nt
        if 0<=t_idx<num_nt: onehot[t_idx]=1.0
        fin=deg_in[u]/max_in; fout=deg_out[u]/max_out; ftot=deg_tot[u]/max_tot
        x_rows.append(onehot+[fin,fout,ftot])
    x=torch.tensor(x_rows, dtype=torch.float32)
    # edges
    ei_src=[]; ei_dst=[]; ea_rows=[]
    span=max(1,(max_ts-min_ts)) if (min_ts is not None and max_ts is not None) else 1
    for u,v,et,ts in edges:
        ei_src.append(nid[u]); ei_dst.append(nid[v])
        et_oh=[0.0]*num_et
        idx=event_type2idx.get(et, None)
        if idx is not None: et_oh[idx]=1.0
        tnorm=((ts-min_ts)/span) if (min_ts is not None) else 0.0
        ea_rows.append(et_oh+[float(max(0.0,min(1.0,tnorm)))])
    edge_index=torch.tensor([ei_src, ei_dst], dtype=torch.long)
    edge_attr=torch.tensor(ea_rows, dtype=torch.float32)
    meta={"node_index_map":{u:nid[u] for u in nodelist},
          "node_type2idx":node_type2idx,
          "event_type2idx":event_type2idx,
          "x_dim":x.shape[1],
          "edge_attr_dim":edge_attr.shape[1]}
    return x, edge_index, edge_attr, meta

# ------------- collectors (successive windows) ----------
def collect_successive_windows_from_file(
    file: Path, win_ns: int, stride_ns: int, edges_cap: Optional[int], max_windows: int
):
    """
    Yield up to max_windows successive shard-local windows from a single file.
    Returns tuples: (ws, we, edges, node_uuids, min_ts, max_ts)
    """
    pf = pq.ParquetFile(file)
    schema = set(pf.schema_arrow.names)
    cols = [c for c in EVENT_COLS if c in schema]
    if not {"timestampNanos","subject_uuid"}.issubset(set(cols)):
        return

    current_ws=None; current_we=None
    edges=[]; node_uuids=set(); min_ts=None; max_ts=None
    windows_yielded=0
    cap = None if not EDGES_PER_WINDOW or EDGES_PER_WINDOW <= 0 else int(EDGES_PER_WINDOW)

    def flush_current():
        nonlocal edges, node_uuids, min_ts, max_ts
        if current_ws is None: return None
        out = (current_ws, current_we, edges, set(node_uuids), min_ts, max_ts)
        edges=[]; node_uuids.clear(); min_ts=None; max_ts=None
        return out

    # vector-friendly: convert batch to columns once
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=cols):
        tbl = pa.Table.from_batches([batch])
        schema = tbl.schema
        get = schema.get_field_index
        ts_col  = tbl.column(get("timestampNanos")).to_pylist()
        sub_col = tbl.column(get("subject_uuid")).to_pylist()
        obj1_col = tbl.column(get("object1_uuid")).to_pylist() if get("object1_uuid")!=-1 else [None]*len(ts_col)
        obj2_col = tbl.column(get("object2_uuid")).to_pylist() if get("object2_uuid")!=-1 else [None]*len(ts_col)
        et_col   = tbl.column(get("event_type")).to_pylist()    if get("event_type")!=-1    else [None]*len(ts_col)

        for i, ts in enumerate(ts_col):
            if ts is None: continue
            ts = int(ts)
            if current_ws is None:
                current_ws = (ts // win_ns) * win_ns
                current_we = current_ws + win_ns
            while ts >= current_we:
                out = flush_current()
                if out and out[2]:
                    yield out
                    windows_yielded += 1
                    if windows_yielded >= max_windows:
                        return
                current_ws += stride_ns
                current_we = current_ws + win_ns

            u=sub_col[i]; v1=obj1_col[i]; v2=obj2_col[i]; et=et_col[i]
            min_ts = ts if min_ts is None else min(min_ts, ts)
            max_ts = ts if max_ts is None else max(max_ts, ts)
            if u and v1:
                edges.append((u,v1,et,ts)); node_uuids.add(u); node_uuids.add(v1)
                if cap and len(edges)>=cap:
                    out = flush_current()
                    if out and out[2]:
                        yield out
                        windows_yielded += 1
                        if windows_yielded >= max_windows:
                            return
                    current_ws += stride_ns; current_we = current_ws + win_ns
            if u and v2:
                edges.append((u,v2,et,ts)); node_uuids.add(u); node_uuids.add(v2)
                if cap and len(edges)>=cap:
                    out = flush_current()
                    if out and out[2]:
                        yield out
                        windows_yielded += 1
                        if windows_yielded >= max_windows:
                            return
                    current_ws += stride_ns; current_we = current_ws + win_ns
    out = flush_current()
    if out and out[2]:
        yield out

# ---------------- worker: process one file ----------------
def process_file_worker(
    file_path: str,
    base_dir: str,
    out_dir: str,
    gt_ivs: List[Tuple[int,int]],
    win_ns: int,
    stride_ns: int,
    max_windows: int,
    need_label: int    # 1 for mal, 0 for benign
):
    """
    Returns: list of rows (dict) for index.csv and profile.csv.
    Worker writes tensors to disk; parent writes CSVs.
    """
    t_file_start = time.perf_counter()
    out_rows = []
    file = Path(file_path)
    base = Path(base_dir)
    out_dir = Path(out_dir)

    for (ws,we,edges,node_uuids,min_ts,max_ts) in collect_successive_windows_from_file(
        file, win_ns, stride_ns, EDGES_PER_WINDOW, max_windows
    ):
        label = 1 if overlaps(gt_ivs, ws, we) else 0
        if label != need_label:
            # reset file timer and continue
            t_file_start = time.perf_counter()
            continue

        t0 = time.perf_counter()
        # UUID typing from parquet folders (per window)
        uuid_map = node_types_from_parquet(base, node_uuids)
        t_uuid = time.perf_counter()

        x, edge_index, edge_attr, meta = build_tensors(edges, uuid_map, min_ts, max_ts)
        t_tensor = time.perf_counter()

        name = f"win_{ns_to_iso_safe(ws)}__{ns_to_iso_safe(we)}__{('mal' if label==1 else 'ben')}_{file.stem}.pt"
        out_pt = out_dir / name
        torch.save({"x":x,"edge_index":edge_index,"edge_attr":edge_attr,"meta":meta,"label":label}, str(out_pt))
        t_save = time.perf_counter()

        out_rows.append({
            "graph_file": name,
            "start_ns": ws,
            "end_ns": we,
            "label": label,
            "num_nodes": int(x.shape[0]),
            "num_edges": int(edge_index.shape[1]),
            "file_source": file.name,
            "read_parse_sec": round((t0 - t_file_start), 4),      # since last window/file start
            "uuid_typing_sec": round((t_uuid - t0), 4),
            "tensor_build_sec": round((t_tensor - t_uuid), 4),
            "save_sec": round((t_save - t_tensor), 4),
            "total_window_sec": round((t_save - t0), 4)
        })
        # reset file timer for next window
        t_file_start = time.perf_counter()

    return out_rows

# ------------------------------ main -------------------
def main():
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logging(out_dir)

    # Gather files
    events_dir = INPUT_DIR / "events"
    files = sorted(events_dir.glob("*.parquet"))
    if not files:
        print(f"No parquet in {events_dir}"); sys.exit(1)

    if SHUFFLE_FILES:
        random.shuffle(files)

    mal_files = [f for f in files if f.name.lower().startswith("malicious")]
    ben_files = [f for f in files if f.name.lower().startswith("non_malicious")]
    log.info(f"Files: malicious={len(mal_files)} non_malicious={len(ben_files)}")

    win_ns = WINDOW_MIN * NS_PER_MIN
    stride_ns = STRIDE_MIN * NS_PER_MIN
    gt_ivs = load_attack_intervals(GT_JSON, log)

    # choose subsets to submit to workers (limit overshoot)
    need_mal_files = int((TARGET_MAL + MAL_WINS_PER_FILE - 1) / MAL_WINS_PER_FILE)
    need_ben_files = int((TARGET_BENIGN + BEN_WINS_PER_FILE - 1) / BEN_WINS_PER_FILE)
    mal_submit = mal_files[: min(len(mal_files), int(need_mal_files*(1+SUBMIT_SLACK)) )]
    ben_submit = ben_files[: min(len(ben_files), int(need_ben_files*(1+SUBMIT_SLACK)) )]

    # CSV writers
    index_csv = (out_dir / "index.csv").open("w", newline="", encoding="utf-8")
    prof_csv  = (out_dir / "profile.csv").open("w", newline="", encoding="utf-8")
    index_wr = csv.writer(index_csv)
    prof_wr  = csv.writer(prof_csv)
    index_wr.writerow(["graph_file","start_ns","end_ns","label","num_nodes","num_edges","file_source"])
    prof_wr.writerow(["graph_file","read_parse_sec","uuid_typing_sec","tensor_build_sec","save_sec","total_window_sec"])

    written_mal = 0
    written_ben = 0
    sum_nodes = 0
    sum_edges = 0
    t0 = time.perf_counter()

    # Submit to process pool
    futures = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for f in mal_submit:
            futures.append(
                ex.submit(
                    process_file_worker,
                    str(f), str(INPUT_DIR), str(out_dir),
                    gt_ivs, win_ns, stride_ns, MAL_WINS_PER_FILE, 1
                )
            )
        for f in ben_submit:
            futures.append(
                ex.submit(
                    process_file_worker,
                    str(f), str(INPUT_DIR), str(out_dir),
                    gt_ivs, win_ns, stride_ns, BEN_WINS_PER_FILE, 0
                )
            )

        pbar = tqdm(total=(TARGET_MAL + TARGET_BENIGN), desc="Windows", unit="win")
        for fut in as_completed(futures):
            try:
                rows = fut.result() or []
            except Exception as e:
                log.warning(f"Worker failed: {e}")
                rows = []

            # write rows, respecting targets
            for r in rows:
                if r["label"] == 1 and written_mal >= TARGET_MAL:
                    continue
                if r["label"] == 0 and written_ben >= TARGET_BENIGN:
                    continue
                index_wr.writerow([r["graph_file"], r["start_ns"], r["end_ns"], r["label"],
                                   r["num_nodes"], r["num_edges"], r["file_source"]])
                prof_wr.writerow([r["graph_file"], r["read_parse_sec"], r["uuid_typing_sec"],
                                  r["tensor_build_sec"], r["save_sec"], r["total_window_sec"]])
                sum_nodes += r["num_nodes"]; sum_edges += r["num_edges"]
                if r["label"] == 1:
                    written_mal += 1
                else:
                    written_ben += 1
                pbar.update(1)
                if written_mal >= TARGET_MAL and written_ben >= TARGET_BENIGN:
                    break
            if written_mal >= TARGET_MAL and written_ben >= TARGET_BENIGN:
                break
        pbar.close()

    index_csv.close(); prof_csv.close()

    total_elapsed = time.perf_counter() - t0
    total_written = written_mal + written_ben
    summary = (f"SUMMARY: windows={total_written} (mal={written_mal}, ben={written_ben}) "
               f"| total_nodes={sum_nodes} total_edges={sum_edges} "
               f"| total_time={total_elapsed:.2f}s | out={OUTPUT_DIR}")
    print("\n=== FINAL SUMMARY ===")
    print(f"Windows: {total_written}  (malicious={written_mal}, benign={written_ben})")
    print(f"Total nodes (sum over windows): {sum_nodes}")
    print(f"Total edges (sum over windows): {sum_edges}")
    print(f"Total time: {total_elapsed:.2f} sec")
    print(f"Index: {OUTPUT_DIR/'index.csv'} | Profile: {OUTPUT_DIR/'profile.csv'}")
    log = setup_logging(OUTPUT_DIR)
    log.info(summary)

if __name__ == "__main__":
    main()
