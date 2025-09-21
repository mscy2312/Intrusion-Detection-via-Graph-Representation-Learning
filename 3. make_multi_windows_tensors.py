#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
make_multi_windows_tensors_safe.py

FAST & BALANCED with PROGRESS (Windows-safe filenames):
- Builds many 10-min window tensors (.pt) directly from Parquet (no NetworkX, no global UUID map)
- Balanced targets (e.g., 100 malicious + 200 benign)
- UUID -> node_type mapping (lazy per window; subjects/fileobjects/netflows)
- Shard-local by default (fast: one file per window)
- Progress bars: overall (windows) + per-window (edges)
- Per-window stats and final summary
- **Windows-safe filenames** (no ":"), e.g. win_2019-05-14T13-30-00Z__2019-05-14T13-40-00Z__i0.pt
"""

import argparse, logging, json, csv, random, time
from pathlib import Path
from datetime import datetime, timezone

import torch
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc
from tqdm import tqdm

LOG_LEVEL = logging.INFO
BATCH_SIZE = 100_000
EVENT_COLS = ["timestampNanos","event_type","subject_uuid","object1_uuid","object2_uuid"]

NS_PER_SEC = 1_000_000_000
NS_PER_MIN = 60 * NS_PER_SEC

logging.basicConfig(format='[%(levelname)s %(asctime)s] %(message)s',
                    datefmt='%H:%M:%S', level=LOG_LEVEL)
log = logging.getLogger("win-builder")

# ----- time helpers -----
def ns_to_iso(ns:int)->str:
    # ISO for logs
    return datetime.fromtimestamp(ns/1e9, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def ns_to_iso_safe(ns:int)->str:
    # Windows-safe for filenames (replace ":" with "-")
    return datetime.fromtimestamp(ns/1e9, tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")

# ----- GT intervals -----
def load_attack_intervals(path: Path):
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
    return [(s,e) for s,e in out]

def overlaps(ivs, s, e):
    import bisect
    if not ivs: return False
    starts=[x[0] for x in ivs]
    i=bisect.bisect_right(starts, s)-1
    for j in (i,i+1):
        if 0<=j<len(ivs):
            ss,ee=ivs[j]
            if not (e<=ss or s>=ee): return True
    return False

# ----- UUID -> node_type (lazy per window) -----
def lazy_node_types(base: Path, node_uuids:set):
    if not node_uuids:
        return {}
    uuid2type={u:None for u in node_uuids}
    remaining=set(node_uuids)
    for folder in ["subjects","fileobjects","netflows"]:
        p=base/folder
        if not p.exists(): continue
        dataset=ds.dataset(p, format="parquet")
        filt=pc.field("uuid").isin(list(remaining))
        try:
            table=dataset.to_table(filter=filt, columns=["uuid"])
        except Exception as e:
            log.warning(f"Node typing read error in {folder}: {e}")
            continue
        found=set()
        for r in table.to_pylist():
            u=r.get("uuid")
            if u in remaining:
                uuid2type[u]=folder
                found.add(u)
        remaining-=found
        if not remaining: break
    for u in node_uuids:
        if uuid2type[u] is None: uuid2type[u]="unknown"
    return uuid2type

# ----- tensors -----
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
    import torch
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

# ----- collectors -----
def collect_window_from_file(file:Path, win_ns:int, edges_cap:int|None, pbar_edges:tqdm|None=None):
    pf=pq.ParquetFile(file)
    schema=set(pf.schema_arrow.names)
    cols=[c for c in EVENT_COLS if c in schema]
    if not {"timestampNanos","subject_uuid"}.issubset(set(cols)):
        return None
    window_start=None; window_end=None
    edges=[]; node_uuids=set(); etypes=set(); min_ts=None; max_ts=None
    for batch in pf.iter_batches(batch_size=BATCH_SIZE, columns=cols):
        for r in batch.to_pylist():
            ts=r.get("timestampNanos")
            if ts is None: continue
            ts=int(ts)
            if window_start is None:
                window_start=(ts//win_ns)*win_ns
                window_end=window_start+win_ns
            if ts<window_start: continue
            if ts>=window_end:
                return window_start, window_end, edges, node_uuids, etypes, min_ts, max_ts
            u=r.get("subject_uuid"); v1=r.get("object1_uuid"); v2=r.get("object2_uuid"); et=r.get("event_type")
            min_ts=ts if min_ts is None else min(min_ts,ts)
            max_ts=ts if max_ts is None else max(max_ts,ts)
            if u and v1:
                edges.append((u,v1,et,ts)); node_uuids.add(u); node_uuids.add(v1)
                if et is not None: etypes.add(et)
                if pbar_edges is not None: pbar_edges.update(1)
            if edges_cap and len(edges)>=edges_cap:
                return window_start, window_end, edges, node_uuids, etypes, min_ts, max_ts
            if u and v2:
                edges.append((u,v2,et,ts)); node_uuids.add(u); node_uuids.add(v2)
                if et is not None: etypes.add(et)
                if pbar_edges is not None: pbar_edges.update(1)
            if edges_cap and len(edges)>=edges_cap:
                return window_start, window_end, edges, node_uuids, etypes, min_ts, max_ts
    if window_start is None:
        return None
    return window_start, window_end, edges, node_uuids, etypes, min_ts, max_ts

# ----- main -----
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True, help=r"Base parquet dir (e.g., .\out_parquet_time_only)")
    ap.add_argument("--out-dir", required=True, help=r"Output directory for .pt tensors")
    ap.add_argument("--gt-json", required=True, help="Ground truth JSON for labeling (gt_overlap)")
    ap.add_argument("--target-mal", type=int, default=100, help="Target number of malicious windows")
    ap.add_argument("--target-benign", type=int, default=200, help="Target number of benign windows")
    ap.add_argument("--edges-per-window", type=int, default=100000, help="Cap edges per window (0=unlimited)")
    ap.add_argument("--window-min", type=int, default=10, help="Window size minutes")
    ap.add_argument("--shuffle-files", action="store_true", help="Shuffle file order for sampling")
    args=ap.parse_args()

    t0 = time.perf_counter()

    base=Path(args.parquet_dir)
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    events_dir=base/"events"
    files=sorted(events_dir.glob("*.parquet"))
    if not files: raise FileNotFoundError(f"No parquet in {events_dir}")

    if args.shuffle_files:
        random.shuffle(files)

    gt_ivs=load_attack_intervals(Path(args.gt_json))
    win_ns=args.window_min*NS_PER_MIN
    edges_cap = None if args.edges_per_window<=0 else args.edges_per_window

    mal_files=[f for f in files if f.name.lower().startswith("malicious")]
    ben_files=[f for f in files if f.name.lower().startswith("non_malicious")]
    log.info(f"Files: malicious={len(mal_files)} non_malicious={len(ben_files)}")

    target_total = args.target_mal + args.target_benign
    windows_pbar = tqdm(total=target_total, desc="Windows", unit="win")

    index_path=out_dir/"index.csv"
    with open(index_path,"w",newline="",encoding="utf-8") as fidx:
        wr=csv.writer(fidx)
        wr.writerow(["graph_file","start_ns","end_ns","label","num_nodes","num_edges","file_source"])

        written_mal=0; written_ben=0; idx=0
        sum_nodes=0; sum_edges=0

        def save_window(ws,we,edges,node_uuids,min_ts,max_ts,label,idx,file_source):
            nonlocal sum_nodes, sum_edges
            node_types_map = lazy_node_types(base, node_uuids)
            x, edge_index, edge_attr, meta = build_tensors(edges, node_types_map, min_ts, max_ts)
            # Windows-safe filename (no ':')
            name = f"win_{ns_to_iso_safe(ws)}__{ns_to_iso_safe(we)}__i{idx}.pt"
            out_pt = out_dir / name
            torch.save({"x":x,"edge_index":edge_index,"edge_attr":edge_attr,"meta":meta,"label":label}, str(out_pt))
            wr.writerow([name, ws, we, label, x.shape[0], edge_index.shape[1], file_source])
            sum_nodes += x.shape[0]
            sum_edges += edge_index.shape[1]
            return name, x.shape[0], edge_index.shape[1]

        # --- Pass 1: malicious ---
        for file in mal_files:
            if written_mal >= args.target_mal:
                break
            pbar_edges = tqdm(total=(edges_cap or None), desc=f"Edges (mal) {file.name}", unit="e", leave=False)
            wstart = time.perf_counter()
            res = collect_window_from_file(file, win_ns, edges_cap, pbar_edges)
            pbar_edges.close()
            if res is None: continue
            ws,we,edges,node_uuids,etypes,min_ts,max_ts = res
            label = 1 if overlaps(gt_ivs, ws, we) else 0
            if label != 1: continue
            out_name, n_nodes, n_edges = save_window(ws,we,edges,node_uuids,min_ts,max_ts,label,idx,file.name)
            elapsed = time.perf_counter() - wstart
            log.info(f"[MAL] {out_name} | {ns_to_iso(ws)}..{ns_to_iso(we)} | nodes={n_nodes} edges={n_edges} | {elapsed:.2f}s")
            windows_pbar.update(1); written_mal += 1; idx += 1
            if written_mal >= args.target_mal: break

        # --- Pass 2: benign ---
        for file in ben_files:
            if written_ben >= args.target_benign:
                break
            pbar_edges = tqdm(total=(edges_cap or None), desc=f"Edges (ben) {file.name}", unit="e", leave=False)
            wstart = time.perf_counter()
            res = collect_window_from_file(file, win_ns, edges_cap, pbar_edges)
            pbar_edges.close()
            if res is None: continue
            ws,we,edges,node_uuids,etypes,min_ts,max_ts = res
            label = 1 if overlaps(gt_ivs, ws, we) else 0
            if label != 0: continue
            out_name, n_nodes, n_edges = save_window(ws,we,edges,node_uuids,min_ts,max_ts,label,idx,file.name)
            elapsed = time.perf_counter() - wstart
            log.info(f"[BENIGN] {out_name} | {ns_to_iso(ws)}..{ns_to_iso(we)} | nodes={n_nodes} edges={n_edges} | {elapsed:.2f}s")
            windows_pbar.update(1); written_ben += 1; idx += 1
            if written_ben >= args.target_benign: break

    windows_pbar.close()
    total_elapsed = time.perf_counter() - t0
    total_written = written_mal + written_ben
    log.info(f"SUMMARY: windows={total_written} (mal={written_mal}, ben={written_ben}) "
             f"| total_nodes={sum_nodes} total_edges={sum_edges} "
             f"| total_time={total_elapsed:.2f}s")
    print(f"\n=== FINAL SUMMARY ===")
    print(f"Windows: {total_written}  (malicious={written_mal}, benign={written_ben})")
    print(f"Total nodes (sum over windows): {sum_nodes}")
    print(f"Total edges (sum over windows): {sum_edges}")
    print(f"Total time: {total_elapsed:.2f} sec")
    print(f"Index: {index_path}")

if __name__=="__main__":
    main()
