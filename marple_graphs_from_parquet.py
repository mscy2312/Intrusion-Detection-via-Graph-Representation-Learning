#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build 3-minute provenance graphs (malicious + benign) from Marple Parquet.

Key specs:
- Windows are CLOSED intervals: [start_ns, end_ns] (inclusive).
- Malicious windows parsed from marple_attack_window.json with structure:
    {"attacks_by_host": {"MARPLE 1": [ {id, start_ns, end_ns, ...}, ... ], ...}}
  (The script collects all hosts' windows.)
- Benign windows are all non-overlapping closed 3-minute windows outside attacks,
  then capped to MAX_BENIGN via time-stratified sampling.
- No c_min threshold (keep single-occurrence edges).
- Per-node cap = 54 (one kept edge per (node, direction, event_type)).
- No coarsening.
- Node features:
  * Subject: has_parent, start_pos (within window), cmdLine hash sketch (32), cid_mod8
  * FileObject: file_type categorical only
  * Netflow: original columns preserved in metadata; numeric encodings used in tensors.

Multiprocessing (>=4 workers). CUDA used for tensor allocation if available (preprocessing is CPU-bound).
Run: python "C:\\Users\\Ali\\Desktop\\ChatGPT Scripts\\marple_graphs_from_parquet.py"
"""

import os
import sys
import time
import json
import hashlib
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union

import numpy as np
import pandas as pd
import torch

# ---------- Logging ----------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("builder")

# tqdm (progress bars)
try:
    from tqdm import tqdm
except Exception:
    class tqdm:  # fallback no-op
        def __init__(self, it=None, total=None, **kwargs): self.it, self.total = it, total
        def __iter__(self): return iter(self.it) if self.it is not None else iter(())
        def update(self, n=1): pass
        def set_postfix(self, **kw): pass
        def close(self): pass

# ---------- Hard-coded paths (YOUR MACHINE) ----------
BASE_DIR = r"C:\Users\Ali\Desktop\ChatGPT Scripts\Parquet\Marple_parquet"
EVENTS_DIR   = os.path.join(BASE_DIR, "events")
SUBJECTS_DIR = os.path.join(BASE_DIR, "subjects")
FILEOBJS_DIR = os.path.join(BASE_DIR, "fileobjects")
NETFLOWS_DIR = os.path.join(BASE_DIR, "netflows")
ATTACK_JSON  = r"C:\Users\Ali\Desktop\ChatGPT Scripts\marple_attack_window.json"
OUT_DIR      = r"C:\Users\Ali\Desktop\ChatGPT Scripts\Graphs_Out"

# ---------- Device & workers ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = max(4, os.cpu_count() // 2)  # at least 4 on your 8-core machine
log.info(f"Using device={DEVICE}, workers={NUM_WORKERS}")

# ---------- Constants & required columns ----------
WIN_NS = 3 * 60 * 1_000_000_000  # 3 minutes in ns (closed interval length)
MAX_BENIGN = 1000  # cap benign slices

REQ_EVENT_COLS = [
    "uuid", "event_type", "timestampNanos",
    "subject_uuid", "object1_uuid", "object2_uuid",
    "size", "malicious"
]
REQ_SUBJECT_COLS = [
    "uuid", "subject_type", "cid", "parentSubject", "localPrincipal",
    "startTimestampNanos", "cmdLine"
]
REQ_FILEOBJ_COLS = ["uuid", "file_type"]
REQ_NETFLOW_COLS = ["uuid", "localAddress", "localPort", "remoteAddress", "remotePort", "ipProtocol"]

DROP_SUBJECTS = ["sessionNumber", "source", "hostId", "privilegeLevel"]
DROP_FILEOBJS = ["sessionNumber", "source", "hostId", "size", "fileDescriptor"]
DROP_NETFLOWS = ["sessionNumber", "source", "hostId", "fileDescriptor"]
DROP_EVENTS   = ["sessionNumber", "source", "hostId"]

# ---------- PyArrow dataset (fast filtered reads) ----------
try:
    import pyarrow as pa
    import pyarrow.dataset as ds
except Exception:
    log.error("PyArrow is required. Install with `conda install pyarrow` or `pip install pyarrow`.")
    sys.exit(1)

def _dataset(path: str) -> ds.Dataset:
    return ds.dataset(path, format="parquet")

def _read_table_filtered(path: str, columns: List[str], filter_expr=None) -> pd.DataFrame:
    """Read a Parquet directory with projection + optional filter via PyArrow dataset."""
    dataset = _dataset(path)
    existing_cols = [c for c in columns if c in dataset.schema.names]
    scanner = ds.Scanner.from_dataset(dataset, columns=existing_cols or None, filter=filter_expr)
    table = scanner.to_table()
    # IMPORTANT: use default pandas dtypes (no ArrowDtype) to avoid Arrow fillna/combine issues
    df = table.to_pandas()
    # Drop forbidden columns if any slipped in
    if path == EVENTS_DIR:
        for c in DROP_EVENTS:
            if c in df.columns: df.drop(columns=c, inplace=True)
    elif path == SUBJECTS_DIR:
        for c in DROP_SUBJECTS:
            if c in df.columns: df.drop(columns=c, inplace=True)
    elif path == FILEOBJS_DIR:
        for c in DROP_FILEOBJS:
            if c in df.columns: df.drop(columns=c, inplace=True)
    elif path == NETFLOWS_DIR:
        for c in DROP_NETFLOWS:
            if c in df.columns: df.drop(columns=c, inplace=True)
    return df

# ---------- Robust attack window parsing for your JSON ----------
def _to_ns(value: Union[str, int, float]) -> int:
    """Convert various time forms to ns."""
    if value is None:
        raise ValueError("timestamp value is None")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        v = float(value)
        return int(v if v > 1e12 else v * 1e9)
    if isinstance(value, str):
        s = value.strip()
        if s.isdigit():
            return int(s)
        # ISO8601
        ts = pd.to_datetime(s, utc=True, errors="raise")
        return int(ts.value)  # ns
    raise ValueError(f"unsupported timestamp type: {type(value)}")

def load_attack_windows(json_path: str) -> List[Tuple[int, int, str]]:
    """
    Parses:
      {
        "attacks_by_host": {
          "MARPLE 1": [ {id, start_ns, end_ns, ...}, ... ],
          "MARPLE 2": [ ... ]
        }
      }
    Returns sorted list of (start_ns, end_ns, id_str) as CLOSED intervals.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    wins: List[Tuple[int,int,str]] = []
    if isinstance(data, dict) and "attacks_by_host" in data and isinstance(data["attacks_by_host"], dict):
        for host, arr in data["attacks_by_host"].items():
            if not isinstance(arr, list): 
                continue
            for i, d in enumerate(arr):
                try:
                    s = _to_ns(d.get("start_ns") or d.get("start") or d.get("start_iso") or d.get("startTimestampNanos"))
                    e = _to_ns(d.get("end_ns")   or d.get("end")   or d.get("end_iso")   or d.get("endTimestampNanos"))
                    if s > e: s, e = e, s
                    wid = str(d.get("id", f"{host}_attack_{i}"))
                    wins.append((int(s), int(e), wid))
                except Exception as ex:
                    log.error(f"[windows] skip {host}[{i}]: {ex}")
    else:
        raise ValueError("Unexpected JSON structure: 'attacks_by_host' not found.")

    if not wins:
        raise ValueError("No valid attack windows parsed from JSON")

    wins.sort(key=lambda x: x[0])
    log.info(f"Parsed {len(wins)} attack windows from JSON (all hosts).")
    return wins

# ---------- Windows math (CLOSED intervals) ----------
def unionize(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge possibly overlapping CLOSED intervals."""
    if not intervals: return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le + 1:  # closed intervals: overlap if s <= le
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged

def complement_windows(min_ts: int, max_ts: int, attacks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Produce all CLOSED 3-minute windows [cur, cur+WIN_NS] outside CLOSED attack intervals.
    Windows are back-to-back across the global range.
    """
    attacks_u = unionize(attacks)
    res = []
    cur = min_ts
    idx = 0
    while cur + WIN_NS <= max_ts:  # ensure window end <= max_ts (closed)
        while idx < len(attacks_u) and attacks_u[idx][1] < cur:
            idx += 1
        if idx < len(attacks_u):
            a_s, a_e = attacks_u[idx]
            # overlap test for closed intervals:
            if not (cur + WIN_NS < a_s or a_e < cur):
                cur = a_e + 1  # jump past inclusive end
                continue
        res.append((cur, cur + WIN_NS))
        cur += WIN_NS + 1  # advance by closed length
    return res

def cap_benign_windows(ben_wins: List[Tuple[int,int]], k: int) -> List[Tuple[int,int]]:
    """
    Time-stratified deterministic sampling: pick k evenly spaced windows by index.
    Keeps chronological coverage and reproducibility (no RNG needed).
    """
    n = len(ben_wins)
    if n <= k:
        return ben_wins
    idxs = np.linspace(0, n - 1, num=k, dtype=int)
    return [ben_wins[i] for i in idxs]

# ---------- Utilities ----------
def hash_token_sketch(text: str, bins: int = 32) -> np.ndarray:
    """Stable hashing-based bag-of-words sketch (cmdLine tokens)."""
    if not isinstance(text, str) or not text:
        return np.zeros(bins, dtype=np.float32)
    import re
    arr = np.zeros(bins, dtype=np.float32)
    for tok in re.split(r"[^a-zA-Z0-9_]+", text.lower()):
        if len(tok) < 2:
            continue
        h = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8, person=b"cmd").hexdigest(), 16)
        arr[h % bins] += 1.0
    return arr

def normalize_pos(val: Any, start_ns: int, end_ns: int) -> float:
    """Return position in [0,1] if val ∈ [start_ns, end_ns] (closed), else 0.0."""
    if not isinstance(val, (int, np.integer, np.int64)): return 0.0
    if val < start_ns or val > end_ns: return 0.0
    span = max(end_ns - start_ns, 1)
    return float((val - start_ns) / span)

def cat_encode(series: pd.Series) -> Tuple[np.ndarray, Dict[Any, int]]:
    cats = {v: i+1 for i, v in enumerate(sorted(series.dropna().unique()))}
    return series.map(lambda v: cats.get(v, 0)).astype(np.int64).values, cats

def encode_ip_like(series: pd.Series) -> np.ndarray:
    """Numeric encoding for IP/hostname (keeps raw in metadata)."""
    def to_u32(s: Any) -> int:
        if not isinstance(s, str): return 0
        try:
            parts = s.split(".")
            if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
                return (int(parts[0])<<24) | (int(parts[1])<<16) | (int(parts[2])<<8) | int(parts[3])
            return int(hashlib.blake2b(s.encode(), digest_size=4, person=b"ip").hexdigest(), 16)
        except Exception:
            return 0
    return series.map(to_u32).astype(np.uint32).values

# ---------- Edge building ----------
def dedup_edges(ev_win: pd.DataFrame, etype_to_id: Dict[str, int]) -> pd.DataFrame:
    # Robust coalesce object2_uuid with object1_uuid without Arrow casting
    a = ev_win["object2_uuid"].astype("object")
    b = ev_win["object1_uuid"].astype("object")
    obj_uuid = a.where(pd.notna(a), b)

    tmp = ev_win.assign(object_uuid=obj_uuid.astype("object"))
    ag = (tmp.groupby(["subject_uuid", "object_uuid", "event_type"], dropna=False)
            .agg(count=("uuid", "size"),
                 first_ts=("timestampNanos", "min"),
                 last_ts=("timestampNanos", "max"),
                 size_sum=("size", "sum"),
                 size_max=("size", "max"))
            .reset_index())
    ag["edge_type_id"] = ag["event_type"].map(lambda x: etype_to_id.get(x, 0)).astype(np.int64)
    ag = ag.rename(columns={"subject_uuid":"src","object_uuid":"dst"})
    ag["size_sum"] = ag["size_sum"].fillna(0.0)
    ag["size_max"] = ag["size_max"].fillna(0.0)
    return ag

def score_edges(df: pd.DataFrame) -> pd.Series:
    span = (df["last_ts"].astype(np.int64) - df["first_ts"].astype(np.int64)).clip(lower=1)
    return (
        0.6 * np.log1p(df["count"].astype(np.float64)) +
        0.3 * np.log1p(df["size_sum"].astype(np.float64)) +
        0.1 * (df["count"].astype(np.float64) / span.astype(np.float64))
    )

def enforce_per_node_cap_54(ag: pd.DataFrame) -> pd.DataFrame:
    """Keep at most one edge per (node, direction, event_type)."""
    if ag.empty: return ag.copy()
    ag = ag.copy()
    ag["score"] = score_edges(ag)
    keep_idx = set()
    # Outgoing: for each (src, edge_type) keep best dst
    for (_, _), sub in ag.groupby(["src", "edge_type_id"], sort=False, dropna=False):
        keep_idx.add(sub["score"].idxmax())
    # Incoming: for each (dst, edge_type) keep best src
    for (_, _), sub in ag.groupby(["dst", "edge_type_id"], sort=False, dropna=False):
        keep_idx.add(sub["score"].idxmax())
    kept = ag.loc[list(keep_idx)].reset_index(drop=True)
    return kept

# ---------- Node feature builders ----------
def build_subject_features(df: pd.DataFrame, win_start: int, win_end: int) -> np.ndarray:
    if df.empty:
        return np.zeros((0, 1+1+32+1), dtype=np.float32)
    has_parent = (~df["parentSubject"].isna()).astype(np.float32).values.reshape(-1,1)
    start_pos = df["startTimestampNanos"].apply(
        lambda v: normalize_pos(int(v) if pd.notna(v) else -1, win_start, win_end)
    ).astype(np.float32).values.reshape(-1,1)
    cid_mod8 = (df["cid"].fillna(0).astype(np.int64) % 8).astype(np.float32).values.reshape(-1,1)
    sketches = np.vstack([hash_token_sketch(s, bins=32) for s in df["cmdLine"].fillna("")])
    feats = np.hstack([has_parent, start_pos, sketches, cid_mod8]).astype(np.float32)
    return feats

def build_file_features(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    if df.empty:
        return np.zeros((0, 1), dtype=np.float32), {"file_type_vocab": {}}
    codes, vocab = cat_encode(df["file_type"].fillna("UNK"))
    feats = codes.reshape(-1,1).astype(np.float32)
    return feats, {"file_type_vocab": vocab}

def build_netflow_features(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
    if df.empty:
        return np.zeros((0, 5), dtype=np.float32), {}
    local_addr_num = encode_ip_like(df["localAddress"].fillna(""))
    remote_addr_num = encode_ip_like(df["remoteAddress"].fillna(""))
    local_port = df["localPort"].fillna(0).astype(np.int64).values
    remote_port = df["remotePort"].fillna(0).astype(np.int64).values
    ipproto = df["ipProtocol"].fillna(0).astype(np.int64).values
    feats = np.vstack([local_addr_num, local_port, remote_addr_num, remote_port, ipproto]).T.astype(np.float32)
    meta_cols = df[["localAddress","localPort","remoteAddress","remotePort","ipProtocol"]].copy()
    return feats, {"netflow_raw": meta_cols}

# ---------- Graph packing ----------
def pack_graph(
    kept_edges: pd.DataFrame,
    subjects_map: pd.DataFrame,
    files_map: pd.DataFrame,
    nets_map: pd.DataFrame,
    win_id: str, win_start: int, win_end: int,
    label: int,
    device: torch.device,
) -> Dict[str, Any]:
    subj_ids = subjects_map["uuid"].astype(str).tolist()
    file_ids = files_map["uuid"].astype(str).tolist()
    net_ids  = nets_map["uuid"].astype(str).tolist()

    uuid_to_local_s = {u:i for i,u in enumerate(subj_ids)}
    uuid_to_local_f = {u:i for i,u in enumerate(file_ids)}
    uuid_to_local_n = {u:i for i,u in enumerate(net_ids)}

    def sel(df: pd.DataFrame, src_type: str, dst_type: str) -> pd.DataFrame:
        if src_type == "subject": src_set = set(subj_ids)
        elif src_type == "fileobject": src_set = set(file_ids)
        else: src_set = set(net_ids)
        if dst_type == "subject": dst_set = set(subj_ids)
        elif dst_type == "fileobject": dst_set = set(file_ids)
        else: dst_set = set(net_ids)
        return df[df["src"].isin(src_set) & df["dst"].isin(dst_set)]

    edges_sf = sel(kept_edges, "subject", "fileobject")
    edges_sn = sel(kept_edges, "subject", "netflow")
    edges_ss = sel(kept_edges, "subject", "subject")
    edges_fs = sel(kept_edges, "fileobject", "subject")
    edges_ns = sel(kept_edges, "netflow", "subject")

    def idx_map(df, src_map, dst_map):
        if df.empty: return np.zeros((2,0), dtype=np.int64)
        s = df["src"].map(src_map).astype(np.int64).values
        d = df["dst"].map(dst_map).astype(np.int64).values
        return np.vstack([s,d])

    eidx_sf = idx_map(edges_sf, uuid_to_local_s, uuid_to_local_f)
    eidx_sn = idx_map(edges_sn, uuid_to_local_s, uuid_to_local_n)
    eidx_ss = idx_map(edges_ss, uuid_to_local_s, uuid_to_local_s)
    eidx_fs = idx_map(edges_fs, uuid_to_local_f, uuid_to_local_s)
    eidx_ns = idx_map(edges_ns, uuid_to_local_n, uuid_to_local_s)

    def edge_feats(df):
        if df.empty:
            return np.zeros((0,6), dtype=np.float32)
        pos_first = np.clip((df["first_ts"].values - win_start) / max(win_end - win_start, 1), 0.0, 1.0)
        span_norm = np.clip((df["last_ts"].values - df["first_ts"].values) / max(win_end - win_start, 1), 0.0, 1.0)
        out = np.stack([
            df["edge_type_id"].values.astype(np.int64),
            df["count"].values.astype(np.int64),
            span_norm.astype(np.float32),
            pos_first.astype(np.float32),
            np.log1p(df["size_sum"].values.astype(np.float64)).astype(np.float32),
            np.log1p(df["size_max"].values.astype(np.float64)).astype(np.float32)
        ], axis=1)
        return out.astype(np.float32)

    eattr_sf = edge_feats(edges_sf)
    eattr_sn = edge_feats(edges_sn)
    eattr_ss = edge_feats(edges_ss)
    eattr_fs = edge_feats(edges_fs)
    eattr_ns = edge_feats(edges_ns)

    # Node features
    x_subj = build_subject_features(subjects_map, win_start, win_end)
    x_file, meta_file = build_file_features(files_map)
    x_net,  meta_net  = build_netflow_features(nets_map)

    # Degrees (post-trim)
    def add_degrees(n_nodes, incoming_lists, outgoing_lists):
        deg_in = np.zeros((n_nodes,1), dtype=np.float32)
        deg_out= np.zeros((n_nodes,1), dtype=np.float32)
        for eidx in incoming_lists:
            if eidx.shape[1]:
                counts = np.bincount(eidx[1], minlength=n_nodes)
                deg_in[:,0] += counts.astype(np.float32)
        for eidx in outgoing_lists:
            if eidx.shape[1]:
                counts = np.bincount(eidx[0], minlength=n_nodes)
                deg_out[:,0] += counts.astype(np.float32)
        return np.hstack([deg_in, deg_out, (deg_in+deg_out)])

    x_subj = np.hstack([x_subj, add_degrees(len(subj_ids), [eidx_fs,eidx_ns,eidx_ss], [eidx_sf,eidx_sn,eidx_ss])])
    x_file = np.hstack([x_file, add_degrees(len(file_ids), [eidx_sf], [eidx_fs])])
    x_net  = np.hstack([x_net,  add_degrees(len(net_ids),  [eidx_sn], [eidx_ns])])

    to_t = lambda x: torch.as_tensor(x, device=DEVICE)
    graph = {
        "x": {
            "subject":    to_t(x_subj),
            "fileobject": to_t(x_file),
            "netflow":    to_t(x_net),
        },
        "edge_index": {
            ("subject","events","fileobject"): to_t(eidx_sf),
            ("subject","events","netflow"):    to_t(eidx_sn),
            ("subject","events","subject"):    to_t(eidx_ss),
            ("fileobject","events","subject"): to_t(eidx_fs),
            ("netflow","events","subject"):    to_t(eidx_ns),
        },
        "edge_attr": {
            ("subject","events","fileobject"): to_t(eattr_sf),
            ("subject","events","netflow"):    to_t(eattr_sn),
            ("subject","events","subject"):    to_t(eattr_ss),
            ("fileobject","events","subject"): to_t(eattr_fs),
            ("netflow","events","subject"):    to_t(eattr_ns),
        },
        "y": to_t(np.array([label], dtype=np.int64)),
        "meta": {
            "window_id": win_id,
            "start_ns": int(win_start),
            "end_ns": int(win_end),
            "counts": {
                "N_subject": len(subj_ids),
                "N_fileobject": len(file_ids),
                "N_netflow": len(net_ids),
                "E_sf": eidx_sf.shape[1], "E_sn": eidx_sn.shape[1], "E_ss": eidx_ss.shape[1],
                "E_fs": eidx_fs.shape[1], "E_ns": eidx_ns.shape[1],
            },
            "subjects_raw": subjects_map[["uuid","subject_type","cid","parentSubject","localPrincipal","startTimestampNanos","cmdLine"]].astype(object).to_dict(orient="records"),
            "file_raw":     files_map[["uuid","file_type"]].astype(object).to_dict(orient="records"),
            "netflow_raw":  meta_net.get("netflow_raw", pd.DataFrame()).astype(object).to_dict(orient="records") if meta_net else [],
        }
    }
    return graph

# ---------- Event types (global mapping) ----------
def build_event_type_map() -> Dict[str, int]:
    dataset = _dataset(EVENTS_DIR)
    scanner = ds.Scanner.from_dataset(dataset, columns=["event_type"])
    etypes = set()
    log.info("[*] Scanning event types...")
    processed = 0
    for batch in scanner.to_reader():
        tb = pa.Table.from_batches([batch]).to_pandas()
        etypes.update(tb["event_type"].dropna().unique().tolist())
        processed += len(tb)
        if processed and processed % 500_000 == 0:
            log.info(f"  scanned ~{processed:,} rows for event types...")
    etypes = sorted(etypes)
    log.info(f"  unique event types: {len(etypes)}")
    return {e:i for i,e in enumerate(etypes)}

# ---------- Worker: process one window ----------
def process_one_window(win_tuple: Tuple[int,int,str], label: int, etype_to_id: Dict[str,int], out_dir: str) -> Dict[str, Any]:
    start_ns, end_ns, win_id = win_tuple
    dbg = log.isEnabledFor(logging.DEBUG)
    if dbg:
        log.debug(f"[{win_id}] start processing")

    # CLOSED interval filter: timestampNanos ∈ [start_ns, end_ns]
    filt = (ds.field("timestampNanos") >= pa.scalar(start_ns)) & (ds.field("timestampNanos") <= pa.scalar(end_ns))
    ev = _read_table_filtered(EVENTS_DIR, REQ_EVENT_COLS, filter_expr=filt)
    if ev.empty:
        if dbg: log.debug(f"[{win_id}] no events")
        return {"window_id": win_id, "label": label, "skipped": True, "reason": "no_events"}

    # Ensure types
    ev["timestampNanos"] = ev["timestampNanos"].astype("int64")
    ev["size"] = ev["size"].fillna(0).astype("float64")

    # Dedup and per-node cap
    ag = dedup_edges(ev, etype_to_id)
    kept = enforce_per_node_cap_54(ag)
    if kept.empty:
        if dbg: log.debug(f"[{win_id}] no kept edges after cap")
        return {"window_id": win_id, "label": label, "skipped": True, "reason": "no_kept_edges"}

    # Node UUIDs referenced
    uuids = set(kept["src"].dropna().astype(str)) | set(kept["dst"].dropna().astype(str))
    if not uuids:
        if dbg: log.debug(f"[{win_id}] no nodes")
        return {"window_id": win_id, "label": label, "skipped": True, "reason": "no_nodes"}

    # Read subjects/fileobjects/netflows rows for these UUIDs via IN filter
    uuid_list = list(uuids)
    in_filter = ds.field("uuid").isin(uuid_list)

    subjects = _read_table_filtered(SUBJECTS_DIR, REQ_SUBJECT_COLS, filter_expr=in_filter)
    files    = _read_table_filtered(FILEOBJS_DIR, REQ_FILEOBJ_COLS, filter_expr=in_filter)
    netflows = _read_table_filtered(NETFLOWS_DIR, REQ_NETFLOW_COLS, filter_expr=in_filter)

    # Pack and save
    device = torch.device(DEVICE)
    graph = pack_graph(kept, subjects, files, netflows, win_id, start_ns, end_ns, label, device)
    out_path = Path(out_dir) / f"{'mal' if label==1 else 'ben'}__{win_id}.pt"
    torch.save(graph, out_path)

    counts = graph["meta"]["counts"]
    if dbg:
        log.debug(f"[{win_id}] done: Nsub={counts['N_subject']} Nfile={counts['N_fileobject']} Nnet={counts['N_netflow']} Etotal={sum(counts[k] for k in ['E_sf','E_sn','E_ss','E_fs','E_ns'])}")
    return {
        "window_id": win_id,
        "label": label,
        "N_subject": counts["N_subject"],
        "N_fileobject": counts["N_fileobject"],
        "N_netflow": counts["N_netflow"],
        "E_total": sum(counts[k] for k in ["E_sf","E_sn","E_ss","E_fs","E_ns"]),
        "path": str(out_path),
        "skipped": False
    }

# --- top-level wrapper for Windows pickling ---
def _proc_wrapper_star(args):
    # args is (win_tuple, label, etype_to_id, out_dir)
    return process_one_window(*args)

# ---------- Main ----------
def main():
    t0 = time.time()
    out_dir = Path(OUT_DIR); out_dir.mkdir(parents=True, exist_ok=True)

    # Build global event-type map once
    etype_to_id = build_event_type_map()

    # Determine global time bounds from events (batch scan)
    ds_events = _dataset(EVENTS_DIR)
    scanner_ts = ds.Scanner.from_dataset(ds_events, columns=["timestampNanos"])
    tmin, tmax, processed = None, None, 0
    log.info("[*] Scanning time bounds...")
    for batch in scanner_ts.to_reader():
        tbl = pa.Table.from_batches([batch])
        ser = tbl.column(0).to_pandas()
        if ser.empty:
            continue
        ser = ser.astype("int64")
        bmin = int(ser.min(skipna=True))
        bmax = int(ser.max(skipna=True))
        tmin = bmin if tmin is None else min(tmin, bmin)
        tmax = bmax if tmax is None else max(tmax, bmax)
        processed += len(ser)
        if processed and processed % 1_000_000 == 0:
            log.info(f"  scanned ~{processed:,} rows for time bounds...")
    if tmin is None or tmax is None:
        log.error("Could not determine time range from events.")
        sys.exit(1)
    log.info(f"  time range: [{tmin}, {tmax}] ns (closed)")

    # Load attack windows (robust for your JSON)
    log.info("[*] Loading attack windows...")
    try:
        att_wins = load_attack_windows(ATTACK_JSON)
    except Exception as e:
        log.error(f"Failed to parse attack windows: {e}")
        sys.exit(1)

    # Build benign windows outside closed attack intervals + cap
    full_ben_wins = complement_windows(tmin, tmax, [(s,e) for s,e,_ in att_wins])
    ben_wins = cap_benign_windows(full_ben_wins, MAX_BENIGN)
    log.info(f"[*] Windows → attack: {len(att_wins):,} | benign(total): {len(full_ben_wins):,} | benign(capped): {len(ben_wins):,}")

    # Build task list (keep args small for Windows multiprocessing)
    tasks = []
    for s,e,wid in att_wins:
        tasks.append(((s,e,wid), 1, etype_to_id, str(out_dir)))
    for i,(s,e) in enumerate(ben_wins):
        wid = f"benign_{i:06d}"
        tasks.append(((s,e,wid), 0, etype_to_id, str(out_dir)))

    # Parallel processing with progress bar
    from multiprocessing import Pool

    log.info("[*] Processing windows in parallel...")
    results = []
    with Pool(processes=NUM_WORKERS) as pool:
        for res in tqdm(pool.imap_unordered(_proc_wrapper_star, tasks, chunksize=1),
                        total=len(tasks), desc="Windows"):
            results.append(res)

    # Summary CSV
    summary = pd.DataFrame(results)
    summary_path = Path(out_dir) / "summary.csv"
    summary.to_csv(summary_path, index=False)

    dt = time.time() - t0
    kept = summary[~summary["skipped"]]
    log.info(f"[*] Done in {dt:.1f}s. Saved {kept.shape[0]} graphs → {out_dir}")
    log.info(f"[*] Summary: {summary_path}")
    try:
        n_mal = int((summary["label"]==1).sum())
        n_ben = int((summary["label"]==0).sum())
        log.info(f"    Breakdown: mal={n_mal:,} | ben={n_ben:,} | skipped={int(summary['skipped'].sum()):,}")
        if not kept.empty:
            log.info(f"    Example saved: {kept.iloc[0]['path']}")
    except Exception:
        pass

if __name__ == "__main__":
    # On Windows, protect entry point for multiprocessing
    main()
