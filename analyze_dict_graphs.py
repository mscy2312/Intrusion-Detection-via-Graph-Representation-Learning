#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time
import math
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter, defaultdict
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # allow duplicate OpenMP runtime
os.environ["OMP_NUM_THREADS"] = "1"          # optional: avoid oversubscription for plotting/stats

# ========= Hard-coded paths (your machine) =========
FOLDER = r"C:\Users\Ali\Desktop\ChatGPT Scripts\Graphs_Out"
PLOTS_DIR = os.path.join(FOLDER, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Our relation keys in saved graphs (directed, typed):
RELS = [
    ("subject","events","fileobject"),
    ("subject","events","netflow"),
    ("subject","events","subject"),
    ("fileobject","events","subject"),
    ("netflow","events","subject"),
]

# ========= Feature documentation (matches the builder) =========
FEATURE_DOC = {
    "node": {
        "subject": {
            "dim": 38,
            "layout": [
                ("has_parent", 1),
                ("start_pos_in_window", 1),
                ("cmdline_hash_sketch", 32),
                ("cid_mod8", 1),
                ("deg_in,deg_out,deg_tot", 3)
            ],
            "notes": "start_pos∈[0,1] if startTimestampNanos falls inside the window; degrees are computed after per-node cap trimming."
        },
        "fileobject": {
            "dim": 4,
            "layout": [
                ("file_type_code", 1),
                ("deg_in,deg_out,deg_tot", 3)
            ],
            "notes": "file_type_code is a categorical encoding; raw file_type values included in meta.file_raw."
        },
        "netflow": {
            "dim": 8,
            "layout": [
                ("localAddress_num, localPort, remoteAddress_num, remotePort, ipProtocol", 5),
                ("deg_in,deg_out,deg_tot", 3)
            ],
            "notes": "Addresses encoded numerically; raw values included in meta.netflow_raw."
        }
    },
    "edge": {
        "dim": 6,
        "layout": [
            ("edge_type_id", 1),
            ("count", 1),
            ("span_norm", 1),
            ("pos_first", 1),
            ("log1p_size_sum", 1),
            ("log1p_size_max", 1)
        ],
        "notes": "edge_type_id maps to event types, but names were not stored; per-node cap keeps at most one edge per (node, direction, event_type)."
    }
}

def describe_feature_layout():
    print("\n=== Feature schema ===")
    for nt, spec in FEATURE_DOC["node"].items():
        print(f"- Node[{nt}] dim={spec['dim']}: {', '.join([f'{n}({d})' for n,d in spec['layout']])}")
    e = FEATURE_DOC["edge"]
    print(f"- Edge dim={e['dim']}: {', '.join([f'{n}({d})' for n,d in e['layout']])}")
    print()

# ========= Helpers =========
def safe_len(x):
    try: return len(x)
    except: return 0

def tensor_nrows(t):
    return int(t.shape[0]) if hasattr(t, "shape") else 0

def edge_count_from_index(edge_index):
    # edge_index is shape [2, E]
    if edge_index is None: return 0
    return int(edge_index.shape[1]) if edge_index.ndim == 2 else 0

def get_edge_type_ids(edge_attr):
    # edge_attr shape [E, 6]; column 0 is edge_type_id
    if edge_attr is None or edge_attr.size(0) == 0:
        return np.array([], dtype=np.int64)
    col0 = edge_attr[:, 0].detach().cpu().numpy().astype(np.int64)
    return col0

def numpy_stats(arr):
    arr = np.asarray(arr)
    if arr.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

# ========= Scan and summarize =========
def main():
    describe_feature_layout()

    files = [f for f in os.listdir(FOLDER) if f.endswith(".pt")]
    files.sort()

    per_graph_rows = []
    label_counter = Counter()
    node_totals = Counter()
    edge_totals = Counter()
    per_type_node_totals = Counter()
    per_rel_edge_totals = Counter()
    global_edge_type_ids = Counter()

    # Optional quick previews of feature distributions (degrees)
    subj_deg_tot_all = []
    file_deg_tot_all = []
    net_deg_tot_all  = []

    total_time = 0.0
    total_files = 0

    for fname in tqdm(files, desc="Loading graphs", unit="file"):
        fpath = os.path.join(FOLDER, fname)
        try:
            t0 = time.time()
            g = torch.load(fpath, weights_only=False, map_location="cpu")
            total_time += (time.time() - t0)
            total_files += 1
        except Exception as e:
            print(f"[WARN] Skipping {fname}: {e}")
            continue

        # Structure: dict with keys: "x", "edge_index", "edge_attr", "y", "meta"
        x = g.get("x", {})
        eidx = g.get("edge_index", {})
        eattr = g.get("edge_attr", {})
        y = int(g.get("y", torch.tensor([0])).item())
        meta = g.get("meta", {})

        # Node counts per type
        n_subject = tensor_nrows(x.get("subject", torch.empty((0,0))))
        n_file    = tensor_nrows(x.get("fileobject", torch.empty((0,0))))
        n_net     = tensor_nrows(x.get("netflow", torch.empty((0,0))))
        total_nodes = n_subject + n_file + n_net

        # Edge counts per relation
        e_counts = {}
        for rel in RELS:
            rel_key = rel
            E = edge_count_from_index(eidx.get(rel_key, torch.empty((2,0), dtype=torch.long)))
            e_counts[str(rel_key)] = E
            per_rel_edge_totals[str(rel_key)] += E

        total_edges = sum(e_counts.values())

        # Edge-type IDs across relations
        for rel in RELS:
            ea = eattr.get(rel, None)
            if ea is None: continue
            ids = get_edge_type_ids(ea)
            if ids.size:
                global_edge_type_ids.update(ids.tolist())

        # Aggregate totals across corpus
        label_counter.update([y])
        node_totals.update({"total_nodes": total_nodes})
        edge_totals.update({"total_edges": total_edges})
        per_type_node_totals.update({"subject": n_subject, "fileobject": n_file, "netflow": n_net})

        # Quick degree totals from node features (last column)
        try:
            if n_subject:
                subj_deg_tot_all.extend(x["subject"][:, -1].numpy().tolist())
            if n_file:
                file_deg_tot_all.extend(x["fileobject"][:, -1].numpy().tolist())
            if n_net:
                net_deg_tot_all.extend(x["netflow"][:, -1].numpy().tolist())
        except Exception:
            pass

        # Per-graph row
        per_graph_rows.append({
            "file_name": fname,
            "graph_label": y,
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "n_subject": n_subject,
            "n_fileobject": n_file,
            "n_netflow": n_net,
            **{f"E_{k}": v for k,v in e_counts.items()},
            "window_id": meta.get("window_id", ""),
            "start_ns": meta.get("start_ns", ""),
            "end_ns": meta.get("end_ns", ""),
        })

    # Build DataFrame
    df = pd.DataFrame(per_graph_rows)
    if df.empty:
        print("No .pt graphs found or none could be loaded.")
        return

    # Save per-graph stats
    csv_out = os.path.join(FOLDER, "graph_stats.csv")
    df.to_csv(csv_out, index=False)

    # Corpus-level summary
    global_summary = {
        "num_graphs": int(df.shape[0]),
        "num_malicious": int((df["graph_label"]==1).sum()),
        "num_benign": int((df["graph_label"]==0).sum()),
        "total_nodes_sum": int(df["total_nodes"].sum()),
        "total_edges_sum": int(df["total_edges"].sum()),
        "avg_nodes_per_graph": float(df["total_nodes"].mean()),
        "avg_edges_per_graph": float(df["total_edges"].mean()),
        "avg_degree_overall": float((df["total_edges"]/df["total_nodes"].replace(0, np.nan)).mean()),
        "per_node_type_totals": dict(per_type_node_totals),
        "per_rel_edge_totals": dict(per_rel_edge_totals),
        "top_edge_type_ids": global_edge_type_ids.most_common(30),
        "time_summary": {
            "total_load_time_sec": round(total_time, 3),
            "avg_load_time_per_file_sec": round(total_time / max(total_files,1), 3),
        },
        "feature_schema": FEATURE_DOC,
    }
    json_out = os.path.join(FOLDER, "global_summary.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(global_summary, f, indent=2)

    # ====== Print brief console summary ======
    print("\n=== Corpus Summary ===")
    print(f"Graphs: {global_summary['num_graphs']} | Malicious: {global_summary['num_malicious']} | Benign: {global_summary['num_benign']}")
    print(f"Total nodes: {global_summary['total_nodes_sum']:,} | Total edges: {global_summary['total_edges_sum']:,}")
    print(f"Avg nodes/graph: {global_summary['avg_nodes_per_graph']:.1f} | Avg edges/graph: {global_summary['avg_edges_per_graph']:.1f}")
    print("Per-node-type totals:", global_summary["per_node_type_totals"])
    print("Per-relation edge totals:", global_summary["per_rel_edge_totals"])
    print("Top 10 edge_type_id (by frequency):", global_summary["top_edge_type_ids"][:10])
    print(f"Load time total: {global_summary['time_summary']['total_load_time_sec']}s | per-file: {global_summary['time_summary']['avg_load_time_per_file_sec']}s")

    # ====== Plots ======
    # 1) Histogram: nodes per graph
    plt.figure(figsize=(10,6))
    df["total_nodes"].hist(bins=40)
    plt.title("Number of Nodes per Graph")
    plt.xlabel("Nodes")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "nodes_per_graph_hist.png"))

    # 2) Histogram: edges per graph
    plt.figure(figsize=(10,6))
    df["total_edges"].hist(bins=40)
    plt.title("Number of Edges per Graph")
    plt.xlabel("Edges")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "edges_per_graph_hist.png"))

    # 3) Bar: total nodes per node type (corpus)
    node_type_totals = [per_type_node_totals["subject"], per_type_node_totals["fileobject"], per_type_node_totals["netflow"]]
    plt.figure(figsize=(9,6))
    plt.bar(["subject","fileobject","netflow"], node_type_totals)
    plt.title("Total Nodes per Node Type (corpus)")
    plt.xlabel("Node Type")
    plt.ylabel("Total Nodes")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "nodes_per_type_bar.png"))

    # 4) Bar: total edges per relation (corpus)
    rel_names = list(per_rel_edge_totals.keys())
    rel_vals  = [per_rel_edge_totals[k] for k in rel_names]
    plt.figure(figsize=(12,6))
    plt.bar(rel_names, rel_vals)
    plt.title("Total Edges per Relation (corpus)")
    plt.xlabel("Relation")
    plt.ylabel("Total Edges")
    plt.xticks(rotation=30, ha='right')
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "edges_per_relation_bar.png"))

    # 5) Pie: malicious vs benign graphs
    mal = int((df["graph_label"]==1).sum())
    ben = int((df["graph_label"]==0).sum())
    plt.figure(figsize=(7,7))
    plt.pie([mal, ben], labels=["Malicious","Benign"], autopct="%1.1f%%", startangle=90)
    plt.title("Graphs: Malicious vs Benign")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "graphs_label_pie.png"))

    # 6) Bar: top 20 edge_type_ids by frequency (across all relations)
    top_edge_ids = global_edge_type_ids.most_common(20)
    if top_edge_ids:
        ids, counts = zip(*top_edge_ids)
        ids = [str(i) for i in ids]
        plt.figure(figsize=(12,6))
        plt.bar(ids, counts)
        plt.title("Top 20 edge_type_id (frequency, corpus)")
        plt.xlabel("edge_type_id")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "top_edge_type_ids_bar.png"))

    # 7) Degree distributions (deg_tot per node type)
    def plot_deg_hist(vals, title, fname):
        if len(vals) == 0: return
        plt.figure(figsize=(10,6))
        plt.hist(vals, bins=50)
        plt.title(title)
        plt.xlabel("deg_tot")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, fname))

    plot_deg_hist(subj_deg_tot_all, "Subject: deg_tot distribution", "deg_tot_subject_hist.png")
    plot_deg_hist(file_deg_tot_all,  "FileObject: deg_tot distribution", "deg_tot_file_hist.png")
    plot_deg_hist(net_deg_tot_all,   "Netflow: deg_tot distribution", "deg_tot_netflow_hist.png")

    print(f"\nSaved per-graph stats → {csv_out}")
    print(f"Saved global summary → {json_out}")
    print(f"Saved plots → {PLOTS_DIR}")

if __name__ == "__main__":
    main()
