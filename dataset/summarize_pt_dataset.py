#!/usr/bin/env python3
# summarize_pt_dataset.py
import argparse, json, math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def load_as_data(obj):
    if isinstance(obj, Data):
        return obj
    if isinstance(obj, dict):
        y = obj.get("label", obj.get("y"))
        return Data(
            x=obj.get("x"), edge_index=obj.get("edge_index"),
            edge_attr=obj.get("edge_attr"),
            y=torch.tensor(int(y)) if y is not None else None,
        )
    return None

def is_undirected_with_reciprocals(edge_index: torch.Tensor, num_nodes: int) -> float:
    # returns fraction of directed edges that have a reciprocal
    if edge_index.numel() == 0: return 1.0
    ei = edge_index.cpu().numpy()
    pairs = set(map(tuple, ei.T))
    rev = set(map(tuple, np.flip(ei, 0).T))
    if not pairs: return 1.0
    return len(pairs & rev) / len(pairs)

def count_self_loops(edge_index: torch.Tensor) -> int:
    if edge_index.numel() == 0: return 0
    return int((edge_index[0] == edge_index[1]).sum().item())

def count_isolated(num_nodes: int, edge_index: torch.Tensor) -> int:
    if edge_index.numel() == 0: return num_nodes
    deg = torch.bincount(edge_index.view(-1), minlength=num_nodes)
    return int((deg == 0).sum().item())

def has_nans_or_infs(t: torch.Tensor) -> bool:
    if t is None: return False
    return not torch.isfinite(t).all().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Folder containing *.pt graphs")
    ap.add_argument("--out-dir", required=True, help="Folder to write reports/plots")
    ap.add_argument("--max-files", type=int, default=0, help="Optional: limit number of files for a quick audit")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir); (out_dir / "plots").mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.pt"))
    if args.max_files > 0:
        files = files[:args.max_files]
    if not files:
        raise SystemExit(f"No .pt files found in {data_dir}")

    rows = []
    label_counts = Counter()
    x_dims = Counter()
    e_dims = Counter()
    undirected_fracs = []
    self_loops_all = 0
    isolated_all = 0
    dup_counts = 0
    nan_x = 0; nan_ea = 0

    top_nodes = []
    top_edges = []

    preview = []

    for fp in files:
        obj = torch.load(fp, map_location="cpu")
        d = load_as_data(obj)
        if d is None or d.x is None or d.edge_index is None or d.y is None:
            # skip malformed
            continue

        N = int(d.num_nodes); E = int(d.num_edges)
        x_dim = int(d.x.size(-1))
        ea_dim = int(d.edge_attr.size(-1)) if getattr(d, "edge_attr", None) is not None else 0
        y = int(d.y.item())

        # undirectedness
        frac = is_undirected_with_reciprocals(d.edge_index, N)
        undirected_fracs.append(frac)

        # duplicates (treat edge as unordered pair)
        if E > 0:
            ei = d.edge_index.cpu().numpy()
            unordered = np.sort(ei, axis=0)  # make (min,max)
            tup = list(map(tuple, unordered.T))
            dup_counts += (len(tup) - len(set(tup)))

        # self-loops & isolated
        sl = count_self_loops(d.edge_index)
        iso = count_isolated(N, d.edge_index)
        self_loops_all += sl
        isolated_all += iso

        # NaN/Inf checks
        if has_nans_or_infs(d.x): nan_x += 1
        if getattr(d, "edge_attr", None) is not None and has_nans_or_infs(d.edge_attr): nan_ea += 1

        label_counts[y] += 1
        x_dims[x_dim] += 1
        e_dims[ea_dim] += 1

        rows.append({
            "filename": fp.name, "label": y,
            "num_nodes": N, "num_edges": E,
            "x_dim": x_dim, "edge_attr_dim": ea_dim,
            "undirected_frac": frac, "self_loops": sl, "isolated_nodes": iso
        })

        top_nodes.append((N, fp.name))
        top_edges.append((E, fp.name))

        if len(preview) < 30:
            preview.append({
                "filename": fp.name, "label": y, "num_nodes": N, "num_edges": E,
                "x_dim": x_dim, "edge_attr_dim": ea_dim, "undirected_frac": round(frac, 3),
                "self_loops": sl, "isolated_nodes": iso
            })

    if not rows:
        raise SystemExit("All graphs were malformed or filtered. Nothing to summarize.")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_file_stats.csv", index=False)

    # Basic plots (small, fast)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # class distribution
        vals, cnts = zip(*sorted(label_counts.items()))
        plt.figure(); plt.bar(vals, cnts); plt.title("Class distribution"); plt.xlabel("label"); plt.ylabel("count")
        plt.savefig(out_dir / "plots" / "class_distribution.png"); plt.close()

        # nodes, edges
        plt.figure(); df["num_nodes"].hist(bins=30); plt.title("Nodes per graph")
        plt.savefig(out_dir / "plots" / "nodes_hist.png"); plt.close()
        plt.figure(); df["num_edges"].hist(bins=30); plt.title("Edges per graph")
        plt.savefig(out_dir / "plots" / "edges_hist.png"); plt.close()

        # undirected frac
        plt.figure(); df["undirected_frac"].hist(bins=30); plt.title("Undirected reciprocal fraction")
        plt.savefig(out_dir / "plots" / "undirected_frac_hist.png"); plt.close()
    except Exception:
        pass

    summary = {
        "num_files_scanned": len(files),
        "label_counts": dict(label_counts),
        "x_dim_counts": dict(x_dims),
        "edge_attr_dim_counts": dict(e_dims),
        "undirected_frac_mean": float(np.mean(undirected_fracs)),
        "undirected_frac_median": float(np.median(undirected_fracs)),
        "total_self_loops": int(self_loops_all),
        "total_isolated_nodes": int(isolated_all),
        "files_with_nan_in_x": int(nan_x),
        "files_with_nan_in_edge_attr": int(nan_ea),
        "largest_by_nodes": sorted(top_nodes, reverse=True)[:10],
        "largest_by_edges": sorted(top_edges, reverse=True)[:10],
    }
    with (out_dir / "dataset_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "preview.jsonl").open("w") as f:
        for p in preview:
            f.write(json.dumps(p) + "\n")

    print(f"Wrote: {out_dir/'dataset_summary.json'}, {out_dir/'per_file_stats.csv'}, preview.jsonl, plots/")
if __name__ == "__main__":
    main()
