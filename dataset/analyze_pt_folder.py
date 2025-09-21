#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_pt_folder_summary.py

Analyze all .pt files in a given directory. During processing, shows only a progress bar.
After processing all files, prints out summaries per file *and* overall summary:
- Number of files
- Count of malicious vs non-malicious
- Total nodes, total edges
- Min / Max / Average nodes, edges

Usage:
    python analyze_pt_folder_summary.py --dir path/to/folder
    python analyze_pt_folder_summary.py --dir path/to/folder --save report.json
"""

import argparse
from pathlib import Path
import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected
from tqdm import tqdm
import json

def summarize_graph(d: Data) -> dict:
    """
    Given a Data graph (or dict format), return summary dictionary without modifying d.
    """
    summary = {}

    # number of nodes
    try:
        summary["num_nodes"] = int(d.num_nodes)
    except Exception:
        if getattr(d, "x", None) is not None:
            summary["num_nodes"] = int(d.x.size(0))
        else:
            summary["num_nodes"] = None

    # number of (directed) edges
    try:
        summary["num_edges_directed"] = int(d.num_edges)
    except Exception:
        if getattr(d, "edge_index", None) is not None:
            summary["num_edges_directed"] = int(d.edge_index.size(1))
        else:
            summary["num_edges_directed"] = None

    # feature dims
    summary["node_feature_dim"] = d.x.size(1) if getattr(d, "x", None) is not None else None

    if getattr(d, "edge_attr", None) is not None:
        summary["has_edge_attr"] = True
        try:
            summary["edge_attr_dim"] = int(d.edge_attr.size(1))
        except Exception:
            summary["edge_attr_dim"] = None
    else:
        summary["has_edge_attr"] = False
        summary["edge_attr_dim"] = None

    # label
    if getattr(d, "y", None) is not None:
        try:
            summary["label"] = int(d.y)
        except:
            try:
                summary["label"] = int(d.y.item())
            except:
                summary["label"] = str(d.y)
    else:
        summary["label"] = None

    # is undirected originally
    try:
        und = is_undirected(d.edge_index, num_nodes=d.num_nodes)
        summary["is_undirected_orig"] = bool(und)
    except Exception:
        summary["is_undirected_orig"] = False

    return summary

def analyze_file(fp: Path) -> dict:
    """
    Load one .pt file and generate summary dictionary.
    Includes 'file' name, maybe error, else summary.
    """
    summary = {"file": fp.name}
    try:
        obj = torch.load(fp, map_location='cpu')
    except Exception as e:
        summary["error"] = f"load_failed: {e}"
        return summary

    if isinstance(obj, Data):
        d = obj
    elif isinstance(obj, dict):
        lab = obj.get('label', obj.get('y', None))
        d = Data(
            x = obj.get('x', None),
            edge_index = obj.get('edge_index'),
            edge_attr = obj.get('edge_attr', None),
            y = torch.tensor(int(lab), dtype=torch.long) if lab is not None else None,
        )
    else:
        summary["error"] = f"unsupported_type: {type(obj)}"
        return summary

    if d.x is None or d.edge_index is None:
        summary["error"] = "missing x or edge_index"
        return summary

    summ = summarize_graph(d)
    summary.update(summ)
    return summary

def analyze_folder_with_summary(dirpath: Path, save_path: Path = None) -> list:
    """
    Analyze all .pt files under dirpath. Show progress bar while reading/processing.
    After finishing, prints summaries per file plus overall summary.
    Returns list of file summaries.
    """
    if not dirpath.exists():
        print(f"Directory {dirpath} does not exist.")
        return []

    pt_files = list(dirpath.glob("*.pt"))
    if len(pt_files) == 0:
        print(f"No .pt files found in directory {dirpath}")
        return []

    reports = []
    # Accumulators for overall summary
    total_files = 0
    count_malicious = 0
    count_non_malicious = 0
    node_counts = []
    edge_counts = []

    # Processing with progress bar
    for fp in tqdm(pt_files, desc="Analyzing .pt files"):
        rep = analyze_file(fp)
        reports.append(rep)
        total_files += 1

        if "error" in rep:
            continue

        # label
        lab = rep.get("label")
        if lab is not None:
            if lab == 1:
                count_malicious += 1
            elif lab == 0:
                count_non_malicious += 1
            else:
                # If labels other than 0/1, count in neither
                pass

        # nodes/edges
        nn = rep.get("num_nodes")
        ne = rep.get("num_edges_directed")
        if (nn is not None) and (isinstance(nn, (int, float))):
            node_counts.append(nn)
        if (ne is not None) and (isinstance(ne, (int, float))):
            edge_counts.append(ne)

    # After processing, print summaries per file
    print(f"\nCompleted analysis of {total_files} files in {dirpath}\n")
    for rep in reports:
        if "error" in rep:
            print(f"{rep['file']}: ERROR -> {rep['error']}")
        else:
            print(f"{rep['file']}: nodes={rep.get('num_nodes')}, edges={rep.get('num_edges_directed')}, label={rep.get('label')}, undirected_orig={rep.get('is_undirected_orig')}")

    # Overall summary
    print("\n=== Overall Summary ===")
    print(f"Total files: {total_files}")
    print(f"Malicious (label=1): {count_malicious}")
    print(f"Non-Malicious (label=0): {count_non_malicious}")
    # Edge: if labels other than 0/1, might need to adjust

    # Nodes stats
    if node_counts:
        total_nodes_all = sum(node_counts)
        min_nodes = min(node_counts)
        max_nodes = max(node_counts)
        avg_nodes = total_nodes_all / len(node_counts)
        print(f"Nodes — Total: {total_nodes_all}, Min: {min_nodes}, Max: {max_nodes}, Avg: {avg_nodes:.2f}")
    else:
        print("Nodes — No valid node count data")

    # Edges stats
    if edge_counts:
        total_edges_all = sum(edge_counts)
        min_edges = min(edge_counts)
        max_edges = max(edge_counts)
        avg_edges = total_edges_all / len(edge_counts)
        print(f"Edges — Total: {total_edges_all}, Min: {min_edges}, Max: {max_edges}, Avg: {avg_edges:.2f}")
    else:
        print("Edges — No valid edge count data")

    # Optionally save
    if save_path is not None:
        try:
            with open(save_path, "w") as f:
                json.dump({
                    "file_summaries": reports,
                    "overall_summary": {
                        "total_files": total_files,
                        "malicious": count_malicious,
                        "non_malicious": count_non_malicious,
                        "nodes_total": total_nodes_all if node_counts else None,
                        "nodes_min": min_nodes if node_counts else None,
                        "nodes_max": max_nodes if node_counts else None,
                        "nodes_avg": avg_nodes if node_counts else None,
                        "edges_total": total_edges_all if edge_counts else None,
                        "edges_min": min_edges if edge_counts else None,
                        "edges_max": max_edges if edge_counts else None,
                        "edges_avg": avg_edges if edge_counts else None,
                    }
                }, f, indent=2)
            print(f"\nSaved report to {save_path}")
        except Exception as e:
            print(f"Failed to save report to {save_path}: {e}")

    return reports

def main():
    parser = argparse.ArgumentParser(description="Analyze folder of .pt graph files and give overall summary")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing .pt files")
    parser.add_argument("--save", type=str, default=None, help="Path to save JSON report (optional)")
    args = parser.parse_args()

    dirpath = Path(args.dir)
    save_path = Path(args.save) if args.save else None

    analyze_folder_with_summary = analyze_folder_with_summary if False else analyze_folder_with_summary  # placeholder

    analyze_folder_with_summary(dirpath, save_path)

if __name__ == "__main__":
    # Fix naming: call function
    parser = argparse.ArgumentParser(description="Analyze folder of .pt graph files with overall summary")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing .pt files")
    parser.add_argument("--save", type=str, default=None, help="Path to save JSON report (optional)")
    args = parser.parse_args()

    dirpath = Path(args.dir)
    save_path = Path(args.save) if args.save else None

    analyze_folder_with_summary(dirpath, save_path)
