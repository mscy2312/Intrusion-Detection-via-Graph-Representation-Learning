#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_pt_file.py

Analyze a single .pt graph file. Prints summary stats such as:
- number of nodes
- number of edges (directed / stored)
- node feature dimension
- edge_attr existence and dimension
- whether the graph is already undirected (based on edge_index)
- first few edge samples

Usage:
    python analyze_pt_file.py --file path/to/graph.pt
"""

import argparse
from pathlib import Path
import torch
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected

def analyze_file(fp: Path):
    """
    Load a single .pt file, interpret as torch_geometric Data or dict with fields,
    and print summary.
    """
    # Load
    obj = torch.load(fp, map_location='cpu')
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
        print(f"[{fp.name}] Unsupported object type: {type(obj)}")
        return

    # Check prerequisites
    if d.x is None:
        print(f"[{fp.name}] Missing node features (x); cannot analyze feature dim.")
    if d.edge_index is None:
        print(f"[{fp.name}] Missing edge_index; cannot analyze edges.")
    # We'll assume label may or may not exist
    # Summary
    num_nodes = d.num_nodes if getattr(d, 'num_nodes', None) is not None else (d.x.size(0) if d.x is not None else None)
    num_edges = d.num_edges if getattr(d, 'num_edges', None) is not None else (d.edge_index.size(1) if d.edge_index is not None else None)

    print(f"File: {fp.name}")
    print(f"  Number of nodes: {num_nodes}")
    print(f"  Number of (directed) edges: {num_edges}")
    if d.x is not None:
        print(f"  Node feature dimension: {d.x.size(1)}")
    if getattr(d, 'edge_attr', None) is not None:
        print(f"  Has edge_attr: True, dimension: {d.edge_attr.size(1)}")
    else:
        print(f"  Has edge_attr: False")
    if getattr(d, 'y', None) is not None:
        try:
            lab = int(d.y)
        except:
            lab = d.y
        print(f"  Label: {lab}")
    # undirected check
    try:
        und = is_undirected(d.edge_index, num_nodes=d.num_nodes)
    except Exception as e:
        und = False
        print(f"  Warning: undirected check failed: {e}")
    print(f"  is_undirected_orig: {und}")
    # Edge samples
    if d.edge_index is not None:
        sample_n = min(10, d.edge_index.size(1))
        samples = d.edge_index[:, :sample_n].tolist()
        print(f"  First {sample_n} edges: {samples}")

def main():
    parser = argparse.ArgumentParser(description="Analyze a single .pt graph file")
    parser.add_argument("--file", type=str, required=True, help="Path to .pt file to analyze")
    args = parser.parse_args()

    fp = Path(args.file)
    if not fp.exists():
        print(f"File {fp} does not exist.")
        return
    analyze_file(fp)

if __name__ == "__main__":
    main()
