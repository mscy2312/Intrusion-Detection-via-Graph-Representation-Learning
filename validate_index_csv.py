#!/usr/bin/env python3
# validate_index_csv.py
import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--per-file-stats", required=True, help="per_file_stats.csv from summarize_pt_dataset.py")
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    idx = pd.read_csv(args.index_csv)
    stats = pd.read_csv(args.per_file_stats)
    data_files = {p.name for p in Path(args.data_dir).glob("*.pt")}

    # Missing files referenced in index
    missing = sorted(set(idx["filename"]) - data_files) if "filename" in idx.columns else []
    # Files not referenced in index
    orphan = sorted(data_files - set(idx["filename"])) if "filename" in idx.columns else []

    # Merge to compare dims, nodes, edges if present
    problems = []
    if "filename" in idx.columns:
        merged = idx.merge(stats, on="filename", how="left", suffixes=("_idx","_scan"))
        for _, r in merged.iterrows():
            for col_pair in [("x_dim","x_dim"), ("edge_attr_dim","edge_attr_dim"), ("num_nodes","num_nodes"), ("num_edges","num_edges")]:
                a, b = col_pair
                if a in merged.columns and b in merged.columns and pd.notna(r[a]) and pd.notna(r[b]):
                    if int(r[a]) != int(r[b]):
                        problems.append({"filename": r["filename"], "field": a, "index_val": int(r[a]), "scanned_val": int(r[b])})

    report = {"missing_in_folder": missing, "unlisted_in_index": orphan, "mismatches": problems}
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
