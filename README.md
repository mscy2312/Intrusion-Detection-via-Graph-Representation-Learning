# Intrusion-Detection-via-Graph-Representation-Learning
## Project at a glance
Topic: Provenance-based intrusion detection on DARPA Engagement 5 (FiveDirections/MARPLE) using time-windowed graphs and GNNs.
Raw → Labeled Parquet: E5 .bin.gz → TA3 consumer + TCCDMDatum.avsc → Parquet per record type (events/, subjects/, fileobjects/, netflows/…), with a malicious label derived from ground-truth attack windows.
Key decision: Label maliciousness purely by time windows (ignore host fields).
Graph building approach: Build 10-minute shard-local windows (per Parquet file), keep all edges inside the window to preserve context, and label the entire graph by GT overlap. Output ready-to-train tensors (.pt) instead of NetworkX graphs.

## What’s done (pipeline & artifacts)
### 1) Preprocessing (done)
Script: preprocess_cdm_time_only_fast.py
Converts TA3 JSON → Parquet and assigns an event-level boolean malicious by time-only overlap with ground truth.
Output folder: out_parquet_time_only/
events/ (each file name prefixed with malicious__ or non_malicious__)
subjects/, fileobjects/, netflows/ (node tables)

Windows run example
python preprocess_cdm_time_only_fast.py `
  --input "D:\Dataset\Engagement 5 Maple Dataset\JSON files\*.json" `
  --attacks .\attack_windows_e5_iso_and_ns_cleaned.json `
  --output-dir .\out_parquet_time_only `
  --compression zstd --batch-size 200000 --debug


### 2) Feature reconnaissance (done)
Script: inspect_features.py
Lists available columns and prints samples per record type.
Decision: For graph windows we’ll start with:
Node features: one-hot node_type (subjects/fileobjects/netflows) + normalized deg_in/deg_out/deg_total.
Edge features: one-hot event_type + normalized time within window.
(Per-edge malicious is not used for training the graph label; we keep window-level labels from GT.)

Windows run
python inspect_features.py --input-dir .\out_parquet_time_only --samples 5

### 3) Single-window sanity checks (done)
Script: test_one_window.py
Builds a single 10-min graph from the first events file, with & without UUID→type mapping.
Logs nodes/edges, top degrees, and a .gpickle for quick visualization (not used for GNN training).

Windows run
python test_one_window.py

### 4) Fast multi-window tensor builder (done)
We iterated to a no-args, hard-coded, parallel builder that:
samples 100 malicious + 200 benign windows (2:1),
takes multiple windows per file (mal: 2 per shard; ben: 1),
uses 10-min windows with 10-min stride, edge cap = 100k,
labels by GT overlap,
UUID→type mapping resolved on the fly from Parquet (no DB),
parallelized by file (ProcessPool),
Arrow-vectorized inner loop,
profiles each window (read/parse, uuid typing, tensor build, save).

Recommended scripts (final)
No-args single-process (simple): make_multi_windows_tensors.py
Easy, reproducible; good when debugging.
No-args parallel + profiling: make_multi_windows_tensors_parallel.py ✅ Use this for scale

Outputs:
<OUTPUT_DIR>\*.pt (one tensor per window)
<OUTPUT_DIR>\index.csv (filename, start/end ns, label, nodes, edges, source file)
<OUTPUT_DIR>\profile.csv (per-phase timings)
<OUTPUT_DIR>\win_build.log (per-window stats)

Windows run (no args)
python make_multi_windows_tensors_parallel.py


## Where we are right now (train-ready assets)
Training graphs: <OUTPUT_DIR>\*.pt (PyTorch tensors)
Each file contains:
x: node feature matrix (one-hot node_type + normalized degrees),
edge_index: COO edges (2×E),
edge_attr: edge features (one-hot event_type + normalized time),
meta: mappings (node id ↔ uuid, dictionaries for types),
label: graph label (1 = malicious window, 0 = benign).
Index: index.csv for splits (train/val/test) and bookkeeping.
Profile: profile.csv for finding bottlenecks if you need more speed.
These are directly consumable by a PyTorch Geometric dataset/loader.

What still needs to be done (next chat: GNN training)

## Dataset & splits
Read index.csv, shuffle by source file (to reduce leakage), split train/val/test (e.g., 70/15/15) with class balance.
Optionally cap max windows per source to avoid overfitting a few shards.

## Model(s) to start with
Graph-level classification: GraphSAGE, GIN/GINE (edge features friendly), GAT (attention over event types). Start with GINEConv (edge attributes helpful for event_type/time).
Readout: global mean + max pool concat, then MLP → 2-class logits.

## Training setup
Loss: BCEWithLogitsLoss or cross-entropy with class weights (to handle benign>malicious).
Optimizer: AdamW, lr ~1e-3 (tune), weight decay ~1e-4.
Batching: mini-batches of graphs (start with batch size 4–8; change based on GPU RAM).
Early stopping: monitor val AUROC (better than accuracy for imbalance).
Logging: loss, AUROC, precision/recall/F1; save best checkpoint.

## Quality checks / ablations
Feature ablations: (a) no time, (b) no degrees, (c) no node types → confirm gains.
Window sensitivity: 5, 10, 15 mins; edge cap 50k vs 100k.
Sampling: if needed, oversample malicious windows or use weighted sampler.

## File-by-file quick reference (and why):
### preprocess_cdm_time_only_fast.py
Converts TA3 JSON → Parquet; labels events by time-only GT overlap.
Why: small, fast, streaming; produces the canonical Parquet corpora for everything else.

### make_multi_windows_tensors.py (no-args, simple)
Builds a fixed number of windows → .pt tensors (single-process).
Why: reproducible baseline; easy to debug.

### make_multi_windows_tensors_parallel.py ✅ (no-args, parallel + profiling)
Same outputs as above, but parallelized and Arrow-vectorized with per-phase profiling.
Why: production data builder for training.

## How to run (Windows 10 recap)
Build training tensors (parallel)
cd "C:\Users\Ali\Desktop\ChatGPT Scripts"
pip install torch pyarrow tqdm
python make_multi_windows_tensors_parallel.py
Outputs: win_tensors_balanced\*.pt, index.csv, profile.csv, win_build.log.

## Final notes / expectations
We now have graph-level labeled tensors (malicious/benign windows) with useful structural and semantic features.
Training can start immediately with GINE or GraphSAGE; we’ll wire up a PyG Dataset that reads .pt files listed in index.csv, create loaders, and train a classifier with class weighting and early stopping.
If you need more speed or more windows later, the parallel builder + profiling will show exactly where to optimize next.
