import os
import torch
import glob
import matplotlib.pyplot as plt
import numpy as np
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def visualize_dataset(data_dir: str, save_dir: str):
    pt_files = glob.glob(os.path.join(data_dir, "*.pt"))
    if not pt_files:
        logger.error(f"No .pt files found in {data_dir}")
        return

    node_counts = []
    edge_counts = []
    labels = []

    for file_path in pt_files:
        try:
            loaded_data = torch.load(file_path)
            if not isinstance(loaded_data, dict):
                continue
            x = loaded_data.get('x')
            edge_index = loaded_data.get('edge_index')
            label = loaded_data.get('label')
            if x is None or edge_index is None or label is None:
                continue
            node_counts.append(x.shape[0])
            edge_counts.append(edge_index.shape[1])
            labels.append(int(label))
        except Exception as e:
            logger.warning(f"Failed to load {file_path}: {str(e)}")
            continue

    if not node_counts:
        logger.error("No valid data found")
        return

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(node_counts, bins=50, color='blue', alpha=0.7, label='All Graphs')
    plt.xlabel('Node Count')
    plt.ylabel('Frequency')
    plt.title('Node Count Distribution')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(edge_counts, bins=50, color='green', alpha=0.7, label='All Graphs')
    plt.xlabel('Edge Count')
    plt.ylabel('Frequency')
    plt.title('Edge Count Distribution')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dataset_distributions.png"))
    plt.close()

    # Separate by class
    node_counts_benign = [n for n, l in zip(node_counts, labels) if l == 0]
    node_counts_malicious = [n for n, l in zip(node_counts, labels) if l == 1]
    edge_counts_benign = [e for e, l in zip(edge_counts, labels) if l == 0]
    edge_counts_malicious = [e for e, l in zip(edge_counts, labels) if l == 1]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(node_counts_benign, bins=50, color='blue', alpha=0.5, label='Benign')
    plt.hist(node_counts_malicious, bins=50, color='red', alpha=0.5, label='Malicious')
    plt.xlabel('Node Count')
    plt.ylabel('Frequency')
    plt.title('Node Count by Class')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(edge_counts_benign, bins=50, color='blue', alpha=0.5, label='Benign')
    plt.hist(edge_counts_malicious, bins=50, color='red', alpha=0.5, label='Malicious')
    plt.xlabel('Edge Count')
    plt.ylabel('Frequency')
    plt.title('Edge Count by Class')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dataset_distributions_by_class.png"))
    plt.close()

    logger.info(f"Plots saved to {save_dir}")

if __name__ == "__main__":
    data_dir = r"C:\Users\Ali\Desktop\ChatGPT Scripts\win_tensors_balanced"
    save_dir = r"C:\Users\Ali\Desktop\ChatGPT Scripts\GrokVisualizations"
    visualize_dataset(data_dir, save_dir)