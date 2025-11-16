#!/usr/bin/env python3
"""
Prepare CIFAR-10 dataset for DiskANN and compute ground truth neighbors,
including class labels per base vector.
"""

import numpy as np
import struct
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from torchvision import datasets, transforms

# -----------------------------
# Helper functions
# -----------------------------
def save_bin_float(filename, data: np.ndarray):
    num, dim = data.shape
    with open(filename, "wb") as f:
        f.write(struct.pack("I", num))
        f.write(struct.pack("I", dim))
        data.astype(np.float32).tofile(f)
    print(f"✅ Saved {filename}: {num} vectors, {dim} dims")

def save_bin_uint32(filename, data: np.ndarray):
    num, k = data.shape
    with open(filename, "wb") as f:
        f.write(struct.pack("I", num))
        f.write(struct.pack("I", k))
        data.astype(np.uint32).tofile(f)
    print(f"✅ Saved {filename}: {num} queries × {k} neighbors")

def save_labels_txt(filename, labels):
    with open(filename, "w") as f:
        for lbl in labels:
            f.write(f"{lbl}\n")
    print(f"✅ Saved {filename}: {len(labels)} class labels")

# -----------------------------
# Load CIFAR-10
# -----------------------------
print("📥 Loading CIFAR-10 (train + test)...")
transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

X_train = trainset.data.reshape(len(trainset.data), -1).astype(np.float32)
X_test = testset.data.reshape(len(testset.data), -1).astype(np.float32)

y_train = np.array(trainset.targets, dtype=np.int32)
y_test = np.array(testset.targets, dtype=np.int32)

# -----------------------------
# Normalize vectors
# -----------------------------
print("⚙️ Normalizing (L2)...")
def normalize_rows(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-10
    return v / norm

X_train = normalize_rows(X_train)
X_test = normalize_rows(X_test)

# -----------------------------
# Base + query split
# -----------------------------
base_data = X_train
query_data = X_test
base_labels = y_train
query_labels = y_test
print(f"📊 Base: {base_data.shape}, Query: {query_data.shape}")

# -----------------------------
# Ground truth via brute-force search
# -----------------------------
K = 100
print(f"🔍 Computing {K}-NN ground truth...")
nbrs = NearestNeighbors(n_neighbors=K, algorithm="brute", metric="euclidean")
nbrs.fit(base_data)
_, indices = nbrs.kneighbors(query_data)

# -----------------------------
# Save all outputs
# -----------------------------
save_bin_float("cifar10_base.fbin", base_data)
save_bin_float("cifar10_query.fbin", query_data)
save_bin_uint32("cifar10_gt.ibin", indices)
save_labels_txt("cifar10_labels.txt", base_labels)
save_labels_txt("cifar10_query_labels.txt", query_labels)

print("\n🎯 All done! Generated files:")
print("  - cifar10_base.fbin")
print("  - cifar10_query.fbin")
print("  - cifar10_gt.ibin")
print("  - cifar10_labels.txt          (train/base labels)")
print("  - cifar10_query_labels.txt    (test/query labels)")
