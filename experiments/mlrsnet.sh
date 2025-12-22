#!/bin/bash

# -----------------------------
# 1. Compute Ground Truth
# -----------------------------
compute_groundtruth_for_filters() {
    local ds_name=$1

    "$build_path/apps/utils/compute_groundtruth_for_filters" \
        --data_type float \
        --dist_fn l2 \
        --base_file "$index_prefix/train_embeddings.bin" \
        --query_file "$index_prefix/test_embeddings.bin" \
        --gt_file "$index_prefix/ground_truth.bin" \
        --K 100 \
        --label_file "$ds_path/train_labels.txt" \
        --filter_label_file "$ds_path/test_labels.txt"
}

compute_groundtruth() {
    local ds_name=$1

    "$build_path/apps/utils/compute_groundtruth" \
        --data_type float \
        --dist_fn l2 \
        --base_file "$index_prefix/train_embeddings.bin" \
        --query_file "$index_prefix/test_embeddings.bin" \
        --gt_file "$index_prefix/ground_truth.bin" \
        --K 100 
}


# -----------------------------
# 2. Build Memory Index
# -----------------------------
build_memory_index() {
    local ds_name=$1

    mkdir $index_prefix
    "$build_path/apps/build_memory_index" \
        --data_type float \
        --dist_fn $distance \
        --data_path  "$index_prefix/train_embeddings.bin" \
        --index_path_prefix "$index_prefix/R32_L50_filtered_index" \
        -R 32 \
        --FilteredLbuild 50 \
        --alpha 1.2 \
        --label_file "$ds_path/label_file.txt" \
        --filtered_medoids 4 \
        --trained_filtering $training
}

# -----------------------------
# 3. Search Memory Index
# -----------------------------
search_memory_index() {
    local ds_name=$1
    "$build_path/apps/search_memory_index" \
        --data_type float \
        --dist_fn $distance \
        --index_path_prefix "$index_prefix/R32_L50_filtered_index" \
        --query_file "$index_prefix/test_embeddings.bin" \
        --gt_file "$index_prefix/ground_truth.bin" \
        --query_filters_file "$ds_path/test_labels.txt" \
        -K 10 \
        -L 10 20 50 100 \
        --result_path "$index_prefix/results"
}

# -----------------------------
# Full Pipeline Wrapper
# -----------------------------
run_diskann_pipeline() {
    local ds_name=$1

    # Expand ~ in path
    build_path="${build_path/#\~/$HOME}"

    echo "----------------------------------------"
    echo "Running DiskANN pipeline on dataset: $ds_name"
    echo "Data path: $ds_path"
    echo "Build path: $build_path"
    echo "----------------------------------------"

    # Step 1
    echo "[1/3] Computing ground truth..."
    compute_groundtruth "$ds_name"

    # Step 2
    echo "[2/3] Building memory index..."
    build_memory_index "$ds_name"

    # Step 3
    echo "[3/3] Searching memory index..."
    search_memory_index "$ds_name"

    echo "Finished pipeline for: $ds_name"
    echo "========================================"
}

# -----------------------------
# Paths and Dataset List
# -----------------------------
ds_path=~/DiskANN/data/MLRSNet
build_path=~/DiskANN/build
index_path=~/DiskANN/data/index

# datasets='all-mpnet  e5-large  gte-large'
datasets='clip_vitb32_laion2b  dinov2-small   resnet-50'
# -----------------------------
# Run Pipeline for Each Dataset
# -----------------------------
distance=fusion
rm output.txt
for training in 0 20 50 100; do
for ds in $datasets; do
    index_prefix=$ds_path/$ds
    run_diskann_pipeline "$ds" 
done
done