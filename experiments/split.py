#!/usr/bin/env python3
import struct
import sys
import os

"""
Usage:
    python split_bin_auto_labels.py embeddings.bin 80

This creates automatically:
    embeddings_train.bin
    embeddings_test.bin
    label_train.txt
    label_test.txt

And always reads labels from:
    label_file.txt
"""

LABEL_FILENAME = "label_file.txt"

def read_header(f):
    npts = struct.unpack('i', f.read(4))[0]
    ndims = struct.unpack('i', f.read(4))[0]
    return npts, ndims

def write_header(f, npts, ndims):
    f.write(struct.pack('i', npts))
    f.write(struct.pack('i', ndims))

def main():
    if len(sys.argv) != 3:
        print("Usage: python split_bin_auto_labels.py embeddings.bin TRAIN_PERCENT")
        sys.exit(1)

    input_bin = sys.argv[1]
    percent = float(sys.argv[2]) / 100.0

    # Derive base name for output naming
    base = os.path.splitext(input_bin)[0]
    train_bin = base + "_train.bin"
    test_bin  = base + "_test.bin"

    train_lbl = base + "_train_labels.txt"
    test_lbl  = base + "_test_labels.txt"

    # Load labels
    if not os.path.exists(LABEL_FILENAME):
        print(f"ERROR: Missing {LABEL_FILENAME}")
        sys.exit(1)

    with open(LABEL_FILENAME, "r") as lf:
        labels = [line.rstrip("\n") for line in lf]

    with open(input_bin, "rb") as f:
        npts, ndims = read_header(f)
        vec_size = ndims * 4  # float32 per dimension

        if len(labels) != npts:
            print("ERROR: Number of labels does not match number of embeddings!")
            print(f"Labels: {len(labels)}, Embeddings: {npts}")
            sys.exit(1)

        print(f"Loaded binary: {npts} vectors, dim = {ndims}")
        print(f"Loaded labels: {len(labels)}")

        train_count = int(npts * percent)
        test_count = npts - train_count

        print(f"Training vectors: {train_count}")
        print(f"Test vectors: {test_count}")

        with open(train_bin, "wb") as f_train, open(test_bin, "wb") as f_test, \
             open(train_lbl, "w") as lbl_train, open(test_lbl, "w") as lbl_test:

            write_header(f_train, train_count, ndims)
            write_header(f_test, test_count, ndims)

            for i in range(npts):
                vec = f.read(vec_size)
                if not vec:
                    break

                if i < train_count:
                    f_train.write(vec)
                    lbl_train.write(labels[i] + "\n")
                else:
                    f_test.write(vec)
                    lbl_test.write(labels[i] + "\n")

    print("\nFinished splitting embeddings + labels!")
    print("Created:")
    print(" ", train_bin)
    print(" ", test_bin)
    print(" ", train_lbl)
    print(" ", test_lbl)


if __name__ == "__main__":
    main()
