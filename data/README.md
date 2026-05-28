# Data Directory

This directory contains the local dataset structure required to reproduce the experiments.

Raw datasets are **not included** in this repository. Users must download them from their original sources and place the files in the expected paths described below.

## Expected structure

    data/
    ├── raw/
    │   ├── nsl_kdd/
    │   │   ├── KDDTrain+.txt
    │   │   └── KDDTest+.txt
    │   │
    │   └── fiveg_nidd/
    │       └── Encoded.csv
    │
    └── processed/

## NSL-KDD

The NSL-KDD experiments expect the following files:

    data/raw/nsl_kdd/KDDTrain+.txt
    data/raw/nsl_kdd/KDDTest+.txt

The current implementation uses the binary normal-vs-DoS task. Samples labeled as `normal` are mapped to class `0`, DoS attacks are mapped to class `1`, and other attack categories are filtered out.

## 5G-NIDD

The 5G-NIDD experiments expect the following file:

    data/raw/fiveg_nidd/Encoded.csv

The current implementation uses the binary Benign-vs-UDPFlood task. Samples labeled as `Benign` are mapped to class `0`, samples labeled as `UDPFlood` are mapped to class `1`, and other attack categories are filtered out.

## Notes

The preprocessing steps are implemented in:

    src/data/nsl_kdd.py
    src/data/fiveg_nidd.py
    src/data/partition.py

The repository keeps this directory structure for reproducibility, but raw datasets and generated processed files are ignored by Git.