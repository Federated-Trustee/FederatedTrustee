# FederatedTrustee
Explainable detection of poisoned clients in Federated Learning using surrogate decision trees.

FederatedTrustee is a research prototype for detecting malicious clients in Federated Learning (FL). The approach trains local client models in a federated setup, extracts interpretable surrogate decision trees from each local model using TRUSTEE, and compares the behavior of clients through prediction agreement over a shared reference set.

The repository contains the code used to run federated experiments, apply label-flipping attacks, save client checkpoints by round, extract TRUSTEE trees, compute agreement matrices, and generate qualitative figures such as surrogate trees and SHAP explanations.

## Main idea

FederatedTrustee assumes that benign clients tend to produce models with similar behavior, while clients affected by poisoning attacks tend to diverge from the group. Instead of comparing only model parameters or gradients, the method compares interpretable surrogate trees extracted from the trained local models.

The core workflow is:

1. Train local client models with Federated Learning.
2. Save client checkpoints for each federated round.
3. Extract a TRUSTEE surrogate tree for each client model.
4. Evaluate all trees on a common reference set.
5. Compute pairwise agreement between tree predictions.
6. Rank clients by mean agreement with the rest of the federation.

Lower mean agreement indicates more divergent behavior and, therefore, higher suspicion.

## Repository structure

    FederatedTrustee/
    ├── configs/                    # Experiment configuration files
    ├── data/                       # Dataset structure and instructions
    ├── scripts/                    # Reproducibility and analysis scripts
    ├── src/                        # Main source code
    ├── main.py                     # Federated training entry point
    ├── requirements.txt            # Python dependencies
    ├── run_pipeline.sh             # Training + analysis pipeline
    └── README.md

Important source modules:

    src/data/                       # Dataset loaders and preprocessing
    src/fl/                         # Flower client/server logic
    src/models/                     # MLP model and parameter utilities
    src/training/                   # Local training and evaluation routines
    src/attacks/                    # Label-flipping attack
    src/artifacts/                  # Run directories and checkpoint utilities
    src/analysis/                   # Agreement, TRUSTEE, and SHAP utilities

Main scripts:

    scripts/evaluate_checkpoint.py          # Evaluate one saved client checkpoint
    scripts/analyze_model_agreement.py      # Agreement between local neural models
    scripts/analyze_trustee_agreement.py    # Agreement between TRUSTEE surrogate trees
    scripts/export_trustee_trees.py         # Export surrogate trees as figures/text
    scripts/export_shap_comparison.py       # Generate SHAP local explanation figures
    scripts/build_client_round_csv.py       # Consolidate agreement results across runs

Paper/reproducibility runners:

    scripts/paper/run_smoke_experiment.py   # Small end-to-end pipeline smoke test
    scripts/paper/run_paper_experiments.py  # Paper experiment grid runner

## Installation

This project was developed with Python 3.10.

Create and activate a virtual environment:

    python -m venv .venv
    source .venv/bin/activate

Install dependencies:

    pip install -r requirements.txt

## Datasets

Raw datasets are not included in this repository.

Place the datasets in the following paths:

    data/raw/nsl_kdd/KDDTrain+.txt
    data/raw/nsl_kdd/KDDTest+.txt
    data/raw/fiveg_nidd/Encoded.csv

The expected data layout is documented in:

    data/README.md

Implemented binary tasks:

| Dataset | Class 0 | Class 1 |
|---|---|---|
| NSL-KDD | normal | DoS |
| 5G-NIDD | Benign | UDPFlood |

Other classes are filtered out by the dataset loaders.

## Configuration

Experiment settings are stored in YAML files:

    configs/nsl_kdd.yaml
    configs/fiveg_nidd.yaml

Each config defines:

- dataset path and task;
- number of clients and rounds;
- model architecture;
- training hyperparameters;
- label-flipping attack settings;
- TRUSTEE extraction parameters;
- output directory.

Example attack configuration:

    attack:
      enabled: true
      type: "label_flip"
      malicious_client_ids: [0, 1]
      source_label: 1
      target_label: 0
      flip_probability: 1.0

In the implemented label-flipping attack, samples from `source_label` are flipped to `target_label` with probability `flip_probability`.

## Running federated training

Run a federated experiment with:

    python main.py --config configs/fiveg_nidd.yaml

or:

    python main.py --config configs/nsl_kdd.yaml

Each run creates a timestamped directory under:

    runs/

The run directory contains:

    config.json
    run_summary.json
    checkpoints/

Client checkpoints are saved by round:

    runs/<RUN_ID>/checkpoints/round_001/client_000.pt
    runs/<RUN_ID>/checkpoints/round_002/client_000.pt
    runs/<RUN_ID>/checkpoints/round_003/client_000.pt

## Running the full pipeline

To train and analyze only the final round:

    bash run_pipeline.sh configs/fiveg_nidd.yaml final

To train and analyze all available federated rounds:

    bash run_pipeline.sh configs/fiveg_nidd.yaml all-rounds

The `all-rounds` mode is useful for checking whether the malicious-client signal appears early or only in the final round.

## Evaluating a checkpoint

Evaluate one saved client checkpoint on the global test set:

    python scripts/evaluate_checkpoint.py \
      --config configs/fiveg_nidd.yaml \
      --run-dir runs/<RUN_ID> \
      --round 3 \
      --client-id 0

## Agreement between TRUSTEE trees

Extract TRUSTEE surrogate trees for all clients and compute agreement between the pruned trees:

    python scripts/analyze_trustee_agreement.py \
      --config configs/fiveg_nidd.yaml \
      --run-dir runs/<RUN_ID> \
      --round 3

Analyze all available rounds:

    python scripts/analyze_trustee_agreement.py \
      --config configs/fiveg_nidd.yaml \
      --run-dir runs/<RUN_ID> \
      --all-rounds

Outputs:

    runs/<RUN_ID>/trustee/round_003/
    ├── trustee_summary.csv
    └── metadata.json

    runs/<RUN_ID>/agreement/trustee_pruned_trees/round_003/
    ├── agreement_matrix.csv
    ├── agreement_summary.csv
    └── metadata.json

The main ranking file is:

    agreement_summary.csv

It contains the mean agreement of each client with the others. Lower values indicate more divergent behavior.

## Exporting TRUSTEE trees

Export surrogate trees as PNG and text files:

    python scripts/export_trustee_trees.py \
      --config configs/fiveg_nidd.yaml \
      --run-dir runs/<RUN_ID> \
      --round 3 \
      --client-ids 0 1 \
      --tree-kind both

Export only pruned trees:

    python scripts/export_trustee_trees.py \
      --config configs/fiveg_nidd.yaml \
      --run-dir runs/<RUN_ID> \
      --round 3 \
      --client-ids 0 1 \
      --tree-kind pruned

Limit the plotted depth for visualization:

    python scripts/export_trustee_trees.py \
      --config configs/fiveg_nidd.yaml \
      --run-dir runs/<RUN_ID> \
      --round 3 \
      --client-ids 0 1 \
      --tree-kind pruned \
      --max-depth 3

Outputs:

    runs/<RUN_ID>/figures/trustee_trees/round_003/
    ├── client_000_pruned_tree.png
    ├── client_000_pruned_tree.txt
    ├── client_001_pruned_tree.png
    ├── client_001_pruned_tree.txt
    ├── exported_tree_summary.csv
    └── metadata.json

## SHAP comparison figures

SHAP is included as a local-explanation baseline for qualitative comparison.

Generate SHAP explanations for the same sample across one benign client and one or more malicious clients:

    python scripts/export_shap_comparison.py \
      --config configs/fiveg_nidd.yaml \
      --run-dir runs/<RUN_ID> \
      --round 3 \
      --benign-client 1 \
      --malicious-clients 0 \
      --target-class 1 \
      --explain predicted \
      --background-size 100 \
      --nsamples 200 \
      --top-k 10 \
      --min-abs-shap 1e-6

Explanation modes:

    target      # explains the same target class for all clients
    predicted   # explains the class predicted by each client

The `predicted` mode is useful for comparing why two clients made different decisions for the same input sample.

Outputs:

    runs/<RUN_ID>/figures/shap/round_003/sample_<ID>/explain_predicted/
    ├── client_000_malicious_*_shap_bar.png
    ├── client_000_malicious_*_shap_waterfall.png
    ├── client_000_malicious_*_shap_values.csv
    ├── client_001_benign_*_shap_bar.png
    ├── client_001_benign_*_shap_waterfall.png
    ├── client_001_benign_*_shap_values.csv
    ├── selected_sample_raw.csv
    ├── feature_name_mapping.csv
    ├── shap_comparison_summary.csv
    └── metadata.json

## Consolidating results

Build a consolidated CSV from multiple runs:

    python scripts/build_client_round_csv.py \
      --runs-dir runs \
      --output-csv client_round_agreements.csv

Filter runs by a substring:

    python scripts/build_client_round_csv.py \
      --runs-dir runs \
      --run-name-contains 20260518 \
      --output-csv client_round_agreements_20260518_all.csv

The resulting CSV includes:

- dataset;
- seed;
- flip probability;
- number of malicious clients;
- malicious client IDs;
- round;
- client ID;
- malicious/benign label;
- local-model mean agreement;
- TRUSTEE-tree mean agreement;
- run directory.

## Output structure

Generated outputs are saved under `runs/`, which is ignored by Git.

Typical structure:

    runs/<RUN_ID>/
    ├── config.json
    ├── run_summary.json
    ├── checkpoints/
    ├── agreement/
    ├── trustee/
    └── figures/

## Paper experiment runners

This repository also includes helper scripts under `scripts/paper/` to support reproducible execution of the experiments described in the paper.

These scripts do not change the core FederatedTrustee implementation. They only generate temporary experiment configurations and call the existing pipeline.

### Smoke test

Before running large experiment grids, run a small end-to-end smoke test:

    python scripts/paper/run_smoke_experiment.py

By default, this script runs a lightweight 5G-NIDD experiment with:

    dataset = fiveg_nidd
    num_clients = 2
    num_rounds = 1
    malicious_client_ids = [0]
    flip_probability = 1.0
    round_mode = all-rounds
    TRUSTEE num_iter = 2

The goal is only to verify that the full pipeline is working:

    federated training
    checkpoint saving
    local-model agreement
    TRUSTEE extraction
    TRUSTEE-tree agreement

The generated temporary config is saved under:

    tmp/paper_configs/

The `tmp/` directory is ignored by Git.

### Paper experiment grid

To run the experimental grids described in the paper, use:

    python scripts/paper/run_paper_experiments.py \
      --group exp1 \
      --round-mode all-rounds

The runner creates temporary YAML configs under:

    tmp/paper_configs/

and then calls:

    bash run_pipeline.sh <temporary_config> <round_mode>

Each generated config receives a descriptive `experiment_name`, which makes the output run directory easier to identify.

Example run directory:

    runs/paper_exp1_fiveg_nidd_p100_m2_seed001_<timestamp>/

### Available groups

The runner supports the following experiment groups:

    exp1
    exp2
    round-analysis
    all

#### Experiment 1: flip-probability variation

    --group exp1

Configuration grid:

    datasets = nsl_kdd, fiveg_nidd
    num_clients = 10
    malicious_client_ids = [0, 1]
    flip_probability = 1.0, 0.8, 0.6, 0.4
    seeds = 1, 7, 21, 42, 84

Example:

    python scripts/paper/run_paper_experiments.py \
      --group exp1 \
      --round-mode all-rounds

#### Experiment 2: number of malicious clients

    --group exp2

Configuration grid:

    datasets = nsl_kdd, fiveg_nidd
    num_clients = 10
    malicious_client_ids = [0] or [0, 1, 2, 3]
    flip_probability = 1.0, 0.8, 0.6
    seeds = 1, 21, 84


Example:

    python scripts/paper/run_paper_experiments.py \
      --group exp2 \
      --round-mode all-rounds

#### Experiment 3: round analysis

    --group round-analysis

Configuration grid:

    datasets = nsl_kdd, fiveg_nidd
    num_clients = 10
    malicious_client_ids = [0, 1]
    flip_probability = 1.0, 0.6
    seeds = 1, 7, 21, 42, 84

This experiment is a subset of Experiment 1. Therefore, if Experiment 1 was already executed with:

    --round-mode all-rounds

then separate training for the round-analysis group is not required. The same checkpoints from rounds 1, 2, and 3 can be reused.

The `round-analysis` group is useful when only this subset needs to be reproduced.

Example:

    python scripts/paper/run_paper_experiments.py \
      --group round-analysis \
      --round-mode all-rounds

### Useful options

Run only one dataset:

    python scripts/paper/run_paper_experiments.py \
      --group exp1 \
      --dataset fiveg_nidd \
      --round-mode all-rounds

Preview the experiment plan without executing anything:

    python scripts/paper/run_paper_experiments.py \
      --group exp1 \
      --dataset fiveg_nidd \
      --limit 2 \
      --dry-run

Run only the first experiment in a group:

    python scripts/paper/run_paper_experiments.py \
      --group exp1 \
      --dataset fiveg_nidd \
      --limit 1 \
      --round-mode all-rounds

Run all main grids:

    python scripts/paper/run_paper_experiments.py \
      --group all \
      --round-mode all-rounds

### Notes

The paper experiment runner trains each configuration only once.

When `--round-mode all-rounds` is used, the pipeline trains normally up to the configured final round and then analyzes all saved rounds. For a 3-round configuration, the same training run produces outputs for:

    round_001
    round_002
    round_003

This avoids duplicated training for the round-analysis experiment when the corresponding Experiment 1 runs already exist.

## Notes on paper reproduction

## Notes on paper reproduction

This repository contains the core implementation and scripts required to run the FederatedTrustee workflow and generate the main per-run outputs.

The paper experiment runner can reproduce the training and agreement-analysis grids. Aggregate paper-level analyses, such as AUROC, Precision@k, separation margins, confidence intervals, and final plotting scripts, require additional post-processing over the generated CSV files. These post-processing scripts can be added under `scripts/paper/` as the final experimental protocol is consolidated.

## Citation

If you use this repository, please cite the corresponding paper:

    @inproceedings{federatedtrustee2026,
      title     = {FederatedTrustee: Detectando e Explicando Ataques de Envenenamento em Aprendizado Federado},
      author    = {Anonymous},
      booktitle = {Proceedings of SBSeg},
      year      = {2026}
    }
