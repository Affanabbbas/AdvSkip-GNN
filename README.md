# AdvSkip-GNN: Skip Connections and Adversarial Training for Graph Node Classification

A PyTorch research implementation exploring skip connections and feature-level adversarial training for graph neural networks (GNNs), evaluated on citation networks and several non-homophilous graph benchmarks.

> **Research note:** The accuracy and runtime tables below transcribe results previously reported for this project; they were **not rerun while preparing this repository archive**. See [Implementation and reproducibility notes](#implementation-and-reproducibility-notes) before interpreting them as results reproducible from every script in this snapshot.

## Overview

This project builds on a GCNII-style graph convolutional network and includes a trainable skip-connection component in `model.py`. The repository also includes an FGSM-style feature-perturbation routine in `train.py`.

The project investigates two practical questions:

- Can a skip-connection design support node classification in deeper GNNs?
- How does the model perform on several commonly used citation and non-homophilous graph datasets?

Classification accuracy alone does **not** establish adversarial robustness. Robustness claims require evaluation against a stated threat model and appropriate attacks.

## Benchmark datasets

The supplied source archive includes data and/or split files for:

| Dataset | Graph type / use in this project |
|---|---|
| Cora | Citation-network node classification |
| CiteSeer | Citation-network node classification |
| PubMed | Citation-network node classification |
| Chameleon | Non-homophilous graph benchmark |
| Wisconsin | Non-homophilous graph benchmark |
| Texas | Non-homophilous graph benchmark |
| Cornell | Non-homophilous graph benchmark |

The `data/` directory contains the Planetoid-format citation data. The `new_data/` directory contains graph files for Chameleon, Cornell, Texas, and Wisconsin. The `splits/` directory contains pre-generated train/validation/test splits.

## Results reported for the project

### Node-classification accuracy

The following values are the seven-dataset results reported for the project in the manuscript materials. The mean is calculated from the dataset-specific values shown here.

| Dataset | GCNII baseline (%) | AdvSkip-GNN (%) |
|---|---:|---:|
| Cora | 88.01 | **88.25** |
| CiteSeer | 76.99 | 76.99 |
| PubMed | **89.50** | 89.25 |
| Chameleon | 60.66 | **67.43** |
| Wisconsin | 77.06 | 77.06 |
| Texas | 77.84 | **79.46** |
| Cornell | 72.97 | **75.68** |
| **Mean** | **77.58** | **79.16** |

Based on these values, AdvSkip-GNN's mean accuracy is **1.58 percentage points** higher than the GCNII baseline. The result is dataset-dependent: the reported accuracy is higher on four datasets, equal on two, and lower on PubMed.

### Training time

The following times (seconds) are the project measurements recorded for the comparison. They should be interpreted as a controlled comparison only when hardware, software versions, data splits, epoch/early-stopping rules, timer scope, and number of runs are matched.

| Dataset | GCNII baseline (s) | AdvSkip-GNN (s) |
|---|---:|---:|
| Cora | 765.36 | **468.11** |
| CiteSeer | 229.85 | **67.54** |
| PubMed | 4759.30 | **2835.49** |
| Chameleon | 126.82 | **88.25** |
| Wisconsin | 183.48 | **111.57** |
| Texas | **211.46** | 446.39 |
| Cornell | **185.93** | 211.29 |
| **Mean** | **923.17** | **604.09** |

The reported mean training time is approximately **34.6% lower** for AdvSkip-GNN, although it is higher on Texas and Cornell. The GCNII timing values shown here are project measurements, not timing values claimed to have been reported by the original GCNII paper.

## Repository structure

```text
.
├── README.md
├── model.py                  # GCNII-style model and trainable skip connection
├── train.py                  # Semi-supervised training path; contains FGSM-style routine
├── full-supervised.py        # Full-supervised training path
├── process.py                # Data loading and preprocessing
├── utils.py                  # Graph utilities and dataset loading
├── full.sh                   # Full-supervised command examples
├── semi.sh                   # Semi-supervised command examples
├── data/                     # Citation-network data
├── new_data/                 # Chameleon, Cornell, Texas, Wisconsin graph files
├── splits/                   # Pre-generated data splits
├── PyG/                      # Additional PyTorch Geometric examples
└── pretrained/               # Checkpoint output directory
```

## Environment and dependencies

The original README listed this historical environment:

- Python 3.6.9
- PyTorch 1.3.1
- CUDA 10.1
- NetworkX 2.1
- scikit-learn

The Python scripts also use NumPy and SciPy. Versions listed above are historical and may not be suitable for a modern environment. The source archive does not include a pinned `requirements.txt` or environment file, so dependencies should be installed and tested in a dedicated environment before running experiments. The extra examples under `PyG/` may have their own dependency requirements.

## Running the supplied scripts

Run commands from the repository root so relative dataset paths resolve correctly.

### Full-supervised experiments

The included `full.sh` lists the project's full-supervised commands for the supported datasets. For example:

```bash
python -u full-supervised.py --data cora --layer 64 --alpha 0.2 --weight_decay 1e-4
python -u full-supervised.py --data citeseer --layer 64 --weight_decay 5e-6
python -u full-supervised.py --data pubmed --layer 64 --alpha 0.1 --weight_decay 5e-6
python -u full-supervised.py --data chameleon --layer 8 --lamda 1.5 --alpha 0.2 --weight_decay 5e-4
python -u full-supervised.py --data wisconsin --layer 16 --lamda 1 --weight_decay 5e-4
python -u full-supervised.py --data texas --layer 32 --lamda 1.5 --weight_decay 1e-4
python -u full-supervised.py --data cornell --layer 16 --lamda 1 --weight_decay 1e-3
```

To review all commands, see [`full.sh`](full.sh). This script trains across the predefined splits, so runtime and reported metrics depend on the selected configuration and environment.

### Semi-supervised experiments

The `semi.sh` file records the repository's semi-supervised example commands. For example:

```bash
python -u train.py --data cora --layer 64 --test
```

Check the implementation notes below before relying on the adversarial-training path in `train.py`.

## Implementation and reproducibility notes

A review of the supplied source archive identified a distinction between the training paths:

1. `full-supervised.py` performs the full-supervised training loop and does not call the adversarial-example generation routine in `train.py`.
2. `train.py` contains an FGSM-style routine that uses `features.grad` to construct perturbed features. In the supplied code, the input feature tensor is not explicitly marked as requiring gradients before this value is accessed. As written, this path appears likely to fail unless the input-gradient handling is corrected.
3. This archive has been repackaged with an updated README only. **The model and training code have not been changed or rerun as part of packaging.** Therefore, the results above should not be described as newly reproduced by this ZIP. Before using this repository as evidence that the adversarial-training method is fully executable, validate the gradient path and align each reported result with the exact script/configuration that produced it.
4. For a reproducible paper comparison, record the hardware/GPU, Python and library versions, seeds, data splits, stopping criteria, number of runs, and exactly which portion of execution was timed. Compare baseline and proposed model under the same conditions.
5. To support an adversarial-robustness claim, report evaluation under explicit attacks and perturbation budgets, not just clean test accuracy.

## References and source attribution

This package preserves the source files from the repository snapshot provided for this README update. The source code has not been rewritten as part of this packaging operation.

- **Source repository:** [ibrahimkhalilsj/AdvSkip-GNN](https://github.com/ibrahimkhalilsj/AdvSkip-GNN/tree/master)
- **GCNII reference implementation:** [chennnM/GCNII](https://github.com/chennnM/GCNII)
- **Geom-GCN dataset source:** [graphdml-uiuc-jlu/geom-gcn](https://github.com/graphdml-uiuc-jlu/geom-gcn)

**Current repository maintainer:** Affan Abbas

The source ZIP supplied for this packaging did not contain a `LICENSE` file. Before publishing or redistributing the copied source publicly, confirm permission from the original rights holder and comply with any applicable license. A public GitHub repository is not, by itself, permission to republish its code.

---

If you use this code in academic work, cite the relevant original GCNII and dataset papers and cite the AdvSkip-GNN manuscript using its final bibliographic details when available.
