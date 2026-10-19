# AdvSkip-GNN: Improving Node Classification Accuracy and Training Efficiency in Graph Neural Networks

**AdvSkip-GNN** is a PyTorch research implementation for graph node classification built on a GCNII-style architecture. It introduces a trainable skip-connection blend and evaluates node-classification accuracy and training time across seven standard graph benchmarks.

## Overview

AdvSkip-GNN builds on GCNII and adds a trainable connection between each layer's output and its previous hidden representation. Let $H^{(\ell+1)}_{\mathrm{GCNII}}$ denote the representation produced by the GCNII layer and $H^{(\ell)}$ the input representation to that layer. The additional blend is

$$
H^{(\ell+1)}_{\mathrm{AdvSkip}} = \beta H^{(\ell+1)}_{\mathrm{GCNII}} + (1-\beta)H^{(\ell)}.
$$

where $\beta$ is a trainable scalar parameter.

The project evaluates this design on citation networks and non-homophilous graph benchmarks using predefined train/validation/test splits.

## Datasets

| Dataset | Benchmark type | Data location |
| --- | --- | --- |
| Cora | Citation network | `data/` |
| CiteSeer | Citation network | `data/` |
| PubMed | Citation network | `data/` |
| Chameleon | Non-homophilous graph | `new_data/` |
| Wisconsin | Non-homophilous graph | `new_data/` |
| Texas | Non-homophilous graph | `new_data/` |
| Cornell | Non-homophilous graph | `new_data/` |

The predefined train/validation/test splits are stored in `splits/`. The full-supervised script evaluates ten splits for each dataset.

## Results

### Node-classification accuracy

The table compares the GCNII baseline with AdvSkip-GNN across the seven datasets. Changes are reported in **percentage points**.

| Dataset | GCNII baseline (%) | AdvSkip-GNN (%) | Change (points) |
| --- | ---: | ---: | ---: |
| Cora | 88.01 | 88.25 | +0.24 |
| CiteSeer | 76.99 | 76.99 | 0.00 |
| PubMed | 89.50 | 89.25 | -0.25 |
| Chameleon | 60.66 | 67.43 | +6.77 |
| Wisconsin | 77.06 | 77.06 | 0.00 |
| Texas | 77.84 | 79.46 | +1.62 |
| Cornell | 72.97 | 75.68 | +2.71 |
| **Mean across datasets** | **77.58** | **79.16** | **+1.58** |

Across the seven benchmarks, AdvSkip-GNN records a mean accuracy of **79.16%**, compared with **77.58%** for GCNII, an average improvement of **1.58 percentage points**. The largest gain in the table is on Chameleon, followed by Cornell and Texas.

### Training time

The following table reports training time in seconds and the relative change calculated from the listed times.

| Dataset | GCNII baseline (s) | AdvSkip-GNN (s) | Time change |
| --- | ---: | ---: | ---: |
| Cora | 765.36 | 468.11 | 38.8% lower |
| CiteSeer | 229.85 | 67.54 | 70.6% lower |
| PubMed | 4759.30 | 2835.49 | 40.4% lower |
| Chameleon | 126.82 | 88.25 | 30.4% lower |
| Wisconsin | 183.48 | 111.57 | 39.2% lower |
| Texas | 211.46 | 446.39 | 2.11× longer |
| Cornell | 185.93 | 211.29 | 13.6% longer |
| **Mean across datasets** | **923.17** | **604.09** | **34.6% lower mean time** |

The mean training time decreases from **923.17 seconds** for GCNII to **604.09 seconds** for AdvSkip-GNN, a **34.6% reduction** in the mean of the seven dataset-level times. The table includes per-dataset results to show how runtime varies across benchmarks.

## Running the experiments

Run commands from the repository root. The examples below reproduce the full-supervised configurations used for the reported dataset results.

```bash
python -u full-supervised.py --data cora --layer 64 --alpha 0.2 --weight_decay 1e-4
python -u full-supervised.py --data citeseer --layer 64 --weight_decay 5e-6
python -u full-supervised.py --data pubmed --layer 64 --alpha 0.1 --weight_decay 5e-6
python -u full-supervised.py --data chameleon --layer 8 --lamda 1.5 --alpha 0.2 --weight_decay 5e-4
python -u full-supervised.py --data wisconsin --layer 16 --lamda 1 --weight_decay 5e-4
python -u full-supervised.py --data texas --layer 32 --lamda 1.5 --weight_decay 1e-4
python -u full-supervised.py --data cornell --layer 16 --lamda 1 --weight_decay 1e-3
```

The complete set of full-supervised commands, including configurations that use `--variant`, is available in `full.sh`.

## Environment

The project dependencies are:

- Python 3.6.9
- PyTorch 1.3.1
- CUDA 10.1
- NetworkX 2.1
- scikit-learn
- NumPy
- SciPy

## Repository structure

```text
.
├── README.md               # Project overview and experiment results
├── model.py                # GCNII-style model with a trainable skip blend
├── train.py                # Semi-supervised training script
├── full-supervised.py      # Full-supervised training over predefined splits
├── process.py              # Data loading and preprocessing
├── utils.py                # Graph utilities and dataset loading
├── full.sh                 # Full-supervised experiment commands
├── semi.sh                 # Semi-supervised experiment commands
├── data/                   # Citation-network data
├── new_data/               # Chameleon, Cornell, Texas, and Wisconsin data
├── splits/                 # Predefined train/validation/test splits
├── PyG/                    # Additional PyTorch Geometric examples
└── pretrained/             # Model checkpoint output directory
```

## References and attribution

- **GCNII reference implementation:** [chennnM/GCNII](https://github.com/chennnM/GCNII)

- **Geom-GCN datasets:** [graphdml-uiuc-jlu/geom-gcn](https://github.com/graphdml-uiuc-jlu/geom-gcn)

Maintainer: **Muhammad Affan Abbas**

## Citation

If you use this work, cite the AdvSkip-GNN manuscript and the original GCNII and dataset papers relevant to your experiments. The project citation can be formatted as:

```bibtex
@misc{advskipgnn2026,
  title  = {AdvSkip-GNN: Improving Node Classification Accuracy and Training Efficiency in Graph Neural Networks},
  author = {Abbas, Muhammad Affan, Khalil, MD Ibrahim},
  year   = {2026}
}
```
