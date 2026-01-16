# Continuum Multi-Timescale Memory System (CMS) for Continual Learning

This repository provides a **reproducible PyTorch implementation** of the **Continuum Multi-Timescale Memory System (CMS)** introduced in *Nested Learning: The Illusion of Deep Learning Architectures* (NeurIPS 2025), together with **controlled continual-learning experiments** analyzing its impact on catastrophic forgetting.

CMS is implemented as a **multi-level learning system** in which different parameter blocks update at **distinct time scales**, forming a continuum of fast-to-slow memories.

---

## Reference

This repository implements and evaluates the **Continuum Multi-Timescale Memory System (CMS)** proposed in:

> Ali Behrouz et al., *Nested Learning: The Illusion of Deep Learning Architectures*, NeurIPS 2025.

### What this repository adds
- Minimal, transparent PyTorch implementation of CMS as a nested learning system  
- Controlled continual-learning evaluation on Permuted-MNIST  
- Quantitative analysis of catastrophic forgetting (accuracy matrices, final forgetting, statistical tests)  
- Clean separation between model, optimizer, experiments, and analysis  

---

## Method Overview

### Nested Learning perspective
Neural architectures and optimizers are viewed as **nested optimization processes** operating at different update frequencies.  
Each process acts as an **associative memory** that compresses information from its own temporal context.

### Continuum Multi-Timescale Memory System (CMS)
CMS distributes memory across multiple parameter blocks, each updated at a distinct frequency:

- **Fast blocks** adapt quickly to recent data  
- **Slow blocks** evolve gradually and retain long-term knowledge  

All blocks participate in the forward and backward pass at every step, but **parameter updates are applied only at scheduled intervals**.  
This enables fast adaptation while preserving stable long-term representations.

---

## Repository Structure

```
.
├── experiments/
│   ├── run_experiment.py         # Main experiment runner (Baseline vs CMS)
│   ├── metrics.py                # Accuracy & forgetting metrics
│   ├── stats.py                  # Paired statistical tests
│   └── plots.py                  # Figure generation
├── datasets/
│   └── permuted_mnist.py         # Continual Permuted-MNIST generator
├── models/
│   └── mlp.py                    # MLP backbone
├── optimizers/
│   └── cms_optimizer_wrapper.py  # CMS update scheduling
├── notebooks/
│   └── analysis.ipynb            # Result analysis & visualization
├── results/                      # Raw runs, figures, summaries
└── README.md
```

---

## Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pip-tools
make install
```

To regenerate the lockfile:

```bash
make lock
```

---

## Experiment Overview

The experiment compares a **baseline MLP** trained with standard SGD-style updates against the **same MLP wrapped with CMS update scheduling**. Both models are trained on a **sequential Permuted-MNIST** stream where each task applies a fixed pixel permutation, and tasks are learned one after another without revisiting earlier data.

### Protocol
- Train for a fixed number of epochs per task, then move to the next permutation.  
- Evaluate accuracy on all seen tasks after each task completes to build an accuracy matrix.  
- Repeat across multiple runs with different random seeds for statistical stability.  

### What is measured
- **Average accuracy over time** to capture overall performance.  
- **Final forgetting per task** to quantify how much each task degrades after learning later tasks.  
- **Paired statistical tests** comparing CMS vs baseline across runs.  

---

## Reproducing Experiments

```bash
PYTHONPATH=. python3 experiments/cms_architecture_experiment.py \
  --runs 10 \
  --seed 0 \
  --tasks 10 \
  --epochs 8 \
  --batch-size 128 \
  --hidden-dims 256 128 64 \
  --periods 3 2 1 \
  --base-lr 9e-05 \
  --baseline-lr 2.490716714676141e-05 \
  --verbose \
  --dir optimized_lr
```


```bash
PYTHONPATH=. python3 experiments/cms_architecture_hyperparameter_search.py \
  --trials 10 \
  --target cms \
  --runs 10 \
  --seed 0 \
  --tasks 10 \
  --epochs 10 \
  --batch-size 128 \
  --hidden-dims 128 64 32 \
  --periods 3 2 1 
```

```bash
PYTHONPATH=. python3 experiments/cms_optimizer_experiment.py \
  --runs 10 \
  --seed 0 \
  --verbose \
  --tasks 10 \
  --epochs 10 \
  --batch-size 64 \
  --hidden-dims 128 64 32 \
  --muon-lr 1e-4 \
  --M3-lr 1e-4 \
  --m1 0.9 \
  --m2 0.45 \
  --v1 0.9 \
  --alpha 0.5 \
  --frequency 2 \
  --dir optimizer_test
```

---

## Argument details

### General experiment control
- `--runs`: number of independent repetitions (seed + k per run)  
- `--seed`: base random seed  
- `--verbose`: print per-task accuracy tables  

### Continual learning setup
- `--tasks`: number of sequential Permuted-MNIST tasks  
- `--epochs`: training epochs per task  
- `--batch-size`: mini-batch size  

### Model
- `--hidden-dims`: hidden layer widths of the MLP  

### Optimization & CMS
- `--base-lr`: base learning rate  
- `--periods`: CMS update periods (slow → fast)

All parameter blocks receive gradients at every step, but optimizer updates are applied only at scheduled intervals.

### Output
- `--dir`: output directory under `results/`

---

## Outputs

Each run produces:
- `results/{dir}/raw/seed_*.npz`: accuracy matrices  
- `results/{dir}/summary.json`: aggregated metrics  
- `results/{dir}/figures/`: plots  

---

## Analysis

Open the notebook:

```
notebooks/analysis.ipynb
```

Includes:
- Average accuracy over time  
- Final forgetting per task  
- Paired statistical tests  

---

## Reproducibility Notes

- Deterministic for fixed seeds  
- CUDA used when available  
- No task boundaries or replay buffers  

---

## Status

Research code for reproducibility and analysis.  
Not intended for production use.

---

## Citation

```bibtex
@inproceedings{behrouz2025nested,
  title={Nested Learning: The Illusion of Deep Learning Architectures},
  author={Behrouz, Ali and others},
  booktitle={NeurIPS},
  year={2025}
}
```
