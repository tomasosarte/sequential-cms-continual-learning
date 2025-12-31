# Catastrophic Forgetting: CMS vs Baseline on Permuted-MNIST
This repository runs a  experiments testing the effect of principles in the Nested Learninn paper applied to continual learning setups.

## Repository layout

- `experiments/run_experiment.py`: main experiment runner (baseline vs CMS).
- `experiments/metrics.py`: average accuracy/forgetting utilities.
- `experiments/stats.py`: paired t-test helper.
- `experiments/plots.py`: plot generation from saved results.
- `datasets/permuted_mnist.py`: Permuted-MNIST task generator.
- `models/mlp.py`: MLP model definition.
- `optimizers/cms_optimizer_wrapper.py`: CMS optimizer wrapper.
- `results/`: output directory for raw runs, figures, and summary report.
- `notebooks/analysis.ipynb`: analysis notebook to summarize results.

## Setup

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pip-tools
make install
```

If you want to regenerate the lockfile from `requirements/requirements.in`:

```bash
make lock
```

## Run the experiment

Default settings run 5 seeds and save outputs under `results/`:

```bash
PYTHONPATH=. python3 experiments/run_experiment.py \
  --runs 10 \
  --seed 0 \
  --tasks 10 \
  --epochs 8 \
  --batch-size 128 \
  --hidden-dims 256 128 64 \
  --base-lr 5e-4 \
  --periods 4 2 1 \
  --verbose \
```

## Outputs

After a run, the following files are created:

- `results/raw/seed_*.npz`: per-run accuracy matrices (baseline and CMS).
- `results/summary.json`: aggregated metrics and run configuration.
- `results/figures/`: exported plots (accuracy over time, final forgetting per task).

## Analyze results

Open the notebook `analysis.ipynb` to unsderstand better the experiment and the analysis of the results.

## Reproducibility notes

- Each run uses `seed + k` for run index `k`.
- CUDA is used when available; otherwise CPU is used.
- Results and plots are deterministic for a fixed seed and environment.