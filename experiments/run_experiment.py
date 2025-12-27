import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
from pathlib import Path
import json

from datasets.permuted_mnist import PermutedMNIST
from optimizers.cms_optimizer_wrapper import CMSOptimizerWrapper, CMSGroup
from experiments.metrics import compute_avg_forgetting, compute_avg_accuracy
from models.mlp import MLP
from experiments.plots import make_plots_from_results
from experiments.stats import paired_ttest

def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
    return correct / len(loader.dataset)

def run_permuted_mnist_cms_experiment(
    T=9,
    epochs_per_task=8,
    batch_size=128,
    hidden_dims=(256, 128, 64),
    base_lr=5e-4,
    periods=(4, 2, 1),
    device=None,
    seed=None,
    verbose=False,
):
    # --- device ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- reproducibility (optional) ---
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # --- datasets ---
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    perms = [torch.randperm(784, generator=g) for _ in range(T)]
    tasks = [PermutedMNIST(p) for p in perms]

    loss_fn = nn.CrossEntropyLoss()

    # --- baseline ---
    baseline_mlp = MLP(784, list(hidden_dims), 10).to(device)
    baseline_optimizer = CMSOptimizerWrapper(
        groups=[CMSGroup(params=list(baseline_mlp.parameters()), lr=base_lr, chunk=1)],
        base_optim_cls=optim.Adam,
    )

    # --- CMS ---
    frequencies = [base_lr / p for p in periods]
    seq_cms = MLP(784, list(hidden_dims), 10).to(device)
    seq_cms_optimizer = CMSOptimizerWrapper(
        groups=[
            CMSGroup(params=list(seq_cms.slow.parameters()), lr=frequencies[0], chunk=periods[0]),
            CMSGroup(params=list(seq_cms.mid.parameters()),  lr=frequencies[1], chunk=periods[1]),
            CMSGroup(params=list(seq_cms.fast.parameters()), lr=frequencies[2], chunk=periods[2]),
        ],
        base_optim_cls=optim.Adam,
    )

    # --- results ---
    acc_matrix_baseline = np.zeros((T, T), dtype=np.float32)
    acc_matrix_cms      = np.zeros((T, T), dtype=np.float32)

    # --- train sequentially ---
    for current_t in range(T):
        train_loader = tasks[current_t].train_loader(batch_size=batch_size)

        baseline_mlp.train()
        seq_cms.train()

        for _ in range(epochs_per_task):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                # baseline
                baseline_optimizer.zero_grad()
                loss = loss_fn(baseline_mlp(x), y)
                loss.backward()
                baseline_optimizer.step()

                # cms
                seq_cms_optimizer.zero_grad()
                loss = loss_fn(seq_cms(x), y)
                loss.backward()
                seq_cms_optimizer.step()

        # --- evaluation + optional printing ---
        if verbose:
            print("\nEvaluation after training task", current_t + 1)
            print("-" * 55)
            print(f"{'Eval Task':<10} | {'Baseline MLP':<15} | {'Seq CMS':<15}")
            print("-" * 55)

        for i in range(T):
            acc_matrix_baseline[i, current_t] = evaluate(
                baseline_mlp, tasks[i].test_loader(), device
            )
            acc_matrix_cms[i, current_t] = evaluate(
                seq_cms, tasks[i].test_loader(), device
            )

            if verbose:
                print(
                    f"{i+1:<10} | "
                    f"{acc_matrix_baseline[i, current_t]*100:>6.2f}%{'':<7} | "
                    f"{acc_matrix_cms[i, current_t]*100:>6.2f}%"
                )

        if verbose:
            print("-" * 55)

    return acc_matrix_baseline, acc_matrix_cms

def run_n_times(n_runs=5, seed=None, verbose=False,  save_dir="results", **kwargs):
    print("\n" + "=" * 70)
    print("Running Permuted-MNIST Continual Learning Experiment")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print(f"Using device: {device}")

    save_dir = Path(save_dir)
    (save_dir / "raw").mkdir(parents=True, exist_ok=True)
    (save_dir / "figures").mkdir(parents=True, exist_ok=True)

    if verbose:
        print("\n" + "=" * 70)
        print("🚀 Running Permuted-MNIST Continual Learning Experiment")
        print(f"Total runs: {n_runs}")
        print("=" * 70)

    base_forgets, cms_forgets = [], []
    base_accs, cms_accs = [], []

    for k in range(n_runs):

        if verbose:
            print("\n" + "-" * 70)
            print(f"▶ Run {k + 1}/{n_runs}")
            print("-" * 70)

        run_seed = seed + k if seed is not None else None
        A_base, A_cms = run_permuted_mnist_cms_experiment(
            verbose=verbose,
            seed=run_seed,
            device=device,
            **kwargs
        )

        # save raw matrices per run
        np.savez(
            save_dir / "raw" / f"seed_{run_seed}.npz",
            acc_baseline=A_base,
            acc_cms=A_cms,
        )

        bF = compute_avg_forgetting(A_base)
        cF = compute_avg_forgetting(A_cms)
        bA = compute_avg_accuracy(A_base)
        cA = compute_avg_accuracy(A_cms)

        base_forgets.append(bF)
        cms_forgets.append(cF)
        base_accs.append(bA)
        cms_accs.append(cA)

        if verbose:
            print("\n📊 Run summary")
            print(f"  Baseline | Avg Acc: {bA*100:6.2f}% | Forgetting: {bF*100:6.2f}%")
            print(f"  CMS      | Avg Acc: {cA*100:6.2f}% | Forgetting: {cF*100:6.2f}%")

    report = {
        "baseline_avg_acc_mean":    float(np.mean(base_accs)),
        "baseline_avg_acc_std":     float(np.std(base_accs, ddof=1)) if n_runs > 1 else 0.0,
        "cms_avg_acc_mean":         float(np.mean(cms_accs)),
        "cms_avg_acc_std":          float(np.std(cms_accs, ddof=1)) if n_runs > 1 else 0.0,
        "baseline_forgetting_mean": float(np.mean(base_forgets)),
        "baseline_forgetting_std":  float(np.std(base_forgets, ddof=1)) if n_runs > 1 else 0.0,
        "cms_forgetting_mean":      float(np.mean(cms_forgets)),
        "cms_forgetting_std":       float(np.std(cms_forgets, ddof=1)) if n_runs > 1 else 0.0,
        "baseline_forgetting_all":  base_forgets,
        "cms_forgetting_all":       cms_forgets,
        "baseline_acc_all":         base_accs,
        "cms_acc_all":              cms_accs,
    }

    # Perofrm statistical tests
    if n_runs > 1:

        t_stat_acc, p_value_acc = paired_ttest(base_accs, cms_accs)
        t_stat_forg, p_value_forg = paired_ttest(base_forgets, cms_forgets)

        report.update({
            "ttest_avg_acc_t_stat": t_stat_acc,
            "ttest_avg_acc_p_value": p_value_acc,
            "ttest_forgetting_t_stat": t_stat_forg,
            "ttest_forgetting_p_value": p_value_forg,
        })

    # save summary
    summary = {
        "config": {"n_runs": n_runs, "seed0": seed, **kwargs},
        "report": report
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    make_plots_from_results(
        results_dir=save_dir / "raw",
        out_dir=save_dir / "figures",
        show=False
    )

    print("\n" + "=" * 70)
    print("✅ Final Continual Learning Report (mean ± std)")
    print("=" * 70)
    print(f"Average Accuracy  - Baseline: {report['baseline_avg_acc_mean']*100:6.2f}% ± {report['baseline_avg_acc_std']*100:5.2f}%")
    print(f"Average Accuracy  - CMS     : {report['cms_avg_acc_mean']*100:6.2f}% ± {report['cms_avg_acc_std']*100:5.2f}%")
    print(f"Avg Forgetting    - Baseline: {report['baseline_forgetting_mean']*100:6.2f}% ± {report['baseline_forgetting_std']*100:5.2f}%")
    print(f"Avg Forgetting    - CMS     : {report['cms_forgetting_mean']*100:6.2f}% ± {report['cms_forgetting_std']*100:5.2f}%")
    print("=" * 70)
    return report

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Permuted-MNIST Continual Learning experiment (Baseline vs CMS)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- general ---
    parser.add_argument("--runs", type=int, default=10, help="Number of independent experiment runs (different seeds)")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (run k uses seed+k)")
    parser.add_argument("--verbose", action="store_true", help="Print per-task accuracy tables")

    # --- experiment ---
    parser.add_argument("--tasks", type=int, default=10, help="Number of sequential Permuted-MNIST tasks")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs per task")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")

    # --- model ---
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64], help="MLP hidden layer sizes")

    # --- optimization ---
    parser.add_argument("--base-lr", type=float, default=5e-4, help="Base learning rate")
    parser.add_argument("--periods", type=int, nargs="+", default=[4, 2, 1], help="CMS update periods (slow → fast)")

    # --- output ---
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to store results and plots")

    args = parser.parse_args()

    run_n_times(
        n_runs=args.runs,
        seed=args.seed,
        verbose=args.verbose,
        save_dir=args.save_dir,
        T=args.tasks,
        epochs_per_task=args.epochs,
        batch_size=args.batch_size,
        hidden_dims=tuple(args.hidden_dims),
        base_lr=args.base_lr,
        periods=tuple(args.periods),
    )
