import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
from pathlib import Path
import json
import time

from datasets.permuted_mnist import PermutedMNIST
from experiments.metrics import compute_avg_forgetting, compute_avg_accuracy
from models.mlp import MLP
from experiments.stats import paired_ttest
from optimizers.glagm_optimizer import GLAGM


# =========================================================
# EVALUATION
# =========================================================
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
    return correct / len(loader.dataset)


# =========================================================
# SINGLE RUN
# =========================================================
def run_permuted_mnist_glagm_experiment(
    T=9,
    epochs_per_task=8,
    batch_size=128,
    hidden_dims=(256, 128, 64),
    lr=5e-4,
    mu=0.9,
    lam=0.1,
    device=None,
    seed=None,
    verbose=False,
):
    # --- device ---
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- reproducibility ---
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

    # --- baseline: Momentum SGD ---
    momentum_mlp = MLP(784, list(hidden_dims), 10).to(device)
    momentum_optimizer = optim.SGD(
        momentum_mlp.parameters(),
        lr=lr,
        momentum=mu
    )

    # --- GLAGM ---
    glagm_mlp = MLP(784, list(hidden_dims), 10).to(device)
    glagm_optimizer = GLAGM(
        glagm_mlp.parameters(),
        lr=lr,
        mu=mu,
        lam=lam
    )

    # --- results ---
    acc_matrix_momentum = np.zeros((T, T), dtype=np.float32)
    acc_matrix_glagm    = np.zeros((T, T), dtype=np.float32)
    opt_time_momentum = 0.0
    opt_time_glagm = 0.0

    # =====================================================
    # TRAIN SEQUENTIALLY (IDENTICAL STRUCTURE)
    # =====================================================
    for current_t in range(T):
        train_loader = tasks[current_t].train_loader(batch_size=batch_size)

        momentum_mlp.train()
        glagm_mlp.train()

        for _ in range(epochs_per_task):
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                # --- Momentum baseline ---
                start_time = time.time()
                momentum_optimizer.zero_grad()
                loss = loss_fn(momentum_mlp(x), y)
                loss.backward()
                momentum_optimizer.step()
                opt_time_momentum += time.time() - start_time

                # --- GLAGM ---
                start_time = time.time()
                glagm_optimizer.zero_grad()
                loss = loss_fn(glagm_mlp(x), y)
                loss.backward(create_graph=True)
                glagm_optimizer.step()
                opt_time_glagm += time.time() - start_time


        # =================================================
        # EVALUATION + VERBOSE PRINTS (UNCHANGED STYLE)
        # =================================================
        if verbose:
            print("\nEvaluation after training task", current_t + 1)
            print("-" * 55)
            print(f"{'Eval Task':<10} | {'Momentum MLP':<15} | {'GLAGM':<15}")
            print("-" * 55)

        for i in range(T):
            acc_matrix_momentum[i, current_t] = evaluate(
                momentum_mlp, tasks[i].test_loader(), device
            )
            acc_matrix_glagm[i, current_t] = evaluate(
                glagm_mlp, tasks[i].test_loader(), device
            )

            if verbose:
                print(
                    f"{i+1:<10} | "
                    f"{acc_matrix_momentum[i, current_t]*100:>6.2f}%{'':<7} | "
                    f"{acc_matrix_glagm[i, current_t]*100:>6.2f}%"
                )

        if verbose:
            print("-" * 55)

    return acc_matrix_momentum, acc_matrix_glagm, opt_time_momentum, opt_time_glagm


# =========================================================
# MULTI-RUN DRIVER (UNCHANGED LOGIC)
# =========================================================
def run_n_times(n_runs=5, seed=None, verbose=False, dir="test", **kwargs):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 70)
    print("Running Permuted-MNIST Continual Learning Experiment")
    print(f"Using device: {device}")
    print(f"Total runs: {n_runs}")
    print("=" * 70)

    dir = Path("results") / dir
    (dir / "raw").mkdir(parents=True, exist_ok=True)

    mom_forgets, glam_forgets = [], []
    mom_accs, glam_accs = [], []
    mom_opt_times, glam_opt_times = [], []

    for k in range(n_runs):

        if verbose:
            print("\n" + "-" * 70)
            print(f"▶ Run {k + 1}/{n_runs}")
            print("-" * 70)

        run_seed = seed + k if seed is not None else None
        A_mom, A_glam, T_mom, T_glam = run_permuted_mnist_glagm_experiment(
            verbose=verbose,
            seed=run_seed,
            device=device,
            **kwargs
        )

        np.savez(
            dir / "raw" / f"seed_{run_seed}.npz",
            acc_momentum=A_mom,
            acc_glagm=A_glam,
        )

        mF = compute_avg_forgetting(A_mom)
        gF = compute_avg_forgetting(A_glam)
        mA = compute_avg_accuracy(A_mom)
        gA = compute_avg_accuracy(A_glam)

        mom_forgets.append(mF)
        glam_forgets.append(gF)
        mom_accs.append(mA)
        glam_accs.append(gA)
        mom_opt_times.append(T_mom)
        glam_opt_times.append(T_glam)

        if verbose:
            print("\n📊 Run summary")
            print(f"  Momentum | Avg Acc: {mA*100:6.2f}% | Forgetting: {mF*100:6.2f}%")
            print(f"  GLAGM    | Avg Acc: {gA*100:6.2f}% | Forgetting: {gF*100:6.2f}%")
            print(f"  Optimization time (s) - Momentum: {T_mom:.2f}s | GLAGM: {T_glam:.2f}s")

    report = {
        "momentum_avg_acc_mean": float(np.mean(mom_accs)),
        "momentum_avg_acc_std":  float(np.std(mom_accs, ddof=1)) if n_runs > 1 else 0.0,
        "glagm_avg_acc_mean":    float(np.mean(glam_accs)),
        "glagm_avg_acc_std":     float(np.std(glam_accs, ddof=1)) if n_runs > 1 else 0.0,
        "momentum_forgetting_mean": float(np.mean(mom_forgets)),
        "glagm_forgetting_mean":    float(np.mean(glam_forgets)),
        "momentum_opt_time_mean":   float(np.mean(mom_opt_times)),
        "glagm_opt_time_mean":      float(np.mean(glam_opt_times)),
    }

    if n_runs > 1:
        t_acc, p_acc = paired_ttest(mom_accs, glam_accs)
        t_forg, p_forg = paired_ttest(mom_forgets, glam_forgets)
        report.update({
            "ttest_avg_acc_t_stat": t_acc,
            "ttest_avg_acc_p_value": p_acc,
            "ttest_forgetting_t_stat": t_forg,
            "ttest_forgetting_p_value": p_forg,
        })

    summary = {
        "config": {"n_runs": n_runs, "seed0": seed, **kwargs},
        "report": report
    }

    with open(dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("✅ Final Continual Learning Report (mean ± std)")
    print("=" * 70)
    print(f"Average Accuracy  - Momentum: {report['momentum_avg_acc_mean']*100:6.2f}% ± {report['momentum_avg_acc_std']*100:5.2f}%")
    print(f"Average Accuracy  - GLAGM   : {report['glagm_avg_acc_mean']*100:6.2f}% ± {report['glagm_avg_acc_std']*100:5.2f}%")
    print(f"Avg Forgetting    - Momentum: {report['momentum_forgetting_mean']*100:6.2f}%")
    print(f"Avg Forgetting    - GLAGM   : {report['glagm_forgetting_mean']*100:6.2f}%")
    print("=" * 70)

    return report


# =========================================================
# CLI (MATCHING ORIGINAL STYLE)
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Permuted-MNIST Continual Learning experiment (Momentum vs GLAGM)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- general ---
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")

    # --- experiment ---
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)

    # --- model ---
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64])

    # --- optimization ---
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--mu", type=float, default=0.9)
    parser.add_argument("--lam", type=float, default=0.1)

    # --- output ---
    parser.add_argument("--dir", type=str, default="results_glagm")

    args = parser.parse_args()

    run_n_times(
        n_runs=args.runs,
        seed=args.seed,
        verbose=args.verbose,
        dir=args.dir,
        T=args.tasks,
        epochs_per_task=args.epochs,
        batch_size=args.batch_size,
        hidden_dims=tuple(args.hidden_dims),
        lr=args.lr,
        mu=args.mu,
        lam=args.lam,
    )
