import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
from pathlib import Path
import json
import time
import optuna

from datasets.permuted_mnist import PermutedMNIST
from optimizers.cms_optimizer_wrapper import CMS, CMSGroup, adam_update
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

def run_permuted_mnist_cms_architecture_experiment(
    T=9,
    epochs_per_task=8,
    batch_size=128,
    hidden_dims=(256, 128, 64),
    base_lr=5e-4,
    baseline_lr=5e-4,
    periods=(4, 2, 1),
    alpha=0.0,
    device=None,
    seed=None,
    verbose=False,
    use_baseline=True,
    use_cms=True,
    fast_eval=False,
):  
    
    if not use_baseline and not use_cms:
        raise ValueError("At least one of use_baseline or use_cms must be True.")
    
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

    methods = {}
    baseline_mlp = MLP(784, list(hidden_dims), 10).to(device)
    baseline_optimizer = CMS(
        groups=[CMSGroup(params=list(baseline_mlp.parameters()), lr=baseline_lr, chunk=1)],
        update_fn=adam_update
    )
    methods["baseline"] = {
        "model": baseline_mlp,
        "optim": baseline_optimizer,
        "acc": np.zeros((T, T), dtype=np.float32),
        "time": 0.0
    }
    
    frequencies = [base_lr * (p ** alpha) for p in periods]
    seq_cms = MLP(784, list(hidden_dims), 10).to(device)

    assert len(periods) == len(seq_cms.levels), "periods must match number of MLP levels"

    cms_groups = [
        CMSGroup(
            params=list(level.parameters()),
            lr=frequencies[i],
            chunk=periods[i],
        )
        for i, level in enumerate(seq_cms.levels)
    ]
    seq_cms_optimizer = CMS(
        groups=cms_groups,
        update_fn=adam_update
    )
    methods["cms"] = {
        "model": seq_cms,
        "optim": seq_cms_optimizer,
        "acc": np.zeros((T, T), dtype=np.float32),
        "time": 0.0
    }

    train_loaders = [t.train_loader(batch_size) for t in tasks]
    test_loaders  = [t.test_loader() for t in tasks]

    # --- train sequentially ---
    for current_t in range(T):
        
        for m in methods.values():
            m["model"].train()

        for _ in range(epochs_per_task):
            for x, y in train_loaders[current_t]:
                x, y = x.to(device), y.to(device)
                for m in methods.values():
                    start = time.time()
                    m["optim"].zero_grad()
                    loss = loss_fn(m["model"](x), y)
                    loss.backward()
                    m["optim"].step()
                    m["time"] += time.time() - start

        # --- Evaluation ---
        if not fast_eval or current_t == T - 1:
            for m in methods.values():
                m["model"].eval()

            for i in range(T):
                for m in methods.values():
                    m["acc"][i, current_t] = evaluate(
                        m["model"], test_loaders[i], device
                    )

        if verbose:
            print(f"\nEvaluation after training task {current_t + 1}")
            print("-" * 55)
            print(f"{'Eval Task':<10} | " + " | ".join(f"{name.upper():<15}" for name in methods))
            for i in range(T):
                row = f"{i+1:<10}"
                for name in methods:
                    acc = methods[name]["acc"][i, current_t]
                    row += f" | {acc*100:>6.2f}%{'':<8}"
                print(row)
            print("-" * 55)

    return methods

def run_n_times(
    n_runs=5, 
    seed=None, 
    verbose=False, 
    dir="test",
    use_cms=True,
    use_baseline=True,
    save_results=True,
    fast_eval=False,
    trial=None,
    **kwargs
    ):

    if not use_baseline and not use_cms:
        raise ValueError("At least one of use_baseline or use_cms must be True.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if verbose:
        print("\n" + "=" * 70)
        print("Running Sequential CMS on architecture Experiment")
        print(f"Using device: {device}")
        print(f"Total runs: {n_runs}")
        print("=" * 70)

    dir = Path("results") / dir
    if save_results:
        (dir / "raw").mkdir(parents=True, exist_ok=True)
        (dir / "figures").mkdir(parents=True, exist_ok=True)

    base_forgets, cms_forgets = [], []
    base_accs, cms_accs = [], []
    base_opt_times, cms_opt_times = [], []

    for k in range(n_runs):

        if verbose:
            print("\n" + "-" * 70)
            print(f"▶ Run {k + 1}/{n_runs}")
            print("-" * 70)

        run_seed = seed + k if seed is not None else None
        methods = run_permuted_mnist_cms_architecture_experiment(
            verbose=verbose,
            seed=run_seed,
            device=device,
            use_baseline=use_baseline,
            use_cms=use_cms,
            fast_eval=fast_eval,
            **kwargs
        )

        # save raw matrices per run
        if save_results:
            save_dict = {}
            if use_baseline:
                save_dict["acc_baseline"] = methods["baseline"]["acc"]
            if use_cms:
                save_dict["acc_cms"] = methods["cms"]["acc"]

            np.savez(dir / "raw" / f"seed_{run_seed}.npz", **save_dict)

        if use_baseline:
            bF = compute_avg_forgetting(methods["baseline"]["acc"])
            bA = compute_avg_accuracy(methods["baseline"]["acc"])
            base_forgets.append(bF)
            base_accs.append(bA)
            base_opt_times.append(methods["baseline"]["time"])

        if use_cms:
            cF = compute_avg_forgetting(methods["cms"]["acc"])
            cA = compute_avg_accuracy(methods["cms"]["acc"])
            cms_forgets.append(cF)
            cms_accs.append(cA)
            cms_opt_times.append(methods["cms"]["time"])
        
        # --- PRUNING POINT ---
        if trial is not None:
            if use_cms:
                current_mean = float(np.mean(cms_accs))
            else:
                current_mean = float(np.mean(base_accs))

            trial.report(current_mean, step=k)

            if trial.should_prune():
                raise optuna.TrialPruned()

        if verbose:
            print("\n📊 Run summary")
            if use_baseline:
                print(f"  Baseline | Avg Acc: {bA*100:6.2f}% | Forgetting: {bF*100:6.2f}%")
            if use_cms:
                print(f"  CMS      | Avg Acc: {cA*100:6.2f}% | Forgetting: {cF*100:6.2f}%")

            times = []
            if use_baseline:
                times.append(f"Baseline: {methods['baseline']['time']:.2f}s")
            if use_cms:
                times.append(f"CMS: {methods['cms']['time']:.2f}s")
            print("Optimization time (s) - " + " | ".join(times))

    report = {}

    if use_baseline:
        report.update({
            "baseline_avg_acc_mean":    float(np.mean(base_accs)),
            "baseline_avg_acc_std":     float(np.std(base_accs, ddof=1)) if n_runs > 1 else 0.0,
            "baseline_forgetting_mean": float(np.mean(base_forgets)),
            "baseline_forgetting_std":  float(np.std(base_forgets, ddof=1)) if n_runs > 1 else 0.0,
            "baseline_forgetting_all":  base_forgets,
            "baseline_acc_all":         base_accs,
            "baseline_opt_time_mean":   float(np.mean(base_opt_times)),
        })

    if use_cms:
        report.update({
            "cms_avg_acc_mean":         float(np.mean(cms_accs)),
            "cms_avg_acc_std":          float(np.std(cms_accs, ddof=1)) if n_runs > 1 else 0.0,
            "cms_forgetting_mean":      float(np.mean(cms_forgets)),
            "cms_forgetting_std":       float(np.std(cms_forgets, ddof=1)) if n_runs > 1 else 0.0,
            "cms_forgetting_all":       cms_forgets,
            "cms_acc_all":              cms_accs,
            "cms_opt_time_mean":        float(np.mean(cms_opt_times)),
        })

    # Perofrm statistical tests
    if n_runs > 1 and use_cms and use_baseline:

        t_stat_acc, p_value_acc = paired_ttest(base_accs, cms_accs)
        t_stat_forg, p_value_forg = paired_ttest(base_forgets, cms_forgets)

        report.update({
            "ttest_avg_acc_t_stat": t_stat_acc,
            "ttest_avg_acc_p_value": p_value_acc,
            "ttest_forgetting_t_stat": t_stat_forg,
            "ttest_forgetting_p_value": p_value_forg,
        })

        # Make a print

    # save summary
    summary = {
        "config": {"n_runs": n_runs, "seed0": seed, **kwargs},
        "report": report
    }

    if save_results:
        with open(dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        make_plots_from_results(
            results_dir=dir / "raw",
            out_dir=dir / "figures",
            show=False
        )

    if verbose:
        print("\n" + "=" * 70)
        print("✅ Final Continual Learning Report (mean ± std)")
        print("=" * 70)
        if use_baseline:
            print(f"Average Accuracy  - Baseline: {report['baseline_avg_acc_mean']*100:6.2f}% ± {report['baseline_avg_acc_std']*100:5.2f}%")
            print(f"Avg Forgetting    - Baseline: {report['baseline_forgetting_mean']*100:6.2f}% ± {report['baseline_forgetting_std']*100:5.2f}%")

        if use_cms:
            print(f"Average Accuracy  - CMS     : {report['cms_avg_acc_mean']*100:6.2f}% ± {report['cms_avg_acc_std']*100:5.2f}%")
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
    parser.add_argument("--verbose", action="store_true", help="Print per-task accuracy tables.")

    # --- experiment ---
    parser.add_argument("--tasks", type=int, default=10, help="Number of sequential Permuted-MNIST tasks")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs per task")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size")

    # --- model ---
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64], help="MLP hidden layer sizes")

    # --- optimization ---
    parser.add_argument("--base-lr", type=float, default=5e-4, help="Base learning rate")
    parser.add_argument("--periods", type=int, nargs="+", default=[4, 2, 1], help="CMS update periods (slow → fast)")
    parser.add_argument("--alpha", type=float, default=0.0, help="LR scaling exponent: lr = base_lr * period^alpha")
    parser.add_argument("--baseline-lr", type=float, default=5e-4, help="Baseline learning rate")

    # --- output ---
    parser.add_argument("--dir", type=str, default="results", help="Directory to store results and plots")

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
        base_lr=args.base_lr,
        periods=tuple(args.periods),
        alpha=args.alpha,
        baseline_lr=args.baseline_lr,
    )
