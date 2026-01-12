import optuna
import numpy as np
from pathlib import Path

optuna.logging.set_verbosity(optuna.logging.INFO)

DB_PATH = Path(__file__).resolve().parent.parent / "optuna_lr.db"
STORAGE = f"sqlite:///{DB_PATH}"

from experiments.cms_architecture_experiment import run_n_times

def objective(trial, use_cms, use_baseline, base_kwargs):

    if not use_baseline and not use_cms:
        raise ValueError("At least one of use_baseline or use_cms must be True.")
    
    if use_baseline and use_cms:
        raise ValueError("At most one of use_baseline or use_cms must be True.")
    
    # --- sample learning rate ---
    # lr_grid = lr_grid = np.linspace(1e-6, 5e-4, 500).tolist()
    # base_lr = trial.suggest_categorical("base_lr", lr_grid)
    base_lr = trial.suggest_float("base_lr", 1e-6, 5e-4, log=True)

    # --- run experiment (silent, no files) ---
    report = run_n_times(
        base_lr=base_lr,
        verbose=False,
        save_results=False,
        fast_eval=True,
        use_cms=use_cms,
        use_baseline=use_baseline,
        **base_kwargs
    )

    # --- choose optimization target ---
    if use_cms:
        return report["cms_avg_acc_mean"]
    
    return report["baseline_avg_acc_mean"]

def run_optuna(n_trials, use_cms, use_baseline, base_kwargs):
    study_name = "lr_search_baseline" if use_baseline else "lr_search_cms"
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=STORAGE,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(
            trial,
            use_cms=use_cms,
            use_baseline=use_baseline,
            base_kwargs=base_kwargs,
        ),
        n_trials=n_trials,
    )

    print("\n🏆 Optuna finished")
    print("Best value :", study.best_value)
    print("Best params:", study.best_params)
    return study

def main():
    import argparse

    parser = argparse.ArgumentParser("Optuna LR search")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--target", choices=["cms", "baseline"], default="cms")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--periods", type=int, nargs="+", default=[4, 2, 1])
    parser.add_argument("--tasks", type=int, default=10)

    args = parser.parse_args()

    use_cms = args.target == "cms"
    use_baseline = args.target == "baseline"

    base_kwargs = dict(
        n_runs=args.runs,
        seed=args.seed,
        epochs_per_task=args.epochs,
        batch_size=args.batch_size,
        hidden_dims=tuple(args.hidden_dims),
        periods=tuple(args.periods),
        T=args.tasks
    )

    run_optuna(
        n_trials=args.trials,
        use_cms=use_cms,
        use_baseline=use_baseline,
        base_kwargs=base_kwargs,
    )

if __name__ == "__main__":
    main()

# Check for optuna progress
# import optuna

# study = optuna.load_study(
#     study_name="lr_search_baseline",
#     storage="sqlite:///optuna_lr.db",
# )

# print("Completed trials:", len(study.trials))
# print("Best so far:", study.best_value, study.best_params)
