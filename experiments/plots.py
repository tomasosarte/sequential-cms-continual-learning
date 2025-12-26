import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from experiments.metrics import forgetting_matrix

def plot_accuracy_evolution(acc_matrix_baseline_mlp, acc_matrix_seq_cms, max_cols=3):
    num_tasks = acc_matrix_baseline_mlp.shape[0]
    cols = min(max_cols, num_tasks)
    rows = math.ceil(num_tasks / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1) 

    x = np.arange(1, num_tasks + 1)
    for i, t in enumerate(range(num_tasks)):
        ax = axes[i]
        ax.plot(x, acc_matrix_baseline_mlp[t], marker="o")
        ax.plot(x, acc_matrix_seq_cms[t], marker="o")
        ax.set_title(f"Task {t+1}")
        ax.set_xlabel("Eval Task")
        ax.set_ylabel("Accuracy")
        
    fig.legend(["Baseline MLP", "Sequential CMS"], loc="lower center", ncol=2)
    fig.suptitle(f"Accuracy evolution per task", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def plot_task_forgetting(forget_matrix_baseline_mlp, forget_matrix_seq_cms, max_cols=3):
    num_tasks = forget_matrix_baseline_mlp.shape[0]
    cols = min(max_cols, num_tasks)
    rows = math.ceil(num_tasks / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1) 

    for i, t in enumerate(range(num_tasks)):
        x = list(range(t, num_tasks))
        ax = axes[i]
        ax.plot(x, forget_matrix_baseline_mlp[t, t:], marker="o")
        ax.plot(x, forget_matrix_seq_cms[t, t:], marker="o")
        ax.set_title(f"Task {t+1}")
        ax.set_xlabel("Task")
        ax.set_ylabel("Task Forgetting")
        
    fig.legend(["Baseline MLP", "Sequential CMS"], loc="lower center", ncol=2)
    fig.suptitle(f"Task Forgetting evolution per task", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def plot_cumulative_forgetting(forget_matrix_baseline_mlp, forget_matrix_seq_cms, max_cols=3):
    num_tasks = forget_matrix_baseline_mlp.shape[0]
    cols = min(max_cols, num_tasks)
    rows = math.ceil(num_tasks / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1) 

    for i, t in enumerate(range(num_tasks)):
        x = list(range(t, num_tasks))
        ax = axes[i]
        ax.plot(x, np.cumsum(forget_matrix_baseline_mlp[t, t:]), marker="o")
        ax.plot(x, np.cumsum(forget_matrix_seq_cms[t, t:]), marker="o")
        ax.set_title(f"Task {t+1}")
        ax.set_xlabel("Task")
        ax.set_ylabel("Task Forgetting")
        
    fig.legend(["Baseline MLP", "Sequential CMS"], loc="lower center", ncol=2)
    fig.suptitle(f"Cumulative Forgetting per task", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def plot_final_forgetting_per_task(F_base, F_cms, save_path=None, show=True):
    # F_* shape (T,T). We want last column (after final task).
    T = F_base.shape[0]
    fb = F_base[:, -1]
    fc = F_cms[:, -1]

    x = np.arange(1, T+1)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(x - 0.2, fb, width=0.4, label="Baseline")
    ax.bar(x + 0.2, fc, width=0.4, label="CMS")
    ax.set_xlabel("Task")
    ax.set_ylabel("Final forgetting")
    ax.set_title("Final forgetting per task")
    ax.legend()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig

def plot_avg_accuracy_over_time(A_base, A_cms, save_path=None, show=True):
    # A_* shape (T,T). Column t = after training task t.
    mean_base = A_base.mean(axis=0)
    mean_cms  = A_cms.mean(axis=0)

    x = np.arange(1, A_base.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(x, mean_base, marker="o", label="Baseline")
    ax.plot(x, mean_cms, marker="o", label="CMS")
    ax.set_xlabel("After training task")
    ax.set_ylabel("Average accuracy")
    ax.set_title("Average accuracy over time")
    ax.legend()
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return fig

def make_plots_from_results(results_dir="results/raw", out_dir="results/figures", show=False):
    results_dir = Path(results_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(results_dir.glob("seed_*.npz"))
    A_base_runs, A_cms_runs = [], []

    for f in files:
        d = np.load(f)
        A_base_runs.append(d["acc_baseline"])
        A_cms_runs.append(d["acc_cms"])

    A_base = np.mean(A_base_runs, axis=0)
    A_cms  = np.mean(A_cms_runs, axis=0)

    F_base = forgetting_matrix(A_base)
    F_cms  = forgetting_matrix(A_cms)

    plot_final_forgetting_per_task(F_base, F_cms, save_path=out_dir/"final_forgetting_per_task.png", show=show)
    plot_avg_accuracy_over_time(A_base, A_cms, save_path=out_dir/"avg_accuracy_over_time.png", show=show)
