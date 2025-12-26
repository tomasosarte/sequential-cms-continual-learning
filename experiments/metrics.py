import numpy as np

def forgetting_matrix(acc_matrix: np.ndarray) -> np.ndarray:
    T = acc_matrix.shape[0]
    F = np.zeros_like(acc_matrix)

    for i in range(T):          # task
        best_so_far = acc_matrix[i, i]  # first time task i is learned
        for j in range(i, T):   # after task j
            best_so_far = max(best_so_far, acc_matrix[i, j])
            F[i, j] = best_so_far - acc_matrix[i, j]

    return F

def compute_avg_forgetting(acc_matrix: np.ndarray) -> float:
    T = acc_matrix.shape[0]
    forgets = []
    for i in range(T - 1):
        best = acc_matrix[i, i:].max()
        final = acc_matrix[i, T - 1]
        forgets.append(best - final)
    return float(np.mean(forgets))

def compute_avg_accuracy(acc_matrix: np.ndarray) -> float:
    T = acc_matrix.shape[0]
    return float(acc_matrix[:, T - 1].mean())