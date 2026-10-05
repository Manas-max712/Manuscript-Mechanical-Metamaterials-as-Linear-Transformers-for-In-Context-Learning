import numpy as np
import torch
from scipy.linalg import solve


def regression_tasks(sigma, weight_covariance, count, context_length, seed):
    dimension = sigma.shape[0]
    factor_x = np.linalg.cholesky(sigma + 1e-10 * np.eye(dimension))
    factor_w = np.linalg.cholesky(weight_covariance + 1e-10 * np.eye(dimension))
    rng = np.random.RandomState(seed)
    for _ in range(count):
        weights = rng.randn(dimension) @ factor_w.T
        inputs = rng.randn(context_length, dimension) @ factor_x.T
        labels = inputs @ weights
        yield inputs, labels, weights


def encode_lattice_tasks(tasks, compatibility, output_indices):
    free_dofs, members = compatibility.shape
    dimension = len(output_indices)
    tokens, targets = [], []
    for inputs, labels, _ in tasks:
        count = inputs.shape[0]
        empirical_covariance = inputs.T @ inputs / count + 1e-10 * np.eye(dimension)
        force = inputs.T @ labels / count
        target = solve(empirical_covariance, force)
        token = np.zeros((free_dofs, members + 3), dtype=np.float64)
        token[:, :members] = compatibility
        for index in range(dimension):
            token[output_indices[index], members] = force[index]
        token[:, members + 2] = 1.0
        tokens.append(token)
        targets.append(target)
    return (torch.tensor(np.stack(tokens), dtype=torch.float64),
            torch.tensor(np.stack(targets), dtype=torch.float64))

