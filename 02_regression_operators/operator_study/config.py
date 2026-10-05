from dataclasses import dataclass

import numpy as np

DIMENSION = 5
LAYERS = 3
AHN_CONTEXT = 20
EXTRA_OFFSETS = ((2, 1), (1, 2), (-2, 1), (-1, 2))
CASE_NAMES = ("isotropic", "diagonal", "rotated")


@dataclass(frozen=True)
class TrainingSettings:
    lattice_epochs: int = 500
    lattice_context: int = 20000
    lattice_train_tasks: int = 4000
    lattice_validation_tasks: int = 500
    lattice_batch_size: int = 16
    lattice_lr: float = 1e-4
    ahn_iterations: int | None = None
    lattice_rng_ahn_iterations: int = 10000
    ahn_batch_size: int | None = None

    @classmethod
    def smoke(cls):
        return cls(lattice_epochs=2, lattice_context=32,
                   lattice_train_tasks=8, lattice_validation_tasks=4,
                   lattice_batch_size=4, ahn_iterations=2, ahn_batch_size=8,
                   lattice_rng_ahn_iterations=2)


def experiment_case(case):
    import torch

    common = {"ahn_var": 0.0001}
    if case == "isotropic":
        return dict(common, name="Isotropic Gaussian",
                    theorem="Thm 1: x~N(0,I), w*~N(0,I), multi-layer",
                    Sigma=np.eye(DIMENSION), w_prior_cov=np.eye(DIMENSION),
                    U=None, D=None, covariance_factor=np.eye(DIMENSION),
                    ahn_lr=0.001, ahn_clip=1000, ahn_batch_size=4000,
                    ahn_betas=(0.9, 0.9))
    if case == "diagonal":
        sigma = np.diag([4.0, 2.0, 1.0, 0.5, 0.25])
        u = torch.eye(DIMENSION, dtype=torch.float64)
        amplitudes = torch.diag(torch.sqrt(torch.tensor(
            [4.0, 2.0, 1.0, 0.5, 0.25], dtype=torch.float64)))
        name = "Diagonal Non-Isotropic"
        theorem = "Thm 3/4: x~N(0,Σ_diag), w*~N(0,Σ_diag⁻¹), multi-layer"
    elif case == "rotated":
        torch.manual_seed(0)
        draw = torch.FloatTensor(DIMENSION, DIMENSION).uniform_(-1, 1).to(torch.float64)
        u = torch.linalg.svd(draw)[0]
        amplitudes = torch.diag(torch.tensor([1., 1., 0.5, 0.25, 1.],
                                             dtype=torch.float64))
        factor = (u @ amplitudes).numpy()
        sigma = factor @ factor.T
        name = "Rotated Non-Isotropic"
        theorem = "Thm 3/4: x~N(0,Σ), w*~N(0,Σ⁻¹), multi-layer"
    else:
        raise ValueError(f"Unknown case {case!r}; choose from {CASE_NAMES}")
    return dict(common, name=name, theorem=theorem, Sigma=sigma,
                w_prior_cov=np.linalg.inv(sigma), U=u, D=amplitudes,
                covariance_factor=(u @ amplitudes).numpy(),
                ahn_lr=0.1, ahn_clip=0.01, ahn_batch_size=20000,
                ahn_betas=(0.99, 0.9))
