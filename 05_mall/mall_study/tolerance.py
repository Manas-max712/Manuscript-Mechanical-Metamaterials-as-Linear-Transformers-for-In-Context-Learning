from types import SimpleNamespace
import numpy as np
from scipy.linalg import solve as sp_solve
from .geometry import B_strain, ofi, int_idx, nf, m

LAT = SimpleNamespace(d=5, B=B_strain, ofi=ofi, int_idx=int_idx, nf=nf, m=m)


def sym(a):
    return 0.5 * (a + a.T)


def schur_from_D(lat, D):
    K = lat.B @ np.diag(D) @ lat.B.T
    K_oo = K[np.ix_(lat.ofi, lat.ofi)]
    K_oi = K[np.ix_(lat.ofi, lat.int_idx)]
    K_io = K[np.ix_(lat.int_idx, lat.ofi)]
    K_ii = K[np.ix_(lat.int_idx, lat.int_idx)]
    return K_oo - K_oi @ sp_solve(
        K_ii + 1e-12 * np.eye(len(lat.int_idx)), K_io)


def case_setup(case, d):
    if case == "identity":
        Sigma_x = np.eye(d)
        Sigma_w = np.eye(d)
    elif case == "diag":
        if d == 5:
            vals = np.array([4.0, 2.0, 1.0, 0.5, 0.25])
        else:
            vals = np.geomspace(4.0, 0.25, d)
        Sigma_x = np.diag(vals)
        Sigma_w = np.linalg.inv(Sigma_x)
    elif case == "rotated":
        rng = np.random.RandomState(0)
        A = rng.uniform(-1, 1, size=(d, d))
        U, _, _ = np.linalg.svd(A)
        if d == 5:
            vals = np.array([1.0, 1.0, 0.5, 0.25, 1.0])
        else:
            vals = np.geomspace(1.0, 0.25, d)
            vals[:min(2, d)] = 1.0
        UD = U @ np.diag(vals)
        Sigma_x = UD @ UD.T
        Sigma_w = np.linalg.inv(Sigma_x)
    else:
        raise ValueError(case)
    return (
        Sigma_x,
        Sigma_w,
        np.linalg.cholesky(Sigma_x + 1e-12 * np.eye(d)),
        np.linalg.cholesky(Sigma_w + 1e-12 * np.eye(d)),
    )


def make_tasks(seed, n_tasks, d, n_few, L_x, L_w):
    rng = np.random.RandomState(seed)
    tasks = []
    for _ in range(n_tasks):
        w = L_w @ rng.randn(d)
        X = rng.randn(n_few, d) @ L_x.T
        y = X @ w
        xq = L_x @ rng.randn(d)
        yq = float(xq @ w)
        tasks.append((X, y, xq, yq))
    return tasks


def stable_eta(lat, D, Sh, safety=0.95, eta_cap=1.0):
    K_s = schur_from_D(lat, D)
    G = np.linalg.inv(sym(K_s))
    evals, evecs = np.linalg.eigh(sym(G))
    G_sqrt = evecs @ np.diag(np.sqrt(np.clip(evals, 1e-14, None))) @ evecs.T
    A = G_sqrt @ Sh @ G_sqrt
    lam = max(float(np.linalg.eigvalsh(sym(A)).max()), 1e-14)
    return min(eta_cap, safety * 2.0 / lam)


def apply_force_at_output(lat, force_vec):
    f = np.zeros(lat.nf)
    for k in range(lat.d):
        f[lat.ofi[k]] = force_vec[k]
    return f


def forward_pgd(lat, D, X, y, n_stages):
    n = len(X)
    F = X.T @ y / n
    Sh = X.T @ X / n
    K = lat.B @ np.diag(D) @ lat.B.T
    eta = stable_eta(lat, D, Sh)
    w = np.zeros(lat.d)
    for s in range(n_stages):
        r = F - Sh @ w if s > 0 else F
        u = sp_solve(K + 1e-10 * np.eye(lat.nf), apply_force_at_output(lat, r))
        w += eta * u[lat.ofi]
    return w


def evaluate_mse(lat, D, tasks, n_stages):
    vals = []
    for X, y, xq, yq in tasks:
        w_hat = forward_pgd(lat, D, X, y, n_stages)
        vals.append((float(xq @ w_hat) - yq) ** 2)
    return float(np.mean(vals))


def stability_diagnostics(lat, D, tasks):
    K = lat.B @ np.diag(D) @ lat.B.T
    min_K_eigenvalue = float(np.linalg.eigvalsh(sym(K)).min())
    G_control = np.linalg.inv(sym(schur_from_D(lat, D)))
    evals, evecs = np.linalg.eigh(sym(G_control))
    G_half = (evecs * np.sqrt(np.clip(evals, 1e-14, None))) @ evecs.T
    Sh = np.stack([X.T @ X / len(X) for X, *_ in tasks])
    whitened = G_half @ Sh @ G_half
    maxima = np.linalg.eigvalsh(
        0.5 * (whitened + whitened.swapaxes(-1, -2)))[:, -1]
    eta = np.minimum(1.0, 1.9 / np.maximum(maxima, 1e-14))
    G_forward = sp_solve(K + 1e-10 * np.eye(lat.nf),
                         np.eye(lat.nf)[:, lat.ofi])[lat.ofi, :]
    iteration = np.eye(lat.d) - eta[:, None, None] * (G_forward @ Sh)
    max_rho = float(np.max(np.abs(np.linalg.eigvals(iteration))))
    return min_K_eigenvalue, max_rho


def oracle_mse(tasks, d, n_few, lam=0.0):
    vals = []
    for X, y, xq, yq in tasks:
        w = sp_solve(X.T @ X / n_few + lam * np.eye(d), X.T @ y / n_few)
        vals.append((float(xq @ w) - yq) ** 2)
    return float(np.mean(vals))


def fast(lat,D,tasks,n_stages):
 K=lat.B@np.diag(D)@lat.B.T
 G=sp_solve(K+1e-10*np.eye(lat.nf),np.eye(lat.nf)[:,lat.ofi])[lat.ofi,:]
 control=np.linalg.inv(sym(schur_from_D(lat,D)))
 vals,vec=np.linalg.eigh(sym(control));half=vec@np.diag(np.sqrt(np.clip(vals,1e-14,None)))@vec.T
 sh=np.stack([X.T@X/len(X) for X,y,xq,yq in tasks]);F=np.stack([X.T@y/len(X) for X,y,xq,yq in tasks])
 a=half@sh@half;lam=np.linalg.eigvalsh((a+a.swapaxes(-1,-2))/2)[:,-1];eta=np.minimum(1,1.9/np.maximum(lam,1e-14))
 w=np.zeros_like(F)
 for _ in range(n_stages):w+=eta[:,None]*((F-np.einsum('nij,nj->ni',sh,w))@G.T)
 err=(np.einsum('ni,ni->n',np.stack([xq for X,y,xq,yq in tasks]),w)-np.array([yq for X,y,xq,yq in tasks]))**2
 return float(err.mean())

def run_tolerance(results, sigmas=(0.2, 0.3), test_seed=20001234, n_test=1000):
    rows = []
    for case, result in results.items():
        D = np.asarray(result['D_best'], dtype=float)
        _, _, lx, lw = case_setup(case, 5)
        tasks = make_tasks(test_seed, n_test, 5, 20, lx, lw)
        clean = fast(LAT, D, tasks, 15)
        for sigma in sigmas:
            values = {stage: [] for stage in (15,50,100)}
            diagnostics = []
            for rep in range(10):
                rng = np.random.RandomState(1000*rep + int(10000*sigma))
                perturbed = D*np.exp(sigma*rng.randn(len(D)))
                diagnostics.append(stability_diagnostics(LAT, perturbed, tasks))
                for stage in values:
                    values[stage].append(fast(LAT, perturbed, tasks, stage))
            row = dict(case=case, sigma=sigma, clean15=clean, reps=10, tasks=n_test,
                       minimum_stiffness_eigenvalue=min(x[0] for x in diagnostics),
                       maximum_spectral_radius=max(x[1] for x in diagnostics))
            for stage, errors in values.items():
                row[f'noisy{stage}'] = float(np.mean(errors))
                row[f'std{stage}'] = float(np.std(errors))
                row[f'reduction{stage}_pct'] = 100*(1-np.mean(errors)/clean)
            rows.append(row)
            print(f'{case}: sigma={sigma}, perturbed MSE at 100 stages={row["noisy100"]:.6e}',flush=True)
    return rows
