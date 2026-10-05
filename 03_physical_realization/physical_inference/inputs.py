from pathlib import Path
import numpy as np

CASE_NAMES = ('Isotropic Gaussian', 'Diagonal Non-Isotropic', 'Rotated Non-Isotropic')
CASE_SLUGS = ('isotropic', 'diagonal', 'rotated')


def load_cases(operators, covariances=None):
    operators = Path(operators)
    by_name = {}
    with np.load(operators, allow_pickle=False) as data:
        for index in range(int(data['n_results'][0])):
            prefix = f'r{index}_'
            name = str(data[prefix+'name'])
            if name not in CASE_NAMES or name in by_name:
                raise ValueError(f'Unknown or duplicate family: {name}')
            by_name[name] = {key: data[prefix+key].astype(float)
                            for key in ('G','Sigma','covariance_factor','Lattice_whitened_G')}
    if set(by_name) != set(CASE_NAMES):
        raise ValueError('Supply an all-case results.npz from Folder 2 (--case all).')
    results = []
    for index, (name, slug) in enumerate(zip(CASE_NAMES, CASE_SLUGS)):
        item = by_name[name]
        g, sigma, factor = item['G'], item['Sigma'], item['covariance_factor']
        source = str(operators.resolve())
        if covariances is not None:
            with np.load(covariances, allow_pickle=False) as data:
                sigma = data[f'r{index}_Sigma'].astype(float)
                factor = data[f'r{index}_factor'].astype(float)
            source = str(Path(covariances).resolve())
        if any(a.shape != (5,5) or not np.isfinite(a).all() for a in (g,sigma,factor,item['Lattice_whitened_G'])):
            raise ValueError(f'Invalid 5-by-5 matrix for {name}')
        if not np.allclose(sigma, sigma.T, rtol=1e-12, atol=1e-12):
            raise ValueError(f'Asymmetric covariance for {name}')
        values, vectors = np.linalg.eigh(sigma)
        if np.any(values <= 0):
            raise ValueError(f'Covariance must be positive definite: {name}')
        if not np.allclose(factor@factor.T, sigma, rtol=1e-12, atol=1e-12):
            raise ValueError(f'Covariance/factor mismatch: {name}')
        h = (vectors*np.sqrt(values))@vectors.T
        if not np.allclose(h@g@h, item['Lattice_whitened_G'], rtol=1e-10, atol=1e-12):
            raise ValueError(f'Raw whitened operator/covariance mismatch: {name}')
        target = .5*(g+g.T)
        if np.linalg.eigvalsh(target).min() <= 0:
            raise ValueError(f'Symmetric target is not positive definite: {name}')
        results.append(dict(name=name,index=index,slug=slug,G=g,Sigma=sigma,factor=factor,
                            G_target=target,K_target=np.linalg.inv(target),covariance_source=source))
    return results
