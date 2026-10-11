# =============================================================================
# Minimal Gaussian-process Bayesian optimisation (numpy / scipy only) for Phase 4e.
# Maximisation on a box. GP: Matern-5/2, per-dimension length scales, learned noise; inputs
# scaled to [0, 1]^d, outputs standardised; hyper-parameters by marginal likelihood (L-BFGS-B,
# restarts). Acquisition: expected improvement over the best posterior mean at the observed
# points, maximised over random candidates + local refinement.
# =============================================================================
import numpy as np
from scipy import optimize, stats
from scipy.stats import qmc


def _matern52(A, B, ls):
    d = np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) / ls) ** 2, 0).sum(2))
    s5 = np.sqrt(5.0) * d
    return (1 + s5 + s5 ** 2 / 3.0) * np.exp(-s5)


class GP:
    def __init__(self, X, y, seed=0):
        self.X = np.asarray(X, float)
        self.mu, self.sd = float(np.mean(y)), float(np.std(y) or 1.0)
        self.y = (np.asarray(y, float) - self.mu) / self.sd
        d = self.X.shape[1]
        rng = np.random.default_rng(seed)
        bounds = [(np.log(0.05), np.log(20.0))] + [(np.log(0.05), np.log(5.0))] * d + [(np.log(1e-4), np.log(2.0))]
        best = None
        for r in range(6):
            x0 = np.array([0.0] + [np.log(0.5)] * d + [np.log(0.1)]) if r == 0 else \
                np.array([rng.uniform(lo, hi) for lo, hi in bounds])
            res = optimize.minimize(self._nll, x0, method='L-BFGS-B', bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        self.theta = best.x
        self._fit(self.theta)

    def _parts(self, th):
        d = self.X.shape[1]
        return np.exp(th[0]), np.exp(th[1:1 + d]), np.exp(th[-1])

    def _nll(self, th):
        s2, ls, nz = self._parts(th)
        K = s2 * _matern52(self.X, self.X, ls) + (nz + 1e-8) * np.eye(len(self.X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        return 0.5 * self.y @ a + np.log(np.diag(L)).sum()

    def _fit(self, th):
        s2, ls, nz = self._parts(th)
        self.s2, self.ls, self.nz = s2, ls, nz
        K = s2 * _matern52(self.X, self.X, ls) + (nz + 1e-8) * np.eye(len(self.X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def predict(self, Z):
        Z = np.atleast_2d(Z)
        Ks = self.s2 * _matern52(Z, self.X, self.ls)
        m = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(self.s2 - (v ** 2).sum(0), 1e-12)
        return m * self.sd + self.mu, np.sqrt(var) * self.sd


def expected_improvement(gp, Z, best):
    m, s = gp.predict(Z)
    z = (m - best) / s
    return (m - best) * stats.norm.cdf(z) + s * stats.norm.pdf(z)


def propose(gp, d, seed, n_rand=5000, n_local=20):
    """Next point in [0, 1]^d maximising EI over the best posterior mean at observed points."""
    best = gp.predict(gp.X)[0].max()
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 1, (n_rand, d))
    ei = expected_improvement(gp, Z, best)
    starts = Z[np.argsort(-ei)[:n_local]]
    cand, val = starts[0], ei.max()
    for z0 in starts:
        r = optimize.minimize(lambda z: -expected_improvement(gp, z[None], best)[0], z0,
                              method='L-BFGS-B', bounds=[(0, 1)] * d)
        if -r.fun > val:
            cand, val = r.x, -r.fun
    return np.clip(cand, 0, 1)


def initial_design(d, n_sobol, seed, include_center=True):
    pts = qmc.Sobol(d, scramble=True, seed=seed).random(16 if n_sobol <= 16 else 2 ** int(np.ceil(np.log2(n_sobol))))[:n_sobol]
    return np.vstack([np.full(d, 0.5), pts]) if include_center else pts
