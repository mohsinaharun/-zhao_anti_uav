# =============================================================================
# Phase 4 §2: equal-privacy-budget machinery. A schedule is a list of per-round
# noise multipliers sigma_t; each round = STEPS DP-SGD steps with Poisson rate q_c per
# client. Budget = the WORST client's epsilon (delta = 1e-5, Opacus RDP accountant),
# matched to the uniform schedule's.
# =============================================================================
import functools, warnings
import numpy as np
from opacus.accountants import RDPAccountant
from opacus.accountants.analysis import rdp as _rdp
warnings.filterwarnings('ignore', message='Optimal order is the largest alpha')
ALPHAS = RDPAccountant.DEFAULT_ALPHAS

DELTA = 1e-5
SIG_MIN, SIG_MAX = 0.3, 20.0
STEPS = 20


@functools.lru_cache(maxsize=None)
def _rdp_round(q, sigma, steps):
    """RDP curve (over ALPHAS) of one round: exactly what RDPAccountant sums per history entry."""
    return np.asarray(_rdp.compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=ALPHAS))


import os
_GRID = np.exp(np.linspace(np.log(SIG_MIN), np.log(SIG_MAX), 1501))
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache', 'rdp_grid')
_TAB = {}


def _table(q, steps):
    """Exact Opacus RDP curves on a dense log-spaced sigma grid for one sampling rate,
    cached on disk. Used only for SEARCH; every reported epsilon is computed exactly."""
    key = (round(q, 12), steps)
    if key not in _TAB:
        os.makedirs(_CACHE, exist_ok=True)
        f = os.path.join(_CACHE, 'q%.12f_s%d.npy' % key)
        if os.path.exists(f):
            _TAB[key] = np.load(f)
        else:
            _TAB[key] = np.log(np.stack([_rdp_round(float(q), float(s), steps) for s in _GRID]))
            np.save(f, _TAB[key])
    return _TAB[key]


def _rdp_fast(q, sigma, steps):
    T = _table(q, steps)                                   # (grid, alphas), log RDP
    x = np.log(np.clip(sigma, SIG_MIN, SIG_MAX))
    i = int(np.clip(np.searchsorted(np.log(_GRID), x) - 1, 0, len(_GRID) - 2))
    w = (x - np.log(_GRID[i])) / (np.log(_GRID[i + 1]) - np.log(_GRID[i]))
    return np.exp((1 - w) * T[i] + w * T[i + 1])


def eps_client(sigmas, q, steps=STEPS, exact=False):
    """Same math as RDPAccountant.get_epsilon (sum of compute_rdp over the history, then
    get_privacy_spent over DEFAULT_ALPHAS). exact=False interpolates the cached grid."""
    f = (lambda s: _rdp_round(float(q), float(s), steps)) if exact else (lambda s: _rdp_fast(q, s, steps))
    tot = sum(f(s) for s in sigmas)
    eps, _ = _rdp.get_privacy_spent(orders=ALPHAS, rdp=tot, delta=DELTA)
    return float(eps)


def eps_client_opacus(sigmas, q, steps=STEPS):
    acc = RDPAccountant()
    acc.history = [(float(s), q, steps) for s in sigmas]
    return float(acc.get_epsilon(DELTA))


def eps_worst(sigmas, qs, steps=STEPS, exact=False):
    return max(eps_client(sigmas, q, steps, exact) for q in qs)


def scale_to_budget(shape, target_eps, qs, steps=STEPS, tol=1e-3):
    """Multiply the relative noise shape by one global factor so the worst-client epsilon
    equals target_eps (relative tolerance tol). Returns (sigmas, eps) or (None, eps) if the
    budget cannot be met with every sigma_t inside [SIG_MIN, SIG_MAX]."""
    shape = np.asarray(shape, float)
    shape = shape / shape.mean()
    f = lambda c: eps_worst(np.clip(c * shape, SIG_MIN, SIG_MAX), qs, steps)
    lo, hi = SIG_MIN / shape.max(), SIG_MAX / shape.min()
    if f(hi) > target_eps or f(lo) < target_eps:      # eps decreases as c grows
        return None, None
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        e = f(mid)
        if abs(e - target_eps) / target_eps < tol:
            break
        lo, hi = (mid, hi) if e > target_eps else (lo, mid)
    sig = np.clip(mid * shape, SIG_MIN, SIG_MAX)
    return sig.tolist(), f(mid)


def sigma_for_round(history, share, rounds_left, target_eps, qs, steps=STEPS):
    """Online arms (§2/§4): choose this round's sigma. `history` = sigmas already used.
    The agent's `share` (0.25-4) scales this round's spend relative to spending the
    remaining budget evenly over the rounds left. Spend is measured as the uniform sigma
    that would exactly exhaust the budget over the rounds left (sigma_even); this round
    uses sigma_even / sqrt(share) (more share -> less noise), then is pushed up if needed
    so that finishing every later round at SIG_MAX stays within budget. The last round
    solves for the exact remainder."""
    def eps_with(sig_now, rest_sigma):
        return eps_worst(list(history) + [sig_now] + [rest_sigma] * (rounds_left - 1), qs, steps)

    def solve(fn, lo=SIG_MIN, hi=SIG_MAX):            # find sigma with fn(sigma) == target
        if fn(hi) > target_eps:
            return hi
        if fn(lo) <= target_eps:
            return lo
        for _ in range(60):
            mid = np.sqrt(lo * hi)
            lo, hi = (mid, hi) if fn(mid) > target_eps else (lo, mid)
        return hi                                        # the side that stays within budget

    if rounds_left == 1:
        return solve(lambda s: eps_with(s, s))
    sig_even = solve(lambda s: eps_with(s, s))
    sig = float(np.clip(sig_even / np.sqrt(share), SIG_MIN, SIG_MAX))
    if eps_with(sig, SIG_MAX) > target_eps:              # would make the budget infeasible
        sig = solve(lambda s: eps_with(s, SIG_MAX))
    return sig


if __name__ == '__main__':
    # G4a unit test: random shapes and random online share sequences all land on budget.
    rng = np.random.default_rng(0)
    qs = [64 / n for n in (1970, 1884, 1500, 1300, 6031, 6162, 2999, 3300)]   # real client sizes (step-5 frames)
    for _ in range(5):   # fast path == Opacus accountant
        sg = np.exp(rng.normal(0.7, 0.5, 12)); q = qs[int(rng.integers(8))]
        ex, op, fa = eps_client(sg, q, exact=True), eps_client_opacus(sg, q), eps_client(sg, q)
        assert abs(ex - op) < 1e-9 * max(1, ex), (ex, op)
        print('  exact %.6f == Opacus %.6f | grid-interpolated %.6f (rel err %.1e)' % (ex, op, fa, abs(fa - ex) / ex))
    # stored Phase 3 uniform runs: sigma 2, R=12 -> worst eps 1.86
    print('uniform sigma 2, R=12 worst eps (stored run says 1.86): %.4f' % eps_worst([2.0] * 12, qs, exact=True))
    for R, s_uni in ((12, 2.0), (25, 2.0), (12, 4.0)):
        target = eps_worst([s_uni] * R, qs, exact=True)
        worst = 0.0
        for _ in range(20):
            sig, e = scale_to_budget(np.exp(rng.normal(0, 0.5, R)), target, qs)
            if sig is not None:
                worst = max(worst, abs(eps_worst(sig, qs, exact=True) - target) / target)
        on = 0.0
        for _ in range(10):
            h = []
            for t in range(R):
                h.append(sigma_for_round(h, float(rng.uniform(0.25, 4)), R - t, target, qs))
            on = max(on, abs(eps_worst(h, qs, exact=True) - target) / target)
        print('R=%d uniform sigma %g: target eps %.4f | offline max rel err %.2e | online max rel err %.2e'
              % (R, s_uni, target, worst, on))
