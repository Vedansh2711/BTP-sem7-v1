"""Subproblem 2: bandwidth and computation allocation for ONE RSU in ONE slot.
Units: B, H in MHz; s_tau in Mbit; workloads in Gcycles; F in GHz; times in seconds."""
import numpy as np

LN2 = np.log(2.0)


def _C(B, H):
    return np.log1p(H / B) / LN2


def t_tran(B, s_tau, H):
    return s_tau / (B * _C(B, H))


def _dt_dB(B, s_tau, H):
    """d/dB of s_tau/(B C(B)); paper Eq. (23a) bracket. Negative and increasing in B (convex)."""
    C = _C(B, H)
    return s_tau * (-C + H / (LN2 * (H + B))) / (B * C) ** 2


def _b_star(W, lam, s_tau, H, Bmax, bmin, iters):
    """Root of W * dt/dB + lam = 0 per vehicle by bisection; clipped to [bmin, Bmax]."""
    lo = np.full_like(H, bmin); hi = np.full_like(H, Bmax)
    g = lambda B: W * _dt_dB(B, s_tau, H) + lam
    at_hi, at_lo = g(hi) <= 0, g(lo) >= 0              # derivative increasing in B
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        neg = g(mid) < 0
        lo = np.where(neg, mid, lo); hi = np.where(neg, hi, mid)
    B = 0.5 * (lo + hi)
    return np.where(at_hi, Bmax, np.where(at_lo, bmin, B))


def project_simplex(y, z):
    """Euclidean projection of y onto {d >= 0, sum d = z}."""
    n = y.size
    u = np.sort(y)[::-1]; css = np.cumsum(u) - z
    k = np.arange(1, n + 1)
    rho = np.flatnonzero(u - css / k > 0)[-1]
    return np.maximum(y - css[rho] / (rho + 1), 0.0)


def solve_computation(a_v, a_u, F):
    """Exact KKT solution: f_i = F sqrt(a_i) / sum_j sqrt(a_j) over all ICVs and N-ICVs."""
    sv, su = np.sqrt(a_v), np.sqrt(a_u)
    Z = sv.sum() + su.sum()
    return F * sv / Z, F * su / Z


def _objective_bw(B, s_tau, H, w_v, w_u, sel):
    t = t_tran(B, s_tau, H)
    T = np.array([t[s].max() for s in sel]) if len(sel) else np.zeros(0)
    return float(w_v @ t + w_u @ T), t, T


def solve_bandwidth(s_tau, H, w_v, w_u, sel, Bmax, cfg, return_trace=False):
    """Dual (sub)gradient ascent, Alg. 2 (corrected). `sel[u]` = index array of ICVs serving u."""
    nV = s_tau.size
    bmin, iters = cfg['b_min_mhz'], cfg['bisect_iters']
    if nV == 1:
        B = np.array([Bmax]); obj, t, T = _objective_bw(B, s_tau, H, w_v, w_u, sel)
        return dict(B=B, t_tran=t, T_lin=T, obj=obj, iters=0, trace=[obj])
    lam = cfg['lam0']
    delta = [np.full(len(s), w_u[u] / len(s)) for u, s in enumerate(sel)]
    best, trace, prev, stall = None, [], np.inf, 0
    for j in range(1, cfg['max_iter'] + 1):
        W = w_v.copy()
        for u, s in enumerate(sel):
            W[s] += delta[u]
        B_raw = _b_star(W, lam, s_tau, H, Bmax, bmin, iters)
        t = t_tran(B_raw, s_tau, H)
        T_lin = np.array([t[s].max() for s in sel]) if len(sel) else np.zeros(0)
        # primal recovery: rescale to feasibility, evaluate the primal objective T_tcd
        B_p = B_raw * min(1.0, Bmax / B_raw.sum())
        obj, t_p, T_p = _objective_bw(B_p, s_tau, H, w_v, w_u, sel)
        trace.append(obj)
        if best is None or obj < best['obj']:
            best = dict(B=B_p.copy(), t_tran=t_p, T_lin=T_p, obj=obj, iters=j)
        step = 1.0 / np.sqrt(j)
        lam = max(lam + cfg['step_lambda'] * step * (B_raw.sum() - Bmax), 0.0)          # Eq. (26a)
        for u, s in enumerate(sel):                                                     # Eq. (26b) + projection
            delta[u] = project_simplex(delta[u] + cfg['step_delta'] * step * (t[s] - T_lin[u]), w_u[u])
        stall = stall + 1 if abs(obj - prev) <= cfg['eps'] * max(1.0, abs(obj)) else 0
        prev = obj
        if stall >= cfg['patience'] and j >= 20:
            break
    best['trace'] = trace
    best['iters_run'] = j
    return best


def solve_rsu(s_tau, H, w_v, cs_v, w_u, cc_u, sel, Bmax, F, cfg):
    """s_tau: s_v*tau (Mbit); cs_v: c_own*s_own*tau (Gcycles); cc_u: c_cp*tau (Gcycles).
    Returns B (MHz), f_own, f_cp, t_tran, T_lin and the full BCA objective."""
    bw = solve_bandwidth(s_tau, H, w_v, w_u, sel, Bmax, cfg)
    f_own, f_cp = solve_computation(w_v * cs_v, w_u * cc_u, F)
    comp = float((w_v * cs_v / f_own).sum() + (w_u * cc_u / f_cp).sum()) if f_cp.size or f_own.size else 0.0
    return dict(B=bw['B'], f_own=f_own, f_cp=f_cp, t_tran=bw['t_tran'], T_lin=bw['T_lin'],
                obj=bw['obj'] + comp, bw_obj=bw['obj'], iters=bw.get('iters', 0))
