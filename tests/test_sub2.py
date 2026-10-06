import unittest
import numpy as np
from scipy.optimize import minimize
from sub2_bca import solve_bandwidth, solve_computation, t_tran, project_simplex, solve_rsu
import yaml
CFG = yaml.safe_load(open('config.yaml'))['sub2']
BMAX = 20.0


def random_instance(rng, nV=None, nU=None):
    nV = nV or rng.integers(3, 9); nU = nU if nU is not None else rng.integers(0, 3)
    s_tau = rng.uniform(4, 6, nV); H = 10 ** rng.uniform(3.2, 3.8, nV); w_v = rng.integers(1, 8, nV).astype(float)
    perm = rng.permutation(nV); sel, c = [], 0
    for _ in range(nU):
        k = int(rng.integers(1, 4))
        if c + k > nV: break
        sel.append(perm[c:c + k]); c += k
    w_u = rng.integers(1, 8, len(sel)).astype(float)
    return s_tau, H, w_v, w_u, sel


def slsqp_ref(s_tau, H, w_v, w_u, sel, Bmax):
    nV, nU = s_tau.size, len(sel)
    f = lambda x: float(w_v @ t_tran(x[:nV], s_tau, H) + w_u @ x[nV:])
    cons = [{'type': 'ineq', 'fun': lambda x: Bmax - x[:nV].sum()}]
    for u, s in enumerate(sel):
        for v in s:
            cons.append({'type': 'ineq', 'fun': lambda x, u=u, v=v: x[nV + u] - t_tran(x[v], s_tau[v], H[v])})
    best = np.inf
    for k in range(3):
        B0 = np.full(nV, Bmax / nV) * (1 + 0.2 * k * np.random.default_rng(k).uniform(-1, 1, nV))
        B0 *= Bmax / B0.sum()
        x0 = np.concatenate([B0, [t_tran(B0, s_tau, H)[s].max() for s in sel]])
        r = minimize(f, x0, constraints=cons, bounds=[(1e-2, Bmax)] * nV + [(0, 10)] * nU, method='SLSQP',
                     options={'maxiter': 500, 'ftol': 1e-12})
        if r.success or True:
            # feasibility check
            ok = all(c['fun'](r.x) >= -1e-6 for c in cons)
            if ok: best = min(best, r.fun)
    return best


class T(unittest.TestCase):
    def test_closed_form_f_matches_slsqp(self):
        rng = np.random.default_rng(0)
        a_v, a_u = rng.uniform(0.1, 3, 5), rng.uniform(0.1, 3, 2); F = 4.0
        f_v, f_u = solve_computation(a_v, a_u, F)
        self.assertAlmostEqual(f_v.sum() + f_u.sum(), F)
        obj = lambda x: float((a_v / x[:5]).sum() + (a_u / x[5:]).sum())
        r = minimize(obj, np.full(7, F / 7), constraints=[{'type': 'eq', 'fun': lambda x: x.sum() - F}],
                     bounds=[(1e-3, F)] * 7, method='SLSQP', options={'ftol': 1e-14, 'maxiter': 500})
        self.assertLess(abs(obj(np.concatenate([f_v, f_u])) - r.fun) / r.fun, 1e-6)
        np.testing.assert_allclose(np.concatenate([f_v, f_u]), r.x, rtol=1e-3)

    def test_bandwidth_feasible_and_close_to_reference(self):
        rng = np.random.default_rng(5)
        gaps = []
        for _ in range(25):
            s_tau, H, w_v, w_u, sel = random_instance(rng)
            res = solve_bandwidth(s_tau, H, w_v, w_u, sel, BMAX, CFG)
            self.assertLessEqual(res['B'].sum(), BMAX * (1 + 1e-9)); self.assertTrue((res['B'] > 0).all())
            ref = slsqp_ref(s_tau, H, w_v, w_u, sel, BMAX)
            gaps.append((res['obj'] - ref) / ref)
        print('\nBCA gap vs SLSQP: mean %.4f max %.4f min %.4f' % (np.mean(gaps), np.max(gaps), np.min(gaps)))
        self.assertLess(np.mean(gaps), 0.02); self.assertLess(np.max(gaps), 0.05)

    def test_best_iterate_monotone(self):
        rng = np.random.default_rng(6)
        s_tau, H, w_v, w_u, sel = random_instance(rng, 6, 2)
        res = solve_bandwidth(s_tau, H, w_v, w_u, sel, BMAX, CFG)
        run_min = np.minimum.accumulate(res['trace'])
        self.assertTrue((np.diff(run_min) <= 1e-12).all())
        self.assertLessEqual(res['obj'], res['trace'][0] + 1e-12)

    def test_simplex_projection(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            y = rng.normal(0, 2, 4); z = rng.uniform(1, 5)
            d = project_simplex(y, z); self.assertAlmostEqual(d.sum(), z); self.assertTrue((d >= 0).all())

    def test_rsu_wrapper_feasible(self):
        rng = np.random.default_rng(8)
        s_tau, H, w_v, w_u, sel = random_instance(rng, 6, 2)
        r = solve_rsu(s_tau, H, w_v, rng.uniform(.1, .3, 6), w_u, rng.uniform(.5, 1.5, len(sel)), sel, BMAX, 4.0, CFG)
        self.assertLessEqual(r['B'].sum(), BMAX + 1e-9); self.assertAlmostEqual(r['f_own'].sum() + r['f_cp'].sum(), 4.0)


if __name__ == '__main__':
    unittest.main()
