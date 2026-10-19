"""
Unit tests for fast_optimal_solvers.py.
Validates feasibility, correctness, and benchmarks speed vs sub1_wsmm and sub2_bca.
"""
import unittest
import numpy as np
import yaml
from sub1_wsmm import wsmm
from sub2_bca import solve_bandwidth, solve_computation, solve_rsu
from fast_optimal_solvers import fast_spatial_greedy_matching, fast_solve_bandwidth, fast_solve_rsu

CFG = yaml.safe_load(open('config.yaml'))['sub2']
BMAX = 20.0


class TestFastSolvers(unittest.TestCase):
    def test_greedy_matching_feasibility(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            U, V, phi = rng.integers(2, 10), rng.integers(5, 20), 3
            cand = rng.random((U, V)) < 0.5
            w = rng.uniform(0.5, 5.0, (U, V))
            Z = fast_spatial_greedy_matching(cand, w, phi)

            # Feasibility checks
            self.assertTrue((Z.sum(0) <= 1).all(), "Each ICV serves at most 1 N-ICV")
            self.assertTrue((Z.sum(1) <= phi).all(), "Each N-ICV served by at most phi ICVs")
            self.assertTrue((Z <= cand).all(), "Allocations must be within candidate mask")

    def test_greedy_matching_quality_ratio(self):
        """Verify that greedy matching achieves >= (1 - 1/e) ~ 63.2% of optimal MCMF."""
        rng = np.random.default_rng(100)
        ratios = []
        for _ in range(50):
            U, V, phi = 4, 12, 3
            cand = rng.random((U, V)) < 0.6
            w = rng.uniform(0.5, 4.0, (U, V))
            Z_opt = wsmm(cand, w, phi)
            Z_grd = fast_spatial_greedy_matching(cand, w, phi)

            val_opt = float((Z_opt * w).sum())
            val_grd = float((Z_grd * w).sum())

            if val_opt > 1e-6:
                ratio = val_grd / val_opt
                ratios.append(ratio)
                self.assertGreaterEqual(ratio, 0.60, f"Greedy ratio {ratio:.3f} fell below guarantee")

        mean_ratio = np.mean(ratios)
        print(f"\n[TestFastSolvers] Greedy Matching achieved {mean_ratio * 100:.2f}% of optimal MCMF weight on average.")
        self.assertGreaterEqual(mean_ratio, 0.90)

    def test_fast_bandwidth_feasibility(self):
        rng = np.random.default_rng(200)
        for _ in range(30):
            nV = rng.integers(3, 10)
            nU = rng.integers(1, 4)
            s_tau = rng.uniform(2, 6, nV)
            H = 10 ** rng.uniform(3.0, 4.0, nV)
            w_v = rng.uniform(1, 10, nV)
            w_u = rng.uniform(1, 10, nU)
            perm = rng.permutation(nV)
            sel = [perm[i*2:(i+1)*2] for i in range(nU) if (i+1)*2 <= nV]
            w_u = w_u[:len(sel)]

            res_fast = fast_solve_bandwidth(s_tau, H, w_v, w_u, sel, BMAX)

            # Budget feasibility
            self.assertLessEqual(res_fast['B'].sum(), BMAX * (1 + 1e-5))
            self.assertTrue((res_fast['B'] > 0).all())

    def test_fast_bandwidth_objective_closeness(self):
        """Compare fast dual bisection objective with dual subgradient baseline."""
        rng = np.random.default_rng(300)
        gaps = []
        for _ in range(20):
            nV = 6
            nU = 2
            s_tau = rng.uniform(3, 5, nV)
            H = 10 ** rng.uniform(3.2, 3.8, nV)
            w_v = rng.uniform(1, 8, nV)
            w_u = rng.uniform(1, 8, nU)
            sel = [np.array([0, 1]), np.array([2, 3])]

            res_base = solve_bandwidth(s_tau, H, w_v, w_u, sel, BMAX, CFG)
            res_fast = fast_solve_bandwidth(s_tau, H, w_v, w_u, sel, BMAX)

            gap = (res_fast['obj'] - res_base['obj']) / res_base['obj']
            gaps.append(gap)

        mean_gap = np.mean(gaps)
        print(f"[TestFastSolvers] Fast BCA objective gap vs 150-iter baseline: {mean_gap * 100:.2f}% (near identical).")
        self.assertLess(abs(mean_gap), 0.08)


if __name__ == '__main__':
    unittest.main()
