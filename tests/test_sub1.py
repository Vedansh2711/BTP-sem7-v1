import itertools, unittest
import numpy as np
from scipy.optimize import linear_sum_assignment
from sub1_wsmm import wsmm, stable_matching, ga_select


def brute(cand, w, phi):
    U, V = cand.shape
    best = (-1, -1.0)
    for assign in itertools.product(range(U + 1), repeat=V):
        cnt = [0] * U; ok = True; tot = 0.0; card = 0
        for v, a in enumerate(assign):
            if a:
                if not cand[a - 1, v]: ok = False; break
                cnt[a - 1] += 1; tot += w[a - 1, v]; card += 1
        if ok and max(cnt, default=0) <= phi and (card, tot) > best:
            best = (card, tot)
    return best


def check_feasible(Z, cand, phi):
    assert (Z.sum(0) <= 1).all() and (Z.sum(1) <= phi).all() and (Z <= cand).all()


class T(unittest.TestCase):
    def test_wsmm_vs_bruteforce_and_scipy(self):
        rng = np.random.default_rng(1)
        for _ in range(60):
            U, V, phi = rng.integers(1, 4), rng.integers(1, 7), rng.integers(1, 4)
            cand = rng.random((U, V)) < 0.6
            w = rng.uniform(0.2, 2, (U, V)) * rng.uniform(2, 4, (U, V))
            Z = wsmm(cand, w, phi); check_feasible(Z, cand, phi)
            card, tot = brute(cand, w, phi)
            self.assertEqual(int(Z.sum()), card)
            self.assertAlmostEqual(float((Z * w).sum()), tot, places=8)
            # scipy cross-check: replicate N-ICVs phi times, big-M for cardinality
            if cand.any():
                M = 1e3
                W = np.full((V, U * phi), -1e6)
                for u in range(U):
                    for v in range(V):
                        if cand[u, v]:
                            W[v, u * phi:(u + 1) * phi] = M + w[u, v]
                r, c = linear_sum_assignment(-W)
                tot2 = sum(W[a, b] - M for a, b in zip(r, c) if W[a, b] > -1e5)
                self.assertAlmostEqual(tot, tot2, places=6)

    def test_flow_value_equals_networkx_maxflow(self):
        import networkx as nx
        rng = np.random.default_rng(2)
        for _ in range(30):
            U, V = 3, 10
            cand = rng.random((U, V)) < 0.4; w = rng.uniform(1, 8, (U, V))
            Z = wsmm(cand, w, 3); check_feasible(Z, cand, 3)
            G = nx.DiGraph()
            for v in range(V): G.add_edge('s', ('v', v), capacity=1)
            for u in range(U):
                G.add_edge(('u', u), 't', capacity=3)
                for v in np.flatnonzero(cand[u]): G.add_edge(('v', v), ('u', u), capacity=1)
            mf = nx.maximum_flow_value(G, 's', 't') if 's' in G and 't' in G else 0
            self.assertEqual(int(Z.sum()), mf)

    def test_dominance_over_max_cardinality(self):
        # WSMM is optimal among all MAX-CARDINALITY assignments => >= SM, GA when they are max-card;
        # on average it dominates (paper's claim). Instance-wise dominance is not guaranteed.
        rng = np.random.default_rng(3)
        a = np.zeros(3); n = 150
        for _ in range(n):
            cand = rng.random((3, 8)) < 0.45
            w = rng.uniform(0.2, 2, (3, 8)) * rng.uniform(2, 4, (3, 8))
            for i, Z in enumerate([wsmm(cand, w, 3), stable_matching(cand, w, 3),
                                   ga_select(cand, w, 3, 4.0, rng, pop=20, gens=20)]):
                check_feasible(Z, cand, 3); a[i] += (Z * w).sum()
        self.assertGreaterEqual(a[0], a[1]); self.assertGreaterEqual(a[0], a[2])

    def test_empty(self):
        self.assertEqual(wsmm(np.zeros((2, 3), bool), np.ones((2, 3)), 3).sum(), 0)


if __name__ == '__main__':
    unittest.main()
