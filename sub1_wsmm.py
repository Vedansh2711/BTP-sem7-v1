"""Subproblem 1: cooperative ICV selection.
Inputs everywhere: cand (U,V) bool candidate mask, w (U,V) weights = q * s_unit * tau  [Mb].
Output: Z (U,V) int8 with Z[u,v]=1 iff ICV v serves N-ICV u."""
import numpy as np


# ----------------------------------------------------------------------------- WSMM (MCMF)
class MCMF:
    """Min-cost max-flow by successive shortest augmenting paths (Bellman-Ford on the residual
    graph). Cost and flow are separate arrays (ASSUMPTIONS #2)."""

    def __init__(self, n):
        self.n = n
        self.to, self.cap, self.cost, self.flow = [], [], [], []
        self.adj = [[] for _ in range(n)]

    def add_edge(self, u, v, cap, cost):
        self.adj[u].append(len(self.to)); self.to.append(v); self.cap.append(cap); self.cost.append(cost); self.flow.append(0)
        self.adj[v].append(len(self.to)); self.to.append(u); self.cap.append(0); self.cost.append(-cost); self.flow.append(0)
        return len(self.to) - 2

    def solve(self, s, t, tol=1e-12):
        total_flow, total_cost = 0, 0.0
        while True:
            dist = [np.inf] * self.n; dist[s] = 0.0
            prev = [-1] * self.n
            for _ in range(self.n - 1):
                changed = False
                for u in range(self.n):
                    if dist[u] == np.inf:
                        continue
                    for e in self.adj[u]:
                        if self.cap[e] - self.flow[e] > 0 and dist[u] + self.cost[e] < dist[self.to[e]] - tol:
                            dist[self.to[e]] = dist[u] + self.cost[e]; prev[self.to[e]] = e; changed = True
                if not changed:
                    break
            if dist[t] == np.inf:
                return total_flow, total_cost
            f, v = np.inf, t
            while v != s:
                e = prev[v]; f = min(f, self.cap[e] - self.flow[e]); v = self.to[e ^ 1]
            v = t
            while v != s:
                e = prev[v]; self.flow[e] += f; self.flow[e ^ 1] -= f; v = self.to[e ^ 1]
            total_flow += f; total_cost += f * dist[t]


def wsmm(cand, w, phi_max):
    U, V = cand.shape
    Z = np.zeros((U, V), np.int8)
    if U == 0 or V == 0 or not cand.any():
        return Z
    wmax = w[cand].max()
    g = MCMF(V + U + 2)
    vs, vt = V + U, V + U + 1
    for v in range(V):
        g.add_edge(vs, v, 1, 0.0)
    eid = {}
    for u in range(U):
        for v in np.flatnonzero(cand[u]):
            eid[(u, v)] = g.add_edge(v, V + u, 1, wmax - w[u, v])      # cost >= 0
        g.add_edge(V + u, vt, phi_max, 0.0)
    g.solve(vs, vt)
    for (u, v), e in eid.items():
        if g.flow[e] == 1:
            Z[u, v] = 1
    return Z


# ----------------------------------------------------------------------------- SM baseline
def stable_matching(cand, w, phi_max, icv_score=None):
    """Hospital-residents Gale-Shapley, ICVs propose (quota 1), N-ICVs have quota phi_max.
    N-ICVs rank ICVs by w (= q s_unit tau). ICVs rank N-ICVs by icv_score (default: w, the literal spec;
    NOTE this makes SM ~ WSMM, see ASSUMPTIONS #A3; pass e.g. -distance for the non-degenerate variant)."""
    U, V = cand.shape
    Z = np.zeros((U, V), np.int8)
    sc = w if icv_score is None else icv_score
    prefs = [sorted(np.flatnonzero(cand[:, v]), key=lambda u: -sc[u, v]) for v in range(V)]
    nxt = [0] * V
    held = [[] for _ in range(U)]
    free = [v for v in range(V) if prefs[v]]
    while free:
        v = free.pop()
        if nxt[v] >= len(prefs[v]):
            continue
        u = prefs[v][nxt[v]]; nxt[v] += 1
        held[u].append(v)
        if len(held[u]) > phi_max:
            worst = min(held[u], key=lambda x: w[u, x])
            held[u].remove(worst)
            free.append(worst)
    for u in range(U):
        for v in held[u]:
            Z[u, v] = 1
    return Z


# ----------------------------------------------------------------------------- GA baseline
def _decode(chrom, cand, w, phi_max):
    U, V = cand.shape
    Z = np.zeros((U, V), np.int8)
    for v, gne in enumerate(chrom):
        if gne > 0 and cand[gne - 1, v]:
            Z[gne - 1, v] = 1
    for u in range(U):                                  # repair C5: keep the phi_max heaviest
        idx = np.flatnonzero(Z[u])
        if idx.size > phi_max:
            drop = idx[np.argsort(w[u, idx])[: idx.size - phi_max]]
            Z[u, drop] = 0
    return Z


def ga_select(cand, w, phi_max, s_req, rng, pop=50, gens=100, tournament=3, p_cross=0.8, p_mut=0.05, penalty=5.0):
    U, V = cand.shape
    if U == 0 or V == 0 or not cand.any():
        return np.zeros((U, V), np.int8)

    def fit(ch):
        Z = _decode(ch, cand, w, phi_max)
        tot = (Z * w).sum(1)
        return tot.sum() - penalty * np.maximum(s_req - tot, 0).sum()

    P = rng.integers(0, U + 1, size=(pop, V))
    F = np.array([fit(c) for c in P])
    for _ in range(gens):
        new = [P[F.argmax()].copy()]                    # elitism
        while len(new) < pop:
            pa = [P[max(rng.integers(0, pop, tournament), key=lambda i: F[i])] for _ in range(2)]
            c1, c2 = pa[0].copy(), pa[1].copy()
            if rng.random() < p_cross and V > 1:
                cut = rng.integers(1, V)
                c1[cut:], c2[cut:] = pa[1][cut:].copy(), pa[0][cut:].copy()
            for c in (c1, c2):
                m = rng.random(V) < p_mut
                c[m] = rng.integers(0, U + 1, size=m.sum())
                new.append(c)
        P = np.array(new[:pop]); F = np.array([fit(c) for c in P])
    return _decode(P[F.argmax()], cand, w, phi_max)


def select(method, cand, w, phi_max, s_req=4.0, rng=None, ga_cfg=None, icv_score=None):
    if method == 'wsmm':
        return wsmm(cand, w, phi_max)
    if method == 'sm':
        return stable_matching(cand, w, phi_max, icv_score)
    if method == 'ga':
        return ga_select(cand, w, phi_max, s_req, rng or np.random.default_rng(0), **(ga_cfg or {}))
    raise ValueError(method)
