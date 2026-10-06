"""Per-slot pipeline (Sec. 2). Gym-like reset()/step(). Padded arrays of size N (= N_max)."""
import copy
import numpy as np
from env.mobility import make_mobility
from env.channel import gain, H_mhz, rate_mbps
import sub1_wsmm
from sub2_bca import solve_rsu


class DTEnv:
    def __init__(self, cfg, sub1=None, seed=None, predictive=True):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg['seed'] if seed is None else seed)
        self.sub1 = sub1 or cfg['perception']['sub1']
        self.mob = make_mobility(cfg, self.rng)
        self.N, self.K, self.T = self.mob.N, cfg['rsu']['K'], cfg['time']['T']
        self.tau = cfg['time']['tau']
        self.centers = np.asarray(cfg['rsu']['centers'])
        self.feat_dim = 11 + 4 * self.K
        self.obs_dim = self.N * self.feat_dim
        self.predictive = predictive           # if False the agent's y is ignored (pure reactive migration)

    # ------------------------------------------------------------------ reset / slot preparation
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed); self.mob.rng = self.rng
        self.mob.reset()
        N, K = self.N, self.K
        self.t = 0
        self.A = np.zeros(N)
        self.dt_loc = np.zeros(N, int)
        self.pend_t = -np.ones(N, int); self.pend_s = np.zeros(N, int)
        self.x_prev = -np.ones(N, int); self.y_prev = np.zeros(N, int)
        for i in np.flatnonzero(self.mob.active):
            self._init_vehicle(i)
        self.omega_hist = []
        self._prepare_slot()
        return self._obs()

    def _init_vehicle(self, i):
        self.A[i] = 0.0
        self.dt_loc[i] = int(np.argmin(np.abs(self.mob.pos[i] - self.centers)))
        self.pend_t[i] = -1; self.x_prev[i] = -1; self.y_prev[i] = 0

    def _prepare_slot(self):
        cfg, m, K, N = self.cfg, self.mob, self.K, self.N
        pc, cc = cfg['perception'], cfg['comm']
        dx = m.pos[:, None] - self.centers[None, :]                    # (N,K)
        self.dist = np.abs(dx)
        self.in_range = (self.dist <= cfg['rsu']['radius']) & m.active[:, None]
        g = gain(dx, cfg['rsu']['height'], cc['pl0'], cc['pl_slope'], self.rng, cc['fading'])
        self.H = H_mhz(g, cc['p'], cc['noise_dbm_hz'])                  # (N,K) MHz
        icv, nic = np.flatnonzero(m.active & m.is_icv), np.flatnonzero(m.active & ~m.is_icv)
        self.icv_idx, self.nic_idx = icv, nic
        # perception quality and candidate sets
        sep = np.abs(m.pos[nic][:, None] - m.pos[icv][None, :])
        cand = sep <= pc['r_sense']
        if pc['q_model'] == 'uniform':
            q = self.rng.uniform(pc['q_low'], pc['q_high'], sep.shape)
        else:
            q = np.clip(pc['q_high'] * (1 - sep / pc['r_sense']) + pc['q_low'], pc['q_low'], pc['q_high'])
        w = q * m.s_unit[icv][None, :] * self.tau                       # Mb
        self.q_sub, self.cand_sub, self.w_sub = q, cand, w
        ga_cfg = {k: v for k, v in cfg['ga'].items()}
        Zs = sub1_wsmm.select(self.sub1, cand, w, pc['phi_max'], pc['s_req'], self.rng, ga_cfg) if icv.size and nic.size \
            else np.zeros((nic.size, icv.size), np.int8)
        self.Z = np.zeros((N, N), np.int8)                              # Z[u, v] in global indices
        self.Zsub = Zs
        if nic.size and icv.size:
            self.Z[np.ix_(nic, icv)] = Zs
        self.sel_flag = np.zeros(N, bool); self.sel_flag[icv] = Zs.sum(0) > 0 if icv.size else False
        self.W = np.zeros(N)                                            # weighted size of each N-ICV [Mb]
        if nic.size and icv.size:
            self.W[nic] = (Zs * w).sum(1)
        self.n_sel = self.Z.sum(1)
        so = pc['transmit_only_if_selected']
        self.s_v = m.s_own + m.s_unit * (self.sel_flag if so else 1.0)  # Eq (1)  [Mbps]
        self.c_cp = (self.Z * (m.c_unit * m.s_unit)[None, :]).sum(1)    # Eq (7)  [Gcycles/s]

    # ------------------------------------------------------------------ masks and observation
    def action_masks(self):
        m, K = self.mob, self.K
        icv = m.active & m.is_icv
        xm = self.in_range & icv[:, None]
        ym = np.zeros((self.N, K + 1), bool); ym[:, 0] = True
        for k in range(K):
            ym[:, k + 1] = m.active & (self.dt_loc != k)
        return dict(x_mask=xm, y_mask=ym, x_rel=icv, y_rel=m.active.copy())

    def _obs(self):
        m, K, N = self.mob, self.K, self.N
        F = np.zeros((N, self.feat_dim), np.float32)
        a = m.active
        F[:, 0] = a; F[:, 1] = m.is_icv & a; F[:, 2] = m.pos / m.L * a; F[:, 3] = m.dirn * a
        F[:, 4] = np.minimum(self.A, 30) / 10.0
        o = 5
        for i in np.flatnonzero(a):
            F[i, o + self.dt_loc[i]] = 1
            if self.x_prev[i] >= 0: F[i, o + K + self.x_prev[i]] = 1
            if self.pend_t[i] >= 0: F[i, o + 2 * K + self.pend_t[i]] = 1
        F[:, o + 3 * K:o + 4 * K] = self.in_range
        o += 4 * K
        F[:, o] = m.c_own / 0.2 * (m.is_icv & a); F[:, o + 1] = self.c_cp / 3.0
        F[:, o + 2] = self.s_v / 6.0 * a
        F[:, o + 3] = np.where(m.is_icv, self.sel_flag, self.n_sel / self.cfg['perception']['phi_max']) * a
        F[:, o + 4] = self.W / self.cfg['perception']['s_req'] * (~m.is_icv & a)
        F[:, o + 5] = 0.0
        return F.reshape(-1)

    # ------------------------------------------------------------------ one slot
    def step(self, action):
        cfg, m, K, N, tau = self.cfg, self.mob, self.K, self.N, self.tau
        ac, am, mg, pn = cfg['aoi'], cfg['comm'], cfg['migration'], cfg['perception']
        x_act, y_act = np.asarray(action['x'], int), np.asarray(action['y'], int)
        icv, nic = self.icv_idx, self.nic_idx

        # --- access selection (ICV from agent; N-ICV via Eq. (4))
        sx = -np.ones(N, int)
        for v in icv:
            k = x_act[v]
            if not self.in_range[v, k]:                              # safety: out of range -> nearest in range
                k = int(np.argmin(np.where(self.in_range[v], self.dist[v], np.inf)))
            sx[v] = k
        for u in nic:
            sel = np.flatnonzero(self.Z[u])
            if sel.size and (sx[sel] == sx[sel[0]]).all():
                sx[u] = sx[sel[0]]
        c6_ok = np.zeros(N, bool)
        c6_ok[nic] = self.W[nic] >= pn['s_req'] - 1e-9

        # --- migration bookkeeping (y), T_pre logic
        mig_count = np.zeros(K, int)
        if self.predictive:
            for i in np.flatnonzero(m.active):
                k = y_act[i] - 1
                if k >= 0 and k != self.dt_loc[i] and self.pend_t[i] != k:
                    self.pend_t[i], self.pend_s[i] = k, self.t; mig_count[k] += 1
        T_wired, T_pre, T_amigr = np.zeros(N), np.zeros(N), np.zeros(N)
        migrated = np.zeros(N, bool)
        for i in np.flatnonzero(m.active):
            k = sx[i]
            if k >= 0 and k != self.dt_loc[i]:
                d = abs(k - self.dt_loc[i]) * cfg['rsu']['spacing']
                T_wired[i] = mg['t_unit'] * m.D[i] * d                     # Eq (10)
                if self.pend_t[i] == k:
                    T_pre[i] = (self.t - self.pend_s[i]) * tau
                else:
                    mig_count[k] += 1                                      # reactive migration
                T_amigr[i] = max(T_wired[i] - T_pre[i], 0.0)               # Eq (11)
                self.dt_loc[i] = k; self.pend_t[i] = -1; migrated[i] = True

        # --- Subproblem 2 per RSU
        T_tran, T_exe = np.zeros(N), np.zeros(N)
        Bv, fv = np.zeros(N), np.zeros(N)
        iters = 0
        for r in range(K):
            Vr = icv[sx[icv] == r]; Ur = nic[sx[nic] == r]
            if Vr.size == 0:
                continue
            pos_in_V = {v: j for j, v in enumerate(Vr)}
            sel = [np.array([pos_in_V[v] for v in np.flatnonzero(self.Z[u])]) for u in Ur]
            res = solve_rsu(self.s_v[Vr] * tau, self.H[Vr, r], self.A[Vr] + 1.0,
                            m.c_own[Vr] * m.s_own[Vr] * tau, self.A[Ur] + 1.0, self.c_cp[Ur] * tau,
                            sel, am['bmax_hz'] / 1e6, cfg['comp']['F_ghz'], cfg['sub2'])
            iters += res['iters']
            Bv[Vr] = res['B']; fv[Vr] = res['f_own']; fv[Ur] = res['f_cp']
            T_tran[Vr] = res['t_tran']
            T_exe[Vr] = m.c_own[Vr] * m.s_own[Vr] * tau / res['f_own']      # Eq (6)
            for j, u in enumerate(Ur):
                T_tran[u] = res['t_tran'][sel[j]].max()                     # Eq (5), x_u = 1
                T_exe[u] = self.c_cp[u] * tau / res['f_cp'][j]              # Eq (8)
        for u in nic:                                                      # failure cases (ASSUMPTIONS #8)
            if sx[u] < 0:
                T_tran[u] = ac['t_pena']; T_exe[u] = ac['t_epena']

        # --- delays, AoI, reward
        T_sum = np.maximum(T_amigr, T_tran) + T_exe                        # Eq (12)
        T_req = np.where(m.is_icv, ac['t_req_icv'], ac['t_req_nicv'])
        valid = np.ones(N, bool)
        valid[nic] = c6_ok[nic] & (sx[nic] >= 0)                           # ASSUMPTIONS #9
        ok = (T_sum <= T_req) & valid
        a = m.active
        A_new = np.where(ok, 0.0, self.A + 1.0)                            # Eq (13)
        A_new = np.where(a, A_new, 0.0)
        alpha = ac['alpha']
        wA = np.where(m.is_icv, alpha, 1 - alpha) * A_new
        omega = float(wA[a].max()) if a.any() else 0.0                      # Eq (14), A^{t+1}
        pen = float(np.maximum(mig_count - mg['m_max'], 0).sum())          # Eq (30) summed over k
        reward = cfg['reward']['beta_rew'] * omega + cfg['reward']['beta_pen'] * pen
        self.omega_hist.append(omega)
        self.A = A_new
        self.x_prev = sx.copy()

        mig = migrated & a
        info = dict(omega=omega, penalty=pen, T_tran=T_tran, T_exe=T_exe, T_amigr=T_amigr, T_wired=T_wired, T_pre=T_pre,
                    T_sum=T_sum, valid=valid, ok=ok, sx=sx, B=Bv, f=fv, mig_count=mig_count, migrated=mig,
                    active=a.copy(), is_icv=m.is_icv.copy(), bca_iters=iters,
                    comm=float(T_tran[a].mean()) if a.any() else 0.0,
                    comp=float(T_exe[a].mean()) if a.any() else 0.0,
                    mig_delay=float(T_amigr[mig].sum()), mig_events=int(mig.sum()))

        # --- advance
        self.t += 1
        done = self.t >= self.T
        new, gone = m.step()
        for i in new:
            self._init_vehicle(i)
        for i in gone:
            self.A[i] = 0.0
        self._prepare_slot()
        if done:
            info['amwaoi'] = float(np.mean(self.omega_hist))
        return self._obs(), reward, done, info
