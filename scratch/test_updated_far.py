import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_false_cpm import SecureDTEnv, load_config
from false_cpm_resistance import FalseCPMResistance, FalseCPMConfig, CPMReport
from attack_injector import AttackInjector

cfg = load_config()
fcpm_cfg = FalseCPMConfig(
    enable_layer_a=True,
    enable_layer_b=True,
    enable_layer_c=True,
    eta_reward=0.15,
    eta_penalty=0.5,
    quarantine_thresh=0.35
)

for p_mal in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]:
    amw, rmse, frr, far, quar, ht, mt = [], [], [], [], [], [], []
    for s in range(5):
        defence = FalseCPMResistance(fcpm_cfg)
        injector = AttackInjector(p_mal=p_mal)
        env = SecureDTEnv(cfg, defence, injector, seed=s*100+42, use_fast_solvers=True)
        env.reset(seed=s*100+42)
        
        # Override run_episode with the accurate is_falsified tagging
        T = cfg['time']['T']
        pc = cfg['perception']
        metrics = {
            'omega_hist': [], 'dt_rmse_hist': [],
            'false_rejections': 0, 'false_acceptances': 0,
            'true_rejections': 0, 'true_acceptances': 0,
            'quarantine_hist': [], 'honest_trust_hist': [], 'malicious_trust_hist': []
        }

        for t in range(T):
            m = env.mob
            icv_idx = np.flatnonzero(m.active & m.is_icv)
            nic_idx = np.flatnonzero(m.active & ~m.is_icv)
            if icv_idx.size == 0 or nic_idx.size == 0:
                m.step()
                continue

            sep = np.abs(m.pos[nic_idx][:, None] - m.pos[icv_idx][None, :])
            cand = sep <= pc['r_sense']
            q = env.rng.uniform(pc['q_low'], pc['q_high'], sep.shape)

            cpms = {}
            ground_truth_pos = {}

            for ui, u_gid in enumerate(nic_idx):
                ground_truth_pos[int(u_gid)] = float(m.pos[u_gid])
                cpms[int(u_gid)] = []

                for vi, v_gid in enumerate(icv_idx):
                    if not cand[ui, vi]: continue
                    honest = CPMReport(
                        reporter_id=int(v_gid), target_id=int(u_gid),
                        reported_pos=float(m.pos[u_gid]), reported_speed=float(m.speed[u_gid]),
                        quality=float(q[ui, vi]), s_unit=float(m.s_unit[v_gid]), timestamp=t
                    )

                    if injector.is_malicious(int(v_gid)):
                        attacked = injector.attack(honest, env.rng, m.L)
                        if attacked is None: continue
                        report = attacked
                        # Genuinely falsified if content was modified
                        is_falsified = (attacked is not honest)
                    else:
                        report = honest
                        is_falsified = False

                    report._is_malicious = is_falsified
                    cpms[int(u_gid)].append(report)

            fused_pos, trust_vec, fusion_conf = defence.process_slot(t, cpms, true_pos=ground_truth_pos)

            for target_id, reports in cpms.items():
                for r in reports:
                    ts = defence.trust_states.get(r.reporter_id)
                    is_quar = ts.quarantined if ts else False
                    was_accepted = (not is_quar) and (trust_vec.get(r.reporter_id, 1.0) >= defence.cfg.quarantine_thresh)

                    if getattr(r, '_is_malicious', False):
                        if was_accepted:
                            metrics['false_acceptances'] += 1
                        else:
                            metrics['true_rejections'] += 1
                    else:
                        if was_accepted:
                            metrics['true_acceptances'] += 1
                        else:
                            metrics['false_rejections'] += 1

            # Subproblem 1
            w = q * m.s_unit[icv_idx][None, :] * env.tau
            cand_secure = cand.copy()
            w_secure = defence.get_wsmm_weights(w, cand_secure, nic_idx, icv_idx)
            from fast_optimal_solvers import fast_spatial_greedy_matching, fast_solve_rsu
            Zs = fast_spatial_greedy_matching(cand_secure, w_secure, pc['phi_max'],
                                               nic_pos=m.pos[nic_idx], icv_pos=m.pos[icv_idx],
                                               r_sense=pc['r_sense'], road_length=m.L)
            Z_full = np.zeros((env.N, env.N), np.int8)
            if nic_idx.size and icv_idx.size:
                Z_full[np.ix_(nic_idx, icv_idx)] = Zs

            sel_flag = np.zeros(env.N, bool)
            sel_flag[icv_idx] = Zs.sum(0) > 0 if icv_idx.size else False
            W_vetted = np.zeros(env.N)
            if nic_idx.size and icv_idx.size:
                W_vetted[nic_idx] = defence.get_vetted_perceived_size(Zs, w, nic_idx, icv_idx)

            s_v = m.s_own + m.s_unit * sel_flag
            c_cp = (Z_full * (m.c_unit * m.s_unit)[None, :]).sum(1)

            dx = m.pos[:, None] - env.centers[None, :]
            dist = np.abs(dx)
            in_range = (dist <= cfg['rsu']['radius']) & m.active[:, None]
            sx = -np.ones(env.N, int)
            for i in np.flatnonzero(m.active & m.is_icv):
                if in_range[i].any():
                    sx[i] = int(np.argmin(np.where(in_range[i], dist[i], np.inf)))
            for u in nic_idx:
                sel = np.flatnonzero(Z_full[u])
                if sel.size and sx[sel[0]] >= 0:
                    sx[u] = sx[sel[0]]

            c6_ok = np.zeros(env.N, bool)
            c6_ok[nic_idx] = W_vetted[nic_idx] >= (pc['s_req'] - 1e-9)

            from env.channel import gain, H_mhz
            g = gain(dx, cfg['rsu']['height'], cfg['comm']['pl0'], cfg['comm']['pl_slope'],
                      env.rng, cfg['comm']['fading'])
            H = H_mhz(g, cfg['comm']['p'], cfg['comm']['noise_dbm_hz'])

            T_tran, T_exe = np.zeros(env.N), np.zeros(env.N)
            ac = cfg['aoi']
            for r in range(env.K):
                Vr = icv_idx[sx[icv_idx] == r]
                Ur = nic_idx[sx[nic_idx] == r]
                if Vr.size == 0: continue
                pos_in_V = {v: j for j, v in enumerate(Vr)}
                sel = [np.array([pos_in_V[v] for v in np.flatnonzero(Z_full[u]) if v in pos_in_V]) for u in Ur]
                w_v = env.A[Vr] + 1.0
                w_u = env.A[Ur] + 1.0
                w_v_sec, w_u_sec = defence.get_bca_weights(w_v, w_u, Vr, Ur)
                try:
                    res = fast_solve_rsu(s_v[Vr] * env.tau, H[Vr, r], w_v_sec,
                                         m.c_own[Vr] * m.s_own[Vr] * env.tau,
                                         w_u_sec, c_cp[Ur] * env.tau,
                                         sel, cfg['comm']['bmax_hz'] / 1e6, cfg['comp']['F_ghz'])
                    T_tran[Vr] = res['t_tran']
                    T_exe[Vr] = m.c_own[Vr] * m.s_own[Vr] * env.tau / res['f_own']
                    for j, u in enumerate(Ur):
                        if sel[j].size:
                            T_tran[u] = res['t_tran'][sel[j]].max()
                            T_exe[u] = c_cp[u] * env.tau / res['f_cp'][j]
                except Exception:
                    pass

            for u in nic_idx:
                if sx[u] < 0:
                    T_tran[u] = ac['t_pena']
                    T_exe[u] = ac['t_epena']

            T_sum = T_tran + T_exe
            T_req = np.where(m.is_icv, ac['t_req_icv'], ac['t_req_nicv'])
            valid = np.ones(env.N, bool)
            valid[nic_idx] = c6_ok[nic_idx] & (sx[nic_idx] >= 0)
            ok = (T_sum <= T_req) & valid

            A_new = np.where(ok, 0.0, env.A + 1.0)
            A_new = np.where(m.active, A_new, 0.0)
            alpha = ac['alpha']
            wA = np.where(m.is_icv, alpha, 1 - alpha) * A_new
            omega = float(wA[m.active].max()) if m.active.any() else 0.0
            metrics['omega_hist'].append(omega)
            if defence.slot_stats:
                metrics['dt_rmse_hist'].append(defence.slot_stats[-1]['dt_rmse'])
            env.A = A_new

            h_trusts = [defence._ensure_trust(v).trust for v in icv_idx if not injector.is_malicious(v)]
            m_trusts = [defence._ensure_trust(v).trust for v in icv_idx if injector.is_malicious(v)]
            if h_trusts: metrics['honest_trust_hist'].append(float(np.mean(h_trusts)))
            if m_trusts: metrics['malicious_trust_hist'].append(float(np.mean(m_trusts)))
            metrics['quarantine_hist'].append(sum(1 for ts in defence.trust_states.values() if ts.quarantined))
            m.step()

        far_val = metrics['false_acceptances'] / max(metrics['true_rejections'] + metrics['false_acceptances'], 1)
        frr_val = metrics['false_rejections'] / max(metrics['true_acceptances'] + metrics['false_rejections'], 1)
        amw.append(float(np.mean(metrics['omega_hist'])))
        rmse.append(float(np.mean(metrics['dt_rmse_hist'])) if metrics['dt_rmse_hist'] else 0.0)
        far.append(far_val)
        frr.append(frr_val)
        quar.append(metrics['quarantine_hist'][-1])
        if metrics['honest_trust_hist']: ht.append(metrics['honest_trust_hist'][-1])
        if metrics['malicious_trust_hist']: mt.append(metrics['malicious_trust_hist'][-1])

    ht_str = f"{np.mean(ht):.3f}" if ht else "n/a"
    mt_str = f"{np.mean(mt):.3f}" if mt else "n/a"
    print(f"p_mal={p_mal:4.0%} | AMWAoI={np.mean(amw):5.2f}s | RMSE={np.mean(rmse):5.2f}m | FRR={np.mean(frr):4.1%} | FAR={np.mean(far):4.1%} | H-Trust={ht_str} | M-Trust={mt_str} | Q={np.mean(quar):.1f}")

