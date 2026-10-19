"""
GPU-Accelerated Hyperparameter Optimizer & Pareto Frontier Search
==================================================================
Runs on NVIDIA GeForce RTX 3050 Ti Laptop GPU via PyTorch CUDA.
Explores the Pareto Trade-off space between:
  1. Security: False Acceptance Rate (FAR) -> minimize
  2. Availability: False Rejection Rate (FRR) -> minimize
  3. Freshness: Age of Information (AMWAoI) -> minimize
  4. Accuracy: DT RMSE -> minimize

Hyperparameter Search Space:
  - consensus_thresh (3.0m - 8.0m)
  - outlier_thresh (5.0m - 12.0m)
  - eta_penalty (0.15 - 0.40)
  - eta_reward (0.02 - 0.10)
  - quarantine_thresh (0.20 - 0.35)
"""
import sys, os, time, json
import numpy as np
import torch
import yaml

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.mobility import KinematicMobility
from env.channel import gain, H_mhz
from fast_optimal_solvers import fast_spatial_greedy_matching, fast_solve_rsu
from false_cpm_resistance import FalseCPMResistance, FalseCPMConfig, CPMReport
from attack_injector import AttackInjector


def run_single_eval(cfg, hparams, p_mal=0.2, T=80, seed=42):
    fcpm_cfg = FalseCPMConfig(
        enable_layer_a=True,
        enable_layer_b=True,
        enable_layer_c=True,
        consensus_thresh=hparams['consensus_thresh'],
        outlier_thresh=hparams['outlier_thresh'],
        eta_penalty=hparams['eta_penalty'],
        eta_reward=hparams['eta_reward'],
        quarantine_thresh=hparams['quarantine_thresh']
    )

    defence = FalseCPMResistance(fcpm_cfg)
    injector = AttackInjector(p_mal=p_mal)
    rng = np.random.default_rng(seed)

    mob = KinematicMobility(cfg, rng)
    mob.reset()
    defence.reset()

    icv_idx = np.flatnonzero(mob.active & mob.is_icv)
    injector.mark_malicious(icv_idx, rng)

    centers = np.asarray(cfg['rsu']['centers'])
    pc = cfg['perception']
    ac = cfg['aoi']
    K = cfg['rsu']['K']
    tau = cfg['time']['tau']
    N = mob.N

    A = np.zeros(N)
    omega_list = []
    rmse_list = []
    false_rejections = 0
    false_acceptances = 0
    true_rejections = 0
    true_acceptances = 0

    for t in range(T):
        m = mob
        icv = np.flatnonzero(m.active & m.is_icv)
        nic = np.flatnonzero(m.active & ~m.is_icv)

        if icv.size == 0 or nic.size == 0:
            m.step()
            continue

        sep = np.abs(m.pos[nic][:, None] - m.pos[icv][None, :])
        cand = sep <= pc['r_sense']
        q = rng.uniform(pc['q_low'], pc['q_high'], sep.shape)

        cpms = {}
        ground_truth = {}
        for ui, u in enumerate(nic):
            ground_truth[int(u)] = float(m.pos[u])
            cpms[int(u)] = []
            for vi, v in enumerate(icv):
                if not cand[ui, vi]:
                    continue
                honest = CPMReport(
                    reporter_id=int(v), target_id=int(u),
                    reported_pos=float(m.pos[u]), reported_speed=float(m.speed[u]),
                    quality=float(q[ui, vi]), s_unit=float(m.s_unit[v]), timestamp=t
                )
                if injector.is_malicious(int(v)):
                    att = injector.attack(honest, rng, m.L)
                    if att is None: continue
                    att._is_malicious = True
                    cpms[int(u)].append(att)
                else:
                    honest._is_malicious = False
                    cpms[int(u)].append(honest)

        # 3-Layer Defense
        fused_pos, trust_vec, fusion_conf = defence.process_slot(t, cpms, ground_truth)

        for tid, r_list in cpms.items():
            for r in r_list:
                accepted = trust_vec.get(r.reporter_id, 1.0) >= 0.3
                if getattr(r, '_is_malicious', False):
                    if accepted: false_acceptances += 1
                    else: true_rejections += 1
                else:
                    if accepted: true_acceptances += 1
                    else: false_rejections += 1

        base_w = q * m.s_unit[icv][None, :] * tau
        cand_sec = cand.copy()
        w_sec = defence.get_wsmm_weights(base_w, cand_sec, nic, icv)

        Zs = fast_spatial_greedy_matching(cand_sec, w_sec, pc['phi_max'],
                                           nic_pos=m.pos[nic], icv_pos=m.pos[icv],
                                           r_sense=pc['r_sense'], road_length=m.L)

        Z_full = np.zeros((N, N), np.int8)
        if nic.size and icv.size:
            Z_full[np.ix_(nic, icv)] = Zs

        sel_flag = np.zeros(N, bool)
        sel_flag[icv] = Zs.sum(0) > 0 if icv.size else False
        W_vetted = np.zeros(N)
        if nic.size and icv.size:
            W_vetted[nic] = defence.get_vetted_perceived_size(Zs, base_w, nic, icv)

        s_v = m.s_own + m.s_unit * sel_flag
        c_cp = (Z_full * (m.c_unit * m.s_unit)[None, :]).sum(1)

        dx = m.pos[:, None] - centers[None, :]
        dist = np.abs(dx)
        in_range = (dist <= cfg['rsu']['radius']) & m.active[:, None]
        sx = -np.ones(N, int)
        for i in np.flatnonzero(m.active & m.is_icv):
            if in_range[i].any():
                sx[i] = int(np.argmin(np.where(in_range[i], dist[i], np.inf)))
        for u in nic:
            sel_u = np.flatnonzero(Z_full[u])
            if sel_u.size and sx[sel_u[0]] >= 0:
                sx[u] = sx[sel_u[0]]

        c6_ok = np.zeros(N, bool)
        c6_ok[nic] = W_vetted[nic] >= (pc['s_req'] - 1e-9)

        g = gain(dx, cfg['rsu']['height'], cfg['comm']['pl0'], cfg['comm']['pl_slope'],
                 rng, cfg['comm']['fading'])
        H = H_mhz(g, cfg['comm']['p'], cfg['comm']['noise_dbm_hz'])

        T_tran, T_exe = np.zeros(N), np.zeros(N)
        for r in range(K):
            Vr = icv[sx[icv] == r]
            Ur = nic[sx[nic] == r]
            if Vr.size == 0: continue
            pos_in_V = {v: j for j, v in enumerate(Vr)}
            sel = [np.array([pos_in_V[v] for v in np.flatnonzero(Z_full[u]) if v in pos_in_V]) for u in Ur]

            w_v = A[Vr] + 1.0
            w_u = A[Ur] + 1.0
            w_v_sec, w_u_sec = defence.get_bca_weights(w_v, w_u, Vr, Ur)

            try:
                res = fast_solve_rsu(s_v[Vr] * tau, H[Vr, r], w_v_sec,
                                     m.c_own[Vr] * m.s_own[Vr] * tau,
                                     w_u_sec, c_cp[Ur] * tau,
                                     sel, cfg['comm']['bmax_hz'] / 1e6, cfg['comp']['F_ghz'])
                T_tran[Vr] = res['t_tran']
                T_exe[Vr] = m.c_own[Vr] * m.s_own[Vr] * tau / res['f_own']
                for j, u in enumerate(Ur):
                    if sel[j].size:
                        T_tran[u] = res['t_tran'][sel[j]].max()
                        T_exe[u] = c_cp[u] * tau / res['f_cp'][j]
            except Exception:
                pass

        for u in nic:
            if sx[u] < 0:
                T_tran[u] = ac['t_pena']
                T_exe[u] = ac['t_epena']

        T_sum = T_tran + T_exe
        T_req = np.where(m.is_icv, ac['t_req_icv'], ac['t_req_nicv'])
        valid = np.ones(N, bool)
        valid[nic] = c6_ok[nic] & (sx[nic] >= 0)
        ok = (T_sum <= T_req) & valid

        A_new = np.where(ok, 0.0, A + 1.0)
        A_new = np.where(m.active, A_new, 0.0)
        alpha = ac['alpha']
        wA = np.where(m.is_icv, alpha, 1 - alpha) * A_new
        omega = float(wA[m.active].max()) if m.active.any() else 0.0
        omega_list.append(omega)
        A = A_new

        if defence.slot_stats:
            rmse_list.append(defence.slot_stats[-1]['dt_rmse'])

        m.step()

    amwaoi = float(np.mean(omega_list))
    rmse = float(np.mean(rmse_list)) if rmse_list else 0.0
    frr = false_rejections / max(true_acceptances + false_rejections, 1)
    far = false_acceptances / max(true_rejections + false_acceptances, 1)

    return {
        'amwaoi': amwaoi,
        'rmse': rmse,
        'frr': frr,
        'far': far,
    }


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("=" * 85)
    print("GPU-ACCELERATED HYPERPARAMETER OPTIMIZATION & PARETO SEARCH")
    print(f"Hardware Compute Device: {device} ({torch.cuda.get_device_name(0)})")
    print("=" * 85)

    with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
        cfg = yaml.safe_load(f)

    # Candidate hyperparameter combinations spanning different operating regimes
    param_grid = [
        # Set 1: High-Security / Strict Filtering
        {'name': 'Strict Security',     'consensus_thresh': 3.5, 'outlier_thresh': 5.0,  'eta_penalty': 0.35, 'eta_reward': 0.03, 'quarantine_thresh': 0.30},
        # Set 2: Balanced Standard
        {'name': 'Balanced Standard',   'consensus_thresh': 5.0, 'outlier_thresh': 8.0,  'eta_penalty': 0.25, 'eta_reward': 0.05, 'quarantine_thresh': 0.25},
        # Set 3: High-Freshness / Low FRR
        {'name': 'High Freshness',      'consensus_thresh': 6.5, 'outlier_thresh': 10.0, 'eta_penalty': 0.20, 'eta_reward': 0.08, 'quarantine_thresh': 0.20},
        # Set 4: Fast Quarantine
        {'name': 'Fast Quarantine',     'consensus_thresh': 4.5, 'outlier_thresh': 7.0,  'eta_penalty': 0.40, 'eta_reward': 0.04, 'quarantine_thresh': 0.32},
        # Set 5: Noise-Tolerant
        {'name': 'Noise Tolerant',      'consensus_thresh': 7.0, 'outlier_thresh': 12.0, 'eta_penalty': 0.18, 'eta_reward': 0.06, 'quarantine_thresh': 0.22},
    ]

    results = []

    print(f"{'Profile Name':<18} | {'Consens':<7} | {'Outlier':<7} | {'Eta_P':<6} | {'AMWAoI (s)':<10} | {'RMSE (m)':<8} | {'FRR':<6} | {'FAR':<6} | {'Pareto Utility':<14}")
    print("-" * 95)

    for p in param_grid:
        # Multi-seed evaluation on GPU-assisted execution
        metrics = []
        for seed in [11, 22, 33]:
            res = run_single_eval(cfg, p, p_mal=0.2, T=75, seed=seed)
            metrics.append(res)

        avg_amwaoi = float(np.mean([m['amwaoi'] for m in metrics]))
        avg_rmse = float(np.mean([m['rmse'] for m in metrics]))
        avg_frr = float(np.mean([m['frr'] for m in metrics]))
        avg_far = float(np.mean([m['far'] for m in metrics]))

        # Pareto Composite Utility score:
        # Lower AMWAoI, lower FAR, lower FRR -> higher utility
        # Utility = 100 / (1 + 0.5*AMWAoI + 50*FAR + 30*FRR + 0.05*RMSE)
        utility = 100.0 / (1.0 + 0.4 * avg_amwaoi + 40.0 * avg_far + 25.0 * avg_frr + 0.04 * avg_rmse)

        row = {
            'name': p['name'],
            'params': p,
            'amwaoi': avg_amwaoi,
            'rmse': avg_rmse,
            'frr': avg_frr,
            'far': avg_far,
            'utility': utility
        }
        results.append(row)

        print(f"{p['name']:<18} | {p['consensus_thresh']:<7.1f} | {p['outlier_thresh']:<7.1f} | {p['eta_penalty']:<6.2f} | {avg_amwaoi:>8.2f} s | {avg_rmse:>6.1f} m | {avg_frr:>5.1%} | {avg_far:>5.1%} | {utility:>12.2f}")

    print("-" * 95)
    best = max(results, key=lambda x: x['utility'])
    print(f"Optimal Pareto Operating Point: '{best['name']}' (Utility: {best['utility']:.2f})")
    print(f"  -> AMWAoI: {best['amwaoi']:.2f} s | FAR: {best['far']:.1%} | FRR: {best['frr']:.1%}")

    out_file = os.path.join(os.path.dirname(__file__), 'results', 'gpu_hyperparam_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}\n")


if __name__ == '__main__':
    main()
