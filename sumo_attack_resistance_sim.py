"""
Realistic Eclipse SUMO TraCI Traffic Simulation: Attack Resistance Benchmark
=============================================================================
Simulates microscopic vehicular traffic along a 2000m highway using
Wiedemann car-following dynamics, stochastic speed distributions, and 
two-way multi-lane physics.

Compares:
  1. Base Architecture under Real SUMO Traffic (No Defense)
  2. Proposed 3-Layer False CPM Defense + Fast Solvers under Real SUMO Traffic

Tracks:
  - Real SUMO vehicle coordinates vs Digital Twin synchronized state (DT RMSE).
  - Age of Information (AMWAoI) under realistic car-following and lane changes.
  - Attack isolation (FAR, FRR, Trust divergence).
"""
import sys, os, time, json
import numpy as np
import yaml

# Set SUMO environment variables
os.environ['PATH'] = f"/home/junior/Desktop/Coding/Vedansh_Ananya/.venv/bin:{os.environ.get('PATH', '')}"
os.environ['SUMO_HOME'] = "/home/junior/Desktop/Coding/Vedansh_Ananya/.venv/lib/python3.13/site-packages/sumo"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.sumo_mobility import SumoMobility
from env.channel import gain, H_mhz
from fast_optimal_solvers import fast_spatial_greedy_matching, fast_solve_rsu
from false_cpm_resistance import FalseCPMResistance, FalseCPMConfig, CPMReport
from attack_injector import AttackInjector


def run_sumo_experiment(use_defense: bool, p_mal: float = 0.20, T: int = 100, seed: int = 42):
    with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
        cfg = yaml.safe_load(f)

    # Configure SUMO backend
    cfg['mobility']['backend'] = 'sumo'
    cfg['mobility']['sumo'] = {'gui': False}

    rng = np.random.default_rng(seed)
    mob = SumoMobility(cfg, rng)
    mob.reset()

    fcpm_cfg = FalseCPMConfig(
        enable_layer_a=use_defense,
        enable_layer_b=use_defense,
        enable_layer_c=use_defense,
        consensus_thresh=5.0,
        outlier_thresh=8.0,
        eta_penalty=0.25,
        eta_reward=0.05,
        quarantine_thresh=0.25
    )
    defence = FalseCPMResistance(fcpm_cfg)
    injector = AttackInjector(p_mal=p_mal)

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
    step_times = []

    print(f"  [SUMO Sim] Starting {T} slots (use_defense={use_defense}, p_mal={p_mal:.0%})...")

    try:
        for t in range(T):
            t0 = time.perf_counter()
            m = mob
            icv = np.flatnonzero(m.active & m.is_icv)
            nic = np.flatnonzero(m.active & ~m.is_icv)

            if icv.size == 0 or nic.size == 0:
                m.step()
                continue

            # In SUMO, m.pos is extracted directly from TraCI vehicle coordinates
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

            # Process 3-Layer Defense
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

            step_times.append((time.perf_counter() - t0) * 1000.0)

            # Advance real SUMO TraCI simulation
            m.step()

            # Mark newly entering SUMO vehicles if needed
            new_icv = np.flatnonzero(m.active & m.is_icv)
            for v in new_icv:
                if int(v) not in injector.malicious_set and int(v) not in defence.trust_states:
                    if rng.random() < p_mal:
                        injector.malicious_set.add(int(v))
                        injector.attack_type[int(v)] = rng.choice(list(injector.attack_mix.keys()))

    finally:
        mob.close()

    amwaoi = float(np.mean(omega_list))
    rmse = float(np.mean(rmse_list)) if rmse_list else 0.0
    frr = false_rejections / max(true_acceptances + false_rejections, 1)
    far = false_acceptances / max(true_rejections + false_acceptances, 1)
    h_trust = float(np.mean([ts.trust for vid, ts in defence.trust_states.items() if not injector.is_malicious(vid)])) if defence.trust_states else 1.0
    m_trust = float(np.mean([ts.trust for vid, ts in defence.trust_states.items() if injector.is_malicious(vid)])) if defence.trust_states else 0.0

    return {
        'use_defense': use_defense,
        'amwaoi': amwaoi,
        'dt_rmse': rmse,
        'frr': frr,
        'far': far,
        'h_trust': h_trust,
        'm_trust': m_trust,
        'avg_step_ms': float(np.mean(step_times))
    }


def main():
    print("=" * 85)
    print("ECLIPSE SUMO MICROSCOPIC TRAFFIC SIMULATION: V2X ATTACK RESISTANCE")
    print("Highway: 2000m, 2 Lanes/Direction, Wiedemann Car-Following Model (TraCI)")
    print("=" * 85)

    results = {}

    # Run 1: SUMO under Attack without Defense
    print("\n>>> Running SUMO TraCI Simulation 1: Base Architecture (No Defense, p_mal=20%)...")
    res_base = run_sumo_experiment(use_defense=False, p_mal=0.20, T=100, seed=42)
    results['base_no_defense'] = res_base

    # Run 2: SUMO under Attack with Proposed Defense + Fast Solvers
    print("\n>>> Running SUMO TraCI Simulation 2: Proposed 3-Layer Defense + Fast Solvers (p_mal=20%)...")
    res_proposed = run_sumo_experiment(use_defense=True, p_mal=0.20, T=100, seed=42)
    results['proposed_defense'] = res_proposed

    print("\n" + "=" * 90)
    print("REALISTIC SUMO SIMULATION RESULTS SUMMARY")
    print("=" * 90)
    print(f"{'Scenario':<30} | {'AMWAoI (s)':<10} | {'DT RMSE (m)':<12} | {'FAR (Accepted)':<15} | {'FRR (Rejected)':<15} | {'Step Time':<10}")
    print("-" * 90)
    print(f"{'SUMO Baseline (No Defense)':<30} | {res_base['amwaoi']:>8.2f} s | {res_base['dt_rmse']:>10.1f} m | {res_base['far']:>13.1%} | {res_base['frr']:>13.1%} | {res_base['avg_step_ms']:>6.2f} ms")
    print(f"{'SUMO + Proposed 3-Layer':<30} | {res_proposed['amwaoi']:>8.2f} s | {res_proposed['dt_rmse']:>10.1f} m | {res_proposed['far']:>13.1%} | {res_proposed['frr']:>13.1%} | {res_proposed['avg_step_ms']:>6.2f} ms")
    print("-" * 90)

    out_file = os.path.join(os.path.dirname(__file__), 'results', 'sumo_attack_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"SUMO simulation results saved to {out_file}\n")


if __name__ == '__main__':
    main()
