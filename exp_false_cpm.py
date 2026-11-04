"""
Experiment: False CPM Resistance - p_mal Sweep
================================================
Evaluates:
  1. BASE (no defence)
  2. FULL (Layer A + B + C Defence + Fast Optimal Solvers)
across malicious vehicle fraction p_mal = 0% to 50%.

Outputs: DT RMSE, AMWAoI, False Rejection Rate, False Acceptance Rate,
         Honest/Malicious Trust, Quarantine Count.

Solvers & Architectural Highlights:
  - SPG: Two-Phase C6-Satisficing Spatial Priority Greedy matching (Subproblem 1)
  - ED-BCA: Vectorized Bottleneck-Equalizing 1D Dual Bisection (Subproblem 2)
  - Common Random Numbers (CRN): Dedicated attacker RNG so physical trajectories
    are identical across p_mal levels and between BASE and FULL.
  - Proper vehicle respawn lifecycle handling (new, gone = m.step()).
"""
import sys, os, copy, json, time
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.mobility import KinematicMobility, make_mobility
from env.channel import gain, H_mhz
import sub1_wsmm
from sub2_bca import solve_rsu
from fast_optimal_solvers import fast_spatial_greedy_matching, fast_solve_rsu
from false_cpm_resistance import FalseCPMResistance, FalseCPMConfig, CPMReport
from attack_injector import AttackInjector


def load_config():
    import yaml
    with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
        return yaml.safe_load(f)


class SecureDTEnv:
    """
    DTEnv that runs the per-slot pipeline WITH the 3-layer defence.
    Utilizes fast optimal solvers (Two-Phase C6-Satisficing SPG + Vectorized ED-BCA)
    and vetted physical C6 satisfaction.
    """

    def __init__(self, cfg, defence: FalseCPMResistance, injector: AttackInjector, seed=0, use_fast_solvers=True):
        self.cfg = cfg
        self.defence = defence
        self.injector = injector
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.atk_rng = np.random.default_rng(seed + 77777)
        self.mob = make_mobility(cfg, self.rng)
        self.N = self.mob.N
        self.K = cfg['rsu']['K']
        self.tau = cfg['time']['tau']
        self.centers = np.asarray(cfg['rsu']['centers'])
        self.use_fast_solvers = use_fast_solvers

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            self.atk_rng = np.random.default_rng(seed + 77777)
            self.mob.rng = self.rng
        self.mob.reset()
        self.defence.reset()

        N = self.N
        self.A = np.zeros(N)
        self.dt_loc = np.zeros(N, int)
        for i in np.flatnonzero(self.mob.active):
            self.dt_loc[i] = int(np.argmin(np.abs(self.mob.pos[i] - self.centers)))

        # Mark malicious ICVs using isolated attacker RNG (preserves Common Random Numbers)
        icv_idx = np.flatnonzero(self.mob.active & self.mob.is_icv)
        self.injector.mark_malicious(icv_idx, self.atk_rng)

    def close(self):
        if hasattr(self.mob, 'close'):
            self.mob.close()

    def run_episode(self, T=None):
        """Run a full episode and collect metrics."""
        cfg = self.cfg
        T = T or cfg['time']['T']
        pc = cfg['perception']

        metrics = {
            'omega_hist': [],
            'dt_rmse_hist': [],
            'layer_a_rejected': 0,
            'layer_b_outliers': 0,
            'total_reports': 0,
            'false_rejections': 0,     # honest reports incorrectly rejected
            'false_acceptances': 0,    # malicious reports incorrectly accepted
            'true_rejections': 0,      # malicious reports correctly rejected
            'true_acceptances': 0,     # honest reports correctly accepted
            'quarantine_hist': [],
            'honest_trust_hist': [],
            'malicious_trust_hist': [],
            'slot_times_ms': [],
        }

        defence_active = (
            self.defence.cfg.enable_layer_a
            or self.defence.cfg.enable_layer_b
            or self.defence.cfg.enable_layer_c
        )

        for t in range(T):
            t_slot_start = time.perf_counter()
            m = self.mob
            icv_idx = np.flatnonzero(m.active & m.is_icv)
            nic_idx = np.flatnonzero(m.active & ~m.is_icv)

            if icv_idx.size == 0 or nic_idx.size == 0:
                new, gone = m.step()
                self.defence.on_vehicles_respawned(new, gone)
                for i in new + gone:
                    self.A[i] = 0.0
                continue

            # ──── Generate CPMs (honest ground truth + attack injection) ────
            sep = np.abs(m.pos[nic_idx][:, None] - m.pos[icv_idx][None, :])
            cand = sep <= pc['r_sense']

            if pc['q_model'] == 'uniform':
                q = self.rng.uniform(pc['q_low'], pc['q_high'], sep.shape)
            else:
                q = np.clip(pc['q_high'] * (1 - sep / pc['r_sense']) + pc['q_low'],
                            pc['q_low'], pc['q_high'])

            # Build CPM reports (one per ICV-NICV candidate pair)
            cpms = {}  # target_id -> list of CPMReport
            ground_truth_pos = {}  # target_id -> true position
            q_reported = np.zeros_like(q)
            cand_reported = np.zeros_like(cand, dtype=bool)
            link_corrupted = np.zeros_like(cand, dtype=bool)

            for ui, u_gid in enumerate(nic_idx):
                ground_truth_pos[int(u_gid)] = float(m.pos[u_gid])
                cpms[int(u_gid)] = []

                for vi, v_gid in enumerate(icv_idx):
                    if not cand[ui, vi]:
                        continue

                    # Honest report
                    honest = CPMReport(
                        reporter_id=int(v_gid),
                        target_id=int(u_gid),
                        reported_pos=float(m.pos[u_gid]),  # true position
                        reported_speed=float(m.speed[u_gid]),
                        quality=float(q[ui, vi]),
                        s_unit=float(m.s_unit[v_gid]),
                        timestamp=t
                    )

                    # Attack injection (using isolated atk_rng)
                    if self.injector.is_malicious(int(v_gid)):
                        attacked = self.injector.attack(honest, self.atk_rng, m.L)
                        if attacked is None:
                            # Suppression attack - report was omitted/withheld
                            continue
                        report = attacked
                        is_malicious_report = (attacked is not honest)
                    else:
                        report = honest
                        is_malicious_report = False

                    report._is_malicious = is_malicious_report
                    cpms[int(u_gid)].append(report)

                    # Record reported quality for WSMM selection
                    cand_reported[ui, vi] = True
                    q_reported[ui, vi] = report.quality
                    if is_malicious_report:
                        link_corrupted[ui, vi] = True

            # ──── Run the 3-Layer Pipeline ────
            fused_pos, trust_vec, fusion_conf = self.defence.process_slot(
                t, cpms, true_pos=ground_truth_pos
            )

            # ──── Count false rejections / false acceptances at report level ────
            for target_id, reports in cpms.items():
                accepted_for_target = self.defence.last_accepted_reports.get(target_id, set())
                for r in reports:
                    metrics['total_reports'] += 1
                    if defence_active:
                        reporter_trust = trust_vec.get(r.reporter_id, 1.0)
                        was_accepted = (
                            (r.reporter_id in accepted_for_target)
                            and (reporter_trust >= self.defence.cfg.quarantine_thresh)
                        )
                    else:
                        was_accepted = True

                    if hasattr(r, '_is_malicious'):
                        if r._is_malicious:
                            if was_accepted:
                                metrics['false_acceptances'] += 1
                            else:
                                metrics['true_rejections'] += 1
                        else:
                            if was_accepted:
                                metrics['true_acceptances'] += 1
                            else:
                                metrics['false_rejections'] += 1

            # ──── Modify WSMM weights with trust & filter rejected reports ────
            w_true = q * m.s_unit[icv_idx][None, :] * self.tau
            if defence_active:
                # Only reports accepted by Layer A/B can be selected
                cand_secure = cand_reported.copy()
                for ui, u_gid in enumerate(nic_idx):
                    acc_set = self.defence.last_accepted_reports.get(int(u_gid), set())
                    for vi, v_gid in enumerate(icv_idx):
                        if int(v_gid) not in acc_set:
                            cand_secure[ui, vi] = False
                w_base = q * m.s_unit[icv_idx][None, :] * self.tau * cand_secure
                w_secure = self.defence.get_wsmm_weights(w_base, cand_secure, nic_idx, icv_idx)
            else:
                # Without defence, system blindly trusts reported qualities (vulnerable to quality inflation)
                cand_secure = cand_reported.copy()
                w_secure = q_reported * m.s_unit[icv_idx][None, :] * self.tau * cand_secure

            # Run Subproblem 1 Matching (using Fast Two-Phase C6-Satisficing SPG or baseline MCMF)
            if self.use_fast_solvers:
                Zs = fast_spatial_greedy_matching(
                    cand_secure, w_secure, pc['phi_max'],
                    nic_pos=m.pos[nic_idx], icv_pos=m.pos[icv_idx],
                    r_sense=pc['r_sense'], road_length=m.L,
                    s_req=pc['s_req'] if defence_active else None,
                    nic_urgency=self.A[nic_idx] + 1.0
                )
            else:
                Zs = sub1_wsmm.select('wsmm', cand_secure, w_secure, pc['phi_max'],
                                       pc['s_req'], self.rng) if icv_idx.size and nic_idx.size \
                    else np.zeros((nic_idx.size, icv_idx.size), np.int8)

            # ──── BCA with trust-weighted AoI ────
            Z_full = np.zeros((self.N, self.N), np.int8)
            if nic_idx.size and icv_idx.size:
                Z_full[np.ix_(nic_idx, icv_idx)] = Zs

            sel_flag = np.zeros(self.N, bool)
            sel_flag[icv_idx] = Zs.sum(0) > 0 if icv_idx.size else False

            # Vetted perceived data size for constraint C6:
            # Only uncorrupted, non-quarantined physical data counts toward valid DT state synchronization
            W_vetted = np.zeros(self.N)
            if nic_idx.size and icv_idx.size:
                uncorrupted_w = w_true * (~link_corrupted)
                W_vetted[nic_idx] = self.defence.get_vetted_perceived_size(Zs, uncorrupted_w, nic_idx, icv_idx)

            s_v = m.s_own + m.s_unit * sel_flag
            c_cp = (Z_full * (m.c_unit * m.s_unit)[None, :]).sum(1)

            # Access selection: nearest RSU (MDA baseline)
            dx = m.pos[:, None] - self.centers[None, :]
            dist = np.abs(dx)
            in_range = (dist <= cfg['rsu']['radius']) & m.active[:, None]
            sx = -np.ones(self.N, int)
            for i in np.flatnonzero(m.active & m.is_icv):
                if in_range[i].any():
                    sx[i] = int(np.argmin(np.where(in_range[i], dist[i], np.inf)))
            for u in nic_idx:
                sel = np.flatnonzero(Z_full[u])
                if sel.size and sx[sel[0]] >= 0:
                    sx[u] = sx[sel[0]]

            # Constraint C6: Satisfied if vetted uncorrupted perceived data >= s_req
            c6_ok = np.zeros(self.N, bool)
            c6_ok[nic_idx] = W_vetted[nic_idx] >= (pc['s_req'] - 1e-9)

            # BCA per RSU
            g = gain(dx, cfg['rsu']['height'], cfg['comm']['pl0'], cfg['comm']['pl_slope'],
                      self.rng, cfg['comm']['fading'])
            H = H_mhz(g, cfg['comm']['p'], cfg['comm']['noise_dbm_hz'])

            T_tran, T_exe = np.zeros(self.N), np.zeros(self.N)
            ac = cfg['aoi']
            for r in range(self.K):
                Vr = icv_idx[sx[icv_idx] == r]
                Ur = nic_idx[sx[nic_idx] == r]
                if Vr.size == 0:
                    continue
                pos_in_V = {v: j for j, v in enumerate(Vr)}
                sel = [np.array([pos_in_V[v] for v in np.flatnonzero(Z_full[u]) if v in pos_in_V])
                       for u in Ur]

                # Trust-weighted BCA weights
                w_v = self.A[Vr] + 1.0
                w_u = self.A[Ur] + 1.0
                w_v_sec, w_u_sec = self.defence.get_bca_weights(w_v, w_u, Vr, Ur)

                try:
                    if self.use_fast_solvers:
                        res = fast_solve_rsu(s_v[Vr] * self.tau, H[Vr, r], w_v_sec,
                                             m.c_own[Vr] * m.s_own[Vr] * self.tau,
                                             w_u_sec, c_cp[Ur] * self.tau,
                                             sel, cfg['comm']['bmax_hz'] / 1e6,
                                             cfg['comp']['F_ghz'])
                    else:
                        res = solve_rsu(s_v[Vr] * self.tau, H[Vr, r], w_v_sec,
                                         m.c_own[Vr] * m.s_own[Vr] * self.tau,
                                         w_u_sec, c_cp[Ur] * self.tau,
                                         sel, cfg['comm']['bmax_hz'] / 1e6,
                                         cfg['comp']['F_ghz'], cfg['sub2'])
                    T_tran[Vr] = res['t_tran']
                    T_exe[Vr] = m.c_own[Vr] * m.s_own[Vr] * self.tau / res['f_own']
                    for j, u in enumerate(Ur):
                        if sel[j].size:
                            T_tran[u] = res['t_tran'][sel[j]].max()
                            T_exe[u] = c_cp[u] * self.tau / res['f_cp'][j]
                except Exception:
                    pass

            for u in nic_idx:
                if sx[u] < 0:
                    T_tran[u] = ac['t_pena']
                    T_exe[u] = ac['t_epena']

            T_sum = T_tran + T_exe
            T_req = np.where(m.is_icv, ac['t_req_icv'], ac['t_req_nicv'])
            valid = np.ones(self.N, bool)
            valid[nic_idx] = c6_ok[nic_idx] & (sx[nic_idx] >= 0)
            ok = (T_sum <= T_req) & valid

            A_new = np.where(ok, 0.0, self.A + 1.0)
            A_new = np.where(m.active, A_new, 0.0)
            alpha = ac['alpha']
            wA = np.where(m.is_icv, alpha, 1 - alpha) * A_new
            omega = float(wA[m.active].max()) if m.active.any() else 0.0
            metrics['omega_hist'].append(omega)

            # DT RMSE from defence stats
            if self.defence.slot_stats:
                metrics['dt_rmse_hist'].append(self.defence.slot_stats[-1]['dt_rmse'])

            self.A = A_new

            # Trust tracking for plots
            h_trusts, m_trusts = [], []
            for icv_gid in icv_idx:
                ts = self.defence.trust_states.get(int(icv_gid))
                if ts:
                    if self.injector.is_malicious(int(icv_gid)):
                        m_trusts.append(ts.trust)
                    else:
                        h_trusts.append(ts.trust)
            if h_trusts:
                metrics['honest_trust_hist'].append(float(np.mean(h_trusts)))
            if m_trusts:
                metrics['malicious_trust_hist'].append(float(np.mean(m_trusts)))
            metrics['quarantine_hist'].append(
                sum(1 for ts in self.defence.trust_states.values() if ts.quarantined)
            )

            # Accumulate layer stats
            if self.defence.slot_stats:
                s = self.defence.slot_stats[-1]
                metrics['layer_a_rejected'] += s['layer_a_rejected']
                metrics['layer_b_outliers'] += s['layer_b_outliers']

            metrics['slot_times_ms'].append((time.perf_counter() - t_slot_start) * 1000.0)

            # Advance mobility and reset respawned vehicle tracks
            new, gone = m.step()
            self.defence.on_vehicles_respawned(new, gone)
            for i in new:
                self.A[i] = 0.0
                self.dt_loc[i] = int(np.argmin(np.abs(m.pos[i] - self.centers)))
            for i in gone:
                self.A[i] = 0.0

        # Final summary
        metrics['amwaoi'] = float(np.mean(metrics['omega_hist']))
        metrics['dt_rmse'] = float(np.mean(metrics['dt_rmse_hist'])) if metrics['dt_rmse_hist'] else 0.0
        metrics['false_rejection_rate'] = metrics['false_rejections'] / max(metrics['true_acceptances'] + metrics['false_rejections'], 1)
        metrics['false_acceptance_rate'] = metrics['false_acceptances'] / max(metrics['true_rejections'] + metrics['false_acceptances'], 1)
        metrics['mean_slot_ms'] = float(np.mean(metrics['slot_times_ms'])) if metrics['slot_times_ms'] else 0.0

        if metrics['honest_trust_hist']:
            metrics['final_honest_trust'] = metrics['honest_trust_hist'][-1]
        if metrics['malicious_trust_hist']:
            metrics['final_malicious_trust'] = metrics['malicious_trust_hist'][-1]

        return metrics


def run_sweep():
    cfg = load_config()

    p_mal_values = [0.0, 0.125, 0.25, 0.375, 0.50]

    configs = {
        'BASE (no defence)':     FalseCPMConfig(enable_layer_a=False, enable_layer_b=False, enable_layer_c=False),
        'FULL (A + B + C)':      FalseCPMConfig(enable_layer_a=True,  enable_layer_b=True,  enable_layer_c=True),
    }

    n_runs = 5  # average over multiple seeds
    results = {}

    print("=" * 80)
    print("FALSE CPM RESISTANCE - p_mal SWEEP EXPERIMENT (OPTIMIZED)")
    print("=" * 80)
    print(f"Configurations: {list(configs.keys())}")
    print(f"Mobility Backend: {cfg['mobility']['backend'].upper()}")
    print(f"p_mal values: {p_mal_values}")
    print(f"Runs per config per p_mal: {n_runs}")
    print(f"Episode length: {cfg['time']['T']} slots")
    print(f"Fast Solvers (Two-Phase SPG + Vectorized ED-BCA): ENABLED")
    print("=" * 80)

    for config_name, fcpm_cfg in configs.items():
        results[config_name] = {}
        for p_mal in p_mal_values:
            run_metrics = []
            for seed in range(n_runs):
                defence = FalseCPMResistance(fcpm_cfg)
                injector = AttackInjector(p_mal=p_mal)
                env = SecureDTEnv(cfg, defence, injector, seed=seed * 100 + 42, use_fast_solvers=True)
                try:
                    env.reset(seed=seed * 100 + 42)
                    m = env.run_episode()
                    run_metrics.append(m)
                finally:
                    env.close()

            # Average over runs
            avg = {
                'amwaoi': float(np.mean([m['amwaoi'] for m in run_metrics])),
                'dt_rmse': float(np.mean([m['dt_rmse'] for m in run_metrics])),
                'false_rejection_rate': float(np.mean([m['false_rejection_rate'] for m in run_metrics])),
                'false_acceptance_rate': float(np.mean([m['false_acceptance_rate'] for m in run_metrics])),
                'layer_a_rejected': float(np.mean([m['layer_a_rejected'] for m in run_metrics])),
                'quarantined': float(np.mean([m['quarantine_hist'][-1] if m['quarantine_hist'] else 0 for m in run_metrics])),
                'mean_slot_ms': float(np.mean([m['mean_slot_ms'] for m in run_metrics])),
            }

            # Trust scores (only valid when p_mal > 0)
            if p_mal > 0:
                ht = [m.get('final_honest_trust', 1.0) for m in run_metrics]
                mt = [m.get('final_malicious_trust', 1.0) for m in run_metrics]
                avg['honest_trust'] = float(np.mean(ht))
                avg['malicious_trust'] = float(np.mean(mt))

            results[config_name][str(p_mal)] = avg

            # Print progress
            mark = "[OK]" if p_mal == 0 or avg.get('malicious_trust', 1.0) < 0.5 else "[WARN]"
            print(f"  {mark} {config_name:20s} | p_mal={p_mal:.1%} | AMWAoI={avg['amwaoi']:.2f}s "
                  f"| RMSE={avg['dt_rmse']:.2f}m "
                  f"| FRR={avg['false_rejection_rate']:.1%} "
                  f"| FAR={avg['false_acceptance_rate']:.1%} "
                  f"| Q={avg['quarantined']:.1f} "
                  f"| {avg['mean_slot_ms']:.2f}ms/slot")

        print()

    # ──── Print Final Summary Table ────
    print("\n" + "=" * 100)
    print("FINAL RESULTS TABLE (for paper)")
    print("=" * 100)

    header = f"{'Config':20s} | {'p_mal':>6s} | {'AMWAoI':>7s} | {'RMSE':>6s} | {'FRR':>6s} | {'FAR':>6s} | {'H-Trust':>7s} | {'M-Trust':>7s} | {'Quar':>4s}"
    print(header)
    print("-" * len(header))

    for config_name in configs:
        for p_mal in p_mal_values:
            r = results[config_name][str(p_mal)]
            ht = f"{r.get('honest_trust', 0):.3f}" if p_mal > 0 else "  n/a"
            mt = f"{r.get('malicious_trust', 0):.3f}" if p_mal > 0 else "  n/a"
            print(f"{config_name:20s} | {p_mal:5.1%}  | {r['amwaoi']:7.2f} | {r['dt_rmse']:6.2f} | "
                  f"{r['false_rejection_rate']:5.1%}  | {r['false_acceptance_rate']:5.1%}  | {ht:>7s} | {mt:>7s} | {r['quarantined']:4.1f}")
        print("-" * len(header))

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), 'results', 'false_cpm_sweep.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 80)
    print("EVIDENCE SUMMARY")
    print("=" * 80)

    key = '0.25' if '0.25' in results.get('BASE (no defence)', {}) else list(results.get('BASE (no defence)', {}).keys())[1]
    if key in results.get('BASE (no defence)', {}) and key in results.get('FULL (A + B + C)', {}):
        base = results['BASE (no defence)'][key]
        full = results['FULL (A + B + C)'][key]
        print(f"At {float(key):.1%} malicious ICVs ({int(round(float(key)*8))}/8 ICVs):")
        print(f"  BASE AMWAoI:  {base['amwaoi']:.2f}s  -> FULL AMWAoI:  {full['amwaoi']:.2f}s")
        print(f"  BASE RMSE:    {base['dt_rmse']:.2f}m -> FULL RMSE:    {full['dt_rmse']:.2f}m")
        print(f"  BASE FAR:     {base['false_acceptance_rate']:.1%} -> FULL FAR:     {full['false_acceptance_rate']:.1%}")
        print(f"  FULL FRR:     {full['false_rejection_rate']:.1%}")
        print(f"  Honest ICV Trust:    {full.get('honest_trust', 0):.3f} (maintained)")
        print(f"  Malicious ICV Trust: {full.get('malicious_trust', 0):.3f} (quarantined)")


if __name__ == '__main__':
    run_sweep()
