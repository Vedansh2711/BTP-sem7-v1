"""
Experiment: False CPM Resistance — p_mal Sweep
================================================
Integrates the 3-layer defence into Vedansh's base paper reproduction.
Runs ablation: BASE, BASE+A, BASE+A+B, FULL (A+B+C) across p_mal = 0% to 40%.

Outputs: DT RMSE, AMWAoI, False Rejection Rate, False Acceptance Rate,
         Trust Convergence Time, Quarantine Count.

This script does NOT require torch (uses MDA baseline agent for speed).
Run: python3 exp_false_cpm.py
"""
import sys, os, copy, json
import numpy as np

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.mobility import KinematicMobility, make_mobility
from env.channel import gain, H_mhz
import sub1_wsmm
from sub2_bca import solve_rsu
from false_cpm_resistance import FalseCPMResistance, FalseCPMConfig, CPMReport
from attack_injector import AttackInjector


def load_config():
    import yaml
    with open(os.path.join(os.path.dirname(__file__), 'config.yaml')) as f:
        return yaml.safe_load(f)


class SecureDTEnv:
    """
    Simplified DTEnv that runs the per-slot pipeline WITH the 3-layer defence.
    Uses MDA (nearest RSU) access selection for simplicity (no torch needed).
    Focuses on measuring security/trust metrics rather than DRL training.
    """

    def __init__(self, cfg, defence: FalseCPMResistance, injector: AttackInjector, seed=0):
        self.cfg = cfg
        self.defence = defence
        self.injector = injector
        self.rng = np.random.default_rng(seed)
        self.mob = KinematicMobility(cfg, self.rng)
        self.N = self.mob.N
        self.K = cfg['rsu']['K']
        self.tau = cfg['time']['tau']
        self.centers = np.asarray(cfg['rsu']['centers'])

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.mob.rng = self.rng
        self.mob.reset()
        self.defence.reset()

        N = self.N
        self.A = np.zeros(N)
        self.dt_loc = np.zeros(N, int)
        for i in np.flatnonzero(self.mob.active):
            self.dt_loc[i] = int(np.argmin(np.abs(self.mob.pos[i] - self.centers)))

        # Mark malicious ICVs
        icv_idx = np.flatnonzero(self.mob.active & self.mob.is_icv)
        self.injector.mark_malicious(icv_idx, self.rng)

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
        }

        for t in range(T):
            m = self.mob
            icv_idx = np.flatnonzero(m.active & m.is_icv)
            nic_idx = np.flatnonzero(m.active & ~m.is_icv)

            if icv_idx.size == 0 or nic_idx.size == 0:
                m.step()
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
            cpms = {}  # target_id → list of CPMReport
            ground_truth_pos = {}  # target_id → true position

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

                    # Attack injection
                    if self.injector.is_malicious(int(v_gid)):
                        attacked = self.injector.attack(honest, self.rng, m.L)
                        if attacked is None:
                            # Suppression attack — don't add any report
                            continue
                        report = attacked
                        is_malicious_report = True
                    else:
                        report = honest
                        is_malicious_report = False

                    # Tag for metric tracking (NOT visible to defence)
                    report._is_malicious = is_malicious_report
                    cpms[int(u_gid)].append(report)

            # ──── Run the 3-Layer Pipeline ────
            fused_pos, trust_vec, fusion_conf = self.defence.process_slot(
                t, cpms, true_pos=ground_truth_pos
            )

            # ──── Count false rejections / false acceptances ────
            for target_id, reports in cpms.items():
                for r in reports:
                    metrics['total_reports'] += 1
                    # Check if this reporter ended up with trust=0 (quarantined) or failed layers
                    reporter_trust = trust_vec.get(r.reporter_id, 1.0)
                    was_accepted = reporter_trust > 0.3  # approximation

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

            # ──── Modify WSMM weights with trust ────
            w = q * m.s_unit[icv_idx][None, :] * self.tau
            cand_secure = cand.copy()

            # Apply trust to weights (our integration formula)
            w_secure = self.defence.get_wsmm_weights(w, cand_secure, nic_idx, icv_idx)

            # Run WSMM with trust-weighted inputs
            Zs = sub1_wsmm.select('wsmm', cand_secure, w_secure, pc['phi_max'],
                                   pc['s_req'], self.rng) if icv_idx.size and nic_idx.size \
                else np.zeros((nic_idx.size, icv_idx.size), np.int8)

            # ──── BCA with trust-weighted AoI ────
            Z_full = np.zeros((self.N, self.N), np.int8)
            if nic_idx.size and icv_idx.size:
                Z_full[np.ix_(nic_idx, icv_idx)] = Zs

            sel_flag = np.zeros(self.N, bool)
            sel_flag[icv_idx] = Zs.sum(0) > 0 if icv_idx.size else False
            W = np.zeros(self.N)
            if nic_idx.size and icv_idx.size:
                W[nic_idx] = (Zs * w_secure).sum(1)

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

            c6_ok = np.zeros(self.N, bool)
            c6_ok[nic_idx] = W[nic_idx] >= pc['s_req'] - 1e-9

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

            # Advance mobility
            m.step()
            # Re-mark malicious after new vehicles enter
            new_icv = np.flatnonzero(m.active & m.is_icv)
            for v in new_icv:
                if int(v) not in self.injector.malicious_set and int(v) not in self.defence.trust_states:
                    # New vehicle — might be malicious
                    if self.rng.random() < self.injector.p_mal:
                        self.injector.malicious_set.add(int(v))
                        attack_types = list(self.injector.attack_mix.keys())
                        self.injector.attack_type[int(v)] = self.rng.choice(attack_types)

        # Final summary
        metrics['amwaoi'] = float(np.mean(metrics['omega_hist']))
        metrics['dt_rmse'] = float(np.mean(metrics['dt_rmse_hist'])) if metrics['dt_rmse_hist'] else 0.0
        tot = metrics['total_reports']
        metrics['false_rejection_rate'] = metrics['false_rejections'] / max(metrics['true_acceptances'] + metrics['false_rejections'], 1)
        metrics['false_acceptance_rate'] = metrics['false_acceptances'] / max(metrics['true_rejections'] + metrics['false_acceptances'], 1)

        if metrics['honest_trust_hist']:
            metrics['final_honest_trust'] = metrics['honest_trust_hist'][-1]
        if metrics['malicious_trust_hist']:
            metrics['final_malicious_trust'] = metrics['malicious_trust_hist'][-1]

        return metrics


def run_sweep():
    cfg = load_config()

    p_mal_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    
    configs = {
        'BASE (no defence)':     FalseCPMConfig(enable_layer_a=False, enable_layer_b=False, enable_layer_c=False),
        'BASE + Layer A':        FalseCPMConfig(enable_layer_a=True,  enable_layer_b=False, enable_layer_c=False),
        'BASE + A + B':          FalseCPMConfig(enable_layer_a=True,  enable_layer_b=True,  enable_layer_c=False),
        'FULL (A + B + C)':      FalseCPMConfig(enable_layer_a=True,  enable_layer_b=True,  enable_layer_c=True),
    }

    n_runs = 5  # average over multiple seeds
    results = {}

    print("=" * 80)
    print("FALSE CPM RESISTANCE — p_mal SWEEP EXPERIMENT")
    print("=" * 80)
    print(f"Configurations: {list(configs.keys())}")
    print(f"p_mal values: {p_mal_values}")
    print(f"Runs per config per p_mal: {n_runs}")
    print(f"Episode length: {cfg['time']['T']} slots")
    print("=" * 80)

    for config_name, fcpm_cfg in configs.items():
        results[config_name] = {}
        for p_mal in p_mal_values:
            run_metrics = []
            for seed in range(n_runs):
                defence = FalseCPMResistance(fcpm_cfg)
                injector = AttackInjector(p_mal=p_mal)
                env = SecureDTEnv(cfg, defence, injector, seed=seed * 100 + 42)
                env.reset(seed=seed * 100 + 42)
                m = env.run_episode()
                run_metrics.append(m)

            # Average over runs
            avg = {
                'amwaoi': float(np.mean([m['amwaoi'] for m in run_metrics])),
                'dt_rmse': float(np.mean([m['dt_rmse'] for m in run_metrics])),
                'false_rejection_rate': float(np.mean([m['false_rejection_rate'] for m in run_metrics])),
                'false_acceptance_rate': float(np.mean([m['false_acceptance_rate'] for m in run_metrics])),
                'layer_a_rejected': float(np.mean([m['layer_a_rejected'] for m in run_metrics])),
                'quarantined': float(np.mean([m['quarantine_hist'][-1] if m['quarantine_hist'] else 0 for m in run_metrics])),
            }

            # Trust scores (only valid when p_mal > 0)
            if p_mal > 0:
                ht = [m.get('final_honest_trust', 1.0) for m in run_metrics]
                mt = [m.get('final_malicious_trust', 1.0) for m in run_metrics]
                avg['honest_trust'] = float(np.mean(ht))
                avg['malicious_trust'] = float(np.mean(mt))

            results[config_name][str(p_mal)] = avg

            # Print progress
            mark = "✅" if p_mal == 0 or avg.get('malicious_trust', 1.0) < 0.5 else "⚠️"
            print(f"  {mark} {config_name:25s} | p_mal={p_mal:.0%} | AMWAoI={avg['amwaoi']:.2f} "
                  f"| RMSE={avg['dt_rmse']:.2f}m "
                  f"| FRR={avg['false_rejection_rate']:.1%} "
                  f"| FAR={avg['false_acceptance_rate']:.1%} "
                  f"| Q={avg['quarantined']:.0f}")

        print()

    # ──── Print Final Summary Table ────
    print("\n" + "=" * 100)
    print("FINAL RESULTS TABLE (for paper)")
    print("=" * 100)

    header = f"{'Config':25s} | {'p_mal':>6s} | {'AMWAoI':>7s} | {'RMSE':>6s} | {'FRR':>6s} | {'FAR':>6s} | {'H-Trust':>7s} | {'M-Trust':>7s} | {'Quar':>4s}"
    print(header)
    print("-" * len(header))

    for config_name in configs:
        for p_mal in p_mal_values:
            r = results[config_name][str(p_mal)]
            ht = f"{r.get('honest_trust', 0):.3f}" if p_mal > 0 else "  n/a"
            mt = f"{r.get('malicious_trust', 0):.3f}" if p_mal > 0 else "  n/a"
            print(f"{config_name:25s} | {p_mal:5.0%}  | {r['amwaoi']:7.2f} | {r['dt_rmse']:6.2f} | "
                  f"{r['false_rejection_rate']:5.1%}  | {r['false_acceptance_rate']:5.1%}  | {ht:>7s} | {mt:>7s} | {r['quarantined']:4.0f}")
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

    # Compare BASE vs FULL at p_mal=0.2
    if '0.2' in results.get('BASE (no defence)', {}) and '0.2' in results.get('FULL (A + B + C)', {}):
        base = results['BASE (no defence)']['0.2']
        full = results['FULL (A + B + C)']['0.2']
        print(f"At 20% malicious ICVs:")
        print(f"  BASE AMWAoI:  {base['amwaoi']:.2f}   → FULL AMWAoI:  {full['amwaoi']:.2f}")
        print(f"  BASE RMSE:    {base['dt_rmse']:.2f}m  → FULL RMSE:    {full['dt_rmse']:.2f}m")
        print(f"  BASE FAR:     {base['false_acceptance_rate']:.1%}  → FULL FAR:     {full['false_acceptance_rate']:.1%}")
        print(f"  Honest ICV Trust:    {full.get('honest_trust', 0):.3f} (maintained)")
        print(f"  Malicious ICV Trust: {full.get('malicious_trust', 0):.3f} (quarantined)")


if __name__ == '__main__':
    run_sweep()
