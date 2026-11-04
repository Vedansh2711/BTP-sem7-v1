"""Diagnostic: where does the 37.5% vs 50% non-monotonicity come from?

Measures, per p_mal:
  - Layer B pool size (non-quarantined reports entering layer_b_fuse)
  - Layer B wall time per call
  - process_slot wall time
  - final quarantine composition (honest vs malicious)
  - final honest/malicious trust, AMWAoI, RMSE, FRR, FAR
Uses kinematic mobility so it runs without SUMO.
"""
import sys, os, time
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from exp_false_cpm import SecureDTEnv
from false_cpm_resistance import FalseCPMResistance, FalseCPMConfig
from attack_injector import AttackInjector


def run_one(cfg, p_mal, seed):
    defence = FalseCPMResistance(FalseCPMConfig(
        enable_layer_a=True, enable_layer_b=True, enable_layer_c=True))
    injector = AttackInjector(p_mal=p_mal)
    env = SecureDTEnv(cfg, defence, injector, seed=seed, use_fast_solvers=True)
    env.reset(seed=seed)

    pool_sizes, lb_times, ps_times = [], [], []

    orig_lb = defence.layer_b_fuse
    def wrapped_lb(target_id, accepted_reports):
        t0 = time.perf_counter()
        out = orig_lb(target_id, accepted_reports)
        lb_times.append((time.perf_counter() - t0) * 1e6)  # us
        pool_sizes.append(len(accepted_reports))
        return out
    defence.layer_b_fuse = wrapped_lb

    orig_ps = defence.process_slot
    def wrapped_ps(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_ps(*args, **kwargs)
        ps_times.append((time.perf_counter() - t0) * 1e6)  # us
        return out
    defence.process_slot = wrapped_ps

    metrics = env.run_episode()

    active_icv = [i for i in range(env.N) if env.mob.active[i] and env.mob.is_icv[i]]
    mal_set = injector.malicious_set
    hq = sum(1 for i in active_icv if i not in mal_set
             and defence.trust_states.get(i) is not None
             and defence.trust_states[i].quarantined)
    mq = sum(1 for i in active_icv if i in mal_set
             and defence.trust_states.get(i) is not None
             and defence.trust_states[i].quarantined)
    return dict(
        amwaoi=metrics['amwaoi'], rmse=metrics['dt_rmse'],
        frr=metrics['false_rejection_rate'], far=metrics['false_acceptance_rate'],
        ht=metrics.get('final_honest_trust', float('nan')),
        mt=metrics.get('final_malicious_trust', float('nan')),
        # quarantine composition among *active* ICVs at episode end
        honest_quarantined=hq, malicious_quarantined=mq,
        n_honest=len(active_icv) - sum(1 for i in active_icv if i in mal_set),
        n_malicious=sum(1 for i in active_icv if i in mal_set),
        mean_pool=float(np.mean(pool_sizes)) if pool_sizes else 0.0,
        max_pool=int(np.max(pool_sizes)) if pool_sizes else 0,
        lb_us=float(np.mean(lb_times)) if lb_times else 0.0,
        ps_us=float(np.mean(ps_times)) if ps_times else 0.0,
    )


def main():
    with open(os.path.join(ROOT, 'config.yaml')) as f:
        cfg = yaml.safe_load(f)
    cfg['mobility']['backend'] = 'kinematic'   # avoid SUMO dependency

    seeds = [42, 142, 242]
    print(f"{'p_mal':>6} | {'AMWAoI':>7} | {'RMSE':>7} | {'FRR':>6} | {'FAR':>6} | "
          f"{'H-trust':>7} | {'M-trust':>7} | {'Qh':>4} | {'Qm':>4} | "
          f"{'pool':>6} | {'maxpool':>7} | {'LB us':>7} | {'slot us':>8}")
    print("-" * 131)
    for p_mal in [0.25, 0.375, 0.5]:
        rows = [run_one(cfg, p_mal, s) for s in seeds]
        a = lambda k: float(np.mean([r[k] for r in rows]))
        print(f"{p_mal:6.1%} | {a('amwaoi'):7.2f} | {a('rmse'):7.2f} | "
              f"{a('frr'):6.1%} | {a('far'):6.1%} | {a('ht'):7.3f} | {a('mt'):7.3f} | "
              f"{a('honest_quarantined'):4.1f} | {a('malicious_quarantined'):4.1f} | "
              f"{a('mean_pool'):6.2f} | {int(np.max([r['max_pool'] for r in rows])):7d} | "
              f"{a('lb_us'):7.1f} | {a('ps_us'):8.1f}")
    print("\nQh/Qm = honest/malicious ICVs quarantined at episode end")
    print("pool = mean reports entering layer_b_fuse (non-quarantined, Layer-A-passed)")


if __name__ == '__main__':
    main()
