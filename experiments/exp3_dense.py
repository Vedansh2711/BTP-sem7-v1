"""Exp3 (Fig 13): 1000 veh/h, penetration 0.60..0.80. Metrics: comm delay, comp delay, actual migration delay
(mean T_amigr per migration event), AMWAoI. Default trains once per method at the middle penetration and
evaluates across the sweep (`--train-per-point` retrains at each point)."""
import argparse, copy
import numpy as np
from env.dt_env import DTEnv
from sub3_asdmm.train import train, evaluate
from experiments.common import load, make_agent, dump

PEN = [0.60, 0.65, 0.70, 0.75, 0.80]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--methods', default='proposed,ddpg,mda'); ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--eval-eps', type=int, default=5); ap.add_argument('--train-per-point', action='store_true')
    ap.add_argument('--bca-iters', type=int); ap.add_argument('--out', default='results/exp3')
    a = ap.parse_args()
    cfg = load(); cfg['population']['dynamic'] = True
    if a.bca_iters: cfg['sub2']['max_iter'] = a.bca_iters
    res = {}
    for m in a.methods.split(','):
        res[m] = []
        c_mid = copy.deepcopy(cfg); c_mid['population']['penetration'] = 0.70
        env0 = DTEnv(c_mid, seed=0); env0.reset()
        agent = make_agent(m, env0, c_mid)
        predictive, sub1 = env0.predictive, env0.sub1
        if agent.name != 'mda' and not a.train_per_point:
            train(env0, agent, a.epochs, verbose=True)          # one training run at penetration 0.70
        for pen in PEN:
            c = copy.deepcopy(cfg); c['population']['penetration'] = pen
            env = DTEnv(c, seed=0); env.reset()
            env.predictive, env.sub1 = predictive, sub1
            if a.train_per_point and agent.name != 'mda':
                agent = make_agent(m, env, c); train(env, agent, a.epochs, verbose=True)
            r = evaluate(env, agent, a.eval_eps); res[m].append(r)
            print(m, pen, {k: round(v, 3) for k, v in r.items()}, flush=True)
    dump(res, a.out + '.json')
