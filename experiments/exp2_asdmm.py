"""Exp2 (Figs 8-12). Sweeps: |V| in 4..12 with |U|=3 (Figs 8,10) ; |U| in 2..6 with |V|=8 (Figs 9,11);
training curve (Fig 12). Methods: proposed(PPO+WSMM) ddpg mda ppo_sm ddpg_sm mda_sm.
Torch is required for proposed/ddpg*. `--methods mda` runs without torch."""
import argparse, copy, time
import numpy as np
from env.dt_env import DTEnv
from sub3_asdmm.train import train, evaluate
from experiments.common import load, make_agent, dump

PAPER_V = {'proposed': [5.5, 4.5, 3.9, 3.65, 3.5], 'ddpg': [6.8, 6.0, 5.5, 5.35, 5.3], 'mda': [9.0, 8.2, 7.2, 7.1, 6.85]}
PAPER_U = {'proposed': [3.0, 4.3, 6.0, 9.0, 13.4], 'ddpg': [3.7, 5.0, 7.2, 11.7, 15.2], 'mda': [4.9, 8.1, 13.1, 18.2, 20.5]}
VS, US = [4, 6, 8, 10, 12], [2, 3, 4, 5, 6]


def one_point(cfg, nV, nU, method, epochs, eval_eps, seed):
    c = copy.deepcopy(cfg); c['population'].update(n_icv=nV, n_nicv=nU)
    env = DTEnv(c, seed=seed); env.reset()
    agent = make_agent(method, env, c, seed)
    hist = train(env, agent, epochs, verbose=True) if (epochs > 0 and agent.name != 'mda') else []
    return evaluate(env, agent, eval_eps), hist


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', choices=['V', 'U', 'curve'], required=True)
    ap.add_argument('--methods', default='proposed,ddpg,mda'); ap.add_argument('--epochs', type=int, default=600)
    ap.add_argument('--eval-eps', type=int, default=10); ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--bca-iters', type=int, help='cap Subproblem-2 iterations (speed)'); ap.add_argument('--out', default='results/exp2')
    a = ap.parse_args()
    cfg = load(a.config)
    if a.bca_iters: cfg['sub2']['max_iter'] = a.bca_iters
    methods = a.methods.split(',')
    res = {}
    if a.sweep == 'curve':                                     # Fig 12: nV=8, nU=3
        for m in methods:
            r, hist = one_point(cfg, 8, 3, m, a.epochs, a.eval_eps, 0)
            res[m] = dict(eval=r, reward=[h['reward'] for h in hist])
    else:
        xs = VS if a.sweep == 'V' else US
        for m in methods:
            res[m] = []
            for x in xs:
                nV, nU = (x, 3) if a.sweep == 'V' else (8, x)
                t = time.time(); r, _ = one_point(cfg, nV, nU, m, a.epochs, a.eval_eps, 0)
                res[m].append(r); print(m, a.sweep, x, 'AMWAoI %.2f  (%.0fs)' % (r['amwaoi'], time.time() - t), flush=True)
    dump(res, f'{a.out}_{a.sweep}_{"-".join(methods)}.json')
