"""Exp1 (Figs 4-7): WSMM vs SM vs GA on random snapshots. |U|=3, |V| in {4,6,8,10,12}."""
import argparse, json, os, time
import numpy as np, yaml
import sub1_wsmm

PAPER_AVG = {'wsmm': [3.7, 4.65, 5.1, 5.3, 5.35], 'sm': [3.3, 3.8, 4.3, 4.6, 4.85], 'ga': [3.15, 3.6, 3.9, 4.03, 4.17]}
PAPER_VIOL = {'wsmm': [.21, .11, .02, .01, .01], 'sm': [.34, .205, .09, .05, .02], 'ga': [.39, .26, .18, .08, .05]}
VS = [4, 6, 8, 10, 12]


def snapshot(cfg, nV, nU, rng):
    pc, v = cfg['perception'], cfg['vehicle']
    L = cfg['road']['length']
    pu, pv = rng.uniform(0, L, nU), rng.uniform(0, L, nV)
    sep = np.abs(pu[:, None] - pv[None, :])
    cand = sep <= pc['r_sense']
    if pc['q_model'] == 'uniform':
        q = rng.uniform(pc['q_low'], pc['q_high'], sep.shape)
    else:
        q = np.clip(pc['q_high'] * (1 - sep / pc['r_sense']) + pc['q_low'], pc['q_low'], pc['q_high'])
    s_unit = rng.uniform(*v['s_unit'], nV)
    return cand, q * s_unit[None, :] * cfg['time']['tau'], -sep


def run(cfg, nU=3, vs=VS, n_snap=1000, methods=('wsmm', 'sm', 'ga'), n_ga=None, seed=0, sm_pref='q'):
    out = {m: {'avg': [], 'viol': [], 'samples': {}, 'paired_avg': [], 'paired_viol': []} for m in methods}
    phi, sreq = cfg['perception']['phi_max'], cfg['perception']['s_req']
    for nV in vs:
        samples = {m: [] for m in methods}; paired = {m: [] for m in methods}
        rng = np.random.default_rng(seed + nV)
        for s in range(n_snap):
            cand, w, nsep = snapshot(cfg, nV, nU, rng)
            for m in methods:
                if m == 'ga' and n_ga is not None and s >= n_ga:
                    continue
                Z = sub1_wsmm.select(m, cand, w, phi, sreq, np.random.default_rng(s), cfg['ga'],
                                     icv_score=nsep if sm_pref == 'distance' else None)
                samples[m].extend((Z * w).sum(1).tolist())                  # per N-ICV weighted size [Mb]
                if n_ga is None or s < n_ga:
                    paired[m].extend((Z * w).sum(1).tolist())              # identical snapshots for every method
        for m in methods:
            a = np.array(samples[m])
            out[m]['avg'].append(float(a.mean())); out[m]['viol'].append(float((a < sreq).mean()))
            out[m]['samples'][nV] = a
            pa = np.array(paired[m]); out[m]['paired_avg'].append(float(pa.mean())); out[m]['paired_viol'].append(float((pa < sreq).mean()))
    return out


def plot(out, path):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for m in out:
        ax[0].plot(VS, out[m]['avg'], 'o-', label=m.upper()); ax[1].plot(VS, out[m]['viol'], 'o-', label=m.upper())
    ax[0].axhline(4, ls=':', c='k'); ax[0].set(xlabel='|V|', ylabel='avg weighted sum (Mb)', title='Fig 4'); ax[1].set(xlabel='|V|', ylabel='P[sum<4Mb]', title='Fig 5')
    ax[0].legend(); fig.tight_layout(); fig.savefig(path + '_fig4_5.png', dpi=130)
    for fig_no, nV in [(6, 6), (7, 8)]:
        f, a = plt.subplots(figsize=(4.5, 3.6))
        for m in out:
            x = np.sort(out[m]['samples'][nV]); a.plot(x, np.arange(1, x.size + 1) / x.size, label=m.upper())
        a.axvline(4, ls=':', c='k'); a.set(xlabel='weighted sum (Mb)', ylabel='CDF', title=f'Fig {fig_no} (|V|={nV})'); a.legend(); f.tight_layout()
        f.savefig(path + f'_fig{fig_no}.png', dpi=130)


def report(out):
    print('%-5s %-4s | ' % ('meth', ''), ' '.join('V=%-2d' % v for v in VS))
    for m in out:
        for name, key, ref in [('avg', 'avg', PAPER_AVG), ('viol', 'viol', PAPER_VIOL)]:
            print('%-5s %-4s | ' % (m, name), ' '.join('%5.2f' % x for x in out[m][key]), '| paper', ref[m])
    print('--- paired comparison (identical snapshots for all methods) ---')
    for m in out:
        print('%-5s avg  | ' % m, ' '.join('%5.2f' % x for x in out[m]['paired_avg']), ' viol |', ' '.join('%5.2f' % x for x in out[m]['paired_viol']))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml'); ap.add_argument('--snap', type=int, default=1000)
    ap.add_argument('--ga-snap', type=int, default=150); ap.add_argument('--rsense', type=float)
    ap.add_argument('--out', default='results/exp1'); ap.add_argument('--sm-pref', default='q', choices=['q', 'distance'])
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.rsense: cfg['perception']['r_sense'] = a.rsense
    t = time.time(); out = run(cfg, n_snap=a.snap, n_ga=a.ga_snap, sm_pref=a.sm_pref)
    report(out); plot(out, a.out)
    json.dump({m: {k: out[m][k] for k in ('avg', 'viol', 'paired_avg', 'paired_viol')} for m in out}, open(a.out + '.json', 'w'), indent=1)
    print('time %.0fs' % (time.time() - t))
