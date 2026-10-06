"""Plot Figs 8-13 from result JSONs: python -m experiments.plots <sweep_json> <V|U|curve|exp3>"""
import json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from experiments.exp2_asdmm import PAPER_V, PAPER_U, VS, US


def sweep(path, kind, out):
    res = json.load(open(path)); xs, paper = (VS, PAPER_V) if kind == 'V' else (US, PAPER_U)
    plt.figure(figsize=(4.8, 3.6))
    for m, rows in res.items():
        l, = plt.plot(xs, [r['amwaoi'] for r in rows], 'o-', label=m)
        if m in paper: plt.plot(xs, paper[m], 'x--', c=l.get_color(), alpha=.5)
    plt.xlabel('|V|' if kind == 'V' else '|U|'); plt.ylabel('AMWAoI'); plt.legend(title='solid=ours, dashed=paper'); plt.tight_layout(); plt.savefig(out, dpi=130)


def curve(path, out):
    res = json.load(open(path)); plt.figure(figsize=(4.8, 3.6))
    for m, d in res.items():
        if d['reward']: plt.plot(d['reward'], label=m)
        else: plt.axhline(d['eval']['reward'], ls=':', label=m)
    plt.xlabel('epoch'); plt.ylabel('mean per-step reward'); plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=130)


def exp3(path, out):
    res = json.load(open(path)); pen = [0.6, 0.65, 0.7, 0.75, 0.8]
    f, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    for a, (k, t) in zip(ax, [('comm', 'comm delay (s)'), ('comp', 'comp delay (s)'), ('mig_delay', 'actual migration delay (s)'), ('amwaoi', 'AMWAoI')]):
        for m, rows in res.items(): a.plot(pen, [r[k] for r in rows], 'o-', label=m)
        a.set(xlabel='penetration', ylabel=t); a.legend()
    f.tight_layout(); f.savefig(out, dpi=130)


if __name__ == '__main__':
    p, k = sys.argv[1], sys.argv[2]; out = p.replace('.json', '.png')
    {'V': lambda: sweep(p, 'V', out), 'U': lambda: sweep(p, 'U', out), 'curve': lambda: curve(p, out), 'exp3': lambda: exp3(p, out)}[k]()
    print('wrote', out)
