"""Final table: reproduced vs paper (percentage error). Usage: python -m experiments.compare results/exp2_V_*.json V"""
import json, sys
from experiments.exp2_asdmm import PAPER_V, PAPER_U, VS, US

if __name__ == '__main__':
    res = json.load(open(sys.argv[1])); kind = sys.argv[2]
    xs, paper = (VS, PAPER_V) if kind == 'V' else (US, PAPER_U)
    print('%-9s %-4s %8s %8s %8s' % ('method', 'x', 'ours', 'paper', 'err%'))
    for m, rows in res.items():
        for x, r, p in zip(xs, rows, paper.get(m, [float('nan')] * len(xs))):
            print('%-9s %-4s %8.2f %8.2f %7.1f%%' % (m, x, r['amwaoi'], p, 100 * (r['amwaoi'] - p) / p))
