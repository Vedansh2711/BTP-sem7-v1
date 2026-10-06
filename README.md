# dtmig — reproduction of "Cooperative Perception Aided DT Model Update and Migration in Mixed Vehicular Networks"

Requirements: Python 3.8+, numpy, scipy, pyyaml, matplotlib, networkx (tests only), **torch 2.1** (PPO/DDPG), optional cvxpy.
All parameters are in `config.yaml`; every deviation/choice is listed in `ASSUMPTIONS.md`.

## Commands
```
python -m unittest discover -s tests -t .                      # unit tests (also runs under pytest)
python -m experiments.exp1_icv_selection --snap 600 --ga-snap 150           # Figs 4-7 -> results/exp1_*.png
python -m experiments.exp2_asdmm --sweep V --methods proposed,ddpg,mda,ppo_sm,ddpg_sm,mda_sm   # Figs 8, 10
python -m experiments.exp2_asdmm --sweep U --methods proposed,ddpg,mda,ppo_sm,ddpg_sm,mda_sm   # Figs 9, 11
python -m experiments.exp2_asdmm --sweep curve --methods proposed,ddpg,mda                      # Fig 12
python -m experiments.exp3_dense --methods proposed,ddpg,mda                                    # Fig 13
python -m experiments.plots results/<file>.json V|U|curve|exp3 ; python -m experiments.compare results/<file>.json V|U
```
Speed: one 150-slot episode costs ~9 s with the default BCA (`sub2.max_iter=150`), i.e. ~1.5 h per 600-epoch training run on one CPU core.
`--bca-iters 40` roughly halves this at a small accuracy cost. Exp2 trains one agent per sweep point and method (30+ runs): budget accordingly or lower `--epochs`.

## Status (what was actually executed when this repo was written)
| Part | Status |
|---|---|
| env, WSMM/SM/GA, BCA, delay/AoI/migration logic | implemented; **17 unit tests pass** (WSMM = brute force + SciPy + networkx max-flow; BCA within 0.45% of SLSQP worst-case over 25 random instances; closed-form f = SLSQP; hand-computed delays, Eq (4) failure, AoI recursion, T_pre, C9) |
| Exp1 (Figs 4-7) | run (400 snapshots; GA 150) |
| MDA / MDA-SM baselines (Exp2, Exp3) | run, **not calibrated** |
| PPO (proposed), DDPG, PPO-SM, DDPG-SM, training curves | code written and byte-compiled only. **Not executed**: torch was not installable in the sandbox. Expect to debug shapes/hyper-parameters on first run. |
| SUMO/TraCI | stub only (kinematic fallback used) |

## Exp1 results (R_sense = 300 m, q ~ U[0.2,2]), ours vs paper
| | |V|=4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| WSMM avg Mb (paired) | 2.90 | 4.65 | 5.93 | 7.10 | 8.21 |
| paper WSMM | 3.7 | 4.65 | 5.1 | 5.3 | 5.35 |
| SM avg Mb | 2.90 | 4.65 | 5.93 | 7.08 | 8.16 |
| GA avg Mb | 2.86 | 4.50 | 5.82 | 6.99 | 8.11 |
| WSMM viol. prob | .67 | .48 | .35 | .28 | .19 |
| paper WSMM viol. | .21 | .11 | .02 | .01 | .01 |

Findings (honest summary):
- WSMM >= SM >= GA in average weighted size holds on identical snapshots, but **the gaps are 0-3%, far smaller than the paper's** (WSMM-SM ~ 15-25%).
  Cause: under the literal "both sides rank by q*s_unit" rule SM is nearly the same as WSMM (ASSUMPTIONS A3). With ICVs ranking by proximity (`--sm-pref distance`) SM is ~5-8% below WSMM.
- **Violation probability is NOT reproduced** (ours .19-.67 vs paper .01-.21) and GA's violation probability is sometimes *lower* than WSMM's (its fitness penalises C6 shortfall while WSMM maximises total size). The acceptance criterion "WSMM <= SM, GA in violation probability" is therefore not met.
- No R_sense in 150-400 m (either q model) fits both metrics: our curves keep growing with |V| while the paper's saturate at ~5.3 Mb with very low violation probability. That suggests a different candidate-set / weight definition than the one specified; I did not invent one.
- The `sub1`-level "WSMM weight >= SM, GA" property is optimal only among max-cardinality assignments; it is not guaranteed instance-wise (tested on average).

## Exp2 / Exp3 (MDA only, uncalibrated, 3 eval episodes)
MDA AMWAoI for |V|=4..12: 6.9, 10.8, 8.9, 6.6, 10.1 (paper 9.0, 8.2, 7.2, 7.1, 6.85): same order of magnitude, but noisy and without the paper's downward trend
(AMWAoI is a max over vehicles, so heavy-tailed; use many more evaluation episodes). Exp3 MDA (1 episode/point): comm 0.30-0.42 s, comp 1.2-1.9 s, migration 1.4-1.6 s, AMWAoI 30-38 (paper MDA: 2.0->0.35, 1.6->0.3, 1.85->1.05, 39.5->17.5).
**Proposed vs DDPG vs MDA ordering, the -10/-19/-36 reward levels and the 55-70% migration reduction have not been reproduced** — they require running the torch agents.
The migration-delay scale (T_unit=2.4e-4, D, d) makes reactive migration ~1.7 s per event, so predictive migration has headroom; whether PPO learns it is untested.

## Layout
`env/` (mobility, channel, per-slot env) · `sub1_wsmm.py` · `sub2_bca.py` · `sub3_asdmm/` (ppo, ddpg, mda, train) · `experiments/` · `tests/`
