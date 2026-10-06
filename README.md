# dtmig — reproduction of "Cooperative Perception Aided DT Model Update and Migration in Mixed Vehicular Networks" (Lu et al., IEEE T-ITS 2025)

Pipeline: **Subproblem 1** cooperative ICV selection (WSMM = min-cost max-flow; SM and GA baselines) ->
**Subproblem 2** bandwidth/computation allocation (dual ascent + closed-form KKT) ->
**Subproblem 3** ASDMM (PPO access selection + predictive DT migration; DDPG and MDA baselines).
All parameters live in `config.yaml`; every choice that goes beyond the paper is listed in `ASSUMPTIONS.md`.

## 1. Setup (Windows / PowerShell)
```powershell
cd C:\Users\<you>\OneDrive\Desktop\dtmig\dtmig        # the folder that contains config.yaml
python -m venv .venv                                  # skip if you already have one (it may be one folder up: ..\.venv)
.venv\Scripts\Activate.ps1                            # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install numpy scipy pyyaml matplotlib networkx torch traci
python -m unittest discover -s tests -t .             # expect: Ran 21 tests ... OK
```
macOS/Linux: `python -m venv .venv && source .venv/bin/activate`, same `pip install`.

### Optional: SUMO backend
Needs SUMO installed (`sumo`, `netconvert`). Set these in **every new PowerShell window** before running:
```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"      # folder containing bin\ and tools\
$env:PATH += ";$env:SUMO_HOME\bin"
sumo --version
$env:DTMIG_BACKEND = "sumo"                                 # unset (or "kinematic") = built-in fallback mobility
```
The road network and routes are generated automatically (2000 m, two-way, 2 lanes/direction, Wiedemann car following,
speed ~ N(12.5, 3.3^2) capped at 16.7 m/s, 1 s steps). Fixed population (Exp1/2): same-type replacement on exit.
Dynamic (Exp3): Poisson flows at 1000 veh/h, ICV with probability = penetration. `mobility.sumo.gui: true` in the config opens sumo-gui (debugging only).

## 2. Commands
```powershell
# first, confirm the torch agents run at all (about 1 minute)
python -m experiments.exp2_asdmm --sweep curve --methods proposed,ddpg,mda --epochs 3 --eval-eps 1 --bca-iters 40 --out results/smoke

# Exp1 (Figs 4-7), no torch, ~5 min
python -m experiments.exp1_icv_selection --snap 600 --ga-snap 150

# Exp2 main run: training curve + final AMWAoI at |V|=8, |U|=3 (Fig 12), ~1.5-3 h per learned method
python -m experiments.exp2_asdmm --sweep curve --methods proposed,ddpg,mda --epochs 300 --eval-eps 20 --bca-iters 40 --log-every 1 --out results/exp2
python -m experiments.plots results/exp2_curve_proposed-ddpg-mda.json curve

# Exp2 sweeps (Figs 8-11): one agent trained per point -> many hours. Run one method per window.
python -m experiments.exp2_asdmm --sweep V --methods proposed --epochs 300 --eval-eps 20 --bca-iters 40 --out results/exp2
python -m experiments.exp2_asdmm --sweep U --methods proposed --epochs 300 --eval-eps 20 --bca-iters 40 --out results/exp2
#   methods: proposed ddpg mda ppo_sm ddpg_sm mda_sm
python merge_results.py
python -m experiments.plots results/exp2_V_all.json V ; python -m experiments.plots results/exp2_U_all.json U
python -m experiments.compare results/exp2_V_all.json V          # ours vs paper, % error

# Exp3 (Fig 13): dense traffic, penetration 0.60-0.80
python -m experiments.exp3_dense --methods proposed,ddpg,mda --out results/exp3
python -m experiments.plots results/exp3.json exp3

# sanity floor: a learned policy should beat this
python random_baseline.py
```
Use a different `--out` name per backend (e.g. `results/exp2_sumo`) so results are not overwritten.
Cost: ~9 s per 150-slot episode at default BCA settings (`--bca-iters 40` roughly halves it).
`--log-every N` prints progress every N epochs (default 10). Training is slow to show its first line; this is normal.

## 3. What the experiments are
| | Question | Compares | Metric |
|---|---|---|---|
| **Exp1** (Figs 4-7) | Which ICVs should serve each N-ICV? | WSMM vs SM vs GA | avg weighted perception size (Mb), P[size < 4 Mb] |
| **Exp2** (Figs 8-12) | Does PPO + predictive DT migration keep twins fresher? | Proposed (PPO+WSMM) vs DDPG vs MDA (+ `*_sm` variants) | AMWAoI (lower is better), training reward |
| **Exp3** (Fig 13) | Do the gains hold in dense, dynamic traffic? | Proposed vs DDPG vs MDA, penetration 0.60-0.80 | comm delay, comp delay, actual migration delay, AMWAoI |

## 4. Validation status (what has actually been checked)
| Item | Status | Where |
|---|---|---|
| Env, WSMM/SM/GA, BCA, delay/AoI/migration logic | **21 unit tests pass** (17 core + 4 SUMO-logic) | sandbox |
| WSMM vs brute force, SciPy assignment, networkx max-flow | match exactly | tests |
| BCA vs SLSQP | worst-case gap 0.45% over 25 random instances (target < 2%); closed-form f matches SLSQP | tests |
| Hand-computed delays, Eq (4) failure, AoI recursion, T_pre, C9 penalty, masks | pass | tests |
| Exp1 | run (400 snapshots, GA on 150); results below | sandbox |
| MDA / MDA-SM on Exp2/Exp3 (kinematic mobility) | run, uncalibrated, few episodes | sandbox |
| PPO/DDPG code | torch smoke test ran without error on the user's machine; PPO trained 300 epochs (kinematic env) | user machine |
| SUMO/TraCI backend | **logic tested against a mock TraCI only**; not yet run against real SUMO | pending: run the one-episode check below |

SUMO one-episode check (before any long SUMO training):
```powershell
$env:DTMIG_BACKEND = "sumo"
python -m experiments.exp2_asdmm --sweep curve --methods mda --epochs 0 --eval-eps 1 --bca-iters 40 --out results/sumo_check
```

## 5. Results so far
**Exp1** (R_sense = 300 m, q ~ U[0.2,2]; identical snapshots for all methods), ours vs paper:
| | V=4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| WSMM avg Mb | 2.90 | 4.65 | 5.93 | 7.10 | 8.21 |
| paper WSMM | 3.7 | 4.65 | 5.1 | 5.3 | 5.35 |
| SM avg Mb | 2.90 | 4.65 | 5.93 | 7.08 | 8.16 |
| GA avg Mb | 2.86 | 4.50 | 5.82 | 6.99 | 8.11 |
| WSMM P[<4Mb] | .67 | .48 | .35 | .28 | .19 |
| paper WSMM P[<4Mb] | .21 | .11 | .02 | .01 | .01 |

- WSMM >= SM >= GA in average size holds, but the gaps are 0-3% (paper: 15-25%). With the literal rule (both sides rank by q*s_unit) SM is almost WSMM (ASSUMPTIONS A3); ICVs ranking by proximity (`--sm-pref distance`) puts SM ~5-8% below WSMM.
- **Violation probability is not reproduced**, and GA's is sometimes lower than WSMM's (its fitness penalises C6 shortfall). The acceptance criterion "WSMM <= SM, GA in violation probability" is not met.
- No R_sense in 150-400 m (either q model) fits both metrics; our curves keep growing with |V| while the paper's saturate near 5.3 Mb. This points to a different candidate-set or weight definition than the one specified. R_sense was left at 300 m.

**Exp2/Exp3 baselines (MDA, kinematic mobility, 1-3 eval episodes, uncalibrated):** MDA AMWAoI for |V|=4..12: 6.9, 10.8, 8.9, 6.6, 10.1 (paper 9.0, 8.2, 7.2, 7.1, 6.85): same order of magnitude, noisy, no clear downward trend (AMWAoI is a max over vehicles, so it is heavy-tailed; use many evaluation episodes). Exp3 MDA: comm 0.30-0.42 s, comp 1.2-1.9 s, migration 1.4-1.6 s, AMWAoI 30-38.

**PPO training (kinematic env, |V|=8, |U|=3, 300 epochs):** the per-epoch reward fluctuated between about -4.6 and -7.7 (first 10 epochs averaged -6.4) with no clear upward trend in the epochs reported (110-300). **This is inconclusive**: judge learning from the final 20-episode greedy evaluation in the result JSON and from `random_baseline.py`. If PPO is not clearly below the random policy and MDA, try `ppo.entropy_coef: 0.001`, `ppo.lr: 3.0e-4`, and 600 epochs in `config.yaml`.

**Not yet reproduced:** the Proposed < DDPG < MDA ordering, the paper's reward levels (-10/-19/-36) and the 55-70% migration-delay reduction. Absolute levels need not match (the environment is not calibrated to the paper); what matters is the ordering inside your own runs.

## 6. Layout
```
config.yaml  ASSUMPTIONS.md  README.md  random_baseline.py  merge_results.py
env/          mobility.py (kinematic) sumo_mobility.py (TraCI) channel.py dt_env.py (per-slot pipeline)
sub1_wsmm.py  sub2_bca.py
sub3_asdmm/   ppo.py ddpg.py mda.py train.py
experiments/  exp1_icv_selection.py exp2_asdmm.py exp3_dense.py plots.py compare.py common.py
tests/        test_sub1.py test_sub2.py test_env.py test_sumo_mobility.py
```

## 7. Troubleshooting
- `Start directory is not importable: 'tests'`: you are not in the folder containing `tests` and `config.yaml`; `cd` into it (check with `dir`).
- `.venv\Scripts\Activate.ps1` not found: the venv may be one level up; use `..\.venv\Scripts\Activate.ps1`.
- `'sumo' not found` / `traci not found`: set `SUMO_HOME` and `PATH` as in section 1 (per window).
- No training output for a few minutes: normal at ~9 s/epoch with the default 10-epoch print interval; use `--log-every 1`.
