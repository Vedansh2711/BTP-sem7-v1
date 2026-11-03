# BTP-7 — Cooperative Perception Aided Digital Twin with False CPM Resistance

Reproduction of **"Cooperative Perception Aided DT Model Update and Migration in Mixed Vehicular Networks"** (Lu et al., IEEE T-ITS 2025), extended with a **3-Layer False CPM Resistance** defense system against malicious ICV attacks.

---

## Architecture Overview

```
CPM Reception
     │
     ▼
┌─────────────────────────────────────────────────┐
│  3-Layer False CPM Resistance (false_cpm_resistance.py)
│                                                 │
│  Layer A: Kinematic Plausibility Filter         │  ← rejects physically impossible reports
│  Layer B: Cross-Validation & Fusion             │  ← multi-ICV consensus + trust-weighted median
│  Layer C: Trust Management & Quarantine         │  ← EWMA trust scores + quarantine
└─────────────────────────────────────────────────┘
     │   vetted reports + trust vector
     ▼
Subproblem 1: ICV Selection  (fast_optimal_solvers.py — SPG)
     ▼
Subproblem 2: BCA            (fast_optimal_solvers.py — ED-BCA)
     ▼
Subproblem 3: ASDMM          (sub3_asdmm/ — PPO access selection + DT migration)
```

All parameters live in `config.yaml`. Everything beyond the paper is listed in `ASSUMPTIONS.md`.

---

## 1. Setup (Windows / PowerShell)

```powershell
cd "C:\Users\<you>\OneDrive\Desktop\BTP-7"
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install numpy scipy pyyaml matplotlib networkx torch traci
python -m unittest discover -s tests -t .    # expect: Ran 21 tests ... OK
```

macOS/Linux: `python -m venv .venv && source .venv/bin/activate`, same `pip install`.

### Optional: SUMO backend

Set these in **every new PowerShell window** before using the SUMO mobility backend:

```powershell
$env:SUMO_HOME = "C:\Program Files (x86)\Eclipse\Sumo"
$env:PATH += ";$env:SUMO_HOME\bin"
sumo --version
$env:DTMIG_BACKEND = "sumo"     # default (unset) = built-in kinematic mobility
```

---

## 2. Experiments

### Exp 1–3: Base Paper Reproduction (ICV Selection, ASDMM, Dense Traffic)

```powershell
# Exp1 (Figs 4-7): ICV selection — WSMM vs SM vs GA, ~5 min
python -m experiments.exp1_icv_selection --snap 600 --ga-snap 150

# Exp2 main run: PPO training curve + AMWAoI sweep, ~1.5-3 h per learned method
python -m experiments.exp2_asdmm --sweep curve --methods proposed,ddpg,mda --epochs 300 --eval-eps 20 --bca-iters 40 --out results/exp2
python -m experiments.plots results/exp2_curve_proposed-ddpg-mda.json curve

# Exp2 V/U sweeps (Figs 8-11)
python -m experiments.exp2_asdmm --sweep V --methods proposed --epochs 300 --eval-eps 20 --bca-iters 40 --out results/exp2
python -m experiments.exp2_asdmm --sweep U --methods proposed --epochs 300 --eval-eps 20 --bca-iters 40 --out results/exp2
python merge_results.py

# Exp3 (Fig 13): dense traffic, ICV penetration 0.60-0.80
python -m experiments.exp3_dense --methods proposed,ddpg,mda --out results/exp3

# Sanity floor baseline
python random_baseline.py
```

### Exp 4: False CPM Resistance — p_mal Sweep *(BTP extension)*

```powershell
python exp_false_cpm.py
# Output saved to: results/false_cpm_sweep.json
```

Sweeps `p_mal ∈ {0%, 12.5%, 25%, 37.5%, 50%}` across two configurations:

| Config | Description |
|--------|-------------|
| `BASE (no defence)` | Raw pipeline, all defence layers disabled |
| `FULL (A + B + C)` | All three defence layers + fast solvers active |

With `|V| = 8` ICVs, each p_mal point maps to exactly **0, 1, 2, 3, or 4** malicious ICVs (no quantization duplicates).

---

## 3. Module Reference

### `false_cpm_resistance.py` — 3-Layer Defence Pipeline

| Layer | What it does | Key parameters |
|-------|-------------|----------------|
| **A — Kinematic Plausibility** | Rejects reports where the claimed position/speed change is physically impossible given the previous fused position. Also rejects ghost IDs (`target_id < 0`) and inflated quality (`quality > 2.05`). | `v_max=50 m/s`, `a_max=10 m/s²` |
| **B — Cross-Validation & Fusion** | Requires ≥ 2 reporters; rejects outliers whose reported position deviates > `outlier_thresh` from the trust-weighted median; returns a fused position and confidence. | `consensus_thresh=5 m`, `outlier_thresh=8 m` |
| **C — Trust & Quarantine** | Maintains per-ICV EWMA trust score. Passed Layer B → trust reward; failed → faster trust penalty. ICVs below `quarantine_thresh` are quarantined and their weights zeroed. | `eta_reward=0.2`, `eta_penalty=0.5`, `quarantine_thresh=0.35` |

**Important design constraint**: `target_history` (last fused position used by Layer A) is **only updated when at least one report is accepted**. This prevents stale zeros from poisoning kinematic checks when no ICVs are nearby.

**Key classes:**
- [`FalseCPMConfig`](file:///C:/Users/vedan/OneDrive/Desktop/BTP-7/false_cpm_resistance.py#L20) — all tunable parameters
- [`CPMReport`](file:///C:/Users/vedan/OneDrive/Desktop/BTP-7/false_cpm_resistance.py#L49) — one ICV's report about one N-ICV target
- [`ICVTrustState`](file:///C:/Users/vedan/OneDrive/Desktop/BTP-7/false_cpm_resistance.py#L61) — per-ICV trust tracking
- [`FalseCPMResistance`](file:///C:/Users/vedan/OneDrive/Desktop/BTP-7/false_cpm_resistance.py#L86) — the pipeline class; call `process_slot()` each slot

---

### `attack_injector.py` — Threat Model Simulator

Marks a fraction `p_mal` of ICVs as malicious and modifies their CPMs. Four attack types:

| Attack | Behavior |
|--------|----------|
| **Ghost injection** (25%) | Reports a non-existent vehicle at a random road position (negative target ID) |
| **State manipulation** (35%) | Shifts a real vehicle's reported position by ~15 m and speed by ~8 m/s |
| **Quality inflation** (20%) | Overstates perception quality 3× to win WSMM/SPG selection |
| **Suppression** (20%) | Withholds the CPM 70% of slots; occasionally sends truthful data 30% of the time |

`n_mal = int(round(p_mal × n))` — uses `round()` (not `int()`) so distinct p_mal values always map to distinct attacker counts with 8 ICVs.

**FAR definition**: Only genuinely falsified content (ghost, shift, quality) counts as a False Acceptance. A suppression attacker's occasional honest pass-through is **not** counted as a False Acceptance.

---

### `fast_optimal_solvers.py` — Fast Optimal Solvers

Replaces the slower reference solvers for use in the defence experiment:

| Solver | Replaces | Speedup | Optimality |
|--------|----------|---------|------------|
| **SPG** — Spatial-Indexed Priority Greedy (Subproblem 1) | MCMF/Bellman-Ford (`sub1_wsmm.select`) | O(N log N) vs O(N³) | ≥ 63.2% of optimal (submodular); empirically >98% |
| **ED-BCA** — Exact 1D Dual Bisection (Subproblem 2) | 150-iter dual subgradient (`solve_rsu`) | ~240× (25 vs 6000 evals) | Exact (geometric convergence) |

C6 constraint satisfaction uses `get_vetted_perceived_size()` with the **raw physical weights** (not trust-discounted), so quarantined nodes correctly reduce perceived coverage without artificially inflating the constraint threshold.

---

### `exp_false_cpm.py` — Sweep Experiment Driver

Runs `BASE` vs `FULL (A+B+C)` across 5 p_mal points (5 seeds each, 150 slots/episode).

Metrics reported per configuration per p_mal:

| Metric | Description |
|--------|-------------|
| **AMWAoI** | Average Maximum Weighted Age of Information (s) — lower is better |
| **RMSE** | DT position error vs ground truth (m) — only counted for targets with `fusion_confidence > 0` |
| **FRR** | False Rejection Rate — honest reports incorrectly rejected |
| **FAR** | False Acceptance Rate — genuinely falsified reports incorrectly accepted |
| **H-Trust** | Mean EWMA trust of honest ICVs at episode end |
| **M-Trust** | Mean EWMA trust of malicious ICVs at episode end |
| **Q** | Number of quarantined ICVs at episode end |

---

## 4. Repository Layout

```
config.yaml          — all simulation parameters
ASSUMPTIONS.md       — design choices beyond the paper
README.md

false_cpm_resistance.py  — 3-layer defence pipeline (Layer A, B, C)
attack_injector.py       — threat model: ghost, shift, quality, suppression attacks
fast_optimal_solvers.py  — SPG (Sub1) + ED-BCA (Sub2) fast solvers
exp_false_cpm.py         — p_mal sweep experiment (BASE vs FULL)

env/
  mobility.py            — kinematic mobility model
  sumo_mobility.py       — SUMO/TraCI backend
  channel.py             — path-loss + fading channel
  dt_env.py              — per-slot DT pipeline

sub1_wsmm.py             — MCMF-based ICV selection (WSMM, SM, GA)
sub2_bca.py              — dual subgradient BCA (reference solver)

sub3_asdmm/
  ppo.py  ddpg.py  mda.py  train.py

experiments/
  exp1_icv_selection.py
  exp2_asdmm.py
  exp3_dense.py
  plots.py  compare.py  common.py

tests/
  test_sub1.py  test_sub2.py  test_env.py  test_sumo_mobility.py

results/
  false_cpm_sweep.json     — output of exp_false_cpm.py
```

---

## 5. Validation Status

| Item | Status |
|------|--------|
| Unit tests (Env, WSMM/SM/GA, BCA, delay/AoI/migration) | **21 tests pass** |
| WSMM vs brute force, SciPy assignment, networkx max-flow | match exactly |
| BCA vs SLSQP | worst-case gap 0.45% over 25 random instances |
| Defence pipeline (Layer A, B, C) | verified correct at p_mal=0% (AMWAoI ≈ 5.2s, FAR = 0%) |
| Target history poisoning bug | fixed — `th.last_pos` only updated when `accepted_ids` is non-empty |
| p_mal quantization bug | fixed — `int(round(...))` ensures distinct attacker counts |
| FAR inflation from suppression truthful pass-throughs | fixed — only `attacked is not honest` counted |
| SUMO/TraCI backend | logic tested against mock TraCI; not yet run against real SUMO |

---

## 6. Troubleshooting

- **`Start directory is not importable: 'tests'`**: `cd` into the folder containing `config.yaml` and `tests/`.
- **`.venv\Scripts\Activate.ps1` not found**: the venv may be one level up — try `..\\.venv\Scripts\Activate.ps1`.
- **`'sumo' not found` / `traci not found`**: set `SUMO_HOME` and `PATH` as in section 1 (per window).
- **No training output for a few minutes**: normal at ~9 s/epoch with the default 10-epoch print interval; use `--log-every 1`.
- **Windows encoding errors with emoji in output**: the experiment scripts use ASCII `[OK]`/`[WARN]` and `->` to stay cp1252-safe.
