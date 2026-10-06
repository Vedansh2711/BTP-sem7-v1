# ASSUMPTIONS.md

## A. Resolutions of the paper inconsistencies (Section 10 of the master prompt) — all implemented as specified
1. Eq (14): time average over slots (`AMWAoI = mean_t Omega_t`), alpha = 0.5.
2. WSMM = MIN cost / MAX flow; cost and flow stored in separate arrays (`sub1_wsmm.MCMF`).
3. Computation allocation uses the exact KKT solution (`solve_computation`), not Eq (23b)/(24).
4. delta is projected onto {delta>=0, sum = w_u} after the Eq (26b) step (`project_simplex`).
5. Alg. 2 loops until the objective change <= eps (relative, with `patience` consecutive iterations and >=20 iterations).
6. Per slot: WSMM first (action-independent, enters the state), then BCA after the action.
7. tau is applied in Eq (8), not Eq (7) (`c_cp` is Gcycles/s).
8. Empty selection or no common RSU for the selected ICVs => N-ICV failure: T_tran = T_pena, T_exe = T_epena, AoI + 1.
9. C6 violated => N-ICV update invalid that slot => AoI + 1.
10. Weights w = A + 1.
11. Violation probability follows Fig 5 (decreasing with |V|).
12. Eq (1) literal; option `perception.transmit_only_if_selected`.
13. "MAD" read as MDA; Fig 13(c) used.
14. Eq (30) (C9 penalty) implemented per RSU and summed over k.
15. PPO default is on-policy; `ppo.paper_buffer: true` gives the random-replacement buffer (capacity 2000, my choice).
16. Only R_sense / alpha / T_pena / T_unit / beta may be tuned. **No parameter has been changed from the defaults**; the
    R_sense sweep (150..400 m) did not improve the Exp1 fit beyond the default 300 m (see README).

## B. Additional assumptions made while implementing (not in the paper or the master prompt)
- **A1 Mobility.** SUMO/TraCI is not available in the authoring sandbox. `env/mobility.py` provides a kinematic fallback
  (Gaussian speed capped at 16.7 m/s, accel/decel-limited jitter, no car-following, no lane interaction). `SumoMobility` is a stub that raises.
  Exp1/2 use a fixed population with same-type replacement on exit; Exp3 uses Poisson entries at 1000 veh/h with a 160 s warm-up.
- **A2 Omega uses A^{t+1}** (AoI after the slot's update) so the reward for action a^t reflects its outcome.
- **A3 Stable matching degenerates under the literal spec.** An ICV ranks N-ICVs by q*s_unit, which for a fixed ICV is the same ordering as q;
  the N-ICV ranks by q*s_unit_v. The two orders almost coincide, so SM ~ WSMM (gap < 1% in my runs, see README).
  `stable_matching(..., icv_score=...)` and `exp1 --sm-pref distance` provide a non-degenerate variant (ICVs rank by proximity); it is **off by default**.
- **A4 Migration semantics.** Pending predictive transfer = (target RSU, start slot). A repeated identical y while pending is not counted again for C9.
  A y whose target never becomes the serving RSU is wasted (no cost). Reactive migrations (no matching pending y) also count towards C9.
  C9 only acts through the reward penalty. T_wired = T_unit * D * d with d = |k - current| * 500 m. N-ICV DTs migrate by the same rules, only when Eq (4) gives a serving RSU.
- **A5 Access.** An ICV x outside coverage is replaced by the nearest in-range RSU (masks should prevent this). Coverage is longitudinal distance <= 300 m. The N-ICV serving RSU needs no coverage of its own.
- **A6 Initial DT location** = nearest RSU at creation; new/replaced vehicles start with A = 0.
- **A7 BCA numerics.** B in MHz internally. Step sizes `step/sqrt(j)` with `step_lambda=2e-4`, `step_delta=20`, `lam0=1e-2`; the returned B is the best
  primal-feasible iterate (rescaled if sum B > Bmax). The raw primal trace is not monotone; only the best-so-far is (test checks that).
- **A8 PPO details.** Entropy coefficient 0.01, advantage normalisation, TD(0) advantage recomputed at the start of each of the 10 passes, log-prob = sum over relevant heads.
- **A9 DDPG.** Critic takes the continuous (noisy, [0,1]) action vector; decoded by masked argmax; ~1 update per env step, evaluated reactive (`ddpg.predictive=false`).
- **A10 Exp1 snapshots** use i.i.d. uniform positions on the 2 km road (no dynamics); metrics are per N-ICV. GA (pop 50, 100 gens) is compared with the other methods on identical snapshots (`paired_*`).
- **A11 Exp3** trains each method once at penetration 0.70 and evaluates over the sweep (`--train-per-point` retrains). "Actual migration delay" = mean T_amigr per migration event.
- **A12 Tests** use unittest (pytest-compatible) and SciPy SLSQP as the BCA reference because cvxpy is unavailable here.
