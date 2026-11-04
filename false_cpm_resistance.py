"""
False CPM Resistance: Three-Layer Defence Pipeline
===================================================
Layer A: Second-Order Kinematic & Physical Plausibility Filter
         — Rejects ghost IDs, inflated quality, out-of-envelope speeds,
           target dead-reckoning violations, and per-reporter self-kinematic violations.
Layer B: Cross-Validation & Temporal-Anchored Medoid Fusion
         — Joint (position, speed) consensus + temporal trajectory verification
           + epistemic abstention on unverified first-slot disagreements.
Layer C: Trust Management & Quarantine
         — Asymmetric EWMA trust score, rapid quarantine of repeat offenders,
           and vetted physical C6 perception accounting.

This module is MODULAR: each layer can be toggled independently for ablation experiments.
It sits between CPM reception and the WSMM/BCA/ASDMM algorithms.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class FalseCPMConfig:
    """All tuneable parameters for the three-layer pipeline."""
    # Layer A: Kinematic Plausibility
    enable_layer_a: bool = True
    a_max: float = 5.0             # m/s² — max physically plausible highway accel/decel (4.5 + margin)
    v_max: float = 17.2            # m/s  — max physically plausible highway speed (16.7 + margin)
    tau: float = 1.0               # s    — slot duration (matches config.yaml time.tau)

    # Layer B: Cross-Validation & Fusion
    enable_layer_b: bool = True
    consensus_thresh: float = 5.0  # m   — max acceptable pairwise position disagreement
    speed_consensus_thresh: float = 3.0  # m/s — max acceptable pairwise speed disagreement
    outlier_thresh: float = 8.0    # m   — individual report vs medoid reference
    min_reporters: int = 2         # minimum ICVs required for multi-reporter consensus

    # Layer C: Trust & Quarantine
    enable_layer_c: bool = True
    trust_init: float = 1.0        # initial trust for new ICVs
    eta_reward: float = 0.2        # EWMA learning rate on verification pass
    eta_penalty: float = 0.5       # EWMA learning rate on verification fail (faster isolation)
    decay_rate: float = 0.01       # trust decay per slot if dormant > 30 slots
    quarantine_thresh: float = 0.35 # below this trust -> quarantined
    recovery_thresh: float = 0.5   # must reach this trust to exit quarantine


# ──────────────────────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CPMReport:
    """One ICV's CPM report about one target N-ICV."""
    reporter_id: int               # ICV index (global)
    target_id: int                 # N-ICV index (global)
    reported_pos: float            # reported 1D position of the target
    reported_speed: float          # reported speed of the target
    quality: float                 # reported perception quality q
    s_unit: float                  # reported data size s_unit
    timestamp: int                 # slot number


@dataclass
class ICVTrustState:
    """Per-ICV trust tracking maintained by the RSU."""
    trust: float = 1.0
    quarantined: bool = False
    n_pass: int = 0                # total verification passes
    n_fail: int = 0                # total verification failures
    last_seen: int = 0             # last slot this ICV sent a CPM
    fail_streak: int = 0           # consecutive failures


@dataclass
class TargetHistory:
    """Per-target (N-ICV) tracking for kinematic plausibility."""
    last_pos: Optional[float] = None
    last_speed: Optional[float] = None
    last_slot: int = -1
    fused_pos: Optional[float] = None
    fusion_confidence: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# The Three-Layer Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class FalseCPMResistance:
    """
    Sits between CPM reception and the base algorithms (WSMM/SPG, BCA/ED-BCA, ASDMM).
    """

    def __init__(self, fcpm_cfg: FalseCPMConfig = None):
        self.cfg = fcpm_cfg or FalseCPMConfig()
        self.trust_states: Dict[int, ICVTrustState] = {}     # ICV id -> trust state
        self.target_history: Dict[int, TargetHistory] = {}   # N-ICV id -> history
        # Per-(reporter_id, target_id) last report for self-kinematic consistency
        self.reporter_history: Dict[Tuple[int, int], Tuple[float, float, int]] = {}
        # Set of (reporter_id, target_id) pairs verified across >= 2 consecutive slots
        self.temporally_verified: set = set()
        self.slot_stats: List[dict] = []                      # per-slot statistics
        self.last_accepted_reports: Dict[int, set] = {}       # target_id -> set of accepted reporter_ids

    def reset(self):
        """Call at episode reset."""
        self.trust_states.clear()
        self.target_history.clear()
        self.reporter_history.clear()
        self.temporally_verified.clear()
        self.slot_stats.clear()
        self.last_accepted_reports.clear()

    def on_vehicles_respawned(self, new_indices, gone_indices):
        """
        Clear kinematic history for target slots that exited the road or respawned,
        preventing a newly spawned vehicle from being compared against an exited vehicle.
        """
        reset_set = set(int(i) for i in new_indices) | set(int(i) for i in gone_indices)
        if not reset_set:
            return
        for tid in reset_set:
            self.target_history.pop(tid, None)
        stale_keys = [k for k in self.reporter_history if k[1] in reset_set or k[0] in reset_set]
        for k in stale_keys:
            self.reporter_history.pop(k, None)
            self.temporally_verified.discard(k)

    def _ensure_trust(self, icv_id: int) -> ICVTrustState:
        if icv_id not in self.trust_states:
            self.trust_states[icv_id] = ICVTrustState(trust=self.cfg.trust_init)
        return self.trust_states[icv_id]

    def _ensure_target(self, target_id: int) -> TargetHistory:
        if target_id not in self.target_history:
            self.target_history[target_id] = TargetHistory()
        return self.target_history[target_id]

    # ───────────────────────────────────────────────── Layer A
    def layer_a_check(self, report: CPMReport) -> bool:
        """
        Second-Order Kinematic & Physical Plausibility Filter.
        Rejects reports where:
          0. Target ID is invalid (ghost injection: target_id < 0)
          1. Quality exceeds physical sensor bounds (quality inflation: quality > 2.05)
          2. Reported speed violates physical highway speed envelope
          3. Per-reporter consecutive reports for the same target violate self-kinematics
          4. Target dead-reckoning (displacement vs speed) or acceleration is violated
        """
        if not self.cfg.enable_layer_a:
            return True

        # Check 0: Ghost target rejection
        if report.target_id < 0:
            return False

        # Check 1: Quality inflation bound check
        if report.quality > 2.05 or report.quality < 0.0:
            return False

        # Check 2: Physical highway speed envelope check
        eff_vmax = min(self.cfg.v_max, 17.2)
        if report.reported_speed < 0.0 or report.reported_speed > eff_vmax:
            return False

        eff_amax = min(self.cfg.a_max, 5.0)

        # Check 3: Per-(reporter, target) self-kinematic consistency
        r_key = (report.reporter_id, report.target_id)
        prev_rep = self.reporter_history.get(r_key)
        if prev_rep is not None:
            p_pos, p_spd, p_slot = prev_rep
            dt_r = (report.timestamp - p_slot) * self.cfg.tau
            if 0.0 < dt_r <= 2.5 * self.cfg.tau:
                dx_r = abs(report.reported_pos - p_pos)
                if dx_r <= 800.0:  # not a road-end wrap-around
                    dv_r = abs(report.reported_speed - p_spd)
                    if dv_r > eff_amax * dt_r + 0.5:
                        self.temporally_verified.discard(r_key)
                        return False
                    v_avg_r = 0.5 * (p_spd + report.reported_speed)
                    kin_err_r = min(
                        abs(dx_r - report.reported_speed * dt_r),
                        abs(dx_r - v_avg_r * dt_r)
                    )
                    if kin_err_r > 2.5 * dt_r:
                        self.temporally_verified.discard(r_key)
                        return False
                    # Passed 2-slot differential kinematic verification!
                    self.temporally_verified.add(r_key)

        # Check 4: Target history dead-reckoning consistency
        th = self._ensure_target(report.target_id)
        if th.last_pos is not None and 0 < (report.timestamp - th.last_slot) <= 2:
            dt = (report.timestamp - th.last_slot) * self.cfg.tau
            dx = abs(report.reported_pos - th.last_pos)
            if dx <= 800.0:  # not a road-end wrap-around
                if th.last_speed is not None:
                    dv = abs(report.reported_speed - th.last_speed)
                    if dv > eff_amax * dt + 0.5:
                        return False
                    v_avg = 0.5 * (th.last_speed + report.reported_speed)
                    kin_err = min(
                        abs(dx - report.reported_speed * dt),
                        abs(dx - v_avg * dt)
                    )
                    if kin_err > 2.5 * dt + 1.0:
                        return False
                elif dx > eff_vmax * dt * 1.15:
                    return False

        return True

    # ───────────────────────────────────────────────── Layer B
    def layer_b_fuse(
        self,
        target_id: int,
        accepted_reports: List[CPMReport],
        slot_t: Optional[int] = None
    ) -> Tuple[float, float, List[int], bool]:
        """
        Joint (Position, Speed) Temporal-Anchored Medoid Consensus & Fusion.

        Returns:
            (fused_position, fusion_confidence, list_of_accepted_reporter_ids, ambiguous_tie)
        """
        if not self.cfg.enable_layer_b or len(accepted_reports) == 0:
            if len(accepted_reports) == 1:
                r = accepted_reports[0]
                return r.reported_pos, 0.5, [r.reporter_id], False
            elif len(accepted_reports) == 0:
                return 0.0, 0.0, [], False
            positions = [r.reported_pos for r in accepted_reports]
            return float(np.mean(positions)), 1.0, [r.reporter_id for r in accepted_reports], False

        positions = np.array([r.reported_pos for r in accepted_reports], dtype=float)
        speeds = np.array([r.reported_speed for r in accepted_reports], dtype=float)

        th = self._ensure_target(target_id)
        has_recent_history = (
            th.last_pos is not None
            and slot_t is not None
            and 0 < (slot_t - th.last_slot) <= 2
            and th.last_speed is not None
        )

        # Single reporter handling
        if len(accepted_reports) < self.cfg.min_reporters:
            r = accepted_reports[0]
            ts = self._ensure_trust(r.reporter_id)
            is_verified = ((r.reporter_id, target_id) in self.temporally_verified) or has_recent_history
            if not is_verified:
                # First unverified single-reporter snapshot: record for next slot's Check 3,
                # but abstain from locking target_history or scoring RMSE until verified on next slot
                return r.reported_pos, 0.0, [r.reporter_id], True
            conf = 0.85 * ts.trust if not ts.quarantined else 0.1
            return r.reported_pos, conf, [r.reporter_id], False

        # Multi-reporter Joint (Position, Speed) Medoid Consensus
        trusts = np.array([self._ensure_trust(r.reporter_id).trust for r in accepted_reports], dtype=float)
        trusts = np.maximum(trusts, 0.05)

        verified_flags = np.array([
            1.0 if (r.reporter_id, target_id) in self.temporally_verified else 0.0
            for r in accepted_reports
        ], dtype=float)

        # Weight verified trajectories higher in consensus voting
        effective_weights = trusts * (1.0 + 1.0 * verified_flags)

        pos_within = np.abs(positions[:, None] - positions[None, :]) <= self.cfg.consensus_thresh
        spd_within = np.abs(speeds[:, None] - speeds[None, :]) <= self.cfg.speed_consensus_thresh
        within = pos_within & spd_within

        inlier_scores = (within * effective_weights[None, :]).sum(axis=1)

        if has_recent_history:
            dt = (slot_t - th.last_slot) * self.cfg.tau
            dx = np.abs(positions - th.last_pos)
            v_avg = 0.5 * (th.last_speed + speeds)
            kin_err = np.minimum(np.abs(dx - speeds * dt), np.abs(dx - v_avg * dt)) + np.abs(speeds - th.last_speed) * dt
            inlier_scores = inlier_scores + 0.5 * np.exp(-kin_err / 2.5)

        # If reporters disagree on the very first slot before any reporter is temporally verified
        # and before target history exists, abstain for 1 slot so Check 3 eliminates shift attackers
        pos_spread = float(positions.max() - positions.min())
        if pos_spread > self.cfg.consensus_thresh and not has_recent_history and verified_flags.sum() == 0:
            avg_pos = float(np.average(positions, weights=trusts))
            return avg_pos, 0.0, [r.reporter_id for r in accepted_reports], True

        best_medoid = int(np.argmax(inlier_scores))
        ref_pos = positions[best_medoid]
        ref_spd = speeds[best_medoid]

        # Separate consensus inliers from outliers relative to medoid
        inlier_mask = (
            (np.abs(positions - ref_pos) <= self.cfg.outlier_thresh)
            & (np.abs(speeds - ref_spd) <= self.cfg.speed_consensus_thresh * 1.5)
        )
        accepted = [r for i, r in enumerate(accepted_reports) if inlier_mask[i]]

        if len(accepted) == 0:
            closest = min(accepted_reports, key=lambda r: abs(r.reported_pos - ref_pos))
            return closest.reported_pos, 0.2, [closest.reporter_id], False

        # Trust-weighted average of accepted consensus reports
        accepted_positions = positions[inlier_mask]
        accepted_trusts = effective_weights[inlier_mask]
        fused_pos = float(np.average(accepted_positions, weights=accepted_trusts))

        # Fusion confidence is high when verified reporters agree
        n_accepted = len(accepted)
        n_total = len(accepted_reports)
        fusion_conf = min(1.0, 0.7 + 0.3 * (n_accepted / max(self.cfg.min_reporters, 1)))
        if n_total > n_accepted:
            fusion_conf *= (1.0 - 0.2 * (n_total - n_accepted) / n_total)

        return fused_pos, float(np.clip(fusion_conf, 0.1, 1.0)), [r.reporter_id for r in accepted], False

    # ───────────────────────────────────────────────── Layer C
    def layer_c_update(self, icv_id: int, passed: bool, slot_t: int):
        """
        Update trust score for an ICV based on verification result.
        Trust = EWMA with asymmetric rates (faster penalty, slower reward).
        """
        if not self.cfg.enable_layer_c:
            return

        ts = self._ensure_trust(icv_id)
        ts.last_seen = slot_t

        if passed:
            ts.trust = (1 - self.cfg.eta_reward) * ts.trust + self.cfg.eta_reward * 1.0
            ts.n_pass += 1
            ts.fail_streak = 0
            # Recovery from quarantine
            if ts.quarantined and ts.trust >= self.cfg.recovery_thresh:
                ts.quarantined = False
        else:
            ts.trust = (1 - self.cfg.eta_penalty) * ts.trust + self.cfg.eta_penalty * 0.0
            ts.n_fail += 1
            ts.fail_streak += 1
            # Quarantine requires sustained failure streak (>= 2) or critically low trust
            if (ts.trust < self.cfg.quarantine_thresh and ts.fail_streak >= 2) or (ts.trust < 0.15):
                ts.quarantined = True

    def decay_absent(self, active_icv_ids: set, slot_t: int):
        """Only decay trust if an ICV has been dormant/absent for > 30 consecutive slots."""
        if not self.cfg.enable_layer_c:
            return
        for icv_id, ts in self.trust_states.items():
            if icv_id not in active_icv_ids and (slot_t - ts.last_seen > 30) and not ts.quarantined:
                ts.trust = (1 - self.cfg.decay_rate) * ts.trust + self.cfg.decay_rate * 0.8

    # ───────────────────────────────────────────────── Main Pipeline
    def process_slot(
        self,
        slot_t: int,
        cpms: Dict[int, List[CPMReport]],   # target_id -> list of CPM reports from different ICVs
        true_pos: Optional[Dict[int, float]] = None   # ground truth for metrics (None in production)
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        """
        Run the full three-layer pipeline for one time slot.
        """
        stats = {
            'slot': slot_t,
            'total_reports': 0,
            'layer_a_rejected': 0,
            'layer_b_outliers': 0,
            'quarantined_count': 0,
            'dt_rmse': 0.0
        }

        fused_positions = {}
        fusion_confidence = {}
        reporters_this_slot = set()
        self.last_accepted_reports = {}

        for target_id, reports in cpms.items():
            stats['total_reports'] += len(reports)

            # ── Layer A: Kinematic Plausibility ──
            a_passed = []
            for r in reports:
                reporters_this_slot.add(r.reporter_id)
                ts = self._ensure_trust(r.reporter_id)

                # Skip quarantined ICVs immediately
                if ts.quarantined:
                    stats['layer_a_rejected'] += 1
                    continue

                if self.layer_a_check(r):
                    a_passed.append(r)
                    if r.target_id >= 0:
                        self.reporter_history[(r.reporter_id, r.target_id)] = (
                            r.reported_pos, r.reported_speed, r.timestamp
                        )
                else:
                    stats['layer_a_rejected'] += 1
                    self.layer_c_update(r.reporter_id, passed=False, slot_t=slot_t)
                    if r.target_id >= 0:
                        self.reporter_history[(r.reporter_id, r.target_id)] = (
                            r.reported_pos, r.reported_speed, r.timestamp
                        )

            # ── Layer B: Cross-Validation & Fusion ──
            raw_b = self.layer_b_fuse(target_id, a_passed, slot_t=slot_t)
            if len(raw_b) == 4:
                fused_pos, conf, accepted_ids, ambiguous_tie = raw_b
            else:
                fused_pos, conf, accepted_ids = raw_b
                ambiguous_tie = False

            accepted_set = set(accepted_ids)
            self.last_accepted_reports[target_id] = accepted_set

            # Update Layer C trust (unless epistemic abstention on unverified first-slot snapshot)
            if not ambiguous_tie:
                for r in a_passed:
                    if r.reporter_id not in accepted_set:
                        stats['layer_b_outliers'] += 1
                        self.layer_c_update(r.reporter_id, passed=False, slot_t=slot_t)
                    else:
                        self.layer_c_update(r.reporter_id, passed=True, slot_t=slot_t)

            fused_positions[target_id] = fused_pos
            fusion_confidence[target_id] = conf

            # Update target history for next slot's Layer A only if we have verified consensus
            if accepted_ids and not ambiguous_tie:
                th = self._ensure_target(target_id)
                inlier_reports = [r for r in a_passed if r.reporter_id in accepted_set]
                best_report = max(inlier_reports, key=lambda r: self._ensure_trust(r.reporter_id).trust)
                th.last_speed = best_report.reported_speed
                th.last_pos = fused_pos
                th.last_slot = slot_t
                th.fused_pos = fused_pos
                th.fusion_confidence = conf

        # ── Layer C: Decay absent ICVs ──
        self.decay_absent(reporters_this_slot, slot_t)

        # Build trust vector
        trust_vector = {}
        for icv_id, ts in self.trust_states.items():
            trust_vector[icv_id] = 0.0 if ts.quarantined else ts.trust
            if ts.quarantined:
                stats['quarantined_count'] += 1

        # Compute DT RMSE if ground truth available
        if true_pos:
            errors = []
            for tid, fp in fused_positions.items():
                if tid in true_pos and tid in fusion_confidence and fusion_confidence[tid] > 0.0:
                    errors.append((fp - true_pos[tid]) ** 2)
            if errors:
                stats['dt_rmse'] = float(np.sqrt(np.mean(errors)))

        self.slot_stats.append(stats)
        return fused_positions, trust_vector, fusion_confidence

    # ───────────────────────────────────────────────── Helpers for WSMM/BCA Integration
    def get_wsmm_weights(self, base_w: np.ndarray, cand: np.ndarray,
                         nic_idx: np.ndarray, icv_idx: np.ndarray) -> np.ndarray:
        """
        Multiply WSMM weights by trust scores. Quarantined ICVs get weight = 0.
        """
        w = base_w.copy()
        for j, icv_gid in enumerate(icv_idx):
            ts = self._ensure_trust(int(icv_gid))
            if ts.quarantined:
                w[:, j] = 0.0         # quarantined -> zero weight -> never selected
                cand[:, j] = False
            else:
                w[:, j] *= ts.trust   # scale by trust
        return w

    def get_bca_weights(self, base_w_v: np.ndarray, base_w_u: np.ndarray,
                        icv_idx: np.ndarray, nic_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale BCA AoI weights by trust (ICVs) and fusion confidence (N-ICVs).
        """
        w_v = base_w_v.copy()
        w_u = base_w_u.copy()

        for j, icv_gid in enumerate(icv_idx):
            ts = self._ensure_trust(int(icv_gid))
            w_v[j] *= ts.trust

        for i, nic_gid in enumerate(nic_idx):
            th = self.target_history.get(int(nic_gid))
            if th is not None and th.fusion_confidence > 0:
                w_u[i] *= th.fusion_confidence

        return np.maximum(w_v, 1e-6), np.maximum(w_u, 1e-6)

    def get_vetted_perceived_size(self, Z: np.ndarray, base_w: np.ndarray,
                                  nic_idx: np.ndarray, icv_idx: np.ndarray) -> np.ndarray:
        """
        Calculates total vetted perceived data size [Mb] delivered to each N-ICV.
        Excludes quarantined ICVs, but uses actual physical data rates (not artificial trust discounts)
        to evaluate Constraint C6 satisfaction (W_u >= s_req).
        """
        U, V = Z.shape
        W = np.zeros(U)
        for j, icv_gid in enumerate(icv_idx):
            ts = self._ensure_trust(int(icv_gid))
            if not ts.quarantined:
                W += Z[:, j] * base_w[:, j]
        return W

    def get_trust_obs_vector(self, N: int, icv_mask: np.ndarray) -> np.ndarray:
        """
        Build a trust observation vector for the PPO state extension.
        Returns array of shape (N,) with trust scores for ICVs, 1.0 for N-ICVs.
        """
        obs = np.ones(N)
        for i in range(N):
            if icv_mask[i]:
                ts = self.trust_states.get(i)
                if ts is not None:
                    obs[i] = 0.0 if ts.quarantined else ts.trust
        return obs

    def get_migration_trust_metadata(self, icv_id: int) -> dict:
        """
        Package trust metadata for DT migration.
        """
        ts = self._ensure_trust(icv_id)
        return {
            'icv_id': icv_id,
            'trust': ts.trust,
            'quarantined': ts.quarantined,
            'n_pass': ts.n_pass,
            'n_fail': ts.n_fail,
            'fail_streak': ts.fail_streak,
            'last_seen': ts.last_seen
        }

    def receive_migration_trust(self, meta: dict):
        """
        Receive trust metadata from a migrating DT.
        CRITICAL: Never reset trust to 1.0 on migration! (prevents whitewashing)
        """
        icv_id = meta['icv_id']
        ts = self._ensure_trust(icv_id)
        ts.trust = min(ts.trust, meta['trust'])
        ts.quarantined = ts.quarantined or meta['quarantined']
        ts.n_pass = max(ts.n_pass, meta['n_pass'])
        ts.n_fail = max(ts.n_fail, meta['n_fail'])
        ts.fail_streak = max(ts.fail_streak, meta['fail_streak'])
