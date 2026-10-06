"""
False CPM Resistance: Three-Layer Defence Pipeline
===================================================
Layer A: Kinematic Plausibility Filter  — rejects physically impossible single-step reports
Layer B: Cross-Validation & Fusion     — multi-ICV consensus + trust-weighted median fusion
Layer C: Trust Management & Quarantine — long-term EWMA trust score, quarantine repeat offenders

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
    a_max: float = 10.0            # m/s² — max physically plausible acceleration
    v_max: float = 50.0            # m/s  — max physically plausible speed (180 km/h)
    tau: float = 1.0               # s    — slot duration (matches config.yaml time.tau)

    # Layer B: Cross-Validation & Fusion
    enable_layer_b: bool = True
    consensus_thresh: float = 5.0  # m — max acceptable mean pairwise position disagreement
    outlier_thresh: float = 8.0    # m — individual report vs trust-weighted median
    min_reporters: int = 2         # minimum ICVs required for consensus

    # Layer C: Trust & Quarantine
    enable_layer_c: bool = True
    trust_init: float = 1.0        # initial trust for new ICVs
    eta_reward: float = 0.2        # EWMA learning rate on verification pass
    eta_penalty: float = 0.3       # EWMA learning rate on verification fail (asymmetric: faster penalty)
    decay_rate: float = 0.01       # trust decay per slot if no CPM received
    quarantine_thresh: float = 0.3 # below this trust → quarantined (excluded from optimisation)
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
    Sits between CPM reception and the base algorithms (WSMM, BCA, ASDMM).
    
    Usage in dt_env.py:
        defence = FalseCPMResistance(cfg)
        
        # In _prepare_slot(), after collecting CPMs:
        vetted_reports, trust_vec, fusion_conf = defence.process_slot(
            slot_t, cpms, mob
        )
        
        # Then pass trust_vec into WSMM weight computation:
        #   w = q * s_unit * tau * trust[reporter]    (zero if quarantined)
        #
        # And fusion_conf into BCA weight computation:
        #   w_u = (A_u + 1) * fusion_conf[u]          (for N-ICVs)
    """

    def __init__(self, fcpm_cfg: FalseCPMConfig = None):
        self.cfg = fcpm_cfg or FalseCPMConfig()
        self.trust_states: Dict[int, ICVTrustState] = {}     # ICV id → trust state
        self.target_history: Dict[int, TargetHistory] = {}   # N-ICV id → history
        self.slot_stats: List[dict] = []                      # per-slot statistics

    def reset(self):
        """Call at episode reset."""
        self.trust_states.clear()
        self.target_history.clear()
        self.slot_stats.clear()

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
        Kinematic Plausibility Filter.
        Rejects reports where:
          |v_new - v_old| > a_max * tau   (impossible acceleration)
          |x_new - x_old| > v_max * tau   (impossible displacement)
        Returns True if report PASSES, False if REJECTED.
        """
        if not self.cfg.enable_layer_a:
            return True

        th = self._ensure_target(report.target_id)

        # First report for this target — no history to compare against, accept it
        if th.last_pos is None or th.last_slot < report.timestamp - 2:
            return True

        dt = (report.timestamp - th.last_slot) * self.cfg.tau
        if dt <= 0:
            return True

        # Check 1: velocity change feasibility
        if th.last_speed is not None:
            dv = abs(report.reported_speed - th.last_speed)
            if dv > self.cfg.a_max * dt * 1.5:  # 1.5x safety margin for noise
                return False

        # Check 2: position change feasibility
        dx = abs(report.reported_pos - th.last_pos)
        if dx > 1000.0:
            return True
        if dx > self.cfg.v_max * dt * 1.5:  # 1.5x safety margin
            return False

        return True

    # ───────────────────────────────────────────────── Layer B
    def layer_b_fuse(self, target_id: int, accepted_reports: List[CPMReport]) -> Tuple[float, float, List[int]]:
        """
        Cross-Validation & Fusion.
        1. Compute mean pairwise distance between reporters
        2. If consensus fails, reject all
        3. Reject individual outliers vs trust-weighted median
        4. Fuse accepted reports into a single estimate
        
        Returns: (fused_position, fusion_confidence, list_of_accepted_reporter_ids)
        """
        if not self.cfg.enable_layer_b or len(accepted_reports) == 0:
            if len(accepted_reports) == 1:
                r = accepted_reports[0]
                return r.reported_pos, 0.5, [r.reporter_id]  # single report → half confidence
            elif len(accepted_reports) == 0:
                return 0.0, 0.0, []
            # If layer B disabled, just average
            positions = [r.reported_pos for r in accepted_reports]
            return np.mean(positions), 1.0, [r.reporter_id for r in accepted_reports]

        positions = np.array([r.reported_pos for r in accepted_reports])

        # Single reporter → no consensus possible, return with lower confidence
        if len(accepted_reports) < self.cfg.min_reporters:
            r = accepted_reports[0]
            return r.reported_pos, 0.3, [r.reporter_id]

        # Consensus test: mean pairwise distance
        n = len(positions)
        pairwise_sum = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                pairwise_sum += abs(positions[i] - positions[j])
                count += 1
        mean_pairwise = pairwise_sum / max(count, 1)

        if mean_pairwise > self.cfg.consensus_thresh:
            # Consensus fails — group disagrees too much
            # Fall back to trust-weighted median and reject outliers
            pass

        # Compute trust-weighted median
        trusts = np.array([self._ensure_trust(r.reporter_id).trust for r in accepted_reports])
        trusts = np.maximum(trusts, 0.01)  # avoid zero weights
        
        # Weighted median: sort by position, pick where cumulative weight crosses 50%
        order = np.argsort(positions)
        sorted_pos = positions[order]
        sorted_w = trusts[order]
        cum_w = np.cumsum(sorted_w) / sorted_w.sum()
        median_idx = np.searchsorted(cum_w, 0.5)
        median_idx = min(median_idx, len(sorted_pos) - 1)
        weighted_median = sorted_pos[median_idx]

        # Reject individual outliers
        accepted = []
        accepted_positions = []
        accepted_trusts = []
        rejected_ids = []

        for i, r in enumerate(accepted_reports):
            if abs(r.reported_pos - weighted_median) <= self.cfg.outlier_thresh:
                accepted.append(r)
                accepted_positions.append(r.reported_pos)
                accepted_trusts.append(trusts[i])
            else:
                rejected_ids.append(r.reporter_id)

        if len(accepted) == 0:
            # All rejected — use the one closest to median
            closest = min(accepted_reports, key=lambda r: abs(r.reported_pos - weighted_median))
            return closest.reported_pos, 0.1, [closest.reporter_id]

        # Trust-weighted average of accepted reports
        accepted_positions = np.array(accepted_positions)
        accepted_trusts = np.array(accepted_trusts)
        fused_pos = np.average(accepted_positions, weights=accepted_trusts)

        # Fusion confidence: higher with more agreeing reporters
        n_accepted = len(accepted)
        n_total = len(accepted_reports)
        fusion_conf = min(1.0, n_accepted / max(self.cfg.min_reporters, 1)) * (1.0 - 0.3 * (n_total - n_accepted) / max(n_total, 1))

        return fused_pos, fusion_conf, [r.reporter_id for r in accepted]

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
            # Quarantine check
            if ts.trust < self.cfg.quarantine_thresh:
                ts.quarantined = True

    def decay_absent(self, active_icv_ids: set, slot_t: int):
        """Decay trust of ICVs that did NOT send any CPM this slot."""
        if not self.cfg.enable_layer_c:
            return
        for icv_id, ts in self.trust_states.items():
            if icv_id not in active_icv_ids and ts.last_seen < slot_t:
                ts.trust = (1 - self.cfg.decay_rate) * ts.trust + self.cfg.decay_rate * 0.5

    # ───────────────────────────────────────────────── Main Pipeline
    def process_slot(
        self,
        slot_t: int,
        cpms: Dict[int, List[CPMReport]],   # target_id → list of CPM reports from different ICVs
        true_pos: Optional[Dict[int, float]] = None   # ground truth for metrics (None in production)
    ) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
        """
        Run the full three-layer pipeline for one time slot.

        Args:
            slot_t: current slot number
            cpms: dict mapping target_id → list of CPMReports from different ICVs
            true_pos: optional ground truth positions for evaluation

        Returns:
            fused_positions: target_id → fused position estimate
            trust_vector: icv_id → current trust score (0 = quarantined)
            fusion_confidence: target_id → confidence in the fused estimate [0,1]
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
                else:
                    stats['layer_a_rejected'] += 1
                    self.layer_c_update(r.reporter_id, passed=False, slot_t=slot_t)

            # ── Layer B: Cross-Validation & Fusion ──
            fused_pos, conf, accepted_ids = self.layer_b_fuse(target_id, a_passed)
            
            # Identify outlier reporters (in a_passed but not in accepted_ids)
            accepted_set = set(accepted_ids)
            for r in a_passed:
                if r.reporter_id not in accepted_set:
                    stats['layer_b_outliers'] += 1
                    self.layer_c_update(r.reporter_id, passed=False, slot_t=slot_t)
                else:
                    self.layer_c_update(r.reporter_id, passed=True, slot_t=slot_t)

            fused_positions[target_id] = fused_pos
            fusion_confidence[target_id] = conf

            # Update target history for next slot's Layer A
            th = self._ensure_target(target_id)
            if a_passed:
                # Use the fused position as the "true" history for next slot
                best_report = max(a_passed, key=lambda r: self._ensure_trust(r.reporter_id).trust)
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
                if tid in true_pos:
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
        
        Args:
            base_w: (U, V) original weight matrix = q * s_unit * tau
            cand: (U, V) bool candidate mask
            nic_idx: global indices of N-ICVs (U items)
            icv_idx: global indices of ICVs (V items)
            
        Returns:
            modified_w: (U, V) trust-scaled weight matrix
        """
        w = base_w.copy()
        for j, icv_gid in enumerate(icv_idx):
            ts = self._ensure_trust(icv_gid)
            if ts.quarantined:
                w[:, j] = 0.0         # quarantined → zero weight → never selected
                cand[:, j] = False
            else:
                w[:, j] *= ts.trust   # scale by trust
        return w

    def get_bca_weights(self, base_w_v: np.ndarray, base_w_u: np.ndarray,
                        icv_idx: np.ndarray, nic_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale BCA AoI weights by trust (ICVs) and fusion confidence (N-ICVs).
        
        Args:
            base_w_v: AoI weights for ICVs (= A_v + 1)
            base_w_u: AoI weights for N-ICVs (= A_u + 1)
            icv_idx: global ICV indices
            nic_idx: global N-ICV indices
            
        Returns:
            (modified_w_v, modified_w_u)
        """
        w_v = base_w_v.copy()
        w_u = base_w_u.copy()

        # ICVs: scale by trust
        for j, icv_gid in enumerate(icv_idx):
            ts = self._ensure_trust(icv_gid)
            w_v[j] *= ts.trust

        # N-ICVs: scale by fusion confidence
        for i, nic_gid in enumerate(nic_idx):
            th = self.target_history.get(nic_gid)
            if th is not None:
                w_u[i] *= th.fusion_confidence

        return np.maximum(w_v, 1e-6), np.maximum(w_u, 1e-6)

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
        The receiving RSU should call receive_migration_trust() with this data.
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
        # Take the MINIMUM of current and incoming trust (conservative)
        ts.trust = min(ts.trust, meta['trust'])
        ts.quarantined = ts.quarantined or meta['quarantined']
        ts.n_pass = max(ts.n_pass, meta['n_pass'])
        ts.n_fail = max(ts.n_fail, meta['n_fail'])
        ts.fail_streak = max(ts.fail_streak, meta['fail_streak'])
