"""
Attack Injector: Simulates malicious ICVs sending falsified CPMs.
Used to evaluate the 3-layer False CPM Resistance pipeline.

Supports 4 attack types from the threat model:
  1. Ghost Injection      — reports vehicles/objects that don't exist
  2. State Manipulation   — alters a real vehicle's position/speed
  3. Quality Inflation    — overstates quality to win WSMM selection
  4. Suppression          — withholds perception of a real N-ICV
"""
import numpy as np
from false_cpm_resistance import CPMReport


class AttackInjector:
    """
    Marks a fraction p_mal of ICVs as malicious and modifies their CPMs.
    
    Usage:
        injector = AttackInjector(p_mal=0.2, attack_mix={'ghost': 0.3, 'shift': 0.4, ...})
        injector.mark_malicious(icv_indices, rng)
        
        # When generating CPMs:
        if injector.is_malicious(icv_id):
            report = injector.attack(honest_report, rng, mob)
    """

    def __init__(self, p_mal: float = 0.2,
                 attack_mix: dict = None,
                 ghost_range: float = 200.0,     # max displacement for ghost objects
                 shift_mean: float = 15.0,        # mean position shift (meters)
                 shift_std: float = 5.0,           # std of position shift
                 speed_shift_mean: float = 8.0,    # mean speed falsification (m/s)
                 quality_inflate: float = 3.0,     # multiplier for quality inflation
                 suppress_prob: float = 0.7):      # probability of suppression per slot
        self.p_mal = p_mal
        self.attack_mix = attack_mix or {
            'ghost': 0.25,
            'shift': 0.35,
            'quality': 0.20,
            'suppress': 0.20
        }
        self.ghost_range = ghost_range
        self.shift_mean = shift_mean
        self.shift_std = shift_std
        self.speed_shift_mean = speed_shift_mean
        self.quality_inflate = quality_inflate
        self.suppress_prob = suppress_prob

        self.malicious_set = set()
        self.attack_type = {}   # icv_id → assigned attack type

    def mark_malicious(self, icv_indices: np.ndarray, rng: np.random.Generator):
        """Randomly select p_mal fraction of ICVs as malicious."""
        self.malicious_set.clear()
        self.attack_type.clear()

        n = len(icv_indices)
        n_mal = max(1, int(self.p_mal * n)) if self.p_mal > 0 else 0
        if n_mal == 0:
            return

        mal_idx = rng.choice(icv_indices, size=min(n_mal, n), replace=False)
        attack_types = list(self.attack_mix.keys())
        attack_probs = [self.attack_mix[k] for k in attack_types]
        total_p = sum(attack_probs)
        attack_probs = [p / total_p for p in attack_probs]

        for idx in mal_idx:
            self.malicious_set.add(int(idx))
            self.attack_type[int(idx)] = rng.choice(attack_types, p=attack_probs)

    def is_malicious(self, icv_id: int) -> bool:
        return icv_id in self.malicious_set

    def get_attack_type(self, icv_id: int) -> str:
        return self.attack_type.get(icv_id, 'none')

    def attack(self, honest_report: CPMReport, rng: np.random.Generator,
               road_length: float = 2000.0) -> CPMReport:
        """
        Apply the assigned attack to an honest CPM report.
        Returns a falsified CPMReport (or None for suppression).
        """
        icv_id = honest_report.reporter_id
        atype = self.get_attack_type(icv_id)

        if atype == 'ghost':
            # Ghost injection: report a non-existent vehicle at a random position
            return CPMReport(
                reporter_id=icv_id,
                target_id=-abs(hash((icv_id, honest_report.timestamp))) % 10000 - 1000,  # fake target ID
                reported_pos=rng.uniform(0, road_length),
                reported_speed=rng.uniform(5, 30),
                quality=honest_report.quality,
                s_unit=honest_report.s_unit,
                timestamp=honest_report.timestamp
            )

        elif atype == 'shift':
            # State manipulation: shift the real vehicle's position and speed
            shift = rng.normal(self.shift_mean, self.shift_std)
            speed_shift = rng.normal(self.speed_shift_mean, self.speed_shift_mean * 0.3)
            return CPMReport(
                reporter_id=icv_id,
                target_id=honest_report.target_id,
                reported_pos=honest_report.reported_pos + shift,
                reported_speed=honest_report.reported_speed + speed_shift,
                quality=honest_report.quality,
                s_unit=honest_report.s_unit,
                timestamp=honest_report.timestamp
            )

        elif atype == 'quality':
            # Quality inflation: overstate quality to win WSMM selection
            return CPMReport(
                reporter_id=icv_id,
                target_id=honest_report.target_id,
                reported_pos=honest_report.reported_pos,  # position is correct!
                reported_speed=honest_report.reported_speed,
                quality=honest_report.quality * self.quality_inflate,
                s_unit=honest_report.s_unit,
                timestamp=honest_report.timestamp
            )

        elif atype == 'suppress':
            # Suppression: don't send the CPM at all
            if rng.random() < self.suppress_prob:
                return None  # suppressed
            return honest_report  # occasionally sends honest data to avoid detection

        return honest_report  # fallback: no attack
