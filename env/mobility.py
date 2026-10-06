"""Kinematic fallback mobility (same statistics as the SUMO setup) + SUMO stub."""
import numpy as np


class KinematicMobility:
    """Fixed-size arrays (N_max) with an `active` mask. Fixed-population mode replaces an exiting
    vehicle by a new one of the same type; dynamic mode uses Poisson entries and removal on exit."""

    def __init__(self, cfg, rng):
        self.cfg, self.rng = cfg, rng
        p = cfg['population']
        self.dynamic = p['dynamic']
        self.N = p['n_max_dynamic'] if self.dynamic else p['n_max']
        self.L = cfg['road']['length']
        self.tau = cfg['time']['tau']
        N = self.N
        self.active = np.zeros(N, bool)
        self.is_icv = np.zeros(N, bool)
        self.pos = np.zeros(N)
        self.speed = np.zeros(N)
        self.dirn = np.ones(N)
        for k in ['s_unit', 's_own', 'c_unit', 'c_own', 'D']:
            setattr(self, k, np.zeros(N))

    def _speed_draw(self):
        m = self.cfg['mobility']
        return float(np.clip(self.rng.normal(m['speed_mean'], m['speed_std']), m['speed_min'], m['speed_max']))

    def _spawn(self, i, is_icv, pos, dirn):
        v, r = self.cfg['vehicle'], self.rng
        self.active[i], self.is_icv[i] = True, is_icv
        self.pos[i], self.dirn[i], self.speed[i] = pos, dirn, self._speed_draw()
        self.s_unit[i] = r.uniform(*v['s_unit']); self.s_own[i] = r.uniform(*v['s_own'])
        self.c_unit[i] = r.uniform(*v['c_unit']); self.c_own[i] = r.uniform(*v['c_own'])
        self.D[i] = r.uniform(*v['D'])

    def reset(self):
        p, r = self.cfg['population'], self.rng
        self.active[:] = False
        if not self.dynamic:
            n_i, n_u = p['n_icv'], p['n_nicv']
            assert n_i + n_u <= self.N
            for i in range(n_i + n_u):
                self._spawn(i, i < n_i, r.uniform(0, self.L), r.choice([-1.0, 1.0]))
        else:
            for _ in range(int(p['warmup_s'] / self.tau)):
                self.step()

    def step(self):
        """Advance one slot. Returns (new_idx, gone_idx)."""
        m, p, r = self.cfg['mobility'], self.cfg['population'], self.rng
        new, gone = [], []
        a = np.flatnonzero(self.active)
        dv = np.clip(r.normal(0, m['jitter_std'], a.size), -m['decel'], m['accel'])
        self.speed[a] = np.clip(self.speed[a] + dv, m['speed_min'], m['speed_max'])
        self.pos[a] += self.dirn[a] * self.speed[a] * self.tau
        for i in a:
            if self.pos[i] < 0 or self.pos[i] > self.L:
                if self.dynamic:
                    self.active[i] = False; gone.append(i)
                else:
                    d = self.dirn[i]
                    self._spawn(i, self.is_icv[i], 0.0 if d > 0 else self.L, d); new.append(i)
        if self.dynamic:
            lam = p['flow_veh_per_h'] / 3600.0 * self.tau
            for _ in range(r.poisson(lam)):
                free = np.flatnonzero(~self.active)
                if free.size == 0:
                    break
                d = r.choice([-1.0, 1.0])
                i = free[0]
                self._spawn(i, r.random() < p['penetration'], 0.0 if d > 0 else self.L, d)
                new.append(i)
        return new, gone


def make_mobility(cfg, rng):
    if cfg['mobility']['backend'] == 'sumo':
        from env.sumo_mobility import SumoMobility
        return SumoMobility(cfg, rng)
    return KinematicMobility(cfg, rng)
