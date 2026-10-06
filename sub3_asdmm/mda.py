"""MDA baseline: each ICV accesses the nearest RSU, migrates only when its serving RSU changes (T_pre=0)."""
import numpy as np


def mda_action(env):
    x = np.zeros(env.N, int)
    d = np.where(env.in_range, env.dist, np.inf)
    x = np.argmin(d, axis=1)
    return dict(x=x, y=np.zeros(env.N, int))


def random_action(env, rng):
    mk = env.action_masks()
    x = np.array([rng.choice(np.flatnonzero(mk['x_mask'][i])) if mk['x_mask'][i].any() else 0 for i in range(env.N)])
    y = np.array([rng.choice(np.flatnonzero(mk['y_mask'][i])) for i in range(env.N)])
    return dict(x=x, y=y)


class MDAAgent:
    name = 'mda'

    def __init__(self, env, cfg=None, seed=0):
        env.predictive = False

    def act(self, obs, masks, explore=False):
        return None, {}                       # decision uses env state directly (see run_episode)

    def observe(self, *a, **k): pass
    def update(self): return None
