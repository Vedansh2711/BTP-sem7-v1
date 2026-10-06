"""Logic test of env/sumo_mobility.py against a MOCK TraCI (real SUMO is not needed). It validates slot/ID
bookkeeping, replacement, dynamic arrivals and the env integration, NOT SUMO's own dynamics."""
import copy, unittest
import numpy as np, yaml
import env.sumo_mobility as sm
from env.dt_env import DTEnv
from sub3_asdmm.mda import mda_action


class FakeConn:
    def __init__(self, dynamic, seed=0):
        self.dynamic, self.rng = dynamic, np.random.default_rng(seed)
        self.v, self.queue, self.n = {}, [], 0
        self.simulation, self.vehicle = self, self
        self._dep, self._arr = [], []

    def load(self, args): self.v, self.queue, self._dep, self._arr = {}, [], [], []
    def close(self): pass

    def add(self, vid, route, typeID=None, depart=None, departLane=None, departPos='base', departSpeed=None):
        self.queue.append((vid, route[1:], 0.0 if departPos == 'base' else float(departPos)))

    def simulationStep(self):
        self._dep, self._arr = [], []
        for vid, p in list(self.v.items()):
            p['pos'] += p['speed']
            if p['pos'] > 2000: del self.v[vid]; self._arr.append(vid)
        if self.dynamic and self.rng.random() < 0.3:
            self.queue.append((f'f{self.n}', 'AB' if self.rng.random() < .5 else 'BA', 0.0)); self.n += 1
        for vid, edge, pos in self.queue:
            self.v[vid] = dict(edge=edge, pos=pos, speed=float(self.rng.uniform(8, 16))); self._dep.append(vid)
        self.queue = []

    def getDepartedIDList(self): return self._dep
    def getArrivedIDList(self): return self._arr
    def getRoadID(self, vid): return self.v[vid]['edge']
    def getLanePosition(self, vid): return self.v[vid]['pos']
    def getSpeed(self, vid): return self.v[vid]['speed']
    def remove(self, vid): self.v.pop(vid, None)


class FakeTraci:
    def __init__(self, dyn): self.c = FakeConn(dyn)
    def start(self, cmd, label=None): pass
    def getConnection(self, label): return self.c


class T(unittest.TestCase):
    def setUp(self):
        self.orig = (sm._tool, sm._import_traci, sm.subprocess.run)
        sm._tool = lambda n: n
        sm.subprocess.run = lambda *a, **k: None

    def tearDown(self):
        sm._tool, sm._import_traci, sm.subprocess.run = self.orig

    def make(self, dynamic, **pop):
        cfg = yaml.safe_load(open('config.yaml')); cfg['mobility']['backend'] = 'sumo'
        cfg['population'].update(dynamic=dynamic, **pop); cfg['sub2']['max_iter'] = 20
        sm._import_traci = lambda: FakeTraci(dynamic)
        return DTEnv(cfg, seed=0)

    def test_fixed_population_constant_with_replacement(self):
        env = self.make(False, n_icv=5, n_nicv=2); env.reset()
        for _ in range(200):                                   # > road length / speed: forces exits
            env.mob.step()
            self.assertGreaterEqual(int(env.mob.active.sum()), 6)   # at most 1 slot waits for re-insertion... per exit
        self.assertEqual(int(env.mob.is_icv[:5].sum()), 5)     # slot types never change
        self.assertFalse(env.mob.is_icv[5:7].any())

    def test_positions_and_direction(self):
        env = self.make(False, n_icv=4, n_nicv=1); env.reset(); m = env.mob
        a = np.flatnonzero(m.active)
        self.assertTrue(((m.pos[a] >= 0) & (m.pos[a] <= 2000)).all())
        self.assertTrue(set(m.dirn[a]) <= {-1.0, 1.0})

    def test_dynamic_arrivals_and_penetration(self):
        env = self.make(True, penetration=0.7, warmup_s=100); env.reset(); n_icv = n_all = 0
        for _ in range(300):
            env.mob.step(); a = env.mob.active
            n_icv += (a & env.mob.is_icv).sum(); n_all += a.sum()
        self.assertGreater(n_all, 0)
        self.assertGreater(n_icv / n_all, 0.4)

    def test_env_episode_runs_on_sumo_backend(self):
        env = self.make(False, n_icv=4, n_nicv=2); env.cfg['time']['T'] = 20; env.T = 20
        env.reset(); d = False; n = 0
        while not d:
            _, r, d, info = env.step(mda_action(env)); n += 1
            self.assertTrue(np.isfinite(r))
        self.assertEqual(n, 20); self.assertIn('amwaoi', info)


if __name__ == '__main__':
    unittest.main()
