import copy, unittest
import numpy as np, yaml
from env.dt_env import DTEnv
from env.channel import gain, H_mhz, rate_mbps

BASE = yaml.safe_load(open('config.yaml'))


def make(n_icv, n_nicv, **over):
    cfg = copy.deepcopy(BASE)
    cfg['population'].update(n_icv=n_icv, n_nicv=n_nicv)
    for k, v in over.items():
        sec, key = k.split('__'); cfg[sec][key] = v
    env = DTEnv(cfg, seed=0); env.reset()
    return env


def place(env, pos, dt=None):
    for i, p in enumerate(pos):
        env.mob.pos[i] = p; env.mob.dirn[i] = 1.0
        if dt is not None: env.dt_loc[i] = dt[i]
    env._prepare_slot()


def act(env, x=None, y=None):
    return dict(x=np.zeros(env.N, int) if x is None else np.array(x), y=np.zeros(env.N, int) if y is None else np.array(y))


def pad(env, vals, fill=0):
    a = np.full(env.N, fill); a[:len(vals)] = vals; return a


class T(unittest.TestCase):
    def test_hand_computed_single_icv(self):
        env = make(1, 0)
        place(env, [250.0], dt=[0])
        m = env.mob
        _, _, _, info = env.step(act(env, x=pad(env, [0])))
        H = H_mhz(gain(0.0, 10.0))
        s = m.s_own[0] + m.s_unit[0]                                   # Eq (1)
        t_exp = s * 1.0 / rate_mbps(20.0, H)                           # Eq (3), B = Bmax (sole vehicle)
        e_exp = m.c_own[0] * m.s_own[0] * 1.0 / 4.0                    # Eq (6), f = F
        self.assertAlmostEqual(info['T_tran'][0], t_exp, places=9)
        self.assertAlmostEqual(info['T_exe'][0], e_exp, places=9)
        self.assertAlmostEqual(info['T_sum'][0], t_exp + e_exp, places=9)   # no migration: Eq (12)
        self.assertEqual(info['omega'], 0.0 if t_exp + e_exp <= 0.5 else 0.5)

    def test_eq4_failure_penalties(self):
        env = make(2, 1)
        place(env, [500.0, 500.0, 500.0], dt=[0, 1, 0])               # both ICVs in range of RSU 0 and 1
        self.assertEqual(env.Z[2, :2].sum(), 2)                        # both selected
        _, _, _, info = env.step(act(env, x=pad(env, [0, 1])))         # ICVs split across RSUs
        self.assertEqual(info['sx'][2], -1)
        self.assertEqual(info['T_tran'][2], 2.0); self.assertEqual(info['T_exe'][2], 2.0)
        self.assertFalse(info['valid'][2]); self.assertEqual(env.A[2], 1.0)
        place(env, [500.0, 500.0, 500.0], dt=[0, 0, 0])
        _, _, _, info = env.step(act(env, x=pad(env, [0, 0])))         # same RSU -> Eq (4) holds
        self.assertEqual(info['sx'][2], 0); self.assertLess(info['T_tran'][2], 2.0)

    def test_empty_selection_failure(self):
        env = make(1, 1)
        place(env, [100.0, 1900.0], dt=[0, 3])                         # out of sensing range
        self.assertEqual(env.Z.sum(), 0)
        _, _, _, info = env.step(act(env, x=pad(env, [0])))
        self.assertEqual(info['T_exe'][1], 2.0); self.assertEqual(env.A[1], 1.0)

    def test_aoi_recursion(self):
        env = make(1, 0, aoi__t_req_icv=1e-6)
        for n in range(1, 4):
            place(env, [250.0], dt=[0]); env.step(act(env, x=pad(env, [0])))
            self.assertEqual(env.A[0], n)
        env.cfg['aoi']['t_req_icv'] = 10.0
        place(env, [250.0], dt=[0]); env.step(act(env, x=pad(env, [0])))
        self.assertEqual(env.A[0], 0.0)

    def test_predictive_vs_reactive_migration(self):
        env = make(1, 0)
        place(env, [260.0], dt=[0]); D = env.mob.D[0]
        # t=0: agent starts transfer to RSU 1 while still served by RSU 0
        _, _, _, i0 = env.step(act(env, x=pad(env, [0]), y=pad(env, [2])))
        self.assertEqual(i0['mig_count'][1], 1); self.assertEqual(i0['T_amigr'][0], 0.0)
        place(env, [500.0], dt=[0]); env.dt_loc[0] = 0
        _, _, _, i1 = env.step(act(env, x=pad(env, [0]), y=pad(env, [2])))   # same target again: not re-counted
        self.assertEqual(i1['mig_count'].sum(), 0)
        place(env, [600.0], dt=[0]); env.dt_loc[0] = 0
        _, _, _, i2 = env.step(act(env, x=pad(env, [1])))                    # access switches at t=2
        wired = 2.4e-4 * D * 500.0
        self.assertAlmostEqual(i2['T_wired'][0], wired); self.assertAlmostEqual(i2['T_pre'][0], 2.0)
        self.assertAlmostEqual(i2['T_amigr'][0], max(wired - 2.0, 0.0)); self.assertEqual(env.dt_loc[0], 1)
        # reactive: switch to RSU 2 with no prior y
        place(env, [1000.0], dt=[1])
        _, _, _, i3 = env.step(act(env, x=pad(env, [2])))
        self.assertEqual(i3['T_pre'][0], 0.0); self.assertAlmostEqual(i3['T_amigr'][0], 2.4e-4 * D * 500.0)
        self.assertEqual(i3['mig_count'][2], 1)

    def test_c9_penalty(self):
        env = make(4, 0)
        place(env, [250.0] * 4, dt=[0] * 4)
        _, r, _, info = env.step(act(env, x=pad(env, [0] * 4), y=pad(env, [4] * 4)))
        self.assertEqual(info['mig_count'][3], 4); self.assertEqual(info['penalty'], 1.0)
        self.assertAlmostEqual(r, -info['omega'] - 1.0)

    def test_masks_and_obs(self):
        env = make(3, 1)
        place(env, [100.0, 600.0, 1000.0, 1500.0], dt=[0, 1, 2, 3])
        mk = env.action_masks()
        self.assertTrue(mk['x_mask'][0, 0] and not mk['x_mask'][0, 3])
        self.assertFalse(mk['x_mask'][3].any())                         # N-ICV has no x head
        self.assertFalse(mk['y_mask'][1, 2]); self.assertTrue(mk['y_mask'][1, 1])
        self.assertTrue(np.isfinite(env._obs()).all()); self.assertEqual(env._obs().shape, (env.obs_dim,))

    def test_episode_terminates_and_amwaoi(self):
        from sub3_asdmm.mda import mda_action
        env = make(3, 1, time__T=10)
        d = False; n = 0
        while not d:
            _, _, d, info = env.step(mda_action(env)); n += 1
        self.assertEqual(n, 10); self.assertAlmostEqual(info['amwaoi'], np.mean(env.omega_hist))


if __name__ == '__main__':
    unittest.main()
