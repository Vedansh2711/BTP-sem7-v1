"""SUMO/TraCI mobility backend. Same interface as KinematicMobility (arrays of size N, `active` mask,
reset(), step() -> (new_idx, gone_idx)). The road network and route files are generated at runtime into a
private temp dir (so parallel training processes do not clash).

Road: nodes A(0,0) - B(L,0); edges AB and BA, `lanes_per_dir` lanes each (two-way, 2 lanes per direction).
Car following: Wiedemann, accel/decel from config, speedFactor ~ N(1, 3.3/12.5) clipped so the desired speed
~ N(12.5, 3.3^2) capped at 16.7 m/s. Step length = tau.
Fixed population (Exp1/2): n_icv + n_nicv vehicles are inserted at random positions; an exiting vehicle is
replaced by a new one of the same type entering at the road start (so the population stays constant).
Dynamic (Exp3): Poisson flows (`flow_veh_per_h` total, split over both directions); each departing vehicle
is an ICV with probability `penetration`.
"""
import atexit, itertools, os, shutil, subprocess, sys, tempfile
import numpy as np
from env.mobility import KinematicMobility

_counter = itertools.count()


def _write(path, text):
    with open(path, 'w') as f:
        f.write(text)


def _tool(name):
    p = shutil.which(name)
    if p:
        return p
    home = os.environ.get('SUMO_HOME')
    if home:
        for ext in ('', '.exe'):
            c = os.path.join(home, 'bin', name + ext)
            if os.path.exists(c):
                return c
    raise FileNotFoundError(f"'{name}' not found. Add SUMO's bin folder to PATH or set SUMO_HOME.")


def _import_traci():
    try:
        import traci
    except ImportError:
        home = os.environ.get('SUMO_HOME')
        if not home:
            raise ImportError("traci not found: `pip install traci` or set SUMO_HOME")
        sys.path.append(os.path.join(home, 'tools'))
        import traci
    return traci


class SumoMobility(KinematicMobility):
    def __init__(self, cfg, rng):
        super().__init__(cfg, rng)
        sc = cfg['mobility'].get('sumo', {}) or {}
        self.gui = bool(sc.get('gui', False))
        self.label = f"dtmig_{os.getpid()}_{next(_counter)}"
        self.traci = _import_traci()
        self.conn = None
        self.dir = tempfile.mkdtemp(prefix='dtmig_sumo_')
        atexit.register(self.close)
        self.id2slot, self.waiting, self.vid = {}, {}, 0
        self._write_files()

    # ------------------------------------------------------------------ files
    def _write_files(self):
        cfg, m = self.cfg, self.cfg['mobility']
        L, lanes, mean, std, vmax = self.L, cfg['road']['lanes_per_dir'], m['speed_mean'], m['speed_std'], m['speed_max']
        d = self.dir
        _write(f'{d}/road.nod.xml',
               f'<nodes><node id="A" x="0" y="0"/><node id="B" x="{L}" y="0"/></nodes>')
        _write(f'{d}/road.edg.xml',
               f'<edges><edge id="AB" from="A" to="B" numLanes="{lanes}" speed="{mean}"/>'
            f'<edge id="BA" from="B" to="A" numLanes="{lanes}" speed="{mean}"/></edges>')
        subprocess.run([_tool('netconvert'), '-n', f'{d}/road.nod.xml', '-e', f'{d}/road.edg.xml',
                        '-o', f'{d}/road.net.xml', '--no-warnings'], check=True, capture_output=True)
        vt = (f'<vType id="car" vClass="passenger" carFollowModel="Wiedemann" accel="{m["accel"]}" '
              f'decel="{m["decel"]}" emergencyDecel="9.0" length="4.5" minGap="2.5" maxSpeed="{vmax}" '
              f'speedFactor="normc(1.0,{std / mean:.4f},0.2,{vmax / mean:.4f})"/>')
        body = vt + '<route id="rAB" edges="AB"/><route id="rBA" edges="BA"/>'
        if self.dynamic:
            rate = cfg['population']['flow_veh_per_h'] / 3600.0 / 2.0           # veh/s per direction
            for e in ('AB', 'BA'):
                body += (f'<flow id="f{e}" type="car" route="r{e}" begin="0" end="1e9" period="exp({rate:.6f})" '
                         f'departLane="random" departSpeed="desired"/>')
        _write(f'{d}/routes.rou.xml', f'<routes>{body}</routes>')

    # ------------------------------------------------------------------ helpers
    def _draw_consts(self, i):
        v, r = self.cfg['vehicle'], self.rng
        self.s_unit[i] = r.uniform(*v['s_unit']); self.s_own[i] = r.uniform(*v['s_own'])
        self.c_unit[i] = r.uniform(*v['c_unit']); self.c_own[i] = r.uniform(*v['c_own'])
        self.D[i] = r.uniform(*v['D'])

    def _add(self, slot, is_icv, pos=None):
        vid = f'd{self.vid}'; self.vid += 1
        route = 'rAB' if self.rng.random() < 0.5 else 'rBA'
        self.conn.vehicle.add(vid, route, typeID='car', depart='now', departLane='random',
                              departPos='base' if pos is None else f'{pos:.2f}', departSpeed='desired')
        self.waiting[vid] = (slot, bool(is_icv))

    def _advance(self):
        c, p = self.conn, self.cfg['population']
        c.simulationStep()
        new, gone = [], []
        for vid in c.simulation.getArrivedIDList():
            slot = self.id2slot.pop(vid, None)
            if slot is None:
                continue
            self.active[slot] = False; gone.append(slot)
            if not self.dynamic:                                  # same-type replacement (appears next step)
                self._add(slot, self.is_icv[slot])
        for vid in c.simulation.getDepartedIDList():
            if vid in self.waiting:
                slot, is_icv = self.waiting.pop(vid)
            elif self.dynamic and vid not in self.id2slot:
                free = np.flatnonzero(~self.active)
                if free.size == 0:
                    c.vehicle.remove(vid); continue               # road full: drop the entry
                slot, is_icv = int(free[0]), self.rng.random() < p['penetration']
            else:
                continue
            self.id2slot[vid] = slot
            self._draw_consts(slot)
            self.active[slot], self.is_icv[slot] = True, is_icv
            new.append(slot)
        for vid, slot in self.id2slot.items():
            lp = c.vehicle.getLanePosition(vid)
            if c.vehicle.getRoadID(vid) == 'BA':
                self.pos[slot], self.dirn[slot] = self.L - lp, -1.0
            else:
                self.pos[slot], self.dirn[slot] = lp, 1.0
            self.speed[slot] = c.vehicle.getSpeed(vid)
        return new, gone

    # ------------------------------------------------------------------ interface
    def reset(self):
        d = self.dir
        args = ['--net-file', f'{d}/road.net.xml', '--route-files', f'{d}/routes.rou.xml',
                '--step-length', str(self.tau), '--no-step-log', '--no-warnings', '--end', '10000000',
                '--time-to-teleport', '-1', '--seed', str(int(self.rng.integers(1, 2 ** 30)))]
        if self.conn is None:
            self.traci.start([_tool('sumo-gui' if self.gui else 'sumo')] + args, label=self.label)
            self.conn = self.traci.getConnection(self.label)
        else:
            self.conn.load(args)
        self.active[:] = False; self.id2slot, self.waiting = {}, {}
        p = self.cfg['population']
        if not self.dynamic:
            n = p['n_icv'] + p['n_nicv']
            for i in range(n):
                self._add(i, i < p['n_icv'], self.rng.uniform(0, self.L - 30.0))
            for _ in range(60):                                   # let insertions happen (unsafe ones are delayed)
                self._advance()
                if int(self.active.sum()) >= n:
                    break
        else:
            for _ in range(int(p['warmup_s'] / self.tau)):
                self._advance()

    def step(self):
        return self._advance()

    def close(self):
        try:
            if self.conn is not None:
                self.conn.close()
        except Exception:
            pass
        self.conn = None
        shutil.rmtree(self.dir, ignore_errors=True)
