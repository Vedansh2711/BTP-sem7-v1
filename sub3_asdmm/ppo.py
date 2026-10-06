"""PPO agent (ASDMM). Factorised categorical heads: x_i in {0..K-1} (masked to RSUs in range, ICVs only),
y_i in {0 (none), 1..K} (masked: target != DT's current RSU). NOT executed in the authoring sandbox (no torch)."""
import numpy as np
import torch, torch.nn as nn
from torch.distributions import Categorical

NEG = -1e9


class PolicyNet(nn.Module):                      # 256-128-128, Tanh
    def __init__(self, obs_dim, N, K):
        super().__init__()
        self.N, self.K = N, K
        self.body = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 128), nn.Tanh(),
                                  nn.Linear(128, 128), nn.Tanh())
        self.out = nn.Linear(128, N * (2 * K + 1))

    def forward(self, obs, xm, ym):
        o = self.out(self.body(obs)).view(-1, self.N, 2 * self.K + 1)
        return o[..., :self.K].masked_fill(~xm, NEG), o[..., self.K:].masked_fill(~ym, NEG)


class ValueNet(nn.Module):                       # 128-128-64, ReLU
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


def _t(a, dtype=torch.float32):
    return torch.as_tensor(np.asarray(a), dtype=dtype)


class PPOAgent:
    name = 'ppo'

    def __init__(self, env, cfg, seed=0):
        torch.manual_seed(seed)
        self.N, self.K, self.p = env.N, env.K, cfg['ppo']
        self.pi, self.v = PolicyNet(env.obs_dim, self.N, self.K), ValueNet(env.obs_dim)
        self.opt_pi = torch.optim.Adam(self.pi.parameters(), lr=self.p['lr'])
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=self.p['lr'])
        self.buf, self.cap, self.rng = [], 2000, np.random.default_rng(seed)
        env.predictive = True

    def _dists(self, obs, xm, ym):
        lx, ly = self.pi(obs, xm, ym)
        return Categorical(logits=lx), Categorical(logits=ly)

    @staticmethod
    def _logp(dx, dy, xa, ya, xr, yr):
        return (dx.log_prob(xa) * xr).sum(-1) + (dy.log_prob(ya) * yr).sum(-1)

    @torch.no_grad()
    def act(self, obs, masks, explore=True):
        o = _t(obs).unsqueeze(0); xm = _t(masks['x_mask'], torch.bool).unsqueeze(0); ym = _t(masks['y_mask'], torch.bool).unsqueeze(0)
        dx, dy = self._dists(o, xm, ym)
        xa, ya = (dx.sample(), dy.sample()) if explore else (dx.probs.argmax(-1), dy.probs.argmax(-1))
        xr, yr = _t(masks['x_rel']).unsqueeze(0), _t(masks['y_rel']).unsqueeze(0)
        logp = self._logp(dx, dy, xa, ya, xr, yr).item()
        return dict(x=xa[0].numpy(), y=ya[0].numpy()), dict(logp=logp, masks=masks)

    def observe(self, obs, aux, action, r, obs2, done):
        m = aux['masks']
        tr = dict(obs=obs, obs2=obs2, xm=m['x_mask'], ym=m['y_mask'], xr=m['x_rel'], yr=m['y_rel'],
                  xa=action['x'], ya=action['y'], logp=aux['logp'], r=r, done=float(done))
        if self.p['paper_buffer']:                                  # Alg. 3 literally: random replacement
            if len(self.buf) < self.cap: self.buf.append(tr)
            else: self.buf[self.rng.integers(self.cap)] = tr
        else:
            self.buf.append(tr)

    def update(self):
        B = self.buf
        T = lambda k, dt=torch.float32: torch.as_tensor(np.stack([b[k] for b in B]), dtype=dt)
        obs, obs2, r, done, old = T('obs'), T('obs2'), T('r'), T('done'), T('logp')
        xm, ym, xr, yr = T('xm', torch.bool), T('ym', torch.bool), T('xr'), T('yr')
        xa, ya = T('xa', torch.long), T('ya', torch.long)
        n, p, stats = len(B), self.p, []
        for _ in range(p['passes']):
            with torch.no_grad():
                target = r + p['gamma'] * self.v(obs2) * (1 - done)       # V_target = R + gamma V(s')
                adv = target - self.v(obs)                                # A = V_target - V(s)
                if p['adv_norm']: adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            perm = torch.randperm(n)
            for i in range(0, n, p['minibatch']):
                ix = perm[i:i + p['minibatch']]
                dx, dy = self._dists(obs[ix], xm[ix], ym[ix])
                logp = self._logp(dx, dy, xa[ix], ya[ix], xr[ix], yr[ix])
                ratio = torch.exp(logp - old[ix])
                s1, s2 = ratio * adv[ix], torch.clamp(ratio, 1 - p['clip'], 1 + p['clip']) * adv[ix]
                ent = (dx.entropy() * xr[ix]).sum(-1) + (dy.entropy() * yr[ix]).sum(-1)
                loss_pi = -torch.min(s1, s2).mean() - p['entropy_coef'] * ent.mean()
                self.opt_pi.zero_grad(); loss_pi.backward(); self.opt_pi.step()
                loss_v = ((self.v(obs[ix]) - target[ix]) ** 2).mean()
                self.opt_v.zero_grad(); loss_v.backward(); self.opt_v.step()
                stats.append((loss_pi.item(), loss_v.item()))
        if not p['paper_buffer']:
            self.buf = []                                                 # on-policy: discard rollout
        return np.mean(stats, 0)
