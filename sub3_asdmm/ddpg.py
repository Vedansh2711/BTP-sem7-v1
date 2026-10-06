"""DDPG baseline: actor emits [0,1] scores per head, Gaussian exploration noise, masked argmax decoding."""
import numpy as np
import torch, torch.nn as nn

NEG = -1e9


def mlp(i, hs, o, act=nn.ReLU, out_act=None):
    L, d = [], i
    for h in hs:
        L += [nn.Linear(d, h), act()]; d = h
    L.append(nn.Linear(d, o))
    if out_act: L.append(out_act())
    return nn.Sequential(*L)


class DDPGAgent:
    name = 'ddpg'

    def __init__(self, env, cfg, seed=0):
        torch.manual_seed(seed)
        self.N, self.K, self.p = env.N, env.K, cfg['ddpg']
        self.adim = self.N * (2 * self.K + 1)
        self.actor = mlp(env.obs_dim, [256, 128, 128], self.adim, nn.ReLU, nn.Sigmoid)
        self.critic = mlp(env.obs_dim + self.adim, [256, 128, 128], 1)
        self.actor_t, self.critic_t = mlp(env.obs_dim, [256, 128, 128], self.adim, nn.ReLU, nn.Sigmoid), mlp(env.obs_dim + self.adim, [256, 128, 128], 1)
        self.actor_t.load_state_dict(self.actor.state_dict()); self.critic_t.load_state_dict(self.critic.state_dict())
        self.oa = torch.optim.Adam(self.actor.parameters(), lr=self.p['lr']); self.oc = torch.optim.Adam(self.critic.parameters(), lr=self.p['lr'])
        self.cap, self.rng = self.p['buffer'], np.random.default_rng(seed)
        self.S, self.A, self.R, self.S2, self.D = [], [], [], [], []
        self.ptr = 0
        env.predictive = self.p['predictive']            # ddpg_predictive toggle

    def _decode(self, a, masks):
        a = a.reshape(self.N, 2 * self.K + 1)
        sx = np.where(masks['x_mask'], a[:, :self.K], -np.inf); sy = np.where(masks['y_mask'], a[:, self.K:], -np.inf)
        x = np.where(masks['x_mask'].any(1), sx.argmax(1), 0)
        return dict(x=x, y=sy.argmax(1))

    @torch.no_grad()
    def act(self, obs, masks, explore=True):
        a = self.actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        if explore:
            a = np.clip(a + self.rng.normal(0, self.p['noise_std'], a.shape), 0, 1)
        return self._decode(a, masks), dict(a=a.astype(np.float32))

    def observe(self, obs, aux, action, r, obs2, done):
        item = (obs, aux['a'], r, obs2, float(done))
        if len(self.S) < self.cap:
            for L, v in zip((self.S, self.A, self.R, self.S2, self.D), item): L.append(v)
        else:
            for L, v in zip((self.S, self.A, self.R, self.S2, self.D), item): L[self.ptr] = v
            self.ptr = (self.ptr + 1) % self.cap

    def update(self, n_updates=150):          # ~one gradient step per env step (150 slots/episode)
        if len(self.S) < self.p['batch']:
            return np.zeros(2)
        out = []
        for _ in range(n_updates):
            ix = self.rng.integers(len(self.S), size=self.p['batch'])
            f = lambda L: torch.as_tensor(np.stack([L[i] for i in ix]), dtype=torch.float32)
            s, a, r, s2, d = f(self.S), f(self.A), f(self.R), f(self.S2), f(self.D)
            with torch.no_grad():
                y = r + self.p['gamma'] * (1 - d) * self.critic_t(torch.cat([s2, self.actor_t(s2)], -1)).squeeze(-1)
            lc = ((self.critic(torch.cat([s, a], -1)).squeeze(-1) - y) ** 2).mean()
            self.oc.zero_grad(); lc.backward(); self.oc.step()
            la = -self.critic(torch.cat([s, self.actor(s)], -1)).mean()
            self.oa.zero_grad(); la.backward(); self.oa.step()
            with torch.no_grad():
                for n, nt in ((self.actor, self.actor_t), (self.critic, self.critic_t)):
                    for p, pt in zip(n.parameters(), nt.parameters()):
                        pt.mul_(1 - self.p['tau_soft']).add_(self.p['tau_soft'] * p)
            out.append((lc.item(), la.item()))
        return np.mean(out, 0)
