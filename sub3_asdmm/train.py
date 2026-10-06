"""Shared train/eval loops for all agents (PPO, DDPG, MDA)."""
import numpy as np
from sub3_asdmm.mda import mda_action


def run_episode(env, agent, train=True, seed=None):
    obs = env.reset(seed)
    done, R, comm, comp, md, me = False, [], [], [], 0.0, 0
    while not done:
        masks = env.action_masks()
        if agent.name == 'mda':
            action, aux = mda_action(env), {}
        else:
            action, aux = agent.act(obs, masks, explore=train)
        obs2, r, done, info = env.step(action)
        if train and agent.name != 'mda':
            agent.observe(obs, aux, action, r, obs2, done)
        R.append(r); comm.append(info['comm']); comp.append(info['comp']); md += info['mig_delay']; me += info['mig_events']
        obs = obs2
    return dict(reward=float(np.mean(R)), amwaoi=info['amwaoi'], comm=float(np.mean(comm)), comp=float(np.mean(comp)),
                mig_delay=md / max(me, 1), mig_events=me)


def train(env, agent, epochs, log_every=10, verbose=True):
    hist = []
    for ep in range(epochs):
        m = run_episode(env, agent, train=True)
        if agent.name != 'mda':
            agent.update()
        hist.append(m)
        if verbose and (ep + 1) % log_every == 0:
            print('[%s] ep %d  reward %.2f  AMWAoI %.2f' % (agent.name, ep + 1, np.mean([h['reward'] for h in hist[-log_every:]]),
                                                         np.mean([h['amwaoi'] for h in hist[-log_every:]])), flush=True)
    return hist


def evaluate(env, agent, episodes=10, seed0=10_000):
    ms = [run_episode(env, agent, train=False, seed=seed0 + i) for i in range(episodes)]
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
