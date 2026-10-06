import copy, json, os
import yaml


def load(path='config.yaml'):
    cfg = yaml.safe_load(open(path))
    if os.environ.get('DTMIG_BACKEND'):                 # e.g. DTMIG_BACKEND=sumo
        cfg['mobility']['backend'] = os.environ['DTMIG_BACKEND']
    return cfg


def make_agent(name, env, cfg, seed=0):
    """name in {proposed, ddpg, mda, ppo_sm, ddpg_sm, mda_sm}. *_sm swaps WSMM for SM in Subproblem 1."""
    base = name.replace('proposed', 'ppo')
    sub1 = 'sm' if base.endswith('_sm') else cfg['perception']['sub1']
    env.sub1 = sub1
    kind = base.replace('_sm', '')
    if kind == 'ppo':
        from sub3_asdmm.ppo import PPOAgent; return PPOAgent(env, cfg, seed)
    if kind == 'ddpg':
        from sub3_asdmm.ddpg import DDPGAgent; return DDPGAgent(env, cfg, seed)
    from sub3_asdmm.mda import MDAAgent; return MDAAgent(env, cfg, seed)


def dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, 'w'), indent=1)
