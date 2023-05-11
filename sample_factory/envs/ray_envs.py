from filelock import FileLock, Timeout
from ray.tune import register_env

from sample_factory.algorithms.utils.arguments import default_cfg
from sample_factory.envs.dmlab.dmlab_env import DMLAB_ENVS, make_dmlab_env


def register_dmlab_envs_rllib(**kwargs):
    for spec in DMLAB_ENVS:
        def make_env_func(env_config):
            print('Creating env!!!')
            cfg = default_cfg(env=spec.name)
            cfg.pixel_format = 'HWC'  # tensorflow models expect HWC by default

            if 'res_w' in env_config:
                cfg.res_w = env_config['res_w']
            if 'res_h' in env_config:
                cfg.res_h = env_config['res_h']
            if 'renderer' in env_config:
                cfg.renderer = env_config['renderer']
            if 'dmlab_throughput_benchmark' in env_config:
                cfg.renderer = env_config['dmlab_throughput_benchmark']

            env = make_dmlab_env(spec.name, env_config=env_config, cfg=cfg, **kwargs)
            return env

        register_env(spec.name, make_env_func)
