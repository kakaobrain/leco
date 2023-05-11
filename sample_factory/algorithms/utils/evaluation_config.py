import os
from os.path import join

from sample_factory.utils.utils import str2bool


def add_eval_args(parser):
    parser.add_argument('--fps', default=0, type=int, help='Enable sync mode with adjustable FPS. Default (0) means default, e.g. for Doom its FPS (~35), or unlimited if not specified by env. Leave at 0 for Doom multiplayer evaluation')
    parser.add_argument('--render_action_repeat', default=None, type=int, help='Repeat an action that many frames during evaluation. By default uses the value from env config (used during training).')
    parser.add_argument('--no_render', action='store_true', help='Do not render the environment during evaluation')
    parser.add_argument('--policy_index', default=0, type=int, help='Policy to evaluate in case of multi-policy training')
    parser.add_argument('--record_to', default=join(os.getcwd(), '..', 'recs'), type=str, help='Record episodes to this folder. Only used for VizDoom!')

    parser.add_argument('--continuous_actions_sample', default=True, type=str2bool, help='True to sample from a continuous action distribution at test time, False to just take the mean')

def add_test_args(parser):
    parser.add_argument('--test_action_sample', default='argmax', choices=['argmax', 'sample'], type=str, help='Action sampling method')
    parser.add_argument('--test_num_episodes', default=10, type=int, help='The number of episodes to evaluate')
    # parser.add_argument('--test_policy_id', default=[0], nargs='+', type=int, help='Policy id to test. Set -1 to test all policies')
    parser.add_argument('--test_policy_id', default=0, type=int, help='Policy id to test')
    parser.add_argument('--test_dir', default='test', type=str, help='Postfix for test summary dir')
    parser.add_argument('--record_to', default=None, type=str, help='Record episodes to this folder')
    #parser.add_argument('--time_limit', default=False, type=str2bool, help='limit the max episode step')
    #parser.add_argument('--multitask', default=False, type=str2bool, help='Whether to use multitask training or not')
    parser.add_argument('--record_res_h', default=72, type=int, help='Game frame height when recording')
    parser.add_argument('--record_res_w', default=96, type=int, help='Game frame width when recording')

    # Test summary dir is $train_dir/$experiment/$test_dir/.summary

def overwrite_test_cfg(cfg):
    # cfg.num_workers = cfg.test_num_levels
    cfg.test_policy_id = [cfg.test_policy_id]
    cfg.num_policies = len(cfg.test_policy_id)
    # cfg.num_envs_per_worker = 1
    cfg.worker_num_splits = 1
    cfg.with_pbt = False
    cfg.decorrelate_experience_max_seconds = 0
    cfg.decorrelate_envs_on_one_worker = False
    cfg.resume = True
    cfg.set_workers_cpu_affinity = False

    # enforce single node single gpu
    cfg.nproc_per_node = 1
    cfg.local_rank = 0
    os.environ['WORLD_SIZE'] = str(1)

    if hasattr(cfg, 'dmlab_one_task_per_worker'):
        cfg.dmlab_one_task_per_worker = True
    if hasattr(cfg, 'atari_one_task_per_worker'):
        cfg.atari_one_task_per_worker = True
    # if hasattr(cfg, 'episode_life'):
    #     cfg.episode_life = False
    if cfg.record_to is not None:
        cfg.worker_num_splits = 1
        cfg.num_envs_per_worker = 1
    if 'atari_' in cfg.env:
        cfg.record_fps = 30
    if 'dmlab_' in cfg.env:
        cfg.record_fps = 15
    if 'MiniGrid' in cfg.env:
        cfg.record_fps = 10
    return cfg
