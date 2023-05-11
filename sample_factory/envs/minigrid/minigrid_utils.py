import gym
# noinspection PyUnresolvedReferences
import gym_minigrid

from collections import deque
import numpy as np

from sample_factory.utils.utils import log, static_vars
from sample_factory.algorithms.utils.algo_utils import EXTRA_EPISODIC_STATS_PROCESSING, EXTRA_PER_POLICY_SUMMARIES, \
    EXTRA_EPISODIC_TEST_STATS_PROCESSING, EXTRA_TEST_SUMMARIES
from sample_factory.envs.env_wrappers import PixelFormatChwWrapper
from sample_factory.envs.minigrid.wrappers.minigrid_wrappers import RenameImageObsWrapper, MinigridPartialObsWrapper, \
    MiniGridRewardShapingWrapper, MinigridRecordingWrapper, RAW_SCORE_SUMMARY_KEY_SUFFIX
from sample_factory.envs.minigrid.minigrid_model import minigrid_register_models

MINIGRID_INITIALIZED = False

class MiniGridSpec:
    def __init__(self, name, env_id, default_timeout=None):
        self.name = name
        self.env_id = env_id
        self.level = env_id
        # self.default_timeout = default_timeout
        self.has_timer = False


MiniGrid_ENVS = []

MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-MultiRoom-N4-S5', 'MiniGrid-MultiRoom-N4-S5-v0'))
MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-MultiRoom-N6', 'MiniGrid-MultiRoom-N6-v0'))
MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-MultiRoom-N7-S8', 'MiniGrid-MultiRoom-N7-S8-v0'))
MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-MultiRoom-N12-S10', 'MiniGrid-MultiRoom-N12-S10-v0'))
MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-KeyCorridorS3R3', 'MiniGrid-KeyCorridorS3R3-v0'))
MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-KeyCorridorS4R3', 'MiniGrid-KeyCorridorS4R3-v0'))
MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-KeyCorridorS5R3', 'MiniGrid-KeyCorridorS5R3-v0'))
MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-KeyCorridorS6R3', 'MiniGrid-KeyCorridorS6R3-v0'))
MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-ObstructedMaze-1Dlhb', 'MiniGrid-ObstructedMaze-1Dlhb-v0'))
MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-ObstructedMaze-2Dlh', 'MiniGrid-ObstructedMaze-2Dlh-v0'))
MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-ObstructedMaze-2Dlhb', 'MiniGrid-ObstructedMaze-2Dlhb-v0'))
MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-ObstructedMaze-1Q', 'MiniGrid-ObstructedMaze-1Q-v0'))
MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-ObstructedMaze-2Q', 'MiniGrid-ObstructedMaze-2Q-v0'))
MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-ObstructedMaze-Full', 'MiniGrid-ObstructedMaze-Full-v0'))
MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-MultiRoomNoisyTV-N7-S4', 'MiniGrid-MultiRoomNoisyTV-N7-S4-v0'))
MiniGrid_ENVS.append(MiniGridSpec('MiniGrid-MultiRoomNoisyTV-N7-S4-hard', 'MiniGrid-MultiRoomNoisyTV-N7-S4-hard-v0'))


def minigrid_env_by_name(name):
    for cfg in MiniGrid_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown MiniGrid env')

# noinspection PyUnusedLocal
def make_minigrid_env(env_name, cfg, env_config, **kwargs):
    ensure_initialized(cfg, env_name)
    minigrid_spec = minigrid_env_by_name(env_name)

    num_envs = len([minigrid_spec.level])
    cfg.num_envs = num_envs

    task_id = get_task_id(env_config, minigrid_spec, cfg)
    level = task_id_to_level(task_id, minigrid_spec)

    env = gym_minigrid.wrappers.FullyObsWrapper(gym.make(level))
    env.level_name = level
    env.unwrapped.task_id = task_id

    env = RenameImageObsWrapper(env)

    if 'record_to' in cfg and cfg.record_to is not None:
        env = MinigridRecordingWrapper(env, cfg.record_to, cfg.record_fps, 0, cfg.random_action)

    env = MinigridPartialObsWrapper(env)

    if cfg.pixel_format == 'CHW':
        env = PixelFormatChwWrapper(env)

    env = MiniGridRewardShapingWrapper(env, task_id, level)

    return env

def get_task_id(env_config, spec, cfg):
    if env_config is None:
        return 0
    elif isinstance(spec.level, str):
        return 0
    elif isinstance(spec.level, (list, tuple)):
        num_envs = len(spec.level)

        if cfg.minigrid_one_task_per_worker:
            return env_config['worker_index'] % num_envs
        else:
            return env_config['env_id'] % num_envs
    else:
        raise Exception('spec level is either string or a list/tuple')


def task_id_to_level(task_id, spec):
    if isinstance(spec.level, str):
        return spec.level
    elif isinstance(spec.level, (list, tuple)):
        levels = spec.level
        level = levels[task_id]
        return level
    else:
        raise Exception('spec level is either string or a list/tuple')

def list_all_levels_for_experiment(env_name):
    spec = minigrid_env_by_name(env_name)
    if isinstance(spec.level, str):
        return [spec.level]
    elif isinstance(spec.level, (list, tuple)):
        levels = spec.level
        return levels
    else:
        raise Exception('spec level is either string or a list/tuple')

@static_vars(new_level_returns=dict(), env_spec=None)
def minigrid_extra_episodic_stats_processing(policy_id, stat_key, stat_value, cfg):
    if RAW_SCORE_SUMMARY_KEY_SUFFIX not in stat_key:
        return

    new_level_returns = minigrid_extra_episodic_stats_processing.new_level_returns
    if policy_id not in new_level_returns:
        new_level_returns[policy_id] = dict()

    if minigrid_extra_episodic_stats_processing.env_spec is None:
        minigrid_extra_episodic_stats_processing.env_spec = minigrid_env_by_name(cfg.env)

    task_id = int(stat_key.split('_')[1])  # this is a bit hacky but should do the job
    level = task_id_to_level(task_id, minigrid_extra_episodic_stats_processing.env_spec)
    #level_name = atari_level_to_level_name(level)
    level_name = level

    if level_name not in new_level_returns[policy_id]:
        new_level_returns[policy_id][level_name] = []

    new_level_returns[policy_id][level_name].append(stat_value)


@static_vars(all_levels=None)
def minigrid_extra_summaries(policy_id, policy_avg_stats, env_steps, summary_writer, cfg):
    """
    We precisely follow IMPALA repo (scalable_agent) here for the reward calculation.

    The procedure is:
    1. Calculate mean raw episode score for the last few episodes for each level
    2. Calculate human-normalized score using this mean value
    3. Calculate capped score

    The key point is that human-normalization and capping is done AFTER mean, which can lead to slighly higher capped
    scores for levels that exceed the human baseline.

    Another important point: we write the avg score summary only when we have at least one episode result for every
    level. Again, we try to precisely follow IMPALA implementation here.

    """
    new_level_returns = minigrid_extra_episodic_stats_processing.new_level_returns
    if policy_id not in new_level_returns:
        return

    # exit if we don't have at least one episode for all levels
    if minigrid_extra_summaries.all_levels is None:
        minigrid_levels = list_all_levels_for_experiment(cfg.env)
        level_names = minigrid_levels
        minigrid_extra_summaries.all_levels = level_names

    all_levels = minigrid_extra_summaries.all_levels
    for level in all_levels:
        # if len(new_level_returns[policy_id].get(level, [])) < 1:
        if len(new_level_returns[policy_id].get(level, [])) < 100:
            return

    # level_mean_scores = []
    mean_score = 0
    median_score = 0

    for level_idx, level in enumerate(all_levels):
        level_score = new_level_returns[policy_id][level]
        assert len(level_score) > 0
        assert level_idx == 0

        # score = np.mean(level_score)
        mean_score = np.mean(level_score)
        median_score = np.median(level_score)

        # level_mean_scores.append(score)

        level_key = f'{level_idx:02d}_{level}'
        # summary_writer.add_scalar(f'_minigrid/{level_key}_raw_score', score, env_steps)

    # assert len(level_mean_scores) == len(all_levels)

    # mean_score = np.mean(level_mean_scores)
    # median_score = np.median(level_mean_scores)

    # use 000 here to put these summaries on top in tensorboard (it sorts by ASCII)
    summary_writer.add_scalar(f'_minigrid/000_mean_raw_score', mean_score, env_steps)
    summary_writer.add_scalar(f'_minigrid/000_median_raw_score', median_score, env_steps)

    # clear the scores and start anew (this is exactly what IMPALA does)
    minigrid_extra_episodic_stats_processing.new_level_returns[policy_id] = dict()

    # add a new stat that PBT can track
    target_objective_stat = 'dmlab_target_objective'
    if target_objective_stat not in policy_avg_stats:
        policy_avg_stats[target_objective_stat] = [deque(maxlen=1) for _ in range(cfg.num_policies)]

    policy_avg_stats[target_objective_stat][policy_id].append(median_score)

@static_vars(new_level_returns=dict())
def minigrid_extra_test_stats_processing(policy_id, stat_key, stat_value, cfg):
    if RAW_SCORE_SUMMARY_KEY_SUFFIX not in stat_key:
        return

    new_level_returns = minigrid_extra_test_stats_processing.new_level_returns
    if policy_id not in new_level_returns:
        new_level_returns[policy_id] = dict()

    task_id = int(stat_key.split('_')[1])  # this is a bit hacky but should do the job
    level_name = 'random_character'

    if level_name not in new_level_returns[policy_id]:
        new_level_returns[policy_id][level_name] = []

    new_level_returns[policy_id][level_name].append(stat_value)
    return task_id


@static_vars(all_levels=None)
def minigrid_test_summaries(policy_id, policy_avg_stats, env_steps, summary_writer, cfg):
    new_level_returns = minigrid_extra_test_stats_processing.new_level_returns
    if policy_id not in new_level_returns:
        return

    # exit if we don't have at least one episode for all levels
    if minigrid_test_summaries.all_levels is None:
        level_names = ['random_character']
        minigrid_test_summaries.all_levels = level_names

    all_levels = minigrid_test_summaries.all_levels
    for level in all_levels:
        if len(new_level_returns[policy_id].get(level, [])) < 1:
            return

    for level_idx, level in enumerate(all_levels):
        level_score = new_level_returns[policy_id][level]
        assert len(level_score) > 0
        if len(level_score) < cfg.test_num_episodes:
            return

    for level_idx, level in enumerate(all_levels):
        level_score = new_level_returns[policy_id][level]
        level_score = np.array(level_score[:cfg.test_num_episodes])

        mean_score = np.mean(level_score)
        std_score = np.std(level_score)
        median_score = np.median(level_score)

        level_key = f'{level_idx:02d}_{level}'
        summary_writer.add_scalar(f'_minigrid/{level_key}_mean_score', mean_score, env_steps)
        summary_writer.add_scalar(f'_minigrid/{level_key}_std_score', std_score, env_steps)
        summary_writer.add_scalar(f'_minigrid/{level_key}_median_score', median_score, env_steps)
        log.debug('Policy %d %s mean_score: %f', cfg.test_policy_id[policy_id], level, mean_score)
        log.debug('Policy %d %s std_score: %f', cfg.test_policy_id[policy_id], level, std_score)
        log.debug('Policy %d %s median_score: %f', cfg.test_policy_id[policy_id], level, median_score)
        log.debug('Policy %d %s scores: %r', cfg.test_policy_id[policy_id], level, level_score)

    return policy_id


def ensure_initialized(cfg, env_name):
    global MINIGRID_INITIALIZED
    if MINIGRID_INITIALIZED:
        return

    minigrid_register_models()

    if 'MiniGrid' in env_name:
        EXTRA_EPISODIC_STATS_PROCESSING.append(minigrid_extra_episodic_stats_processing)
        EXTRA_PER_POLICY_SUMMARIES.append(minigrid_extra_summaries)

    if cfg.is_test:
        EXTRA_EPISODIC_TEST_STATS_PROCESSING.append(minigrid_extra_test_stats_processing)
        EXTRA_TEST_SUMMARIES.append(minigrid_test_summaries)

    MINIGRID_INITIALIZED = True
