import gym
import gym_minigrid
from gym.spaces import Box
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers.transform_observation import TransformObservation
from sample_factory.envs.env_wrappers import RecordingWrapper
import numpy as np
from typing import List, Tuple
import copy
import cv2
import os

RAW_SCORE_SUMMARY_KEY_SUFFIX = 'minigrid_raw_score'

def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env

class RenameImageObsWrapper(gym.ObservationWrapper):
    """We call the main observation just 'obs' in all algorithms."""

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = env.observation_space
        self.observation_space.spaces['obs'] = self.observation_space.spaces['image']
        self.observation_space.spaces.pop('image')

    def observation(self, observation):
        observation['obs'] = observation['image']
        observation.pop('image')
        observation.pop('mission')
        return observation

class MinigridPartialObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(MinigridPartialObsWrapper, self).__init__(env)
        partial_obs = self.env.gen_obs()['image']
        img_obs_space = self.observation_space.spaces['obs']
        new_img_obs_space = gym.spaces.Box(low=partial_obs.min(), high=partial_obs.max(), shape=partial_obs.shape, dtype=partial_obs.dtype)
        self.observation_space.spaces['obs'] = new_img_obs_space
        self.observation_space.spaces['full_obs'] = img_obs_space
        self.observation_space.spaces['agent_pos'] = gym.spaces.Box(low=0, high=max(img_obs_space.shape[0:2]),
                                                                    shape=env.agent_pos.shape,
                                                                    dtype=env.agent_pos.dtype)

    def observation(self, observation):
        observation['full_obs'] = observation['obs']
        observation['obs'] = self.env.gen_obs()['image']
        observation['agent_pos'] = self.env.agent_pos
        return observation

class MiniGridRewardShapingWrapper(gym.Wrapper):
    def __init__(self, env, task_id, level):
        super().__init__(env)
        self.raw_episode_return = self.episode_length = 0
        self.task_id = task_id
        self.level = level

    def reset(self):
        obs = self.env.reset()
        self.raw_episode_return = self.episode_length = 0
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.raw_episode_return += rew

        raw_rew = rew

        # # reward clipping
        #rew = np.clip(rew, 0, rew + 1)

        self.episode_length += info.get('num_frames', 1)

        if done:
            score = self.raw_episode_return
            if 'episode_extra_stats' not in info:
                info['episode_extra_stats'] = dict()
            level_name = self.level

            # add extra 'z_' to the summary key to put them towards the end on tensorboard (just convenience)
            level_name_key = f'z_{self.task_id:02d}_{level_name}'
            info['episode_extra_stats'][f'{level_name_key}_{RAW_SCORE_SUMMARY_KEY_SUFFIX}'] = score
            info['episode_extra_stats'][f'{level_name_key}_len'] = self.episode_length

        return obs, rew, done, info, raw_rew


class DiscreteGrid:
    def __init__(self, env: gym_minigrid.minigrid.MiniGridEnv):
        self._env = env
        self._tile_size = 20
        self._visit_weight = 40
        self._agent_positions = []
        self._actions = []
        self._grid = None
        self._allow_actions_for_log = [env.actions.pickup, env.actions.drop, env.actions.toggle]

    def add(self, action):
        if self._grid is None:
            self._grid = self._env.grid.render(self._tile_size)
        self._agent_positions.append(self._env.agent_pos)
        self._actions.append(action)

    def reset(self):
        self._grid = None
        self._grid_act = None
        self._agent_positions = []
        self._actions = []

    def _grid_coverage(
        self, grid: np.ndarray, positions: List[Tuple[int]], actions=None
    ) -> np.ndarray:
        """
        Returns an image representing the grid coverage over 1 episode.
        """
        img = grid
        for i, pos in enumerate(positions):
            add_weight = True
            if actions is not None:
                act = actions[i]
                if not act in self._allow_actions_for_log:
                    add_weight = False
            
            if add_weight:
                img[
                    pos[1] * self._tile_size : (pos[1] + 1) * self._tile_size,
                    pos[0] * self._tile_size : (pos[0] + 1) * self._tile_size,
                    2,
                ] += self._visit_weight

        img[img > 255] = 255
        return img
        # return img.transpose(-1, 0, 1)

    def logs(self) -> List:
        if self._grid is None:
            return None, None
        pos_grid = self._grid_coverage(copy.deepcopy(self._grid), self._agent_positions)
        act_grid = self._grid_coverage(copy.deepcopy(self._grid), self._agent_positions, self._actions)
        return pos_grid, act_grid


class MinigridRecordingWrapper(RecordingWrapper):
    def __init__(self, env, record_to, record_fps, player_id, random_action):
        super().__init__(env, record_to, record_fps, player_id, random_action, env.width, env.height)
        self.grid = DiscreteGrid(env)

    # noinspection PyMethodOverriding
    #def render(self, mode, **kwargs):
    #    self.env.render()
    #    frame = self.env.render('rgb_array')
    #    self._record(frame)
    #    return frame
    def reset(self):
        if self._episode_recording_dir is not None and self._record_id > 0:
            reward = self._recorded_episode_reward + self._recorded_episode_shaping_reward
            new_dir_name = self._episode_recording_dir + f'_r{reward:.2f}'
            pos_grid, act_grid = self.grid.logs()
            if pos_grid is not None:
                cv2.imwrite(f'{new_dir_name}_pos_grid.png', pos_grid)
            if act_grid is not None:
                cv2.imwrite(f'{new_dir_name}_act_grid.png', act_grid)

        result = super().reset()
        self.grid.reset()
        return result


    def step(self, action):
        if self._random_action:
            action = np.random.choice(self.env.action_space.n)
        self.grid.add(action)
        observation, reward, done, info = self.env.step(action)

        if isinstance(action, np.ndarray):
            self._recorded_actions.append(action.tolist())
        elif isinstance(action, np.int64):
            self._recorded_actions.append(int(action))
        else:
            self._recorded_actions.append(action)

        if isinstance(reward, np.ndarray):
            self._recorded_rewards.append(reward.tolist())
        else:
            self._recorded_rewards.append(reward)
        self._record(self.env.render('rgb_array'))
        # self._record(self.env.gen_obs()['image'])
        self._recorded_episode_reward += reward
        return observation, reward, done, info