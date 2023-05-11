import math

import numpy as np
import torch
from gym import spaces

from sample_factory.algorithms.appo.appo_utils import copy_dict_structure, iter_dicts_recursively, iterate_recursively
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.action_distributions import calc_num_logits, calc_num_actions
from sample_factory.utils.utils import log


def to_torch_dtype(numpy_dtype):
    """from_numpy automatically infers type, so we leverage that."""
    x = np.zeros([1], dtype=numpy_dtype)
    t = torch.from_numpy(x)
    return t.dtype


def to_numpy(t, num_dimensions):
    arr_shape = t.shape[:num_dimensions]
    arr = np.ndarray(arr_shape, dtype=object)
    to_numpy_func(t, arr)
    return arr


def to_numpy_func(t, arr):
    if len(arr.shape) == 1:
        for i in range(t.shape[0]):
            arr[i] = t[i]
    else:
        for i in range(t.shape[0]):
            to_numpy_func(t[i], arr[i])


def ensure_memory_shared(*tensors):
    """To prevent programming errors, ensure all tensors are in shared memory."""
    for tensor_dict in tensors:
        for _, _, t in iterate_recursively(tensor_dict):
            assert t.is_shared()


class PolicyOutput:
    def __init__(self, name, size):
        self.name = name
        self.size = size


class SharedBuffers:
    def __init__(self, cfg, num_agents, obs_space, action_space):
        self.cfg = cfg
        self.num_agents = num_agents
        self.envs_per_split = cfg.num_envs_per_worker // cfg.worker_num_splits
        self.num_traj_buffers = self.calc_num_trajectory_buffers()

        num_actions = calc_num_actions(action_space)
        num_action_logits = calc_num_logits(action_space)

        hidden_size = get_hidden_size(self.cfg)
        if self.cfg.use_transformer:
            hidden_size_transformer = cfg.hidden_size * (cfg.n_layer+1)
            if not cfg.match_core_input_size: # when cfg.match_core_input_size=True, it satisfies core_input_size=cfg.hidden_size
                if cfg.encoder_custom == 'dmlab_instructions':
                    hidden_size_transformer += 64 * (cfg.n_layer+1)
                if cfg.extended_input:
                    hidden_size_transformer += (num_action_logits + 1) * (cfg.n_layer + 1)
                if cfg.use_intrinsic and cfg.int_type == 'cell' and cfg.extended_input_cell:
                    hidden_size_transformer += cfg.cell_dim * (cfg.n_layer + 1)

            if not cfg.actor_critic_share_weights:
                hidden_size_transformer *= 2

        log.debug('Allocating shared memory for trajectories')
        self.tensors = TensorDict()

        # policy inputs
        obs_dict = TensorDict()
        self.tensors['obs'] = obs_dict
        if isinstance(obs_space, spaces.Dict):
            for space_name, space in obs_space.spaces.items():
                obs_dict[space_name] = self.init_tensor(space.dtype, space.shape)
        else:
            raise Exception('Only Dict observations spaces are supported')

        next_obs_dict = TensorDict()
        self.tensors['next_obs'] = next_obs_dict
        if isinstance(obs_space, spaces.Dict):
            for space_name, space in obs_space.spaces.items():
                next_obs_dict[space_name] = self.init_tensor(space.dtype, space.shape)

        # env outputs
        self.tensors['rewards'] = self.init_tensor(torch.float32, [1])
        self.tensors['dones'] = self.init_tensor(torch.bool, [1])
        self.tensors['raw_rewards'] = self.init_tensor(torch.float32, [1])

        self.tensors['prev_rewards'] = self.init_tensor(torch.float32, [1])
        self.tensors['prev_raw_rewards'] = self.init_tensor(torch.float32, [1])
        self.tensors['prev_actions'] = self.init_tensor(torch.int64, [1])
        self.tensors['episode_id'] = self.init_tensor(torch.int64, [1])
        self.tensors['episode_step'] = self.init_tensor(torch.int64, [1])

        # policy outputs
        if self.cfg.use_transformer:
            policy_outputs = [
                ('actions', num_actions),
                ('action_logits', num_action_logits),
                ('log_prob_actions', 1),
                ('values', 1),
                ('normalized_values', 1),
                ('policy_version', 1),
            ]
        else:
            policy_outputs = [
                ('actions', num_actions),
                ('action_logits', num_action_logits),
                ('log_prob_actions', 1),
                ('values', 1),
                ('normalized_values', 1),
                ('policy_version', 1),
                ('rnn_states', hidden_size)
            ]

        if cfg.use_intrinsic:
            policy_outputs += [('int_rewards', 1)]
            policy_outputs += [('int_values', 1)]
            if cfg.int_type in ['cell'] or cfg.int_type in ['bebold', 'agac', 'ride'] and cfg.use_episodic_cnt:
                policy_outputs += [('episodic_cnt', 1)]
                if cfg.extended_input_cell or cfg.is_test:
                    from sample_factory.algorithms.appo.modules.cell import CellSpec
                    cell_spec = CellSpec(cfg.cell_spec, type=cfg.cell_type)
                    if cfg.cell_type.startswith('ae'):
                        cell_size = cfg.hash_dim
                    else:
                        cell_size = np.prod(cell_spec.resolution)
                    policy_outputs += [('cell_ids', cell_size)]
            if cfg.int_type == 'agac':
                int_hidden_size = hidden_size // 2 if not cfg.actor_critic_share_weights else hidden_size
                policy_outputs += [('int_action_logits', num_action_logits),
                                   ('int_rnn_states', int_hidden_size)]

        policy_outputs = [PolicyOutput(*po) for po in policy_outputs]
        policy_outputs = sorted(policy_outputs, key=lambda policy_output: policy_output.name)

        for po in policy_outputs:
            self.tensors[po.name] = self.init_tensor(torch.float32, [po.size])

        ensure_memory_shared(self.tensors)

        # this is for performance optimization
        # indexing in numpy arrays is faster than in PyTorch tensors
        self.tensors_individual_transitions = self.tensor_dict_to_numpy(len(self.tensor_dimensions()))
        self.tensor_trajectories = self.tensor_dict_to_numpy(len(self.tensor_dimensions()) - 1)

        # create a shared tensor to indicate when the learner is done with the trajectory buffer and
        # it can be used to store the next trajectory
        traj_buffer_available_shape = [
            self.cfg.num_workers,
            self.cfg.worker_num_splits,
            self.envs_per_split,
            self.num_agents,
            self.num_traj_buffers,
        ]
        self.is_traj_tensor_available = torch.ones(traj_buffer_available_shape, dtype=torch.uint8)
        self.is_traj_tensor_available.share_memory_()
        self.is_traj_tensor_available = to_numpy(self.is_traj_tensor_available, 2)

        # copying small policy outputs (e.g. individual value predictions & action logits) to shared memory is a
        # bottleneck on the policy worker. For optimization purposes we create additional tensors to hold
        # just concatenated policy outputs. Rollout workers parse the data and add it to the trajectory buffers
        # in a proper format
        policy_outputs_combined_size = sum(po.size for po in policy_outputs)
        policy_outputs_shape = [
            self.cfg.num_workers,
            self.cfg.worker_num_splits,
            self.envs_per_split,
            self.num_agents,
            policy_outputs_combined_size,
        ]

        self.policy_outputs = policy_outputs
        self.policy_output_tensors = torch.zeros(policy_outputs_shape, dtype=torch.float32)
        self.policy_output_tensors.share_memory_()
        self.policy_output_tensors = to_numpy(self.policy_output_tensors, 4)

        self.policy_versions = torch.zeros([self.cfg.num_policies], dtype=torch.int32)
        self.policy_versions.share_memory_()

        # a list of boolean flags to be shared among components that indicate that experience collection should be
        # temporarily stopped (e.g. due to too much experience accumulated on the learner)
        self.stop_experience_collection = torch.ones([self.cfg.num_policies], dtype=torch.bool)
        self.stop_experience_collection.share_memory_()

        self.task_ids = torch.zeros([self.cfg.num_workers, self.cfg.worker_num_splits, self.envs_per_split], dtype=torch.uint8)
        self.task_ids.share_memory_()

        if self.cfg.use_transformer:

            if self.cfg.max_mems_buffer_len < 0:
                self.max_mems_buffer_len = self.cfg.mem_len + self.cfg.rollout * (self.num_traj_buffers + 1)
            else:
                self.max_mems_buffer_len = self.cfg.max_mems_buffer_len

            self.mems_dimensions = [self.cfg.num_workers, self.cfg.worker_num_splits, self.envs_per_split, self.num_agents, self.max_mems_buffer_len]
            self.mems_dimensions.append(hidden_size_transformer)

            self.mems_dones_dimensions = [self.cfg.num_workers, self.cfg.worker_num_splits, self.envs_per_split, self.num_agents, self.max_mems_buffer_len]
            self.mems_dones_dimensions.append(1)

    def calc_num_trajectory_buffers(self):
        # calculate how many buffers are required per env runner to collect one "macro batch" for training
        # once macro batch is collected, all buffers will be released
        # we could have just copied the tensors on the learner to avoid this complicated logic, but it's better for
        # performance to keep data in shared buffers until they're needed
        samples_per_iteration = self.cfg.num_batches_per_iteration * self.cfg.batch_size * self.cfg.num_policies
        num_traj_buffers = samples_per_iteration / (self.cfg.num_workers * self.cfg.num_envs_per_worker * self.num_agents * self.cfg.rollout)

        # make sure we definitely have enough buffers to actually never wait
        # usually it'll be just two buffers and we swap back and forth
        num_traj_buffers *= 3

        # make sure we have at least two to swap between so we never actually have to wait
        num_traj_buffers = math.ceil(max(num_traj_buffers, self.cfg.min_traj_buffers_per_worker))
        log.info('Using %d sets of trajectory buffers', num_traj_buffers)
        return num_traj_buffers

    def init_tensor(self, tensor_type, tensor_shape):
        if not isinstance(tensor_type, torch.dtype):
            tensor_type = to_torch_dtype(tensor_type)

        dimensions = self.tensor_dimensions()
        final_shape = dimensions + list(tensor_shape)
        t = torch.zeros(final_shape, dtype=tensor_type)
        t.share_memory_()
        return t

    def tensor_dimensions(self):
        dimensions = [
            self.cfg.num_workers,
            self.cfg.worker_num_splits,
            self.envs_per_split,
            self.num_agents,
            self.num_traj_buffers,
            self.cfg.rollout,
        ]
        return dimensions

    def tensor_dict_to_numpy(self, num_dimensions):
        numpy_dict = copy_dict_structure(self.tensors)
        for d1, d2, key, curr_t, value2 in iter_dicts_recursively(self.tensors, numpy_dict):
            assert isinstance(curr_t, torch.Tensor)
            assert value2 is None
            d2[key] = to_numpy(curr_t, num_dimensions)
            assert isinstance(d2[key], np.ndarray)
        return numpy_dict


class TensorDict(dict):
    def index(self, indices):
        return self.index_func(self, indices)

    def index_func(self, x, indices):
        if isinstance(x, (dict, TensorDict)):
            res = TensorDict()
            for key, value in x.items():
                res[key] = self.index_func(value, indices)
            return res
        else:
            t = x[indices]
            return t

    def set_data(self, index, new_data):
        self.set_data_func(self, index, new_data)

    def set_data_func(self, x, index, new_data):
        if isinstance(new_data, (dict, TensorDict)):
            for new_data_key, new_data_value in new_data.items():
                self.set_data_func(x[new_data_key], index, new_data_value)
        elif isinstance(new_data, torch.Tensor):
            x[index].copy_(new_data)
        elif isinstance(new_data, np.ndarray):
            t = torch.from_numpy(new_data)
            x[index].copy_(t)
        else:
            raise Exception(f'Type {type(new_data)} not supported in set_data_func')
