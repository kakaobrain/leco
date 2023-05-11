import multiprocessing
import signal
import time
from collections import deque
from queue import Empty

import numpy as np
import psutil
import copy
import torch
from torch.multiprocessing import Process as TorchProcess

from sample_factory.algorithms.appo.appo_utils import TaskType, memory_stats, cuda_envvars_for_policy
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import AttrDict, log, join_or_kill
from sample_factory.algorithms.appo.ema import ExponentialMovingAverage


def dict_of_lists_append(dict_of_lists, new_data, index):
    for key, x in new_data.items():
        if key in dict_of_lists:
            dict_of_lists[key].append(x[index])
        else:
            dict_of_lists[key] = [x[index]]


def dict_of_lists_dummy_append(dict_of_lists, new_data, index):
    _index = (0,)
    for _ in index[:-1]:
        _index += (0,)
    for key, x in new_data.items():
        if key in dict_of_lists:
            dict_of_lists[key].append(torch.zeros_like(x[_index]))
        else:
            dict_of_lists[key] = [torch.zeros_like(x[_index])]


class PolicyWorker:
    def __init__(
        self, worker_idx, policy_id, cfg, obs_space, action_space, shared_buffers, policy_queue, actor_queues,
        report_queue, task_queue, policy_lock, resume_experience_collection_cv
    ):
        log.info('Initializing policy worker %d for policy %d', worker_idx, policy_id)

        self.worker_idx = worker_idx
        self.policy_id = policy_id
        self.cfg = cfg

        self.obs_space = obs_space
        self.action_space = action_space

        self.device = None
        self.actor_critic = None
        self.int_module = None
        self.shared_model_weights = None
        self.policy_lock = policy_lock
        self.resume_experience_collection_cv = resume_experience_collection_cv

        self.policy_queue = policy_queue
        self.actor_queues = actor_queues
        self.report_queue = report_queue

        # queue other components use to talk to this particular worker
        self.task_queue = task_queue

        self.initialized = False
        self.terminate = False
        self.initialized_event = multiprocessing.Event()
        self.initialized_event.clear()

        self.shared_buffers = shared_buffers
        self.num_traj_buffers = self.shared_buffers.num_traj_buffers
        self.tensors_individual_transitions = self.shared_buffers.tensors_individual_transitions
        self.policy_versions = shared_buffers.policy_versions
        self.stop_experience_collection = shared_buffers.stop_experience_collection

        self.latest_policy_version = -1000
        self.num_policy_updates = 0

        self.requests = []

        self.total_num_samples = 0

        if self.cfg.use_transformer:
            max_batch_size = self.cfg.num_workers * self.cfg.num_envs_per_worker
            mem_T_dim = self.cfg.mem_len
            mem_D_dim = self.shared_buffers.mems_dimensions[-1]
            mem_dones_D_dim = 1
            self.mems = torch.zeros([max_batch_size, mem_T_dim, mem_D_dim]).float()
            self.mems_dones = torch.zeros([max_batch_size, mem_T_dim, mem_dones_D_dim]).float()

            self.max_mems_buffer_len = shared_buffers.max_mems_buffer_len

        self.process = TorchProcess(target=self._run, daemon=True)

    def start_process(self):
        self.process.start()

    def _init(self):
        log.info('Policy worker %d-%d initialized', self.policy_id, self.worker_idx)
        self.initialized = True
        self.initialized_event.set()

    def _handle_policy_steps(self, timing):
        with torch.no_grad():
            with timing.timeit('deserialize'):
                observations = AttrDict()
                next_observations = AttrDict() if self.cfg.use_intrinsic else None
                prev_observations = None
                if self.cfg.use_intrinsic and \
                   (self.cfg.is_test or (self.cfg.int_type == 'cell' and self.cfg.extended_input_cell)):
                    prev_observations = AttrDict()

                rnn_states = []
                int_rnn_states = None
                if self.cfg.use_intrinsic:
                    if self.cfg.int_type == 'agac':
                        int_rnn_states = []
                first_rollout_list = []
                actions = []
                rewards = []
                dones = []
                dones_rollout = []
                traj_tensors = self.shared_buffers.tensors_individual_transitions
                task_ids = []
                rollout_step_list = []
                actor_env_step_list = []
                episodic_stats = []

                r_idx = 0
                for request in self.requests:
                    actor_idx, split_idx, request_data = request

                    if self.cfg.use_intrinsic:
                        for data in request_data:
                            if self.cfg.use_transformer:
                                env_idx, agent_idx, traj_buffer_idx, rollout_step, _, _ = data
                            else:
                                env_idx, agent_idx, traj_buffer_idx, rollout_step = data

                            index = actor_idx, split_idx, env_idx, agent_idx, traj_buffer_idx, rollout_step
                            stat = {
                                'index': (actor_idx, split_idx, agent_idx, env_idx),
                                'episode_id': traj_tensors['episode_id'][index].item(),
                                'task_id': self.shared_buffers.task_ids[actor_idx][split_idx][env_idx].unsqueeze(0).item(),
                                'done': traj_tensors['dones'][index].item(),
                                'score': traj_tensors['raw_rewards'][index].item(),
                                'episode_step': traj_tensors['episode_step'][index].item()
                            }
                            # if stat['score'] > 0:
                            #     print(stat)
                            episodic_stats.append(stat)

                            if prev_observations is not None:
                                if (rollout_step > 0 or traj_buffer_idx > 0):
                                    # get prev observations
                                    # FIXME: this code assumes that there is enough size of traj_buffer (after an episode ended, traj_buffer_idx is set to 0).
                                    # FIXME: the assumption may not be true in some environments. thus episode_id should be considered.
                                    if rollout_step == 0 and traj_buffer_idx > 0:
                                        _traj_buffer_idx = traj_buffer_idx - 1
                                        _rollout_step = self.cfg.rollout - 1
                                    else:
                                        _traj_buffer_idx = traj_buffer_idx
                                        _rollout_step = rollout_step - 1
                                    prev_index = actor_idx, split_idx, env_idx, agent_idx, _traj_buffer_idx, _rollout_step
                                    dict_of_lists_append(prev_observations, traj_tensors['obs'], prev_index)
                                else:
                                    # append dummy data
                                    dict_of_lists_dummy_append(prev_observations, traj_tensors['obs'], index)
                            else:
                                if next_observations is not None:
                                    dict_of_lists_append(next_observations, traj_tensors['next_obs'], index)

                    if self.cfg.use_transformer:
                        for env_idx, agent_idx, traj_buffer_idx, rollout_step, first_rollout, actor_env_step in request_data:
                            index = actor_idx, split_idx, env_idx, agent_idx, traj_buffer_idx, rollout_step
                            with timing.timeit("write_done_on_mems_dones_buffer"):
                                self._write_done_on_mems_dones_buffer(index, actor_env_step)
                            timing["write_done_on_mems_dones_buffer"] *= len(self.requests) * len(request_data)
                            dict_of_lists_append(observations, traj_tensors['obs'], index)

                            # read memory
                            s_idx = (actor_env_step - self.cfg.mem_len) % self.max_mems_buffer_len
                            e_idx = actor_env_step % self.max_mems_buffer_len
                            with timing.timeit("mems_copy"):
                                if s_idx > e_idx:
                                    self.mems[r_idx] = torch.cat([self.mems_buffer[actor_idx, split_idx, env_idx, agent_idx, s_idx:],
                                                                  self.mems_buffer[actor_idx, split_idx, env_idx, agent_idx, :e_idx]])
                                    self.mems_dones[r_idx] = torch.cat([self.mems_dones_buffer[actor_idx, split_idx, env_idx, agent_idx, s_idx:],
                                                                        self.mems_dones_buffer[actor_idx, split_idx, env_idx, agent_idx, :e_idx]])
                                else:
                                    self.mems[r_idx] = self.mems_buffer[actor_idx, split_idx, env_idx, agent_idx, s_idx:e_idx]
                                    self.mems_dones[r_idx] = self.mems_dones_buffer[actor_idx, split_idx, env_idx, agent_idx, s_idx:e_idx]
                            timing["mems_copy"] *= len(self.requests) * len(request_data)
                            r_idx += 1
                            first_rollout_list.append(first_rollout)

                            # index handling for reading done of previous step
                            actions.append(traj_tensors['prev_actions'][index])
                            rewards.append(traj_tensors['prev_rewards'][index])
                            dones.append(traj_tensors['dones'][index])
                            task_ids.append(self.shared_buffers.task_ids[actor_idx][split_idx][env_idx].unsqueeze(0))
                            rollout_step_list.append(rollout_step)
                            actor_env_step_list.append(actor_env_step)
                            self.total_num_samples += 1
                    else:
                        for env_idx, agent_idx, traj_buffer_idx, rollout_step in request_data:
                            index = actor_idx, split_idx, env_idx, agent_idx, traj_buffer_idx, rollout_step
                            dict_of_lists_append(observations, traj_tensors['obs'], index)
                            rnn_states.append(traj_tensors['rnn_states'][index])
                            if self.cfg.use_intrinsic and self.cfg.int_type == 'agac':
                                int_rnn_states.append(traj_tensors['int_rnn_states'][index])
                            actions.append(traj_tensors['prev_actions'][index])
                            rewards.append(traj_tensors['prev_rewards'][index])
                            dones.append(traj_tensors['dones'][index])
                            task_ids.append(self.shared_buffers.task_ids[actor_idx][split_idx][env_idx].unsqueeze(0))
                            self.total_num_samples += 1

            if self.cfg.use_transformer:
                with timing.timeit('reordering_mems'):
                    n_batch = len(actor_env_step_list)
                    if self.cfg.mem_len > 0:
                        mems_dones = self.mems_dones[:n_batch]
                        actor_env_step = torch.tensor(actor_env_step_list) # (n_batch)
                        mem_begin_index = self.actor_critic.cores[0].get_mem_begin_index(mems_dones, actor_env_step)
                        self.actor_critic.actor_env_step = actor_env_step
                    else:
                        mem_begin_index = [0] * n_batch

            with timing.timeit('stack'):
                for key, x in observations.items():
                    observations[key] = torch.stack(x)
                if next_observations is not None:
                    for key, x in next_observations.items():
                        next_observations[key] = torch.stack(x)
                if prev_observations is not None:
                    for key, x in prev_observations.items():
                        prev_observations[key] = torch.stack(x)
                actions = torch.stack(actions)
                rewards = torch.stack(rewards)
                task_ids = torch.stack(task_ids)
                if not self.cfg.use_transformer:
                    rnn_states = torch.stack(rnn_states)
                    if self.cfg.use_intrinsic and self.cfg.int_type == 'agac':
                        int_rnn_states = torch.stack(int_rnn_states)
                    num_samples = rnn_states.shape[0]
                dones = torch.stack(dones)

            with timing.timeit('obs_to_device'):
                for key, x in observations.items():
                    device, dtype = self.actor_critic.device_and_type_for_input_tensor(key)
                    observations[key] = x.to(device).type(dtype)
                if next_observations is not None:
                    for key, x in next_observations.items():
                        device, dtype = self.actor_critic.device_and_type_for_input_tensor(key)
                        next_observations[key] = x.to(device).type(dtype)
                if prev_observations is not None:
                    for key, x in prev_observations.items():
                        device, dtype = self.actor_critic.device_and_type_for_input_tensor(key)
                        prev_observations[key] = x.to(device).type(dtype)

                actions = actions.to(self.device).long()
                rewards = rewards.to(self.device).float()
                task_ids = task_ids.to(self.device).long()

                if not self.cfg.use_transformer:
                    rnn_states = rnn_states.to(self.device).float()
                    if self.cfg.use_intrinsic and self.cfg.int_type == 'agac':
                        int_rnn_states = int_rnn_states.to(self.device).float()
                    else:
                        int_rnn_states = []
                    dones = dones.to(self.device).float()

            if self.cfg.use_transformer:
                with timing.timeit('mems_slice'):
                    # slice
                    num_samples = actions.shape[0]
                    mems = self.mems[:num_samples]

            with timing.timeit('forward'):
                with_action_distribution = self.cfg.is_test and self.cfg.test_action_sample == 'argmax'
                if self.cfg.use_half_policy_worker:
                    for key in observations:
                        obs = observations[key]
                        if obs.dtype == torch.float32:
                            observations[key] = obs.half()
                    if self.cfg.use_transformer:
                        mems = mems.half()
                    else:
                        rnn_states = rnn_states.half()
                    rewards = rewards.half()

                if self.cfg.use_ema:
                    if self.cfg.use_transformer:
                        policy_outputs = self.actor_critic_ema.module(observations, actions.squeeze(1), rewards.squeeze(1),
                                                       mems=mems.transpose(0, 1), rollout_step_list=rollout_step_list, mem_begin_index=mem_begin_index,
                                                       task_ids=task_ids, is_seq=False,
                                                       with_action_distribution=with_action_distribution, is_transformer=True)
                    else:
                        policy_outputs = self.actor_critic_ema.module(observations, actions.squeeze(1), rewards.squeeze(1),
                                                           rnn_states=rnn_states, dones=dones.squeeze(1), task_ids=task_ids, is_seq=False,
                                                           with_action_distribution=with_action_distribution)
                else:
                    if self.int_module is None:
                        if self.cfg.use_transformer:
                            policy_outputs = self.actor_critic(observations, actions.squeeze(1), rewards.squeeze(1),
                                                           mems=mems.transpose(0, 1), rollout_step_list=rollout_step_list, mem_begin_index=mem_begin_index,
                                                           task_ids=task_ids, is_seq=False,
                                                           with_action_distribution=with_action_distribution, is_transformer=True)
                        else:
                            policy_outputs = self.actor_critic(observations, actions.squeeze(1), rewards.squeeze(1),
                                                               rnn_states=rnn_states, dones=dones.squeeze(1), task_ids=task_ids, is_seq=False,
                                                               with_action_distribution=with_action_distribution)
                    else:
                        # instr = self.actor_critic.encoder.forward_instr(observations)

                        instr = None
                        cell_reps, cells = None, None

                        if self.cfg.int_type == 'cell':
                            if self.cfg.is_test:
                                cell_reps, cells = self.int_module.get_cell_representation(observations, actions.squeeze(1), rewards.squeeze(1), instr)

                        head_output = self.actor_critic.forward_head(observations, actions.squeeze(1), rewards.squeeze(1), cells=cell_reps)
                        if self.cfg.use_transformer:
                            core_output, new_memes, attn_entropy = self.actor_critic.forward_core_transformer(head_output, mems.transpose(0, 1), rollout_step_list=rollout_step_list, mem_begin_index=mem_begin_index)
                            policy_outputs = self.actor_critic.forward_tail(core_output, task_ids, with_action_distribution=with_action_distribution)
                            policy_outputs.mems = new_memes
                            policy_outputs.attn_entropy = attn_entropy
                        else:
                            core_output, new_rnn_states, _ = self.actor_critic.forward_core_rnn(head_output, rnn_states, dones.squeeze(1), is_seq=False)
                            policy_outputs = self.actor_critic.forward_tail(core_output, task_ids, with_action_distribution=with_action_distribution)

                # TODO: handling int_mdoule when self.cfg.use_half=True is not applied by now
                if self.int_module is not None:
                    return_int_rewards = self.cfg.is_test
                    return_int_values = self.cfg.use_ppo or self.cfg.is_test
                    if self.cfg.is_test and 'obs' in prev_observations:
                        tmp = observations
                        observations = prev_observations
                        next_observations = tmp
                    int_rewards, int_values, output_dict = self.int_module.forward_policy(observations,
                                                       next_observations,
                                                       actions.squeeze(1),
                                                       rewards.squeeze(1),
                                                       int_rnn_states,
                                                       dones.squeeze(1),
                                                       core_output,
                                                       policy_outputs.action_logits,
                                                       instr,
                                                       is_seq=False,
                                                       episodic_stats=episodic_stats,
                                                       return_int_rewards=return_int_rewards,
                                                       return_int_values=return_int_values)

                if not self.cfg.use_half_learner_worker and self.cfg.use_half_policy_worker:
                    for key in policy_outputs:
                        policy_output = policy_outputs[key]
                        if (key != 'action_distribution'
                            and policy_output.dtype == torch.float16):
                            policy_outputs[key] = policy_output.float()

                if self.cfg.is_test and self.cfg.test_action_sample == 'argmax':
                    policy_outputs.actions = policy_outputs.action_distribution.sample_max()
                    del policy_outputs['action_distribution']

            # write mems in mems_buffer
            if self.cfg.use_transformer:
                midx = 0
                for request in self.requests:
                    actor_idx, split_idx, request_data = request
                    for env_idx, agent_idx, traj_buffer_idx, rollout_step, first_rollout, actor_env_step in request_data:
                        mem_index = actor_idx, split_idx, env_idx, agent_idx, actor_env_step % self.max_mems_buffer_len
                        self.mems_buffer[mem_index] = policy_outputs['mems'][midx]
                        self.mems_policy_version_buffer[mem_index] = self.latest_policy_version
                        midx += 1
                del policy_outputs['mems']

            with timing.timeit('to_cpu'):
                for key, output_value in policy_outputs.items():
                    policy_outputs[key] = output_value.cpu()
                if self.int_module is not None:
                    int_rewards = int_rewards.cpu() if int_rewards is not None else torch.zeros_like(actions).cpu()
                    policy_outputs['int_rewards'] = int_rewards
                    int_values = int_values.cpu() if int_values is not None else torch.zeros_like(actions).cpu()
                    policy_outputs['int_values'] = int_values
                    if self.cfg.int_type in ['cell'] or self.cfg.int_type in ['bebold', 'agac', 'ride'] and self.cfg.use_episodic_cnt:
                        policy_outputs['episodic_cnt'] = output_dict['episodic_cnt'].cpu()
                        if self.cfg.extended_input_cell or self.cfg.is_test:
                            policy_outputs['cell_ids'] = cells.long().reshape(len(cells), -1).cpu()
                    if self.cfg.int_type == 'agac':
                        policy_outputs['int_action_logits'] = output_dict['int_action_logits'].cpu()
                        policy_outputs['int_rnn_states'] = output_dict['int_new_rnn_states'].cpu()
                    if not self.cfg.use_transformer:
                        policy_outputs['rnn_states'] = new_rnn_states.cpu()

            with timing.timeit('format_outputs'):
                policy_outputs.policy_version = torch.empty([num_samples]).fill_(self.latest_policy_version)

                # concat all tensors into a single tensor for performance
                output_tensors = []
                for policy_output in self.shared_buffers.policy_outputs:
                    tensor_name = policy_output.name
                    output_value = policy_outputs[tensor_name].float()
                    if len(output_value.shape) == 1:
                        output_value.unsqueeze_(dim=1)
                    output_tensors.append(output_value)

                output_tensors = torch.cat(output_tensors, dim=1)

            with timing.timeit('postprocess'):
                self._enqueue_policy_outputs(self.requests, output_tensors)

        self.requests = []

    def _enqueue_policy_outputs(self, requests, output_tensors):
        output_idx = 0
        outputs_ready = set()

        policy_outputs = self.shared_buffers.policy_output_tensors

        for request in requests:
            actor_idx, split_idx, request_data = request
            worker_outputs = policy_outputs[actor_idx, split_idx]
            if self.cfg.use_transformer:
                for env_idx, agent_idx, traj_buffer_idx, rollout_step, _, _ in request_data: # writing at shared buffer
                    worker_outputs[env_idx, agent_idx].copy_(output_tensors[output_idx])
                    output_idx += 1
            else:
                for env_idx, agent_idx, traj_buffer_idx, rollout_step in request_data: # writing at shared buffer
                    worker_outputs[env_idx, agent_idx].copy_(output_tensors[output_idx])
                    output_idx += 1
            outputs_ready.add((actor_idx, split_idx))

        for actor_idx, split_idx in outputs_ready:
            advance_rollout_request = dict(split_idx=split_idx, policy_id=self.policy_id)
            self.actor_queues[actor_idx].put((TaskType.ROLLOUT_STEP, advance_rollout_request))

    def _init_model(self, init_model_data):
        if self.cfg.use_transformer:
            policy_version, state_dict, mems_buffer, mems_dones_buffer, mems_policy_version_buffer = init_model_data

            self.mems_buffer = mems_buffer
            self.mems_dones_buffer = mems_dones_buffer
            self.mems_policy_version_buffer = mems_policy_version_buffer
        else:
            policy_version, state_dict = init_model_data

        self.actor_critic.load_state_dict(state_dict, strict=False)
        if self.int_module is not None:
            self.int_module.load_state_dict(state_dict, strict=False)
        self.shared_model_weights = state_dict
        self.latest_policy_version = policy_version

    def _write_done_on_mems_dones_buffer(self, raw_index, actor_env_step):
        traj_tensors = self.shared_buffers.tensors_individual_transitions
        actor_idx, split_idx, env_idx, agent_idx, traj_buffer_idx, rollout_step = raw_index
        if rollout_step == 0:
            index_for_done = actor_idx, split_idx, env_idx, agent_idx, (traj_buffer_idx - 1) % self.num_traj_buffers, rollout_step - 1
        else:
            index_for_done = actor_idx, split_idx, env_idx, agent_idx, traj_buffer_idx, rollout_step - 1
        done = traj_tensors['dones'][index_for_done]
        index_for_mems_dones = actor_idx, split_idx, env_idx, agent_idx, (actor_env_step - 1) % self.max_mems_buffer_len
        self.mems_dones_buffer[index_for_mems_dones] = bool(done)

    def _update_weights(self, timing):
        learner_policy_version = self.policy_versions[self.policy_id].item()
        if self.latest_policy_version < learner_policy_version - self.cfg.policy_update_minimum_diff and self.shared_model_weights is not None:
            with timing.timeit('weight_update'):
                with self.policy_lock:
                    self.actor_critic.load_state_dict(self.shared_model_weights, strict=False)
                    if self.int_module is not None:
                        self.int_module.load_state_dict(self.shared_model_weights, strict=False)
                    if self.cfg.use_ema:
                        with torch.no_grad():
                            self.actor_critic_ema.update(self.actor_critic, step=self.num_policy_updates)

            self.latest_policy_version = learner_policy_version

            if self.num_policy_updates % 10 == 0:
                log.info(
                    'Updated weights on worker %d-%d, policy_version %d (%.5f)',
                    self.policy_id, self.worker_idx, self.latest_policy_version, timing.weight_update,
                )

            self.num_policy_updates += 1

    # noinspection PyProtectedMember
    def _run(self):
        # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        psutil.Process().nice(min(self.cfg.default_niceness + 2, 20))

        gpu_mask = [self.cfg.local_rank] if self.cfg.local_rank >= 0 else None
        cuda_envvars_for_policy(self.policy_id, 'inference', gpu_mask=gpu_mask)
        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        with timing.timeit('init'):
            # initialize the Torch modules
            log.info('Initializing model on the policy worker %d-%d...', self.policy_id, self.worker_idx)

            torch.set_num_threads(1)

            if self.cfg.device == 'gpu':
                # we should already see only one CUDA device, because of env vars
                assert torch.cuda.device_count() == 1
                self.device = torch.device('cuda', index=0)
            else:
                self.device = torch.device('cpu')

            self.actor_critic = create_actor_critic(self.cfg, self.obs_space, self.action_space, timing)
            self.actor_critic.model_to_device(self.device)
            if self.cfg.use_half_policy_worker:
                self.actor_critic.half()

            for p in self.actor_critic.parameters():
                p.requires_grad = False  # we don't train anything here

            if self.cfg.use_intrinsic:
                from .intrinsic import IntrinsicModule
                self.int_module = IntrinsicModule(self.cfg,
                                                  self.obs_space,
                                                  self.action_space,
                                                  self.actor_critic.core.get_core_out_size(),
                                                  timing=timing)
                self.int_module.to(device=self.device)
                if self.cfg.use_half_policy_worker:
                    self.int_module.half()
                for p in self.int_module.parameters():
                    p.requires_grad = False  # we don't train anything here


            log.info('Initialized model on the policy worker %d-%d!', self.policy_id, self.worker_idx)

            if self.cfg.use_ema:
                actor_critic_ema = create_actor_critic(self.cfg, self.obs_space, self.action_space, timing)
                self.actor_critic_ema = ExponentialMovingAverage(actor_critic_ema, mu=0.9999, data_parallel=True)
                self.actor_critic_ema.module.model_to_device(self.device)
                self.actor_critic_ema.eval()
                self.actor_critic_ema.update(self.actor_critic, step=-1) # init model param (overwrite)
                log.info(f'Created exponential moving average model')

            if self.cfg.use_transformer:
                self.mems = self.mems.to(self.device)
                self.mems_dones = self.mems_dones.to(self.device)

        last_report = last_cache_cleanup = time.time()
        last_report_samples = 0
        request_count = deque(maxlen=50)

        # very conservative limit on the minimum number of requests to wait for
        # this will almost guarantee that the system will continue collecting experience
        # at max rate even when 2/3 of workers are stuck for some reason (e.g. doing a long env reset)
        # Although if your workflow involves very lengthy operations that often freeze workers, it can be beneficial
        # to set min_num_requests to 1 (at a cost of potential inefficiency, i.e. policy worker will use very small
        # batches)
        min_num_requests = self.cfg.num_workers // (self.cfg.num_policies * self.cfg.policy_workers_per_policy)
        min_num_requests //= self.cfg.policy_worker_batch_size_factor # default is 2 (originally 3 in SF)
        min_num_requests = max(1, min_num_requests)
        if self.cfg.is_test:
            min_num_requests = 1
        log.info('Min num requests: %d', min_num_requests)

        # Again, very conservative timer. Only wait a little bit, then continue operation.
        wait_for_min_requests = 0.025

        while not self.terminate:
            try:
                while self.stop_experience_collection[self.policy_id]:
                    with self.resume_experience_collection_cv:
                        self.resume_experience_collection_cv.wait(timeout=0.05)

                waiting_started = time.time()
                while len(self.requests) < min_num_requests and time.time() - waiting_started < wait_for_min_requests:
                    try:
                        with timing.timeit('wait_policy'), timing.add_time('wait_policy_total'):
                            policy_requests = self.policy_queue.get_many(timeout=0.005)
                        self.requests.extend(policy_requests)
                    except Empty:
                        pass

                self._update_weights(timing)

                with timing.timeit('one_step'), timing.add_time('handle_policy_step'):
                    if self.initialized:
                        if len(self.requests) > 0:
                            request_count.append(len(self.requests))
                            self._handle_policy_steps(timing)

                try:
                    task_type, data = self.task_queue.get_nowait()

                    # task from the task_queue
                    if task_type == TaskType.INIT:
                        self._init()
                    elif task_type == TaskType.TERMINATE:
                        self.terminate = True
                        break
                    elif task_type == TaskType.INIT_MODEL:
                        self._init_model(data)

                    self.task_queue.task_done()
                except Empty:
                    pass

                if time.time() - last_report > 3.0 and 'one_step' in timing:
                    #timing_stats = dict(wait_policy=timing.wait_policy, step_policy=timing.one_step)
                    timing_stats = {
                        key: val for key, val in timing.items()
                        if key not in ['wait_policy_total', 'handle_policy_step']
                    } # summary timeit stats of timing
                    samples_since_last_report = self.total_num_samples - last_report_samples

                    stats = memory_stats('policy_worker', self.device)
                    if len(request_count) > 0:
                        stats['avg_request_count'] = np.mean(request_count)

                    self.report_queue.put(dict(
                        times_policy_worker=timing_stats, samples=samples_since_last_report, policy_id=self.policy_id, stats=stats,
                    ))
                    last_report = time.time()
                    last_report_samples = self.total_num_samples

                if time.time() - last_cache_cleanup > 300.0 or (not self.cfg.benchmark and self.total_num_samples < 1000):
                    if self.cfg.device == 'gpu':
                        torch.cuda.empty_cache()
                    last_cache_cleanup = time.time()

            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected on worker %d-%d', self.policy_id, self.worker_idx)
                self.terminate = True
            except:
                log.exception('Unknown exception on policy worker')
                self.terminate = True

        time.sleep(0.2)
        log.info('Policy worker avg. requests %.2f, timing: %s', np.mean(request_count), timing)

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.initialized_event.wait()

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        join_or_kill(self.process)
