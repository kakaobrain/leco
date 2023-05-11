import glob
import os
import shutil
import signal
import threading
import time
from collections import OrderedDict, deque
from os.path import join
from queue import Empty, Queue, Full
from threading import Thread

import numpy as np
import random
import psutil
import torch

import torch_optimizer
from torch.multiprocessing import Process, Event as MultiprocessingEvent


if os.name == 'nt':
    from sample_factory.utils import Queue as MpQueue
else:
    from faster_fifo import Queue as MpQueue

from sample_factory.algorithms.appo.appo_utils import TaskType, list_of_dicts_to_dict_of_lists, memory_stats, cuda_envvars_for_policy, \
    TensorBatcher, iter_dicts_recursively, copy_dict_structure, ObjectPool
from sample_factory.algorithms.appo.model import CPCA, create_actor_critic
from sample_factory.algorithms.appo.population_based_training import PbtTask
from sample_factory.algorithms.utils.action_distributions import get_action_distribution, is_continuous_action_space
from sample_factory.algorithms.utils.algo_utils import calculate_gae, EPS
from sample_factory.algorithms.utils.pytorch_utils import to_scalar
from sample_factory.utils.decay import LinearDecay
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log, AttrDict, experiment_dir, ensure_dir_exists, join_or_kill, safe_get, safe_put
from sample_factory.utils.profile import Profiler
from dist.dist_utils import dist_init, dist_broadcast_model, dist_all_reduce_buffers, dist_reduce_gradient
from sample_factory.algorithms.optim.lamb import create_lamb_optimizer
from sample_factory.algorithms.utils.pytorch_utils import build_rnn_inputs


class LearnerWorker:
    def __init__(
        self, worker_idx, policy_id, cfg, obs_space, action_space, report_queue, policy_worker_queues, shared_buffers,
        policy_lock, resume_experience_collection_cv,
    ):
        log.info('Initializing the learner %d for policy %d', worker_idx, policy_id)

        self.worker_idx = worker_idx
        self.policy_id = policy_id

        self.cfg = cfg

        # PBT-related stuff
        self.should_save_model = True  # set to true if we need to save the model to disk on the next training iteration
        self.load_policy_id = None  # non-None when we need to replace our parameters with another policy's parameters
        self.pbt_mutex = threading.Lock()
        self.new_cfg = None  # non-None when we need to update the learning hyperparameters

        self.terminate = False
        self.num_batches_processed = 0

        self.obs_space = obs_space
        self.action_space = action_space

        self.rollout_tensors = shared_buffers.tensor_trajectories
        self.traj_tensors_available = shared_buffers.is_traj_tensor_available
        self.policy_versions = shared_buffers.policy_versions
        self.stop_experience_collection = shared_buffers.stop_experience_collection

        self.stop_experience_collection_num_msgs = self.resume_experience_collection_num_msgs = 0

        self.device = None
        self.actor_critic = None
        self.aux_loss_module = None
        self.int_module = None
        self.optimizer = None
        self.policy_lock = policy_lock
        self.resume_experience_collection_cv = resume_experience_collection_cv

        self.task_queue = MpQueue()
        self.report_queue = report_queue

        self.initialized_event = MultiprocessingEvent()
        self.initialized_event.clear()

        self.model_saved_event = MultiprocessingEvent()
        self.model_saved_event.clear()

        # queues corresponding to policy workers using the same policy
        # we send weight updates via these queues
        self.policy_worker_queues = policy_worker_queues

        self.experience_buffer_queue = Queue()

        self.tensor_batch_pool = ObjectPool()
        self.tensor_batcher = TensorBatcher(self.tensor_batch_pool)

        self.with_training = True if not self.cfg.is_test else False # set to False for debugging no-training regime
        self.train_in_background = self.cfg.train_in_background_thread  # set to False for debugging

        self.training_thread = Thread(target=self._train_loop) if self.train_in_background else None
        self.train_thread_initialized = threading.Event()

        self.is_training = False

        self.train_step = self.env_steps = self.optimizer_step_count = 0
        if 'dmlab_' in self.cfg.env:
            from sample_factory.envs.dmlab.dmlab_env import list_all_levels_for_experiment
        elif 'atari_' in self.cfg.env:
            from sample_factory.envs.atari.atari_utils import list_all_levels_for_experiment
        elif 'MiniGrid' in self.cfg.env:
            from sample_factory.envs.minigrid.minigrid_utils import list_all_levels_for_experiment

        self.all_levels = list_all_levels_for_experiment(self.cfg.env)
        self.env_steps_per_level = [0] * len(self.all_levels)

        # decay rate at which summaries are collected
        # save summaries every 20 seconds in the beginning, but decay to every 4 minutes in the limit, because we
        # do not need frequent summaries for longer experiments
        self.summary_rate_decay_seconds = LinearDecay([(0, 20), (100000, 120), (1000000, 240)])
        self.last_summary_time = 0

        self.last_saved_time = self.last_milestone_time = 0

        self.discarded_experience_over_time = deque([], maxlen=30)
        self.discarded_experience_timer = time.time()
        self.num_discarded_rollouts = 0

        self.timing = Timing()
        self.process = Process(target=self._run, daemon=True)

        if is_continuous_action_space(self.action_space) and self.cfg.exploration_loss == 'symmetric_kl':
            raise NotImplementedError('KL-divergence exploration loss is not supported with '
                                      'continuous action spaces. Use entropy exploration loss')

        self.exploration_loss_coeff_base = float(self.cfg.exploration_loss_coeff)
        if self.cfg.exploration_loss_coeff == 0.0:
            self.exploration_loss_func = lambda action_distr: 0.0
        elif self.cfg.exploration_loss == 'entropy':
            self.exploration_loss_func = self.entropy_exploration_loss
        elif self.cfg.exploration_loss == 'symmetric_kl':
            self.exploration_loss_func = self.symmetric_kl_exploration_loss
        else:
            raise NotImplementedError(f'{self.cfg.exploration_loss} not supported!')

        if self.cfg.save_milestones_step > 0:
            self.next_milestone_step = self.cfg.save_milestones_step

        if self.cfg.use_transformer:
            self.mems_dimensions = shared_buffers.mems_dimensions
            self.mems_dones_dimensions = shared_buffers.mems_dones_dimensions
            self.max_mems_buffer_len = shared_buffers.max_mems_buffer_len


    def start_process(self):
        self.process.start()

    def _init(self):
        log.info('Waiting for the learner to initialize...')
        self.train_thread_initialized.wait()
        log.info('Learner %d initialized', self.worker_idx)
        self.initialized_event.set()

    def _terminate(self):
        self.terminate = True

    def _broadcast_model_weights(self):
        state_dict = self.actor_critic.state_dict()
        if self.int_module is not None:
            state_dict.update(self.int_module.state_dict())
        policy_version = self.train_step

        log.debug('Broadcast model weights for model version %d', policy_version)
        if self.cfg.use_transformer:
            mems_buffer = self.mems_buffer
            mems_dones_buffer = self.mems_dones_buffer
            mems_policy_version_buffer = self.mems_policy_version_buffer
            model_state = (policy_version, state_dict, mems_buffer, mems_dones_buffer, mems_policy_version_buffer)
        else:
            model_state = (policy_version, state_dict)
        for q in self.policy_worker_queues:
            q.put((TaskType.INIT_MODEL, model_state))

    def _calculate_gae(self, buffer, curr_values):
        """
        Calculate advantages using Generalized Advantage Estimation.
        This is leftover the from previous version of the algorithm.
        Perhaps should be re-implemented in PyTorch tensors, similar to V-trace for uniformity.
        """

        rewards = buffer.rewards.view(-1, self.cfg.recurrence).cpu().numpy()
        dones = buffer.dones.view(-1, self.cfg.recurrence).cpu().numpy()  # [E, T]
        values_arr = curr_values.view(-1, self.cfg.recurrence).cpu().numpy()  # [E, T]

        # calculating fake values for the last step in the rollout
        # this will make sure that advantage of the very last action is always zero
        values = []
        for i in range(len(values_arr)):
            last_value, last_reward = values_arr[i][-1], rewards[i, -1]
            next_value = (last_value - last_reward) / self.cfg.gamma
            values.append(list(values_arr[i]))
            values[i].append(float(next_value))  # [T] -> [T+1]

        # calculating returns and GAE
        rewards = rewards.transpose((1, 0))  # [E, T] -> [T, E]
        dones = dones.transpose((1, 0))  # [E, T] -> [T, E]
        values = np.asarray(values).transpose((1, 0))  # [E, T+1] -> [T+1, E]

        advantages, returns = calculate_gae(rewards, dones, values, self.cfg.gamma, self.cfg.gae_lambda)

        advantages = advantages.transpose((1, 0))  # [T, E] -> [E, T]
        returns = returns.transpose((1, 0))  # [T, E] -> [E, T]

        advantages = torch.tensor(advantages).reshape(-1)
        returns = torch.tensor(returns).reshape(-1)

        return returns, advantages

    def _calculate_vtrace(self, values, rewards, dones, vtrace_rho, vtrace_c,
                          num_trajectories, recurrence, gamma, exclude_last=False):
        values_cpu = values.cpu()
        rewards_cpu = rewards.cpu()
        dones_cpu = dones.cpu()
        vtrace_rho_cpu = vtrace_rho.cpu()
        vtrace_c_cpu = vtrace_c.cpu()

        vs = torch.zeros((num_trajectories * recurrence))
        adv = torch.zeros((num_trajectories * recurrence))

        bootstrap_values = values_cpu[recurrence - 1::recurrence]
        values_BT = values_cpu.view(-1, recurrence)
        next_values = torch.cat([values_BT[:, 1:], bootstrap_values.view(-1, 1)], dim=1).view(-1)
        next_vs = next_values[recurrence - 1::recurrence]

        not_done = 1.0 - dones_cpu
        masked_gammas = not_done * gamma

        if exclude_last:
            rollout_recurrence = recurrence - 1
            adv[recurrence - 1::recurrence] = rewards_cpu[recurrence - 1::recurrence] + (
                    masked_gammas[recurrence - 1::recurrence] - 1) * next_vs
            vs[recurrence - 1::recurrence] = (next_vs * vtrace_rho_cpu[recurrence - 1::recurrence]
                                              * adv[recurrence - 1::recurrence])
        else:
            rollout_recurrence = recurrence

        for i in reversed(range(rollout_recurrence)):
            rewards = rewards_cpu[i::recurrence]
            not_done_times_gamma = masked_gammas[i::recurrence]

            curr_values = values_cpu[i::recurrence]
            curr_next_values = next_values[i::recurrence]
            curr_vtrace_rho = vtrace_rho_cpu[i::recurrence]
            curr_vtrace_c = vtrace_c_cpu[i::recurrence]

            delta_s = curr_vtrace_rho * (rewards + not_done_times_gamma * curr_next_values - curr_values)
            adv[i::recurrence] = rewards + not_done_times_gamma * next_vs - curr_values
            next_vs = curr_values + delta_s + not_done_times_gamma * curr_vtrace_c * (next_vs - curr_next_values)
            vs[i::recurrence] = next_vs

        return vs, adv

    def _calculate_vtrace_for_lirpg(self, values, rewards, dones, vtrace_rho, vtrace_c,
                          num_trajectories, recurrence, gamma, exclude_last=False):
        vs = torch.zeros_like(values.squeeze())
        adv = torch.zeros_like(rewards.squeeze())

        bootstrap_values = values[recurrence - 1::recurrence]
        values_BT = values.view(-1, recurrence)
        next_values = torch.cat([values_BT[:, 1:], bootstrap_values.view(-1, 1)], dim=1).view(-1)
        next_vs = next_values[recurrence - 1::recurrence]

        not_done = 1.0 - dones
        masked_gammas = not_done * gamma
        if exclude_last:
            rollout_recurrence = recurrence - 1
            adv[recurrence - 1::recurrence] = rewards[recurrence - 1::recurrence] + (
                    masked_gammas[recurrence - 1::recurrence] - 1) * next_vs
            vs[recurrence - 1::recurrence] = (next_vs * vtrace_rho[recurrence - 1::recurrence]
                                              * adv[recurrence - 1::recurrence])
        else:
            rollout_recurrence = recurrence

        for i in reversed(range(rollout_recurrence)):
            _rewards = rewards[i::recurrence]
            not_done_times_gamma = masked_gammas[i::recurrence]

            curr_values = values[i::recurrence]
            curr_next_values = next_values[i::recurrence]
            curr_vtrace_rho = vtrace_rho[i::recurrence]
            curr_vtrace_c = vtrace_c[i::recurrence]

            delta_s = curr_vtrace_rho * (_rewards + not_done_times_gamma * curr_next_values - curr_values)
            adv[i::recurrence] = _rewards + not_done_times_gamma * next_vs - curr_values
            next_vs = curr_values + delta_s + not_done_times_gamma * curr_vtrace_c * (next_vs - curr_next_values)
            vs[i::recurrence] = next_vs

        return vs, adv

    def _mark_rollout_buffer_free(self, rollout):
        r = rollout
        self.traj_tensors_available[r.worker_idx, r.split_idx][r.env_idx, r.agent_idx, r.traj_buffer_idx] = 1

    def _prepare_train_buffer(self, rollouts, macro_batch_size):
        trajectories = [AttrDict(r['t']) for r in rollouts]

        with self.timing.timeit('buffers'):
            buffer = AttrDict()

            # by the end of this loop the buffer is a dictionary containing lists of numpy arrays
            buffer['task_idx'] = []
            for i, t in enumerate(trajectories):
                for key, x in t.items():
                    if key not in buffer:
                        buffer[key] = []
                    buffer[key].append(x)
                buffer['task_idx'].append(rollouts[i]['task_idx'].repeat(self.cfg.recurrence).unsqueeze(1))

            # convert lists of dict observations to a single dictionary of lists
            for key, x in buffer.items():
                if isinstance(x[0], (dict, OrderedDict)):
                    buffer[key] = list_of_dicts_to_dict_of_lists(x)

        with self.timing.timeit('batching'):
            # concatenate rollouts from different workers into a single batch efficiently
            # that is, if we already have memory for the buffers allocated, we can just copy the data into
            # existing cached tensors instead of creating new ones. This is a performance optimization.
            use_pinned_memory = self.cfg.device == 'gpu'
            buffer = self.tensor_batcher.cat(buffer, macro_batch_size, use_pinned_memory, self.timing)

        with self.timing.timeit('tensors_gpu_float'):
            device_buffer = self._copy_train_data_to_device(buffer)

        with self.timing.timeit('buff_ready'):
            for r in rollouts:
                self._mark_rollout_buffer_free(r)

        with self.timing.timeit('squeeze'):
            # will squeeze actions only in simple categorical case
            tensors_to_squeeze = [
                'actions', 'log_prob_actions', 'policy_version', 'values',
                'rewards', 'dones', 'rewards_cpu', 'dones_cpu', 'raw_rewards',
            ]
            for tensor_name in tensors_to_squeeze:
                device_buffer[tensor_name].squeeze_()

        # we no longer need the cached buffer, and can put it back into the pool
        self.tensor_batch_pool.put(buffer)
        return device_buffer

    def _macro_batch_size(self, batch_size):
        return self.cfg.num_batches_per_iteration * batch_size

    def _process_macro_batch(self, rollouts, batch_size):
        macro_batch_size = self._macro_batch_size(batch_size)

        assert macro_batch_size % self.cfg.rollout == 0
        assert self.cfg.rollout % self.cfg.recurrence == 0
        assert macro_batch_size % self.cfg.recurrence == 0

        samples = env_steps = 0
        for rollout in rollouts:
            samples += rollout['length']
            env_steps += rollout['env_steps']

        with self.timing.timeit('prepare'):
            buffer = self._prepare_train_buffer(rollouts, macro_batch_size) # from cpu to gpu
            self.experience_buffer_queue.put((buffer, batch_size, samples, env_steps))

            if not self.cfg.benchmark and self.cfg.train_in_background_thread:
                # in PyTorch 1.4.0 there is an intense memory spike when the very first batch is being processed
                # we wait here until this is over so we can continue queueing more batches onto a GPU without having
                # a risk to run out of GPU memory
                while self.num_batches_processed < 1:
                    log.debug('Waiting for the first batch to be processed')
                    time.sleep(0.5)

    def _process_rollouts(self, rollouts):
        # batch_size can potentially change through PBT, so we should keep it the same and pass it around
        # using function arguments, instead of using global self.cfg

        batch_size = self.cfg.batch_size
        rollouts_in_macro_batch = self._macro_batch_size(batch_size) // self.cfg.rollout

        if len(rollouts) < rollouts_in_macro_batch:
            return rollouts

        discard_rollouts = 0
        policy_version = self.train_step
        select_rollout_ids = []
        chance_to_discard_no_reward_rollout = 1 - self.cfg.chance_discard_no_rew
        for i, r in enumerate(rollouts):
            rollout_min_version = r['t']['policy_version'].min().item()
            policy_lag = policy_version - rollout_min_version
            do_discard = False

            if policy_lag >= self.cfg.max_policy_lag:
                do_discard = True
            else:
                if random.random() > chance_to_discard_no_reward_rollout:
                    rollout_raw_rewards_abs_sum = r['t']['raw_rewards'].abs().sum().item()
                    if rollout_raw_rewards_abs_sum < 1e-6:
                        do_discard = True

            if do_discard:
                discard_rollouts += 1
                self._mark_rollout_buffer_free(r)
            else:
                select_rollout_ids.append(i)

        if discard_rollouts > 0:
            log.warning(
                'Discarding %d old rollouts, cut by policy lag threshold %d (learner %d)',
                discard_rollouts, self.cfg.max_policy_lag, self.policy_id,
            )
            rollouts = [rollouts[i] for i in select_rollout_ids]
            self.num_discarded_rollouts += discard_rollouts

        if len(rollouts) >= rollouts_in_macro_batch:
            # process newest rollouts
            rollouts_to_process = rollouts[:rollouts_in_macro_batch]
            rollouts = rollouts[rollouts_in_macro_batch:]

            self._process_macro_batch(rollouts_to_process, batch_size)
            # log.info('Unprocessed rollouts: %d (%d samples)', len(rollouts), len(rollouts) * self.cfg.rollout)

        return rollouts

    def _get_minibatches(self, batch_size, experience_size):
        """Generating minibatches for training."""
        assert self.cfg.rollout % self.cfg.recurrence == 0
        #assert experience_size % batch_size == 0, f'experience size: {experience_size}, batch size: {batch_size}'

        if self.cfg.num_batches_per_iteration == 1:
            return [None]  # single minibatch is actually the entire buffer, we don't need indices

        # indices that will start the mini-trajectories from the same episode (for bptt)
        indices = np.arange(0, experience_size, self.cfg.recurrence)
        indices = np.random.permutation(indices)

        # complete indices of mini trajectories, e.g. with recurrence==4: [4, 16] -> [4, 5, 6, 7, 16, 17, 18, 19]
        indices = [np.arange(i, i + self.cfg.recurrence) for i in indices]
        indices = np.concatenate(indices)

        assert len(indices) == experience_size

        num_minibatches = experience_size // batch_size
        minibatches = np.split(indices, num_minibatches)
        return minibatches

    @staticmethod
    def _get_minibatch(buffer, indices):
        if indices is None:
            # handle the case of a single batch, where the entire buffer is a minibatch
            return buffer

        mb = AttrDict()

        for item, x in buffer.items():
            if isinstance(x, (dict, OrderedDict)):
                mb[item] = AttrDict()
                for key, x_elem in x.items():
                    mb[item][key] = x_elem[indices]
            else:
                if len(indices) != len(x):
                    continue
                mb[item] = x[indices]

        return mb

    def _should_save_summaries(self):
        summaries_every_seconds = self.summary_rate_decay_seconds.at(self.train_step)
        if time.time() - self.last_summary_time < summaries_every_seconds:
            return False

        return True

    def _after_optimizer_step(self):
        """A hook to be called after each optimizer step."""
        self.train_step += 1
        if self.distenv.master:
            self._maybe_save()

    def _maybe_save(self):
        if time.time() - self.last_saved_time >= self.cfg.save_every_sec or self.should_save_model:
            self._save()
            self.model_saved_event.set()
            self.should_save_model = False
            self.last_saved_time = time.time()

    @staticmethod
    def checkpoint_dir(cfg, policy_id):
        checkpoint_dir = join(experiment_dir(cfg=cfg), f'checkpoint_p{policy_id}')
        return ensure_dir_exists(checkpoint_dir)

    @staticmethod
    def get_checkpoints(checkpoints_dir):
        checkpoints = glob.glob(join(checkpoints_dir, 'checkpoint_*'))
        return sorted(checkpoints)

    def _get_checkpoint_dict(self):
        checkpoint = {
            'train_step': self.train_step,
            'env_steps': self.env_steps,
            'model': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        if self.aux_loss_module is not None:
            checkpoint['aux_loss_module'] = self.aux_loss_module.state_dict()

        if self.int_module is not None:
            checkpoint['int_module'] = self.int_module.state_dict()
            if self.cfg.separate_int_optimizer:
                checkpoint['int_optimizer'] = self.int_optimizer.state_dict()

        return checkpoint

    def _save(self):
        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None

        checkpoint_dir = self.checkpoint_dir(self.cfg, self.policy_id)
        tmp_filepath = join(checkpoint_dir, '.temp_checkpoint')
        checkpoint_name = f'checkpoint_{self.train_step:09d}_{self.env_steps}.pth'
        filepath = join(checkpoint_dir, checkpoint_name)
        log.info('Saving %s...', tmp_filepath)
        torch.save(checkpoint, tmp_filepath)
        log.info('Renaming %s to %s', tmp_filepath, filepath)
        os.rename(tmp_filepath, filepath)

        if self.cfg.save_milestones_sec > 0:
            # milestones enabled
            if time.time() - self.last_milestone_time >= self.cfg.save_milestones_sec:
                milestones_dir = ensure_dir_exists(join(checkpoint_dir, 'milestones'))
                milestone_path = join(milestones_dir, f'{checkpoint_name}.milestone')
                log.debug('Saving a milestone %s', milestone_path)
                shutil.copy(filepath, milestone_path)
                self.last_milestone_time = time.time()

        if self.cfg.save_milestones_step > 0:
            if self.env_steps > self.next_milestone_step:
                milestones_dir = ensure_dir_exists(join(checkpoint_dir, 'milestones'))
                milestone_path = join(milestones_dir, f'{checkpoint_name}.milestone')
                log.debug('Saving a milestone %s', milestone_path)
                shutil.copy(filepath, milestone_path)
                self.next_milestone_step += self.cfg.save_milestones_step

        while len(self.get_checkpoints(checkpoint_dir)) > self.cfg.keep_checkpoints:
            oldest_checkpoint = self.get_checkpoints(checkpoint_dir)[0]
            if os.path.isfile(oldest_checkpoint):
                log.debug('Removing %s', oldest_checkpoint)
                os.remove(oldest_checkpoint)

    # @staticmethod
    def _policy_loss(self, ratio, log_prob_actions, adv, clip_ratio_low, clip_ratio_high, ppo=True, cfg=None, exclude_last=False):
        if ppo:
            clipped_ratio = torch.clamp(ratio, clip_ratio_low, clip_ratio_high)
            loss_unclipped = ratio * adv
            loss_clipped = clipped_ratio * adv
            loss = torch.min(loss_unclipped, loss_clipped)
        else:
            loss = log_prob_actions * adv
        if exclude_last:
            loss_mask = torch.ones_like(loss).view(-1, self.cfg.recurrence)
            loss_mask[:, -1] = 0
            loss = loss * loss_mask.contiguous().view(-1)

        if cfg.loss_type == 'mean':
            loss = -loss.mean()
        elif cfg.loss_type == 'sum':
            loss = loss.reshape([-1, cfg.recurrence])
            loss = loss.sum(dim=1)
            loss = -loss.mean()
        elif cfg.loss_type == 'sum_ori':
            loss = -loss.sum()
        else:
            raise NotImplementedError
        return loss

    def _value_loss(self, new_values, old_values, target, clip_value, ppo=True, cfg=None, exclude_last=False):
        if ppo and cfg.ppo_vclip:
            value_clipped = old_values + torch.clamp(new_values - old_values, -clip_value, clip_value)
            value_original_loss = (new_values - target).pow(2)
            value_clipped_loss = (value_clipped - target).pow(2)
            value_loss = torch.max(value_original_loss, value_clipped_loss)
        else:
            value_loss = (new_values - target).pow(2)

        if exclude_last:
            loss_mask = torch.ones_like(value_loss).view(-1, self.cfg.recurrence)
            loss_mask[:, -1] = 0
            value_loss = value_loss * loss_mask.contiguous().view(-1)

        if cfg.loss_type == 'mean':
            value_loss = value_loss.mean()
        elif cfg.loss_type == 'sum':
            value_loss = value_loss.reshape([-1, cfg.recurrence])
            value_loss = value_loss.sum(dim=1)
            value_loss = value_loss.mean()
        elif cfg.loss_type == 'sum_ori':
            value_loss = value_loss.sum()
        else:
            raise NotImplementedError

        value_loss *= self.cfg.value_loss_coeff

        return value_loss

    def entropy_exploration_loss(self, action_distribution, exclude_last=False):
        entropy = action_distribution.entropy()
        if exclude_last:
            entropy = entropy.view(-1, self.cfg.recurrence)[:, :-1].contiguous().view(-1)

        if self.cfg.loss_type == 'mean':
            entropy_loss = -self.cfg.exploration_loss_coeff * entropy.mean()
        elif self.cfg.loss_type == 'sum':
            entropy = entropy.reshape([-1, self.cfg.recurrence])
            entropy = entropy.sum(dim=1)
            entropy_loss = -self.cfg.exploration_loss_coeff * entropy.mean()
        elif self.cfg.loss_type == 'sum_ori':
            entropy_loss = -self.cfg.exploration_loss_coeff * entropy.sum()
        else:
            raise NotImplementedError
        return entropy_loss

    def symmetric_kl_exploration_loss(self, action_distribution):
        kl_prior = action_distribution.symmetric_kl_with_uniform_prior()
        kl_prior = kl_prior.mean()
        if not torch.isfinite(kl_prior):
            kl_prior = torch.zeros(kl_prior.shape)
        kl_prior = torch.clamp(kl_prior, max=30)
        kl_prior_loss = self.cfg.exploration_loss_coeff * kl_prior
        return kl_prior_loss

    def _prepare_observations(self, obs_tensors, gpu_buffer_obs):
        for d, gpu_d, k, v, _ in iter_dicts_recursively(obs_tensors, gpu_buffer_obs):
            device, dtype = self.actor_critic.device_and_type_for_input_tensor(k)
            tensor = v.detach().to(device, copy=True).type(dtype)
            gpu_d[k] = tensor

    def _copy_train_data_to_device(self, buffer):
        device_buffer = copy_dict_structure(buffer)

        for key, item in buffer.items():
            if key == 'obs':
                self._prepare_observations(item, device_buffer['obs'])
            elif key == 'next_obs':
                self._prepare_observations(item, device_buffer['next_obs'])
            else:
                device_tensor = item.detach().to(self.device, copy=True, non_blocking=True)
                device_buffer[key] = device_tensor.float()

        device_buffer['dones_cpu'] = buffer.dones.to('cpu', copy=True, non_blocking=True).float()
        device_buffer['rewards_cpu'] = buffer.rewards.to('cpu', copy=True, non_blocking=True).float()

        return device_buffer

    def _clip_grads(self, grads):
        norm_type = 2.0
        total_norm = torch.norm(
            torch.stack([torch.norm(grad.detach(), norm_type).to(self.device) for grad in grads if grad is not None]),
            norm_type)
        clip_coef = self.cfg.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for grad in grads:
                if grad is not None:
                    grad.detach().mul_(clip_coef.to(grad.device))

    def _train(self, gpu_buffer, batch_size, experience_size):
        with torch.no_grad():
            if self.int_module is not None and self.int_module.should_prepare_obs_norm():
                indices = np.arange(0, experience_size)
                next_obs = self._get_minibatch(gpu_buffer, indices).next_obs.obs
                self.int_module.prepare_obs_norm(next_obs)
                stats_and_summaries = None
                return stats_and_summaries

            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_actor_loss = 1e9
            epoch_actor_losses = []

            # V-trace parameters
            # noinspection PyArgumentList
            rho_hat = torch.Tensor([self.cfg.vtrace_rho])
            # noinspection PyArgumentList
            c_hat = torch.Tensor([self.cfg.vtrace_c])

            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high

            clip_value = self.cfg.ppo_clip_value
            gamma = self.cfg.gamma
            recurrence = self.cfg.recurrence

            if self.cfg.with_vtrace:
                assert recurrence == self.cfg.rollout and recurrence > 1, \
                    'V-trace requires to recurrence and rollout to be equal'

            num_sgd_steps = 0

            stats_and_summaries = None
            if not self.with_training:
                return stats_and_summaries

        for epoch in range(self.cfg.ppo_epochs):
            with self.timing.timeit('epoch_init'):
                if early_stop or self.terminate:
                    break

                summary_this_epoch = force_summaries = False

                minibatches = self._get_minibatches(batch_size, experience_size)

            for batch_num in range(len(minibatches)):
                with self.timing.timeit('minibatch_init'):
                    indices = minibatches[batch_num]

                    # current minibatch consisting of short trajectory segments with length == recurrence
                    mb = self._get_minibatch(gpu_buffer, indices)

                # calculate policy head outside of recurrent loop
                with self.timing.timeit('forward_head'):
                    cells = None
                    # instr = self.actor_critic.encoder.forward_instr(mb.obs)
                    instr = None
                    if hasattr(mb, 'cell_ids'):
                        cells, _ = self.int_module.get_cell_representation(mb.obs, mb.prev_actions.squeeze().long(), mb.prev_rewards.squeeze(), instr)
                    head_outputs = self.actor_critic.forward_head(mb.obs, mb.prev_actions.squeeze().long(), mb.prev_rewards.squeeze(), cells=cells)

                # initial rnn states
                with self.timing.timeit('bptt_initial'):
                    if self.cfg.use_transformer:
                        if self.cfg.mem_len > 0:
                            mems_indices = mb.mems_indices.type(torch.long)
                            for bidx in range(mems_indices.shape[0]):
                                s_idx = mems_indices[bidx, 4]
                                e_idx = mems_indices[bidx, 5]
                                if s_idx > e_idx:
                                    self.mems[:, bidx, :] = torch.cat(
                                        [self.mems_buffer[mems_indices[bidx, 0], mems_indices[bidx, 1],
                                         mems_indices[bidx, 2], mems_indices[bidx, 3], s_idx:, :],
                                         self.mems_buffer[mems_indices[bidx, 0], mems_indices[bidx, 1],
                                         mems_indices[bidx, 2], mems_indices[bidx, 3], :e_idx, :]])
                                    self.mems_dones[:, bidx, :] = torch.cat(
                                        [self.mems_dones_buffer[mems_indices[bidx, 0], mems_indices[bidx, 1],
                                         mems_indices[bidx, 2], mems_indices[bidx, 3], s_idx:, :],
                                         self.mems_dones_buffer[mems_indices[bidx, 0], mems_indices[bidx, 1],
                                         mems_indices[bidx, 2], mems_indices[bidx, 3], :e_idx, :]])
                                    mems_policy_version = torch.cat(
                                        [self.mems_policy_version_buffer[mems_indices[bidx, 0], mems_indices[bidx, 1],
                                         mems_indices[bidx, 2], mems_indices[bidx, 3], s_idx:, :],
                                         self.mems_policy_version_buffer[mems_indices[bidx, 0], mems_indices[bidx, 1],
                                         mems_indices[bidx, 2], mems_indices[bidx, 3], :e_idx, :]])

                                else:
                                    self.mems[:, bidx, :] = self.mems_buffer[mems_indices[bidx, 0],
                                                            mems_indices[bidx, 1], mems_indices[bidx, 2],
                                                            mems_indices[bidx, 3], s_idx:e_idx, :]
                                    self.mems_dones[:, bidx, :] = self.mems_dones_buffer[mems_indices[bidx, 0],
                                                                  mems_indices[bidx, 1], mems_indices[bidx, 2],
                                                                  mems_indices[bidx, 3], s_idx:e_idx, :]
                                    mems_policy_version = self.mems_policy_version_buffer[mems_indices[bidx, 0],
                                                                  mems_indices[bidx, 1], mems_indices[bidx, 2],
                                                                  mems_indices[bidx, 3], s_idx:e_idx, :]
                                    mems_version_diff = self.policy_versions - mems_policy_version.to('cpu')
                            mem_begin_index = [0] * (self.cfg.batch_size // self.cfg.recurrence)
                            dones = mb.dones.view(self.cfg.batch_size // self.cfg.recurrence,
                                                  self.cfg.chunk_size).transpose(0, 1)

                            actor_env_step_of_rollout_begin = mb.actor_env_step - self.cfg.rollout
                            mems_dones = self.mems_dones.transpose(0, 1)
                            mem_begin_index = self.actor_critic.cores[0].get_mem_begin_index(mems_dones,
                                                                                         actor_env_step_of_rollout_begin)
                        else:  # cfg.mem_len == 0
                            mems = None
                            mem_begin_index = [0] * (self.cfg.batch_size // self.cfg.recurrence)
                            dones = mb.dones.view(self.cfg.batch_size // self.cfg.recurrence,
                                                  self.cfg.chunk_size).transpose(0, 1)
                            mems_policy_version = []


                    elif self.cfg.use_rnn:
                        if self.cfg.packed_seq:
                            head_output_seq, rnn_states, inverted_select_inds = build_rnn_inputs(
                                head_outputs, mb.dones_cpu, mb.rnn_states, recurrence, self.cfg.actor_critic_share_weights, len(self.actor_critic.cores)
                            )
                        else:
                            if self.cfg.actor_critic_share_weights:
                                head_output_seq = head_outputs
                                rnn_states = mb.rnn_states[::recurrence]
                            else:
                                head_output_seq = head_outputs.chunk(len(self.actor_critic.cores), dim=1)
                                rnn_states = mb.rnn_states[::recurrence].chunk(len(self.actor_critic.cores), dim=1)
                    else:
                        rnn_states = mb.rnn_states[::recurrence]

                # calculate RNN outputs for each timestep in a loop
                # calculate transformer outputs for each timestep in a loop
                with self.timing.timeit('bptt'):
                    if self.cfg.use_rnn:
                        cudnn_enable = True
                        if self.int_module and self.int_module.use_lirpg:
                            cudnn_enable = False
                        with torch.backends.cudnn.flags(enabled=cudnn_enable):
                            with self.timing.timeit('bptt_forward_core'):
                                if self.cfg.packed_seq:
                                    #core_outputs = build_core_out_from_seq(core_output_seq, inverted_select_inds)
                                    core_outputs, _, all_hidden = self.actor_critic.forward_core_rnn(head_output_seq,
                                                                                                        rnn_states,
                                                                                                        mb.dones,
                                                                                                        is_seq=True, inverted_select_inds=inverted_select_inds)
                                else:
                                    core_outputs, _, all_hidden = self.actor_critic.forward_core_rnn(head_output_seq,
                                                                                                        rnn_states,
                                                                                                        mb.dones,
                                                                                                        is_seq=True)
                    elif self.cfg.use_transformer:
                        core_outputs, mems, attn_entropy = self.actor_critic.forward_core_transformer(head_outputs, self.mems,
                                                                                          mem_begin_index=mem_begin_index,
                                                                                          dones=dones)
                    else:
                        core_outputs, _, _ = self.actor_critic.forward_core_rnn(head_outputs, rnn_states, mb.dones, is_seq=False)

                num_trajectories = head_outputs.size(0) // recurrence
                if self.aux_loss_module is not None:
                    with self.timing.timeit('aux_loss'):
                        aux_loss = self.aux_loss_module(mb.actions.view(num_trajectories, recurrence, -1),
                                                       (1.0 - mb.dones).view(num_trajectories, recurrence, 1),
                                                       head_outputs.view(num_trajectories, recurrence, -1),
                                                       core_outputs.view(num_trajectories, recurrence, -1)
                                                       )



                with self.timing.timeit('tail'):
                    assert core_outputs.shape[0] == head_outputs.shape[0]

                    # calculate policy tail outside of recurrent loop
                    result = self.actor_critic.forward_tail(core_outputs, mb.task_idx.long(), with_action_distribution=True)

                    action_distribution = result.action_distribution
                    log_prob_actions = action_distribution.log_prob(mb.actions)
                    ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

                    # super large/small values can cause numerical problems and are probably noise anyway
                    ratio = torch.clamp(ratio, 0.01, 100.0)

                    values = result.values.squeeze()
                    normalized_values = result.normalized_values.squeeze()

                int_loss = 0
                if self.int_module is not None:
                    int_rewards, int_values, int_output_dict = self.int_module.forward(mb, core_outputs, instr)
                    if hasattr(int_output_dict, 'int_loss'):
                        int_loss = int_output_dict['int_loss']

                if self.cfg.use_task_specific_gamma:
                    level_to_mask = ['arbitrary_visuomotor', 'continuous_recognition',
                                     'sequential_comparison', 'visual_search']
                    gamma_mask = [any(ltm in self.all_levels[_level[0].int().data]
                                      for ltm in level_to_mask)
                                  for _level in mb.task_idx]
                    gamma_mask = torch.FloatTensor(gamma_mask)
                    gamma = (1. - gamma_mask) * self.cfg.gamma + gamma_mask * self.cfg.task_specific_gamma
                else:
                    gamma = self.cfg.gamma

                if self.int_module and self.int_module.use_lirpg:
                    with self.timing.timeit('calc_vtrace'):
                        rho_hat = torch.Tensor([self.cfg.vtrace_rho]).to(self.device)
                        c_hat = torch.Tensor([self.cfg.vtrace_c]).to(self.device)
                        vtrace_rho = torch.min(rho_hat, ratio)
                        vtrace_c = self.cfg.vtrace_lambda * torch.min(c_hat, ratio)
                        r_ex_in = self.cfg.ext_coef_rnd * mb.rewards + self.cfg.int_coef_rnd * int_rewards

                        if type(gamma) == torch.Tensor:
                            gamma = gamma.to(self.device)

                        targets, adv = self._calculate_vtrace_for_lirpg(values.detach().squeeze(), r_ex_in, mb.dones, vtrace_rho.detach(), vtrace_c.detach(), num_trajectories, recurrence, gamma, exclude_last=self.cfg.exclude_last)
                        if self.cfg.use_popart:
                            vs = targets.clone().detach()
                            targets = (targets - result.mus.squeeze(1)) / result.sigmas.squeeze(1)
                            adv = ((adv + values) - result.mus.squeeze(1)) / result.sigmas.squeeze(
                                1) - normalized_values.detach()

                        if not self.cfg.use_vmpo:
                            adv = adv * vtrace_rho.detach()

                        adv_mean = adv.mean().detach()
                        adv_std = adv.std().detach()
                        if self.cfg.use_adv_normalization:
                            adv = (adv - adv_mean) / max(1e-3, adv_std)  # normalize advantage
                else:
                    with torch.no_grad():  # these computations are not the part of the computation graph
                        if self.cfg.with_vtrace:
                            # using vtrace
                            with self.timing.timeit('calc_vtrace'):
                                # V-trace parameters
                                # noinspection PyArgumentList
                                rho_hat = torch.Tensor([self.cfg.vtrace_rho]).to(self.device)
                                # noinspection PyArgumentList
                                c_hat = torch.Tensor([self.cfg.vtrace_c]).to(self.device)
                                vtrace_rho = torch.min(rho_hat, ratio)
                                vtrace_c = self.cfg.vtrace_lambda * torch.min(c_hat, ratio)
                                rewards = mb.rewards
                                if self.int_module is not None:
                                    if self.cfg.int_type == 'agac':
                                        n_updates = self.cfg.train_for_env_steps // self.cfg.batch_size
                                        frac = 1.0 - self.train_step / n_updates
                                        agac_c = max(self.cfg.int_coef_rnd * frac, 0.0)
                                        int_rewards *= agac_c
                                        int_values *= agac_c
                                    else:
                                        if not self.int_module.has_value_head() or self.int_module.use_lirpg:
                                            rewards = self.cfg.ext_coef_rnd * rewards + self.cfg.int_coef_rnd * int_rewards

                                targets, adv = self._calculate_vtrace(values, rewards, mb.dones,
                                                                  vtrace_rho, vtrace_c, num_trajectories, recurrence,
                                                                  gamma, exclude_last=self.cfg.exclude_last
                                                                  )

                                if self.cfg.use_popart:
                                    vs = targets.clone().detach()
                                    targets = (targets - result.mus.squeeze(1).cpu()) / result.sigmas.squeeze(1).cpu()
                                    adv = ((adv + values.cpu()) - result.mus.squeeze(1).cpu()) / result.sigmas.squeeze(
                                        1).cpu() - normalized_values.cpu()

                                if self.int_module is not None:
                                    if self.int_module.has_value_head() and not self.int_module.use_lirpg:
                                        int_targets, int_adv = self._calculate_vtrace(int_values, int_rewards,
                                                                              torch.zeros_like(mb.dones), vtrace_rho, vtrace_c, num_trajectories, recurrence,
                                                                              self.cfg.int_gamma, exclude_last=self.cfg.exclude_last
                                                                              )
                                        adv, int_adv = self.int_module.define_advantage(adv, int_adv, self.cfg.int_coef_rnd)
                                        adv += int_adv
                                    elif self.cfg.int_type == 'agac':
                                        if self.cfg.use_episodic_cnt:
                                            _, adv = self._calculate_vtrace(values, int_output_dict['rewards_ep'], mb.dones,
                                                                            vtrace_rho, vtrace_c, num_trajectories, recurrence,
                                                                            self.cfg.gamma, exclude_last=self.cfg.exclude_last,
                                                                            )
                                            if self.cfg.use_popart:
                                                adv = ((adv + values.cpu()) - result.mus.squeeze(
                                                    1).cpu()) / result.sigmas.squeeze(
                                                    1).cpu() - normalized_values.cpu()

                                        adv += int_rewards.cpu()
                                        targets += int_values.cpu()

                                if not self.cfg.use_vmpo:
                                    adv = adv * vtrace_rho.cpu()
                        else:
                            # using regular GAE
                            with self.timing.timeit('calc_gae'):
                                targets, adv = self._calculate_gae(mb, values)

                    if self.cfg.use_vmpo:
                        indices = torch.sort(adv, descending=True).indices.to(self.device)
                        #adv[indices[adv.size(0) // 2:]] = -1e9
                        #if self.cfg.with_vtrace:
                        #    adv = vtrace_rho * torch.exp(adv/self.actor_critic.vmpo_eta.detach().cpu())
                        #else:
                        #    adv = torch.exp(adv / self.actor_critic.vmpo_eta.detach().cpu())
                        #adv_mean = adv.mean()
                        #adv_std = adv.std()
                        #adv = (adv - adv_mean) / max(1e-3, adv_std)  # normalize advantage
                    else:
                        if self.cfg.exclude_last:
                            adv_mean = adv.view(-1, recurrence)[:, :-1].mean()
                            adv_std = adv.view(-1, recurrence)[:, :-1].std()
                        else:
                            adv_mean = adv.mean()
                            adv_std = adv.std()
                        if self.cfg.use_adv_normalization:
                            adv = (adv - adv_mean) / max(1e-3, adv_std)  # normalize advantage
                    adv = adv.to(self.device)

                with self.timing.timeit('losses'):
                    if self.cfg.use_vmpo:
                        old_action_distribution = get_action_distribution(self.actor_critic.action_space, mb.action_logits)
                        kl = torch.clamp(torch.sum(old_action_distribution.probs.detach() * (old_action_distribution.log_probs.detach() - action_distribution.log_probs), dim=1), min=0.0)
                        alpha_loss = torch.mean(self.actor_critic.vmpo_alpha*(self.cfg.vmpo_eps_alpha-kl.detach())+self.actor_critic.vmpo_alpha.detach()*kl)
                        #if self.cfg.loss_type == 'mean':
                        #    alpha_loss = alpha_loss / (adv.size(0))
                        advhalf = torch.index_select(adv, 0, indices[:adv.size(0) // 2])
                        if self.cfg.with_vtrace:
                            vtrace_rho_half = torch.index_select(vtrace_rho, 0, indices[:vtrace_rho.size(0) // 2])
                            expadvhalf = vtrace_rho_half.to(self.device).detach() * torch.exp(advhalf/self.actor_critic.vmpo_eta)
                            #expadvhalf = torch.exp(advhalf / self.actor_critic.vmpo_eta)
                        else:
                            expadvhalf = torch.exp(advhalf/self.actor_critic.vmpo_eta)
                        eta_loss = self.actor_critic.vmpo_eta*self.cfg.vmpo_eps_eta+self.actor_critic.vmpo_eta*torch.log(torch.mean(expadvhalf))
                        #if self.cfg.loss_type == 'mean':
                        #    eta_loss = eta_loss / (advhalf.size(0))
                        #psi = expadvhalf/ max(1e-3, torch.sum(expadvhalf))
                        psi = torch.nn.functional.softmax(advhalf)
                        log_prob_actions_half = torch.index_select(log_prob_actions, 0, indices[:log_prob_actions.size(0) // 2])
                        #adv / max(1e-6, torch.sum(adv[indices[:adv.size(0) // 2]]))
                        #adv[indices[adv.size(0) // 2:]] = 0.0
                        #adv_mean = adv.mean()
                        #adv_std = adv.std()
                        #adv = (adv - adv_mean.detach()) / max(1e-3, adv_std.detach())
                        policy_loss = self._policy_loss(ratio, log_prob_actions_half, psi.detach(), clip_ratio_low, clip_ratio_high, ppo=False, cfg=self.cfg)
                        #policy_loss = self._policy_loss(ratio, log_prob_actions, adv.detach(), clip_ratio_low, clip_ratio_high, ppo=False, cfg=self.cfg)
                        policy_loss += eta_loss + alpha_loss
                    else:
                        if self.int_module and self.int_module.use_lirpg:
                            policy_loss = self._policy_loss(ratio, log_prob_actions, adv, clip_ratio_low, clip_ratio_high, ppo=self.cfg.use_ppo, cfg=self.cfg, exclude_last=self.cfg.exclude_last)
                        else:
                            policy_loss = self._policy_loss(ratio, log_prob_actions, adv.detach(), clip_ratio_low, clip_ratio_high, ppo=self.cfg.use_ppo, cfg=self.cfg, exclude_last=self.cfg.exclude_last)

                    if self.cfg.exploration_loss_decaying_step > 0:  # exploration loss coeff decaying
                        relative = np.exp(
                            -0.69314 * self.env_steps * self.distenv.world_size / self.cfg.exploration_loss_decaying_step)
                        self.cfg.exploration_loss_coeff = relative * self.exploration_loss_coeff_base
                        if self.cfg.exploration_loss_coeff_end > 0:
                            self.cfg.exploration_loss_coeff = max(self.cfg.exploration_loss_coeff,
                                                                  self.cfg.exploration_loss_coeff_end)

                    exploration_loss = self.exploration_loss_func(action_distribution, exclude_last=self.cfg.exclude_last)
                    actor_loss = policy_loss + exploration_loss
                    epoch_actor_losses.append(actor_loss.item())

                    targets = targets.to(self.device)
                    old_values = mb.values.squeeze()
                    old_normalized_values = mb.normalized_values.squeeze()
                    if self.cfg.use_popart:
                        if self.cfg.use_vmpo:
                            value_loss = self._value_loss(normalized_values, old_normalized_values.detach(), targets.detach(), clip_value,
                                                          ppo=False, cfg=self.cfg, exclude_last=self.cfg.exclude_last)
                        else:
                            value_loss = self._value_loss(normalized_values, old_normalized_values.detach(), targets.detach(), clip_value,
                                                          ppo=self.cfg.use_ppo, cfg=self.cfg, exclude_last=self.cfg.exclude_last)
                    else:
                        if self.cfg.use_vmpo:
                            value_loss = self._value_loss(values, old_values.detach(), targets.detach(), clip_value,
                                                          ppo=False, cfg=self.cfg, exclude_last=self.cfg.exclude_last)
                        else:
                            value_loss = self._value_loss(values, old_values.detach(), targets.detach(), clip_value,
                                                          ppo=self.cfg.use_ppo, cfg=self.cfg, exclude_last=self.cfg.exclude_last)
                    critic_loss = value_loss

                    int_value_loss = torch.zeros_like(value_loss)
                    if self.int_module:
                        if self.int_module.has_value_head() and not self.int_module.use_lirpg:
                            int_targets = int_targets.to(self.device)
                            int_value_loss = self._value_loss(int_values, mb.int_values.squeeze(), int_targets.detach(), clip_value, ppo=self.cfg.use_ppo, cfg=self.cfg, exclude_last=self.cfg.exclude_last)
                            critic_loss += int_value_loss

                    loss = actor_loss + critic_loss

                    if self.int_module is not None:
                        if self.cfg.separate_int_optimizer:
                            if self.int_module.use_lirpg:
                                pass
                            else:
                                int_loss *= self.cfg.int_loss_cost
                        else:
                            loss = loss + self.cfg.int_loss_cost * int_loss

                    high_loss = 30.0
                    if abs(to_scalar(policy_loss)) > high_loss or abs(to_scalar(value_loss)) > high_loss or abs(to_scalar(exploration_loss)) > high_loss:
                        log.warning(
                            'High loss value: %.4f %.4f %.4f %.4f (recommended to adjust the --reward_scale parameter)',
                            to_scalar(loss), to_scalar(policy_loss), to_scalar(value_loss), to_scalar(exploration_loss),
                        )
                        force_summaries = True

                    if self.aux_loss_module is not None:
                        loss = loss + aux_loss

                    if self.cfg.use_pbl:
                        novelty = None
                        if self.cfg.pbl_with_novelty:
                            novelty = int_output_dict.int_rewards_before_norm

                        if self.cfg.use_transformer:
                            pbl_loss, (pbl_l_forward, pbl_l_backward, pbl_l_for_reg, pbl_l_back_reg) = self.actor_critic.pbl.calc_loss(mb, core_outputs, num_trajectories, recurrence, novelty=novelty)
                        elif self.cfg.use_rnn:
                            pbl_loss, (pbl_l_forward, pbl_l_backward, pbl_l_for_reg, pbl_l_back_reg) = self.actor_critic.pbl.calc_loss(mb, all_hidden, num_trajectories, recurrence, novelty=novelty)
                        else:
                            pbl_loss, (pbl_l_forward, pbl_l_backward, pbl_l_for_reg, pbl_l_back_reg) = 0.0, (0.0, 0.0, 0.0, 0.0)
                        loss = loss + pbl_loss

                    if self.cfg.aux_task_classifier:
                        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
                        task_cls_loss = loss_fn(result.task_logits, mb.task_idx.squeeze(-1).long())
                        loss = loss + task_cls_loss

                        correct = 0
                        total = 0
                        with torch.no_grad():
                            _, predicted = torch.max(result.task_logits, -1)
                            total = mb.task_idx.size(0)
                            correct = (predicted == mb.task_idx.squeeze(-1)).sum().item()
                            task_cls_acc = correct / total

                    if self.cfg.encoder_type == 'resnet_encoder_decoder':
                        reconstruction_loss = self.actor_critic.encoder.basic_encoder.get_decoder_loss(mb.obs)
                        loss = loss + reconstruction_loss

                    if self.cfg.use_reward_prediction:
                        save_memory = True
                        if save_memory:
                            reward_prediction = self.actor_critic.forward_reward_prediction(
                                core_outputs, mb.actions, mb.next_obs, mb.task_idx.long(),
                                head_out=head_outputs)
                        else:
                            reward_prediction = self.actor_critic.forward_reward_prediction(core_outputs,
                                                                                            mb.actions,
                                                                                            mb.next_obs,
                                                                                            mb.task_idx.long())
                        reward_prediction_loss = self.cfg.reward_prediction_loss_coeff \
                                                 * torch.pow(reward_prediction - mb.rewards, 2).mean()
                        loss = loss + reward_prediction_loss

                # update the weights
                with self.timing.timeit('update'):
                    if self.int_module and self.int_module.use_lirpg:
                        with self.timing.timeit('grads_lirpg'):
                            params = OrderedDict(self.actor_critic.named_parameters())
                            grads = torch.autograd.grad(policy_loss, params.values(), retain_graph=True, create_graph=True, allow_unused=True)
                            # if self.distenv.world_size > 1:
                            #     dist_reduce_gradient(None, grads=grads)

                            # gradient clipping
                            if self.cfg.max_grad_norm > 0.0:
                                self._clip_grads(grads)

                            updated_params = self.int_module.update_params(params, grads, self.optimizer)

                        with self.timing.timeit('lirpg'):
                            _x = self.actor_critic.forward_head(mb.obs, mb.prev_actions.squeeze().long(), mb.prev_rewards.squeeze(), cells=cells, params=updated_params)
                            if self.cfg.use_rnn:
                                inverted_select_inds = None
                                if self.cfg.packed_seq:
                                    _x, rnn_states, inverted_select_inds = build_rnn_inputs(
                                        _x, mb.dones_cpu, mb.rnn_states, recurrence,
                                        self.cfg.actor_critic_share_weights, len(self.actor_critic.cores)
                                    )
                                else:
                                    if self.cfg.actor_critic_share_weights:
                                        rnn_states = mb.rnn_states[::recurrence]
                                    else:
                                        _x = _x.chunk(len(self.actor_critic.cores), dim=1)
                                        rnn_states = mb.rnn_states[::recurrence].chunk(len(self.actor_critic.cores), dim=1)

                                with torch.backends.cudnn.flags(enabled=False):
                                    _x, _, _ = self.actor_critic.forward_core_rnn(_x, rnn_states, mb.dones, is_seq=True, inverted_select_inds=inverted_select_inds, params=updated_params)
                            else:
                                _x, _, _ = self.actor_critic.forward_core_transformer(_x, self.mems, mem_begin_index=mem_begin_index, dones=dones, params=updated_params)

                            new_result = self.actor_critic.forward_tail(_x, mb.task_idx.long(), with_action_distribution=True, params=updated_params)

                            new_action_distribution = new_result.action_distribution
                            new_log_prob_actions = new_action_distribution.log_prob(mb.actions)
                            new_ratio = torch.exp(new_log_prob_actions - log_prob_actions.detach())
                            ex_values = int_values

                            vtrace_rho = torch.min(rho_hat, new_ratio)
                            vtrace_c = self.cfg.vtrace_lambda * torch.min(c_hat, new_ratio)
                            r_ex = self.cfg.ext_coef_rnd * mb.rewards
                            ex_targets, new_adv = self._calculate_vtrace(ex_values, r_ex, mb.dones, vtrace_rho, vtrace_c, num_trajectories, recurrence, self.cfg.gamma, exclude_last=self.cfg.exclude_last)

                            # TODO: separating parameters of popart for mix(r_ex + r_i), which is "poparted" above codes, and r_ex to "popart" here.
                            # new_targets = (new_targets - new_result.mus.squeeze(1).cpu()) / new_result.sigmas.squeeze(1).cpu()
                            # new_adv = ((new_adv + new_values.cpu()) - new_result.mus.squeeze(1).cpu()) / new_result.sigmas.squeeze(1).cpu() - new_normalized_values.cpu()

                            new_adv = new_adv * vtrace_rho.cpu()
                            if self.cfg.use_adv_normalization:
                                new_adv_mean = new_adv.mean()
                                new_adv_std = new_adv.std()
                                new_adv = (new_adv - new_adv_mean) / max(1e-3, new_adv_std)  # normalize advantage

                            new_adv = new_adv.to(self.device)
                            ex_targets = ex_targets.to(self.device)
                            new_policy_loss = self._policy_loss(new_ratio, new_log_prob_actions, new_adv.detach(), clip_ratio_low, clip_ratio_high, ppo=self.cfg.use_ppo, cfg=self.cfg, exclude_last=self.cfg.exclude_last)
                            # new_exploration_loss = self.exploration_loss_func(new_action_distribution, exclude_last=self.cfg.exclude_last)
                            # new_actor_loss = new_policy_loss + new_exploration_loss
                            new_actor_loss = new_policy_loss
                            int_value_loss = self._value_loss(ex_values, None, ex_targets.detach(), clip_value, ppo=self.cfg.use_ppo, cfg=self.cfg, exclude_last=self.cfg.exclude_last)
                            lirpg_loss = new_actor_loss + (self.cfg.vex_coef/self.cfg.value_loss_coeff) * int_value_loss
                            int_loss += self.cfg.lirpg_loss_cost * lirpg_loss
                            int_loss *= self.cfg.int_loss_cost

                    # following advice from https://youtu.be/9mS1fIYj1So set grad to None instead of optimizer.zero_grad()
                    for p in self.actor_critic.parameters():
                        p.grad = None
                    if self.aux_loss_module is not None:
                        for p in self.aux_loss_module.parameters():
                            p.grad = None
                    if self.int_module is not None:
                        self.int_module.zero_grad()

                    int_module_should_update = self.int_module and not self.cfg.separate_int_optimizer and not self.int_module.use_lirpg
                    with self.timing.timeit('backward'):
                        if self.int_module and self.int_module.use_lirpg:
                            loss.backward(retain_graph=True)
                            if self.cfg.separate_int_optimizer:
                                int_params = OrderedDict(self.int_module.named_parameters())

                                # filter p.requires_grad=False
                                filter_params = []
                                for name, p in int_params.items():
                                    if not p.requires_grad:
                                        filter_params.append(name)
                                for name in filter_params:
                                    del int_params[name]

                                int_grads = torch.autograd.grad(int_loss, int_params.values(), allow_unused=True)
                                for p, grad in zip(int_params.values(), int_grads):
                                    p.grad = grad
                            else:
                                int_loss.backward()
                        else:
                            loss.backward()
                            if self.int_module and self.cfg.separate_int_optimizer:
                                int_loss.backward()
                    # if self.cfg.use_transformer:
                    #     internal_grad_norm_dict = self.actor_critic.cores[0].get_internal_value_dict()

                    if self.distenv.world_size > 1:
                        with self.timing.timeit('dist_all_reduce_gradient'):
                            dist_reduce_gradient(self.actor_critic)
                            if self.aux_loss_module is not None:
                                dist_reduce_gradient(self.aux_loss_module)
                            if int_module_should_update:
                                dist_reduce_gradient(self.int_module.unwrapped)

                    grad_norm_before_clip = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in self.actor_critic.parameters()
                        if p.grad is not None
                    )
                    if self.aux_loss_module is not None:
                        grad_norm_before_clip += sum(
                            p.grad.data.norm(2).item() ** 2
                            for p in self.aux_loss_module.parameters()
                            if p.grad is not None
                        )

                    if self.int_module is not None:
                        grad_norm_before_clip += self.int_module.calc_grad_norm()
                    grad_norm_before_clip = grad_norm_before_clip ** 0.5

                    if self.cfg.max_grad_norm > 0.0:
                        with self.timing.timeit('clip'):
                            params = list(self.actor_critic.parameters())
                            if self.aux_loss_module is not None:
                                params += list(self.aux_loss_module.parameters())
                            if int_module_should_update:
                                params += list(self.int_module.parameters())
                            torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm)

                    grad_norm = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in self.actor_critic.parameters()
                        if p.grad is not None
                    )

                    if int_module_should_update:
                        grad_norm += self.int_module.calc_grad_norm()

                    grad_norm = grad_norm ** 0.5

                    curr_policy_version = self.train_step  # policy version before the weight update

                    if self.optimizer_step_count < self.cfg.warmup_optimizer:
                        lr_original = self.optimizer.param_groups[0]['lr']
                        self.optimizer.param_groups[0]['lr'] = 0.0

                    with self.policy_lock:
                        if self.distenv.world_size > 1:
                            if self.distenv.master:
                                self.optimizer.step()
                            dist_broadcast_model(self.actor_critic)
                            if self.aux_loss_module is not None:
                                dist_broadcast_model(self.aux_loss_module)
                            if int_module_should_update:
                                dist_broadcast_model(self.int_module.unwrapped)
                        else:
                            self.optimizer.step()

                        if self.cfg.scheduler is not None:
                            #self.scheduler.step(self.env_steps * int(os.environ.get('WORLD_SIZE', '1')))
                            self.scheduler.step(self.env_steps)

                    if self.int_module and (self.cfg.separate_int_optimizer or self.int_module.use_lirpg):
                        if self.cfg.separate_int_optimizer:
                            int_optim = self.int_optimizer
                        else:
                            int_optim = self.optimizer

                        if self.distenv.world_size > 1:
                            dist_reduce_gradient(self.int_module.unwrapped)

                        if self.cfg.max_grad_norm > 0.0:
                            torch.nn.utils.clip_grad_norm_(self.int_module.parameters(), self.cfg.max_grad_norm)

                        with self.policy_lock:
                            if self.distenv.world_size > 1:
                                if self.distenv.master:
                                    int_optim.step()
                                dist_broadcast_model(self.int_module.unwrapped)
                            else:
                                int_optim.step()

                    if self.cfg.use_popart:
                        if self.cfg.with_vtrace:
                            mu, sigma, oldmu, oldsigma = self.actor_critic.update_mu_sigma(vs.to(self.device), mb.task_idx.long(), cfg=self.cfg)
                        else:
                            mu, sigma, oldmu, oldsigma = self.actor_critic.update_mu_sigma(targets.to(self.device), mb.task_idx.long(), cfg=self.cfg)
                        if self.distenv.world_size > 1:
                            dist_all_reduce_buffers(self.actor_critic)
                        self.actor_critic.update_parameters(mu, sigma, oldmu, oldsigma)

                        self.mu = mu.detach().clone().cpu()
                        self.sigma = sigma.detach().clone().cpu()

                    if self.optimizer_step_count < self.cfg.warmup_optimizer:
                        self.optimizer.param_groups[0]['lr'] = lr_original
                    self.optimizer_step_count += 1

                    if self.cfg.use_vmpo:
                        with torch.no_grad():
                            self.actor_critic.vmpo_eta.copy_(torch.clamp(self.actor_critic.vmpo_eta, min=1e-8))
                            self.actor_critic.vmpo_alpha.copy_(torch.clamp(self.actor_critic.vmpo_alpha, min=1e-8))

                    num_sgd_steps += 1

                with torch.no_grad():
                    with self.timing.timeit('after_optimizer'):
                        self._after_optimizer_step()

                        # collect and report summaries
                        with_summaries = self._should_save_summaries() or force_summaries
                        if with_summaries and not summary_this_epoch:
                            stats_and_summaries = self._record_summaries(AttrDict(locals()))
                            # if self.cfg.use_transformer:
                            #     stats_and_summaries['internal_grad_norm'] = internal_grad_norm_dict
                                # stats_and_summaries['internal_attn_len'] = internal_attn_len_dict

                            summary_this_epoch = True
                            force_summaries = False

            # end of an epoch
            # this will force policy update on the inference worker (policy worker)
            self.policy_versions[self.policy_id] = self.train_step

            new_epoch_actor_loss = np.mean(epoch_actor_losses)
            loss_delta_abs = abs(prev_epoch_actor_loss - new_epoch_actor_loss)
            if loss_delta_abs < early_stopping_tolerance:
                early_stop = True
                log.debug(
                    'Early stopping after %d epochs (%d sgd steps), loss delta %.7f',
                    epoch + 1, num_sgd_steps, loss_delta_abs,
                )
                break

            prev_epoch_actor_loss = new_epoch_actor_loss
            epoch_actor_losses = []

        return stats_and_summaries

    def _record_summaries(self, train_loop_vars):
        var = train_loop_vars

        self.last_summary_time = time.time()
        stats = AttrDict()

        stats.grad_norm = var.grad_norm
        stats.grad_norm_before_clip = var.grad_norm_before_clip
        stats.loss = var.loss
        stats.value = var.result.values.mean()
        stats.entropy = var.action_distribution.entropy().mean()

        stats.exploration_coeff = self.cfg.exploration_loss_coeff
        stats.policy_loss = var.policy_loss
        stats.value_loss = var.value_loss
        stats.exploration_loss = var.exploration_loss
        stats.max_memory_allocated = torch.cuda.max_memory_allocated(self.device)
        if self.aux_loss_module is not None:
            stats.aux_loss = var.aux_loss
        if self.int_module is not None:
            stats.int_loss = var.int_loss
            stats.int_reward = var.int_rewards.mean()
            stats.int_value_loss = var.int_value_loss
            if hasattr(var.int_output_dict, 'int_rewards_only'):
                stats.int_rewards_only = var.int_output_dict.int_rewards_only
            if hasattr(var.int_output_dict, 'episodic_count_rate'):
                stats.episodic_count_rate = var.int_output_dict.episodic_count_rate
            if hasattr(var.int_output_dict, 'best_episodic_count_rate'):
                stats.best_episodic_count_rate = var.int_output_dict.best_episodic_count_rate
            if hasattr(var.int_output_dict, 'loss_vq'):
                stats.vqvae_loss = var.int_output_dict.loss_recons + var.int_output_dict.loss_vq
                stats.vqvae_recon = var.int_output_dict.loss_recons
                stats.vqvae_vq = var.int_output_dict.loss_vq
                if hasattr(var.int_output_dict, 'loss_reg'):
                    stats.vqvae_reg = var.int_output_dict.loss_reg
            if hasattr(var.int_output_dict, 'rnd_loss'):
                stats.rnd_loss = var.int_output_dict.rnd_loss
            if hasattr(var.int_output_dict, 'cell_probs'):
                stats.cell_probs_mean = var.int_output_dict.cell_probs.mean()
                stats.cell_probs_max = var.int_output_dict.cell_probs.max()
                stats.cell_probs_min = var.int_output_dict.cell_probs.min()
            if hasattr(var.int_output_dict, 'portion_cell_probs'):
                stats.portion_cell_probs_mean = var.int_output_dict.portion_cell_probs.mean()
                stats.portion_cell_probs_max = var.int_output_dict.portion_cell_probs.max()
                stats.portion_cell_probs_min = var.int_output_dict.portion_cell_probs.min()
            if hasattr(var.int_output_dict, 'state_novelty'):
                stats.state_novelty = var.int_output_dict.state_novelty.mean()
            if self.int_module.use_lirpg:
                stats.new_actor_loss = var.new_actor_loss.mean()
                stats.lirpg_loss = var.lirpg_loss.mean()
                stats.ex_adv_mean = var.new_adv.mean()
                stats.ex_adv_std = var.new_adv.std()
                stats.ex_adv_min = var.new_adv.min()
                stats.ex_adv_max = var.new_adv.max()
                stats.new_ratio_mean = var.new_ratio.mean()
                stats.new_ratio_std = var.new_ratio.std()
                stats.new_ratio_min = var.new_ratio.min()
                stats.new_ratio_max = var.new_ratio.max()
                stats.ex_values = var.ex_values.mean()
                stats.ex_targets = var.ex_targets.mean()


        if self.cfg.use_pbl:
            stats.pbl_loss = var.pbl_loss
            stats.pbl_l_forward = var.pbl_l_forward
            stats.pbl_l_backward = var.pbl_l_backward
        if self.cfg.aux_task_classifier:
            stats.task_cls_loss = var.task_cls_loss
            stats.task_cls_acc = var.task_cls_acc
        if self.cfg.encoder_type == 'resnet_encoder_decoder':
            stats.reconstruction_loss = var.reconstruction_loss
        if self.cfg.use_reward_prediction:
            stats.reward_prediction_loss = var.reward_prediction_loss

        stats.adv_min = var.adv.min()
        stats.adv_max = var.adv.max()
        stats.adv_avg = var.adv.mean()
        stats.adv_std = var.adv.std()
        stats.max_abs_logit = torch.abs(var.mb.action_logits).max()
        stats.max_abs_logprob = torch.abs(torch.log_softmax(var.mb.action_logits, dim=-1)).max()
        stats.max_prob = torch.softmax(var.mb.action_logits, dim=-1).max()
        stats.min_prob = torch.softmax(var.mb.action_logits, dim=-1).min()

        stats.rollout_pos_rew_count = (var.mb.raw_rewards.view(self.cfg.batch_size // self.cfg.rollout, -1) > 0).float().sum(dim=1).mean()
        stats.rollout_neg_rew_count = (var.mb.raw_rewards.view(self.cfg.batch_size // self.cfg.rollout, -1) < 0).float().sum(dim=1).mean()


        # attn_entropy
        if self.cfg.use_transformer:
            stats.internal_attn = AttrDict()
            stats.internal_attn.entropy_all = to_scalar( torch.mean(var.attn_entropy) ) # n_layers x n_heads
            n_layers, n_heads = var.attn_entropy.size()
            for i in range(n_layers):
                key = f'layer{i}_attn_entropy'
                stats.internal_attn[key] = to_scalar( torch.mean(var.attn_entropy[i,:]) )
                for j in range(n_heads):
                    key = f'layer{i}_head{j}_attn_entropy'
                    stats.internal_attn[key] = to_scalar( var.attn_entropy[i, j] )

        if hasattr(var.action_distribution, 'summaries'):
            stats.update(var.action_distribution.summaries())
        if var.epoch == self.cfg.ppo_epochs - 1 and var.batch_num == len(var.minibatches) - 1:
            # we collect these stats only for the last PPO batch, or every time if we're only doing one batch, IMPALA-style
            ratio_mean = torch.abs(1.0 - var.ratio).mean().detach()
            ratio_min = var.ratio.min().detach()
            ratio_max = var.ratio.max().detach()
            # log.debug('Learner %d ratio mean min max %.4f %.4f %.4f', self.policy_id, ratio_mean.cpu().item(), ratio_min.cpu().item(), ratio_max.cpu().item())

            value_delta = torch.abs(var.values - var.old_values)
            value_delta_avg, value_delta_max = value_delta.mean(), value_delta.max()

            # calculate KL-divergence with the behaviour policy action distribution
            old_action_distribution = get_action_distribution(
                self.actor_critic.action_space, var.mb.action_logits,
            )
            kl_old = var.action_distribution.kl_divergence(old_action_distribution)
            kl_old_mean = kl_old.mean()

            stats.kl_divergence = kl_old_mean
            stats.value_delta = value_delta_avg
            stats.value_delta_max = value_delta_max
            stats.fraction_clipped = ((var.ratio < var.clip_ratio_low).float() + (var.ratio > var.clip_ratio_high).float()).mean()
            stats.ratio_mean = ratio_mean
            stats.ratio_min = ratio_min
            stats.ratio_max = ratio_max
            stats.num_sgd_steps = var.num_sgd_steps
            if self.cfg.use_vmpo:
                stats.vmpo_eta = self.actor_critic.vmpo_eta
                stats.vmpo_alpha = self.actor_critic.vmpo_alpha
                stats.vmpo_eta_loss = var.eta_loss
                stats.vmpo_alpha_loss = var.alpha_loss

        # this caused numerical issues on some versions of PyTorch with second moment reaching infinity
        if 'adam' in self.cfg.optimizer_type:
            adam_max_second_moment = 0.0
            for key, tensor_state in self.optimizer.state.items():
                adam_max_second_moment = max(tensor_state['exp_avg_sq'].max().item(), adam_max_second_moment)
            stats.adam_max_second_moment = adam_max_second_moment

        version_diff = var.curr_policy_version - var.mb.policy_version
        stats.version_diff_avg = version_diff.mean()
        stats.version_diff_min = version_diff.min()
        stats.version_diff_max = version_diff.max()

        if self.cfg.use_transformer:
            if self.cfg.mem_len > 0:
                version_diff_mems = var.curr_policy_version - var.mems_policy_version
                stats.version_diff_mems_avg = version_diff_mems.mean()
                stats.version_diff_mems_min = version_diff_mems.min()
                stats.version_diff_mems_max = version_diff_mems.max()

        for key, value in stats.items():
            stats[key] = to_scalar(value)

        return stats

    def _update_pbt(self):
        """To be called from the training loop, same thread that updates the model!"""
        with self.pbt_mutex:
            if self.load_policy_id is not None:
                assert self.cfg.with_pbt

                log.debug('Learner %d loads policy from %d', self.policy_id, self.load_policy_id)
                self.load_from_checkpoint(self.load_policy_id)
                self.load_policy_id = None

            if self.new_cfg is not None:
                for key, value in self.new_cfg.items():
                    if self.cfg[key] != value:
                        log.debug('Learner %d replacing cfg parameter %r with new value %r', self.policy_id, key, value)
                        self.cfg[key] = value

                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.cfg.learning_rate
                    param_group['betas'] = (self.cfg.adam_beta1, self.cfg.adam_beta2)
                    log.debug('Updated optimizer lr to value %.7f, betas: %r', param_group['lr'], param_group['betas'])

                self.new_cfg = None

    @staticmethod
    def fix_old_model_mismatch(checkpoint_dict):
        if checkpoint_dict['model'].get('core.core.weight_ih', False) is not False:  # old models
            checkpoint_dict['model']["core.core.weight_ih_l0"] = checkpoint_dict['model'].pop("core.core.weight_ih")
            checkpoint_dict['model']["core.core.weight_hh_l0"] = checkpoint_dict['model'].pop("core.core.weight_hh")
            checkpoint_dict['model']["core.core.bias_ih_l0"] = checkpoint_dict['model'].pop("core.core.bias_ih")
            checkpoint_dict['model']["core.core.bias_hh_l0"] = checkpoint_dict['model'].pop("core.core.bias_hh")
            checkpoint_dict['model']["action_parameterization.distribution_linear.weight"] = checkpoint_dict[
                'model'].pop("distribution_linear.weight")
            checkpoint_dict['model']["action_parameterization.distribution_linear.bias"] = checkpoint_dict['model'].pop(
                "distribution_linear.bias")
        return checkpoint_dict

    @staticmethod
    def load_checkpoint(checkpoints, device):
        if len(checkpoints) <= 0:
            log.warning('No checkpoints found')
            return None
        else:
            latest_checkpoint = checkpoints[-1]

            # extra safety mechanism to recover from spurious filesystem errors
            num_attempts = 3
            for attempt in range(num_attempts):
                try:
                    log.warning('Loading state from checkpoint %s...', latest_checkpoint)
                    checkpoint_dict = torch.load(latest_checkpoint, map_location=device)
                    return checkpoint_dict
                except Exception:
                    log.exception(f'Could not load from checkpoint, attempt {attempt}')

    def _load_state(self, checkpoint_dict, load_progress=True):
        if load_progress:
            self.train_step = checkpoint_dict['train_step']
            self.env_steps = checkpoint_dict['env_steps']
            if self.cfg.resume:
                self.train_step = int(checkpoint_dict['train_step'] * self.cfg.checkpoint_world_size / self.distenv.world_size)
                self.env_steps = int(checkpoint_dict['env_steps'] * self.cfg.checkpoint_world_size / self.distenv.world_size)
        self.actor_critic.load_state_dict(checkpoint_dict['model'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        if self.aux_loss_module is not None:
            self.aux_loss_module.load_state_dict(checkpoint_dict['aux_loss_module'])
        if self.int_module is not None:
            log.info("batch_size: %d, recurrence: %d", self.cfg.batch_size, self.cfg.recurrence)
            self.int_module.load_state_dict(checkpoint_dict['int_module'], strict=False)
            if self.cfg.separate_int_optimizer:
                self.int_optimizer.load_state_dict(checkpoint_dict['int_optimizer'])
        log.info('Loaded experiment state at training iteration %d, env step %d', self.train_step, self.env_steps)

    def init_model(self):
        self.actor_critic = create_actor_critic(self.cfg, self.obs_space, self.action_space, self.timing, is_learner_worker=True)
        if self.cfg.use_half_learner_worker:
            self.actor_critic.half()

        self.actor_critic.model_to_device(self.device)
        self.actor_critic.share_memory()

        if self.cfg.use_cpc:
            self.aux_loss_module = CPCA(self.cfg, self.action_space)

        if self.aux_loss_module is not None:
            self.aux_loss_module.to(device=self.device)

        if self.cfg.use_intrinsic:
            from .intrinsic import IntrinsicModule
            self.int_module = IntrinsicModule(self.cfg,
                                              self.obs_space,
                                              self.action_space,
                                              self.actor_critic.core.get_core_out_size(),
                                              timing=self.timing, use_shared_memory=True)
            self.int_module.to(self.device)

        if self.cfg.use_transformer:
            self.mems_buffer = torch.zeros(self.mems_dimensions).to(self.device)
            if self.cfg.use_half_learner_worker:
                self.mems_buffer = self.mems_buffer.half()
            self.mems_buffer.share_memory_()

            self.mems_dones_buffer = torch.zeros(self.mems_dones_dimensions, dtype=torch.bool).to(self.device)
            self.mems_dones_buffer.share_memory_()

            self.mems_policy_version_buffer = torch.zeros(self.mems_dones_dimensions).to(self.device)
            self.mems_policy_version_buffer.share_memory_()

            # mems: [mem_len x batch_size/chunk_size x hidden_size_transformer]
            if self.cfg.mem_len:
                self.mems = torch.zeros([self.cfg.mem_len, self.cfg.batch_size // self.cfg.chunk_size, self.mems_dimensions[-1]], device=self.device)
                if self.cfg.use_half_learner_worker:
                    self.mems = self.mems.half()
                self.mems_dones = torch.zeros([self.cfg.mem_len, self.cfg.batch_size // self.cfg.chunk_size, 1], dtype=torch.bool, device=self.device)
            else:
                self.mems = None
                self.mems_dones = None

    def profile_model(self):
        profiler = Profiler(self.actor_critic)
        profiler_core = Profiler(self.actor_critic.cores[0])
        profiler_encoder = Profiler(self.actor_critic.encoders[0])

        params = profiler.params()
        params_core = profiler_core.params()
        params_enc = profiler_encoder.params()

        if self.int_module is not None:
            profiler_int_module = Profiler(self.int_module.unwrapped)
            params_int_module = profiler_int_module.params()
            params += params_int_module

        # params_enc = profiler.params(name_filter=lambda name: 'encoder' in name)
        # params_core = profiler.params(name_filter=lambda name: 'core' in name)
        # params_critic = profiler.params(name_filter=lambda name: 'critic' in name)

        return {'params': params, 'params_enc': params_enc, 'params_core': params_core}

    def load_from_checkpoint(self, policy_id, checkpoint_filename=None):
        if checkpoint_filename is None:
            checkpoints = self.get_checkpoints(self.checkpoint_dir(self.cfg, policy_id))
        else:
            checkpoints = [join(self.checkpoint_dir(self.cfg, policy_id), checkpoint_filename)]
        checkpoint_dict = self.load_checkpoint(checkpoints, self.device)
        # checkpoint_dict = self.fix_old_model_mismatch(checkpoint_dict)

        if checkpoint_dict is None:
            log.debug('Did not load from checkpoint, starting from scratch!')
        else:
            log.debug('Loading model from checkpoint')

            # if we're replacing our policy with another policy (under PBT), let's not reload the env_steps
            load_progress = policy_id == self.policy_id if not self.cfg.is_test else True
            self._load_state(checkpoint_dict, load_progress=load_progress)

    def initialize(self):
        with self.timing.timeit('init'):
            # initialize the Torch modules
            if self.cfg.seed is None:
                log.info('Starting seed is not provided')
            else:
                log.info('Setting fixed seed %d', self.cfg.seed)
                torch.manual_seed(self.cfg.seed)
                np.random.seed(self.cfg.seed)

            # this does not help with a single experiment
            # but seems to do better when we're running more than one experiment in parallel
            torch.set_num_threads(1)

            if self.cfg.device == 'gpu':
                torch.backends.cudnn.benchmark = True

                if self.cfg.local_rank < 0: # for debugging when running directly from run_algorithm.py
                    self.cfg.local_rank = 0
                    os.environ['WORLD_SIZE'] = str(1)
                # we should already see only one CUDA device, because of env vars
                assert torch.cuda.device_count() == 1
                self.device = torch.device('cuda', index=0)
            else:
                self.device = torch.device('cpu')

            self.init_model()
            self.num_params = self.profile_model()

            # dist env init
            self.distenv = dist_init(self.cfg)
            if self.distenv.world_size > 1:
                assert not self.cfg.with_pbt
                assert self.cfg.num_policies == 1
                dist_broadcast_model(self.actor_critic)
                if self.aux_loss_module is not None:
                    dist_broadcast_model(self.aux_loss_module)
                if self.int_module is not None:
                    dist_broadcast_model(self.int_module)

            params = list(filter(lambda p: p.requires_grad, self.actor_critic.parameters()))

            if self.aux_loss_module is not None:
                params += list(self.aux_loss_module.parameters())
            if self.int_module is not None:
                self.int_module.set_dist(self.distenv)
                print(self.distenv)
                if not self.cfg.separate_int_optimizer:
                    params += list(self.int_module.parameters())

            self.cfg.adam_eps = 1e-4 if self.cfg.use_half_learner_worker else self.cfg.adam_eps
            if self.cfg.scheduler == 'cosine':
                if self.cfg.optimizer_type == "adam":
                    self.optimizer = torch.optim.Adam(
                        params,
                        0,
                        betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
                        eps=self.cfg.adam_eps,
                    )
                elif self.cfg.optimizer_type == "rmsprop":
                    self.optimizer = torch.optim.RMSprop(
                        params,
                        lr=0,
                        momentum=self.cfg.momentum,
                        eps=self.cfg.rmsprop_eps,
                        alpha=self.cfg.alpha,
                    )
                elif self.cfg.optimizer_type == "adamw":
                    self.optimizer = torch.optim.AdamW(
                        params,
                        lr=0,
                        betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
                        eps=self.cfg.adam_eps
                    )
                elif self.cfg.optimizeer_type == "lamb":
                    self.optimizer = create_lamb_optimizer(
                        self.actor_critic, lr=0, weight_decay=1.5, bias_correction=False)
                else:
                    NotImplementedError
                from sample_factory.algorithms.utils.optim_utils import CosineAnnealingWarmUpRestarts
                self.scheduler = CosineAnnealingWarmUpRestarts(self.optimizer, self.cfg.train_for_env_steps, T_mult=1, eta_max=self.cfg.learning_rate, T_up=self.cfg.warmup, gamma=1., last_epoch=-1)


            elif self.cfg.scheduler is None or self.cfg.scheduler == 'linear':
                if self.cfg.optimizer_type == "adam":
                    self.optimizer = torch.optim.Adam(
                        params,
                        self.cfg.learning_rate,
                        betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
                        eps=self.cfg.adam_eps,
                    )
                elif self.cfg.optimizer_type == "rmsprop":
                    self.optimizer = torch.optim.RMSprop(
                        params,
                        lr=self.cfg.learning_rate,
                        momentum=self.cfg.momentum,
                        eps=self.cfg.rmsprop_eps,
                        alpha=self.cfg.alpha,
                    )
                elif self.cfg.optimizer_type == 'adamw':
                    self.optimizer = torch.optim.AdamW(
                        params,
                        lr=self.cfg.learning_rate,
                        betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
                        eps=self.cfg.adam_eps
                    )
                elif self.cfg.optimizer_type == "lamb":
                    self.optimizer = create_lamb_optimizer(
                                    self.actor_critic, lr=self.cfg.learning_rate, weight_decay=1.5, bias_correction=False)
                else:
                    NotImplementedError
                if self.cfg.scheduler == 'linear':
                    from sample_factory.algorithms.utils.optim_utils import LinearLR
                    self.scheduler = LinearLR(self.optimizer,1.0, 0.0, self.cfg.train_for_env_steps)
            else:
                NotImplementedError

            if self.int_module is not None:
                if self.cfg.separate_int_optimizer:
                    if self.cfg.optimizer_type == "adamw":
                        self.int_optimizer = torch.optim.AdamW(
                            self.int_module.parameters(),
                            self.cfg.learning_rate * self.cfg.int_lr_nu,
                            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
                            eps=self.cfg.adam_eps,
                        )
                    else:
                        self.int_optimizer = torch.optim.Adam(
                            self.int_module.parameters(),
                            self.cfg.learning_rate * self.cfg.int_lr_nu,
                            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
                            eps=self.cfg.adam_eps,
                        )

            if self.cfg.resume:
                if self.cfg.is_test:
                    self.load_from_checkpoint(self.cfg.test_policy_id[self.policy_id], self.cfg.checkpoint_filename)
                else:
                    self.load_from_checkpoint(self.policy_id, self.cfg.checkpoint_filename)

            if self.cfg.pretrained:
                print('load pretrained model weights... ')
                if self.cfg.encoder_subtype == 'resnet_18':
                    raise Exception('Not available in this code')
                elif self.cfg.encoder_subtype == 'resnet_50':
                    raise Exception('Not available in this code')
                else:
                    raise NotImplementedError
                self.actor_critic.encoder.basic_encoder.encoder.load_state_dict(state_dict, strict=False)

            self._broadcast_model_weights()  # sync the very first version of the weights

        self.train_thread_initialized.set()

    def _process_training_data(self, data, wait_stats=None):
        self.is_training = True

        buffer, batch_size, samples, env_steps = data

        assert samples == batch_size * self.cfg.num_batches_per_iteration

        if not self.cfg.is_test:
            self.env_steps += env_steps
            task_ids, counts = np.unique(np.array(buffer['task_idx'].cpu()), return_counts=True)
            for task_id, count in zip(task_ids, counts):
                self.env_steps_per_level[int(task_id)] += count * self.cfg.env_frameskip

        experience_size = buffer.rewards.shape[0]

        stats = dict(learner_env_steps=self.env_steps, policy_id=self.policy_id, times_learner_worker=dict())
        stats['learner_env_steps_per_level'] = self.env_steps_per_level

        task_ids_splits = torch.split(buffer['task_idx'], self.cfg.rollout)
        task_ids = [_task_ids[0].int().item() for _task_ids in task_ids_splits]
        if 'episodic_cnt' in buffer:
            episodic_cnt_rate_per_level = {}
            episodic_cnt_splits = torch.split(buffer['episodic_cnt'], self.cfg.rollout)
            for task_id, cnt in zip(task_ids, episodic_cnt_splits):
                rate = (cnt <= self.cfg.int_n_neighbors).sum()/cnt.shape[0]
                episodic_cnt_rate_per_level[task_id] = rate.item()
            stats['episodic_cnt_rate_per_level'] = episodic_cnt_rate_per_level

        if 'best_episodic_cnt' in buffer:
            best_episodic_cnt_rate_per_level = {}
            best_episodic_cnt_splits = torch.split(buffer['best_episodic_cnt'], self.cfg.rollout)
            for task_id, cnt in zip(task_ids, best_episodic_cnt_splits):
                rate = (cnt > 0).sum()/cnt.shape[0]
                best_episodic_cnt_rate_per_level[task_id] = rate.item()
            stats['best_episodic_cnt_rate_per_level'] = best_episodic_cnt_rate_per_level

        if 'raw_rewards' in buffer:
            rew_rate_per_level = {}
            rewards_splits = torch.split(buffer['raw_rewards'], self.cfg.rollout)
            for task_id, rewards in zip(task_ids, rewards_splits):
                rate = (rewards > 0).sum()/rewards.shape[0]
                rew_rate_per_level[task_id] = rate.item()
            stats['rew_rate_per_level'] = rew_rate_per_level

        with self.timing.timeit('train'):
            discarding_rate = self._discarding_rate()

            self._update_pbt()
            train_stats = self._train(buffer, batch_size, experience_size)

            if train_stats is not None:
                if 'internal_attn' in train_stats:
                    attn_stats = train_stats.pop('internal_attn')
                    stats['internal_attn'] = attn_stats

                if 'internal_grad_norm' in train_stats:
                    attn_stats = train_stats.pop('internal_grad_norm')
                    stats['internal_grad_norm'] = attn_stats

                stats['train'] = train_stats

                if wait_stats is not None:
                    wait_avg, wait_min, wait_max = wait_stats
                    stats['times_learner_worker']['wait_avg'] = wait_avg
                    stats['times_learner_worker']['wait_min'] = wait_min
                    stats['times_learner_worker']['wait_max'] = wait_max

                for key, value in self.timing.items():
                    stats['times_learner_worker'][key] = value

                stats['train']['discarded_rollouts'] = self.num_discarded_rollouts
                stats['train']['discarding_rate'] = discarding_rate

                stats['stats'] = memory_stats('learner', self.device)

                if self.cfg.scheduler is not None:
                    stats['train']['lr'] = self.scheduler.get_lr()[0]
                else:
                    stats['train']['lr'] = self.cfg.learning_rate

                if self.cfg.use_popart:
                    #mu = self.actor_critic.mu.detach().clone().cpu()
                    #sigma = self.actor_critic.sigma.detach().clone().cpu()
                    #stats['train']['popart_stat'] = dict(mu=self.mu, sigma=self.sigma, mu_ema=mu, sigma_ema=sigma)
                    stats['train']['popart_stat'] = dict(mu=self.mu, sigma=self.sigma)

                stats['train']['params'] = self.num_params['params'] / (1000 * 1000)
                stats['train']['params_enc'] = self.num_params['params_enc'] / (1000 * 1000)
                stats['train']['params_core'] = self.num_params['params_core'] / (1000 * 1000)

        self.is_training = False

        try:
            safe_put(self.report_queue, stats, queue_name='report')
        except Full:
            log.warning('Could not report training stats, the report queue is full!')

    def _train_loop(self):
        self.initialize()

        wait_times = deque([], maxlen=self.cfg.num_workers)
        last_cache_cleanup = time.time()

        while not self.terminate:
            with self.timing.timeit('train_wait'):
                data = safe_get(self.experience_buffer_queue)

            if self.terminate:
                break

            wait_stats = None
            wait_times.append(self.timing.train_wait)

            if len(wait_times) >= wait_times.maxlen:
                wait_times_arr = np.asarray(wait_times)
                wait_avg = np.mean(wait_times_arr)
                wait_min, wait_max = wait_times_arr.min(), wait_times_arr.max()
                # log.debug(
                #     'Training thread had to wait %.5f s for the new experience buffer (avg %.5f)',
                #     self.timing.train_wait, wait_avg,
                # )
                wait_stats = (wait_avg, wait_min, wait_max)

            self._process_training_data(data, wait_stats)
            self.num_batches_processed += 1

            if time.time() - last_cache_cleanup > 300.0 or (not self.cfg.benchmark and self.num_batches_processed < 50):
                if self.cfg.device == 'gpu':
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                last_cache_cleanup = time.time()

        time.sleep(0.3) # -> Why sleep 0.3 sec?
        log.info('Train loop self.timing: %s', self.timing)
        del self.actor_critic
        del self.device

    def _experience_collection_rate_stats(self):
        now = time.time()
        if now - self.discarded_experience_timer > 1.0:
            self.discarded_experience_timer = now
            self.discarded_experience_over_time.append((now, self.num_discarded_rollouts))

    def _discarding_rate(self):
        if len(self.discarded_experience_over_time) <= 1:
            return 0

        first, last = self.discarded_experience_over_time[0], self.discarded_experience_over_time[-1]
        delta_rollouts = last[1] - first[1]
        delta_time = last[0] - first[0]
        discarding_rate = delta_rollouts / (delta_time + EPS)
        return discarding_rate

    def _extract_rollouts(self, data):
        data = AttrDict(data)
        worker_idx, split_idx, traj_buffer_idx = data.worker_idx, data.split_idx, data.traj_buffer_idx

        rollouts = []
        for rollout_data in data.rollouts:
            env_idx, agent_idx = rollout_data['env_idx'], rollout_data['agent_idx']
            tensors = self.rollout_tensors.index((worker_idx, split_idx, env_idx, agent_idx, traj_buffer_idx))

            # read memory
            if self.cfg.use_transformer:
                actor_env_step = rollout_data['actor_env_step']
                s_idx = (actor_env_step - self.cfg.mem_len - self.cfg.rollout) % self.max_mems_buffer_len
                e_idx = (actor_env_step - self.cfg.rollout) % self.max_mems_buffer_len

                tensors['actor_env_step'] = torch.tensor([actor_env_step])
                tensors['mems_indices'] = torch.tensor([[worker_idx, split_idx, env_idx, agent_idx, s_idx, e_idx]], dtype=torch.long)

            rollout_data['t'] = tensors
            rollout_data['worker_idx'] = worker_idx
            rollout_data['split_idx'] = split_idx
            rollout_data['traj_buffer_idx'] = traj_buffer_idx
            rollouts.append(AttrDict(rollout_data))

        return rollouts

    def _process_pbt_task(self, pbt_task):
        task_type, data = pbt_task

        with self.pbt_mutex:
            if task_type == PbtTask.SAVE_MODEL:
                policy_id = data
                assert policy_id == self.policy_id
                self.should_save_model = True
            elif task_type == PbtTask.LOAD_MODEL:
                policy_id, new_policy_id = data
                assert policy_id == self.policy_id
                assert new_policy_id is not None
                self.load_policy_id = new_policy_id
            elif task_type == PbtTask.UPDATE_CFG:
                policy_id, new_cfg = data
                assert policy_id == self.policy_id
                self.new_cfg = new_cfg

    def _accumulated_too_much_experience(self, rollouts):
        max_minibatches_to_accumulate = self.cfg.num_minibatches_to_accumulate
        if max_minibatches_to_accumulate == -1:
            # default value
            max_minibatches_to_accumulate = 2 * self.cfg.num_batches_per_iteration

        # allow the max batches to accumulate, plus the minibatches we're currently training on
        max_minibatches_on_learner = max_minibatches_to_accumulate + self.cfg.num_batches_per_iteration

        minibatches_currently_training = int(self.is_training) * self.cfg.num_batches_per_iteration

        rollouts_per_minibatch = self.cfg.batch_size / self.cfg.rollout

        # count contribution from unprocessed rollouts
        minibatches_currently_accumulated = len(rollouts) / rollouts_per_minibatch

        # count minibatches ready for training
        minibatches_currently_accumulated += self.experience_buffer_queue.qsize() * self.cfg.num_batches_per_iteration

        total_minibatches_on_learner = minibatches_currently_training + minibatches_currently_accumulated

        return total_minibatches_on_learner >= max_minibatches_on_learner

    def _run(self):
        # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            psutil.Process().nice(self.cfg.default_niceness)
        except psutil.AccessDenied:
            log.error('Low niceness requires sudo!')

        if self.cfg.device == 'gpu':
            gpu_mask = [self.cfg.local_rank] if self.cfg.local_rank >= 0 else None
            cuda_envvars_for_policy(self.policy_id, 'learner', gpu_mask=gpu_mask)

        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(self.cfg.learner_main_loop_num_cores)

        rollouts = []

        if self.train_in_background:
            self.training_thread.start()
        else:
            self.initialize()
            log.error(
                'train_in_background set to False on learner %d! This is slow, use only for testing!', self.policy_id,
            )

        while not self.terminate:
            while True:
                try:
                    tasks = self.task_queue.get_many(timeout=0.005)

                    for task_type, data in tasks:
                        if task_type == TaskType.TRAIN:
                            with self.timing.timeit('extract'):
                                rollouts.extend(self._extract_rollouts(data))
                                # log.debug('Learner %d has %d rollouts', self.policy_id, len(rollouts))
                        elif task_type == TaskType.INIT:
                            self._init()
                        elif task_type == TaskType.TERMINATE:
                            time.sleep(0.3)
                            log.info('GPU learner self.timing: %s', self.timing)
                            self._terminate()
                            break
                        elif task_type == TaskType.PBT:
                            self._process_pbt_task(data)
                except Empty:
                    break

            if self._accumulated_too_much_experience(rollouts):
                # if we accumulated too much experience, signal the policy workers to stop experience collection
                if not self.stop_experience_collection[self.policy_id]:
                    self.stop_experience_collection_num_msgs += 1
                    # TODO: add a logger function for this
                    if self.stop_experience_collection_num_msgs >= 50:
                        log.info(
                            'Learner %d accumulated too much experience, stop experience collection! '
                            'Learner is likely a bottleneck in your experiment (%d times)',
                            self.policy_id, self.stop_experience_collection_num_msgs,
                        )
                        self.stop_experience_collection_num_msgs = 0

                self.stop_experience_collection[self.policy_id] = True
            elif self.stop_experience_collection[self.policy_id]:
                # otherwise, resume the experience collection if it was stopped
                self.stop_experience_collection[self.policy_id] = False
                with self.resume_experience_collection_cv:
                    self.resume_experience_collection_num_msgs += 1
                    if self.resume_experience_collection_num_msgs >= 50:
                        log.debug('Learner %d is resuming experience collection!', self.policy_id)
                        self.resume_experience_collection_num_msgs = 0
                    self.resume_experience_collection_cv.notify_all()

            with torch.no_grad():
                rollouts = self._process_rollouts(rollouts)

            if not self.train_in_background:
                while not self.experience_buffer_queue.empty():
                    training_data = self.experience_buffer_queue.get()
                    self._process_training_data(training_data)

            self._experience_collection_rate_stats()

        if self.train_in_background:
            self.experience_buffer_queue.put(None)
            self.training_thread.join()

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.initialized_event.wait()

    def save_model(self, timeout=None):
        self.model_saved_event.clear()
        save_task = (PbtTask.SAVE_MODEL, self.policy_id)
        self.task_queue.put((TaskType.PBT, save_task))
        log.debug('Wait while learner %d saves the model...', self.policy_id)
        if self.model_saved_event.wait(timeout=timeout):
            log.debug('Learner %d saved the model!', self.policy_id)
        else:
            log.warning('Model saving request timed out!')
        self.model_saved_event.clear()

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        join_or_kill(self.process)
