"""
Algorithm entry poing.
Methods of the APPO class initiate all other components (rollout & policy workers and learners) in the main thread,
and then fork their separate processes.
All data structures that are shared between processes are also created during the construction of APPO.

This class contains the algorithm main loop. All the actual work is done in separate worker processes, so
the only task of the main loop is to collect summaries and stats from the workers and log/save them to disk.

Hyperparameters specific to policy gradient algorithms are defined in this file. See also algorithm.py.

"""

import json
import math
import multiprocessing
import os
import time
from collections import deque
from os.path import join
from queue import Empty

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.multiprocessing import JoinableQueue as TorchJoinableQueue

from sample_factory.algorithms.algorithm import ReinforcementLearningAlgorithm
from sample_factory.algorithms.appo.actor_worker import ActorWorker
from sample_factory.algorithms.appo.appo_utils import make_env_func, iterate_recursively, set_global_cuda_envvars
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.policy_worker import PolicyWorker
from sample_factory.algorithms.appo.population_based_training import PopulationBasedTraining
from sample_factory.algorithms.appo.shared_buffers import SharedBuffers
from sample_factory.algorithms.utils.algo_utils import EXTRA_PER_POLICY_SUMMARIES, EXTRA_EPISODIC_STATS_PROCESSING, EXTRA_TEST_SUMMARIES, EXTRA_EPISODIC_TEST_STATS_PROCESSING, \
    ExperimentStatus
from sample_factory.envs.env_utils import get_default_reward_shaping
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import summaries_dir, experiment_dir, records_dir, log, str2bool, memory_consumption_mb, cfg_file, \
    ensure_dir_exists, list_child_processes, kill_processes, AttrDict, done_filename, save_git_diff
from sample_factory.algorithms.utils.evaluation_config import overwrite_test_cfg
from dist.dist_utils import AsyncDistSummaryWrapper
from sample_factory.utils.tbmon.writer import SummaryWriterThread

if os.name == 'nt':
    from sample_factory.utils import Queue as MpQueue
else:
    from faster_fifo import Queue as MpQueue


torch.multiprocessing.set_sharing_strategy('file_system')


class APPO(ReinforcementLearningAlgorithm):
    """Async PPO."""

    @classmethod
    def add_cli_args(cls, parser):
        p = parser
        super().add_cli_args(p)
        p.add_argument('--experiment_summaries_interval', default=300, type=int, help='How often in seconds we write avg. statistics about the experiment (reward, episode length, extra stats...)')
        p.add_argument('--train_summaries_interval', default=300, type=int,
                       help='How often in seconds we write train summaries (grad_norm, loss, ...)')
        p.add_argument('--report_interval', default=300, type=float,
                       help='How often in seconds we write report summaries (fps, ...)')

        # optimizer
        p.add_argument('--optimizer_type', default="adam", choices=["adam", "adamw", "rmsprop", "lamb"], help="type of optimizer")
        p.add_argument('--adam_eps', default=1e-6, type=float, help='Adam epsilon parameter (1e-8 to 1e-5 seem to reliably work okay, 1e-3 and up does not work)')
        p.add_argument('--adam_beta1', default=0.9, type=float, help='Adam momentum decay coefficient')
        p.add_argument('--adam_beta2', default=0.999, type=float, help='Adam second momentum decay coefficient')

        p.add_argument("--alpha", default=0.99, type=float, help="RMSProp smoothing constant.")
        p.add_argument("--momentum", default=0, type=float, help="RMSProp momentum.")
        p.add_argument("--rmsprop_eps", default=0.01, type=float, help="RMSProp epsilon.")

        p.add_argument('--gae_lambda', default=0.95, type=float, help='Generalized Advantage Estimation discounting (only used when V-trace is False')

        p.add_argument(
            '--rollout', default=32, type=int,
            help='Length of the rollout from each environment in timesteps.'
                 'Once we collect this many timesteps on actor worker, we send this trajectory to the learner.'
                 'The length of the rollout will determine how many timesteps are used to calculate bootstrapped'
                 'Monte-Carlo estimates of discounted rewards, advantages, GAE, or V-trace targets. Shorter rollouts'
                 'reduce variance, but the estimates are less precise (bias vs variance tradeoff).'
                 'For RNN policies, this should be a multiple of --recurrence, so every rollout will be split'
                 'into (n = rollout / recurrence) segments for backpropagation. V-trace algorithm currently requires that'
                 'rollout == recurrence, which what you want most of the time anyway.'
                 'Rollout length is independent from the episode length. Episode length can be both shorter or longer than'
                 'rollout, although for PBT training it is currently recommended that rollout << episode_len'
                 '(see function finalize_trajectory in actor_worker.py)',
        )

        p.add_argument('--num_workers', default=multiprocessing.cpu_count(), type=int, help='Number of parallel environment workers. Should be less than num_envs and should divide num_envs')

        p.add_argument(
            '--recurrence', default=32, type=int,
            help='Trajectory length for backpropagation through time. If recurrence=1 there is no backpropagation through time, and experience is shuffled completely randomly'
                 'For V-trace recurrence should be equal to rollout length.',
        )

        p.add_argument('--use_rnn', default=False, type=str2bool, help='Whether to use RNN core in a policy or not')
        p.add_argument('--rnn_type', default='gru', choices=['gru', 'lstm'], type=str, help='Type of RNN cell to use if use_rnn is True')
        p.add_argument('--rnn_num_layers', default=1, type=int, help='Number of RNN layers to use if use_rnn is True')

        p.add_argument('--use_transformer', default=False, type=str2bool, help='Whether to use transformer core in a policy or not')
        p.add_argument('--n_layer', default=12, type=int, help="num layers in transformer decoder")
        parser.add_argument("--n_heads", default=8, type=int,
                            help="number of MHA heads")
        parser.add_argument("--d_head", default=64, type=int,
                            help="MHA head dimension")
        parser.add_argument("--d_inner", default=2048, type=int,
                            help="the position wise ff network dimension -> d_model x d_inner")
        parser.add_argument("--use_gate", action='store_true',
                            help="whether to use gating in transformer decoder")
        parser.add_argument("--mem_len", default=512, type=int,
                            help="Length of memory segment for TXL")
        parser.add_argument("--chunk_size", default=96, type=int,
                           help="Size of chunks to chop batch into")
        parser.add_argument('--use_pe', default=True, type=str2bool, help='Whether to use relative positional encoding')
        parser.add_argument('--use_ema', default=False, type=str2bool, help='Whether to use exponential moving average on policy worker')

        p.add_argument('--use_ppo', default=True, type=str2bool, help='Whether to use PPO')
        p.add_argument('--ppo_clip_ratio', default=0.1, type=float, help='We use unbiased clip(x, 1+e, 1/(1+e)) instead of clip(x, 1+e, 1-e) in the paper')
        p.add_argument('--ppo_clip_value', default=1.0, type=float, help='Maximum absolute change in value estimate until it is clipped. Sensitive to value magnitude')
        p.add_argument('--batch_size', default=1024, type=int, help='Minibatch size for SGD')
        p.add_argument(
            '--num_batches_per_iteration', default=1, type=int,
            help='How many minibatches we collect before training on the collected experience. It is generally recommended to set this to 1 for most experiments, because any higher value will increase the policy lag.'
                 'But in some specific circumstances it can be beneficial to have a larger macro-batch in order to shuffle and decorrelate the minibatches.'
                 'Here and throughout the codebase: macro batch is the portion of experience that learner processes per iteration (consisting of 1 or several minibatches)',
        )
        p.add_argument('--ppo_epochs', default=1, type=int, help='Number of training epochs before a new batch of experience is collected')
        p.add_argument('--ppo_vclip', default=True, type=str2bool, help='whether to clip value when using ppo')

        p.add_argument(
            '--num_minibatches_to_accumulate', default=-1, type=int,
            help='This parameter governs the maximum number of minibatches the learner can accumulate before further experience collection is stopped.'
                 'The default value (-1) will set this to 2 * num_batches_per_iteration, so if the experience collection is faster than the training,'
                 'the learner will accumulate enough minibatches for 2 iterations of training (but no more). This is a good balance between policy-lag and throughput.'
                 'When the limit is reached, the learner will notify the actor workers that they ought to stop the experience collection until accumulated minibatches'
                 'are processed. Set this parameter to 1 * num_batches_per_iteration to further reduce policy-lag.' 
                 'If the experience collection is very non-uniform, increasing this parameter can increase overall throughput, at the cost of increased policy-lag.'
                 'A value of 0 is treated specially. This means the experience accumulation is turned off, and all experience collection will be halted during training.'
                 'This is the regime with potentially lowest policy-lag.'
                 'When this parameter is 0 and num_workers * num_envs_per_worker * rollout == num_batches_per_iteration * batch_size, the algorithm is similar to'
                 'regular synchronous PPO.',
        )

        p.add_argument('--max_grad_norm', default=4.0, type=float, help='Max L2 norm of the gradient vector')

        # components of the loss function
        p.add_argument('--exploration_loss_coeff', default=0.003, type=float,
                       help='Coefficient for the exploration component of the loss function.')
        p.add_argument('--exploration_loss_coeff_end', default=-1, type=float,
                       help='End value when exploration loss decaying is applied')
        p.add_argument('--value_loss_coeff', default=0.5, type=float, help='Coefficient for the critic loss')
        p.add_argument('--exploration_loss', default='entropy', type=str, choices=['entropy', 'symmetric_kl'],
                       help='Usually the exploration loss is based on maximizing the entropy of the probability'
                            ' distribution. Note that mathematically maximizing entropy of the categorical probability '
                            'distribution is exactly the same as minimizing the (regular) KL-divergence between'
                            ' this distribution and a uniform prior. The downside of using the entropy term '
                            '(or regular asymmetric KL-divergence) is the fact that penalty does not increase as '
                            'probabilities of some actions approach zero. I.e. numerically, there is almost '
                            'no difference between an action distribution with a probability epsilon > 0 for '
                            'some action and an action distribution with a probability = zero for this action.'
                            ' For many tasks the first (epsilon) distribution is preferrable because we keep some '
                            '(albeit small) amount of exploration, while the second distribution will never explore '
                            'this action ever again.'
                            'Unlike the entropy term, symmetric KL divergence between the action distribution '
                            'and a uniform prior approaches infinity when entropy of the distribution approaches zero,'
                            ' so it can prevent the pathological situations where the agent stops exploring. '
                            'Empirically, symmetric KL-divergence yielded slightly better results on some problems.',
                       )

        # APPO-specific
        p.add_argument(
            '--num_envs_per_worker', default=2, type=int,
            help='Number of envs on a single CPU actor, in high-throughput configurations this should be in 10-30 range for Atari/VizDoom'
                 'Must be even for double-buffered sampling!',
        )
        p.add_argument(
            '--worker_num_splits', default=2, type=int,
            help='Typically we split a vector of envs into two parts for "double buffered" experience collection'
                 'Set this to 1 to disable double buffering. Set this to 3 for triple buffering!',
        )

        p.add_argument('--num_policies', default=1, type=int, help='Number of policies to train jointly')
        p.add_argument('--policy_workers_per_policy', default=1, type=int, help='Number of policy workers that compute forward pass (per policy)')
        p.add_argument(
            '--max_policy_lag', default=1000, type=int,
            help='Max policy lag in policy versions. Discard all experience that is older than this. This should be increased for configurations with multiple epochs of SGD because naturally'
                 'policy-lag may exceed this value.',
        )
        p.add_argument(
            '--min_traj_buffers_per_worker', default=4, type=int,
            help='How many shared rollout tensors to allocate per actor worker to exchange information between actors and learners'
                 'Default value of 2 is fine for most workloads, except when differences in 1-step simulation time are extreme, like with some DMLab environments.'
                 'If you see a lot of warnings about actor workers having to wait for trajectory buffers, try increasing this to 4-6, this should eliminate the problem at a cost of more RAM.',
        )
        p.add_argument(
            '--decorrelate_experience_max_seconds', default=10, type=int,
            help='Decorrelating experience serves two benefits. First: this is better for learning because samples from workers come from random moments in the episode, becoming more "i.i.d".'
                 'Second, and more important one: this is good for environments with highly non-uniform one-step times, including long and expensive episode resets. If experience is not decorrelated'
                 'then training batches will come in bursts e.g. after a bunch of environments finished resets and many iterations on the learner might be required,'
                 'which will increase the policy-lag of the new experience collected. The performance of the Sample Factory is best when experience is generated as more-or-less'
                 'uniform stream. Try increasing this to 100-200 seconds to smoothen the experience distribution in time right from the beginning (it will eventually spread out and settle anyway)',
        )
        p.add_argument(
            '--decorrelate_envs_on_one_worker', default=True, type=str2bool,
            help='In addition to temporal decorrelation of worker processes, also decorrelate envs within one worker process'
                 'For environments with a fixed episode length it can prevent the reset from happening in the same rollout for all envs simultaneously, which makes experience collection more uniform.',
        )

        p.add_argument('--with_vtrace', default=True, type=str2bool, help='Enables V-trace off-policy correction. If this is True, then GAE is not used')
        p.add_argument('--vtrace_rho', default=1.0, type=float, help='rho_hat clipping parameter of the V-trace algorithm (importance sampling truncation)')
        p.add_argument('--vtrace_c', default=1.0, type=float, help='c_hat clipping parameter of the V-trace algorithm. Low values for c_hat can reduce variance of the advantage estimates (similar to GAE lambda < 1)')
        p.add_argument('--use_adv_normalization', default=True, type=str2bool, help='Whether to use advantage normalization or not')

        p.add_argument(
            '--set_workers_cpu_affinity', default=True, type=str2bool,
            help='Whether to assign workers to specific CPU cores or not. The logic is beneficial for most workloads because prevents a lot of context switching.'
                 'However for some environments it can be better to disable it, to allow one worker to use all cores some of the time. This can be the case for some DMLab environments with very expensive episode reset'
                 'that can use parallel CPU cores for level generation.',
        )
        p.add_argument(
            '--force_envs_single_thread', default=True, type=str2bool,
            help='Some environments may themselves use parallel libraries such as OpenMP or MKL. Since we parallelize environments on the level of workers, there is no need to keep this parallel semantic.'
                 'This flag uses threadpoolctl to force libraries such as OpenMP and MKL to use only a single thread within the environment.'
                 'Default value (True) is recommended unless you are running fewer workers than CPU cores.',
        )
        p.add_argument('--reset_timeout_seconds', default=120, type=int, help='Fail worker on initialization if not a single environment was reset in this time (worker probably got stuck)')

        p.add_argument('--default_niceness', default=0, type=int, help='Niceness of the highest priority process (the learner). Values below zero require elevated privileges.')

        p.add_argument(
            '--train_in_background_thread', default=True, type=str2bool,
            help='Using background thread for training is faster and allows preparing the next batch while training is in progress.'
                 'Unfortunately debugging can become very tricky in this case. So there is an option to use only a single thread on the learner to simplify the debugging.',
        )
        p.add_argument('--learner_main_loop_num_cores', default=1, type=int, help='When batching on the learner is the bottleneck, increasing the number of cores PyTorch uses can improve the performance')
        p.add_argument('--actor_worker_gpus', default=[], type=int, nargs='*', help='By default, actor workers only use CPUs. Changes this if e.g. you need GPU-based rendering on the actors')

        p.add_argument('--resume', default=False, type=str2bool,
                       help='Resume training from the last checkpoint')
        p.add_argument('--pretrained', default=False, type=str2bool,
                       help='use pretrained model for resenet-18/50')
        p.add_argument('--checkpoint_filename', default=None, type=str, help='load from the specific checkpoint file')

        # PBT stuff
        p.add_argument('--with_pbt', default=False, type=str2bool, help='Enables population-based training basic features')
        p.add_argument('--pbt_mix_policies_in_one_env', default=True, type=str2bool, help='For multi-agent envs, whether we mix different policies in one env.')
        p.add_argument('--pbt_period_env_steps', default=int(5e6), type=int, help='Periodically replace the worst policies with the best ones and perturb the hyperparameters')
        p.add_argument('--pbt_start_mutation', default=int(2e7), type=int, help='Allow initial diversification, start PBT after this many env steps')
        p.add_argument('--pbt_replace_fraction', default=0.3, type=float, help='A portion of policies performing worst to be replace by better policies (rounded up)')
        p.add_argument('--pbt_mutation_rate', default=0.15, type=float, help='Probability that a parameter mutates')
        p.add_argument('--pbt_replace_reward_gap', default=0.1, type=float, help='Relative gap in true reward when replacing weights of the policy with a better performing one')
        p.add_argument('--pbt_replace_reward_gap_absolute', default=1e-6, type=float, help='Absolute gap in true reward when replacing weights of the policy with a better performing one')
        p.add_argument('--pbt_optimize_batch_size', default=False, type=str2bool, help='Whether to optimize batch size or not (experimental)')
        p.add_argument(
            '--pbt_target_objective', default='true_reward', type=str,
            help='Policy stat to optimize with PBT. true_reward (default) is equal to raw env reward if not specified, but can also be any other per-policy stat.'
                 'For DMlab-30 use value "dmlab_target_objective" (which is capped human normalized score)',
        )

        # CPC|A options
        p.add_argument('--use_cpc', default=False, type=str2bool, help='Use CPC|A as an auxiliary loss durning learning')
        p.add_argument('--cpc_forward_steps', default=8, type=int, help='Number of forward prediction steps for CPC')
        p.add_argument('--cpc_time_subsample', default=6, type=int, help='Number of timesteps to sample from each batch. This should be less than recurrence to decorrelate experience.')
        p.add_argument('--cpc_forward_subsample', default=2, type=int, help='Number of forward steps to sample for loss computation. This should be less than cpc_forward_steps to decorrelate gradients.')
        p.add_argument('--cpc_loss_coeff', default=0.1, type=float, help='CPC|A loss weight')


        # debugging options
        p.add_argument('--benchmark', default=False, type=str2bool, help='Benchmark mode')
        p.add_argument('--sampler_only', default=False, type=str2bool, help='Do not send experience to the learner, measuring sampling throughput')

        # custom
        p.add_argument('--extended_input', default=False, type=str2bool, help='Whether to use extended input (+previous action and reward) to RNN')
        p.add_argument('--nonlinear_inplace', default=False, type=str2bool, help='Whether to use inplace on nonlinear functions')
        p.add_argument('--use_popart', default=False, type=str2bool, help='Whether to use PopArt normalization or not')
        p.add_argument('--popart_beta', default=0.0003, type=float, help='decay rate to track mean and standard derivation of the values')

        p.add_argument('--use_vmpo', default=False, type=str2bool, help='Whether to use v_mpo rather than ppo')
        p.add_argument('--vmpo_eps_eta', default=0.5, type=float, help='eps_eta for vmpo')
        p.add_argument('--vmpo_eps_alpha', default=0.1, type=float, help='eps_alpha for vmpo')

        # distributed training
        p.add_argument('--dist_backend', default='nccl', type=str, help='distributed backend')
        p.add_argument('--local_rank', default=0, type=int,
                            help='Used for multi-process training. Can either be manually set ' +
                                 'or automatically set by using \'python -m dist.launch\'.')
        p.add_argument("--nproc_per_node", type=int, default=1,
                            help='The number of processes to launch on each node. Can either be manually set ' +
                                 'or automatically set by using \'python -m dist.launch\'.')
        p.add_argument("--loss_type", type=str, default='mean', help='mean or sum or sum_ori loss')
        p.add_argument("--popart_clip_min", type=float, default=0.0001, help='popart clip min')

        p.add_argument('--packed_seq', default=True, type=str2bool, help='Use torch.nn.utils.rnn.PackedSequence')
        p.add_argument('--max_mems_buffer_len', default=-1, type=int, help='Lenth of mems buffer. Autoset if -1')
        p.add_argument('--extra_fc_critic', default=0, type=int, help='number of extra fc layers to be inserted')
        p.add_argument('--extra_fc_critic_hidden_size', default=-1, type=int,
                       help='hidden size of extra fc layers to be inserted, -1 means using default core_out_size')
        p.add_argument('--extra_fc_critic_nonlinearity', default='relu',
                       type=str, choices=['relu', 'elu'], help='nonlinearity of extra fc layers to be inserted')
        p.add_argument('--match_core_input_size', default=False, type=str2bool,
                       help="when using extended_input=True, whether to add an one-layer mlp after encoder to match input size of core as hidden size or not")

        p.add_argument('--policy_update_minimum_diff', default=0, type=int, help='update if policy_version < learner_version - policy_update_minimum_diff')

        p.add_argument('--use_pbl', default=False, type=str2bool, help='Add PBL aux loss')

        # intrinsic module
        p.add_argument('--use_intrinsic', default=False, type=str2bool, help='Whether to include intrinsic reward')
        p.add_argument('--int_type', default='rnd', type=str, choices=['rnd', 'bebold', 'cell', 'agac', 'lirpg', 'ride'], help='intrinsic reward type')
        p.add_argument('--separate_int_optimizer', default=False, type=str2bool, help='Whether to use separated optimizer for network producing intrinsic reward')
        p.add_argument('--int_lr_nu', default=0.3, type=float, help='relative learning rate for intrinsic loss')
        p.add_argument('--int_loss_cost', default=0.00004, type=float, help='intrinsic loss cost for agac')
        p.add_argument('--int_hidden_dim', default=256, type=int, help='hidden size in intrinsic network')
        p.add_argument('--int_gamma', default=0.99, type=float, help='Discount factor for intrinsic reward')
        p.add_argument('--int_coef_rnd', default=1.0, type=float, help='intrinsic coeff for advantage in rnd')
        p.add_argument('--ext_coef_rnd', default=2.0, type=float, help='extrinsic coeff for advantage in rnd')
        p.add_argument('--rnd_type', default='medium', type=str, help='type of RND network')
        p.add_argument('--use_shared_rnd', default=False, type=str2bool, help='Whether to use shared CNN encoder of RND')
        p.add_argument('--int_rnd_update_proportion', default=0.25, type=float, help='distill update proportion in rnd')
        p.add_argument('--int_obs_norm', default=True, type=str2bool, help='Whether to use observation normalization for intrinsic network')
        p.add_argument('--int_rew_norm', default=True, type=str2bool, help='Whether to use observation normalization for intrinsic network')
        p.add_argument('--int_pre_obs_norm_steps', default=50, type=int, help='number of initial steps for initializing observation normalization')
        p.add_argument('--int_n_neighbors', default=0, type=int, help='number of neighbors used in episodic pseudo counts')
        p.add_argument('--cell_type', default=None, type=str,  help='type of cell representation')
        p.add_argument('--cell_spec', default='30,20,2x3x4', type=str, help='Specify the cell spec. make form as \'offset-y,offset-x,HxWxD\'')
        p.add_argument('--cell_dim', default=256, type=int, help='representation size of cell in VQCellRep')
        p.add_argument('--cell_rep_cost', default=0.1, type=float, help='representation learning cost cost for VQCellRep')
        p.add_argument('--cell_reg', default='l2,1e-2', type=str, help='cell regularization type and coefficient for VQCellRep')
        p.add_argument('--extended_input_cell', default=False, type=str2bool, help='Whether to use extended input previous cell to actor')
        p.add_argument('--use_approx_norm_obs', default=False, type=str2bool, help='Whether to use approximated normalization for obs')
        p.add_argument('--use_approx_norm_int', default=False, type=str2bool, help='Whether to use approximated normalization for int_rew')
        p.add_argument('--use_episodic_cnt', default=True, type=str2bool, help='Whether to use episodic count in bebold')
        p.add_argument('--use_indicator', default=True, type=str2bool, help='Whether to use I(N_ep = 1) or 1/sqrt(N_ep) in bebold')
        p.add_argument('--int_scale_fac', default=0.5, type=float, help='scaling factor in bebold')
        p.add_argument('--separate_int_value', default=True, type=str2bool, help='Whether to use separated intrinsic value network')
        p.add_argument('--chance_discard_no_rew', default=0, type=float, help='chance to discard zero-rewarded rollout')
        p.add_argument('--int_agac_count_c', default=0.0125, type=float, help='reward weight of episodic state count for agac')

        p.add_argument('--checkpoint_world_size', default=16, type=int, help='Checkpoint world size')
        p.add_argument('--pbl_with_novelty', default=False, type=str2bool, help='')
        p.add_argument('--pbl_obs_norm', default=False, type=str2bool, help='')

        p.add_argument('--use_half_policy_worker', default=False, type=str2bool, help='Use half-precision in Policy Worker')
        p.add_argument('--use_half_learner_worker', default=False, type=str2bool, help='Use half-precision in Learner Worker')
        p.add_argument('--vtrace_lambda', default=1.0, type=float, help='rho value for vtrace')

        p.add_argument('--policy_worker_batch_size_factor', default=2, type=int, help='PolicyWorker batch size controlling factor (1<=, <inf)')
        p.add_argument('--pretrained_vq_path', default=None, type=str, help='Load specified pretrained vq-vae weights before learning')
        p.add_argument('--freeze_vq', default=False, type=str2bool, help='Freeze vq-vae weights')
        p.add_argument('--cell_learning', default='none', type=str, choices=['none', 'lirpg', 'lirpg_ca', 'lirpg_ca_naive'])
        p.add_argument('--vex_coef', default=1.0, type=float, help='coef of extrinsic value (r_ex) loss in LIRPG')
        p.add_argument('--int_coef_count', default=1.0, type=float, help='coef of episodic count in LECO')
        p.add_argument('--int_coef_lirpg', default=1.0, type=float, help='coef of intrinsic reward of LIRPG in LECO')
        p.add_argument('--lirpg_loss_cost', default=1.0, type=float, help='coef of intrinsic reward of LIRPG')
        p.add_argument('--rescale_leco_reward', default=True, type=str2bool, help='Whether to use rescale LECO reward')
        p.add_argument('--lirpg_spec', default='tanh,adam', type=str, help='Specify r_i activation and method that update new params')
        p.add_argument('--leco_temp', default=0.01, type=float, help='softmax temperature of LECO')
        p.add_argument('--rnd_loss_coef', default=1.0, type=float, help='distilation loss coeffient')
        p.add_argument('--cell_enc_hidden', default=1024, type=int, help='hidden size of cell encoder in LECO')
        p.add_argument('--return_reps', default=-1, type=int, help='hidden size of cell encoder in LECO')
        p.add_argument('--int_init', default=False, type=str2bool, help='whether to initialize int_module following policy_initialization')
        p.add_argument('--steps_start_decay', default=0, type=int, help='whether to initialize int_module following policy_initialization')
        p.add_argument('--steps_end_decay', default=0, type=int, help='whether to initialize int_module following policy_initialization')
        p.add_argument('--hash_dim', default=62, type=int, help='whether to initialize int_module following policy_initialization')



        p.add_argument('--aux_task_classifier', default=False, type=str2bool, help='Whether to use auxiliary task classifier')
        p.add_argument('--fixed_reset_seed', default=False, type=str2bool, help='Whether to use fixed dmlab episode')
        p.add_argument('--exploration_loss_decaying_step', default=-1, type=float,
                       help='expnentially 1/2 entropy loss coefficient for every given step, do not apply if it is less than 0')
        p.add_argument('--tau', default=1.0, type=float, help='trxl attention logit scaling factor')

        p.add_argument('--load_encoder_only', default=False, type=str2bool, help='Load encoder weights only from the checkpoint')
        p.add_argument('--freeze_encoder', default=False, type=str2bool, help='Freeze encoder weights')

        p.add_argument('--reconstruction_loss_coeff', default=1e-2, type=float,
                       help='Reconstruction loss coefficient for normalized input')
        p.add_argument('--reconstruction_loss_type', default='MSE', type=str, choices=['MSE', 'MSE_Inverted', 'CE'],
                       help='Reconstruction loss type (MSE or CE)')
        p.add_argument('--reconstruction_from_core_hidden', default=False, type=str2bool,
                       help='Reconstruc from core hidden or conv output')
        p.add_argument('--use_long_skip_reconstruction', default=False, type=str2bool,
                       help='Use long-skip connection in reconstruction or not')
        p.add_argument('--apply_tanh_for_mse_reconstruction', default=True, type=str2bool,
                       help='Use tanh nonlinearity for mse reconstruction')

        p.add_argument('--use_reward_prediction', default=False, type=str2bool,
                       help='Use reward prediction or not')
        p.add_argument('--reward_prediction_loss_coeff', default=1e-2, type=float,
                       help='Reward prediction loss (MSE) coefficient')

        p.add_argument('--use_task_specific_gamma', default=False, type=str2bool,
                       help='use task specific gamma for 3 of the psychlab tasks')
        p.add_argument('--task_specific_gamma', default=0.95, type=float, help='task specific gamma value')

        p.add_argument('--exclude_last', default=False, type=str2bool,
                       help='exclude last element of each rollout in vtrace calc, as they are falsely calculated')

    def __init__(self, cfg):
        super().__init__(cfg)

        assert not (self.cfg.use_transformer and self.cfg.use_rnn)
        if self.cfg.use_transformer:
            assert self.cfg.chunk_size == self.cfg.recurrence

        self.cfg.train_for_env_steps = int(self.cfg.train_for_env_steps) // int(os.environ.get('WORLD_SIZE', '1'))
        self.cfg.steps_start_decay = int(self.cfg.steps_start_decay) // int(os.environ.get('WORLD_SIZE', '1'))
        self.cfg.steps_end_decay = int(self.cfg.steps_end_decay) // int(os.environ.get('WORLD_SIZE', '1'))
        self.cfg.save_milestones_step = self.cfg.save_milestones_step // int(os.environ.get('WORLD_SIZE', '1'))
        self.cfg.warmup = self.cfg.warmup // int(os.environ.get('WORLD_SIZE', '1'))
        self.cfg.warmup_optimizer = int(self.cfg.warmup_optimizer)

        self.fps_per_level = True

        if 'dmlab_' in self.cfg.env:
            from sample_factory.envs.dmlab.dmlab_env import list_all_levels_for_experiment

        elif 'atari_' in self.cfg.env:
            from sample_factory.envs.atari.atari_utils import list_all_levels_for_experiment

        elif 'MiniGrid' in self.cfg.env:
            from sample_factory.envs.minigrid.minigrid_utils import list_all_levels_for_experiment

        self.all_levels = list_all_levels_for_experiment(self.cfg.env)
        self.num_levels = len(self.all_levels)

        if self.cfg.is_test:
            self.cfg = overwrite_test_cfg(self.cfg)
            self.test_policy_completed = []
            self.test_level_completed = []
            self.test_episodes_collected = [[0] * self.cfg.num_policies for _ in range(self.num_levels)]

            if self.cfg.record_to is not None:
                self.cfg.record_to = records_dir(experiment_dir(cfg=self.cfg), cfg=self.cfg)

        # we should not use CUDA in the main thread, only on the workers
        set_global_cuda_envvars(self.cfg)

        tmp_env = make_env_func(self.cfg, env_config=None)
        self.obs_space = tmp_env.observation_space
        self.action_space = tmp_env.action_space
        self.num_agents = tmp_env.num_agents

        self.reward_shaping_scheme = None
        if self.cfg.with_pbt:
            self.reward_shaping_scheme = get_default_reward_shaping(tmp_env)

        tmp_env.close()

        # shared memory allocation
        self.traj_buffers = SharedBuffers(self.cfg, self.num_agents, self.obs_space, self.action_space)

        self.actor_workers = None

        #self.report_queue = MpQueue(40 * 1000 * 1000)
        self.report_queue = MpQueue(400 * 1000 * 1000)
        self.policy_workers = dict()
        self.policy_queues = dict()

        self.learner_workers = dict()

        self.workers_by_handle = None

        self.policy_inputs = [[] for _ in range(self.cfg.num_policies)]
        self.policy_outputs = dict()
        for worker_idx in range(self.cfg.num_workers):
            for split_idx in range(self.cfg.worker_num_splits):
                self.policy_outputs[(worker_idx, split_idx)] = dict()

        self.policy_avg_stats = dict()
        self.policy_lag = [dict() for _ in range(self.cfg.num_policies)]

        self.last_timing = dict()
        self.env_steps = dict()
        self.env_steps_per_level = dict()
        self.samples_collected = [0 for _ in range(self.cfg.num_policies)]
        self.total_env_steps_since_resume = 0
        if self.fps_per_level:
            self.total_env_steps_since_resume_per_level = [0] * self.num_levels

        # currently this applies only to the current run, not experiment as a whole
        # to change this behavior we'd need to save the state of the main loop to a filesystem
        self.total_train_seconds = 0

        self.last_report = time.time()
        self.last_experiment_summaries = 0
        self.last_train_summaries = 0
        self.last_test_summaries = 0

        self.report_interval = self.cfg.report_interval  # sec
        self.experiment_summaries_interval = self.cfg.experiment_summaries_interval  # sec
        self.train_summaries_interval = self.cfg.train_summaries_interval  # sec

        self.avg_stats_intervals = (2, 12, 60)  # 10 seconds, 1 minute, 5 minutes

        self.fps_stats = deque([], maxlen=max(self.avg_stats_intervals))
        self.fps_stats_per_level = deque([], maxlen=max(self.avg_stats_intervals))
        self.throughput_stats = [deque([], maxlen=5) for _ in range(self.cfg.num_policies)]
        self.avg_stats = dict()
        self.stats = dict()  # regular (non-averaged) stats

        self.writers = dict()
        writer_keys = list(range(self.cfg.num_policies))
        self.resource_monitors = list(range(self.cfg.num_policies))
        for key in writer_keys:
            if self.cfg.is_test:
                summary_dir = join(summaries_dir(experiment_dir(cfg=self.cfg), cfg=self.cfg), str(self.cfg.test_policy_id[key]))
            else:
                summary_dir = join(summaries_dir(experiment_dir(cfg=self.cfg), cfg=self.cfg), str(key))
            summary_dir = ensure_dir_exists(summary_dir)
            # if (self.cfg.node_rank == 0 and self.cfg.local_rank == 0) or (self.cfg.local_rank == -1):
            #     writer = SummaryWriter(summary_dir, flush_secs=20)
            # else:
            #     writer = None
            # writer = SummaryWriter(summary_dir, flush_secs=20)
            self.writers[key] = AsyncDistSummaryWrapper(summary_dir, flush_secs=20, key=key, default_scalar_op='avg', default_step_op='sum')
            if hasattr(self.writers[key], 'writer'):
                self.resource_monitors[key] = SummaryWriterThread(self.writers[key].writer, interval_secs=self.cfg.experiment_summaries_interval)
                self.resource_monitors[key].start()

        self.pbt = PopulationBasedTraining(self.cfg, self.reward_shaping_scheme, self.writers)
        self.report_cfg()

        assert self.cfg.use_half_policy_worker or not self.cfg.use_half_learner_worker, (
            "Using half-precisiong only for learner worker is not supported."
        )

    def _cfg_dict(self):
        if isinstance(self.cfg, dict):
            return self.cfg
        else:
            return vars(self.cfg)

    def _save_cfg(self):
        cfg_dict = self._cfg_dict()
        with open(cfg_file(self.cfg, is_test=self.cfg.is_test), 'w') as json_file:
            json.dump(cfg_dict, json_file, indent=2)

    def initialize(self):
        self._save_cfg()
        save_git_diff(experiment_dir(cfg=self.cfg))

    def finalize(self):
        pass

    def create_actor_worker(self, idx, actor_queue):
        learner_queues = {p: w.task_queue for p, w in self.learner_workers.items()}

        return ActorWorker(
            self.cfg, self.obs_space, self.action_space, self.num_agents, idx, self.traj_buffers,
            task_queue=actor_queue, policy_queues=self.policy_queues,
            report_queue=self.report_queue, learner_queues=learner_queues,
        )

    # noinspection PyProtectedMember
    def init_subset(self, indices, actor_queues):
        """
        Initialize a subset of actor workers (rollout workers) and wait until the first reset() is completed for all
        envs on these workers.

        This function will retry if the worker process crashes during the initial reset.

        :param indices: indices of actor workers to initialize
        :param actor_queues: task queues corresponding to these workers
        :return: initialized workers
        """

        reset_timelimit_seconds = self.cfg.reset_timeout_seconds  # fail worker if not a single env was reset in that time

        workers = dict()
        last_env_initialized = dict()
        for i in indices:
            w = self.create_actor_worker(i, actor_queues[i])
            w.init()
            w.request_reset()
            workers[i] = w
            last_env_initialized[i] = time.time()

        total_num_envs = self.cfg.num_workers * self.cfg.num_envs_per_worker
        envs_initialized = [0] * self.cfg.num_workers
        workers_finished = set()

        while len(workers_finished) < len(workers):
            failed_worker = -1

            try:
                report = self.report_queue.get(timeout=1.0)

                if 'initialized_env' in report:
                    worker_idx, split_idx, env_i = report['initialized_env']
                    last_env_initialized[worker_idx] = time.time()
                    envs_initialized[worker_idx] += 1

                    log.debug(
                        'Progress for %d workers: %d/%d envs initialized...',
                        len(indices), sum(envs_initialized), total_num_envs,
                    )
                elif 'finished_reset' in report:
                    workers_finished.add(report['finished_reset'])
                elif 'critical_error' in report:
                    failed_worker = report['critical_error']
            except Empty:
                pass

            for worker_idx, w in workers.items():
                if worker_idx in workers_finished:
                    continue

                time_passed = time.time() - last_env_initialized[worker_idx]
                timeout = time_passed > reset_timelimit_seconds

                if timeout or failed_worker == worker_idx or not w.process.is_alive():
                    envs_initialized[worker_idx] = 0

                    log.error('Worker %d is stuck or failed (%.3f). Reset!', w.worker_idx, time_passed)
                    log.debug('Status: %r', w.process.is_alive())
                    stuck_worker = w
                    stuck_worker.process.kill()

                    new_worker = self.create_actor_worker(worker_idx, actor_queues[worker_idx])
                    new_worker.init()
                    new_worker.request_reset()

                    last_env_initialized[worker_idx] = time.time()
                    workers[worker_idx] = new_worker
                    del stuck_worker

        return workers.values()

    # noinspection PyUnresolvedReferences
    def init_workers(self):
        """
        Initialize all types of workers and start their worker processes.
        """

        actor_queues = [MpQueue(2 * 1000 * 1000) for _ in range(self.cfg.num_workers)]

        policy_worker_queues = dict()
        for policy_id in range(self.cfg.num_policies):
            policy_worker_queues[policy_id] = []
            for i in range(self.cfg.policy_workers_per_policy):
                policy_worker_queues[policy_id].append(TorchJoinableQueue())

        log.info('Initializing learners...')
        policy_locks = [multiprocessing.Lock() for _ in range(self.cfg.num_policies)]
        resume_experience_collection_cv = [multiprocessing.Condition() for _ in range(self.cfg.num_policies)]

        learner_idx = 0
        for policy_id in range(self.cfg.num_policies):
            learner_worker = LearnerWorker(
                learner_idx, policy_id, self.cfg, self.obs_space, self.action_space,
                self.report_queue, policy_worker_queues[policy_id], self.traj_buffers,
                policy_locks[policy_id], resume_experience_collection_cv[policy_id],
            )
            learner_worker.start_process()
            learner_worker.init()

            self.learner_workers[policy_id] = learner_worker
            learner_idx += 1

        log.info('Initializing policy workers...')
        for policy_id in range(self.cfg.num_policies):
            self.policy_workers[policy_id] = []

            policy_queue = MpQueue()
            self.policy_queues[policy_id] = policy_queue

            for i in range(self.cfg.policy_workers_per_policy):
                policy_worker = PolicyWorker(
                    i, policy_id, self.cfg, self.obs_space, self.action_space, self.traj_buffers,
                    policy_queue, actor_queues, self.report_queue, policy_worker_queues[policy_id][i],
                    policy_locks[policy_id], resume_experience_collection_cv[policy_id],
                )
                self.policy_workers[policy_id].append(policy_worker)
                policy_worker.start_process()

        log.info('Initializing actors...')

        # We support actor worker initialization in groups, which can be useful for some envs that
        # e.g. crash when too many environments are being initialized in parallel.
        # Currently the limit is not used since it is not required for any envs supported out of the box,
        # so we parallelize initialization as hard as we can.
        # If this is required for your environment, perhaps a better solution would be to use global locks,
        # like FileLock (see doom_gym.py)
        self.actor_workers = []
        max_parallel_init = int(1e9)  # might be useful to limit this for some envs
        worker_indices = list(range(self.cfg.num_workers))
        for i in range(0, self.cfg.num_workers, max_parallel_init):
            workers = self.init_subset(worker_indices[i:i + max_parallel_init], actor_queues)
            self.actor_workers.extend(workers)

    def init_pbt(self):
        if self.cfg.with_pbt:
            self.pbt.init(self.learner_workers, self.actor_workers)

    def finish_initialization(self):
        """Wait until policy workers are fully initialized."""
        for policy_id, workers in self.policy_workers.items():
            for w in workers:
                log.debug('Waiting for policy worker %d-%d to finish initialization...', policy_id, w.worker_idx)
                w.init()
                log.debug('Policy worker %d-%d initialized!', policy_id, w.worker_idx)

    def update_env_steps_actor(self):
        for w in self.actor_workers:
            w.update_env_steps(self.env_steps)

    def process_report(self, report):
        """Process stats from various types of workers."""

        if 'policy_id' in report:
            policy_id = report['policy_id']

            if 'learner_env_steps' in report:
                if policy_id not in self.env_steps_per_level.keys() and self.fps_per_level:
                    self.env_steps_per_level[policy_id] = [0] * self.num_levels
                if policy_id in self.env_steps:
                    delta = report['learner_env_steps'] - self.env_steps[policy_id]
                    self.total_env_steps_since_resume += delta
                    if self.fps_per_level:
                        for task_id, count in enumerate(report['learner_env_steps_per_level']):
                            self.total_env_steps_since_resume_per_level[task_id] += count - self.env_steps_per_level[policy_id][task_id]
                self.env_steps[policy_id] = report['learner_env_steps']
                if self.fps_per_level:
                    for task_id, count in enumerate(report['learner_env_steps_per_level']):
                        self.env_steps_per_level[policy_id][task_id] = count

            if 'episodic' in report:
                s = report['episodic']
                for _, key, value in iterate_recursively(s):
                    if key not in self.policy_avg_stats:
                        self.policy_avg_stats[key] = [deque(maxlen=self.cfg.stats_avg) for _ in range(self.cfg.num_policies)]

                    self.policy_avg_stats[key][policy_id].append(value)

                    if not self.cfg.is_test:
                        for extra_stat_func in EXTRA_EPISODIC_STATS_PROCESSING:
                            extra_stat_func(policy_id, key, value, self.cfg)

                    # enforce extra summary if is_test
                    if self.cfg.is_test:
                        for extra_stat_func in EXTRA_EPISODIC_TEST_STATS_PROCESSING:
                            task_id = extra_stat_func(policy_id, key, value, self.cfg)

                            if task_id is not None:
                                self.test_episodes_collected[task_id][policy_id] += 1

                            for level_idx, level_episode_collected in enumerate(self.test_episodes_collected):
                                all_episode_collected = True
                                for p in level_episode_collected:
                                    all_episode_collected &= p >= self.cfg.test_num_episodes
                                if all_episode_collected:
                                    if level_idx not in self.test_level_completed:
                                        self.test_level_completed.append(level_idx)

                                    # close finished level workers
                                    for worker_idx, w in enumerate(self.actor_workers):
                                        # got stuck if close too many actors. seems related with traj_buffers
                                        if (worker_idx % self.num_levels) == level_idx and w.process.is_alive:
                                            #log.debug('Collected all episodes of task_%d. Terminate actor %d', level_idx, worker_idx)
                                            w.close()

                            if time.time() - self.last_test_summaries > 1:
                                level_remain = list(set(range(self.num_levels)) - set(self.test_level_completed))
                                log.debug('Level remains %r', level_remain)
                                log.debug('Episode collected %r', self.test_episodes_collected)
                                self.last_test_summaries = time.time()

                        if policy_id not in self.test_policy_completed and len(self.env_steps) == self.cfg.num_policies:
                            for extra_summaries_func in EXTRA_TEST_SUMMARIES:
                                policy_completed = extra_summaries_func(policy_id, self.policy_avg_stats, self.env_steps[policy_id],
                                                                        self.writers[policy_id], self.cfg)
                                if policy_completed is not None:
                                    self.test_policy_completed.append(policy_completed)
                                    log.debug('Test policy %d completed. Terminate ...', self.cfg.test_policy_id[policy_completed])
                                    for w in self.policy_workers[policy_completed]:
                                        w.close()
                                    self.learner_workers[policy_completed].close()

            if 'train' in report:
                self.report_train_summaries(report['train'], policy_id)

            if 'internal_attn' in report:
                self.report_internal_summaries(report['internal_attn'], policy_id, name_internal='internal_attn')

            if 'internal_grad_norm' in report:
                self.report_internal_summaries(report['internal_grad_norm'], policy_id, name_internal='internal_grad_norm')

            if 'samples' in report:
                self.samples_collected[policy_id] += report['samples']

            if 'episodic_cnt_rate_per_level' in report:
                self.report_task_summaries(report['episodic_cnt_rate_per_level'], policy_id, 'epiCnt_rate')

            if 'rew_rate_per_level' in report:
                self.report_task_summaries(report['rew_rate_per_level'], policy_id, 'rewarded_rate')

            if 'best_episodic_cnt_rate_per_level' in report:
                self.report_task_summaries(report['best_episodic_cnt_rate_per_level'], policy_id, 'best_epiCnt_rate')

        key_timings = ['times_learner_worker', 'times_actor_worker', 'times_policy_worker']
        for key in key_timings:
            if key in report:
                self.report_time_summaries(report[key], report['policy_id'], key)
                for k, v in report[key].items():
                    if k not in self.avg_stats:
                        self.avg_stats[k] = deque([], maxlen=50)
                    self.avg_stats[k].append(v)

        if 'stats' in report:
            self.stats.update(report['stats'])

    def report(self):
        """
        Called periodically (every X seconds, see report_interval).
        Print experiment stats (FPS, avg rewards) to console and dump TF summaries collected from workers to disk.
        """

        if len(self.env_steps) < self.cfg.num_policies:
            return

        now = time.time()
        self.fps_stats.append((now, self.total_env_steps_since_resume))
        if self.fps_per_level:
            self.fps_stats_per_level.append((now, self.total_env_steps_since_resume_per_level.copy()))
        if len(self.fps_stats) <= 1:
            return

        fps = []
        if self.fps_per_level:
            fps_per_level = []
        for avg_interval in self.avg_stats_intervals:
            past_moment, past_frames = self.fps_stats[max(0, len(self.fps_stats) - 1 - avg_interval)]
            fps.append((self.total_env_steps_since_resume - past_frames) / (now - past_moment))

            if self.fps_per_level:
                past_moment, past_frames = self.fps_stats_per_level[max(0, len(self.fps_stats_per_level) - 1 - avg_interval)]
                _fps_per_level = [0] * self.num_levels
                for level_id, past_frame in enumerate(past_frames):
                    _fps_per_level[level_id] = (self.total_env_steps_since_resume_per_level[level_id] - past_frame) / (now - past_moment)
                fps_per_level.append(_fps_per_level)

        sample_throughput = dict()
        for policy_id in range(self.cfg.num_policies):
            self.throughput_stats[policy_id].append((now, self.samples_collected[policy_id]))
            if len(self.throughput_stats[policy_id]) > 1:
                past_moment, past_samples = self.throughput_stats[policy_id][0]
                sample_throughput[policy_id] = (self.samples_collected[policy_id] - past_samples) / (now - past_moment)
            else:
                sample_throughput[policy_id] = math.nan

        total_env_steps = sum(self.env_steps.values())
        self.print_stats(fps, sample_throughput, total_env_steps)

        if time.time() - self.last_experiment_summaries > self.experiment_summaries_interval and not self.cfg.is_test:
            if self.fps_per_level:
                self.report_experiment_summaries(fps[0], sample_throughput, fps_per_level=fps_per_level[0], env_steps_per_level=self.env_steps_per_level[0])
            else:
                self.report_experiment_summaries(fps[0], sample_throughput)
            self.last_experiment_summaries = time.time()

    def print_stats(self, fps, sample_throughput, total_env_steps):
        fps_str = []
        for interval, fps_value in zip(self.avg_stats_intervals, fps):
            fps_str.append(f'{int(interval * self.report_interval)} sec: {fps_value:.1f}')
        fps_str = f'({", ".join(fps_str)})'

        samples_per_policy = ', '.join([f'{p}: {s:.1f}' for p, s in sample_throughput.items()])

        lag_stats = self.policy_lag[0]
        lag = AttrDict()
        for key in ['min', 'avg', 'max']:
            lag[key] = lag_stats.get(f'version_diff_{key}', -1)
        policy_lag_str = f'min: {lag.min:.1f}, avg: {lag.avg:.1f}, max: {lag.max:.1f}'

        log.debug(
            'Fps is %s. Total num frames: %d. Throughput: %s. Samples: %d. Policy #0 lag: (%s)',
            fps_str, total_env_steps, samples_per_policy, sum(self.samples_collected), policy_lag_str,
        )

        if 'reward' in self.policy_avg_stats:
            policy_reward_stats = []
            for policy_id in range(self.cfg.num_policies):
                reward_stats = self.policy_avg_stats['reward'][policy_id]
                if len(reward_stats) > 0:
                    policy_reward_stats.append((policy_id, f'{np.mean(reward_stats):.3f}'))
            log.debug('Avg episode reward: %r', policy_reward_stats)

    def report_train_summaries(self, stats, policy_id):
        if time.time() - self.last_train_summaries > self.train_summaries_interval and not self.cfg.is_test:
            for key, scalar in stats.items():
                if 'discarded_rollouts' == key:
                    self.writers[policy_id].add_scalar(f'train/{key}', scalar, self.env_steps[policy_id],
                                                       scalar_op='sum')
                elif 'popart_stat' == key:
                    for level_id, level_name in enumerate(self.all_levels):
                        level_name = level_name.replace('contributed/dmlab30/', '')
                        self.writers[policy_id].add_scalar(f'popart/{level_id:02d}_{level_name}_mu', scalar['mu'][level_id], self.env_steps[policy_id], scalar_op='avg')
                        #self.writers[policy_id].add_scalar(f'popart/{level_id:02d}_{level_name}_mu_ema', scalar['mu_ema'][level_id], self.env_steps[policy_id], scalar_op='avg')
                        self.writers[policy_id].add_scalar(f'popart/{level_id:02d}_{level_name}_sigma', scalar['sigma'][level_id], self.env_steps[policy_id], scalar_op='avg')
                        #self.writers[policy_id].add_scalar(f'popart/{level_id:02d}_{level_name}_sigma_ema', scalar['sigma_ema'][level_id], self.env_steps[policy_id], scalar_op='avg')
                elif 'grad_norm' in key:
                    self.writers[policy_id].add_scalar(f'train/{key}', scalar, self.env_steps[policy_id], scalar_op=None)
                else:
                    if 'max' in key:
                        self.writers[policy_id].add_scalar(f'train/{key}', scalar, self.env_steps[policy_id], scalar_op='max')
                    elif 'min' in key:
                        self.writers[policy_id].add_scalar(f'train/{key}', scalar, self.env_steps[policy_id], scalar_op='min')
                    else:
                        self.writers[policy_id].add_scalar(f'train/{key}', scalar, self.env_steps[policy_id])
                if 'version_diff' in key:
                    self.policy_lag[policy_id][key] = scalar
            self.last_train_summaries = time.time()

    def report_time_summaries(self, stats, policy_id, name_times):
        if policy_id in self.env_steps:
            for key, scalar in stats.items():
                self.writers[policy_id].add_scalar(f'{name_times}/{key}', scalar, self.env_steps[policy_id])

    def report_internal_summaries(self, stats, policy_id, name_internal='internal'):
        if policy_id in self.env_steps:
            for key, scalar in stats.items():
                self.writers[policy_id].add_scalar(f'{name_internal}/{key}', scalar, self.env_steps[policy_id], scalar_op=None)

    def report_task_summaries(self, stats, policy_id, key, name_internal='stats_task'):
        if policy_id in self.env_steps:
            for level_id, scalar in stats.items():
                level_name = self.all_levels[level_id]
                level_name = level_name.replace('contributed/dmlab30/', '')
                self.writers[policy_id].add_scalar(f'{name_internal}/{level_id:02d}_{level_name}_{key}', scalar, self.env_steps[policy_id], scalar_op=None)

    def report_experiment_summaries(self, fps, sample_throughput, fps_per_level=None, env_steps_per_level=None):
        memory_mb = memory_consumption_mb()

        default_policy = 0
        for policy_id, env_steps in self.env_steps.items():
            if policy_id == default_policy:
                self.writers[policy_id].add_scalar('0_aux/_fps', fps, env_steps, scalar_op='sum')
                if self.fps_per_level:
                    self.writers[policy_id].add_scalar('fps/000_all', fps, env_steps, scalar_op='sum')

                    for level_id, _fps in enumerate(fps_per_level):
                        level_name = self.all_levels[level_id].replace('contributed/dmlab30/', '')
                        self.writers[policy_id].add_scalar(f'fps/zz_{level_id:02d}_{level_name}', _fps, env_steps, scalar_op='sum')
                        self.writers[policy_id].add_scalar(f'fps/zzz_{level_id:02d}_{level_name}_ratio', _fps/(fps+1e-5), env_steps, scalar_op='avg')

                    for level_id, _env_steps_per_level in enumerate(env_steps_per_level):
                        level_name = self.all_levels[level_id].replace('contributed/dmlab30/', '')
                        self.writers[policy_id].add_scalar(f'fps/{level_id:02d}_{level_name}_env_steps', _env_steps_per_level, env_steps, scalar_op='sum')

                self.writers[policy_id].add_scalar('0_aux/master_process_memory_mb', float(memory_mb), env_steps)
                for key, value in self.avg_stats.items():
                    if len(value) >= value.maxlen or (len(value) > 10 and self.total_train_seconds > 300):
                        self.writers[policy_id].add_scalar(f'stats/{key}', np.mean(value), env_steps)

                for key, value in self.stats.items():
                    self.writers[policy_id].add_scalar(f'stats/{key}', value, env_steps)

            if not math.isnan(sample_throughput[policy_id]):
                self.writers[policy_id].add_scalar('0_aux/_sample_throughput', sample_throughput[policy_id], env_steps, scalar_op='sum')

            for key, stat in self.policy_avg_stats.items():
                if len(stat[policy_id]) >= stat[policy_id].maxlen or (len(stat[policy_id]) > 10 and self.total_train_seconds > 300):
                    stat_value = np.mean(stat[policy_id])
                    writer = self.writers[policy_id]

                    # custom summaries have their own sections in tensorboard
                    if '/' in key:
                        avg_tag = key
                        min_tag = f'{key}_min'
                        max_tag = f'{key}_max'
                    else:
                        avg_tag = f'0_aux/avg_{key}'
                        std_tag = f'0_aux/avg_{key}_std'
                        min_tag = f'0_aux/avg_{key}_min'
                        max_tag = f'0_aux/avg_{key}_max'

                    writer.add_scalar(avg_tag, float(stat_value), env_steps)

                    # for key stats report min/max as well
                    if key in ('reward', 'true_reward', 'len'):
                        writer.add_scalar(min_tag, float(min(stat[policy_id])), env_steps, scalar_op='min')
                        writer.add_scalar(max_tag, float(max(stat[policy_id])), env_steps, scalar_op='max')

                    # report std of episode raw scores
                    if 'raw_score' in key:
                        writer.add_scalar(std_tag, float(np.std(stat[policy_id])), env_steps)

            if not self.cfg.is_test:
                for extra_summaries_func in EXTRA_PER_POLICY_SUMMARIES:
                    extra_summaries_func(policy_id, self.policy_avg_stats, env_steps, self.writers[policy_id], self.cfg)

    def report_cfg(self):
        if not hasattr(self, 'wrote_cfg'):
            if hasattr(self.writers[0], 'writer'):
                text_total = ''
                for key, val in self.cfg.items():
                    text_total = text_total + f'"{key}": {val}  \n'
                self.writers[0].writer.add_text('cfg', text_total, 0)
                self.wrote_cfg = True

    def _should_end_training(self):
        end = len(self.env_steps) > 0 and all(s > self.cfg.train_for_env_steps for s in self.env_steps.values())
        end |= self.total_train_seconds > self.cfg.train_for_seconds

        if self.cfg.benchmark:
            end |= self.total_env_steps_since_resume >= int(2e6)
            end |= sum(self.samples_collected) >= int(1e6)

        if self.cfg.is_test:
            end = len(self.test_policy_completed) == self.cfg.num_policies

        return end

    def run(self):
        """
        This function contains the main loop of the algorithm, as well as initialization/cleanup code.

        :return: ExperimentStatus (SUCCESS, FAILURE, INTERRUPTED). Useful in testing.
        """

        status = ExperimentStatus.SUCCESS

        if os.path.isfile(done_filename(self.cfg)) and not self.cfg.is_test:
            log.warning('Training already finished! Remove "done" file to continue training')
            return status

        self.init_workers()
        self.init_pbt()
        self.finish_initialization()

        log.info('Collecting experience...')

        timing = Timing()
        with timing.timeit('experience'):
            # noinspection PyBroadException
            try:
                while not self._should_end_training():
                    try:
                        reports = self.report_queue.get_many(timeout=0.1)
                        for report in reports:
                            self.process_report(report)
                    except Empty:
                        pass

                    if time.time() - self.last_report > self.report_interval:
                        self.report()

                        now = time.time()
                        self.total_train_seconds += now - self.last_report
                        self.last_report = now

                        #if not self.cfg.is_test:
                        self.update_env_steps_actor()

                    self.pbt.update(self.env_steps, self.policy_avg_stats)

            except Exception:
                log.exception('Exception in driver loop')
                status = ExperimentStatus.FAILURE
            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected in driver loop, exiting...')
                status = ExperimentStatus.INTERRUPTED

        for k, writer in self.writers.items():
            writer.terminate()

        for learner in self.learner_workers.values():
            # timeout is needed here because some environments may crash on KeyboardInterrupt (e.g. VizDoom)
            # Therefore the learner train loop will never do another iteration and will never save the model.
            # This is not an issue with normal exit, e.g. due to desired number of frames reached.
            if not self.cfg.is_test:
                learner.save_model(timeout=5.0)

        all_workers = self.actor_workers
        for workers in self.policy_workers.values():
            all_workers.extend(workers)
        all_workers.extend(self.learner_workers.values())

        child_processes = list_child_processes()

        time.sleep(0.1)
        log.debug('Closing workers...')
        for i, w in enumerate(all_workers):
            w.close()
            time.sleep(0.01)
        for i, w in enumerate(all_workers):
            w.join()
        log.debug('Workers joined!')

        # VizDoom processes often refuse to die for an unidentified reason, so we're force killing them with a hack
        kill_processes(child_processes)

        fps = self.total_env_steps_since_resume / timing.experience
        log.info('Collected %r, FPS: %.1f', self.env_steps, fps)
        log.info('Timing: %s', timing)

        if self._should_end_training() and not self.cfg.is_test:
            with open(done_filename(self.cfg), 'w') as fobj:
                fobj.write(f'{self.env_steps}')


        time.sleep(0.5)
        log.info('Done!')

        return status
