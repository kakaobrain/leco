import math
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist
from sample_factory.algorithms.appo.model_utils import get_obs_shape
import numpy as np
import copy
from torchvision import transforms
from abc import ABC
from dist.dist_utils import DistEnv
from sample_factory.utils.utils import AttrDict
from sample_factory.algorithms.utils.action_distributions import get_action_distribution
from sample_factory.algorithms.utils.pytorch_utils import build_rnn_inputs
from sample_factory.algorithms.appo.model_utils import create_encoder, create_core, normalize_obs_return, nonlinearity
from sample_factory.algorithms.appo.modules.meta_modules import update_params_by_adamw, update_params_by_adam, update_params_by_sgd, update_params_by_rms


class IntrinsicRewardNet(nn.Module):
    def __init__(self, cfg, obs_space, action_space, core_out_size):
        super().__init__()

        self.cfg = cfg
        self.obs_space = obs_space
        self.action_space = action_space

        if self.cfg.separate_int_value:
            self.int_critic_linear = nn.Linear(core_out_size, 1)

        if not self.cfg.is_test and self.cfg.int_rew_norm:
            self.register_buffer("rew_rms_mean", torch.zeros(()))  # (1,1,h,w)
            self.register_buffer("rew_rms_var", torch.ones(()))  # (1,1,h,w)
            self.register_buffer("rew_rms_count", torch.ones(()) * 1e-4)
            self.register_buffer("rewems", torch.zeros(cfg.batch_size // cfg.recurrence))

        self.rewems_triggered = True if cfg.resume or cfg.is_test else False
        self.gamma = self.cfg.int_gamma

        self.is_dmlab = cfg.env.lower().startswith('dmlab')
        self.is_atari = cfg.env.lower().startswith('atari')
        self.is_minigrid = cfg.env.lower().startswith('minigrid')

    def forward_critic(self, core_output):
        assert core_output is not None
        if self.cfg.separate_int_value:
            if self.cfg.encoder_custom == 'atari_atarinet':
                int_values = self.int_critic_linear(self.extra_layer(core_output) + core_output)
            else:
                int_values = self.int_critic_linear(core_output)
        else:
            int_values = torch.zeros(core_output.size(0)).type_as(core_output)
        return int_values

    @torch.no_grad()
    def rew_rms_update(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.rew_rms_mean
        tot_count = self.rew_rms_count + batch_count

        self.rew_rms_mean = self.rew_rms_mean + delta * batch_count / tot_count
        m_a = self.rew_rms_var * (self.rew_rms_count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + torch.square(delta) * self.rew_rms_count * batch_count / tot_count

        self.rew_rms_var = M2 / tot_count
        self.rew_rms_count = tot_count

    @torch.no_grad()
    def rewems_update(self, rews):
        if not self.rewems_triggered:
            self.rewems = rews
            self.rewems_triggered = True
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

    def initialize_weights(self, layer):
        # gain = nn.init.calculate_gain(self.cfg.nonlinearity)
        gain = self.cfg.policy_init_gain

        if self.cfg.policy_initialization == 'orthogonal':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.orthogonal_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(0)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:  # TODO: test for LSTM
                nn.init.orthogonal_(layer.weight_ih, gain=gain)
                nn.init.orthogonal_(layer.weight_hh, gain=gain)
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
            # elif type(layer) == nn.GRU or type(layer) == nn.LSTM:
            #     for name, param in layer.named_parameters():
            #         if 'weight' in name:
            #             nn.init.orthogonal_(param, gain=gain)
            #         elif 'bias' in name:
            #             param.data.fill_(0)
            else:
                pass
        elif self.cfg.policy_initialization == 'xavier_uniform':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.xavier_uniform_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(0)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:
                nn.init.xavier_uniform_(layer.weight_ih, gain=gain)
                nn.init.xavier_uniform_(layer.weight_hh, gain=gain)
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
            # elif type(layer) == nn.GRU or type(layer) == nn.LSTM:
            #     for name, param in layer.named_parameters():
            #         if 'weight' in name:
            #             nn.init.xavier_uniform_(param, gain=gain)
            #         elif 'bias' in name:
            #             param.data.fill_(0)
            else:
                pass

    def _get_obs_norm(self, obs, use_rms=True):
        if use_rms:
            obs_norm = ((obs - self.obs_rms_mean) / torch.sqrt(self.obs_rms_var)).clip(-5, 5)
        else:
            # scale -1 to 1
            obs_norm = (obs - self.cfg.obs_subtract_mean) / self.cfg.obs_scale
        return obs_norm


class RND(ABC):
    def build_rnd(self, cfg, obs_space=None):
        self.is_dmlab = cfg.env.lower().startswith('dmlab')
        self.is_atari = cfg.env.lower().startswith('atari')
        self.is_minigrid = cfg.env.lower().startswith('minigrid')

        if self.is_dmlab:
            from .modules.rnd_models import RNDModel4DMLab
            target_type = cfg.rnd_type.split(',')[0]
            predictor_type = cfg.rnd_type.split(',')[-1]
            rnd = RNDModel4DMLab(cfg.int_hidden_dim,
                                      target_type=target_type,
                                      predictor_type=predictor_type,
                                      use_shared_encoder=cfg.use_shared_rnd)
        elif self.is_atari:
            from .modules.rnd_models import RNDModel
            rnd = RNDModel()
        elif self.is_minigrid:
            from .modules.rnd_models import RNDModel4MiniGrid
            rnd = RNDModel4MiniGrid(cfg, obs_space)

        self.objective = torch.nn.MSELoss(reduction='none')
        self.rnd = rnd

        if not cfg.is_test and cfg.int_obs_norm:
            if self.is_dmlab or self.is_minigrid:
                n_channel = 3
            else:
                n_channel = 1

            rms_shape = get_obs_shape(obs_space)
            rms_shape = [1, n_channel, rms_shape.obs[-2], rms_shape.obs[-1]]
            self.register_buffer("obs_rms_mean", torch.zeros(rms_shape))  # ()
            self.register_buffer("obs_rms_var", torch.ones(rms_shape))  # ()
            self.register_buffer("obs_rms_count", torch.ones(()) * 1e-4)

    @torch.no_grad()
    def obs_rms_update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.obs_rms_mean
        tot_count = self.obs_rms_count + batch_count

        self.obs_rms_mean = self.obs_rms_mean + delta * batch_count / tot_count
        m_a = self.obs_rms_var * (self.obs_rms_count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + torch.square(delta) * self.obs_rms_count * batch_count / tot_count

        self.obs_rms_var = M2 / tot_count
        self.obs_rms_count = tot_count

    def _get_obs_norm(self, obs, use_rms=True):
        if use_rms:
            obs_norm = ((obs - self.obs_rms_mean) / torch.sqrt(self.obs_rms_var)).clip(-5, 5)
        else:
            # scale -1 to 1
            obs_norm = (obs - self.cfg.obs_subtract_mean) / self.cfg.obs_scale
        return obs_norm

    def get_obs(self, obs):
        if not self.is_dmlab:
            obs = obs[:, -1:, :, :].contiguous()
        return obs

    def calc_distil_loss(self, predict_feature, target_feature):
        device = predict_feature.device
        loss = self.objective(predict_feature, target_feature.detach()).mean(-1)
        mask = torch.rand(len(loss)).to(device)
        mask = (mask < self.cfg.int_rnd_update_proportion).type(torch.FloatTensor).to(device)
        return loss, mask

    def get_novelty(self, obs):
        target_feature = self.rnd.target(obs)
        predict_feature = self.rnd.predictor(obs)
        novelty = (target_feature - predict_feature).pow(2).sum(1) / 2
        return novelty, predict_feature, target_feature


class RNDNet(IntrinsicRewardNet, RND):
    def __init__(self, cfg, obs_space, action_space, core_out_size, timing):
        super(RNDNet, self).__init__(cfg, obs_space, action_space, core_out_size)

        if self.cfg.encoder_custom == 'atari_atarinet':
            self.extra_layer = nn.Sequential(
                nn.Linear(core_out_size, core_out_size),
                nn.ReLU()
            )

        self.build_rnd(cfg, obs_space)
        self.train()

    def forward(self, obs_dict, core_output=None, return_int_rewards=True, return_int_values=True):
        int_rewards, int_values, predict_next_feature, target_next_feature = None, None, None, None

        if return_int_rewards:
            if self.is_dmlab:
                # use RGB channels
                obs = copy.deepcopy(obs_dict)['obs'].float()
            else:
                obs = copy.deepcopy(obs_dict)['obs'][:, -1:, :, :].float()

            obs = self._get_obs_norm(obs)
            int_rewards, predict_next_feature, target_next_feature = self.get_novelty(obs)

        if return_int_values:
            int_values = self.forward_critic(core_output)

        return int_rewards, int_values, predict_next_feature, target_next_feature


class BeboldNet(RNDNet):
    def __init__(self, cfg, obs_space, action_space, core_out_size, timing):
        super(BeboldNet, self).__init__(cfg, obs_space, action_space, core_out_size, timing)
        from .modules.cell import CellCounter
        self.counter = CellCounter(cfg)
        self.learnable_hash = self.counter.learnable_hash

    def _get_novelty(self, obs_norm):
        if self.rnd.random_enc:
            obs_norm = self.rnd.encoder(obs_norm).detach()  # random encoder
        target_next_feature = self.rnd.target(obs_norm)
        predict_next_feature = self.rnd.predictor(obs_norm)
        int_rewards = torch.norm(target_next_feature - predict_next_feature, dim=1, p=2)

        return int_rewards, target_next_feature, predict_next_feature

    def forward(self, obs_dict, next_obs_dict, core_output=None,
                episodic_stats=None, is_seq=False, return_int_rewards=True, return_int_values=True):
        output_dict = {}
        obs = copy.deepcopy(obs_dict)['obs']
        int_rewards, int_values, next_target_feature, next_predict_feature = None, None, None, None

        if return_int_rewards:
            if self.is_minigrid:
                obs_norm = self._get_obs_norm(obs, use_rms=False)
            else:
                obs_norm = self._get_obs_norm(obs)
            novelty, target_feature, predict_feature = self._get_novelty(obs_norm)

            next_obs = copy.deepcopy(next_obs_dict)['obs']
            if self.is_minigrid:
                next_obs_norm = self._get_obs_norm(next_obs, use_rms=False)
            else:
                next_obs_norm = self._get_obs_norm(next_obs)
            next_novelty, next_target_feature, next_predict_feature = self._get_novelty(next_obs_norm)

            int_rewards = torch.clamp(next_novelty - self.cfg.int_scale_fac * novelty, min=0)

        if self.cfg.use_episodic_cnt and episodic_stats is not None:
            if self.is_minigrid:
                #obs = copy.deepcopy(obs_dict)['obs'].float()
                #full_obs = copy.deepcopy(obs_dict)['full_obs'].float()
                #result = self.counter.add(full_obs, episodic_stats)
                #agent_pos = copy.deepcopy(obs_dict)['agent_pos'].float()
                #result = self.counter.add(agent_pos, episodic_stats)
                result = self.counter.add(obs, episodic_stats)
            else:
                result = self.counter.add(obs, episodic_stats)
            output_dict['episodic_cnt'] = torch.LongTensor(result.counts_in_this_episode())

        if episodic_stats is None and self.learnable_hash:
            loss = self.counter.cell_rep.calc_loss(obs, output_dict)
            output_dict['aux_loss'] = loss

        if return_int_values:
            int_values = self.forward_critic(core_output)

        return int_rewards, int_values, next_predict_feature, next_target_feature, output_dict


class RIDENet(IntrinsicRewardNet):
    def __init__(self, cfg, obs_space, action_space, core_out_size, timing):
        super(RIDENet, self).__init__(cfg, obs_space, action_space, core_out_size)
        from .modules.ride_models import ForwardDynamicsNet, InverseDynamicsNet
        from .modules.cell import CellCounter

        self.counter = CellCounter(cfg)
        self.learnable_hash = self.counter.learnable_hash

        if self.cfg.encoder_custom == 'atari_atarinet':
            self.extra_layer = nn.Sequential(
                nn.Linear(core_out_size, core_out_size),
                nn.ReLU()
            )

        num_actions = action_space.n
        self.state_embedding_model = create_encoder(cfg, obs_space, timing, action_space.n)
        self.forward_dynamics_model = ForwardDynamicsNet(
            num_actions, cfg.int_hidden_dim, self.state_embedding_model.get_encoder_out_size(),
            self.state_embedding_model.get_encoder_out_size()
        )
        self.inverse_dynamics_model = InverseDynamicsNet(
            num_actions, cfg.int_hidden_dim, self.state_embedding_model.get_encoder_out_size(),
            self.state_embedding_model.get_encoder_out_size()
        )

        self.train()

    def compute_forward_dynamics_loss(self, pred_next_emb, next_emb):
        forward_dynamics_loss = torch.norm(pred_next_emb - next_emb, dim=-1, p=2)
        return forward_dynamics_loss

    def compute_inverse_dynamics_loss(self, pred_actions, true_actions):
        inverse_dynamics_loss = F.nll_loss(
            F.log_softmax(pred_actions, dim=-1),
            target=torch.flatten(true_actions),
            reduction='none')
        return inverse_dynamics_loss

    def forward(self, obs_dict, next_obs_dict, actions, core_output, episodic_stats=None, return_int_values=True, return_int_rewards=True):
        int_loss, int_rewards, int_values = None, None, None
        output_dict = {}
        obs = copy.deepcopy(obs_dict)['obs']

        if self.cfg.use_episodic_cnt and episodic_stats is not None:
            result = self.counter.add(obs, episodic_stats)
            output_dict['episodic_cnt'] = torch.LongTensor(result.counts_in_this_episode())

        if episodic_stats is None and self.learnable_hash:
            loss = self.counter.cell_rep.calc_loss(obs, output_dict)
            output_dict['aux_loss'] = loss

        if return_int_rewards:
            obs_dict_ = normalize_obs_return(obs_dict, self.cfg)
            next_obs_dict_ = normalize_obs_return(next_obs_dict, self.cfg)
            state_emb = self.state_embedding_model(obs_dict_)
            next_state_emb = self.state_embedding_model(next_obs_dict_)
            pred_next_state_emb = self.forward_dynamics_model(state_emb, actions.detach())
            pred_actions = self.inverse_dynamics_model(state_emb, next_state_emb)

            forward_loss = self.cfg.int_loss_cost * self.compute_forward_dynamics_loss(pred_next_state_emb, next_state_emb)
            inverse_loss = self.cfg.int_loss_cost * self.compute_inverse_dynamics_loss(pred_actions, actions.detach())
            int_loss = forward_loss + inverse_loss

            # Proportion of exp used for predictor update
            mask = torch.rand_like(int_loss)
            mask = (mask < self.cfg.int_rnd_update_proportion).type_as(int_loss)
            int_loss = (int_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)

            int_rewards = torch.norm(next_state_emb - state_emb, dim=-1, p=2)

        if return_int_values:
            int_values = self.forward_critic(core_output)

        return int_loss, int_rewards, int_values, output_dict


class AGACNet(IntrinsicRewardNet):
    def __init__(self, cfg, obs_space, action_space, core_out_size, timing):
        super(AGACNet, self).__init__(cfg, obs_space, action_space, core_out_size)

        if self.is_dmlab:
            from .modules.rnd_models import AdvModel4DMLab
            self.adv = AdvModel4DMLab(cfg, obs_space, action_space, timing)
        elif self.is_minigrid:
            from .modules.agac_models import AdvModel4MiniGrid
            self.adv = AdvModel4MiniGrid(cfg, obs_space, action_space, timing)

        self.apply(self.initialize_weights)
        self.train()

        from .modules.cell import CellCounter
        self.counter = CellCounter(cfg)

    def forward(self, obs_dict, actions, rewards, int_rnn_states, dones, action_logits, core_output=None, is_seq=False, episodic_stats=None, return_int_rewards=True, return_int_values=True):
        int_head_outputs = self.adv.forward_head(obs_dict, actions, rewards)
        if self.cfg.use_rnn and is_seq:
            if self.cfg.packed_seq:
                int_head_output_seq, int_rnn_states, int_inverted_select_inds = build_rnn_inputs(
                    int_head_outputs, dones, int_rnn_states, self.cfg.recurrence, True, 1
                )
            else:
                int_head_output_seq = int_head_outputs
                int_rnn_states = int_rnn_states[::self.cfg.recurrence]
            if self.cfg.packed_seq:
                int_core_outputs, int_new_rnn_states = self.adv.forward_core(int_head_output_seq, int_rnn_states, dones, is_seq,
                                                                   inverted_select_inds=int_inverted_select_inds)
            else:
                int_core_outputs, int_new_rnn_states = self.adv.forward_core(int_head_output_seq, int_rnn_states, dones, is_seq)
        else:
            int_core_outputs, int_new_rnn_states = self.adv.forward_core(int_head_outputs, int_rnn_states,
                                                                                dones, is_seq=False)
        int_action_logits = self.adv.forward_tail(int_core_outputs)

        output_dict = {}
        int_rewards, int_values = None, None
        action_distribution = get_action_distribution(self.adv.action_space, action_logits)
        int_action_distribution = get_action_distribution(self.adv.action_space, int_action_logits)
        if return_int_rewards and torch.min(actions).item() >= 0:
            int_rewards = action_distribution.log_prob(actions) - int_action_distribution.log_prob(actions)

        if self.cfg.use_episodic_cnt and episodic_stats is not None:
            if self.is_minigrid:
                obs = copy.deepcopy(obs_dict)['obs'].float()
                result = self.counter.add(obs, episodic_stats)
            else:
                obs = copy.deepcopy(obs_dict)['obs'].float()
                result = self.counter.add(obs, episodic_stats)
            cnts = result.counts_in_this_episode()
            output_dict['episodic_cnt'] = torch.LongTensor(cnts)

        if return_int_values:
            if self.cfg.separate_int_value:
                int_values = self.forward_critic(core_output)
            else:
                int_values = action_distribution.kl_divergence(int_action_distribution)

        output_dict['int_action_logits'] = int_action_logits
        output_dict['int_new_rnn_states'] = int_new_rnn_states

        return int_rewards, int_values, output_dict


class LinearScheduledCoef:
    def __init__(self, start_factor, end_factor, total_iters, start_step=0, step_size=1):
        self.coef = start_factor
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
        self.step_size = step_size
        self.step_count = 0
        self._step_count = 0
        self.start_step = start_step
        self._w = (self.end_factor - self.start_factor) / (self.total_iters - self.start_step)

        print(f"from {start_factor} to {end_factor} within {start_step} to {total_iters} steps, w: {self._w}, step_size: {step_size}")

    def step(self):
        if self.start_step < self._step_count:
            self.step_count += self.step_size
        self._step_count += self.step_size

    def get_coef(self):
        if self.step_count > (self.total_iters - self.start_step):
            return self.end_factor
        return self.step_count * self._w + self.start_factor


class LirPG(RND):
    ACTIV_CLASS = {
        'sigm': nn.Sigmoid,
        'tanh': nn.Tanh
    }

    UPDATE_METHODS = {
        'adamw': update_params_by_adamw,
        'adam': update_params_by_adam,
        'sgd': update_params_by_sgd,
        'rms': update_params_by_rms
    }

    def build_lirpg(self, cfg, obs_space, action_space):
        self.cfg = cfg
        self.is_dmlab = cfg.env.lower().startswith('dmlab')
        self.is_minigrid = cfg.env.lower().startswith('minigrid')
        self.num_layers = 1

        self.use_code_reps = 'code' in cfg.cell_learning
        self.detach_reps = 'det' in cfg.cell_learning
        self.use_prev_act = 'pa' in cfg.cell_learning
        self.use_cur_act = 'ca' in cfg.cell_learning
        self.use_cur_obs = self.use_prev_act or self.use_cur_act
        self.use_naive_combination = 'naive' in cfg.cell_learning

        activ_cls, self.update_method, _ = self.parse_spec(cfg.lirpg_spec)

        dummy_input = torch.rand((1,) + obs_space['obs'].shape)
        _, (code_reps, enc_reps) = self.counter.obs2cell(dummy_input, to_numpy=False, return_reps=self.cfg.return_reps)
        reps = enc_reps
        if self.use_code_reps:
            reps = code_reps
        kernel_size = reps.shape[2:]
        self.cell_enc_for_rew = self.build_cell_encoder(kernel_size)
        self.cell_enc_for_critic = self.build_cell_encoder(kernel_size, for_critic=True)

        rew_input_size = cfg.cell_enc_hidden
        if self.use_cur_act or self.use_prev_act:
            rew_input_size += action_space.n
        critic_input_size = cfg.cell_enc_hidden
        if self.use_naive_combination:
            rew_input_size += 1
        self.int_rew_predictor = self.build_intrinsic_reward_net(rew_input_size, action_space.n, activ_cls=activ_cls)
        self.int_critic_linear = self.build_critic_net(critic_input_size)

    def update_params(self, params, grads, optim):
        return self.UPDATE_METHODS[self.update_method](params, grads, optim, lr=self.cfg.learning_rate)

    def build_intrinsic_reward_net(self, input_size, output_size, activ_cls='tanh'):
        layers = []
        for i in range(self.num_layers):
            is_last = i+1 == self.num_layers
            if is_last:
                layers.append(nn.Linear(input_size, output_size))
            else:
                layers.append(nn.Linear(input_size, self.cfg.hidden_size))
                layers.append(nonlinearity(self.cfg))
                input_size = self.cfg.hidden_size

        layers.append(self.ACTIV_CLASS[activ_cls]())
        net = nn.Sequential(*layers)
        return net

    def build_critic_net(self, input_size):
        layers = []
        for i in range(self.num_layers):
            is_last = i+1 == self.num_layers
            if is_last:
                layers.append(nn.Linear(input_size, 1))
            else:
                layers.append(nn.Linear(input_size, self.cfg.hidden_size))
                layers.append(nonlinearity(self.cfg))
                input_size = self.cfg.hidden_size
        net = nn.Sequential(*layers)
        return net

    def parse_spec(self, lirpg_spec):
        items = lirpg_spec.split(',')
        activ_cls = items[0]
        update_method = items[1]
        if len(items) == 3:
            num_critic_layers = int(items[2])
        else:
            num_critic_layers = 1

        return activ_cls, update_method, num_critic_layers

    def build_cell_encoder(self, kernel_size, for_critic=False):
        in_channels = self.counter.cell_dim

        layers = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.cfg.cell_enc_hidden,
                kernel_size=kernel_size,
                stride=1),
            nn.LeakyReLU(),
            nn.Flatten()
        ]

        cell_encoder = nn.Sequential(*layers)

        return cell_encoder

    def initialize_weights(self, layer):
        gain = self.cfg.policy_init_gain
        if self.cfg.policy_initialization == 'orthogonal':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.orthogonal_(layer.weight.data, gain=gain)
                layer.bias.data.fill_(0)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:  # TODO: test for LSTM
                nn.init.orthogonal_(layer.weight_ih, gain=gain)
                nn.init.orthogonal_(layer.weight_hh, gain=gain)
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
            else:
                pass


class CellCounterWrapper(nn.Module, LirPG):
    def __init__(self, cfg, obs_space, action_space, core_out_size, timing):
        super().__init__()
        self.cfg = cfg
        from .modules.cell import CellCounter
        self.counter = CellCounter(cfg, action_space)
        self.action_space = action_space

        self.learnable_cell = 'lirpg' in self.cfg.cell_learning

        if self.learnable_cell:
            self.build_lirpg(cfg, obs_space, action_space)

        self.learnable_hash = self.cfg.cell_type.startswith('vq') or self.cfg.cell_type.startswith('ae')

        if self.cfg.int_init:
            self.apply(self.initialize_weights)

    def get_representations(self, cells):
        reps = self.counter.get_representations(cells)
        reps = reps.permute(0, 3, 1, 2).contiguous()
        reps = self.cell_encoder(reps)
        return reps

    def get_cell_representation(self, obs_dict, actions, rewards, instr):
        if type(obs_dict) == AttrDict or type(obs_dict) == dict:
            obs = copy.deepcopy(obs_dict)['obs']
        else:
            obs = obs_dict
        ext_data = None
        cells, reps = self.counter.obs2cell(obs, to_numpy=False, ext_data=ext_data, return_reps=True)
        reps = None
        return reps, cells

    def get_ext_data(self, actions, rewards, instr):
        if not self.augmented_action and not self.augmented_instr:
            return None

        ext_data = []
        if self.augmented_action:
            done_ids = actions.eq(-1).nonzero(as_tuple=False)
            actions[done_ids] = 0
            prev_actions = torch.nn.functional.one_hot(actions, self.action_space.n).to(rewards.dtype)
            prev_actions[done_ids] = 0.
            ext_data.append(prev_actions)

        if self.augmented_reward:
            ext_data.append(rewards.clamp(-1, 1).unsqueeze(1))

        if self.augmented_instr:
            assert instr is not None
            ext_data.append(instr)

        ext_data = torch.cat(ext_data, dim=1)

        return ext_data

    def get_act_embedding(self, actions):
        done_ids = actions.eq(-1).nonzero(as_tuple=False)
        actions[done_ids] = 0
        ret = torch.nn.functional.one_hot(actions, self.action_space.n)
        ret[done_ids] = 0
        return ret

    def extend_cell_reps(self, cell_reps, actions):
        act_embedding = self.get_act_embedding(actions)
        w, h = cell_reps.shape[-2:]
        act_embedding = act_embedding[:, :, None, None]
        act_embedding = torch.tile(act_embedding, (1, 1, w, h))
        cell_reps = torch.cat([cell_reps, act_embedding], dim=1)
        return cell_reps

    def forward(self, obs_dict, core_outputs, rewards, episodic_stats=None, actions=None, prev_actions=None, input_int_rewards=None):
        int_rewards = None
        int_values = None
        output_dict = {}

        ext_data = None

        if episodic_stats is not None:
            obs = copy.deepcopy(obs_dict)['obs']

            result = self.counter.add(obs, episodic_stats)
            if self.cfg.extended_input_cell or self.cfg.is_test:
                output_dict['cells'] = torch.LongTensor(result.cells)

            episodic_count = torch.LongTensor(result.counts_in_this_episode())
            output_dict["episodic_cnt"] = episodic_count

            int_rewards = 1 / torch.sqrt((episodic_count.float()))
        else:
            if self.learnable_hash or self.learnable_cell:
                obs = copy.deepcopy(obs_dict)['obs']

            if self.learnable_cell:
                _, (code_reps, enc_reps) = self.counter.obs2cell(obs, to_numpy=False, return_reps=self.cfg.return_reps)
                reps = enc_reps if not self.use_code_reps else code_reps
                if self.detach_reps or self.use_naive_combination:
                    reps = reps.detach()

                ext_reps = self.cell_enc_for_rew(reps)
                if self.use_prev_act:
                    act_reps = self.get_act_embedding(prev_actions)
                    ext_reps = torch.cat([ext_reps, act_reps], dim=-1)
                elif self.use_cur_act:
                    act_reps = self.get_act_embedding(actions)
                    actions = prev_actions
                    ext_reps = torch.cat([ext_reps, act_reps], dim=-1)

                if self.use_naive_combination:
                    ext_reps = torch.cat([ext_reps, input_int_rewards.unsqueeze(-1)], dim=-1)

                # reward prediction
                int_rewards = self.int_rew_predictor(ext_reps)
                done_ids = actions.eq(-1).nonzero(as_tuple=False)
                actions[done_ids] = 0
                int_rewards = int_rewards.gather(1, actions.unsqueeze(-1)).squeeze()
                output_dict['done_ids'] = done_ids

                # ex_value prediction
                int_values = self.int_critic_linear(self.cell_enc_for_critic(reps))
            else:
                int_values = torch.zeros_like(rewards)

            if self.learnable_hash:
                loss = self.counter.calc_loss(obs, output_dict, ext_data=ext_data)
                output_dict['aux_loss'] = loss

        return int_rewards, int_values, output_dict


class LirPGNet(nn.Module, LirPG):
    def __init__(self, cfg, obs_space, action_space, core_out_size, timing):
        super().__init__()
        self.cfg = cfg
        self.action_space = action_space

        activ_cls, self.update_method, num_critic_layers = self.parse_spec(cfg.lirpg_spec)

        self.int_encoder = create_encoder(cfg, obs_space, timing, self.action_space.n)
        encoder_output_size = self.int_encoder.get_encoder_out_size()
        self.predictor = self.build_intrinsic_reward_net(encoder_output_size, self.action_space.n, activ_cls=activ_cls)
        self.int_critic_linear = self.build_critic_net(cfg, encoder_output_size, num_layers=num_critic_layers)

    def forward(self, obs_dict, core_outputs, actions, episodic_stats=None):
        int_rewards = None
        int_values = None
        output_dict = {}

        if episodic_stats is not None:
            pass
        else:
            obs_dict = normalize_obs_return(obs_dict, self.cfg)  # before normalize, [0,255]
            x = self.int_encoder(obs_dict)
            int_rewards = self.predictor(x)
            int_values = self.int_critic_linear(x)
            done_ids = actions.eq(-1).nonzero(as_tuple=False)
            actions[done_ids] = 0
            int_rewards = int_rewards.gather(1, actions.unsqueeze(-1)).squeeze()
            int_rewards[done_ids] = 0.
            output_dict['done_ids'] = done_ids

        return int_rewards, int_values, output_dict


class IntrinsicModule:
    TYPE = {
        'rnd': RNDNet,
        'ride': RIDENet,
        'bebold': BeboldNet,
        'agac': AGACNet,
        'cell': CellCounterWrapper,
        'lirpg': LirPGNet,
    }

    def __init__(self, cfg, obs_space, action_space, core_out_size, timing=None, distenv=None, use_shared_memory=False):
        self.cfg = cfg
        self.obs_space = obs_space
        self.action_space = action_space
        self.core_out_size = core_out_size
        self.timing = timing
        self.distenv = distenv
        self.group = None
        self.world_size = None

        if self.cfg.use_intrinsic and self.cfg.int_type in ['rnd', 'bebold', 'made']:
            self.int_pre_obs_norm_cnt = 0

        self.unwrapped = self._create(self.cfg.int_type, use_shared_memory=use_shared_memory)

        self._prepare_obs_norm_cond = self.cfg.int_type in ['rnd', 'bebold', 'made'] and self.cfg.int_obs_norm
        self._prepare_obs_norm_cond = self._prepare_obs_norm_cond and (not self.cfg.resume) and (not self.cfg.is_test)

    def __getattr__(self, e):
        return getattr(self.unwrapped, e)

    def get_int_coef(self):
        coef = self.int_coef_scheduler.get_coef()
        self.int_coef_scheduler.step()
        return coef

    @property
    def use_lirpg(self):
        cond = self.cfg.int_type in ['cell'] and 'lirpg' in self.cfg.cell_learning
        cond = cond or self.cfg.int_type.startswith('lirpg')
        return cond

    def set_dist(self, distenv: DistEnv):
        self.distenv = distenv

        n_gpu = self.cfg.nproc_per_node
        nnodes = distenv.world_size // n_gpu
        my_rank = math.floor(self.distenv.world_rank / n_gpu)
        backend = self.cfg.dist_backend

        if n_gpu == 1 and nnodes == 1:
            return

        def make_pg(ranks):
            return dist.new_group(ranks, backend=backend), len(ranks)

        global_ranks = list(range(distenv.world_size))
        int_group_list = []
        obs_group_list = []
        int_prepared, obs_prepared = False, False
        '''
        dist.new_group function requires that all processes in the main group (i.e. all
        processes that are part of the distributed job) enter this function, even
        if they are not going to be members of the group.
        '''
        for node_rank in range(nnodes):
            local_ranks = [node_rank * n_gpu + i for i in range(n_gpu)]
            _int_ranks = local_ranks if self.cfg.use_approx_norm_int else global_ranks
            _obs_ranks = local_ranks if self.cfg.use_approx_norm_obs else global_ranks

            if not int_prepared:
                _int_pg, _int_world_size = make_pg(_int_ranks)
                int_group_list.append(_int_pg)
                if node_rank == my_rank or not self.cfg.use_approx_norm_int:
                    int_ranks = _int_ranks
                    int_world_size = _int_world_size
                    int_pg = _int_pg
                    if not self.cfg.use_approx_norm_int:
                        int_prepared = True

            if not obs_prepared:
                _obs_pg, _obs_world_size = make_pg(_obs_ranks)
                obs_group_list.append(_obs_pg)
                if node_rank == my_rank or not self.cfg.use_approx_norm_obs:
                    obs_ranks = _obs_ranks
                    obs_world_size = _obs_world_size
                    obs_pg = _obs_pg
                    if not self.cfg.use_approx_norm_obs:
                        obs_prepared = True

        print(f'int_world_size:{int_world_size}, int_ranks:{int_ranks}')
        print(f'obs_world_size:{obs_world_size}, obs_ranks:{obs_ranks}')

        def get_group(mode='obs'):
            if mode == 'obs':
                return obs_pg
            return int_pg

        def get_world_size(mode='obs'):
            if mode == 'obs':
                return obs_world_size
            return int_world_size

        self.group = get_group
        self.world_size = get_world_size

        # gathering data that belongs to it's group
        tensor_ranks = torch.Tensor([self.distenv.world_rank]).cuda()
        _ranks = self.all_gather(tensor_ranks)
        print(_ranks)

    def _create(self, int_type, use_shared_memory=False):
        assert int_type in self.TYPE
        int_module = self.TYPE[int_type](self.cfg, self.obs_space, self.action_space, self.core_out_size, self.timing)
        if use_shared_memory:
            int_module.share_memory()
        return int_module

    def get_episodic_count(self, cnts, reverse=False):
        if reverse:
            return torch.where(cnts > self.cfg.int_n_neighbors, torch.ones_like(cnts), torch.zeros_like(cnts))
        return torch.where(cnts <= (self.cfg.int_n_neighbors + 1), torch.ones_like(cnts), torch.zeros_like(cnts))

    def forward(self, mb, core_outputs, instr):
        output_dict = {}

        task_ids_splits = torch.split(mb.task_idx.squeeze(), self.cfg.rollout)
        task_ids = [_task_ids[0].int().item() for _task_ids in task_ids_splits]

        int_loss = 0
        with self.timing.timeit('int_reward'):
            if self.cfg.int_type == 'rnd':
                int_rewards, int_values, predict_next_feature, target_next_feature = self.unwrapped(mb.next_obs, core_outputs)
            elif self.cfg.int_type == 'ride':
                int_loss, int_rewards, int_values, output_dict = self.unwrapped(mb.obs, mb.next_obs, mb.actions.long(), core_outputs)
                episodic_count = mb.episodic_cnt.squeeze()
                output_dict['episodic_count'] = self.get_episodic_count(episodic_count)
                with torch.no_grad():
                    int_rewards[self.cfg.recurrence - 1::self.cfg.recurrence] = 0.
                    int_rewards *= 1 / torch.sqrt(episodic_count.roll(-1).float())

            elif self.cfg.int_type == 'bebold':
                int_rewards, int_values, predict_next_feature, target_next_feature, output_dict = self.unwrapped(mb.obs, mb.next_obs, core_outputs, is_seq=True)
                output_dict['int_rewards_only'] = int_rewards.clone().detach().mean()
                if self.cfg.use_episodic_cnt:
                    if self.cfg.use_indicator:
                        output_dict['episodic_count'] = self.get_episodic_count(mb.episodic_cnt.squeeze())
                    else:
                        output_dict['episodic_count'] = 1 / torch.sqrt((mb.episodic_cnt.squeeze().float()))

                    with torch.no_grad():
                        int_rewards[self.cfg.recurrence - 1::self.cfg.recurrence] = 0.
                        #output_dict['int_rewards_only'] = int_rewards.clone().detach().mean()
                        _episodic_count = output_dict['episodic_count']
                        int_rewards *= _episodic_count.roll(-1).float()

            elif self.cfg.int_type == 'cell':
                _obs = mb.next_obs
                if self.unwrapped.learnable_cell:
                    if self.unwrapped.use_cur_obs:
                        _obs = mb.obs
                int_rewards = mb.int_rewards.squeeze()
                int_rewards = int_rewards.roll(-1)
                int_rewards[self.cfg.recurrence - 1::self.cfg.recurrence] = 0.
                pred_int_rewards, int_values, output_dict = self.unwrapped(_obs, core_outputs, mb.prev_rewards.squeeze(),
                                                                           actions=mb.actions.squeeze().long(),
                                                                           prev_actions=mb.prev_actions.squeeze().long(),
                                                                           input_int_rewards=int_rewards)

                if 'state_novelty' in output_dict:
                    int_rewards = output_dict['state_novelty'].squeeze()

                output_dict['episodic_count'] = self.get_episodic_count(mb.episodic_cnt.squeeze())

                if 'done_ids' in output_dict:
                    int_rewards[output_dict['done_ids']] = 0.

                if self.unwrapped.learnable_cell:
                    output_dict['int_rewards_only'] = pred_int_rewards.clone().detach().mean()
                    if self.unwrapped.use_naive_combination:
                        int_rewards = self.cfg.int_coef_count * pred_int_rewards
                    else:
                        int_rewards = self.cfg.int_coef_count * int_rewards + self.cfg.int_coef_lirpg * pred_int_rewards
                        if self.cfg.rescale_leco_reward:
                            scale_factor = 1/(self.cfg.int_coef_count + self.cfg.int_coef_lirpg)
                            int_rewards *= scale_factor
            elif self.cfg.int_type == 'agac':
                _, _, output_dict = self.unwrapped(mb.obs, mb.actions, mb.rewards,
                                                                      mb.int_rnn_states, mb.dones_cpu, mb.action_logits, core_outputs, is_seq=True, return_int_rewards=False, return_int_values=False)

                old_action_distribution = get_action_distribution(self.action_space, mb.action_logits)
                old_int_action_distribution = get_action_distribution(self.action_space, mb.int_action_logits)
                int_rewards = mb.log_prob_actions - old_int_action_distribution.log_prob(mb.actions)
                int_values = old_action_distribution.kl_divergence(old_int_action_distribution)

                if self.cfg.use_episodic_cnt:
                    episodic_cnt = mb.episodic_cnt.squeeze()
                    output_dict['episodic_count'] = self.get_episodic_count(episodic_cnt)
                    state_count_rewards = torch.floor(1.0 / torch.sqrt(episodic_cnt.roll(-1)))
                    state_count_rewards[self.cfg.recurrence - 1::self.cfg.recurrence] = 0.
                    rewards_ep = mb.rewards + self.cfg.int_agac_count_c * state_count_rewards
                    output_dict['rewards_ep'] = rewards_ep
            elif self.cfg.int_type.startswith('lirpg'):
                int_rewards, int_values, output_dict = self.unwrapped(mb.obs, core_outputs, mb.prev_actions.squeeze().long())
                int_rewards[self.cfg.recurrence - 1::self.cfg.recurrence] = 0.

        if self.cfg.int_type in ['cell', 'bebold', 'ride']:
            episodic_count = output_dict['episodic_count']
            output_dict['episodic_count_rate'] = episodic_count.sum() / episodic_count.shape[0]
            output_dict['state_novelty'] = mb.int_rewards
        elif self.cfg.int_type =='agac':
            episodic_count = output_dict['episodic_count']
            output_dict['episodic_count_rate'] = episodic_count.sum() / episodic_count.shape[0]
        output_dict['int_rewards_before_norm'] = int_rewards.clone().detach()
        int_rewards = self.normalize_int_rew(int_rewards)
        int_values = int_values.squeeze()

        self.update_obs_norm_factor(mb.next_obs['obs'].clone())

        if self.cfg.int_type in ['rnd', 'bebold', 'made']:
            int_loss, mask = self.unwrapped.calc_distil_loss(predict_next_feature, target_next_feature)
            int_loss = (int_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(int_loss.device))
        elif self.cfg.int_type == 'agac':
            int_action_distribution = get_action_distribution(self.action_space, output_dict['int_action_logits'])
            int_loss = old_action_distribution.kl_divergence(int_action_distribution)
            if self.cfg.loss_type == 'mean':
                int_loss = int_loss.mean()
            elif self.cfg.loss_type == 'sum':
                int_loss = int_loss.reshape([-1, self.cfg.recurrence])
                int_loss = int_loss.sum(dim=1)
                int_loss = int_loss.mean()

        aux_loss = 0
        if 'aux_loss' in output_dict:
            aux_loss = output_dict['aux_loss']
            if self.cfg.cell_type.startswith('vq') or self.cfg.cell_type.startswith('ae'):
                aux_loss *= self.cfg.cell_rep_cost

        if 'rnd_loss' in output_dict:
            aux_loss += self.cfg.rnd_loss_coef * output_dict['rnd_loss'].mean()

        int_loss = int_loss + aux_loss
        output_dict['int_loss'] = int_loss

        return int_rewards, int_values, AttrDict(output_dict)

    def forward_policy(self, obs, next_obs, actions, rewards, int_rnn_states, dones, core_output, action_logits, instr, is_seq, episodic_stats=None,
                       return_int_rewards=False, return_int_values=False):
        int_rewards, int_values, output_dict = None, None, {}
        if self.cfg.int_type == 'rnd':
            int_rewards, int_values, _, _ = self.unwrapped(obs, core_output, return_int_rewards=return_int_rewards, return_int_values=return_int_values)
        elif self.cfg.int_type == 'ride':
            _, int_rewards, int_values, output_dict = self.unwrapped(obs, next_obs, actions, core_output, episodic_stats=episodic_stats, return_int_rewards=return_int_rewards, return_int_values=return_int_values)
        elif self.cfg.int_type == 'bebold':
            int_rewards, int_values, _, _, output_dict = self.unwrapped(
                obs, next_obs, core_output, episodic_stats=episodic_stats, return_int_rewards=return_int_rewards, return_int_values=return_int_values)
        elif self.cfg.int_type == 'cell':
            int_rewards, int_values, output_dict = self.unwrapped(obs, core_output, rewards, episodic_stats=episodic_stats)
        elif self.cfg.int_type == 'agac':
            int_rewards, int_values, output_dict = self.unwrapped(
                obs, actions, rewards, int_rnn_states, dones, action_logits, core_output, is_seq, episodic_stats=episodic_stats, return_int_rewards=return_int_rewards, return_int_values=return_int_values)
        elif self.cfg.int_type.startswith('lirpg'):
            pass

        return int_rewards, int_values, output_dict

    def define_advantage(self, ext_adv, int_adv, int_coef):
        ext_adv *= self.cfg.ext_coef_rnd
        int_adv *= int_coef
        return ext_adv, int_adv

    def has_value_head(self):
        return hasattr(self.unwrapped, 'int_critic_linear')

    def should_prepare_obs_norm(self):
        return self._prepare_obs_norm_cond and self.int_pre_obs_norm_cnt < self.cfg.int_pre_obs_norm_steps

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def calc_grad_norm(self):
        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.parameters() if p.grad is not None)
        return grad_norm

    def prepare_obs_norm(self, next_obs):
        next_obs = self.unwrapped.get_obs(next_obs)
        next_obs = self.all_gather(next_obs)
        self.unwrapped.obs_rms_update(next_obs)
        self.int_pre_obs_norm_cnt += 1

    def normalize_int_rew(self, int_rewards):
        if self.cfg.int_rew_norm and self.cfg.int_type in ['rnd', 'bebold', 'made']:
            with torch.no_grad():
                self.update_int_rew_norm_factor(int_rewards)
            int_rewards /= torch.sqrt(self.unwrapped.rew_rms_var.detach())
        return int_rewards

    def update_int_rew_norm_factor(self, int_rewards):
        # running mean intrinsic reward
        total_int_reward = int_rewards.view(-1, self.cfg.recurrence)
        total_int_reward = self.all_gather(total_int_reward, mode='int')
        # print(f"int-{self.distenv.local_rank}:", total_int_reward.sum())
        total_reward_per_env = \
            torch.stack([self.unwrapped.rewems_update(reward_per_step) for reward_per_step in total_int_reward.T])
        mean, std, count = torch.mean(total_reward_per_env), torch.std(total_reward_per_env), len(total_reward_per_env)
        self.unwrapped.rew_rms_update(mean, std ** 2, count)

    def update_obs_norm_factor(self, next_obs):
        if self.cfg.int_obs_norm and self.cfg.int_type in ['rnd', 'bebold', 'made', 'cell']:
            with torch.no_grad():
                next_obs = next_obs.clone()
                next_obs = self.unwrapped.get_obs(next_obs)
                next_obs = self.all_gather(next_obs)
                # print(f"obs-{self.distenv.local_rank}:", next_obs.sum())
                self.unwrapped.obs_rms_update(next_obs)

    def _broadcast_norm_factor(self, data, mode='obs'):
        group = self.group(mode)

        if not self.distenv.local_rank == 0:
            dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
        else:
            data = self.all_gather(data, mode=mode)
            src = self.distenv.world_rank
            if mode == 'obs':
                self.unwrapped.obs_rms_update(data)
                dist.broadcast(self.unwrapped.obs_rms_mean, src, group=group)
                dist.broadcast(self.unwrapped.obs_rms_var, src, group=group)
                dist.broadcast(self.unwrapped.obs_rms_count, src, group=group)
            else:
                total_reward_per_env = \
                    torch.stack([self.unwrapped.rewems_update(reward_per_step) for reward_per_step in data.T])
                mean, std, count = torch.mean(total_reward_per_env), torch.std(total_reward_per_env), len(total_reward_per_env)
                self.unwrapped.rew_rms_update(mean, std ** 2, count)
                dist.broadcast(self.unwrapped.rew_rms_mean, src, group=group)
                dist.broadcast(self.unwrapped.rew_rms_var, src, group=group)
                dist.broadcast(self.unwrapped.rew_rms_count, src, group=group)
            dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
            if mode == 'obs':
                print(f'rank-{self.distenv.local_rank}, obs_rms_count:{self.unwrapped.obs_rms_count}')

        return data

    def all_gather(self, data, mode='obs'):
        if self.world_size is None:
            return data

        world_size = self.world_size(mode)
        group = self.group(mode)

        if world_size == 1:
            return data

        data_list = [torch.zeros_like(data) for _ in range(world_size)]
        dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
        dist.all_gather(data_list, data, group=group)

        return torch.cat(data_list, dim=0)
