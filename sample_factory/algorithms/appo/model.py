from typing import Tuple

import math
import gym
import torch
from torch import nn
from torch.nn import functional as F

from sample_factory.algorithms.appo.model_utils import create_encoder, create_core, ActionParameterizationContinuousNonAdaptiveStddev, \
    ActionParameterizationDefault, normalize_obs_return, nonlinearity
from sample_factory.algorithms.utils.action_distributions import sample_actions_log_probs, is_continuous_action_space, calc_num_actions, calc_num_logits
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import AttrDict, use_lirpg
from sample_factory.algorithms.appo.pbl import PBL
from sample_factory.algorithms.appo.modules.meta_modules import get_child_dict, Linear, Sequential


class CPCA(nn.Module):
    def __init__(self, cfg, action_space):
        super().__init__()
        self.k: int = cfg.cpc_forward_steps
        self.time_subsample: int = cfg.cpc_time_subsample
        self.forward_subsample: int = cfg.cpc_forward_subsample
        self.hidden_size: int = cfg.hidden_size
        self.num_actions: int = calc_num_actions(action_space)
        if isinstance(action_space, gym.spaces.Discrete):
            self.action_sizes = [action_space.n]
        else:
            self.action_sizes = [space.n for space in action_space.spaces]

        self.rnn = nn.GRU(32 * self.num_actions, cfg.hidden_size)
        self.action_embed = nn.Embedding(calc_num_logits(action_space), 32)

        self.predictor = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, 1),
        )
        self.cpc_loss_coeff = cfg.cpc_loss_coeff

    def embed_actions(self, actions):
        embedded = []
        offset = 0
        for i in range(self.num_actions):
            embedded.append(self.action_embed(actions[..., i] + offset).squeeze())
            offset += self.action_sizes[i]

        return torch.cat(embedded, -1)

    def _build_unfolded(self, x, k: int):
        return (
            torch.cat((x, x.new_zeros(x.size(0), k, x.size(2))), 1)
            .unfold(1, size=k, step=1)
            .permute(3, 0, 1, 2)
        )

    def _build_mask_and_subsample(
        self, not_dones
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        t = not_dones.size(1)

        not_dones_unfolded = self._build_unfolded(
            not_dones[:, :-1].to(dtype=torch.bool), self.k
        )

        time_subsample = torch.randperm(
            t - 1, device=not_dones.device, dtype=torch.long
        )[0:self.time_subsample]

        forward_mask = (
            torch.cumprod(not_dones_unfolded.index_select(2, time_subsample), dim=0)
            .to(dtype=torch.bool)
            .flatten(1, 2)
        )

        max_k = forward_mask.flatten(1).any(-1).nonzero().max().item() + 1

        unroll_subsample = torch.randperm(max_k, dtype=torch.long)[0:self.forward_subsample]

        max_k = unroll_subsample.max().item() + 1

        unroll_subsample = unroll_subsample.to(device=not_dones.device)
        forward_mask = forward_mask.index_select(0, unroll_subsample)

        return forward_mask, unroll_subsample, time_subsample, max_k

    def forward(self, actions, not_dones, rnn_inputs, rnn_outputs):
        n = actions.size(0)
        t = actions.size(1)

        mask_res = self._build_mask_and_subsample(not_dones)
        (forward_mask, unroll_subsample, time_subsample, max_k,) = mask_res

        actions = self.embed_actions(actions.long())
        actions_unfolded = self._build_unfolded(actions[:, :-1], max_k).index_select(
            2, time_subsample
        )

        rnn_outputs_subsampled = rnn_outputs[:, :-1].index_select(1, time_subsample)
        forward_preds, _ = self.rnn(
            actions_unfolded.contiguous().flatten(1, 2),
            rnn_outputs_subsampled.contiguous().view(1, -1, self.hidden_size),
        )
        forward_preds = forward_preds.index_select(0, unroll_subsample)
        forward_targets = self._build_unfolded(rnn_inputs[:, 1:], max_k)
        forward_targets = (
            forward_targets.index_select(2, time_subsample)
            .index_select(0, unroll_subsample)
            .flatten(1, 2)
        )

        positives = self.predictor(torch.cat((forward_preds, forward_targets), dim=-1))
        positive_loss = F.binary_cross_entropy_with_logits(
            positives, torch.broadcast_tensors(positives, positives.new_ones(()))[1], reduction="none"
        )
        positive_loss = torch.masked_select(positive_loss, forward_mask).mean()

        forward_negatives = torch.randint(
            0, n * t, size=(self.forward_subsample * self.time_subsample * n * 20,), dtype=torch.long, device=actions.device
        )
        forward_negatives = (
            rnn_inputs.flatten(0, 1)
            .index_select(0, forward_negatives)
            .view(self.forward_subsample, self.time_subsample * n, 20, -1)
        )
        negatives = self.predictor(
            torch.cat(
                (
                    forward_preds.view(self.forward_subsample, self.time_subsample * n, 1, -1)
                        .expand(-1, -1, 20, -1),
                    forward_negatives,
                ),
                dim=-1,
            )
        )
        negative_loss = F.binary_cross_entropy_with_logits(
            negatives, torch.broadcast_tensors(negatives, negatives.new_zeros(()))[1], reduction="none"
        )
        negative_loss = torch.masked_select(
            negative_loss, forward_mask.unsqueeze(2)
        ).mean()

        return self.cpc_loss_coeff * (positive_loss + negative_loss)


class _ActorCriticBase(nn.Module):
    def __init__(self, action_space, cfg, timing):
        super().__init__()
        self.cfg = cfg
        self.action_space = action_space
        self.timing = timing
        self.encoders = []
        self.cores = []

    def get_action_parameterization(self, core_output_size):
        if not self.cfg.adaptive_stddev and is_continuous_action_space(self.action_space):
            action_parameterization = ActionParameterizationContinuousNonAdaptiveStddev(
                self.cfg, core_output_size, self.action_space,
            )
        else:
            action_parameterization = ActionParameterizationDefault(self.cfg, core_output_size, self.action_space)

        return action_parameterization

    def model_to_device(self, device):
        self.to(device)
        for e in self.encoders:
            e.model_to_device(device)

    def device_and_type_for_input_tensor(self, input_tensor_name):
        return self.encoders[0].device_and_type_for_input_tensor(input_tensor_name)

    def initialize_weights(self, layer):
        # gain = nn.init.calculate_gain(self.cfg.nonlinearity)
        gain = self.cfg.policy_init_gain
        if self.cfg.policy_initialization == 'orthogonal':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.orthogonal_(layer.weight.data, gain=gain)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:  # TODO: test for LSTM
                nn.init.orthogonal_(layer.weight_ih, gain=gain)
                nn.init.orthogonal_(layer.weight_hh, gain=gain)
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
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
        elif self.cfg.policy_initialization == 'kaiming':
            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                nn.init.kaiming_normal(layer.weight.data, mode='fan_in')
                layer.bias.data.fill_(0)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:
                nn.init.kaiming_normal(layer.weight_ih, mode='fan_in')
                nn.init.kaiming_normal(layer.weight_hh, mode='fan_in')
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
            else:
                pass

        elif self.cfg.policy_initialization == 'transformer':
            classname = layer.__class__.__name__
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

            if isinstance(layer, nn.LayerNorm):
                nn.init.constant_(layer.bias, 0)
                nn.init.constant_(layer.weight, 1.0)

            if 'TransformerLM' in classname:
                if hasattr(layer, 'r_emb'):
                    nn.init.normal_(layer.r_emb, 0.0, 0.02)
                if hasattr(layer, 'r_w_bias'):
                    nn.init.normal_(layer.r_w_bias, 0.0, 0.02)
                if hasattr(layer, 'r_r_bias'):
                    nn.init.normal_(layer.r_r_bias, 0.0, 0.02)
                if hasattr(layer, 'r_bias'):
                    nn.init.constant_(layer.r_bias, 0.0)

        # def _init_weights(self, m):
        #     if isinstance(m, nn.Linear):
        #         trunc_normal_(m.weight, std=.02)
        #         if isinstance(m, nn.Linear) and m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.LayerNorm):
        #         nn.init.constant_(m.bias, 0)
        #         nn.init.constant_(m.weight, 1.0)

        elif self.cfg.policy_initialization == 'impala':

            def _no_grad_trunc_normal_(tensor, mean, std, a, b):
                # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
                def norm_cdf(x):
                    # Computes standard normal cumulative distribution function
                    return (1 + math.erf(x / math.sqrt(2.))) / 2.

                with torch.no_grad():
                    # Values are generated by using a truncated uniform distribution and
                    # then using the inverse CDF for the normal distribution.
                    # Get upper and lower cdf values
                    l = norm_cdf((a - mean) / std)
                    u = norm_cdf((b - mean) / std)

                    # Uniformly fill tensor with values from [l, u], then translate to
                    # [2l-1, 2u-1].
                    tensor.uniform_(2 * l - 1, 2 * u - 1)

                    # Use inverse cdf transform for normal distribution to get truncated
                    # standard normal
                    tensor.erfinv_()

                    # Transform to proper mean, std
                    tensor.mul_(std * math.sqrt(2.))
                    tensor.add_(mean)

                    # Clamp to ensure it's in the proper range
                    tensor.clamp_(min=a, max=b)
                    return tensor

            def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
                # type: (Tensor, float, float, float, float) -> Tensor
                r"""Fills the input `Tensor` with values drawn from a truncated
                normal distribution. The values are effectively drawn from the
                normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
                with values outside :math:`[a, b]` redrawn until they are within
                the bounds.
                Args:
                    tensor: an n-dimensional `torch.Tensor`
                    mean: the mean of the normal distribution
                    std: the standard deviation of the normal distribution
                    a: the minimum cutoff value
                    b: the maximum cutoff value
                Examples:
                    >>> w = torch.empty(3, 5)
                    >>> nn.init.trunc_normal_(w)
                """
                return _no_grad_trunc_normal_(tensor, mean, std, a, b)

            if type(layer) == nn.Conv2d or type(layer) == nn.Linear:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight.data)
                stddev = 1/math.sqrt(fan_in)
                trunc_normal_(layer.weight.data, mean=0, std=stddev, a=-2*stddev, b=2*stddev)
                layer.bias.data.fill_(0)
            elif type(layer) == nn.GRUCell or type(layer) == nn.LSTMCell:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight_ih.data)
                stddev = 1/math.sqrt(fan_in)
                trunc_normal_(layer.weight_ih.data, mean=0, std=stddev, a=-2*stddev, b=2*stddev)
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight_hh.data)
                stddev = 1/math.sqrt(fan_in)
                trunc_normal_(layer.weight_hh.data, mean=0, std=stddev, a=-2*stddev, b=2*stddev)
                layer.bias_ih.data.fill_(0)
                layer.bias_hh.data.fill_(0)
                import matplotlib.pyplot as plt
                tmp = layer.weight_ih.data.flatten().numpy()
                plt.hist(tmp)
                plt.show()
            else:
                pass

    def weights_init(self, m):
        def init_weight(weight):
            # if args.init == 'uniform':
            #    nn.init.uniform_(weight, -args.init_range, args.init_range)
            # elif args.init == 'normal':
            nn.init.normal_(weight, 0.0, 0.02)  # args.init_std)

        def init_bias(bias):
            nn.init.constant_(bias, 0.0)

        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init_weight(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('AdaptiveEmbedding') != -1:
            if hasattr(m, 'emb_projs'):
                for i in range(len(m.emb_projs)):
                    if m.emb_projs[i] is not None:
                        nn.init.normal_(m.emb_projs[i], 0.0, 0.01)  # args.proj_init_std)
        elif classname.find('Embedding') != -1:
            if hasattr(m, 'weight'):
                init_weight(m.weight)
        elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
            if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
                init_weight(m.cluster_weight)
            if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
                init_bias(m.cluster_bias)
            if hasattr(m, 'out_projs'):
                for i in range(len(m.out_projs)):
                    if m.out_projs[i] is not None:
                        nn.init.normal_(m.out_projs[i], 0.0, 0.01)  # args.proj_init_std)
        elif classname.find('LayerNorm') != -1:
            if hasattr(m, 'weight'):
                nn.init.normal_(m.weight, 1.0, 0.02)  # args.init_std)
            if hasattr(m, 'bias') and m.bias is not None:
                init_bias(m.bias)
        elif classname.find('TransformerLM') != -1:
            # print('FOUND TRNASFORMER LM')
            if hasattr(m, 'r_emb'):
                init_weight(m.r_emb)
            if hasattr(m, 'r_w_bias'):
                init_weight(m.r_w_bias)
            if hasattr(m, 'r_r_bias'):
                init_weight(m.r_r_bias)
            if hasattr(m, 'r_bias'):
                init_bias(m.r_bias)

    def update_mu_sigma(self, vs, task_ids, cfg=None):
        oldnu = self.nu.clone()
        oldsigma = torch.sqrt(oldnu - self.mu ** 2)
        oldsigma[torch.isnan(oldsigma)] = self.cfg.popart_clip_min
        clamp_max = 1e4 if hasattr(self, 'need_half') and self.need_half else 1e6
        oldsigma = torch.clamp(oldsigma, min=cfg.popart_clip_min, max=clamp_max)
        oldmu = self.mu.clone()

        vs = vs.reshape(-1, self.cfg.recurrence)
        task_ids_per_epi = task_ids.reshape(-1, self.cfg.recurrence)[:,0]   # same task ids over all time steps within a single episode
        for i in range(len(task_ids_per_epi)):
            task_id = task_ids_per_epi[i]
            v = torch.mean(vs[i])
            #if self.popart_update_triggered[task_id]:
            #    self.mu[task_id] = (1 - self.beta) * self.mu[task_id] + self.beta * v
            #    self.nu[task_id] = (1 - self.beta) * self.nu[task_id] + self.beta * (v ** 2)
            #else:
            #    self.mu[task_id] = v
            #    self.nu[task_id] = torch.var(vs[i])
            #    self.popart_update_triggered[task_id] = True
            self.mu[task_id] = (1 - self.beta) * self.mu[task_id] + self.beta * v
            self.nu[task_id] = (1 - self.beta) * self.nu[task_id] + self.beta * (v ** 2)

        sigma = torch.sqrt(self.nu - self.mu ** 2)
        sigma[torch.isnan(sigma)] = self.cfg.popart_clip_min
        sigma = torch.clamp(sigma, min=cfg.popart_clip_min, max=clamp_max)

        return self.mu, sigma, oldmu, oldsigma

    def update_parameters(self, mu, sigma, oldmu, oldsigma):
        self.critic_linear.weight.data = (self.critic_linear.weight.t() * oldsigma / sigma).t()
        self.critic_linear.bias.data = (oldsigma * self.critic_linear.bias + oldmu - mu) / sigma

    def _get_extra_critic_mlp(self):
        core_out_size = self.cores[0].get_core_out_size()
        nonlinearity = nn.ELU(inplace=True) if self.cfg.extra_fc_critic_nonlinearity == 'elu' else nn.ReLU(inplace=True)
        input_size = core_out_size
        output_size = core_out_size if self.cfg.extra_fc_critic_hidden_size > 0 else core_out_size

        extra_critic_layers = list()
        for _ in range(self.cfg.extra_fc_critic):
            extra_critic_layers.append(nn.Linear(input_size, output_size))
            extra_critic_layers.append(nonlinearity)
            input_size = output_size
        extra_critic_layers = nn.Sequential(
            *extra_critic_layers
        )
        return extra_critic_layers

    def _get_mlp_dim_match(self, action_space):
        if self.cfg.match_core_input_size:
            # One-layer MLP for dimension matching, (hidden_size + action_dim + 1) -> (hidden_size)
            core_input_size = self.encoders[0].get_encoder_out_size()
            if self.cfg.extended_input:
                core_input_size += action_space.n + 1

            if self.cfg.use_intrinsic and self.cfg.int_type == 'cell' and self.cfg.extended_input_cell:
                core_input_size += self.cfg.cell_dim

            if core_input_size != self.cfg.hidden_size:
                mlp_dim_match = nn.Sequential(
                    nn.Linear(core_input_size, self.cfg.hidden_size),
                    nonlinearity(self.cfg)
                )
            return mlp_dim_match


class _ActorCriticSharedWeights(_ActorCriticBase):
    def __init__(self, make_encoder, make_core, action_space, cfg, timing, is_learner_worker=False):
        super().__init__(action_space, cfg, timing)
        self.is_learner_worker = is_learner_worker
        # in case of shared weights we're using only a single encoder and a single core
        self.encoder = make_encoder(action_space.n)
        # if self.is_learner_worker:
        if self.is_learner_worker and self.cfg.encoder_type == 'resnet_encoder_decoder':
                self.encoder.basic_encoder.set_learner_worker()
        self.encoders = [self.encoder]

        self.mlp_dim_match = self._get_mlp_dim_match(action_space)

        #if cfg.action_reward_inputs
        self.core = make_core(self.encoder, action_space.n)
        self.cores = [self.core]

        core_out_size = self.core.get_core_out_size()
        self.extra_critic_mlp = self._get_extra_critic_mlp()  # extra fc layers for critic: (core_out_size, core_out_size) x n

        if self.cfg.use_reward_prediction:
            action_embedding_size = 32
            self.action_embedding = nn.Embedding(action_space.n, action_embedding_size)
            reward_prediction_input_size = (core_out_size  # core out
                                            + self.encoders[0].get_encoder_out_size()  # next state encoding
                                            + action_embedding_size)  # action

        if self.cfg.use_popart:
            self.register_buffer('mu', torch.zeros(cfg.num_envs, requires_grad=False))
            self.register_buffer('nu', torch.ones(cfg.num_envs, requires_grad=False))
            self.critic_linear = nn.Linear(core_out_size, cfg.num_envs)
            self.beta = self.cfg.popart_beta
            #self.register_buffer('popart_update_triggered', torch.zeros(cfg.num_envs, requires_grad=False).to(torch.bool))
            if self.cfg.use_reward_prediction:
                self.reward_linear = nn.Linear(reward_prediction_input_size, cfg.num_envs)
        else:
            self.critic_linear = nn.Linear(core_out_size, 1)
            if self.cfg.use_reward_prediction:
                self.reward_linear = nn.Linear(reward_prediction_input_size, 1)

        if self.cfg.use_vmpo:
            self.vmpo_eta = torch.nn.parameter.Parameter(torch.tensor(1.0))
            self.vmpo_alpha = torch.nn.parameter.Parameter(torch.tensor(5.0))

        if self.cfg.use_pbl:
            pbl_encoder_f = make_encoder()
            self.pbl = PBL(self.cfg, pbl_encoder_f, core_out_size, action_space)

        self.action_parameterization = self.get_action_parameterization(core_out_size)

        #self.apply(self.initialize_weights)
        self.encoder.apply(self.initialize_weights)
        self.core.apply(self.weights_init)
        self.action_parameterization.apply(self.weights_init)
        self.critic_linear.apply(self.weights_init)
        self.extra_critic_mlp.apply(self.weights_init)
        if not self.mlp_dim_match is None: self.mlp_dim_match.apply(self.weights_init)
        if self.cfg.use_reward_prediction: self.reward_linear.apply(self.weights_init)

        self.train()  # eval() for inference?
        self.need_half = (not self.is_learner_worker and self.cfg.use_half_policy_worker) \
                         or (self.is_learner_worker and self.cfg.use_half_learner_worker)

    def forward_head(self, obs_dict, actions, rewards, **kwargs):
        obs_dict = normalize_obs_return(obs_dict, self.cfg) # before normalize, [0,255]
        if self.need_half:
            obs_dict['obs'] = obs_dict['obs'].half()

        x = self.encoder(obs_dict)
        x_extended = []
        if self.cfg.extended_input:
            # -1 is transformed into all zero vector
            assert torch.min(actions) >= -1 and torch.max(actions) < self.action_space.n
            done_ids = actions.eq(-1).nonzero(as_tuple=False)
            actions[done_ids] = 0
            prev_actions = torch.nn.functional.one_hot(actions, self.action_space.n).float()
            prev_actions[done_ids] = 0.
            x_extended.append(prev_actions)
            x_extended.append(rewards.clamp(-1, 1).unsqueeze(1))

        if kwargs.get('cells', None) is not None:
            x_extended.append(kwargs.get('cells'))

        x = torch.cat([x] + x_extended, dim=-1)

        if self.need_half:
            x = x.half()

        if self.mlp_dim_match is not None:
            return self.mlp_dim_match(x)
        else:
            return x

    def forward_core_rnn(self, head_output, rnn_states, dones, is_seq, inverted_select_inds=None):
        x, new_rnn_states, all_hidden = self.core(head_output, rnn_states, dones, is_seq)
        if inverted_select_inds is not None:
            x = x.data.index_select(0, inverted_select_inds)
        return x, new_rnn_states, all_hidden

    def forward_core_transformer(self, head_output, mems=None, rollout_step_list=None, mem_begin_index=None, dones=None):
        x, new_mems, attn_entropy = self.core(head_output, mems, rollout_step_list, mem_begin_index, dones=dones)
        return x, new_mems, attn_entropy

    def forward_tail(self, core_output, task_ids, with_action_distribution=False):
        values = self.critic_linear(
            self.extra_critic_mlp(core_output)
        )
        normalized_values = values.clone()
        sigmas = torch.ones((values.size(0), 1), requires_grad=False)
        mus = torch.zeros((values.size(0), 1), requires_grad=False)
        if self.cfg.use_popart:
            normalized_values = normalized_values.gather(dim=1, index=task_ids)
            with torch.no_grad():
                nus = self.nu.index_select(dim=0, index=task_ids.squeeze(1)).unsqueeze(1)
                mus = self.mu.index_select(dim=0, index=task_ids.squeeze(1)).unsqueeze(1)
                sigmas = torch.sqrt(nus - mus ** 2)
                sigmas[torch.isnan(sigmas)] = self.cfg.popart_clip_min
                clamp_max = 1e4 if self.need_half else 1e6
                sigmas = torch.clamp(sigmas, min=self.cfg.popart_clip_min, max=clamp_max)
                values = normalized_values * sigmas + mus

        action_distribution_params, action_distribution = self.action_parameterization(core_output)

        # for non-trivial action spaces it is faster to do these together
        actions, log_prob_actions = sample_actions_log_probs(action_distribution)

        result = AttrDict(dict(
            actions=actions,
            action_logits=action_distribution_params,  # perhaps `action_logits` is not the best name here since we now support continuous actions
            log_prob_actions=log_prob_actions,
            values=values,
            normalized_values=normalized_values,
            sigmas=sigmas,
            mus=mus
        ))

        if with_action_distribution:
            result.action_distribution = action_distribution

        return result

    def _slice_obs(self, obs_dict, idx1, idx2, skip=1):
        dict_copy = type(obs_dict)()
        for k, v in obs_dict.items():
            dict_copy[k] = v[idx1:idx2:skip]
        return dict_copy

    def forward_reward_prediction(self, core_out, actions, next_obs_dict, task_ids, head_out=None):
        if head_out is not None:
            next_obs_dict = normalize_obs_return(
                self._slice_obs(next_obs_dict, self.cfg.recurrence-1, core_out.shape[0], skip=self.cfg.recurrence),
                self.cfg
            )
            if self.need_half:
                next_obs_dict['obs'] = next_obs_dict['obs'].half()
            with torch.no_grad():
                x_next = self.encoder(next_obs_dict)
            x_next = torch.cat([
                head_out.view(head_out.shape[0]//self.cfg.recurrence,
                              self.cfg.recurrence, *head_out.shape[1:])[:, 1:, :x_next.shape[1]],
                x_next.unsqueeze(1)],
                dim=1
            ).view(core_out.shape[0], x_next.shape[1])
        else:
            next_obs_dict = normalize_obs_return(next_obs_dict, self.cfg)  # before normalize, [0,255]
            if self.need_half:
                next_obs_dict['obs'] = next_obs_dict['obs'].half()
            with torch.no_grad():
                x_next = self.encoder(next_obs_dict)

        done_ids = actions.eq(-1).nonzero(as_tuple=False)
        actions_clone = actions.clone().int()
        actions_clone[done_ids] = 0
        action_embedding = self.action_embedding(actions_clone)
        action_embedding[done_ids] = 0

        x_reward_prediction = torch.cat([core_out, action_embedding, x_next], dim=1)
        reward_prediction = self.reward_linear(x_reward_prediction)
        if self.cfg.use_popart:
            reward_prediction = reward_prediction.gather(dim=1, index=task_ids)
        return reward_prediction

    def forward(self, obs_dict, actions, rewards, rnn_states=None, mems=None, rollout_step_list=None, mem_begin_index=None, dones=None, task_ids=None, is_seq=None, with_action_distribution=False, is_transformer=False):
        x = self.forward_head(obs_dict, actions, rewards)
        if is_transformer:
            x, new_mems, attn_entropy = self.forward_core_transformer(x, mems, rollout_step_list=rollout_step_list,
                                                          mem_begin_index=mem_begin_index)
        else:
            x, new_rnn_states, _ = self.forward_core_rnn(x, rnn_states, dones, is_seq)

        assert not x.isnan().any()
        result = self.forward_tail(x, task_ids, with_action_distribution=with_action_distribution)
        if is_transformer:
            result.mems = new_mems
            result.attn_entropy = attn_entropy
        else:
            result.rnn_states = new_rnn_states

        return result


class _ActorCriticSharedWeightsMeta(_ActorCriticBase):
    def __init__(self, make_encoder, make_core, action_space, cfg, timing, is_learner_worker=False):
        super().__init__(action_space, cfg, timing)
        self.is_learner_worker = is_learner_worker
        # in case of shared weights we're using only a single encoder and a single core
        self.encoder = make_encoder(action_space.n)
        if self.is_learner_worker and self.cfg.encoder_type == 'resnet_encoder_decoder':
            self.encoder.basic_encoder.set_learner_worker()
        self.encoders = [self.encoder]

        self.mlp_dim_match = self._get_mlp_dim_match(action_space)

        #if cfg.action_reward_inputs
        self.core = make_core(self.encoder, action_space.n)
        self.cores = [self.core]

        core_out_size = self.core.get_core_out_size()
        self.extra_critic_mlp = self._get_extra_critic_mlp()  # extra fc layers for critic: (core_out_size, core_out_size) x n
        if self.cfg.use_popart:
            self.register_buffer('mu', torch.zeros(cfg.num_envs, requires_grad=False))
            self.register_buffer('nu', torch.ones(cfg.num_envs, requires_grad=False))
            self.critic_linear = Linear(core_out_size, cfg.num_envs)
            self.beta = self.cfg.popart_beta
            #self.register_buffer('popart_update_triggered', torch.zeros(cfg.num_envs, requires_grad=False).to(torch.bool))
        else:
            self.critic_linear = Linear(core_out_size, 1)

        from sample_factory.algorithms.appo.modules.meta_modules import ActionParameterizationDefault as ActionParameterizationDefaultMeta
        self.action_parameterization = ActionParameterizationDefaultMeta(cfg, core_out_size, action_space)

        #self.apply(self.initialize_weights)
        self.encoder.apply(self.initialize_weights)
        self.core.apply(self.weights_init)
        self.action_parameterization.apply(self.weights_init)
        self.critic_linear.apply(self.weights_init)
        self.extra_critic_mlp.apply(self.weights_init)
        if not self.mlp_dim_match is None: self.mlp_dim_match.apply(self.weights_init)

        self.train()  # eval() for inference?
        self.need_half = (not self.is_learner_worker and self.cfg.use_half_policy_worker) \
                         or (self.is_learner_worker and self.cfg.use_half_learner_worker)

    def _get_mlp_dim_match(self, action_space):
        if self.cfg.match_core_input_size:
            # One-layer MLP for dimension matching, (hidden_size + action_dim + 1) -> (hidden_size)
            core_input_size = self.encoders[0].get_encoder_out_size()
            if self.cfg.extended_input:
                core_input_size += action_space.n + 1

            if self.cfg.use_intrinsic and self.cfg.int_type == 'cell' and self.cfg.extended_input_cell:
                core_input_size += self.cfg.cell_dim

            if core_input_size != self.cfg.hidden_size:
                mlp_dim_match = Sequential(
                    ('linear', Linear(core_input_size, self.cfg.hidden_size)),
                    ('_', nonlinearity(self.cfg))
                )
            return mlp_dim_match


    def forward_head(self, obs_dict, actions, rewards, params=None, **kwargs):
        obs_dict = normalize_obs_return(obs_dict, self.cfg) # before normalize, [0,255]
        if self.need_half:
            obs_dict['obs'] = obs_dict['obs'].half()

        x = self.encoder(obs_dict, params=get_child_dict(params, 'encoder'))
        x_extended = []
        if self.cfg.extended_input:
            # -1 is transformed into all zero vector
            assert torch.min(actions) >= -1 and torch.max(actions) < self.action_space.n
            done_ids = actions.eq(-1).nonzero(as_tuple=False)
            actions[done_ids] = 0
            prev_actions = torch.nn.functional.one_hot(actions, self.action_space.n).float()
            prev_actions[done_ids] = 0.
            x_extended.append(prev_actions)
            x_extended.append(rewards.clamp(-1, 1).unsqueeze(1))

        if kwargs.get('cells', None) is not None:
            x_extended.append(kwargs.get('cells'))

        x = torch.cat([x] + x_extended, dim=-1)

        if self.need_half:
            x = x.half()

        if self.mlp_dim_match is not None:
            return self.mlp_dim_match(x, params=get_child_dict(params, 'mlp_dim_match'))
        else:
            return x

    def forward_core_rnn(self, head_output, rnn_states, dones, is_seq, inverted_select_inds=None, params=None):
        x, new_rnn_states, all_hidden = self.core(head_output, rnn_states, dones, is_seq, params=get_child_dict(params, 'core'))
        if inverted_select_inds is not None:
            x = x.data.index_select(0, inverted_select_inds)
        return x, new_rnn_states, all_hidden

    def forward_core_transformer(self, head_output, mems=None, rollout_step_list=None, mem_begin_index=None, dones=None, params=None):
        x, new_mems, attn_entropy = self.core(head_output, mems, rollout_step_list, mem_begin_index, dones=dones, params=get_child_dict(params, 'core'))
        return x, new_mems, attn_entropy

    def forward_tail(self, core_output, task_ids, with_action_distribution=False, params=None):
        values = self.critic_linear(
            self.extra_critic_mlp(core_output), params=get_child_dict(params, 'critic_linear')
        )
        normalized_values = values.clone()
        sigmas = torch.ones((values.size(0), 1), requires_grad=False)
        mus = torch.zeros((values.size(0), 1), requires_grad=False)
        if self.cfg.use_popart:
            normalized_values = normalized_values.gather(dim=1, index=task_ids)
            with torch.no_grad():
                nus = self.nu.index_select(dim=0, index=task_ids.squeeze(1)).unsqueeze(1)
                mus = self.mu.index_select(dim=0, index=task_ids.squeeze(1)).unsqueeze(1)
                sigmas = torch.sqrt(nus - mus ** 2)
                sigmas[torch.isnan(sigmas)] = self.cfg.popart_clip_min
                clamp_max = 1e4 if self.need_half else 1e6
                sigmas = torch.clamp(sigmas, min=self.cfg.popart_clip_min, max=clamp_max)
                values = normalized_values * sigmas + mus

        action_distribution_params, action_distribution = self.action_parameterization(core_output, params=get_child_dict(params, 'action_parameterization'))

        # for non-trivial action spaces it is faster to do these together
        actions, log_prob_actions = sample_actions_log_probs(action_distribution)

        result = AttrDict(dict(
            actions=actions,
            action_logits=action_distribution_params,  # perhaps `action_logits` is not the best name here since we now support continuous actions
            log_prob_actions=log_prob_actions,
            values=values,
            normalized_values=normalized_values,
            sigmas=sigmas,
            mus=mus
        ))

        if with_action_distribution:
            result.action_distribution = action_distribution

        return result

    def forward(self, obs_dict, actions, rewards, rnn_states=None, mems=None, rollout_step_list=None, mem_begin_index=None, dones=None, task_ids=None, is_seq=None, with_action_distribution=False, is_transformer=False, params=None):
        x = self.forward_head(obs_dict, actions, rewards, params=params)
        if is_transformer:
            x, new_mems, attn_entropy = self.forward_core_transformer(x, mems, rollout_step_list=rollout_step_list,
                                                          mem_begin_index=mem_begin_index, params=params)
        else:
            x, new_rnn_states, _ = self.forward_core_rnn(x, rnn_states, dones, is_seq)

        assert not x.isnan().any()
        result = self.forward_tail(x, task_ids, with_action_distribution=with_action_distribution, params=params)
        if is_transformer:
            result.mems = new_mems
            result.attn_entropy = attn_entropy
        else:
            result.rnn_states = new_rnn_states

        return result



class _ActorCriticSeparateWeights(_ActorCriticBase):
    def __init__(self, make_encoder, make_core, action_space, cfg, timing):
        super().__init__(action_space, cfg, timing)
        #num_actions = action_space.n if self.cfg.extended_input else -1 # -1 is going to be used for cancel out the dimension of reward (1): No extended input of action and reward
        num_actions = action_space.n

        self.actor_encoder = make_encoder(num_actions)
        self.actor_core = make_core(self.actor_encoder, num_actions)

        self.critic_encoder = make_encoder(num_actions)
        self.critic_core = make_core(self.critic_encoder, num_actions)

        #self.core = self.critic_core
        #self.encoder = self.critic_encoder

        self.actor_mlp_dim_match = self._get_mlp_dim_match(action_space)
        self.critic_mlp_dim_match = self._get_mlp_dim_match(action_space)

        self.encoders = [self.actor_encoder, self.critic_encoder]
        self.cores = [self.actor_core, self.critic_core]
        self.mlp_dim_matches = [self.actor_mlp_dim_match, self.critic_mlp_dim_match]

        self.extra_critic_mlp = self._get_extra_critic_mlp()  # extra fc layers for critic: (core_out_size, core_out_size) x n
        if self.cfg.use_popart:
            self.register_buffer('mu', torch.zeros(cfg.num_envs, requires_grad=False))
            self.register_buffer('nu', torch.ones(cfg.num_envs, requires_grad=False))
            self.critic_linear = nn.Linear(self.critic_core.get_core_out_size(), cfg.num_envs)
            self.beta = self.cfg.popart_beta
            # self.register_buffer('popart_update_triggered', torch.zeros(cfg.num_envs, requires_grad=False).to(torch.bool))
        else:
            self.critic_linear = nn.Linear(self.critic_core.get_core_out_size(), 1)

        if self.cfg.use_vmpo:
            self.vmpo_eta = torch.nn.parameter.Parameter(torch.tensor(1.0))
            self.vmpo_alpha = torch.nn.parameter.Parameter(torch.tensor(5.0))

        assert self.cfg.use_pbl is False, "PBL is currently not supported for _ActorCriticSeparateWeights."
        self.action_parameterization = self.get_action_parameterization(self.actor_core.get_core_out_size())

        #self.apply(self.initialize_weights)
        [encoder.apply(self.initialize_weights) for encoder in self.encoders]
        [core.apply(self.weights_init) for core in self.cores]
        self.action_parameterization.apply(self.weights_init)
        self.critic_linear.apply(self.weights_init)
        self.extra_critic_mlp.apply(self.weights_init)
        [mlp_dim_match.apply(self.initialize_weights) if mlp_dim_match else 0 for mlp_dim_match in self.mlp_dim_matches]

        self.train()

    def _core_rnn(self, head_output, rnn_states, dones, is_seq, inverted_select_inds):
        """
        This is actually pretty slow due to all these split and cat operations.
        Consider using shared weights when training RNN policies.
        """
        num_cores = len(self.cores)
        if is_seq:
            head_outputs_split = head_output
            rnn_states_split = rnn_states
        else:
            head_outputs_split = head_output.chunk(num_cores, dim=1)
            rnn_states_split = rnn_states.chunk(num_cores, dim=1)

        outputs, new_rnn_states, all_hs, all_cs = [], [], [], []
        for i, c in enumerate(self.cores):
            output, new_rnn_state, (all_h, all_c) = c(head_outputs_split[i], rnn_states_split[i], dones, is_seq)
            if inverted_select_inds is not None:
                output = output.data.index_select(0, inverted_select_inds)
            outputs.append(output)
            new_rnn_states.append(new_rnn_state)
            all_hs.append(all_h)
            all_cs.append(all_c)

        outputs = torch.cat(outputs, dim=1)
        new_rnn_states = torch.cat(new_rnn_states, dim=1)
        all_hs = torch.cat(all_hs, dim=2)
        all_cs = torch.cat(all_cs, dim=2)
        return outputs, new_rnn_states, (all_hs, all_cs)

    def _core_trxl(self, head_output, mems, rollout_step_list, mem_begin_index, dones):
        """
        This is actually pretty slow due to all these split and cat operations.
        Consider using shared weights when training RNN policies.
        """

        num_cores = len(self.cores)
        head_outputs_split = head_output.chunk(num_cores, dim=1)
        memses_split = mems.chunk(num_cores, dim=-1)

        outputs, new_memses = [], []
        for i, c in enumerate(self.cores):
            # S# Temporary code for avoiding a bug in learner.py attn for both actor and critic should be considered
            output, new_mems, attn_entropy = c(head_outputs_split[i], memses_split[i], rollout_step_list, mem_begin_index, dones=dones)
            # E#
            outputs.append(output)
            new_memses.append(new_mems)

        outputs = torch.cat(outputs, dim=1)
        new_memses = torch.cat(new_memses, dim=-1)
        return outputs, new_memses, attn_entropy

    #@staticmethod
    #def _core_empty(head_output, mems, rollout_step_list=None, mem_begin_index=None, dones=None):
    #    """Optimization for the feed-forward case."""
    #    return head_output, mems

    def forward_head(self, obs_dict, actions, rewards):
        obs_dict = normalize_obs_return(obs_dict, self.cfg)

        assert torch.min(actions) >= -1 and torch.max(actions) < self.action_space.n
        done_ids = actions.eq(-1).nonzero(as_tuple=False)
        actions[done_ids] = 0
        prev_actions = torch.nn.functional.one_hot(actions, self.action_space.n).float()
        prev_actions[done_ids] = 0.

        head_outputs = []
        for i, e in enumerate(self.encoders):
            x = e(obs_dict)
            if self.cfg.extended_input:
                x = torch.cat((x, prev_actions, rewards.clamp(-1, 1).unsqueeze(1)), dim=1)

            if self.mlp_dim_matches[i]:
                x = self.mlp_dim_matches[i](x)

            head_outputs.append(x)

        return torch.cat(head_outputs, dim=1)

    def forward_core_rnn(self, head_output, rnn_states, dones, is_seq, inverted_select_inds=None):
        x, new_rnn_states, all_hidden = self._core_rnn(head_output, rnn_states, dones, is_seq, inverted_select_inds)
        return x, new_rnn_states, all_hidden

    def forward_core_transformer(self, head_output, mems=None, rollout_step_list=None, mem_begin_index=None, dones=None):
        x, new_mems, attn_entropy = self._core_trxl(head_output, mems, rollout_step_list, mem_begin_index, dones=dones)
        return x, new_mems, attn_entropy

    def forward_tail(self, core_output, task_ids, with_action_distribution=False):
        core_outputs = core_output.chunk(len(self.cores), dim=1)

        values = self.critic_linear(
            self.extra_critic_mlp(core_outputs[1])
        )
        normalized_values = values.clone()
        sigmas = torch.ones((values.size(0), 1), requires_grad=False)
        mus = torch.zeros((values.size(0), 1), requires_grad=False)
        if self.cfg.use_popart:
            normalized_values = normalized_values.gather(dim=1, index=task_ids)
            with torch.no_grad():
                nus = self.nu.index_select(dim=0, index=task_ids.squeeze(1)).unsqueeze(1)
                mus = self.mu.index_select(dim=0, index=task_ids.squeeze(1)).unsqueeze(1)
                sigmas = torch.sqrt(nus - mus ** 2)
                sigmas[torch.isnan(sigmas)] = self.cfg.popart_clip_min
                sigmas = torch.clamp(sigmas, min=self.cfg.popart_clip_min, max=1e+6)
                values = normalized_values * sigmas + mus

        action_distribution_params, action_distribution = self.action_parameterization(core_outputs[0])

        # for non-trivial action spaces it is faster to do these together
        actions, log_prob_actions = sample_actions_log_probs(action_distribution)

        result = AttrDict(dict(
            actions=actions,
            action_logits=action_distribution_params,
            # perhaps `action_logits` is not the best name here since we now support continuous actions
            log_prob_actions=log_prob_actions,
            values=values,
            normalized_values=normalized_values,
            sigmas=sigmas,
            mus=mus
        ))

        if with_action_distribution:
            result.action_distribution = action_distribution

        return result

    def forward(self,
                obs_dict,
                actions,
                rewards,
                rnn_states=None,
                mems=None,
                rollout_step_list=None,
                mem_begin_index=None,
                dones=None, task_ids=None,
                is_seq=None,
                with_action_distribution=False,
                is_transformer=False):
        x = self.forward_head(obs_dict, actions, rewards)
        if is_transformer:
            x, new_mems, attn_entropy = self.forward_core_transformer(x, mems, rollout_step_list=rollout_step_list,
                                                          mem_begin_index=mem_begin_index)
        else:
            x, new_rnn_states, _ = self.forward_core_rnn(x, rnn_states, dones, is_seq)

        assert not x.isnan().any()
        result = self.forward_tail(x, task_ids, with_action_distribution=with_action_distribution)
        if is_transformer:
            result.mems = new_mems
            result.attn_entropy = attn_entropy
        else:
            result.rnn_states = new_rnn_states

        return result

def create_actor_critic(cfg, obs_space, action_space, timing=None, is_learner_worker=False):
    if timing is None:
        timing = Timing()

    def make_encoder(action_sizes=-1):
        return create_encoder(cfg, obs_space, timing, action_sizes, is_learner_worker=is_learner_worker)

    def make_core(encoder, action_sizes=-1):
        core_input_size = encoder.get_encoder_out_size()
        if cfg.extended_input:
            core_input_size += action_sizes + 1

        if cfg.use_intrinsic and cfg.int_type == 'cell' and cfg.extended_input_cell:
            core_input_size += cfg.cell_dim

        if (
            cfg.match_core_input_size
            and core_input_size != cfg.hidden_size
        ):
            core_input_size = cfg.hidden_size

        return create_core(cfg, core_input_size)

    if cfg.actor_critic_share_weights:
        if use_lirpg(cfg):
            return _ActorCriticSharedWeightsMeta(make_encoder, make_core, action_space, cfg, timing, is_learner_worker)
        else:
            return _ActorCriticSharedWeights(make_encoder, make_core, action_space, cfg, timing, is_learner_worker)

    else:
        return _ActorCriticSeparateWeights(make_encoder, make_core, action_space, cfg, timing)


if __name__ == '__main__':
    from sample_factory.algorithms.appo.modules.meta_modules import update_params_by_sgd
    import argparse
    import os
    from sample_factory.envs.dmlab.dmlab_model import dmlab_register_models
    from collections import OrderedDict

    dmlab_register_models()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=1, help='')
    parser.add_argument('--extra_fc_critic_hidden_size', type=int, default=256, help='')
    parser.add_argument('--extra_fc_critic', type=int, default=1, help='')
    parser.add_argument('--extra_fc_critic_nonlinearity', type=str, default='relu', help='')
    parser.add_argument('--n_heads', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--obs_subtract_mean', type=int, default=128, help='')
    parser.add_argument('--obs_scale', type=int, default=128, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--use_pe', type=bool, default=True, help='')
    parser.add_argument('--use_popart', type=bool, default=False, help='')
    parser.add_argument('--use_gate', type=bool, default=False, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')
    parser.add_argument('--batch_size', type=int, default=1536, help='')
    parser.add_argument('--chunk_size', type=int, default=96, help='')
    parser.add_argument('--mem_len', type=int, default=0, help='')
    parser.add_argument('--hidden_size', type=int, default=256, help='')
    parser.add_argument('--extended_input', type=bool, default=False, help='')
    parser.add_argument('--nonlinearity', type=str, default='relu', help='')
    parser.add_argument('--encoder_subtype', type=str, default='resnet_impala', help='')
    parser.add_argument('--dmlab_instr_encoder_device', type=str, default='cpu', help='')
    parser.add_argument('--nonlinear_inplace', type=bool, default=False, help='')
    parser.add_argument('--use_transformer', type=bool, default=True, help='')
    parser.add_argument('--use_rnn', type=bool, default=False, help='')
    parser.add_argument('--encoder_extra_fc_layers', type=int, default=1, help='')
    parser.add_argument('--packed_seq', type=bool, default=False, help='')
    parser.add_argument('--recurrence', type=int, default=96, help='')
    parser.add_argument('--actor_critic_share_weights', type=bool, default=True, help='')
    parser.add_argument('--use_intrinsic', type=bool, default=True, help='')
    parser.add_argument('--use_half_policy_worker', type=bool, default=False, help='')
    parser.add_argument('--match_core_input_size', type=bool, default=False, help='')
    parser.add_argument('--int_type', type=str, default='lirpg', help='')
    parser.add_argument('--encoder_custom', type=str, default='dmlab_instructions', help='')
    parser.add_argument('--policy_initialization', type=str, default='orthogonal', help='')
    parser.add_argument('--env', type=str, default='dmlab_lasertag', help='')
    parser.add_argument('--policy_init_gain', default=1.0, type=float, help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    args.n_token = 10000

    data = torch.ones((args.batch_size, args.d_model)).random_(0, args.n_token).float().to(device)
    mems = None
    mem_begin_index = [0] * (args.batch_size // args.recurrence)
    cutoffs = [args.n_token // 2]

    div_val = 1
    d_embed = 100

    obs_space = AttrDict({
        'obs': torch.zeros((3, 72, 96)),
        'INSTR': torch.zeros(16).long(),
        'spaces': OrderedDict([('obs', torch.zeros((3, 72, 96))), ('INSTR', torch.zeros(16).long())])
    })

    action_space = gym.spaces.discrete.Discrete(15)

    model = create_actor_critic(args, obs_space, action_space, None, False).to(device)
    model_ = create_actor_critic(args, obs_space, action_space, None, False).to(device)

    model_.load_state_dict(model.state_dict())
    params = OrderedDict(model.named_parameters())
    optim = torch.optim.SGD(params.values(), lr=1e-4)
    print(sum(p.numel() for p in model.parameters()))

    dones = torch.zeros(args.batch_size)
    dones = dones.view(args.batch_size // args.recurrence,
                          args.chunk_size).transpose(0, 1)


    def make_data():
        obs = AttrDict({
            'obs': torch.randn((args.batch_size, 3, 72, 96)),
            'INSTR': torch.zeros((args.batch_size, 16)).long()
        })
        prev_actions = torch.randint(0, 15, (args.batch_size, 1)).long()
        prev_rewards = torch.zeros((args.batch_size, 1))
        return obs, prev_actions, prev_rewards


    obs, prev_actions, prev_rewards = make_data()
    x = model.forward_head(obs, prev_actions.squeeze().long(), prev_rewards.squeeze())
    x, new_mems, attn_entropy = model.forward_core_transformer(x, mems, mem_begin_index=mem_begin_index, dones=dones)
    x_ = model_.forward_head(obs, prev_actions.squeeze().long(), prev_rewards.squeeze())
    x_, new_mems_, attn_entropy_ = model_.forward_core_transformer(x_, mems, mem_begin_index=mem_begin_index,
                                                                   dones=dones)
    print((x - x_).sum())
    print((new_mems - new_mems_).sum())
    print((attn_entropy - attn_entropy_).sum())

    for i in range(2):
        obs, prev_actions, prev_rewards = make_data()
        x = model.forward_head(obs, prev_actions.squeeze().long(), prev_rewards.squeeze())
        x, new_mems, attn_entropy = model.forward_core_transformer(x, mems, mem_begin_index=mem_begin_index, dones=dones)
        result = model.forward_tail(x, None, with_action_distribution=True)

        dist = result.action_distribution
        dist = dist.log_prob(prev_actions.squeeze())
        dummy_target = torch.randn_like(dist)

        loss = F.mse_loss(dist, dummy_target)
        grads = torch.autograd.grad(loss, params.values(), retain_graph=True, allow_unused=True)
        updated_params = update_params_by_sgd(params, grads, lr=1e-4)

        optim.zero_grad()
        loss.backward()
        optim.step()

        # gradient check
        # for name, param in OrderedDict(model.named_parameters()).items():
        #     manual_param = updated_params[name]
        #     diff = torch.abs(param - manual_param).sum()
        #     print(name, diff.item())

        x = model.forward_head(obs, prev_actions.squeeze().long(), prev_rewards.squeeze())
        x, new_mems, attn_entropy = model.forward_core_transformer(x, mems, mem_begin_index=mem_begin_index, dones=dones)
        result = model.forward_tail(x, None, with_action_distribution=True)
        dist = result.action_distribution
        dist = dist.log_prob(prev_actions.squeeze())

        x_ = model_.forward_head(obs, prev_actions.squeeze().long(), prev_rewards.squeeze(), params=updated_params)
        x_, new_mems_, attn_entropy_ = model_.forward_core_transformer(x_, mems, mem_begin_index=mem_begin_index,dones=dones, params=updated_params)
        result_ = model_.forward_tail(x_, None, with_action_distribution=True, params=updated_params)
        dist_ = result_.action_distribution
        dist_ = dist_.log_prob(prev_actions.squeeze())
        print((x - x_).sum())
        print((new_mems - new_mems_).sum())
        print((attn_entropy - attn_entropy_).sum())
        print((dist - dist_).sum())

