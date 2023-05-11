import torch
from torch import nn
from sample_factory.algorithms.appo.model_utils import create_encoder, create_core, ActionParameterizationContinuousNonAdaptiveStddev, ActionParameterizationDefault, normalize_obs_return
from sample_factory.algorithms.utils.action_distributions import sample_actions_log_probs, is_continuous_action_space, calc_num_actions, calc_num_logits

class AdvModel4MiniGrid(nn.Module):
    def __init__(self, cfg, obs_space, action_space, timing):
        super(AdvModel4MiniGrid, self).__init__()

        self.cfg = cfg
        self.action_space = action_space
        num_actions = action_space.n
        self.int_encoder = create_encoder(cfg, obs_space, timing, num_actions)

        if cfg.extended_input:
            self.int_core = create_core(cfg, self.int_encoder.get_encoder_out_size() + num_actions + 1)
        else:
            self.int_core = create_core(cfg, self.int_encoder.get_encoder_out_size())

        int_core_out_size = self.int_core.get_core_out_size()

        self.int_action_parameterization = self.get_action_parameterization(int_core_out_size,
                                                                                layers_dim=[int_core_out_size // 2])

    def get_action_parameterization(self, core_output_size, layers_dim=None):
        if not self.cfg.adaptive_stddev and is_continuous_action_space(self.action_space):
            action_parameterization = ActionParameterizationContinuousNonAdaptiveStddev(
                self.cfg, core_output_size, self.action_space,
            )
        else:
            action_parameterization = ActionParameterizationDefault(self.cfg, core_output_size, self.action_space, layers_dim)

        return action_parameterization

    def forward_head(self, obs_dict, actions, rewards):
        #normalize_obs(obs_dict, self.cfg) # before normalize, [0,255]
        #x = self.encoder(obs_dict)
        obs_dict_ = normalize_obs_return(obs_dict, self.cfg)  # before normalize, [0,255]
        x = self.int_encoder(obs_dict_)
        if self.cfg.extended_input:
            # -1 is transformed into all zero vector
            assert torch.min(actions) >= -1 and torch.max(actions) < self.action_space.n
            done_ids = actions.eq(-1).nonzero(as_tuple=False)
            actions[done_ids] = 0
            prev_actions = torch.nn.functional.one_hot(actions, self.action_space.n).float()
            prev_actions[done_ids] = 0.

            return torch.cat((x, prev_actions, rewards.clamp(-1, 1).unsqueeze(1)), dim=1)
        else:
            return x

    def forward_core(self, int_head_output, int_rnn_states, dones, is_seq, inverted_select_inds=None):
        int_core_output, int_new_rnn_states, _ = self.int_core(int_head_output, int_rnn_states, dones, is_seq)
        if inverted_select_inds is not None:
            int_core_output = int_core_output.data.index_select(0, inverted_select_inds)
        return int_core_output, int_new_rnn_states

    def forward_tail(self, int_core_output):
        action_distribution_params, _ = self.int_action_parameterization(int_core_output)
        return action_distribution_params

class AdvModel4DMLab(nn.Module):
    def __init__(self, cfg, obs_space, action_space, timing):
        super(AdvModel4DMLab, self).__init__()

        self.cfg = cfg
        self.action_space = action_space
        num_actions = action_space.n
        self.int_encoder = create_encoder(cfg, obs_space, timing, num_actions)

        if cfg.extended_input:
            self.int_core = create_core(cfg, self.int_encoder.get_encoder_out_size() + num_actions + 1)
        else:
            self.int_core = create_core(cfg, self.int_encoder.get_encoder_out_size())

        int_core_out_size = self.int_core.get_core_out_size()

        self.int_action_parameterization = self.get_action_parameterization(int_core_out_size,
                                                                                layers_dim=[int_core_out_size // 2])

    def get_action_parameterization(self, core_output_size, layers_dim=None):
        if not self.cfg.adaptive_stddev and is_continuous_action_space(self.action_space):
            action_parameterization = ActionParameterizationContinuousNonAdaptiveStddev(
                self.cfg, core_output_size, self.action_space,
            )
        else:
            action_parameterization = ActionParameterizationDefault(self.cfg, core_output_size, self.action_space, layers_dim)

        return action_parameterization

    def forward_head(self, obs_dict, actions, rewards):
        #normalize_obs(obs_dict, self.cfg) # before normalize, [0,255]
        #x = self.encoder(obs_dict)
        obs_dict_ = normalize_obs_return(obs_dict, self.cfg)  # before normalize, [0,255]
        x = self.int_encoder(obs_dict_)
        if self.cfg.extended_input:
            # -1 is transformed into all zero vector
            assert torch.min(actions) >= -1 and torch.max(actions) < self.action_space.n
            done_ids = actions.eq(-1).nonzero(as_tuple=False)
            actions[done_ids] = 0
            prev_actions = torch.nn.functional.one_hot(actions, self.action_space.n).float()
            prev_actions[done_ids] = 0.

            return torch.cat((x, prev_actions, rewards.clamp(-1, 1).unsqueeze(1)), dim=1)
        else:
            return x

    def forward_core(self, int_head_output, int_rnn_states, dones, is_seq, inverted_select_inds=None):
        int_core_output, int_new_rnn_states, _ = self.int_core(int_head_output, int_rnn_states, dones, is_seq)
        if inverted_select_inds is not None:
            int_core_output = int_core_output.data.index_select(0, inverted_select_inds)
        return int_core_output, int_new_rnn_states

    def forward_tail(self, int_core_output):
        action_distribution_params, _ = self.int_action_parameterization(int_core_output)
        return action_distribution_params