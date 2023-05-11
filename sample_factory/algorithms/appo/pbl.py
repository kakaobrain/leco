import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
from sample_factory.algorithms.utils.action_distributions import sample_actions_log_probs, is_continuous_action_space, calc_num_actions, calc_num_logits
# from sample_factory.algorithms.appo.learner import build_rnn_inputs
from sample_factory.algorithms.appo.model_utils import normalize_obs_return


class PBL(nn.Module):
    def __init__(self, cfg, encoder_f, core_out_size, action_space):
        super(PBL, self).__init__()
        self.cfg = cfg
        self.rnn_num_layers = cfg.rnn_num_layers
        self.action_space = action_space
        self.n_action = action_space.n
        self.horizon_k: int = 20
        self.time_subsample: int = 6
        self.forward_subsample: int = 2
        self.core_out_size = core_out_size

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_sizes = [action_space.n]
        else:
            self.action_sizes = [space.n for space in action_space.spaces]

        self.encoder_f = encoder_f
        if self.cfg.use_transformer:
            self.h_p = nn.GRU(self.n_action, self.core_out_size, 1)
            g_input_size, g_output_size = self.core_out_size, self.core_out_size
        elif self.cfg.use_rnn:
            self.h_p = nn.LSTM(self.n_action, self.core_out_size, self.rnn_num_layers)
            g_input_size, g_output_size = self.core_out_size * self.rnn_num_layers, self.core_out_size +    80

        self.g = nn.Sequential(
            nn.Linear(g_input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, g_output_size)
        )
        self.g_prime = nn.Sequential(
            nn.Linear(g_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, g_input_size)
        )

        self.mse_loss = nn.MSELoss(reduction='none')

    def _forward_pred_loss(self, mb, obs, b_t, num_traj, recurrence, novelty=None):
        not_dones = (1.0 - mb.dones).view(num_traj, recurrence, 1)
        mask_res = self._build_mask_and_subsample(not_dones)
        (forward_mask, unroll_subsample, time_subsample, max_k) = mask_res

        actions = mb.actions.view(num_traj, recurrence, -1).long()
        actions = nn.functional.one_hot(actions, num_classes=self.n_action).squeeze(2).float()
        actions_unfolded = self._build_unfolded(actions[:, :-1], max_k).index_select(2, time_subsample)  # max_k, 32, 6, 32  seq, batch, time, dim
        actions_unfolded = actions_unfolded.flatten(1,2) # max_k, 192, 32 = max_k, batch * sampled t * embed_dim

        if novelty is not None:
            novelty = novelty.view(num_traj, recurrence, -1)
            novelty = self._build_unfolded(novelty[:, :-1], max_k)
            novelty = torch.cumsum(novelty, 2).index_select(2, time_subsample)
            novelty = novelty.flatten(1, 2)
            novelty = novelty.index_select(0, unroll_subsample)
            norm_factor, _ = torch.max(novelty, 1)
            novelty /= (norm_factor.unsqueeze(-1) + 1e-8)  # scale 0 to 1

        if self.cfg.use_transformer:
            h_pred_stack = self._make_gru_pred(b_t, actions_unfolded, num_traj, recurrence, time_subsample, max_k)
        elif self.cfg.use_rnn:
            h_pred_stack = self._make_lstm_pred(b_t, actions_unfolded, num_traj, recurrence, time_subsample, max_k)
        h_pred_stack = h_pred_stack.index_select(0, unroll_subsample)

        final_pred = self.g(h_pred_stack) # 2, 192, 336
        regularize_loss = 0.02 * torch.square(torch.norm(final_pred, dim=2) - 1)
        final_pred = nn.functional.normalize(final_pred, p=2.0, dim=2, eps=1e-8)

        with torch.no_grad():
            z_t = self._encode_f(obs, mb.prev_actions.squeeze().long(), mb.prev_rewards.squeeze())  # 1024 , 336
            z_t = z_t.view(num_traj, recurrence, -1)  # 32, 24, 592 = batch, T, dim

            forward_targets = self._build_unfolded(z_t[:, 1:], max_k)  # max_k, batch, T, dim
            forward_targets = forward_targets.index_select(2, time_subsample) # max_k, batch, 6, dim
            forward_targets = forward_targets.index_select(0, unroll_subsample) # 2, batch, T, dim
            forward_targets = forward_targets.flatten(1, 2) # 2, batch*6, dim
            forward_targets = nn.functional.normalize(forward_targets, p=2.0, dim=2, eps=1e-8)

        loss = self.mse_loss(final_pred, forward_targets)

        if novelty is None:
            certainty = torch.ones_like(regularize_loss).unsqueeze(-1)
        else:
            certainty = 1 - novelty

        loss *= certainty
        regularize_loss *= certainty.squeeze(-1)

        loss = loss.mean(-1, keepdim=True)
        loss = torch.masked_select(loss, forward_mask).mean()
        return loss, regularize_loss.mean()

    def _make_gru_pred(self, b_t, actions_unfolded, num_traj, recurrence, time_subsample, max_k):
        c = b_t.reshape(num_traj, recurrence, -1)  # 32, 24, 3*512 = batch, T, n_layer*lstm_hidden
        c_subsampled = c[:, :-1].index_select(1, time_subsample)  # 32, 6, 768 = batch * sampled_t * nlayer_lstm_hidden
        c_subsampled = c_subsampled.view(c_subsampled.size(0) * c_subsampled.size(1), 1, self.core_out_size)
        c_subsampled = c_subsampled.permute(1, 0, 2)  # 3, 32*6, 512

        h_pred_stack = self._predict_gru(actions_unfolded, c_subsampled, max_k)
        return h_pred_stack

    def _make_lstm_pred(self, b_t, actions_unfolded, num_traj, recurrence, time_subsample, max_k):
        h, c = b_t

        h = h.permute(2, 0, 1, 3)
        h = h.reshape(num_traj, recurrence, -1)  # 32, 24, 768
        h_subsampled = h[:, :-1].index_select(1, time_subsample)  # 32, 6, 768
        h_subsampled = h_subsampled.view(h_subsampled.size(0) * h_subsampled.size(1), self.rnn_num_layers, self.core_out_size)
        h_subsampled = h_subsampled.permute(1, 0, 2)  # 3, 32*6, 512

        c = c.permute(2, 0, 1, 3)  # 32, 24, 3, 512 = batch, T, n_layer, lstm_hidden
        c = c.reshape(num_traj, recurrence, -1)  # 32, 24, 3*512 = batch, T, n_layer*lstm_hidden
        c_subsampled = c[:, :-1].index_select(1, time_subsample)  # 32, 6, 768 = batch * sampled_t * nlayer_lstm_hidden
        c_subsampled = c_subsampled.view(c_subsampled.size(0) * c_subsampled.size(1), self.rnn_num_layers, self.core_out_size)
        c_subsampled = c_subsampled.permute(1, 0, 2)  # 3, 32*6, 512

        h_pred_stack = self._predict_lstm(actions_unfolded, h_subsampled, c_subsampled, max_k)
        return h_pred_stack

    def _predict_gru(self, actions_unfolded, c_subsampled, max_k):
        actions_unfolded = actions_unfolded.contiguous()
        c = c_subsampled.contiguous()

        lst = []
        for i in range(max_k):
            out, c = self.h_p(actions_unfolded[i].unsqueeze(0), c)
            lst.append(c)

        x = torch.stack(lst)
        return x.squeeze(1)

    def _predict_lstm(self, actions_unfolded, h_subsampled, c_subsampled, max_k):
        actions_unfolded = actions_unfolded.contiguous()
        h_subsampled = h_subsampled.contiguous()
        c_subsampled = c_subsampled.contiguous()

        h, c = h_subsampled, c_subsampled
        lst = []
        for i in range(max_k):
            out, (h, c) = self.h_p(actions_unfolded[i].unsqueeze(0), (h, c))
            lst.append(h)

        x = torch.stack(lst)
        x = x.permute(0, 2, 1, 3)
        return x.reshape(max_k, -1, self.rnn_num_layers * self.core_out_size)

    def _build_unfolded(self, x, k: int):
        tobe_cat = x.new_zeros(x.size(0), k, x.size(2))
        cat = torch.cat((x, tobe_cat), 1)
        cat = cat.unfold(1, size=k, step=1)
        cat = cat.permute(3, 0, 1, 2)
        return cat

    def _build_mask_and_subsample(self, not_dones):
        t = not_dones.size(1)

        not_dones_unfolded = self._build_unfolded(not_dones[:, :-1].to(dtype=torch.bool), self.horizon_k) # 10, 32, 24, 1
        time_subsample = torch.randperm(t - 1, device=not_dones.device, dtype=torch.long)[0:self.time_subsample]

        forward_mask = torch.cumprod(not_dones_unfolded.index_select(2, time_subsample), dim=0).to(dtype=torch.bool) # 10, 32, 6, 1
        forward_mask = forward_mask.flatten(1, 2) # 10, 192, 1

        max_k = forward_mask.flatten(1).any(-1).nonzero().max().item() + 1

        unroll_subsample = torch.randperm(max_k, dtype=torch.long)[0:self.forward_subsample]

        max_k = unroll_subsample.max().item() + 1

        unroll_subsample = unroll_subsample.to(device=not_dones.device)
        forward_mask = forward_mask.index_select(0, unroll_subsample)

        return forward_mask, unroll_subsample, time_subsample, max_k

    def _backward_pred_loss(self, mb, obs, b_t, num_traj, recurrence, novelty=None):
        z_t = self._encode_f(obs, mb.prev_actions.squeeze().long(), mb.prev_rewards.squeeze())  # 1024 , 336
        z_t = z_t.reshape(num_traj, recurrence, -1)
        regularize_loss = 0.02 * torch.square(torch.norm(z_t, dim=2) - 1)
        z_t = nn.functional.normalize(z_t, p=2.0, dim=2, eps=1e-8)
        pred = self.g_prime(z_t)

        if self.cfg.use_rnn:
            h, c = b_t
            h_clone = h.clone().detach()  # c : recurrence, nlayer, num_traj, dim
            h_clone = h_clone.permute(2, 0, 1, 3)  # c : num_traj, recurrence, nlayer, dim
        elif self.cfg.use_transformer:
            h_clone = b_t.clone().detach()  # c : recurrence, nlayer, num_traj, dim
        h_clone = h_clone.reshape(num_traj, recurrence, -1)  # c : num_traj * recurrence,  nlayer * dim  = 1024 , 256*3
        loss = self.mse_loss(pred, h_clone)

        # if novelty is not None:
        #     novelty = novelty.reshape(num_traj, recurrence, -1)
        #     novelty = torch.cumsum(novelty, 1)
        #     norm_factor, _ = torch.max(novelty, 1)
        #     novelty /= (norm_factor.unsqueeze(-1) + 1e-8)  # scale 0 to 1
        #     certainty = (1 - novelty)
        # else:
        #     certainty = torch.ones_like(regularize_loss).unsqueeze(-1)
        # loss *= certainty
        # regularize_loss *= certainty.squeeze(-1)

        return loss.mean(), regularize_loss.mean()

    def calc_loss(self, mb, b_t, num_traj, recurrence, novelty=None):
        # b_t : (h,c)   h: recurrence, n_layer, num_traj, lstm dim
        # obs : 1024, 3, 72, 96
        obs = mb.obs
        if self.cfg.pbl_obs_norm:
            obs = normalize_obs_return(obs, self.cfg)

        l_forward, l_forward_reg = self._forward_pred_loss(mb, obs, b_t, num_traj, recurrence, novelty=novelty)
        l_backward, l_backward_reg = self._backward_pred_loss(mb, obs, b_t, num_traj, recurrence, novelty=novelty)

        loss = l_forward + l_backward + l_forward_reg + l_backward_reg
        return loss, (l_forward.item(), l_backward.item(), l_forward_reg.item(), l_backward_reg.item())


    def _encode_f(self, obs_dict, actions, rewards):
        x = self.encoder_f(obs_dict)

        if self.cfg.extended_input:
            # -1 is transformed into all zero vector
            assert torch.min(actions) >= -1 and torch.max(actions) < self.n_action
            done_ids = actions.eq(-1).nonzero(as_tuple=False)
            actions[done_ids] = 0
            prev_actions = torch.nn.functional.one_hot(actions, self.n_action).float()
            prev_actions[done_ids] = 0.

            return torch.cat((x, prev_actions, rewards.clamp(-1, 1).unsqueeze(1)), dim=1)
        else:
            return x

