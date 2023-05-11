import math
import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import _VF
from torch.nn.utils.rnn import PackedSequence

from sample_factory.algorithms.appo.model_utils import nonlinearity, EncoderBase, get_obs_shape, calc_num_elements, fc_after_encoder_size
from sample_factory.algorithms.appo.model_utils import create_standard_encoder, EncoderBase, PolicyCoreBase
from sample_factory.envs.dmlab.dmlab30 import DMLAB_VOCABULARY_SIZE, DMLAB_INSTRUCTIONS
from sample_factory.utils.utils import log


def get_child_dict(params, key=None):
    """
    Constructs parameter dictionary for a network module.

    Args:
      params (dict): a parent dictionary of named parameters.
      key (str, optional): a key that specifies the root of the child dictionary.

    Returns:
      child_dict (dict): a child dictionary of model parameters.
    """
    if params is None:
        return None
    if key is None or (isinstance(key, str) and key == ''):
        return params

    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    if not any(filter(key_re.match, params.keys())):  # handles nn.DataParallel
        key_re = re.compile(r'^module\.{0}\.(.+)'.format(re.escape(key)))
    child_dict = OrderedDict(
        (key_re.sub(r'\1', k), value) for (k, value)
        in params.items() if key_re.match(k) is not None)
    return child_dict


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride, padding, bias=bias)

    def forward(self, x, params=None, episode=None):
        if params is None:
            x = super(Conv2d, self).forward(x)
        else:
            weight, bias = params.get('weight'), params.get('bias')
            if weight is None:
                weight = self.weight
            if bias is None:
                bias = self.bias
            x = F.conv2d(x, weight, bias, self.stride, self.padding)
        return x


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__(in_features, out_features, bias=bias)

    def forward(self, x, params=None):
        if params is None:
            x = super(Linear, self).forward(x)
        else:
            weight, bias = params.get('weight'), params.get('bias')
            if weight is None:
                weight = self.weight
            if bias is None:
                bias = self.bias
            x = F.linear(x, weight, bias)
        return x


class Sequential(nn.Sequential):
    def __init__(self, *args):
        super(Sequential, self).__init__(*args)

    def forward(self, x, params=None):
        if params is None:
            for module in self:
                x = module(x)
        else:
            for name, module in self._modules.items():
                if name.startswith('_'):
                    x = module(x)
                else:
                    x = module(x, params=get_child_dict(params, name))
        return x


class Embedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super(Embedding, self).__init__(*args, **kwargs)

    def forward(self, input, params=None):
        if params is None:
            return super(Embedding, self).forward(input)
        return F.embedding(
            input, params.get('weight'), self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args):
        super(LayerNorm, self).__init__(*args)

    def forward(self, input, params=None):
        if params is None:
            return super(LayerNorm, self).forward(input)
        return F.layer_norm(
            input, self.normalized_shape, params.get('weight'), params.get('bias'), self.eps)


class LSTM(nn.LSTM):
    def __init__(self, *args, **kwargs):
        super(LSTM, self).__init__(*args, **kwargs)

    def flatten_parameters(self, weights=None) -> None:
        """Resets parameter data pointer so that they can use faster code paths.

        Right now, this works only if the module is on the GPU and cuDNN is enabled.
        Otherwise, it's a no-op.
        """

        if weights is None:
            weights = self._flat_weights

        # Short-circuits if _flat_weights is only partially instantiated
        if len(weights) != len(self._flat_weights_names):
            return

        for w in weights:
            if not isinstance(w, torch.Tensor):
                return
        # Short-circuits if any tensor in self._flat_weights is not acceptable to cuDNN
        # or the tensors in _flat_weights are of different dtypes

        first_fw = weights[0]
        dtype = first_fw.dtype
        for fw in weights:
            if (not isinstance(fw.data, torch.Tensor) or not (fw.data.dtype == dtype) or
                    not fw.data.is_cuda or
                    not torch.backends.cudnn.is_acceptable(fw.data)):
                return

        # If any parameters alias, we fall back to the slower, copying code path. This is
        # a sufficient check, because overlapping parameter buffers that don't completely
        # alias would break the assumptions of the uniqueness check in
        # Module.named_parameters().
        unique_data_ptrs = set(p.data_ptr() for p in weights)
        if len(unique_data_ptrs) != len(weights):
            return

        with torch.cuda.device_of(first_fw):
            import torch.backends.cudnn.rnn as rnn

            # Note: no_grad() is necessary since _cudnn_rnn_flatten_weight is
            # an inplace operation on weights
            with torch.no_grad():
                if torch._use_cudnn_rnn_flatten_weight():
                    num_weights = 4 if self.bias else 2
                    if self.proj_size > 0:
                        num_weights += 1
                    torch._cudnn_rnn_flatten_weight(
                        weights, num_weights,
                        self.input_size, rnn.get_cudnn_mode(self.mode),
                        self.hidden_size, self.proj_size, self.num_layers,
                        self.batch_first, bool(self.bidirectional))

    def forward(self, input, hx=None, params=None):
        if params is None:
            return super(LSTM, self).forward(input, hx=hx)

        weights = [params.get(wn) if wn in params else None for wn in self._flat_weights_names]
        self.flatten_parameters(weights)
        orig_input = input
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            sorted_indices = None
            unsorted_indices = None

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
            h_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, real_hidden_size,
                                  dtype=input.dtype, device=input.device)
            c_zeros = torch.zeros(self.num_layers * num_directions,
                                  max_batch_size, self.hidden_size,
                                  dtype=input.dtype, device=input.device)
            hx = (h_zeros, c_zeros)
        else:
            # Each batch of the hidden state should match the input sequence that
            # the user believes he/she is passing in.
            hx = self.permute_hidden(hx, sorted_indices)

        self.check_forward_args(input, hx, batch_sizes)
        if batch_sizes is None:
            result = _VF.lstm(input, hx, weights, self.bias, self.num_layers,
                              self.dropout, self.training, self.bidirectional, self.batch_first)
        else:
            result = _VF.lstm(input, batch_sizes, hx, weights, self.bias,
                              self.num_layers, self.dropout, self.training, self.bidirectional)
        output = result[0]
        hidden = result[1:]
        # xxx: isinstance check needs to be in conditional for TorchScript to compile
        if isinstance(orig_input, PackedSequence):
            output_packed = PackedSequence(output, batch_sizes, sorted_indices, unsorted_indices)
            return output_packed, self.permute_hidden(hidden, unsorted_indices)
        else:
            return output, self.permute_hidden(hidden, unsorted_indices)


class ResBlock(nn.Module):
    def __init__(self, cfg, input_ch, output_ch):
        super().__init__()

        self.res_block_core = Sequential(OrderedDict([
            ('_', nonlinearity(cfg)),
            ('conv0', Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1)),
            ('_', nonlinearity(cfg)),
            ('conv1', Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1)),
        ]))

    def forward(self, x, params=None):
        identity = x
        out = self.res_block_core(x, params=get_child_dict(params, 'res_block_core'))
        out = out + identity
        return out


class EncoderBaseMeta(EncoderBase):
    def init_fc_blocks(self, input_size):
        layers = []
        fc_layer_size = fc_after_encoder_size(self.cfg)
        fc_layer_size = self.cfg.encoder_fc_hidden_size if self.cfg.encoder_fc_hidden_size > 0 else fc_layer_size

        for i in range(self.cfg.encoder_extra_fc_layers):
            size = input_size if i == 0 else fc_layer_size
            is_last_layer = i == self.cfg.encoder_extra_fc_layers - 1
            fc_layer_size = self.cfg.hidden_size if is_last_layer else fc_layer_size

            layers.extend([
                (f'fc{i}', Linear(size, fc_layer_size)),
                ('_', nonlinearity(self.cfg)),
            ])

        if len(layers) > 0:
            self.fc_after_enc = Sequential(OrderedDict(layers))
            self.encoder_out_size = fc_layer_size
        else:
            self.encoder_out_size = input_size

    def forward_fc_blocks(self, x, params=None):
        if self.fc_after_enc is not None:
            x = self.fc_after_enc(x, params=get_child_dict(params, 'fc_after_enc'))
        return x


class ResnetEncoder(EncoderBaseMeta):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg, None)

        obs_shape = get_obs_shape(obs_space)
        input_ch = obs_shape.obs[0]
        if cfg.encoder_subtype == 'resnet_impala':
            # configuration from the IMPALA paper
            resnet_conf = [[16, 2], [32, 2], [32, 2]]
        elif cfg.encoder_subtype == 'resnet_impala_large':
            resnet_conf = [[32, 2], [64, 2], [64, 2]]
        else:
            raise NotImplementedError(f'Unknown resnet subtype {cfg.encoder_subtype}')

        curr_input_channels = input_ch
        layers = []
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            layers.extend([
                (f'conv{i}', Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=1, padding=1)),  # padding SAME
                (f'_pool{i}', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),  # padding SAME
            ])

            for j in range(res_blocks):
                layers.append((f'res{j}_{i}', ResBlock(cfg, out_channels, out_channels)))

            curr_input_channels = out_channels

        layers.append(('_', nonlinearity(cfg)))

        self.conv_head = Sequential(OrderedDict(layers))
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)

        self.init_fc_blocks(self.conv_head_out_size)

    def forward(self, obs_dict, params=None):
        x = self.conv_head(obs_dict['obs'], params=get_child_dict(params, 'conv_head'))
        x = x.contiguous().view(-1, self.conv_head_out_size)
        x = self.forward_fc_blocks(x, params=params)
        return x


class ResnetEncoderDecoder(EncoderBaseMeta):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg, None)
        self.cfg = cfg
        self.is_learner_worker = False # default value

        obs_shape = get_obs_shape(obs_space)

        if cfg.encoder_subtype == 'resnet_impala':
            # configuration from the IMPALA paper
            resnet_conf = [[16, 2], [32, 2], [32, 2]]
        elif cfg.encoder_subtype == 'resnet_impala_small':
            resnet_conf = [[16, 2], [32, 2]]
        elif cfg.encoder_subtype == 'resnet_impala_large':
            resnet_conf = [[32, 2], [64, 2], [64, 2]]
        else:
            raise NotImplementedError(f'Unknown resnet subtype {cfg.encoder_subtype}')

        layers_encoder = list()
        layers_decoder = list()

        curr_input_channels = obs_shape.obs[0]
        for i, (out_channels, res_blocks) in enumerate(resnet_conf):
            _layer_encoder = list()
            stride_dec = 2
            if self.cfg.encoder_pooling == 'stride':
                stride_enc = 2
                pool = nn.Identity()
            else:
                stride_enc = 1
                pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if self.cfg.encoder_pooling == 'max' \
                       else nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

            is_last_cnn = i == len(resnet_conf) - 1
            if self.cfg.no_pooling_last_cnn and is_last_cnn:
                stride_enc = stride_dec = 1
                pool = nn.Identity()

            _layer_encoder.extend([
                (f'conv{i}', Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=stride_enc, padding=1)),
                (f'_pool{i}', pool)
            ])
            for j in range(res_blocks):
                _layer_encoder.append((f'res{j}_{i}', ResBlock(cfg, out_channels, out_channels)))
            layer_encoder = Sequential(OrderedDict(_layer_encoder))
            layers_encoder.append(layer_encoder)

            _layer_decoder = list()
            output_paddig = 1 if stride_dec == 2 else 0
            _layer_decoder.append(
                nn.ConvTranspose2d(out_channels, curr_input_channels, kernel_size=3, stride=stride_dec,
                                   padding=1, output_padding=output_paddig)
            )
            for j in range(res_blocks):
                _layer_decoder.append(ResBlock(cfg, curr_input_channels, curr_input_channels))
            layer_decoder = nn.Sequential(*_layer_decoder)
            layers_decoder.append(layer_decoder)

            curr_input_channels = out_channels

        layers_decoder.reverse()
        self.layers_encoder = nn.ModuleList(layers_encoder)
        self.layers_decoder = nn.ModuleList(layers_decoder)
        self.conv_head_out_size = calc_num_elements(nn.Sequential(*self.layers_encoder), obs_shape.obs)

        self.init_fc_blocks(self.conv_head_out_size)

    def set_learner_worker(self):
        self.is_learner_worker = True

    def forward(self, obs_dict, params=None):
        x = obs_dict['obs']

        check = torch.all(torch.logical_and(torch.all(obs_dict['obs'] >= -1),
                                            torch.all(obs_dict['obs'] <= 1)))
        assert check, "Observations should be normalized for deconvolution"

        x_enc = x
        encodings = list()
        for i in range(len(self.layers_encoder)):
            layer_encoder = self.layers_encoder[i]
            x_enc = layer_encoder(x_enc, params=get_child_dict(params, f'layers_encoder.{i}'))
            encodings.append(x_enc)
        x_enc = nonlinearity(self.cfg)(x_enc)

        x_dec = x_enc
        if self.is_learner_worker: # We reconstruct only for learner_worker
            encodings.pop(-1) # pop out last encoding since it is not used
            encodings.reverse() # Reverse the order
            for i in range(len(self.layers_decoder) - 1):
                layer_decoder = self.layers_decoder[i]
                x_dec = layer_decoder(x_dec)

                if self.cfg.use_long_skip_reconstruction:
                    x_dec = x_dec + encodings[i]
            if self.cfg.reconstruction_loss_type == 'MSE':
                x_dec = torch.tanh(self.layers_decoder[-1](x_dec))
            else:
                x_dec = torch.sigmoid(self.layers_decoder[-1](x_dec))
            self.x_dec = x_dec

        x_enc = x_enc.contiguous().view(-1, self.conv_head_out_size)
        x = self.forward_fc_blocks(x_enc, params=params)

        return x

    def get_decoder_loss(self, obs_dict):
        if self.cfg.reconstruction_loss_type == 'MSE':
            mean = self.cfg.obs_subtract_mean
            scale = self.cfg.obs_scale
            obs_normalized = (obs_dict['obs'] - mean) / scale
            loss = self.cfg.reconstruction_loss_coeff * torch.mean(torch.pow(obs_normalized - self.x_dec, 2))
        elif self.cfg.reconstruction_loss_type == 'MSE_Inverted':
            obs_normalized = (255 - obs_dict['obs']) / 255 # Invert and normalize to make white board as zero
            loss = self.cfg.reconstruction_loss_coeff * torch.mean(torch.pow(obs_normalized - self.x_dec, 2))
        elif self.cfg.reconstruction_loss_type == 'CE':
            obs_normalized = obs_dict['obs'] / 255 # 0 ~ 1
            loss = self.cfg.reconstruction_loss_coeff * torch.mean(
                    -obs_normalized * torch.log(self.x_dec + 1e-10)
                    - (1 - obs_normalized) * torch.log(1 - self.x_dec + 1e-10))
        else:
            assert False, f"loss type '{self.cfg.reconstruction_loss_type}' is not supported for reconstruction"
        return loss


class DmlabEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing, action_sizes):
        super().__init__(cfg, timing)
        if cfg.encoder_type == 'resnet_encoder_decoder':
            encoder = ResnetEncoderDecoder(cfg, obs_space)
        else:
            encoder = ResnetEncoder(cfg, obs_space)
        self.basic_encoder = encoder
        self.encoder_out_size = self.basic_encoder.encoder_out_size

        # same as IMPALA paper
        self.embedding_size = 20
        self.instructions_lstm_units = 64
        self.instructions_lstm_layers = 1

        padding_idx = 0
        self.word_embedding = Embedding(
            num_embeddings=DMLAB_VOCABULARY_SIZE,
            embedding_dim=self.embedding_size,
            padding_idx=padding_idx
        )

        self.instructions_lstm = LSTM(
            input_size=self.embedding_size,
            hidden_size=self.instructions_lstm_units,
            num_layers=self.instructions_lstm_layers,
            batch_first=True,
        )

        self.encoder_out_size += self.instructions_lstm_units
        log.debug('Policy head output size: %r', self.encoder_out_size)

        if cfg.dmlab_instr_encoder_device == 'cpu':
            self.instr_device = torch.device('cpu')
        else:
            assert torch.cuda.device_count() == 1
            self.instr_device = torch.device('cuda', index=0)

        self.cudnn_enable = False

    def model_to_device(self, device):
        self.to(device)
        self.instr_device = device
        self.word_embedding.to(self.instr_device)
        self.instructions_lstm.to(self.instr_device)

    def device_and_type_for_input_tensor(self, input_tensor_name):
        if input_tensor_name == DMLAB_INSTRUCTIONS:
            return self.instr_device, torch.int64
        else:
            return self.model_device(), torch.float32

    def forward_instr(self, obs_dict, params=None):
        with torch.no_grad():
            instr = obs_dict[DMLAB_INSTRUCTIONS]
            instr_lengths = (instr != 0).sum(axis=1)
            instr_lengths = torch.clamp(instr_lengths, min=1)
            max_instr_len = torch.max(instr_lengths).item()
            instr = instr[:, :max_instr_len]
            instr_lengths_cpu = instr_lengths.to('cpu')

        instr_embed = self.word_embedding(instr, params=get_child_dict(params, 'word_embedding'))
        instr_packed = torch.nn.utils.rnn.pack_padded_sequence(
            instr_embed, instr_lengths_cpu, batch_first=True, enforce_sorted=False,
        )
        with torch.backends.cudnn.flags(enabled=self.cudnn_enable):
            rnn_output, _ = self.instructions_lstm(instr_packed, params=get_child_dict(params, 'instructions_lstm'))
        rnn_outputs, sequence_lengths = torch.nn.utils.rnn.pad_packed_sequence(rnn_output, batch_first=True)

        first_dim_idx = torch.arange(rnn_outputs.shape[0])
        last_output_idx = sequence_lengths - 1
        last_outputs = rnn_outputs[first_dim_idx, last_output_idx]

        return last_outputs

    def forward(self, obs_dict, params=None, **kwargs):
        x = self.basic_encoder(obs_dict, params=get_child_dict(params, 'basic_encoder'))

        if kwargs.get('instr', None) is None:
            ext_data = self.forward_instr(obs_dict, params=params)
        else:
            ext_data = kwargs.get('instr')

        if self.instr_device.type == 'cpu':
            ext_data = ext_data.to(x.device)  # for some reason this is very slow

        x = torch.cat((x, ext_data), dim=1)
        return x


class MiniGridBeboldEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing, action_sizes):
        super().__init__(cfg, timing)

        from sample_factory.envs.minigrid.minigrid_model import init
        init_ = lambda m: init(m, nn.init.orthogonal_,
           lambda x: nn.init.constant_(x, 0),
           nn.init.calculate_gain('relu'))

        self.feature = Sequential(OrderedDict([
            ('conv0', init_(Conv2d(in_channels=obs_space.spaces['obs'].shape[0], out_channels=32, kernel_size=3, stride=1, padding=1))),
            ('_act0', nn.ELU()),
            ('conv1', init_(Conv2d(in_channels=32, out_channels=128, kernel_size=3, stride=2, padding=1))),
            ('_act1', nn.ELU()),
            ('conv2', init_(Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=2, padding=1))),
            ('_act2', nn.ELU()),
            ('_', nn.Flatten()),
            ('linear0', init_(Linear(2048, 1024))),
            ('_act3', nn.ReLU()),
            ('linear2', init_(Linear(1024, self.cfg.hidden_size))),
            ('_act4', nn.ReLU())
        ]))

        self.encoder_out_size = self.cfg.hidden_size

    def forward(self, obs_dict, params=None):
        x = self.feature(obs_dict['obs'], params=get_child_dict(params, 'feature'))
        return x


class MiniGridAGACEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing, action_sizes):
        super().__init__(cfg, timing)

        from sample_factory.envs.minigrid.minigrid_model import init
        init_ = lambda m: init(m, nn.init.orthogonal_,
           lambda x: nn.init.constant_(x, 0),
           nn.init.calculate_gain('relu'))

        self.feature = Sequential(OrderedDict([
            ('conv0', init_(Conv2d(in_channels=obs_space.spaces['obs'].shape[0], out_channels=32, kernel_size=3, stride=2, padding=4))),
            ('_act0', nn.ELU()),
            ('conv1', init_(Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=4))),
            ('_act1', nn.ELU()),
            ('conv2', init_(Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=4))),
            ('_act2', nn.ELU()),
            ('_', nn.Flatten()),
            ('linear0', init_(Linear(1568, self.cfg.hidden_size))),
        ]))

        self.encoder_out_size = self.cfg.hidden_size

    def forward(self, obs_dict, params=None):
        x = self.feature(obs_dict['obs'], params=get_child_dict(params, 'feature'))
        return x


class PolicyCoreRNN(PolicyCoreBase):
    def __init__(self, cfg, input_size):
        super().__init__(cfg)

        self.cfg = cfg
        self.is_gru = False

        assert cfg.rnn_type == 'lstm', f'Unknown RNN type {cfg.rnn_type}'

        self.core = LSTM(input_size, cfg.hidden_size, cfg.rnn_num_layers)

        self.core_output_size = cfg.hidden_size
        self.rnn_num_layers = cfg.rnn_num_layers

    def forward(self, head_output, rnn_states, dones, is_seq, params=None):
        #is_seq = not torch.is_tensor(head_output)
        if not is_seq:
            head_output = head_output.unsqueeze(0)

        if self.rnn_num_layers > 1:
            rnn_states = rnn_states.view(rnn_states.size(0), self.cfg.rnn_num_layers, -1)
            rnn_states = rnn_states.permute(1, 0, 2)
        else:
            rnn_states = rnn_states.unsqueeze(0)

        h, c = torch.split(rnn_states, self.cfg.hidden_size, dim=2)
        core_params = get_child_dict(params, 'core')
        if self.cfg.packed_seq or (not is_seq):
            x, (h, c) = self.core(head_output, (h.contiguous(), c.contiguous()), params=core_params)
            new_rnn_states = torch.cat((h, c), dim=2)
            h_stack, c_stack = h, c
        else:
            outputs = []
            hs, cs = [], []
            num_trajectories = head_output.size(0) // self.cfg.recurrence
            head_output = head_output.view(num_trajectories, self.cfg.recurrence, -1)  # B x T x D
            is_new_episode = dones.clone().detach().view((-1, self.cfg.recurrence))
            is_new_episode = is_new_episode.roll(1, 1)
            is_new_episode[:, 0] = 0
            for t in range(self.cfg.recurrence):
                h = (1.0 - is_new_episode[:, t]).view(1, -1, 1) * h
                c = (1.0 - is_new_episode[:, t]).view(1, -1, 1) * c
                output, (h, c) = self.core(head_output[:, t, :].unsqueeze(0), (h.contiguous(), c.contiguous()), params=core_params)
                outputs.append(output.squeeze(0))
                hs.append(h)
                cs.append(c)
            x = torch.stack(outputs)    # T x B x D
            x = x.permute(1, 0, 2).flatten(0,1) # (BT) x D
            new_rnn_states = torch.cat((h, c), dim=2)

            h_stack = torch.stack(hs)
            c_stack = torch.stack(cs)

        if not is_seq:
            x = x.squeeze(0)

        if self.rnn_num_layers > 1:
            new_rnn_states = new_rnn_states.permute(1, 0, 2)
            new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
        else:
            new_rnn_states = new_rnn_states.squeeze(0)

        return x, new_rnn_states, (h_stack, c_stack)


class ConvEncoder(EncoderBase):
    class ConvEncoderImpl(nn.Module):
        """
        After we parse all the configuration and figure out the exact architecture of the model,
        we devote a separate module to it to be able to use torch.jit.script (hopefully benefit from some layer
        fusion).
        """
        def __init__(self, activation, conv_filters, fc_layer_size, encoder_extra_fc_layers, obs_shape):
            super(ConvEncoder.ConvEncoderImpl, self).__init__()
            conv_layers = []
            for i, layer in enumerate(conv_filters):
                if layer == 'maxpool_2x2':
                    conv_layers.append((f'_maxpool{i}', nn.MaxPool2d((2, 2))))
                elif isinstance(layer, (list, tuple)):
                    inp_ch, out_ch, filter_size, stride = layer
                    conv_layers.append((f'conv{i}', Conv2d(inp_ch, out_ch, filter_size, stride=stride)))
                    conv_layers.append((f'_', activation))
                else:
                    raise NotImplementedError(f'Layer {layer} not supported!')

            self.conv_head = Sequential(OrderedDict(conv_layers))
            self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)

            fc_layers = []
            for i in range(encoder_extra_fc_layers):
                size = self.conv_head_out_size if i == 0 else fc_layer_size
                fc_layers.append((f'linear{i}', Linear(size, fc_layer_size)))
                fc_layers.append((f'_', activation))

            self.fc_layers = Sequential(OrderedDict(fc_layers))

        def forward(self, obs, params=None):
            x = self.conv_head(obs, params=get_child_dict(params, 'conv_head'))
            x = x.contiguous().view(-1, self.conv_head_out_size)
            x = self.fc_layers(x, params=get_child_dict(params, 'fc_layers'))
            return x

    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        obs_shape = get_obs_shape(obs_space)
        input_ch = obs_shape.obs[0]
        log.debug('Num input channels: %d', input_ch)

        if cfg.encoder_subtype == 'convnet_simple':
            conv_filters = [[input_ch, 32, 8, 4], [32, 64, 4, 2], [64, 128, 3, 2]]
        elif cfg.encoder_subtype == 'convnet_impala':
            conv_filters = [[input_ch, 16, 8, 4], [16, 32, 4, 2]]
        elif cfg.encoder_subtype == 'minigrid_convnet_tiny':
            conv_filters = [[3, 16, 3, 1], [16, 32, 2, 1], [32, 64, 2, 1]]
        else:
            raise NotImplementedError(f'Unknown encoder {cfg.encoder_subtype}')

        activation = nonlinearity(self.cfg)
        fc_layer_size = fc_after_encoder_size(self.cfg)
        encoder_extra_fc_layers = self.cfg.encoder_extra_fc_layers

        self.enc = self.ConvEncoderImpl(activation, conv_filters, fc_layer_size, encoder_extra_fc_layers, obs_shape)

        self.encoder_out_size = calc_num_elements(self.enc, obs_shape.obs)
        log.debug('Encoder output size: %r', self.encoder_out_size)

    def forward(self, obs_dict, params=None):
        return self.enc(obs_dict['obs'], params=get_child_dict(params, 'enc'))


from sample_factory.algorithms.appo.model_utils import ActionsParameterizationBase, calc_num_logits, get_action_distribution
class ActionParameterizationDefault(ActionsParameterizationBase):
    """
    A single fully-connected layer to output all parameters of the action distribution. Suitable for
    categorical action distributions, as well as continuous actions with learned state-dependent stddev.

    """

    def __init__(self, cfg, core_out_size, action_space, layers_dim=None):
        super().__init__(cfg, action_space)

        num_action_outputs = calc_num_logits(action_space)
        if layers_dim is None:
            self.distribution_linear = Linear(core_out_size, num_action_outputs)
        else:
            num_neurons = [core_out_size] + list(layers_dim) + [num_action_outputs]
            num_neurons = zip(num_neurons[:-1], num_neurons[1:])
            layers = []
            for i, in_dim, out_dim in enumerate(num_neurons):
                layers.extend([
                    (f'linear{i}', Linear(in_dim, out_dim)),
                    (f'_act{i}', nn.ReLU())
                ])
            layers.pop()  # remove last activation
            self.distribution_linear = Sequential(OrderedDict(layers))

    def forward(self, actor_core_output, params=None):
        """Just forward the FC layer and generate the distribution object."""
        action_distribution_params = self.distribution_linear(actor_core_output, params=get_child_dict(params, 'distribution_linear'))
        action_distribution = get_action_distribution(self.action_space, raw_logits=action_distribution_params)
        return action_distribution_params, action_distribution





def update_params_by_sgd(params, grads, optim=None, lr=None):
    assert lr is not None
    updated_params = OrderedDict()
    for (name, param), grad in zip(params.items(), grads):
        if grad is None:
            continue
        updated_params[name] = param.add(grad, alpha=-lr)
    return updated_params


def update_params_by_rms(params, grads, optim=None, lr=None):
    assert optim is not None
    updated_params = OrderedDict()
    for group in optim.param_groups:
        lr = group['lr']
        alpha = group['alpha']
        eps = group['eps']
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        centered = group['centered']

        for i, (name, p) in enumerate(params.items()):
            grad = grads[i]
            if grad is None:
                continue
            state = optim.state[p]

            # State initialization
            if len(state) == 0:
                square_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
                if momentum > 0:
                    buf = torch.zeros_like(p, memory_format=torch.preserve_format)
                if centered:
                    grad_avg = torch.zeros_like(p, memory_format=torch.preserve_format)
            else:
                square_avg = state['square_avg']
                if momentum > 0:
                    buf = state['momentum_buffer']
                if centered:
                    grad_avg = state['grad_avg']

            if weight_decay != 0:
                grad = grad.add(p, alpha=weight_decay)

            square_avg = square_avg.mul(alpha).addcmul(grad, grad, value=1 - alpha)

            if centered:
                grad_avg = grad_avg.mul(alpha).add(grad, alpha=1 - alpha)
                avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt().add(eps)
            else:
                avg = square_avg.sqrt().add(eps)

            if momentum > 0:
                buf = buf.mul(momentum).addcdiv(grad, avg)
                p = p.add(buf, alpha=-lr)
            else:
                p = p.addcdiv(grad, avg, value=-lr)
            updated_params[name] = p
    return updated_params


def update_params_by_adamw(params, grads, optim=None, lr=None, use_adam=False):
    assert optim is not None
    updated_params = OrderedDict()
    for group in optim.param_groups:
        amsgrad = group['amsgrad']
        beta1, beta2 = group['betas']
        lr = group['lr']
        weight_decay = group['weight_decay']
        eps = group['eps']

        for i, (name, p) in enumerate(params.items()):
            grad = grads[i]
            if grad is None:
                continue
            state = optim.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            step = state['step']

            if step == 0:
                step = 1

            if amsgrad:
                max_exp_avg_sq = state['max_exp_avg_sq']

            # Perform stepweight decay
            if weight_decay != 0:
                if use_adam:
                    grad = grad.add(p, alpha=weight_decay)
                else:
                    p = p * (1 - lr * weight_decay)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            # Decay the first and second moment running average coefficient

            exp_avg = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
            exp_avg_sq = exp_avg_sq.mul(beta2).addcmul(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add(eps)
            else:
                # denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add(eps)
                denom = (torch.sqrt(exp_avg_sq + eps) / math.sqrt(bias_correction2)).add(eps)

            step_size = lr / bias_correction1
            updated_params[name] = p.addcdiv(exp_avg, denom, value=-step_size)

    return updated_params


def update_params_by_adam(params, grads, optim=None, lr=None):
    return update_params_by_adamw(params, grads, optim=optim, lr=lr, use_adam=True)


def _gradient_check(module1, module2, input, output_shape, **kwargs):
    params = OrderedDict(module1.named_parameters())
    optim = torch.optim.SGD(params.values(), lr=1e-4)

    out = module1(input, **kwargs)
    if type(out) == tuple:
        out = out[0]

    dummy_target = torch.randn(output_shape)
    loss = F.mse_loss(out, dummy_target)
    grads = torch.autograd.grad(loss, params.values(), retain_graph=True, allow_unused=True)
    updated_params = update_params_by_sgd(params, grads, lr=1e-4)

    optim.zero_grad()
    loss.backward()
    optim.step()

    out_after_update_by_optim = module1(input, **kwargs)
    if type(out_after_update_by_optim) == tuple:
        out_after_update_by_optim = out_after_update_by_optim[0]
    out_after_update_by_grads = module2(input, params=updated_params, **kwargs)
    if type(out_after_update_by_grads) == tuple:
        out_after_update_by_grads = out_after_update_by_grads[0]
    assert torch.abs(out_after_update_by_optim - out_after_update_by_grads).sum().item() == 0


if __name__ == '__main__':
    import argparse
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    p = argparse.ArgumentParser(description='unit test')

    p.add_argument('--nonlinearity', type=str, default='relu', help='')
    p.add_argument('--encoder_subtype', type=str, default='resnet_impala', help='')
    p.add_argument('--dmlab_instr_encoder_device', type=str, default='cpu', help='')
    p.add_argument('--nonlinear_inplace', type=bool, default=False, help='')
    p.add_argument('--hidden_size', type=int, default=512, help='')
    p.add_argument('--encoder_extra_fc_layers', type=int, default=1, help='')
    p.add_argument('--rnn_type', type=str, default='lstm', help='')
    p.add_argument('--rnn_num_layers', type=int, default=2, help='')
    p.add_argument('--packed_seq', type=bool, default=False, help='')
    p.add_argument('--recurrence', type=int, default=96, help='')
    p.add_argument('--encoder_fc_hidden_size', default=-1, type=int,
                   help='Size of hidden layer in the encoder fc, it is set as hidden_size when negative')
    p.add_argument('--reconstruction_loss_coeff', default=1e-2, type=float,
                   help='Reconstruction loss coefficient for normalized input')
    p.add_argument('--reconstruction_loss_type', default='MSE', type=str, choices=['MSE', 'MSE_Inverted', 'CE'],
                   help='Reconstruction loss type (MSE or CE)')
    p.add_argument('--reconstruction_from_core_hidden', default=False, type=bool,
                   help='Reconstruc from core hidden or conv output')
    p.add_argument('--use_long_skip_reconstruction', default=False, type=bool,
                   help='Use long-skip connection in reconstruction or not')
    p.add_argument('--apply_tanh_for_mse_reconstruction', default=True, type=bool,
                   help='Use tanh nonlinearity for mse reconstruction')
    p.add_argument('--encoder_pooling', default='max', choices=['stride', 'max', 'avg'], type=str,
                   help='pooling method in CNN')
    p.add_argument('--no_pooling_last_cnn', default=False, type=bool, help='apply pooling for last cnn or not')

    args = p.parse_args()

    # unit test for ResBlock
    input_ch, output_ch = 32, 32
    model = ResBlock(args, input_ch, output_ch)
    model_ = ResBlock(args, input_ch, output_ch)
    model_.load_state_dict(model.state_dict())

    with torch.no_grad():
        x = torch.randn((1, 32, 2, 3))
        out = model(x)
        out2 = model_(x)
        assert torch.abs(out - out2).sum() == 0
    for i in range(10):
        x = torch.randn((1, 32, 2, 3))
        _gradient_check(model, model_, x, out.shape)


    # unit test for ResNetEncoder
    x = torch.randn((3, 72, 96))
    model = ResnetEncoder(args, x)
    model_ = ResnetEncoder(args, x)
    model_.load_state_dict(model.state_dict())
    obs_dict = {'obs': x.unsqueeze(0)}
    with torch.no_grad():
        out = model(obs_dict)
        out2 = model_(obs_dict)
        assert torch.abs(out - out2).sum() == 0
    for i in range(10):
        obs_dict = {'obs': torch.randn((1, 3, 72, 96))}
        _gradient_check(model, model_, obs_dict, out.shape)


    # unit test for ResnetEncoderDecoder
    x = torch.randn((3, 72, 96))
    model = ResnetEncoderDecoder(args, x)
    model_ = ResnetEncoderDecoder(args, x)
    model_.load_state_dict(model.state_dict())
    obs_dict = {'obs': x.unsqueeze(0)}
    with torch.no_grad():
        out = model(obs_dict)
        out2 = model_(obs_dict)
        assert torch.abs(out - out2).sum() == 0
    for i in range(2):
        obs_dict = {'obs': torch.randn((1, 3, 72, 96))}
        _gradient_check(model, model_, obs_dict, out.shape)


    # unit test for DMLabEncoder
    x = torch.randn((3, 72, 96))
    model = DmlabEncoder(args, x, None, None)
    model_ = DmlabEncoder(args, x, None, None)
    model_.load_state_dict(model.state_dict())

    obs_dict = {
        'obs': x.unsqueeze(0),
        'INSTR': torch.zeros((1, 16)).long()
    }
    with torch.no_grad():
        out = model(obs_dict)
        out2 = model_(obs_dict)
        assert torch.abs(out - out2).sum() == 0

    for i in range(10):
        obs_dict = {
            'obs': torch.randn((1, 3, 72, 96)),
            'INSTR': torch.randint(0, DMLAB_VOCABULARY_SIZE, (1, 16)).long()
        }
        _gradient_check(model, model_, obs_dict, out.shape)

    # unit test for MinigridEncoder
    x = {
        'spaces': {
            'obs': torch.randn((3, 7, 7))
        }
    }
    from sample_factory.utils.utils import AttrDict
    x = AttrDict(x)
    model = MiniGridBeboldEncoder(args, x, None, None)
    model_ = MiniGridBeboldEncoder(args, x, None, None)
    model_.load_state_dict(model.state_dict())
    obs_dict = {
        'obs': x.spaces['obs'].unsqueeze(0),
    }
    with torch.no_grad():
        out = model(obs_dict)
        out2 = model_(obs_dict)
        assert torch.abs(out - out2).sum() == 0
    for i in range(10):
        obs_dict = {
            'obs': torch.randn((1, 3, 7, 7))
        }
        _gradient_check(model, model_, obs_dict, out.shape)

    # unit test for PolicyRNN
    batch_size = 768
    x = torch.randn((batch_size, args.hidden_size))
    states = torch.randn((batch_size // args.recurrence, args.hidden_size * 2 * args.rnn_num_layers))
    model = PolicyCoreRNN(args, args.hidden_size)
    model_ = PolicyCoreRNN(args, args.hidden_size)
    model_.load_state_dict(model.state_dict())
    dones = torch.zeros(batch_size)
    with torch.no_grad():
        out, _, _ = model(x, states, dones, is_seq=True)
        out2, _, _ = model_(x, states, dones, is_seq=True)
        assert torch.abs(out - out2).sum() == 0
    for i in range(5):
        x = torch.randn((batch_size, args.hidden_size))
        states = torch.randn((batch_size // args.recurrence, args.hidden_size * 2 * args.rnn_num_layers))
        _gradient_check(model, model_, x, out.shape, rnn_states=states, dones=dones, is_seq=True)

    # unit test for standard convnet
    x = torch.randn((3, 7, 7))
    args.encoder_subtype = 'minigrid_convnet_tiny'
    model = ConvEncoder(args, x, None)
    model_ = ConvEncoder(args, x, None)
    model_.load_state_dict(model.state_dict())
    with torch.no_grad():
        obs_dict = {
            'obs': x.unsqueeze(0)
        }
        out = model(obs_dict)
        out2 = model_(obs_dict)
        assert torch.abs(out - out2).sum() == 0
    for i in range(10):
        obs_dict = {
            'obs': torch.randn((1, 3, 7, 7))
        }
        _gradient_check(model, model_, obs_dict, out.shape)
