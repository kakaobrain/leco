import math
import copy
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from collections import OrderedDict

from sample_factory.algorithms.utils.action_distributions import calc_num_logits, get_action_distribution, is_continuous_action_space
from sample_factory.algorithms.utils.algo_utils import EPS
from sample_factory.algorithms.utils.pytorch_utils import calc_num_elements
from sample_factory.utils.utils import AttrDict, use_lirpg
from sample_factory.utils.utils import log

from sample_factory.algorithms.appo.transformer import *

import torchvision


# register custom encoders
ENCODER_REGISTRY = dict()


def register_custom_encoder(custom_encoder_name, encoder_cls):
    assert issubclass(encoder_cls, EncoderBase), 'Custom encoders must be derived from EncoderBase'
    assert custom_encoder_name not in ENCODER_REGISTRY

    log.debug('Adding model class %r to registry (with name %s)', encoder_cls, custom_encoder_name)
    ENCODER_REGISTRY[custom_encoder_name] = encoder_cls


def get_hidden_size(cfg):
    if cfg.use_rnn:
        size = cfg.hidden_size * cfg.rnn_num_layers
    else:
        size = 1

    if cfg.rnn_type == 'lstm':
        size *= 2

    if not cfg.actor_critic_share_weights:
        # actor and critic need separate states
        size *= 2

    return size


def get_basic_encoder_weights(state_dict):
    encoder_weights = OrderedDict()
    for key in state_dict:
        if 'encoder.basic_encoder.' in key:
            _key = key.replace('encoder.basic_encoder.', '')
            encoder_weights[_key] = state_dict[key]
    return encoder_weights


def fc_after_encoder_size(cfg):
    return cfg.hidden_size  # make configurable?


def nonlinearity(cfg):
    if cfg.nonlinearity == 'elu':
        return nn.ELU(inplace=cfg.nonlinear_inplace)
    elif cfg.nonlinearity == 'relu':
        return nn.ReLU(inplace=cfg.nonlinear_inplace)
    elif cfg.nonlinearity == 'tanh':
        return nn.Tanh()
    else:
        raise Exception('Unknown nonlinearity')


def get_obs_shape(obs_space):
    obs_shape = AttrDict()
    if hasattr(obs_space, 'spaces'):
        for key, space in obs_space.spaces.items():
            obs_shape[key] = space.shape
    else:
        obs_shape.obs = obs_space.shape

    return obs_shape


def normalize_obs(obs_dict, cfg):
    with torch.no_grad():
        mean = cfg.obs_subtract_mean
        scale = cfg.obs_scale

        if obs_dict['obs'].dtype != torch.float:
            obs_dict['obs'] = obs_dict['obs'].float()

        if abs(mean) > EPS:
            obs_dict['obs'].sub_(mean)

        if abs(scale - 1.0) > EPS:
            obs_dict['obs'].mul_(1.0 / scale)


def normalize_obs_return(obs_dict, cfg, half=False):
    with torch.no_grad():
        mean = cfg.obs_subtract_mean
        scale = cfg.obs_scale

        normalized_obs_dict = copy.deepcopy(obs_dict)

        if normalized_obs_dict['obs'].dtype != torch.float:
            normalized_obs_dict['obs'] = normalized_obs_dict['obs'].float()

        if abs(mean) > EPS:
            normalized_obs_dict['obs'].sub_(mean)

        if abs(scale - 1.0) > EPS:
            normalized_obs_dict['obs'].mul_(1.0 / scale)

        if half:
            normalized_obs_dict['obs'] = normalized_obs_dict['obs'].half()

    return normalized_obs_dict

class EncoderBase(nn.Module):
    def __init__(self, cfg, timing):
        super().__init__()

        self.cfg = cfg
        self.timing = timing

        self.fc_after_enc = None
        self.encoder_out_size = -1  # to be initialized in the constuctor of derived class

    def get_encoder_out_size(self):
        return self.encoder_out_size

    # resnet_coberl
    def init_fc_blocks_coberl(self, input_size):
        layers = []

        fc_layer_size = fc_after_encoder_size(self.cfg)

        # for coberl encoder,
        fc_layer_size = [512, 432]

        for i in range(self.cfg.encoder_extra_fc_layers):
            size = input_size if i == 0 else fc_layer_size[i-1]

            layers.extend([
                nn.Linear(size, fc_layer_size[i]),
                nonlinearity(self.cfg),
            ])

        if len(layers) > 0:
            self.fc_after_enc = nn.Sequential(*layers)
            #self.encoder_out_size = fc_layer_size
            self.encoder_out_size = fc_layer_size[-1]
        else:
            self.encoder_out_size = input_size

    # resnet_impala
    def init_fc_blocks(self, input_size):
        layers = []
        fc_layer_size = fc_after_encoder_size(self.cfg)
        fc_layer_size = self.cfg.encoder_fc_hidden_size if self.cfg.encoder_fc_hidden_size > 0 else fc_layer_size

        for i in range(self.cfg.encoder_extra_fc_layers):
            size = input_size if i == 0 else fc_layer_size
            is_last_layer = i == self.cfg.encoder_extra_fc_layers - 1
            fc_layer_size = self.cfg.hidden_size if is_last_layer else fc_layer_size

            layers.extend([
                nn.Linear(size, fc_layer_size),
                nonlinearity(self.cfg),
            ])

        if len(layers) > 0:
            self.fc_after_enc = nn.Sequential(*layers)
            self.encoder_out_size = fc_layer_size
        else:
            self.encoder_out_size = input_size

    def model_to_device(self, device):
        """Default implementation, can be overridden in derived classes."""
        self.to(device)

    def device_and_type_for_input_tensor(self, _):
        """Default implementation, can be overridden in derived classes."""
        return self.model_device(), torch.float32

    def model_device(self):
        return next(self.parameters()).device

    def forward_fc_blocks(self, x):
        if self.fc_after_enc is not None:
            x = self.fc_after_enc(x)

        return x


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
            for layer in conv_filters:
                if layer == 'maxpool_2x2':
                    conv_layers.append(nn.MaxPool2d((2, 2)))
                elif isinstance(layer, (list, tuple)):
                    inp_ch, out_ch, filter_size, stride = layer
                    conv_layers.append(nn.Conv2d(inp_ch, out_ch, filter_size, stride=stride))
                    conv_layers.append(activation)
                else:
                    raise NotImplementedError(f'Layer {layer} not supported!')

            self.conv_head = nn.Sequential(*conv_layers)
            self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)

            fc_layers = []
            for i in range(encoder_extra_fc_layers):
                size = self.conv_head_out_size if i == 0 else fc_layer_size
                fc_layers.extend([nn.Linear(size, fc_layer_size), activation])

            self.fc_layers = nn.Sequential(*fc_layers)

        def forward(self, obs):
            x = self.conv_head(obs)
            x = x.contiguous().view(-1, self.conv_head_out_size)
            x = self.fc_layers(x)
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

        enc = self.ConvEncoderImpl(activation, conv_filters, fc_layer_size, encoder_extra_fc_layers, obs_shape)
        self.enc = torch.jit.script(enc)

        self.encoder_out_size = calc_num_elements(self.enc, obs_shape.obs)
        log.debug('Encoder output size: %r', self.encoder_out_size)

    def forward(self, obs_dict):
        return self.enc(obs_dict['obs'])


class ResBlock(nn.Module):
    def __init__(self, cfg, input_ch, output_ch, timing):
        super().__init__()

        self.timing = timing

        layers = [
            nonlinearity(cfg),
            nn.Conv2d(input_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
            nonlinearity(cfg),
            nn.Conv2d(output_ch, output_ch, kernel_size=3, stride=1, padding=1),  # padding SAME
        ]

        self.res_block_core = nn.Sequential(*layers)

    def forward(self, x):
        with self.timing.timeit('res_block'):
            identity = x
            out = self.res_block_core(x)
            with self.timing.timeit('res_block_plus'):
                out = out + identity
            return out


class BottleneckBlock(nn.Module):
    def __init__(self, input_ch, inplane, output_ch, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_ch, inplane, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(inplane, inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(inplane, output_ch, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if residual.size() != out.size():
            print(out.size(), residual.size())
        out += residual
        out = self.relu(out)

        return out


class ResnetEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing, action_sizes):
        super().__init__(cfg, timing)

        self.cfg = cfg

        obs_shape = get_obs_shape(obs_space)
        input_ch = obs_shape.obs[0]
        log.debug('Num input channels: %d', input_ch)

        if cfg.encoder_subtype == 'resnet_18':
            import timm
            self.encoder = timm.create_model('resnet18', pretrained=False, num_classes=0)
            self.encoder_out_size = 512
        elif cfg.encoder_subtype == 'resnet_50':
            import timm
            self.encoder = timm.create_model('resnet50', pretrained=False, num_classes=0)
            self.encoder_out_size = 2048
        else:
            if cfg.encoder_subtype == 'resnet_impala':
                # configuration from the IMPALA paper
                resnet_conf = [[16, 2], [32, 2], [32, 2]]
            elif cfg.encoder_subtype == 'resnet_impala_large':
                resnet_conf = [[32, 2], [64, 2], [64, 2]]
            elif cfg.encoder_subtype == 'resnet_coberl':
                # configurtion from the coberl paper
                resnet_conf = [[16, 2], [32, 4], [64, 6], [128, 2]]
            else:
                raise NotImplementedError(f'Unknown resnet subtype {cfg.encoder_subtype}')

            curr_input_channels = input_ch
            layers = []
            if 'resnet_impala' in cfg.encoder_subtype:
                for i, (out_channels, res_blocks) in enumerate(resnet_conf):
                    if self.cfg.encoder_pooling == 'stride':
                        enc_stride = 2
                        pool = nn.Identity
                    else:
                        enc_stride = 1
                        pool = nn.MaxPool2d if self.cfg.encoder_pooling == 'max' else nn.AvgPool2d
                    layers.extend([
                        nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=enc_stride, padding=1),  # padding SAME
                        pool(kernel_size=3, stride=2, padding=1),  # padding SAME
                    ])

                    for j in range(res_blocks):
                        layers.append(ResBlock(cfg, out_channels, out_channels, self.timing))

                    curr_input_channels = out_channels

                layers.append(nonlinearity(cfg))
            elif cfg.encoder_subtype == 'resnet_coberl':
                for i, (inplane, res_blocks) in enumerate(resnet_conf):
                    layers.append(nn.Conv2d(curr_input_channels, inplane * 4, kernel_size=3, stride=2, padding=1))
                    input_ch = inplane * 4
                    output_ch = inplane * 4
                    for j in range(res_blocks):
                        layers.append(BottleneckBlock(input_ch, inplane, output_ch, downsample=None))

                    # group norm w/ group size 8
                    layers.append(nn.GroupNorm(8, output_ch))
                    curr_input_channels = output_ch
            else:
                raise NotImplementedError(f'Unknown resnet subtype {cfg.encoder_subtype}')

            self.conv_head = nn.Sequential(*layers)
            self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape.obs)

            log.debug('Convolutional layer output size: %r', self.conv_head_out_size)

            # if cfg.encoder_subtype == 'resnet_impala':
            if 'resnet_impala' in cfg.encoder_subtype:
                self.init_fc_blocks(self.conv_head_out_size)
            elif cfg.encoder_subtype == 'resnet_coberl':
                self.init_fc_blocks_coberl(self.conv_head_out_size)
            else:
                NotImplementedError(f'Unknown resnet subtype {cfg.encoder_subtype}')

        if  cfg.encoder_subtype == 'resnet_18' or cfg.encoder_subtype == 'resnet_50':
            self.transforms = torchvision.transforms.Resize([224, 224])


    def forward(self, obs_dict):
        if self.cfg.encoder_subtype == 'resnet_18' or self.cfg.encoder_subtype == 'resnet_50':
            # resize to 224 x 224
            x = self.encoder(self.transforms(obs_dict['obs']))
        else:
            x = self.conv_head(obs_dict['obs'])
            x = x.contiguous().view(-1, self.conv_head_out_size)

            x = self.forward_fc_blocks(x)
        return x


class ResnetEncoderDecoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing, action_sizes):
        super().__init__(cfg, timing)
        self.cfg = cfg
        self.is_learner_worker = False # default value

        obs_shape = get_obs_shape(obs_space)

        if cfg.encoder_subtype == 'resnet_18':
            raise NotImplementedError(f"Decoder is not Implemented for resnet_18")
        elif cfg.encoder_subtype == 'resnet_50':
            raise NotImplementedError(f"Decoder is not Implemented for resnet_50")
        else:
            if cfg.encoder_subtype == 'resnet_impala':
                # configuration from the IMPALA paper
                resnet_conf = [[16, 2], [32, 2], [32, 2]]
            elif cfg.encoder_subtype == 'resnet_impala_small':
                resnet_conf = [[16, 2], [32, 2]]
            elif cfg.encoder_subtype == 'resnet_impala_large':
                resnet_conf = [[32, 2], [64, 2], [64, 2]]
            elif cfg.encoder_subtype == 'resnet_coberl':
                # configurtion from the coberl paper
                resnet_conf = [[16, 2], [32, 4], [64, 6], [128, 2]]
            else:
                raise NotImplementedError(f'Unknown resnet subtype {cfg.encoder_subtype}')

            layers_encoder = list()
            layers_decoder = list()

            curr_input_channels = obs_shape.obs[0]
            if cfg.encoder_subtype in ['resnet_impala', 'resnet_impala_small', 'resnet_impala_large']:
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
                        nn.Conv2d(curr_input_channels, out_channels, kernel_size=3, stride=stride_enc, padding=1),
                        pool
                    ])
                    for j in range(res_blocks):
                        _layer_encoder.append(ResBlock(cfg, out_channels, out_channels, self.timing))
                    layer_encoder = nn.Sequential(*_layer_encoder)
                    layers_encoder.append(layer_encoder)

                    _layer_decoder = list()
                    output_paddig = 1 if stride_dec == 2 else 0
                    _layer_decoder.append(
                        nn.ConvTranspose2d(out_channels, curr_input_channels, kernel_size=3, stride=stride_dec,
                                           padding=1, output_padding=output_paddig)
                    )
                    for j in range(res_blocks):
                        _layer_decoder.append(ResBlock(cfg, curr_input_channels, curr_input_channels, self.timing))
                    layer_decoder = nn.Sequential(*_layer_decoder)
                    layers_decoder.append(layer_decoder)

                    curr_input_channels = out_channels

            elif cfg.encoder_subtype == 'resnet_coberl':
                for i, (inplane, res_blocks) in enumerate(resnet_conf):
                    _layer_encoder = list()
                    _layer_encoder.append(nn.Conv2d(curr_input_channels, inplane * 4, kernel_size=3, stride=2, padding=1))
                    input_ch = inplane * 4
                    output_ch = inplane * 4
                    for j in range(res_blocks):
                        _layer_encoder.append(BottleneckBlock(input_ch, inplane, output_ch, downsample=None))

                    # group norm w/ group size 8
                    _layer_encoder.append(nn.GroupNorm(output_ch // 8, output_ch))
                    layer_encoder = nn.Sequential(*_layer_encoder)
                    layers_encoder.append(layer_encoder)

                    _layer_decoder = list()
                    # group norm w/ group size 8
                    _layer_decoder.append(nn.GroupNorm(output_ch // 8, input_ch))
                    input_ch = inplane * 4
                    output_ch = inplane * 4
                    for j in range(res_blocks):
                        _layer_decoder.append(BottleneckBlock(output_ch, inplane, input_ch, downsample=None))
                    output_padding = (0, 1) if i == len(resnet_conf) - 1 else 1 # Dimension matching setting
                    _layer_decoder.append(
                        nn.ConvTranspose2d(inplane * 4, curr_input_channels, kernel_size=3, stride=2,
                                           padding=1, output_padding=output_padding))
                    layer_decoder = nn.Sequential(*_layer_decoder)
                    layers_decoder.append(layer_decoder)

                    curr_input_channels = output_ch

            layers_decoder.reverse()
            self.layers_encoder = nn.ModuleList(layers_encoder)
            self.layers_decoder = nn.ModuleList(layers_decoder)
            self.conv_head_out_size = calc_num_elements(nn.Sequential(*self.layers_encoder), obs_shape.obs)

            if cfg.encoder_subtype in ['resnet_impala', 'resnet_impala_small', 'resnet_impala_large', 'resnet_coberl']:
                self.init_fc_blocks(self.conv_head_out_size)

            if cfg.reconstruction_from_core_hidden:
                self.init_decoder_fc_blocks(self.conv_head_out_size)

    # resnet_impala
    def init_decoder_fc_blocks(self, enc_size):
        layers = []

        for i in range(1):
            size = self.cfg.hidden_size

            layers.extend([
                nn.Linear(size, enc_size),
                nonlinearity(self.cfg),
            ])
        self.fc_decoder = nn.Sequential(*layers)

    def set_learner_worker(self):
        self.is_learner_worker = True

    def forward(self, obs_dict):
        if self.cfg.encoder_subtype == 'resnet_18' or self.cfg.encoder_subtype == 'resnet_50':
            raise NotImplementedError("Not implemented for resnet_18 or resent_50")
        elif self.cfg.encoder_subtype in ['resnet_impala', 'resnet_impala_small', 'resnet_impala_large']:
            x = obs_dict['obs']

            check = torch.all(torch.logical_and(torch.all(obs_dict['obs'] >= -1),
                                                torch.all(obs_dict['obs'] <= 1)))
            assert check, "Observations should be normalized for deconvolution"

            x_enc = x
            encodings = list()
            for i in range(len(self.layers_encoder)):
                layer_encoder = self.layers_encoder[i]
                x_enc = layer_encoder(x_enc)
                encodings.append(x_enc)
            x_enc = nonlinearity(self.cfg)(x_enc)

            x_dec = x_enc
            if self.cfg.reconstruction_from_core_hidden:
                # When reconstruct from core hidden, we apply fc_after_enc before decoing
                shape_enc = x_enc.shape
                x_enc = x_enc.contiguous().view(-1, self.conv_head_out_size)
                x = self.forward_fc_blocks(x_enc)

            if self.is_learner_worker: # We reconstruct only for learner_worker
                if self.cfg.reconstruction_from_core_hidden:
                    x_dec = self.fc_decoder(x).reshape(shape_enc)

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

        elif self.cfg.encoder_subtype == 'resnet_coberl':
            x = obs_dict['obs'] # normalized obs
            x_enc = x
            for i in range(len(self.layers_encoder)):
                layer_encoder = self.layers_encoder[i]
                x_enc = layer_encoder(x_enc)
            x_enc = nonlinearity(self.cfg)(x_enc)

            x_dec = x_enc
            if self.cfg.reconstruction_from_core_hidden:
                # When reconstruct from core hidden, we apply fc_after_enc before decoing
                shape_enc = x_enc.shape
                x_enc = x_enc.contiguous().view(-1, self.conv_head_out_size)
                x = self.forward_fc_blocks(x_enc)

            if self.is_learner_worker:  # We reconstruct only for learner_worker
                if self.cfg.reconstruction_from_core_hidden:
                    x_dec = self.fc_decoder(x).reshape(shape_enc)

                for i in range(len(self.layers_decoder)):
                    layer_decoder = self.layers_decoder[i]
                    x_dec = layer_decoder(x_dec)

                if self.cfg.reconstruction_loss_type == 'MSE' and self.cfg.apply_tanh_for_mse_reconstruction:
                    x_dec = torch.tanh(x_dec)
                elif self.cfg.reconstruction_loss_type == 'CE':
                    x_dec = torch.sigmoid(x_dec)
                self.x_dec = x_dec

        if not self.cfg.reconstruction_from_core_hidden:
            # When we do not reconstruct from core hidden, we apply fc_after_enc after decoding
            x_enc = x_enc.contiguous().view(-1, self.conv_head_out_size)
            x = self.forward_fc_blocks(x_enc)

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


class MlpEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        obs_shape = get_obs_shape(obs_space)
        assert len(obs_shape.obs) == 1

        if cfg.encoder_subtype == 'mlp_mujoco':
            fc_encoder_layer = cfg.hidden_size
            encoder_layers = [
                nn.Linear(obs_shape.obs[0], fc_encoder_layer),
                nonlinearity(cfg),
                nn.Linear(fc_encoder_layer, fc_encoder_layer),
                nonlinearity(cfg),
            ]
        else:
            raise NotImplementedError(f'Unknown mlp encoder {cfg.encoder_subtype}')

        self.mlp_head = nn.Sequential(*encoder_layers)
        self.init_fc_blocks(fc_encoder_layer)

    def forward(self, obs_dict):
        x = self.mlp_head(obs_dict['obs'])
        x = self.forward_fc_blocks(x)
        return x

def fc_layer(in_features, out_features, bias=True, spec_norm=False):
    if spec_norm:
        return spectral_norm(nn.Linear(in_features, out_features, bias))
    return nn.Linear(in_features, out_features, bias)

def create_encoder(cfg, obs_space, timing, action_sizes, is_learner_worker=False):
    if cfg.encoder_custom:
        encoder_name = cfg.encoder_custom
        if use_lirpg(cfg):
            encoder_name += '_meta'
        encoder_cls = ENCODER_REGISTRY[encoder_name]
        encoder = encoder_cls(cfg, obs_space, timing, action_sizes)
        if use_lirpg(cfg) and not is_learner_worker:
            encoder.cudnn_enable = True
    else:
        encoder = create_standard_encoder(cfg, obs_space, timing, action_sizes)

    return encoder


def create_standard_encoder(cfg, obs_space, timing, action_sizes):
    if cfg.encoder_type == 'conv':
        if use_lirpg(cfg):
            from sample_factory.algorithms.appo.modules.meta_modules import ConvEncoder as ConvEncoderMeta
            encoder = ConvEncoderMeta(cfg, obs_space, timing)
        else:
            encoder = ConvEncoder(cfg, obs_space, timing)
    elif cfg.encoder_type == 'resnet':
        encoder = ResnetEncoder(cfg, obs_space, timing, action_sizes)
    elif cfg.encoder_type == 'mlp':
        encoder = MlpEncoder(cfg, obs_space, timing)
    elif cfg.encoder_type == 'resnet_encoder_decoder':
        if use_lirpg(cfg):
            from sample_factory.algorithms.appo.modules.meta_modules import ResnetEncoderDecoder as ResnetEncoderDecoderMeta
            encoder = ResnetEncoderDecoderMeta(cfg, obs_space)
        else:
            encoder = ResnetEncoderDecoder(cfg, obs_space, timing, action_sizes)
    else:
        raise Exception('Encoder type not supported')

    return encoder


class PolicyCoreBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.core_output_size = -1

    def get_core_out_size(self):
        return self.core_output_size


class PolicyCoreRNN(PolicyCoreBase):
    def __init__(self, cfg, input_size):
        super().__init__(cfg)

        self.cfg = cfg
        self.is_gru = False

        if cfg.rnn_type == 'gru':
            self.core = nn.GRU(input_size, cfg.hidden_size, cfg.rnn_num_layers)
            self.is_gru = True
        elif cfg.rnn_type == 'lstm':
            self.core = nn.LSTM(input_size, cfg.hidden_size, cfg.rnn_num_layers)
        else:
            raise RuntimeError(f'Unknown RNN type {cfg.rnn_type}')

        self.core_output_size = cfg.hidden_size
        self.rnn_num_layers = cfg.rnn_num_layers

    def forward(self, head_output, rnn_states, dones, is_seq):
        #is_seq = not torch.is_tensor(head_output)
        if not is_seq:
            head_output = head_output.unsqueeze(0)

        if self.rnn_num_layers > 1:
            rnn_states = rnn_states.view(rnn_states.size(0), self.cfg.rnn_num_layers, -1)
            rnn_states = rnn_states.permute(1, 0, 2)
        else:
            rnn_states = rnn_states.unsqueeze(0)

        if self.is_gru:
            x, new_rnn_states = self.core(head_output, rnn_states.contiguous())
        else:
            h, c = torch.split(rnn_states, self.cfg.hidden_size, dim=2)
            if self.cfg.packed_seq or (not is_seq):
                x, (h, c) = self.core(head_output, (h.contiguous(), c.contiguous()))
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
                    output, (h, c) = self.core(head_output[:, t, :].unsqueeze(0), (h.contiguous(), c.contiguous()))
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


class PolicyCoreFeedForward(PolicyCoreBase):
    """A noop core (no recurrency)."""

    def __init__(self, cfg, input_size):
        super().__init__(cfg)
        self.cfg = cfg
        self.core_output_size = input_size

    def forward(self, head_output, fake_rnn_states, dones, is_seq):
        return head_output, fake_rnn_states, head_output


def create_core(cfg, core_input_size):
    if cfg.use_rnn:
        if use_lirpg(cfg):
            from sample_factory.algorithms.appo.modules.meta_modules import PolicyCoreRNN as PolicyCoreRNNMeta
            core = PolicyCoreRNNMeta(cfg, core_input_size)
        else:
            core = PolicyCoreRNN(cfg, core_input_size)
    elif cfg.use_transformer:
        if use_lirpg(cfg):
            from sample_factory.algorithms.appo.modules.meta_transformer import MemTransformerLM as MemTransformerLMMeta
            core = MemTransformerLMMeta(cfg, n_token=None, n_layer=cfg.n_layer, n_head=cfg.n_heads,
                                    d_head=cfg.d_head, d_model=core_input_size, d_inner=cfg.d_inner,
                                    dropout=0.0, dropatt=0.0, mem_len=cfg.mem_len,
                                    use_stable_version=True, use_gate=cfg.use_gate)
        else:
            core = MemTransformerLM(cfg, n_token=None, n_layer=cfg.n_layer, n_head=cfg.n_heads,
                                    d_head=cfg.d_head, d_model=core_input_size, d_inner=cfg.d_inner,
                                    dropout=0.0, dropatt=0.0, mem_len=cfg.mem_len,
                                    use_stable_version=True, use_gate=cfg.use_gate)
    else:
        core = PolicyCoreFeedForward(cfg, core_input_size)
    return core


class ActionsParameterizationBase(nn.Module):
    def __init__(self, cfg, action_space):
        super().__init__()
        self.cfg = cfg
        self.action_space = action_space


class ActionParameterizationDefault(ActionsParameterizationBase):
    """
    A single fully-connected layer to output all parameters of the action distribution. Suitable for
    categorical action distributions, as well as continuous actions with learned state-dependent stddev.

    """

    def __init__(self, cfg, core_out_size, action_space, layers_dim=None):
        super().__init__(cfg, action_space)

        num_action_outputs = calc_num_logits(action_space)
        if layers_dim is None:
            self.distribution_linear = nn.Linear(core_out_size, num_action_outputs)
        else:
            num_neurons = [core_out_size] + list(layers_dim) + [num_action_outputs]
            num_neurons = zip(num_neurons[:-1], num_neurons[1:])
            layers = []
            # self._layers_without_activations = []
            for in_dim, out_dim in num_neurons:
                layer = nn.Linear(in_dim, out_dim)
                # self._layers_without_activations.append(layer)
                layers.append(layer)
                layers.append(nn.ReLU())
            layers.pop()  # remove last activation
            self.distribution_linear = nn.Sequential(*layers)

    def forward(self, actor_core_output):
        """Just forward the FC layer and generate the distribution object."""
        action_distribution_params = self.distribution_linear(actor_core_output)
        action_distribution = get_action_distribution(self.action_space, raw_logits=action_distribution_params)
        return action_distribution_params, action_distribution


class ActionParameterizationContinuousNonAdaptiveStddev(ActionsParameterizationBase):
    """Use a single learned parameter for action stddevs."""

    def __init__(self, cfg, core_out_size, action_space):
        super().__init__(cfg, action_space)

        assert not cfg.adaptive_stddev
        assert is_continuous_action_space(self.action_space), \
            'Non-adaptive stddev makes sense only for continuous action spaces'

        num_action_outputs = calc_num_logits(action_space)

        # calculate only action means using the policy neural network
        self.distribution_linear = nn.Linear(core_out_size, num_action_outputs // 2)

        # stddev is a single learned parameter
        initial_stddev = torch.empty([num_action_outputs // 2])
        initial_stddev.fill_(math.log(self.cfg.initial_stddev))
        self.learned_stddev = nn.Parameter(initial_stddev, requires_grad=True)

    def forward(self, actor_core_output):
        action_means = self.distribution_linear(actor_core_output)

        batch_size = action_means.shape[0]
        action_stddevs = self.learned_stddev.repeat(batch_size, 1)
        action_distribution_params = torch.cat((action_means, action_stddevs), dim=1)
        action_distribution = get_action_distribution(self.action_space, raw_logits=action_distribution_params)
        return action_distribution_params, action_distribution
