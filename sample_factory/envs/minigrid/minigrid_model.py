import torch

from torch import nn

from sample_factory.algorithms.appo.model_utils import create_standard_encoder, EncoderBase, register_custom_encoder
from sample_factory.utils.utils import log
import numpy as np
from sample_factory.algorithms.appo.modules.meta_modules import MiniGridBeboldEncoder as MiniGridBeboldEncoderMeta
from sample_factory.algorithms.appo.modules.meta_modules import MiniGridAGACEncoder as MiniGridAGACEncoderMeta


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class MiniGridBeboldEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing, action_sizes):
        super().__init__(cfg, timing)

        init_ = lambda m: init(m, nn.init.orthogonal_,
           lambda x: nn.init.constant_(x, 0),
           nn.init.calculate_gain('relu'))

        self.feature = nn.Sequential(
            init_(nn.Conv2d(
                in_channels=obs_space.spaces['obs'].shape[0],
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(
                in_channels=32,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1)),
            nn.ELU(),
            init_(nn.Conv2d(
                in_channels=128,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1)),
            nn.ELU(),
            nn.Flatten(),
            init_(nn.Linear(
                2048,
                1024)),
            nn.ReLU(),
            init_(nn.Linear(
                1024,
                self.cfg.hidden_size)),
            nn.ReLU()
        )

        self.encoder_out_size = self.cfg.hidden_size

    def forward(self, obs_dict):
        x = self.feature(obs_dict['obs'])
        return x

class MiniGridAGACEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing, action_sizes):
        super().__init__(cfg, timing)

        init_ = lambda m: init(m, nn.init.orthogonal_,
           lambda x: nn.init.constant_(x, 0),
           nn.init.calculate_gain('relu'))

        self.feature = nn.Sequential(
            init_(nn.Conv2d(
                in_channels=obs_space.spaces['obs'].shape[0],
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=4)),
            nn.ELU(),
            init_(nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=4)),
            nn.ELU(),
            init_(nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=4)),
            nn.ELU(),
            nn.Flatten(),
            init_(nn.Linear(
                1568,
                self.cfg.hidden_size)),
            nn.ReLU()
        )

        self.encoder_out_size = self.cfg.hidden_size

    def forward(self, obs_dict):
        x = self.feature(obs_dict['obs'])
        return x


def minigrid_register_models():
    register_custom_encoder('minigrid_beboldnet', MiniGridBeboldEncoder)
    register_custom_encoder('minigrid_beboldnet_meta', MiniGridBeboldEncoderMeta)
    register_custom_encoder('minigrid_agacnet', MiniGridAGACEncoder)
    register_custom_encoder('minigrid_agacnet_meta', MiniGridAGACEncoderMeta)