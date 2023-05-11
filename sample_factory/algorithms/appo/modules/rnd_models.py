from torch import nn
from torch.nn import functional as F
import numpy as np
from sample_factory.algorithms.appo.model_utils import ResBlock, nonlinearity


class RNDModel(nn.Module):
    def __init__(self):
        super(RNDModel, self).__init__()

        self.random_enc = False
        feature_output = 7 * 7 * 64
        self.predictor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(feature_output, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

        self.target = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=8,
                stride=4),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(feature_output, 512)
        )

        # for p in self.modules():
        #     if isinstance(p, nn.Conv2d):
        #         nn.init.orthogonal_(p.weight, np.sqrt(2))
        #         p.bias.data.zero_()
        #
        #     if isinstance(p, nn.Linear):
        #         nn.init.orthogonal_(p.weight, np.sqrt(2))
        #         p.bias.data.zero_()

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, next_obs):
        target_feature = self.target(next_obs)
        predict_feature = self.predictor(next_obs)

        return predict_feature, target_feature


class RNDEncoder(nn.Module):
    def __init__(self, hdim, channels=None, shared_encoder=None):
        super(RNDEncoder, self).__init__()
        if shared_encoder is not None:
            self.encoder = shared_encoder
        else:
            assert channels is not None
            feature_output = 5 * 8 * channels[-1]
            self.encoder = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=channels[0],
                    kernel_size=8,
                    stride=4),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=channels[0],
                    out_channels=channels[1],
                    kernel_size=4,
                    stride=2),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=channels[1],
                    out_channels=channels[2],
                    kernel_size=3,
                    stride=1),
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(feature_output, hdim),
                nn.ReLU(),
                nn.Linear(hdim, hdim),
                nn.ReLU()
            )

        self.feature_head = nn.Linear(hdim, hdim)

    def forward(self, obs):
        encoded = self.encoder(obs)
        return self.feature_head(encoded)


class RNDModel4DMLab(RNDModel):
    CHANNELS_DICT = {
        'xsmall': [8, 16, 16],
        'small': [16, 32, 32],
        'medium': [32, 64, 64],
        'large': [64, 128, 128],
        'xlarge': [128, 256, 256]
    }

    def __init__(self, hdim,
                 target_type='medium',
                 predictor_type='medium',
                 use_shared_encoder=False):
        super(RNDModel4DMLab, self).__init__()

        assert target_type in self.CHANNELS_DICT and predictor_type in self.CHANNELS_DICT

        self.hdim = hdim
        target_channels = self.CHANNELS_DICT[target_type]
        predictor_channels = self.CHANNELS_DICT[predictor_type]

        if use_shared_encoder:
            assert target_type == predictor_type

        self.target = self.build(target_channels)
        shared_encoder = None
        if use_shared_encoder:
            shared_encoder = self.target.encoder
        self.predictor = self.build(predictor_channels, shared_encoder=shared_encoder)

        for param in self.target.parameters():
            param.requires_grad = False

    def build(self, channels, shared_encoder=None):
        net = RNDEncoder(self.hdim, channels=channels, shared_encoder=shared_encoder)
        return net

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class RNDModel4MiniGrid(nn.Module):
    def __init__(self, cfg, obs_space):
        super(RNDModel4MiniGrid, self).__init__()

        self.cfg = cfg
        self.random_enc = True

        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))

        self.encoder = nn.Sequential(
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

        self.predictor = nn.Sequential(
            nn.Linear(self.cfg.hidden_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        self.target = nn.Sequential(
            nn.Linear(self.cfg.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        '''
        for p in self.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.orthogonal_(p.weight, nn.init.calculate_gain('relu'))
                p.bias.data.zero_()

            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, nn.init.calculate_gain('relu'))
                p.bias.data.zero_()
        '''
        for param in self.target.parameters():
            param.requires_grad = False