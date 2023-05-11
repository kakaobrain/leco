import torch
from torch import nn
from torch.nn import functional as F


class VAEModel4DMLab(nn.Module):
    def __init__(self, cfg, num_actions):
        super(VAEModel4DMLab, self).__init__()

        self.num_layers = 4
        self.num_filters = 32
        self.channel = 3 + num_actions
        self.output_dim_h = 9
        self.output_dim_w = 12

        self.output_logits = False
        self.feature_dim = 128

        self.encoder = nn.Sequential(
            nn.Conv2d(self.channel, self.num_filters, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_filters, self.num_filters * 2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_filters * 2, self.num_filters * 4, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc_mu = nn.Sequential(
            nn.Linear(
                self.num_filters * 4 * self.output_dim_h * self.output_dim_w, self.feature_dim
            ),
        )
        self.fc_var = nn.Sequential(
            nn.Linear(
                self.num_filters * 4 * self.output_dim_h * self.output_dim_w, self.feature_dim
            ),
        )

        self.decode_fc = nn.Sequential(
            nn.Linear(
                self.feature_dim, self.num_filters * 4 * self.output_dim_h * self.output_dim_w
            ),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.num_filters * 4, self.num_filters * 2,
                               kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_filters * 2, self.num_filters,
                               kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_filters, self.channel,
                               kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )
        self.final_conv = nn.Sequential(nn.Conv2d(self.channel, self.channel, 3, stride=1, padding=1), nn.Tanh())

        self.outputs = dict()

    def encode(self, input):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decode_fc(z)
        result = result.view(-1, self.num_filters * 4, self.output_dim_h, self.output_dim_w)
        result = self.decoder(result)
        result = self.final_conv(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        #input[:, :3] = input[:, :3] / 255.0
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        ir = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1) + torch.mean(
            F.mse_loss(recons, input, reduction='none'), dim=(-3, -2, -1))
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss, 'ir': ir}