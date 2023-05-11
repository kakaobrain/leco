import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from torch.autograd import Function


class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        with torch.no_grad():
            embedding_size = codebook.size(1)
            inputs_size = inputs.size()
            inputs_flatten = inputs.view(-1, embedding_size)

            codebook_sqr = torch.sum(codebook ** 2, dim=1)
            inputs_sqr = torch.sum(inputs_flatten ** 2, dim=1, keepdim=True)

            # Compute the distances to the codebook
            distances = torch.addmm(codebook_sqr + inputs_sqr,
                inputs_flatten, codebook.t(), alpha=-2.0, beta=1.0)

            _, indices_flatten = torch.min(distances, dim=1)
            indices = indices_flatten.view(*inputs_size[:-1])
            ctx.mark_non_differentiable(indices)

            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization`. The function `VectorQuantization` '
            'is not differentiable. Use `VectorQuantizationStraightThrough` '
            'if you want a straight-through estimator of the gradient.')


class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        indices = vq(inputs, codebook)
        indices_flatten = indices.view(-1)
        ctx.save_for_backward(indices_flatten, codebook)
        ctx.mark_non_differentiable(indices_flatten)

        codes_flatten = torch.index_select(codebook, dim=0,
            index=indices_flatten)
        codes = codes_flatten.view_as(inputs)

        return (codes, indices_flatten)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        grad_inputs, grad_codebook = None, None

        if ctx.needs_input_grad[0]:
            # Straight-through estimator
            grad_inputs = grad_output.clone()
        if ctx.needs_input_grad[1]:
            # Gradient wrt. the codebook
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)

            grad_output_flatten = (grad_output.contiguous()
                                              .view(-1, embedding_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, indices, grad_output_flatten)

        return (grad_inputs, grad_codebook)


vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply


def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VAE(nn.Module):
    def __init__(self, input_dim, dim, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, z_dim * 2, 3, 1, 0),
            nn.BatchNorm2d(z_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, dim, 3, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)

        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()

        x_tilde = self.decoder(q_z_x.rsample())
        return x_tilde, kl_div


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def select(self, indices):
        return torch.index_select(self.embedding.weight, dim=0, index=indices)

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight, dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


def build_encoder(input_dim, dim, kernel_size, num_res_blocks, num_blocks=3):
    encoder_layers = []
    for i in range(num_blocks):
        if i > 0:
            _input_dim = dim
        else:
            _input_dim = input_dim
        encoder_layers.append(nn.Conv2d(_input_dim, dim, kernel_size, 2, 1))
        encoder_layers.append(nn.BatchNorm2d(dim))
        encoder_layers.append(nn.ReLU(True))
    encoder_layers.append(nn.Conv2d(dim, dim, kernel_size, 2, 1))
    for _ in range(num_res_blocks):
        encoder_layers.append(ResBlock(dim))

    return nn.Sequential(*encoder_layers)


def build_decoder(input_dim, dim, kernel_size, num_res_blocks, num_blocks=3):
    decoder_layers = []
    for _ in range(num_res_blocks):
        decoder_layers.append(ResBlock(dim))
    for i in range(num_blocks):
        decoder_layers.append(nn.ReLU(True))
        decoder_layers.append(nn.ConvTranspose2d(dim, dim, kernel_size, 2, 1))
        decoder_layers.append(nn.BatchNorm2d(dim))
    decoder_layers.append(nn.ConvTranspose2d(dim, input_dim, kernel_size, 2, 1))
    decoder_layers.append(nn.Tanh())

    return nn.Sequential(*decoder_layers)


class BasicEncBlock(nn.Module):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
            downsample=None,
            is_last=False
    ):
        super().__init__()

        if is_last:
            self.block = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size, stride, padding),
                nn.BatchNorm2d(planes)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size, stride, padding),
                nn.BatchNorm2d(planes),
                nn.ReLU()
            )
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(x)
            out += identity
        return out


class BasicDecBlock(nn.Module):
    def __init__(
            self,
            inplanes: int,
            planes: int,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
            upsample=None,
            is_last=False
    ):
        super().__init__()

        if is_last:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes, kernel_size, stride, padding)
            )
        else:
            self.block = nn.Sequential(
                nn.ReLU(),
                nn.ConvTranspose2d(inplanes, planes, kernel_size, stride, padding),
                nn.BatchNorm2d(planes),
            )
        self.is_last = is_last
        self.upsample = upsample
        self.stride = stride

    def forward(self, x):
        out = self.block(x)
        if self.upsample is not None:
            identity = self.upsample(x)
            out += identity
        return out


def build_encoder_res18(input_dim, dim, kernel_size, num_cnn_blocks=5, num_res_blocks=0, learnable=False, return_module_list=False):
    if not learnable:
        downsample = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)
    else:
        downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    blocks = [
        BasicEncBlock(input_dim, dim, kernel_size=kernel_size),
        BasicEncBlock(dim, dim, kernel_size=kernel_size),
    ]

    for i in range(num_cnn_blocks-3):
        blocks.append(BasicEncBlock(dim, dim, kernel_size=kernel_size, downsample=downsample))
    blocks.append(BasicEncBlock(dim, dim, kernel_size=kernel_size, downsample=downsample, is_last=True))

    for i in range(num_res_blocks):
        blocks.append(ResBlock(dim))

    if return_module_list:
        return nn.ModuleList(blocks)

    return nn.Sequential(*blocks)


def build_decoder_res18(input_dim, dim, kernel_size, num_cnn_blocks=5, num_res_blocks=0, learnable=False, return_module_list=False):
    if not learnable:
        upsample = nn.Upsample(mode='bilinear', scale_factor=2)
    else:
        upsample = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

    blocks = []

    for i in range(num_res_blocks):
        blocks.append(ResBlock(dim))

    for i in range(num_cnn_blocks-2):
        blocks.append(BasicDecBlock(dim, dim, kernel_size=kernel_size, upsample=upsample))

    blocks.extend([
        BasicDecBlock(dim, dim, kernel_size=kernel_size),
        BasicDecBlock(dim, input_dim, kernel_size=kernel_size, is_last=True),
        nn.Tanh()
    ])

    if return_module_list:
        return nn.ModuleList(blocks)

    return nn.Sequential(*blocks)


def build_encoder_sres18(input_dim, dim, kernel_size, num_res_blocks, learnable=False):
    if not learnable:
        downsample = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)
    else:
        downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    blocks = [
        BasicEncBlock(input_dim, dim, kernel_size=kernel_size),
        BasicEncBlock(dim, dim, kernel_size=kernel_size, downsample=downsample),
        BasicEncBlock(dim, dim, kernel_size=kernel_size, downsample=downsample),
        BasicEncBlock(dim, dim, kernel_size=kernel_size, downsample=downsample, is_last=True),
    ]

    for i in range(num_res_blocks):
        blocks.append(ResBlock(dim))

    return nn.Sequential(*blocks)


def build_decoder_sres18(input_dim, dim, kernel_size, num_res_blocks, learnable=False):
    if not learnable:
        upsample = nn.Upsample(mode='bilinear', scale_factor=2)
    else:
        upsample = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

    blocks = []

    for i in range(num_res_blocks):
        blocks.append(ResBlock(dim))

    blocks.extend([
        BasicDecBlock(dim, dim, kernel_size=kernel_size, upsample=upsample),
        BasicDecBlock(dim, dim, kernel_size=kernel_size, upsample=upsample),
        BasicDecBlock(dim, dim, kernel_size=kernel_size, upsample=upsample),
        BasicDecBlock(dim, input_dim, kernel_size=kernel_size, is_last=True),
        nn.Tanh()
    ])

    return nn.Sequential(*blocks)

def build_encoder_res_mg(input_dim, dim, kernel_size, num_res_blocks, learnable=False):
    if not learnable:
        downsample = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)
    else:
        downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=4, stride=2, padding=1),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    blocks = [
        BasicEncBlock(input_dim, dim, kernel_size=3, stride=1, padding=1),
        BasicEncBlock(dim, dim, kernel_size=kernel_size, downsample=downsample),
        BasicEncBlock(dim, dim, kernel_size=kernel_size, downsample=downsample, is_last=True),
    ]

    for i in range(num_res_blocks):
        blocks.append(ResBlock(dim))

    return nn.Sequential(*blocks)


def build_decoder_res_mg(input_dim, dim, kernel_size, num_res_blocks, learnable=False):
    if not learnable:
        upsample = nn.Upsample(mode='bilinear', scale_factor=2)
    else:
        upsample = nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

    blocks = []

    for i in range(num_res_blocks):
        blocks.append(ResBlock(dim))

    blocks.extend([
        BasicDecBlock(dim, dim, kernel_size=kernel_size, upsample=upsample),
        BasicDecBlock(dim, dim, kernel_size=kernel_size, upsample=upsample),
        BasicDecBlock(dim, input_dim, kernel_size=3, stride=1, padding=1, is_last=True),
        nn.Tanh()
    ])

    return nn.Sequential(*blocks)


class AE(nn.Module):
    def encode(self, x, return_reps=False):
        reps = self.encoder(x)
        states = self.embedding(reps)

        noise = -self.alpha * 2 + torch.rand_like(states) + self.alpha
        codes = torch.sigmoid(states + noise)

        if return_reps:
            return codes, (states, reps)
        return codes

    def decode(self, codes):
        states = self.de_embedding(codes)
        n = states.shape[0]
        states = states.view((n,) + self.resolution).contiguous()
        x_tilde = self.decoder(states)
        return x_tilde

    def forward(self, x):
        codes = self.encode(x)
        x_tilde = self.decode(codes)
        return x_tilde, codes

    def calc_losses(self, x):
        x_tilde, codes = self.forward(x)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, x)

        # binarization loss
        loss_vq = torch.minimum(torch.pow(1 - codes, 2), torch.pow(codes, 2)).mean()

        return loss_recons, loss_vq


class AE4SimHash(AE):
    def __init__(self, input_dim, dim,
                 D=512, num_blocks=3, kernel_size=4,
                 arch='basic', alpha=0.3, resolution=(2, 3)):
        super().__init__()
        self.D = D
        self.alpha = alpha
        self.dim = dim
        self.resolution = (dim,) + resolution
        if arch == 'basic':
            self.encoder = build_encoder(input_dim, dim, kernel_size, num_res_blocks=num_blocks)
        elif arch == 'res':
            self.encoder = build_encoder_res18(input_dim, dim, kernel_size, num_res_blocks=num_blocks)
        elif arch == 'resl':
            self.encoder = build_encoder_res18(input_dim, dim, kernel_size, num_res_blocks=num_blocks, learnable=True)
        elif arch == 'sres':
            self.encoder = build_encoder_sres18(input_dim, dim, kernel_size, num_res_blocks=num_blocks)
        elif arch == 'sresl':
            self.encoder = build_encoder_sres18(input_dim, dim, kernel_size, num_res_blocks=num_blocks, learnable=True)

        self.encoded_dim = np.prod(self.resolution)

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoded_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.D)
        )

        self.de_embedding = nn.Sequential(
            nn.Linear(self.D, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.encoded_dim)
        )

        if arch == 'basic':
            self.decoder = build_decoder(input_dim, dim, kernel_size, num_res_blocks=num_blocks)
        elif arch == 'res':
            self.decoder = build_decoder_res18(input_dim, dim, kernel_size, num_res_blocks=num_blocks)
        elif arch == 'resl':
            self.decoder = build_decoder_res18(input_dim, dim, kernel_size, num_res_blocks=num_blocks, learnable=True)
        elif arch == 'sres':
            self.decoder = build_decoder_sres18(input_dim, dim, kernel_size, num_res_blocks=num_blocks)
        elif arch == 'sresl':
            self.decoder = build_decoder_sres18(input_dim, dim, kernel_size, num_res_blocks=num_blocks, learnable=True)

        self.apply(weights_init)


class AE4SimHashForMG(AE):
    def __init__(self, input_dim, dim,
                 D=512, num_blocks=3, kernel_size=4,
                 arch='basic', alpha=0.3, resolution=(2, 3)):
        super().__init__()
        self.D = D
        self.alpha = alpha
        self.dim = dim
        self.resolution = (dim,) + resolution
        if arch == 'res':
            self.encoder = build_encoder_res_mg(input_dim, dim, kernel_size, num_res_blocks=num_blocks)
        elif arch == 'resl':
            self.encoder = build_encoder_res_mg(input_dim, dim, kernel_size, num_res_blocks=num_blocks, learnable=True)

        self.encoded_dim = np.prod(self.resolution)

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.encoded_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.D)
        )

        self.de_embedding = nn.Sequential(
            nn.Linear(self.D, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.encoded_dim)
        )

        if arch == 'res':
            self.decoder = build_decoder_res_mg(input_dim, dim, kernel_size, num_res_blocks=num_blocks)
        elif arch == 'resl':
            self.decoder = build_decoder_res_mg(input_dim, dim, kernel_size, num_res_blocks=num_blocks, learnable=True)

        self.apply(weights_init)


def forward_module_list(module_list, x):
    encodings = []
    x_enc = x
    for i in range(len(module_list)):
        layer = module_list[i]
        x_enc = layer(x_enc)
        encodings.append(x_enc)
    return encodings


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim,
                 K=512, num_cnn_blocks=5, num_res_blocks=3, kernel_size=4,
                 reg_type='l2', arch='basic'):
        super().__init__()
        self.K = K
        self.reg_type = reg_type
        if arch == 'res':
            self.encoder = build_encoder_res18(input_dim, dim, kernel_size, num_cnn_blocks=num_cnn_blocks, num_res_blocks=num_res_blocks, return_module_list=True)
        elif arch == 'resl':
            self.encoder = build_encoder_res18(input_dim, dim, kernel_size, num_cnn_blocks=num_cnn_blocks, num_res_blocks=num_res_blocks, learnable=True, return_module_list=True)

        self.codebook = VQEmbedding(K, dim)

        if arch == 'res':
            self.decoder = build_decoder_res18(input_dim, dim, kernel_size, num_cnn_blocks=num_cnn_blocks, num_res_blocks=num_res_blocks)
        elif arch == 'resl':
            self.decoder = build_decoder_res18(input_dim, dim, kernel_size, num_cnn_blocks=num_cnn_blocks, num_res_blocks=num_res_blocks, learnable=True)

        self.apply(weights_init)

    def encode(self, x, return_reps=False):
        if type(self.encoder) == torch.nn.ModuleList:
            encodings = forward_module_list(self.encoder, x)
            z_e_x = encodings[-1]
        else:
            z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        if return_reps:
            z_q_x_st, _ = self.codebook.straight_through(z_e_x)
            reps = z_e_x
            if type(return_reps) == int:
                reps = encodings[return_reps]
            return latents, (z_q_x_st, reps)
        else:
            return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        if type(self.encoder) == torch.nn.ModuleList:
            encodings = forward_module_list(self.encoder, x)
            z_e_x = encodings[-1]
        else:
            z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x

    def calc_reg_loss(self, reg_type=None):
        loss = 0
        reg_type = reg_type or self.reg_type
        if reg_type == 'l1':
            loss = torch.abs(self.codebook.embedding.weight).mean()
        elif reg_type == 'l2':
            loss = torch.square(torch.norm(self.codebook.embedding.weight, dim=1)).mean()

        return loss

    def calc_losses(self, x):
        x_tilde, z_e_x, z_q_x = self.forward(x)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, x)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss_reg = self.calc_reg_loss()

        return loss_recons, loss_vq, loss_commit, loss_reg

    def select_codes(self, indices):
        if not type(indices) == torch.Tensor:
            indices = torch.LongTensor(indices).to(next(self.codebook.parameters()).device)
        return self.codebook.embedding(indices)

class VectorQuantizedVAEforMG(nn.Module):
    def __init__(self, input_dim, dim,
                 K=512, num_blocks=3, kernel_size=4,
                 reg_type='l2', arch='basic'):
        super().__init__()
        self.K = K
        self.reg_type = reg_type
        if arch == 'res':
            self.encoder = build_encoder_res_mg(input_dim, dim, kernel_size, num_res_blocks=num_blocks)
        elif arch == 'resl':
            self.encoder = build_encoder_res_mg(input_dim, dim, kernel_size, num_res_blocks=num_blocks, learnable=True)

        self.codebook = VQEmbedding(K, dim)

        if arch == 'res':
            self.decoder = build_decoder_res_mg(input_dim, dim, kernel_size, num_res_blocks=num_blocks)
        elif arch == 'resl':
            self.decoder = build_decoder_res_mg(input_dim, dim, kernel_size, num_res_blocks=num_blocks, learnable=True)

        self.apply(weights_init)

    def encode(self, x, return_reps=False):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)

        if return_reps:
            z_q_x_st, _ = self.codebook.straight_through(z_e_x)
            return latents, (z_q_x_st, z_e_x)
        else:
            return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x

    def calc_reg_loss(self, reg_type=None):
        loss = 0
        reg_type = reg_type or self.reg_type
        if reg_type == 'l1':
            loss = torch.abs(self.codebook.embedding.weight).mean()
        elif reg_type == 'l2':
            loss = torch.square(torch.norm(self.codebook.embedding.weight, dim=1)).mean()

        return loss

    def calc_losses(self, x):
        x_tilde, z_e_x, z_q_x = self.forward(x)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, x)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss_reg = self.calc_reg_loss()

        return loss_recons, loss_vq, loss_commit, loss_reg

    def select_codes(self, indices):
        if not type(indices) == torch.Tensor:
            indices = torch.LongTensor(indices).to(next(self.codebook.parameters()).device)
        return self.codebook.embedding(indices)

    def encode_and_reps(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        z_q_x_st, _ = self.codebook.straight_through(z_e_x)
        # z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)

        return latents, z_e_x, z_q_x_st


class ExtendedVectorQuantizedVAEforMG(VectorQuantizedVAEforMG):
    def __init__(self, input_dim, dim,
                 K=512, num_blocks=3, kernel_size=4,
                 reg_type='l2', arch='basic', ext_dim=None):
        super(ExtendedVectorQuantizedVAEforMG, self).__init__(input_dim, dim, K, num_blocks, kernel_size, reg_type=reg_type, arch=arch)
        encoder_input_dim = input_dim + ext_dim
        if arch == 'res':
            self.encoder = build_encoder_res_mg(encoder_input_dim, dim, kernel_size, num_res_blocks=num_blocks)
        elif arch == 'resl':
            self.encoder = build_encoder_res_mg(encoder_input_dim, dim, kernel_size, num_res_blocks=num_blocks, learnable=True)
        # self.mlp = nn.Sequential(
        #     nn.Linear(dim + ext_dim, dim),
        #     nn.ReLU(True)
        # )

    # def _extends(self, x, x_):
    #     x_ = x_.unsqueeze(-1).unsqueeze(-1)
    #     x_ = x_.repeat([1, 1] + list(x.size())[2:])
    #     x = torch.cat([x, x_], dim=1)
    #     x = x.permute(0, 2, 3, 1).contiguous()
    #     x = self.mlp(x)
    #     x = x.permute(0, 3, 1, 2).contiguous()
    #     return x

    def _extends(self, x, x_):
        w, h = x.shape[-2:]
        x_ = x_[:, :, None, None]
        x_ = torch.tile(x_, (1, 1, w, h))
        x = torch.cat([x, x_], dim=1)
        return x

    def encode(self, x, x_, return_reps=False):
        x = self._extends(x, x_)
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        if return_reps:
            z_q_x_st, _ = self.codebook.straight_through(z_e_x)
            return latents, (z_q_x_st, z_e_x)
        else:
            return latents

    def forward(self, x, x_):
        x = self._extends(x, x_)
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x

    def calc_losses(self, x, x_):
        x_tilde, z_e_x, z_q_x = self.forward(x, x_)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, x)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss_reg = self.calc_reg_loss()

        return loss_recons, loss_vq, loss_commit, loss_reg

    def encode_and_reps(self, x, x_):
        z_e_x = self.encoder(x)
        z_e_x = self._extends(z_e_x, x_)
        latents = self.codebook(z_e_x)
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)

        return latents, z_e_x, z_q_x


class ExtendedVectorQuantizedVAE(VectorQuantizedVAE):
    def __init__(self, input_dim, dim,
                 K=512, num_blocks=3, kernel_size=4,
                 reg_type='l2', arch='basic', ext_dim=None):
        super(ExtendedVectorQuantizedVAE, self).__init__(input_dim, dim, K, num_blocks, kernel_size, reg_type=reg_type, arch=arch)
        self.mlp = nn.Sequential(
            nn.Linear(dim + ext_dim, dim),
            nn.ReLU(True)
        )

    def _extends(self, x, x_):
        x_ = x_.unsqueeze(-1).unsqueeze(-1)
        x_ = x_.repeat([1, 1] + list(x.size())[2:])
        x = torch.cat([x, x_], dim=1)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def encode(self, x, x_, return_reps=False):
        z_e_x = self.encoder(x)
        z_e_x = self._extends(z_e_x, x_)
        latents = self.codebook(z_e_x)
        if return_reps:
            z_q_x_st, _ = self.codebook.straight_through(z_e_x)
            return latents, (z_q_x_st, z_e_x)
        else:
            return latents

    def forward(self, x, x_):
        z_e_x = self.encoder(x)
        z_e_x = self._extends(z_e_x, x_)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x

    def calc_losses(self, x, x_):
        x_tilde, z_e_x, z_q_x = self.forward(x, x_)

        # Reconstruction loss
        loss_recons = F.mse_loss(x_tilde, x)
        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

        loss_reg = self.calc_reg_loss()

        return loss_recons, loss_vq, loss_commit, loss_reg


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        self.class_cond_embedding = nn.Embedding(
            n_classes, 2 * dim
        )

        kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()

        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert + h[:, :, None, None])

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10):
        super().__init__()
        self.dim = dim

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        self.apply(weights_init)

    def forward(self, x, label):
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, W)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)

    def generate(self, label, shape=(8, 8), batch_size=64):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x
