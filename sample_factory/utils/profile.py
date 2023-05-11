# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging

import numpy as np
import torch
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.batchnorm import _BatchNorm


LOGGER = logging.getLogger(__name__)


class Profiler:
    def __init__(self, module):
        self.module = module

    def params(self, name_filter=lambda name: True):
        #for name, params in self.module.named_parameters():
        #    if name_filter(name):
        #        print(name)
        return np.sum([params.numel() for name, params in self.module.named_parameters() if name_filter(name)])

    def flops(self, *inputs, name_filter=lambda name: 'skeleton' not in name and 'loss' not in name):
        operation_flops = []

        def get_hook(name):
            def counting(module, inp, outp):
                class_name = module.__class__.__name__

                fn = None
                module_type = type(module)
                if not name_filter(str(module_type)):
                    pass
                elif isinstance(module, torch.nn.Linear):
                    fn = count_linear_flops
                elif isinstance(module, _ConvNd):
                    fn = count_conv_flops
                elif isinstance(module, _BatchNorm):
                    fn = count_elements_flops
                elif 'swish' in class_name.lower():
                    fn = count_elements_flops
                elif 'attention' == class_name.lower():
                    fn = count_attention_flops
                elif 'token_performer' == class_name.lower() or 'tokenperformer' == class_name.lower():
                    fn = count_token_performer_flops
                elif 't2tattention' == class_name.lower():
                    fn = count_t2tattention_flops
                else:
                    pass
                    # LOGGER.warning('Not implemented for %s', module_type)

                flops = fn(module, inp, outp) if fn is not None else 0
                data = {
                    'name': name,
                    'class_name': class_name,
                    'flops': flops,
                }
                operation_flops.append(data)
            return counting

        handles = []
        for name, module in self.module.named_modules():
            # if len(list(module.children())) > 0:  # pylint: disable=len-as-condition
            #     continue
            handle = module.register_forward_hook(get_hook(name))
            handles.append(handle)

        with torch.no_grad():
            _ = self.module(*inputs)

        # remove hook
        _ = [h.remove() for h in handles]

        return np.sum([data['flops'] for data in operation_flops if name_filter(data['name'])])

def count_attention_flops(attention_module, inputs, outputs):
    inputs = inputs[0]
    num_head = attention_module.num_heads
    dim = attention_module.qkv.in_features
    head_dim = dim // num_head

    attn_einsum_ops = inputs.shape[1] * inputs.shape[1] * head_dim * num_head
    x_einsum_ops = num_head * inputs.shape[1] * head_dim  * inputs.shape[1]
    return int(attn_einsum_ops + x_einsum_ops)

def count_t2tattention_flops(t2tattention_module, inputs, outputs):
    inputs = inputs[0]
    token_dim = t2tattention_module.token_dim
    num_head = t2tattention_module.num_heads
    attn_einsum_ops = num_head * inputs.shape[1] * inputs.shape[1] * token_dim
    x_einsum_ops = num_head * inputs.shape[1] * token_dim * inputs.shape[1]

    return int(attn_einsum_ops + x_einsum_ops)


def count_token_performer_flops(token_performer_module, inputs, outputs):
    inputs = inputs[0]
    emb = token_performer_module.emb
    # prm_exp
    w_shape = token_performer_module.w.shape
    prm_pow = inputs.shape[1] * inputs.shape[2]
    prm_einsum = inputs.shape[1] * w_shape[0] * w_shape[1]
    prm_exp = inputs.shape[1] * w_shape[0]

    prm_ops = 2 * (prm_pow + prm_einsum + prm_exp)
    # kp shape = [inputs.shape[1], w_shape[0]]
    D_einsum_ops = inputs.shape[1] * w_shape[0]
    kptv_einsum_ops = inputs.shape[1] * emb * w_shape[0]
    y_einsum_ops = inputs.shape[1] * w_shape[0] * emb + inputs.shape[1] * emb
    return int(prm_ops + D_einsum_ops + kptv_einsum_ops + y_einsum_ops)


# base code from https://github.com/JaminFong/DenseNAS/blob/master/tools/multadds_count.py
def count_conv_flops(conv_module, inputs, outputs):
    inputs = inputs[0]

    batch_size = inputs.shape[0]
    output_height, output_width = outputs.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels

    conv_per_position_flops = (kernel_height * kernel_width * in_channels * out_channels) / conv_module.groups

    active_elements_count = batch_size * output_height * output_width

    if hasattr(conv_module, '__mask__') and conv_module.__mask__ is not None:
        # (b, 1, h, w)
        flops_mask = conv_module.__mask__.expand(batch_size, 1, output_height, output_width)
        active_elements_count = flops_mask.sum()

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops
    return int(overall_flops)


def count_linear_flops(linear_module, inputs, outputs):
    inputs = inputs[0]
    batch_size = inputs.shape[0]
    num_repeats = 1
    for i in (inputs[0].shape[:-1]):
        num_repeats *= i

    overall_flops = linear_module.in_features * linear_module.out_features * batch_size * num_repeats

    bias_flops = 0
    #if linear_module.bias is not None:
    #    bias_flops = outputs.numel()

    overall_flops = overall_flops + bias_flops
    return int(overall_flops)


def count_elements_flops(module, inputs, outputs):
    return inputs[0][0].numel()

