
'''
This model can be ran as the original transformer XL or the stable transformer XL. Much of this code came from
https://github.com/kimiyoung/transformer-xl  but now has added functionality to use gating as well as different
orderings of the submodules as done in https://arxiv.org/pdf/1910.06764.pdf


TO DO:
    1. Figure out how we'll actually run this (how is cache kept and so on?)
        a) need to rewrite the training script (will assume functionality for batching previous examples exists)
    2. CHECK THIS: Have I applied layer norms in correct order?
    3. Initialize b_g (one of the bias' in GRU) to 2
    4. They use 256dim embedding, 512 memory size
    5. Add in action set from table 5 of paper (15 actions) (is even more simple for some in table 6)

Remember: Receptive field in transformer XL is linear in #layers and segment size




'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm

import numpy as np


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if not pos_emb.shape[-1] == self.demb:
            # They should be same originally, but it doesn't when self.demb is an odd number
            # In this case, we cut out last one element in the last dimension for dimension matching,
            # Cutting out one cosine value is same as the original paper section 3.5 (https://arxiv.org/pdf/1706.03762.pdf)
            pos_emb = torch.split(pos_emb, [self.demb, 1], dim=-1)[0]

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


'''
GRU gating layer used in Stabilizing transformers in RL.
Note that all variable names follow the notation from section: "Gated-Recurrent-Unit-type gating" 
in https://arxiv.org/pdf/1910.06764.pdf
'''
class GRUGate(nn.Module):

    def __init__(self,d_model):
        #d_model is dimension of embedding for each token as input to layer (want to maintain this in the gate)
        super(GRUGate,self).__init__()

        # TODO: DEBUG Make sure intitialize bias of linear_w_z to -3
        self.linear_w_r = nn.Linear(d_model,d_model,bias=False)
        self.linear_u_r = nn.Linear(d_model,d_model,bias=False)
        self.linear_w_z = nn.Linear(d_model,d_model)               ### Giving bias to this layer (will count as b_g so can just initialize negative)
        self.linear_u_z = nn.Linear(d_model, d_model,bias=False)
        self.linear_w_g = nn.Linear(d_model, d_model,bias=False)
        self.linear_u_g = nn.Linear(d_model, d_model,bias=False)

        self.init_bias()

    def get_internal_value_dict(self, name_prefix=''):
        internal_value_dict = dict()
        internal_value_dict[f'{name_prefix}gate_w_r_weight_grad_norm'] = self.linear_w_r.weight.grad.norm().item()
        internal_value_dict[f'{name_prefix}gate_u_r_weight_grad_norm'] = self.linear_u_r.weight.grad.norm().item()
        internal_value_dict[f'{name_prefix}gate_w_z_weight_grad_norm'] = self.linear_w_z.weight.grad.norm().item()
        internal_value_dict[f'{name_prefix}gate_w_z_bias_grad_norm'] = self.linear_w_z.bias.grad.norm().item()
        internal_value_dict[f'{name_prefix}gate_u_z_weight_grad_norm'] = self.linear_u_z.weight.grad.norm().item()
        internal_value_dict[f'{name_prefix}gate_w_g_weight_grad_norm'] = self.linear_w_g.weight.grad.norm().item()
        internal_value_dict[f'{name_prefix}gate_u_g_weight_grad_norm'] = self.linear_u_g.weight.grad.norm().item()
        return internal_value_dict

    def init_bias(self):
        with torch.no_grad():
            self.linear_w_z.bias.fill_(-2)  # Manually setting this bias to allow starting with markov process
            # Note -2 is the setting used in the paper stable transformers

    def forward(self,x,y):
        ### Here x,y follow from notation in paper
        # TODO: DEBUG MAKE SURE THIS IS APPLIED ON PROPER AXIS
        z = torch.sigmoid(self.linear_w_z(y) + self.linear_u_z(x))  #MAKE SURE THIS IS APPLIED ON PROPER AXIS
        r = torch.sigmoid(self.linear_w_r(y) + self.linear_u_r(x))
        h_hat = torch.tanh(self.linear_w_g(y) + self.linear_u_g(r*x))  #Note elementwise multiplication of r and x
        return (1.-z)*x + z*h_hat


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )


    #QUESTION: WHAT IS shape of inp to make this work (elementwise fnn and have batch dim)
    def forward(self, inp):

        ##### positionwise feed-forward (this is what's used in original transformer)
        core_out = self.CoreNet(inp)

        return core_out


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,use_gate,use_stable_version, use_pe, tau,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.use_gate = use_gate
        self.use_stable_version = use_stable_version

        self.use_pe = use_pe
        self.tau = tau

        if self.use_gate:
            self.gate_mha = GRUGate(d_model)
            self.gate_mlp = GRUGate(d_model)

        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model,
                            d_head, dropout, use_pe, tau, **kwargs)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout)
        if not use_stable_version:
            self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def get_internal_value_dict(self, name_prefix=''):
        internal_value_dict = dict()
        if self.use_gate:
            internal_value_dict.update(
                self.gate_mha.get_internal_value_dict(name_prefix=f'{name_prefix}mha_')
            )
            internal_value_dict.update(
                self.gate_mlp.get_internal_value_dict(name_prefix=f'{name_prefix}mlp_')
            )
        internal_value_dict.update( self.dec_attn.get_internal_value_dict(name_prefix=name_prefix) )

        if not self.use_stable_version:
            internal_value_dict[f'{name_prefix}layer_norm1_weight_grad_norm'] = self.layer_norm1.weight.grad.norm().item()
            internal_value_dict[f'{name_prefix}layer_norm1_bias_grad_norm'] = self.layer_norm1.bias.grad.norm().item()

        internal_value_dict[f'{name_prefix}layer_norm2_weight_grad_norm'] = self.layer_norm2.weight.grad.norm().item()
        internal_value_dict[f'{name_prefix}layer_norm2_bias_grad_norm'] = self.layer_norm2.bias.grad.norm().item()

        return internal_value_dict

    def forward_orig(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                               attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.layer_norm1(dec_inp+output)
        output2 = self.pos_ff(output)
        output2 = self.layer_norm2(output+output2)
        return output2


    def forward_stable(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        # dec_input: [q_len, B, 256], r: [k_len, 1, 256],
        # r_w_bias: [4, 64], r_r_bias: [4, 64], dec_attn_mask: [q_len, k_len, 1],
        # mems: [m_len, B, 256]

        #Layer norm will be applied at start of MHA module on both dec_inp2 and mems
        #First Layer norm will be applied within dec_attn

        dec_inp2, attn_entropy = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                                attn_mask=dec_attn_mask,
                                mems=mems, use_stable_version=self.use_stable_version)

        #NOTE: In stable transformer they apply Relu before the layernorm/gate (in appendix C.3)
        if self.use_gate:
            dec_inp2 = self.gate_mha(dec_inp, F.relu(dec_inp2))
        else:
            dec_inp2 = dec_inp + F.relu(dec_inp2)

        dec_inp3 = self.layer_norm2(dec_inp2)

        dec_inp3 = self.pos_ff(dec_inp3)

        if self.use_gate:
            dec_inp3 = self.gate_mlp(dec_inp2, F.relu(dec_inp3))
        else:
            dec_inp3 = F.relu(dec_inp3) + dec_inp2

        return dec_inp3, attn_entropy


    def forward(self,dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):

        if self.use_stable_version:
            return self.forward_stable(dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask, mems)

        return self.forward_orig(dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask, mems)


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, use_pe, tau, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False):
        super(RelMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        #Get query, key and value for each token (NOTE SOME Inefficiency since
        #don't need query for any of the memory. Parallelization must make up for it
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        self.use_pe = use_pe
        self.tau = tau

    def _parallelogram_mask(self, h, w, left=False):
        # UserWarning: masked_fill_ received a mask with dtype torch.uint8,
        # this behavior is now deprecated,please use a mask with dtype torch.bool instead.
        # changed .byte() to .bool()
        # mask = torch.ones((h, w)).byte()
        mask = torch.ones((h, w)).bool()
        m = min(h, w)
        mask[:m,:m] = torch.triu(mask[:m,:m])
        mask[-m:,-m:] = torch.tril(mask[-m:,-m:])

        if left:
            return mask
        else:
            return mask.flip(0)

    def _shift(self, x, qlen, klen, mask, left=False):
        if qlen > 1:
            zero_pad = torch.zeros((x.size(0), qlen-1, x.size(2), x.size(3)),
                                    device=x.device, dtype=x.dtype)
        else:
            zero_pad = torch.zeros(0, device=x.device, dtype=x.dtype)

        if left:
            mask = mask.flip(1)
            x_padded = torch.cat([zero_pad, x], dim=1).expand(qlen, -1, -1, -1)
        else:
            x_padded = torch.cat([x, zero_pad], dim=1).expand(qlen, -1, -1, -1)

        x = x_padded.masked_select(mask[:,:,None,None]) \
                    .view(qlen, klen, x.size(2), x.size(3))

        return x

    def _rel_shift(self, x, zero_triu=False):
        # x: (q_len, k_len, n_batch, n_head)
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype) # (q_len, 1, n_batch, n_head)
        x_padded = torch.cat([zero_pad, x], dim=1) # (q_len, k_len + 1, n_batch, n_head)

        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:]) # (k_len + 1, q_len, n_batch, n_head)

        x = x_padded[1:].view_as(x) # (q_len, k_len, n_batch, n_head), relative shift should be applied at k_len for each dim of q_len

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError




class RelPartialLearnableMultiHeadAttn(RelMultiHeadAttn):
    def __init__(self, *args, **kwargs):
        super(RelPartialLearnableMultiHeadAttn, self).__init__(*args, **kwargs)

        if self.use_pe:
            self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def get_internal_value_dict(self, name_prefix=''):
        # funcion to be used for monitoring network internal values
        internal_value_dict = dict()
        if self.use_pe:
            grad_w_r_net = self.r_net.weight.grad.reshape(self.d_model, self.n_head, self.d_head)
        grad_w_qkv_net = self.qkv_net.weight.grad.reshape(self.d_model, self.n_head, 3, self.d_head)
        grad_w_o_net = self.o_net.weight.grad.reshape(self.n_head, self.d_head, self.d_model)

        for i in range(self.n_head):
            if self.use_pe:
                internal_value_dict[f'{name_prefix}head{i}_r_net_weight_grad_norm'] = grad_w_r_net[:, i, :].norm().item()
            internal_value_dict[f'{name_prefix}head{i}_q_net_weight_grad_norm'] = grad_w_qkv_net[:, i, 0, :].norm().item()
            internal_value_dict[f'{name_prefix}head{i}_k_net_weight_grad_norm'] = grad_w_qkv_net[:, i, 1, :].norm().item()
            internal_value_dict[f'{name_prefix}head{i}_v_net_weight_grad_norm'] = grad_w_qkv_net[:, i, 2, :].norm().item()
            internal_value_dict[f'{name_prefix}head{i}_o_net_weight_grad_norm'] = grad_w_o_net[i, :, :].norm().item()

        internal_value_dict[f'{name_prefix}layer_norm_weight_grad_norm'] = self.layer_norm.weight.grad.norm().item()
        internal_value_dict[f'{name_prefix}layer_norm_bias_grad_norm'] = self.layer_norm.bias.grad.norm().item()

        return internal_value_dict

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None, use_stable_version=False):
        if not self.use_pe:
            assert r is None
            assert r_r_bias is None

        # w: (n_seq, n_batch, dim)
        qlen, bsz = w.size(0), w.size(1)
        if self.use_pe:
           rlen = r.size(0)

        #if using stable version, then want layernorm of memory as well before MHA
        if mems.size(0) > 0:
            cat = torch.cat([mems, w], 0) # (m_len + q_len) x B x dim

            w_heads = self.qkv_net(cat) if not use_stable_version else self.qkv_net(self.layer_norm(cat))             # w_heads: 200x 16 x (256 x 3)

            if self.use_pe:
                if mems.dtype == torch.float16:
                    r = r.half() # TODO: should be handled with cfg
                r_head_k = self.r_net(r) # 200 x 1 x 256

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            w_head_q = w_head_q[-qlen:]
        else:
            w_heads = self.qkv_net(w) if not use_stable_version else self.qkv_net(self.layer_norm(w))
            if self.use_pe:
                r_head_k = self.r_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)           # qlen x bsz x n_head x d_head
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)           # klen x bsz x n_head x d_head

        if self.use_pe:
            r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)                # qlen x n_head x d_head

        #### compute attention score
        rw_head_q = w_head_q + r_w_bias                                        # qlen x bsz x n_head x d_head
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))             # qlen x klen x bsz x n_head

        if self.use_pe:
            rr_head_q = w_head_q + r_r_bias
            BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))              # qlen x klen x bsz x n_head
            BD = self._rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        #attn_score = AC + BD
        attn_score = AC
        if self.use_pe:
            attn_score += BD

        attn_score.mul_(self.scale) # 100 x 200 x 16 x 4 (qlen x klen x bsz x n_head)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score = attn_score.float().masked_fill(
                    attn_mask[None,:,:,None], -float("inf")).type_as(attn_score)
            elif attn_mask.dim() == 3: #THIS IS WHAT IS Usually executed
                attn_score = attn_score.float().masked_fill(
                    attn_mask[:,:,:,None], -float("inf")).type_as(attn_score)


        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score * self.tau, dim=1)
        attn_prob = self.dropatt(attn_prob)
        self.attn_prob = attn_prob

        with torch.no_grad():
            temp = -1 * attn_prob * torch.log(attn_prob)
            temp = temp.masked_fill(
                attn_mask[:, :, :, None], 0).type_as(temp)
            attn_entropy = torch.sum(temp, dim=1) # qlen x bsz x n_head
            #attn_entropy = torch.sum(temp, dim=1) / torch.sum(1 - attn_mask.long(), dim=1).unsqueeze(-1).repeat(1, 1, temp.size(-1))
            attn_entropy = torch.mean(attn_entropy, dim=[0, 1]) # n_head

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        return attn_out, attn_entropy


# TODO : DEBUG, sanity check the memtransformerLM implementation with the one in the Stabilizing paper
class MemTransformerLM(nn.Module):
    def __init__(self, cfg, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tie_weight=True, d_embed=None,
                 div_val=1,
                 tgt_len=None, ext_len=0, mem_len=1,
                 cutoffs=[], adapt_inp=False,
                 same_length=False, clamp_len=-1,
                 use_gate=True, use_stable_version=True):
        super(MemTransformerLM, self).__init__()
        self.cfg = cfg
        self.n_token = n_token # TODO : Check this is not being used anywhere

        self.d_embed = d_model
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        # self.state_emb = State_Embedder()

        self.drop = nn.Dropout(dropout)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len
        #self.max_klen = tgt_len + ext_len + mem_len

        self.layers = nn.ModuleList()

        for i in range(n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    n_head=n_head, d_model=d_model, d_head=d_head, d_inner=d_inner, dropout=dropout,
                    use_stable_version=use_stable_version, use_gate=use_gate,
                    tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                    dropatt=dropatt, use_pe=cfg.use_pe, tau=cfg.tau)
            )

        #To do: Look into sample softmax and adaptive softmax for future, not relevant here though
        # are useful when need fast softmax over many classes

        self.same_length = same_length
        self.clamp_len = clamp_len

        # create positional encoding-related parameters
        self._create_params()

    def get_core_out_size(self):
        return self.d_model

    def init_gru_bias(self):
        for l in self.layers:
            l.gate_mha.init_bias()
            l.gate_mlp.init_bias()

    def backward_compatible(self):
        self.sample_softmax = -1

    def _create_params(self):
        if self.cfg.use_pe:
            self.pos_emb = PositionalEmbedding(self.d_model)
            self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self):
        mems = []
        param = next(self.parameters())
        for i in range(self.n_layer+1):
            empty = torch.empty(0, dtype=param.dtype, device=param.device)
            mems.append(empty)

        return mems

    def get_internal_value_dict(self, name_prefix=''):
        # funcion to be used for monitoring network internal values
        internal_value_dict = dict()
        if self.cfg.use_pe:
            for i in range(self.n_head):
                internal_value_dict[f'{name_prefix}head{i}_r_w_bias_grad_norm'] = self.r_w_bias.grad[i].norm().item()
                internal_value_dict[f'{name_prefix}head{i}_r_r_bias_grad_norm'] = self.r_r_bias.grad[i].norm().item()

        for i, layer in enumerate(self.layers):
            _internal_value_dict = layer.get_internal_value_dict(name_prefix=f'{name_prefix}layer{i}_')
            internal_value_dict.update(_internal_value_dict)
        return internal_value_dict


    #NOTE: qlen looks to be number of characters in one example
    #      mlen is memory size
    def _update_mems(self, hids, mems, mlen, qlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len) # ext_len looks to usually be 0 (in their experiments anyways

            # TODO: I have changed beg_idx to 0 since want to use all memory, may want to change
            #       this once move to larger environments (THIS HAS NOW BEEN CHANGED)

            #HERE IS THE PROBLEM.
            #print('hids shape: ', hids[0].shape)

            beg_idx = max(0, end_idx - self.mem_len) #if hids[0].shape[0] > 1 else 0
            #print('BEG IND: ', beg_idx)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0) # (m_len + q) x B x dim
                if beg_idx == end_idx: # cfg.mem_len=0
                    new_mems.append(torch.zeros(cat[0:1].size()))
                else: # cfg.mem_len > 0
                    new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    # def _forward(self, dec_inp, obs_emb, mems=None):
    # TODO : We dropped dec_input since the first 2 dims of obs_emb should be the same as
    # that of dec_input, which is unrolled length = query length and batch_size
    # we saw this from             core_input = core_input.view(T, B, -1) line 668 in monobeast_test.py
    def _forward(self, obs_emb, mems=None, rollout_step_list=None, mem_begin_index=None, dones=None):
        qlen, bsz, _ = obs_emb.size() #qlen is number of characters in input ex

        if mems is not None:
            mlen = mems[0].size(0)
            # print('HERE: mlen: {}, len mems: {}, mems[0] shape: {}'.format(mlen, len(mems),mems[0].shape))
        else:
            mlen = 0
        # mlen = mems[0].size(0) if mems is not None else 0

        klen = mlen + qlen

        # create the mask taking in consideration the mlen as well. All memory should be attended by the first query
        # dec_attn_mask = torch.triu(
        #     obs_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()[:,:,None]

        # dec_attn_mask = torch.triu(
        #     obs_emb.new_ones(qlen, klen), diagonal=1+mlen).bool().unsqueeze(-1).repeat(1, 1, bsz)

        dec_attn_mask = (torch.triu(
             obs_emb.new_ones(qlen, klen), diagonal=1+mlen)
            + torch.tril(
             obs_emb.new_ones(qlen, klen), diagonal=-1)).bool().unsqueeze(-1).repeat(1, 1, bsz)

        # print(dec_attn_mask.size())
        # print(len(rollout_step_list))

        # if rollout_step_list is not None:
        #     for b in range(len(rollout_step_list)):
        #         rollout_step = rollout_step_list[b]
        #         mask_begin_idx = int(mlen // 2) + rollout_step - self.cfg.mem_len
        #         mask_end_idx = mask_begin_idx + self.cfg.mem_len
        #         dec_attn_mask[:, :mask_begin_idx, b] = True
        #         dec_attn_mask[:, mask_end_idx:mlen, b] = True
        #         #dec_attn_mask[:, rollout_step:mlen, b] = True

        for b in range(bsz):
            dec_attn_mask[:, :(mlen - max(0, mem_begin_index[b])), b] = True
            if dones is not None:
                query_done_index = torch.where(dones[:, b] > 0)
                for q in query_done_index[0]:
                    # Going to mask out elements before done for new episode
                    dec_attn_mask[q + 1:, :(mlen + q + 1), b] = True

                #query_done_index = (1 - dones[:, b]).long().argmin(0)
                #dec_attn_mask[query_done_index+1:, :(mlen + query_done_index + 1), b] = True

        hids = []
        if self.cfg.use_pe:
            pos_seq = torch.arange(klen-1, -1, -1.0, device=obs_emb.device,
                                   dtype=obs_emb.dtype) # [99,...0]
            if self.clamp_len > 0:
                pos_seq.clamp_(max=self.clamp_len)
            pos_emb = self.pos_emb(pos_seq) # T x 1 x dim
            pos_emb = self.drop(pos_emb)

        core_out = self.drop(obs_emb)

        hids.append(core_out)
        #SEEMS THAT THEY store memory per layer which makes sense to attend to (for ex if at first layer, if we were
        #applying attention to memory and this new data, this would give us the same result.
        attn_entropy = []
        for i, layer in enumerate(self.layers):
            #print('HIDDEN iter: {}, output: {}'.format(i, core_out[-1,0, :10]))

            # TODO : The memory should be the same hidden layer's state of the previous T timesteps
            mems_i = None if mems is None else mems[i]
            # print('from txl483 shapes : ', core_out.shape, pos_emb.shape, self.r_w_bias.shape, self.r_r_bias.shape, dec_attn_mask.shape, mems_i.shape)
            if self.cfg.use_pe:
                core_out, attn_entropy_layer = layer(core_out, pos_emb, self.r_w_bias,
                        self.r_r_bias, dec_attn_mask=dec_attn_mask, mems=mems_i)
            else:
                core_out, attn_entropy_layer = layer(core_out, None, self.r_w_bias,
                                                     None, dec_attn_mask=dec_attn_mask, mems=mems_i)
            hids.append(core_out)
            attn_entropy.append(attn_entropy_layer)

        core_out = self.drop(core_out)
        #print('before update mems hids shape: {}, mems shape {}'.format(hids[0].shape,mems[0].shape if mems else None))
        new_mems = self._update_mems(hids, mems, mlen, qlen)

        return core_out, new_mems, torch.stack(attn_entropy, dim=0)


    def forward(self, data, mems, rollout_step_list=None, mem_begin_index=None, dones=None):
        # from policy worker, data is either [1 x 256] or [2 x 256]
        #if torch.sum(mems) == 0:
        # data dim: B x dim
        # mems dim: t x B x (dim * num_layers)
        if mems is None: # never happens
            # print('INITIALIZED MEMS')
            mems = self.init_mems()
        else:
            # reshape mems: t x B x (dim x (n_layer+1)) -> [n_layer+1] x (t x B x hidden_dim)
            mems = torch.split(mems, self.d_model, dim=-1)

        if rollout_step_list is None: # from learner,
            data = data.reshape(
                int(self.cfg.batch_size // self.cfg.recurrence),
                self.cfg.chunk_size,
                -1).transpose(0, 1) # T x B x dim
        else: # from policy worker,
            data = data.unsqueeze(0) # 1 x B x dim
        # input observation should be either (1 x B x dim) or (T x B x dim)
        hidden, new_mems, attn_entropy = self._forward(data, mems=mems, rollout_step_list=rollout_step_list, mem_begin_index=mem_begin_index,
                                                       dones=dones)

        # reshape hidden: T x B x dim -> TB x dim
        hidden = hidden.transpose(0, 1).reshape(hidden.size(0) * hidden.size(1), -1)

        if rollout_step_list is not None: # if from actor, return only the last mems
            new_mems = [m[-1] for m in new_mems]

        new_mems = torch.cat(new_mems, dim=-1)
        #assert new_mems.size(-1) == (self.cfg.n_layer + 1) * self.d_model

        return hidden, new_mems, attn_entropy

    def get_mem_begin_index(self, mems_dones, actor_env_step):
        # mems_dones: (n_batch, n_seq, 1)
        # actor_env_step: (n_batch)
        assert mems_dones.shape[0] == actor_env_step.shape[0], (
            f'The number of batches should be same for mems_done ({mems_dones.shape[0]})'
            + f' and actor_env_step ({actor_env_step.shape[0]})'
        )
        mems_dones = mems_dones.squeeze(-1).cpu()
        actor_env_step = actor_env_step.cpu()

        arange = torch.arange(1, self.cfg.mem_len + 1, 1).unsqueeze(0)  # 0 ~ self.cfg.mem_len - 1, (1, n_seq)
        step_count_dones = mems_dones * arange  # (n_batch, n_seq)
        step_count_last_dones = step_count_dones.max(dim=-1).values  # (n_batch)
        numel_to_be_attentioned = self.cfg.mem_len - step_count_last_dones
        mem_begin_index = torch.min(numel_to_be_attentioned, actor_env_step)
        mem_begin_index = mem_begin_index.int().tolist()

        return mem_begin_index


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='unit test')

    parser.add_argument('--n_layer', type=int, default=4, help='')
    parser.add_argument('--n_rel_layer', type=int, default=4, help='')
    parser.add_argument('--n_head', type=int, default=2, help='')
    parser.add_argument('--d_head', type=int, default=2, help='')
    parser.add_argument('--d_model', type=int, default=200, help='')
    parser.add_argument('--d_embed', type=int, default=200, help='')
    parser.add_argument('--d_inner', type=int, default=200, help='')
    parser.add_argument('--dropout', type=float, default=0.0, help='')
    parser.add_argument('--cuda', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=1111, help='')
    parser.add_argument('--multi_gpu', action='store_true', help='')

    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    B = 4
    tgt_len, mem_len, ext_len = 36, 36, 0
    data_len = tgt_len * 20
    args.n_token = 10000

    import data_utils

    data = torch.LongTensor(data_len*B).random_(0, args.n_token).to(device)
    diter = data_utils.LMOrderedIterator(data, B, tgt_len, device=device, ext_len=ext_len)

    cutoffs = [args.n_token // 2]
    tie_projs = [False] + [True] * len(cutoffs)

    for div_val in [1, 2]:
        for d_embed in [200, 100]:
            model = MemTransformerLM(args.n_token, args.n_layer, args.n_head,
                            args.d_model, args.d_head, args.d_inner, args.dropout,
                            dropatt=args.dropout, tie_weight=True,
                            d_embed=d_embed, div_val=div_val,
                            tie_projs=tie_projs, pre_lnorm=True,
                            tgt_len=tgt_len, ext_len=ext_len, mem_len=mem_len,
                            cutoffs=cutoffs, attn_type=0).to(device)

            print(sum(p.numel() for p in model.parameters()))

            mems = tuple()
            for idx, (inp, tgt, seqlen) in enumerate(diter):
                print('batch {}'.format(idx))
                out = model(inp, tgt, *mems)
                mems = out[1:]