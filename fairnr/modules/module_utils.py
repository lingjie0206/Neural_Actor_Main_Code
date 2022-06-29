# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import (
    LayerNorm, TransformerDecoderLayer,
    TransformerEncoderLayer)
from fairseq.utils import get_activation_fn

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m
    

class IndTransformerDecoderLayer(TransformerDecoderLayer):
    """
    This is a wrapper for the original TransformerDecoderLayer
    However, no self-attention is performed between points.
    """
    def __init__(self, args):
        super().__init__(args)

        self.self_attn = None  # disable self-attention
        self.self_attn_layer_norm = None

    def forward(self, x, encoder_out):
        residual = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        x, attn = self.encoder_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=None,
            incremental_state=None,
            static_kv=True,
            need_weights=True,
            need_head_weights=False,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn, None


class JointMeshTransformerEncoderLayer(TransformerEncoderLayer):
    """
    This is a wrapper for the original TransformerDecoderLayer
    However, no self-attention is performed between points.
    """
    def __init__(self, args):
        super().__init__(args)

        import copy
        self.encoder_attn = self.build_encoder_attention(args.encoder_embed_dim, args)
        self.reverse_encoder_attn = copy.deepcopy(self.encoder_attn)
        self.encoder_attn_layer_norm = copy.deepcopy(self.self_attn_layer_norm)
        self.reverse_encoder_attn_layer_norm = copy.deepcopy(self.self_attn_layer_norm)

    def build_encoder_attention(self, embed_dim, args):
        from fairseq.modules import MultiheadAttention
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True
        )

    def forward_self_attention(self, x):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=None,
            attn_mask=None,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
        return x

    def forward_endec_attention(self, x, v):
        residual = v
        if self.normalize_before:
            v = self.reverse_encoder_attn_layer_norm(v)
        v, attn = self.reverse_encoder_attn(
            query=v,
            key=x,
            value=x,
            key_padding_mask=None,
            incremental_state=None,
            static_kv=True,
            need_weights=False,
            need_head_weights=False,
        )
        v = self.dropout_module(v)
        v = residual + v
        if not self.normalize_before:
            v = self.reverse_encoder_attn_layer_norm(v)

        residual = x
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, attn = self.encoder_attn(
            query=x,
            key=v,
            value=v,
            key_padding_mask=None,
            incremental_state=None,
            static_kv=True,
            need_weights=False,
            need_head_weights=False,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        
        return x, v

    def forward_ffn(self, x):
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x

    def forward(self, x, v):
        # self attention between joints
        x = self.forward_self_attention(x)
        x, v = self.forward_endec_attention(x, v)
        x = self.forward_ffn(x)
        v = self.forward_ffn(v)
        return x, v


class PosEmbLinear(nn.Module):

    def __init__(self, in_dim, out_dim, no_linear=False, scale=1, *args, **kwargs):
        super().__init__()
        assert out_dim % (2 * in_dim) == 0, "dimension must be dividable"
        half_dim = out_dim // 2 // in_dim
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        
        self.emb = nn.Parameter(emb, requires_grad=False)
        self.linear = Linear(out_dim, out_dim) if not no_linear else None
        self.scale = scale
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cat_input = False

    def forward(self, x):
        assert x.size(-1) == self.in_dim, "size must match"
        sizes = x.size()
        x = self.scale * x.unsqueeze(-1) @ self.emb.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(*sizes[:-1], self.out_dim)
        if self.linear is not None:
            return self.linear(x)
        return x


class NeRFPosEmbLinear(nn.Module):

    def __init__(self, in_dim, out_dim, angular=False, no_linear=False, cat_input=False, no_pi=False):
        super().__init__()
        assert out_dim % (2 * in_dim) == 0, "dimension must be dividable"
        L = out_dim // 2 // in_dim
        emb = torch.exp(torch.arange(L, dtype=torch.float) * math.log(2.))
        
        if (not angular) and (not no_pi):
            emb = emb * math.pi
            
        self.emb = nn.Parameter(emb, requires_grad=False)
        self.angular = angular
        self.linear = Linear(out_dim, out_dim) if not no_linear else None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cat_input = cat_input

    def forward(self, x):
        assert x.size(-1) == self.in_dim, "size must match"
        sizes = x.size() 
        inputs = x.clone()

        if self.angular:
            x = torch.acos(x.clamp(-1 + 1e-6, 1 - 1e-6))
        x = x.unsqueeze(-1) @ self.emb.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(*sizes[:-1], self.out_dim)
        if self.linear is not None:
            x = self.linear(x)
        if self.cat_input:
            x = torch.cat([x, inputs], -1)
        return x

    def extra_repr(self) -> str:
        outstr = 'Sinusoidal (in={}, out={}, angular={})'.format(
            self.in_dim, self.out_dim, self.angular)
        if self.cat_input:
            outstr = 'Cat({}, {})'.format(outstr, self.in_dim)
        return outstr


class FCLayer(nn.Module):
    """
    Reference:
        https://github.com/vsitzmann/pytorch_prototyping/blob/10f49b1e7df38a58fd78451eac91d7ac1a21df64/pytorch_prototyping.py
    """
    def __init__(self, in_dim, out_dim, with_ln=True, use_softplus=False, non_linear=True):
        super().__init__()
        self.net = [nn.Linear(in_dim, out_dim)]
        if with_ln:
            self.net += [nn.LayerNorm([out_dim])]
        if non_linear:
            self.net += [nn.ReLU()] if not use_softplus else [nn.Softplus(beta=100)]
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x) 


class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False,
                 with_ln=True,
                 use_softplus=False):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features, hidden_ch, with_ln, use_softplus))
        for i in range(num_hidden_layers):
            self.net.append(FCLayer(hidden_ch, hidden_ch, with_ln, use_softplus))
        if outermost_linear:
            self.net.append(Linear(hidden_ch, out_features))
        else:
            self.net.append(FCLayer(hidden_ch, out_features, with_ln, use_softplus))
        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights)

    def __getitem__(self, item):
        return self.net[item]

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, input):
        return self.net(input)


class InvertableMapping(nn.Module):
    def __init__(self, style='simple'):
        super().__init__()
        self.style = style

    def f(self, x):  # (0, 1) --> (0, +inf)
        if self.style == 'simple':
            return x / (1 - x + 1e-7)
        raise NotImplementedError
    
    def g(self, y):  # (0, +inf) --> (0, 1)
        if self.style == 'simple':
            return y / (1 + y)
        raise NotImplementedError

    def dy(self, x):
        if self.style == 'simple':
            return 1 / ((1 - x) ** 2 + 1e-7)
        raise NotImplementedError