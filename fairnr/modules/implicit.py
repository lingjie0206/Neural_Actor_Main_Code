# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.utils import get_activation_fn
from fairnr.modules.hyper import HyperFC, MoEFCLayer
from fairnr.modules.module_utils import FCLayer, Linear


class BackgroundField(nn.Module):
    """
    Background (we assume a uniform color)
    """
    def __init__(self, out_dim=3, bg_color="1.0,1.0,1.0", min_color=-1, stop_grad=False, background_depth=5.0):
        super().__init__()

        if out_dim == 3:  # directly model RGB
            bg_color = [float(b) for b in bg_color.split(',')] if isinstance(bg_color, str) else [bg_color]
            if min_color == -1:
                bg_color = [b * 2 - 1 for b in bg_color]
            if len(bg_color) == 1:
                bg_color = bg_color + bg_color + bg_color
            bg_color = torch.tensor(bg_color)
        else:    
            bg_color = torch.ones(out_dim).uniform_()
            if min_color == -1:
                bg_color = bg_color * 2 - 1
        self.out_dim = out_dim
        self.bg_color = nn.Parameter(bg_color, requires_grad=not stop_grad)
        self.depth = background_depth

    def forward(self, x, **kwargs):
        return self.bg_color.unsqueeze(0).expand(
            *x.size()[:-1], self.out_dim)


class ImplicitField(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, 
                outmost_linear=False, with_ln=True, skips=None, 
                spec_init=True, use_softplus=False, experts=1):
        super().__init__()
        self.skips = skips
        self.net = []
        self.total_experts = experts
        prev_dim = in_dim
        for i in range(num_layers):
            next_dim = out_dim if i == (num_layers - 1) else hidden_dim
            if (i == (num_layers - 1)) and outmost_linear:
                if experts <= 1:
                    module = nn.Linear(prev_dim, next_dim)
                else:
                    module = MoEFCLayer(prev_dim, next_dim, experts, nonlinear=False)
            else:
                if experts <= 1:
                    module = FCLayer(prev_dim, next_dim, with_ln=with_ln, use_softplus=use_softplus)
                else:
                    module = MoEFCLayer(prev_dim, next_dim, experts, nonlinear=True)
            self.net.append(module)
                
            prev_dim = next_dim
            if (self.skips is not None) and (i in self.skips) and (i != (num_layers - 1)):
                prev_dim += in_dim
        
        if num_layers > 0:
            self.net = nn.ModuleList(self.net)
            if spec_init:
                self.net.apply(self.init_weights)

    def forward(self, x, sizes=None):
        y = self.net[0]([x, sizes] if self.total_experts > 1 else x)
        for i in range(len(self.net) - 1):
            if (self.skips is not None) and (i in self.skips):
                y = torch.cat((x, y), dim=-1) / math.sqrt(2)    # BUG: I found IDR has sqrt(2)
            y = self.net[i+1]([y, sizes] if self.total_experts > 1 else y)
        return y

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


class HyperImplicitField(nn.Module):

    def __init__(self, hyper_in_dim, in_dim, out_dim, hidden_dim, num_layers, 
                outmost_linear=False):
        super().__init__()

        self.hyper_in_dim = hyper_in_dim
        self.in_dim = in_dim
        self.net = HyperFC(
            hyper_in_dim,
            1, 256, 
            hidden_dim,
            num_layers,
            in_dim,
            out_dim,
            outermost_linear=outmost_linear
        )

    def forward(self, x, c):
        assert (x.size(-1) == self.in_dim) and (c.size(-1) == self.hyper_in_dim)
        
        if x.dim() == 2:
            return self.net(c)(x.unsqueeze(0)).squeeze(0)
        return self.net(c)(x)


class SignedDistanceField(ImplicitField):
    """
    Predictor for density or SDF values.
    """
    def __init__(self, in_dim, hidden_dim, num_layers=1, 
                recurrent=False, with_ln=True, spec_init=True,
                experts=0):
        super().__init__(
            in_dim, in_dim, in_dim, num_layers-1, 
            with_ln=with_ln, spec_init=spec_init, experts=experts)
        self.recurrent = recurrent
        self.experts = experts
        if recurrent:
            assert num_layers > 1
            assert experts <= 1

            self.hidden_layer = nn.LSTMCell(input_size=in_dim, hidden_size=hidden_dim)
            self.hidden_layer.apply(init_recurrent_weights)
            lstm_forget_gate_init(self.hidden_layer)
        else:
            if num_layers > 0:
                if self.experts > 1:
                    self.hidden_layer = MoEFCLayer(in_dim, hidden_dim, experts, nonlinear=True)
                else:
                    self.hidden_layer = FCLayer(in_dim, hidden_dim, with_ln)
            else:
                self.hidden_layer = None
        prev_dim = hidden_dim if num_layers > 0 else in_dim
        if self.experts > 1:
            self.output_layer = MoEFCLayer(prev_dim, 1, experts, nonlinear=False)
        else:
            self.output_layer = nn.Linear(prev_dim, 1)
            self.output_layer.bias.data.fill_(0.5)   # set a bias for density
        
    def forward(self, x, sizes=None):
        if self.recurrent:
            assert sizes is None
            shape = x.size()
            state = None
            state = self.hidden_layer(x.view(-1, shape[-1]), state)
            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-5, max=5))
            return self.output_layer(state[0].view(*shape[:-1], -1)).squeeze(-1), state
        
        else:
            if self.experts > 1:
                if self.hidden_layer is not None:
                    x = self.hidden_layer([x, sizes])
                return self.output_layer([x, sizes]).squeeze(-1), None
            if self.hidden_layer is not None:
                return self.output_layer(self.hidden_layer(x)).squeeze(-1), None
            return self.output_layer(x).squeeze(-1), None

class TextureField(ImplicitField):
    """
    Pixel generator based on 1x1 conv networks
    """
    def __init__(self, in_dim, hidden_dim, num_layers, 
                with_alpha=False, with_ln=True, 
                spec_init=True, experts=0):
        out_dim = 3 if not with_alpha else 4
        super().__init__(in_dim, out_dim, hidden_dim, num_layers, 
            outmost_linear=True, with_ln=with_ln, spec_init=spec_init, experts=experts)


# ------------------ #
# helper functions   #
# ------------------ #
def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name: continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.)


def clip_grad_norm_hook(x, max_norm=10):
    total_norm = x.norm()
    total_norm = total_norm ** (1 / 2.)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return x * clip_coef