import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
import math

class HNNLayer(nn.Module):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, act, use_bias):
        super(HNNLayer, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c, dropout, use_bias)
        self.hyp_act = HypAct(manifold, c, c, act)

    def forward(self, x):
        h = self.linear.forward(x)
        h = self.hyp_act.forward(h)
        return h
    
class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """

    def __init__(self, manifold, in_features, out_features, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = manifold.c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
        mv = self.manifold.mobius_matvec(drop_weight, x)
        res = self.manifold.projx(mv)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1))
            hyp_bias = self.manifold.expmap0(bias)
            hyp_bias = self.manifold.projx(hyp_bias)
            res = self.manifold.mobius_add(res, hyp_bias,)
            res = self.manifold.projx(res)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )

class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold_in, manifold_out, act):
        super(HypAct, self).__init__()
        self.manifold_in = manifold_in
        self.manifold_out = manifold_out
        self.act = act

    def forward(self, x):
        xt = self.act(self.manifold_in.logmap0(x))
        xt = self.manifold_out.proj_tan0(xt)
        return self.manifold_out.projx(self.manifold_out.expmap0(xt))

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.manifold_in.c, self.manifold_out.c
        )