import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class TemporalCoderNonLinear(nn.Module):
    def __init__(self, feat_size):
        super(TemporalCoderNonLinear, self).__init__()
        self.feat_size = feat_size

    def forward(self, x, times):
        tcodes = torch.add(torch.cos(3.14159 * times), 1.0)
        return torch.mul(x, tcodes.repeat(1, self.feat_size, 1))


class TemporalCoderLinear(nn.Module):
    def __init__(self, dt):
        super(TemporalCoderLinear, self).__init__()
        self.tsize = dt

    def forward(self, x, times):
        tcodes = torch.div(times, self.tsize)
        return torch.clamp((x - tcodes.repeat(1, self.feat_size, 1)), -1.0, 1.0)


class TemporalCoderPhase(nn.Module):
    def __init__(self, feat_size):
        super(TemporalCoderPhase, self).__init__()
        self.ls_by_2 = int(feat_size / 2)
        self.b = torch.arange(0, self.ls_by_2).type("torch.FloatTensor").cuda()
        self.b_by_latent = self.b / feat_size
        self.pow_vec_reci = 1 / torch.pow(1000, (self.b_by_latent))

    def forward(self, x, times):
        t = times.view(times.shape[0], -1, 1)

        pes = torch.sin(100 * t * self.pow_vec_reci)
        pec = torch.cos(100 * t * self.pow_vec_reci)

        pe = torch.stack([pes, pec], axis=2).view(
            t.shape[0], t.shape[1], self.ls_by_2 * 2
        )
        return x + pe.permute(0, 2, 1)
