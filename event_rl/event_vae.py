import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from tcoder import *


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class EventVAE(nn.Module):
    def __init__(
        self, data_len, latent_size, params, tc=False,
    ):
        """
        data_len (int)      : 2 for X and Y, 3 to include polarity as well
        latent_size (int)            : Size of latent vector
        tc (bool)           : True if temporal coding is included
        """
        super(EventVAE, self).__init__()

        # Context feature dimensionality can be tuned
        self.feat_size = 1024
        self.latent_size = latent_size

        if data_len == 2:
            self.featnet = nn.Sequential(
                nn.Conv1d(data_len, 64, 1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 1024, 1),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
            )

        # Empirically better results without batch norm for pol + tc - YMMV
        elif data_len == 3:
            self.featnet = nn.Sequential(
                nn.Conv1d(data_len, 64, 1),
                nn.ReLU(),
                nn.Conv1d(64, 128, 1),
                nn.ReLU(),
                nn.Conv1d(128, 1024, 1),
                nn.ReLU(),
            )

        self.encoder = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, latent_size * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, params[0] * params[1]),
            nn.Tanh(),
        )

        self.weight_init()
        self.tcode = tc

        # See te.py for other types of encoding
        self.temporal_coder = TemporalCoderPhase(self.feat_size)
        self.m = torch.nn.ReLU()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def sample_latent(self, mu, logvar):
        # Reparameterization trick

        std = torch.exp(0.5 * logvar)
        eps = Variable(std.data.new(std.size()).normal_())

        return mu + eps * std

    def encode(self, x, times=None):
        # ECN computes per-event spatial features
        x = self.featnet(x)

        # Temporal embeddings are added to per-event features
        if self.tcode:
            x = self.m(self.temporal_coder(x, times))

        # Symmetric function to reduce N features to 1 a la PointNet
        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(-1, self.feat_size)

        # Compress to latent space
        dist = self.encoder(x)

        # Compute latent vector
        mu = dist[:, : self.latent_size]
        logvar = dist[:, self.latent_size :]
        z = self.sample_latent(mu, logvar)

        return z

    def decode(self, z):
        # Decode to event frame given a latent vector
        x_o = self.decoder(z)

        return x_o

    def compute_feature(self, x):
        # Only go through the ECN to provide 'spatial context feature'
        x = self.featnet(x)

        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(-1, self.feat_size)

        return x

    def forward(self, x, times=None):
        x = self.featnet(x)

        if self.tcode:
            x = self.m(self.temporal_coder(x, times))

        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(-1, 1024)
        dist = self.encoder(x)

        mu = dist[:, : self.latent_size]
        logvar = dist[:, self.latent_size :]
        z = self.sample_latent(mu, logvar)

        # Compute reconstruction
        x_o = self.decoder(z)

        return x_o, z, mu, logvar
