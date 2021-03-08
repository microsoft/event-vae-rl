import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

from tcoder import *


class EventVAE(nn.Module):
    def __init__(
        self, data_len, latent_size, params, tc=False, decoder="image",
    ):
        """
        data_len (int)      : 2 for X and Y, 3 to include polarity as well
        latent_size (int)            : Size of latent vector
        tc (bool)           : True if temporal coding is included
        params (list)       : Currently just the image resolution (H, W)
        """
        super(EventVAE, self).__init__()

        # Context feature dimensionality can be tuned
        self.feat_size = 1024
        self.latent_size = latent_size

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

        self.encoder = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, latent_size * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, params[0] * params[1]),
        )

        # Reconstructed frame is between [0, 1] if we don't consider polarity,
        # otherwise, range is [-1, 1]
        if decoder == "image":
            if data_len == 2:
                self.opa = nn.Sigmoid()
            elif data_len == 3:
                self.opa = nn.Tanh()

        self.weight_init()
        self.m = torch.nn.ReLU()
        self.tcode = tc
        self.temporal_coder = TemporalCoderPhase(self.feat_size)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def weight_init(self):
        self.featnet.apply(self.kaiming_init)
        self.encoder.apply(self.kaiming_init)
        self.decoder.apply(self.kaiming_init)

    def sample_latent(self, mu, logvar):
        # Reparameterization trick

        std = torch.exp(0.5 * logvar)
        eps = Variable(std.data.new(std.size()).normal_())

        return mu + eps * std

    def forward(self, x, times=None):
        # ECN computes per-event spatial features
        x = self.featnet(x)

        # Temporal embeddings are added to per-event features
        if self.tcode:
            x = self.m(self.temporal_coder(x, times))

        # Symmetric function to reduce N features to 1 a la PointNet
        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(-1, 1024)

        # Compress to latent space
        dist = self.encoder(x)

        # Compute latent vector
        mu = dist[:, : self.latent_size]
        logvar = dist[:, self.latent_size :]
        z = self.sample_latent(mu, logvar)

        # Decode to event frame given a latent vector
        x_o = self.decoder(z)
        x_o = self.opa(x_o)

        return x_o, z, mu, logvar
