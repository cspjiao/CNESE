# coding=utf-8
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import compute_ricci, get_degree_list

from .GAN import Discriminator
from .VAE import VAE


class CNESE(nn.Module):
    def __init__(self, config, graph, features):
        super(CNESE, self).__init__()
        self.config = config
        self.G = graph

        self.adj = torch.from_numpy(nx.adjacency_matrix(self.G).todense()).to(
            self.config.device, dtype=torch.float32)

        self.features = torch.from_numpy(features).to(self.config.device, dtype=torch.float32)

        self.config.struct[0] = self.features.shape[1]
        self.degree = torch.from_numpy(get_degree_list(self.G)).to(self.config.device,
                                                                   dtype=torch.float32).reshape(
            -1, 1)
        self.degree = torch.log(self.degree + 1)

        self.vae = VAE(self.G, self.config).to(self.config.device, dtype=torch.float32)
        self.discriminator = Discriminator(self.config).to(self.config.device,
                                                           dtype=torch.float32)
        self.mlp = nn.ModuleList([
            nn.Linear(self.config.struct[-1], self.config.struct[-1]),
            nn.Linear(self.config.struct[-1], 1)
        ]).to(self.config.device, dtype=torch.float32)
        for i in range(len(self.mlp)):
            nn.init.xavier_uniform_(self.mlp[i].weight)
            nn.init.uniform_(self.mlp[i].bias)
        self.mseLoss = nn.MSELoss()
        self.bceLoss = nn.BCEWithLogitsLoss()
        self.maeLoss = nn.L1Loss()

    def generate_fake(self, h_state):
        z = torch.from_numpy(np.random.normal(0, 1, h_state.size())).to(self.config.device,
                                                                        dtype=torch.float32)
        return z

    # node-level loss
    def gan_loss(self, embedding):
        valid = torch.ones(embedding.size(0), 1).to(self.config.device,
                                                    dtype=torch.float32)
        fake = torch.zeros(embedding.size(0), 1).to(self.config.device,
                                                    dtype=torch.float32)
        z = self.generate_fake(embedding)
        d_logits = self.discriminator(embedding)
        real_loss = F.binary_cross_entropy_with_logits(self.discriminator(z), valid)
        fake_loss = F.binary_cross_entropy_with_logits(d_logits, fake)
        g_loss = F.binary_cross_entropy_with_logits(d_logits, valid)
        return fake_loss + real_loss + g_loss

    def mlp_out(self, embedding):
        for i, layer in enumerate(self.mlp):
            embedding = torch.relu(layer(embedding))
        return embedding

    def latent_loss(self, z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def forward(self, input_):
        features = self.features[input_]
        mu, sigma, embedding, vae_out = self.vae(features)

        vae_loss = self.config.alpha * self.mseLoss(vae_out, features)

        valid = torch.ones(embedding.size(0), 1).to(self.config.device,
                                                    dtype=torch.float32)
        fake = torch.zeros(embedding.size(0), 1).to(self.config.device,
                                                    dtype=torch.float32)
        z = self.generate_fake(embedding)
        g_loss = F.binary_cross_entropy_with_logits(self.discriminator(embedding), valid)

        fake_loss = F.binary_cross_entropy_with_logits(self.discriminator(z),
                                                       fake)
        real_loss = F.binary_cross_entropy_with_logits(self.discriminator(embedding.detach()),
                                                       valid)

        pred = self.mlp_out(embedding)
        mlp_loss = self.config.gamma * self.maeLoss(pred, self.degree[input_])

        return vae_loss + mlp_loss + self.config.beta * g_loss, fake_loss + real_loss

    def get_embedding(self):
        return self.vae.get_embedding(self.features)
