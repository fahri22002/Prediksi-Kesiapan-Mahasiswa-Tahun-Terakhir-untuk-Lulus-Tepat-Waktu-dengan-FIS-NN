import torch
import torch.nn as nn
import numpy as np

# Sigma harus float dan positif
class GaussianMF(nn.Module):
    def __init__(self, mean, sigma):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(float(mean)))
        self.sigma = nn.Parameter(torch.tensor(float(sigma)))

    def forward(self, x):
        sigma = torch.clamp(self.sigma, min=1e-3)
        return torch.exp(-0.5 * ((x - self.mean) / sigma) ** 2)

class ANFIS(nn.Module):
    def __init__(self, n_inputs=6, mfs_per_input=2, n_classes=3):  # tambah argumen n_classes
        super().__init__()
        self.n_inputs = n_inputs
        self.mfs_per_input = mfs_per_input
        self.n_rules = mfs_per_input ** n_inputs
        self.n_classes = n_classes

        self.mf_layers = nn.ModuleList([
            nn.ModuleList([GaussianMF(mean=0.5 + i, sigma=1.0) for i in range(mfs_per_input)])
            for _ in range(n_inputs)
        ])

        # rule_weights sekarang shape: [n_rules, n_inputs + 1, n_classes]
        # karena tiap rule punya parameter linear untuk tiap kelas
        self.rule_weights = nn.Parameter(torch.randn(self.n_rules, n_inputs + 1, n_classes))

    def forward(self, x):
        batch_size = x.size(0)

        mf_values = []
        for i in range(self.n_inputs):
            mf_x = [mf(x[:, i]) for mf in self.mf_layers[i]]
            mf_values.append(torch.stack(mf_x, dim=1))

        from itertools import product
        rule_indices = list(product(range(self.mfs_per_input), repeat=self.n_inputs))

        firing_strengths = []
        for idx in rule_indices:
            rule_mfs = [mf_values[i][:, idx[i]] for i in range(self.n_inputs)]
            prod = torch.stack(rule_mfs, dim=0).prod(dim=0)
            firing_strengths.append(prod)

        firing_strengths = torch.stack(firing_strengths, dim=1)
        norm_strengths = firing_strengths / torch.sum(firing_strengths, dim=1, keepdim=True)

        x_aug = torch.cat([x, torch.ones(batch_size, 1)], dim=1)  # [batch, n_inputs+1]

        # Hitung consequent untuk semua kelas:
        # x_aug shape: [batch, n_inputs+1]
        # rule_weights shape: [n_rules, n_inputs+1, n_classes]
        # output tiap rule per kelas: [batch, n_rules, n_classes]
        consequent = torch.einsum('bi,rij->brj', x_aug, self.rule_weights)  # einsum buat dot product

        # normalisasi dan agregasi per kelas
        # norm_strengths shape: [batch, n_rules], consequent: [batch, n_rules, n_classes]
        output = torch.sum(norm_strengths.unsqueeze(-1) * consequent, dim=1)  # [batch, n_classes]

        return output
