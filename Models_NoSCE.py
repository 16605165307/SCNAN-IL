import torch
from torch import nn
import torch.nn.functional as F


n_components = 20  # 表示a的维度
fault_size = 52
latent_dim = 20  # 噪音维度！
embedding_size = 52


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class Embedding_Net(nn.Module):
    def __init__(self, dim, lambda_):
        super(Embedding_Net, self).__init__()
        self.l11 = nn.Linear(52, dim[0])
        self.l12 = nn.Linear(dim[0], dim[0] * 5)
        self.l13 = nn.Linear(dim[0] * 5, dim[1])
        self.l14 = nn.Linear(2 * dim[1], 52)

        self.l21 = nn.Linear(20, dim[0])
        self.l22 = nn.Linear(dim[0], dim[0] * 5)
        self.l23 = nn.Linear(dim[0] * 5, dim[1])
        self.l24 = nn.Linear(2 * dim[1], 20)

        self.bn1 = nn.BatchNorm1d(dim[0])
        self.bn2 = nn.BatchNorm1d(dim[1])
        self.bn3 = nn.BatchNorm1d(dim[0] * 5)
        self.lambda_ = lambda_

    def compability_loss(self, z1, z2):
        N, D = z1.shape
        c = self.bn2(z1).T @ self.bn2(z2) / N
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambda_[3] * off_diag
        return loss

    def compute_loss(self, z1, z2, x, a, x_, a_):
        loss_R1 = self.lambda_[0] * F.mse_loss(a, a_)
        loss_R2 = self.lambda_[1] * F.mse_loss(x, x_)
        loss_CM = self.compability_loss(z1, z2)
        loss_CM = self.lambda_[2] * loss_CM
        loss = loss_R1 + loss_R2 + loss_CM
        return loss_R1, loss_R2, loss_CM, loss

    def transform(self, x, a):
        z1 = self.l11(x)
        z1 = torch.relu(self.bn1(z1))
        z1 = self.l12(z1)
        z1 = torch.relu(self.bn3(z1))
        z1 = self.l13(z1)

        z2 = self.l21(a)
        z2 = torch.relu(self.bn1(z2))
        z2 = self.l22(z2)
        z2 = torch.relu(self.bn3(z2))
        z2 = self.l23(z2)
        return z1, z2

    def reconstruction(self, z1, z2):
        f1 = torch.cat([z1, z2], dim=1)
        f2 = torch.cat([z2, z1], dim=1)
        x_ = self.l14(f1)
        a_ = torch.sigmoid(self.l24(f2))
        return x_, a_

    def forward(self, x, a):
        z1, z2 = self.transform(x, a)
        x_, a_ = self.reconstruction(z1, z2)
        loss_R1, loss_R2, loss_CM, loss = self.compute_loss(z1, z2, x, a, x_, a_)
        package = {'z1': z1, 'z2': z2, 'x': x, 'x_': x_, 'r1': loss_R1,
                   'r2': loss_R2, 'cm': loss_CM, 'loss': loss}
        return package


class SClassifier(nn.Module):
    def __init__(self, output_dim):
        super(SClassifier, self).__init__()
        self.h1 = torch.nn.Linear(52, 32)
        self.b1 = torch.nn.BatchNorm1d(32)
        self.a1 = torch.nn.LeakyReLU()
        self.h2 = torch.nn.Linear(32, 32)
        self.b2 = torch.nn.BatchNorm1d(32)
        self.a2 = torch.nn.LeakyReLU()
        self.sm = torch.nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.a1(self.b1(self.h1(x)))
        x = self.a2(self.b2(self.h2(x)))
        x = self.sm(x)
        return x


class SClassifier_Online(nn.Module):
    def __init__(self, output_dim):
        super(SClassifier_Online, self).__init__()
        self.h1 = torch.nn.Linear(52, 32)
        self.b1 = torch.nn.BatchNorm1d(32)
        self.a1 = torch.nn.LeakyReLU()
        self.h2 = torch.nn.Linear(32, 32)
        self.b2 = torch.nn.BatchNorm1d(32)
        self.a2 = torch.nn.LeakyReLU()
        self.sm = torch.nn.Linear(32, output_dim)

    def forward(self, x):
        x = self.a1(self.b1(self.h1(x)))
        x = self.a2(self.b2(self.h2(x)))
        x = self.sm(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_components, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(),
            nn.Linear(128, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.LeakyReLU(),
            nn.Linear(128, embedding_size),
            nn.Tanh()
        )

    def forward(self, z, att):
        z = torch.cat([z, att], axis=-1)
        output = self.model(z)
        fault = output.reshape(z.shape[0], embedding_size)
        return fault


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_size + n_components, 256),
            torch.nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, fault, att):
        z = torch.cat([fault.reshape(fault.shape[0], -1), att], axis=-1)
        prob = self.model(z)
        return prob


class UClassifier(nn.Module):
    def __init__(self, output_dim):
        super(UClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(embedding_size, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(),
            nn.Linear(32, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.LeakyReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        output = self.model(x)
        return output