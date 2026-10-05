import torch
from torch import nn


def lat_attn(P, M, Z, Fe):
    Bb, N, F = Z.shape
    Za = Z[:, :, :Fe]
    At = torch.einsum('BNi,Hij,BMj->HBNM', Za, M, Za)
    PZ = torch.einsum('Hij,BNj->HBNi', P, Z)
    return torch.einsum('HBNM,HBMi->BNi', At, PZ) / float(N)


class LatticeTF(nn.Module):
    def __init__(self, nl, m, device="cpu"):
        super().__init__()
        self.nl = nl
        self.m = m
        fd = m + 3
        Fe = m + 2
        self.fd = fd
        self.Fe = Fe
        self.pc = m + 1
        self.mc = m + 2
        self.P = nn.Parameter(torch.zeros(nl, 1, fd, fd, dtype=torch.float64, device=device))
        self.M = nn.Parameter(torch.zeros(nl, 1, Fe, Fe, dtype=torch.float64, device=device))
        self.S = nn.Parameter(torch.zeros(nl, 1, fd, fd, dtype=torch.float64, device=device))
        with torch.no_grad():
            for l in range(nl):
                self.M[l, 0, :m, :m] = 0.01 * torch.eye(m, dtype=torch.float64, device=device)
                self.M[l, 0, m, m] = 0.1
                self.M[l, 0, :m, m] = 0.01
                self.M[l, 0, m, :m] = 0.01
        Pm = torch.ones(fd, dtype=torch.float64, device=device)
        Pm[:m] = 0.0
        Pm[m] = 1.0
        self.register_buffer('Pmask', Pm)

    def forward(self, Z):
        mask = (Z[:, :, self.mc] > 0).float()
        Zl = Z.clone()
        for l in range(self.nl):
            P = self.P[l] * self.Pmask[None, :, None]
            S = (self.S[l] * self.Pmask[:, None])[0]
            Zl = Zl + lat_attn(P, self.M[l], Zl, self.Fe)
            Zl = Zl + torch.nn.functional.linear(Zl, S)
            Zl[:, :, self.mc] = mask
        return Zl


def lat_loss(out, tgt, m, ofi, eps=1e-14):
    p = out[:, ofi, m + 1]
    dd = p - tgt
    return ((dd * dd).sum(1) / ((tgt * tgt).sum(1) + eps)).mean()
