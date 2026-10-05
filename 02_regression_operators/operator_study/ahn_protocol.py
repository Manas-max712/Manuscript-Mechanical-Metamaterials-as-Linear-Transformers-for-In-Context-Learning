import torch
from . import ahn_upstream as ahn

COMMIT = 'b6ddc35f0f83e3f8f26e45548e94bac2adafdd50'


def clip_function(isotropic):
    def clip_and_step(allparam, optimizer, clip_r=None):
        grad = allparam.grad
        norm = None
        if clip_r is not None:
            groups = [grad] if isotropic else grad.reshape(-1, 5, 5)
            for group in groups:
                norm = group.norm().item()
                if norm > clip_r:
                    group.mul_(clip_r / norm)
        optimizer.step()
        return norm
    return clip_and_step


def covariance(case, seed, device):
    if case == 'isotropic':
        return None, None
    if case == 'upstream-rotated':
        draw = torch.FloatTensor(5, 5).uniform_(-1, 1).to(device)
        return torch.linalg.svd(draw)[0], torch.diag(
            torch.FloatTensor([1, 1, .5, .25, 1])).to(device)
    if case == 'paper-rotated':
        generator = torch.Generator(device='cpu').manual_seed(0)
        draw = torch.FloatTensor(5, 5).uniform_(-1, 1, generator=generator).double()
        u = torch.linalg.svd(draw)[0]
        return u.float().to(device), torch.diag(
            torch.FloatTensor([1, 1, .5, .25, 1])).to(device)
    if case == 'diagonal':
        return torch.eye(5, device=device), torch.diag(torch.sqrt(
            torch.tensor([4., 2., 1., .5, .25], device=device)))
    raise ValueError(case)

