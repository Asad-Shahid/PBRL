import os
from glob import glob
import math
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

import torch.nn.functional as F
from torch import distributions as pyd
from torch.distributions.transforms import TanhTransform


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        try:
            self.base_dist = pyd.Normal(loc, scale)
        except (AssertionError, ValueError) as e:
            print(torch.where(torch.isnan(loc)))
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu

    def entropy(self):
        return self.base_dist.entropy()


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


def add_normal_noise(x, std, noise_bounds=None, out_bounds=None):
    noise = torch.normal(torch.zeros(x.shape, dtype=x.dtype, device=x.device),
                         torch.full(x.shape, std, dtype=x.dtype, device=x.device))
    if noise_bounds is not None:
        noise = noise.clamp(noise_bounds[0], noise_bounds[1])
    out = x + noise
    if out_bounds is not None:
        out = out.clamp(out_bounds[0], out_bounds[1])
    return out


def add_mixed_normal_noise(x, std_max, std_min, noise_bounds=None, out_bounds=None):
    std_seq = torch.linspace(std_min, std_max,
                             x.shape[0]).to(x.device).unsqueeze(-1).expand(x.shape)

    noise = torch.normal(torch.zeros(x.shape, dtype=x.dtype, device=x.device),
                         std_seq)
    if noise_bounds is not None:
        noise = noise.clamp(noise_bounds[0], noise_bounds[1])
    out = x + noise
    if out_bounds is not None:
        out = out.clamp(out_bounds[0], out_bounds[1])
    return out


def optimizer_update(optimizer, objective, max_grad_norm=0.5):
    optimizer.zero_grad(set_to_none=True)
    objective.backward()
    if max_grad_norm is not None:
        grad_norm = clip_grad_norm_(parameters=optimizer.param_groups[0]["params"],
                                    max_norm=max_grad_norm)
    else:
        grad_norm = None
    optimizer.step()
    return grad_norm


# def soft_update_target_network(target, source, tau):
#     for target_param, source_param in zip(target.parameters(), source.parameters()):
#         target_param.data.copy_((1 - tau) * source_param.data + tau * target_param.data)


@torch.no_grad()
def soft_update_target_network(target_net, current_net, tau: float):
    for tar, cur in zip(target_net.parameters(), current_net.parameters()):
        tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))


# required when we load optimizer from a checkpoint
def optimizer_cuda(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def get_ckpt_path(base_dir, ckpt_num):
    if ckpt_num is None:
        return get_recent_ckpt_path(base_dir)
    files = glob(os.path.join(base_dir, "*.pt"))
    for f in files:
        if 'ckpt_%08d.pt' % ckpt_num in f:
            return f, ckpt_num
    raise Exception("Did not find ckpt_%s.pt" % ckpt_num)


def get_recent_ckpt_path(base_dir):
    files = glob(os.path.join(base_dir, "*.pt"))
    # files.sort()
    sorted(files)
    if len(files) == 0:
        return None, None
    max_step = max([f.rsplit('_', 1)[-1].split('.')[0] for f in files], key=int)
    paths = [f for f in files if max_step in f]
    if len(paths) == 1:
        return paths[0], int(max_step)
    else:
        raise Exception("Multiple most recent ckpts %s" % paths)
