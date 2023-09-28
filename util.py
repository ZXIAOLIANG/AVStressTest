import numpy as np
import random
import torch
from typing import Callable


def cg(A: Callable, b: torch.Tensor, steps: int, tol: float = 1e-6) -> torch.Tensor: # noqa
    x = torch.zeros_like(b)
    r = b - A(x)
    d = r.clone()
    tol_new = r.t() @ r
    for _ in range(steps):
        if tol_new < tol:
            break
        q = A(d)
        alpha = tol_new / (d.t() @ q)
        x += alpha * d
        r -= alpha * q
        tol_old = tol_new.clone()
        tol_new = r.t() @ r
        beta = tol_new / tol_old
        d = r + beta * d
    return x

def categorical_kl(p_nk: torch.Tensor, q_nk: torch.Tensor):
    # https://github.com/joschu/modular_rl/blob/master/modular_rl/distributions.py
    ratio_nk = p_nk / (q_nk + 1e-6)
    ratio_nk[p_nk == 0] = 1
    ratio_nk[(q_nk == 0) & (p_nk != 0)] = np.inf
    return (p_nk * torch.log(ratio_nk)).sum(dim=1)

def get_flat_params_from(model: torch.nn.Module):
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_flat_params_to(params: torch.nn.Module.parameters, model: torch.nn.Module):
    pointer = 0
    for p in model.parameters():
        p.data.copy_(params[pointer:pointer + p.data.numel()].view_as(p.data))
        pointer += p.data.numel()

# def gae(rewards, values, episode_ends, next_values, gamma, lam):
#     """Compute generalized advantage estimate.
#         rewards: a list of rewards at each step.
#         values: the value estimate of the state at each step.
#         episode_ends: an array of the same shape as rewards, with a 1 if the
#             episode ended at that step and a 0 otherwise.
#         gamma: the discount factor.
#         lam: the GAE lambda parameter.
#     """
#     # Invert episode_ends to have 0 if the episode ended and 1 otherwise
#     # episode_ends = (episode_ends * -1) + 1
#     N = len(rewards)
#     for i in range(len(rewards)):
#         values[i].append(next_values[i])
#     print("gae N: {}".format(N))
#     # T = rewards.shape[1]
#     # gae_step = np.zeros((N, ))
#     advs = []
#     for i in reversed(range(len(rewards))):
#         gae_step = 0
#         for j in reversed(range(len(rewards[i]))):
#             delta = rewards[i][j] + gamma * values[i][j+1] * (not episode_ends[i][j]) - values[i][j]
#             gae_step = delta + gamma * lam * episode_ends[i][j] * gae_step
#             advs.insert(0, gae_step)
#     # print(returns)
#     # for t in reversed(range(T - 1)):
#     #     # First compute delta, which is the one-step TD error
#     #     delta = rewards[:, t] + gamma * values[:, t + 1] * episode_ends[:, t] - values[:, t] 
#     #     # Then compute the current step's GAE by discounting the previous step
#     #     # of GAE, resetting it to zero if the episode ended, and adding this
#     #     # step's delta
#     #     gae_step = delta + gamma * lam * episode_ends[:, t] * gae_step
#     #     # And store it
#     #     returns[:, t] = gae_step + values[:, t]
#     return np.hstack(advs)