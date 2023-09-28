import argparse
import os
import random
import time
from distutils.util import strtobool

import sys
sys.path.append('highway_envs')
import highway_env
highway_env.register_highway_envs()
from highway_env.envs.common.agents import FollowingVictimVulnerable

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gym_wrapper import AutoResetWrapper
from util import *
from conjugate_gradient import cg

n_attackers = 4

# def config_env():
#     env = gym.make("highway-centralized-perfect-target-fast-v0", render_mode="rgb_array")
#     env.configure({"observation": {
#                         "type": "Kinematics",
#                         "see_behind": True,
#                         "vehicles_count": n_attackers+1
#                     },
#                    "attacker_num": n_attackers, 
#                    "controlled_vehicles": n_attackers,
#                    "time_penalty": 0.0,	
#                    "close_vehicle_cost": 15,
#                    "randomize_starting_position": False,
#                    "constraint_env": True,
#                    "vis": True,
#                    "victim_lane_id": None})
#     env.reset()
#     victim_agent = FollowingVictimVulnerable(env)
#     env.load_agents(n_attackers, victim_agent)
#     env = AutoResetWrapper(env)
#     return env

def config_env():
    env = gym.make("highway-centralized-perfect-target-penalty-fast-v0", render_mode="rgb_array")
    env.configure({"observation": {
                        "type": "Kinematics",
                        "see_behind": True,
                        "vehicles_count": n_attackers+1
                    },
                   "attacker_num": n_attackers, 
                   "controlled_vehicles": n_attackers,
                   "time_penalty": 0.0,	
                   "close_vehicle_cost": 5,
                   "randomize_starting_position": False,
                   "constraint_env": True,
                   "vis": True,
                   "victim_index": 2,
                   "victim_lane_id": 1,
                   "testing": True})
    env.reset()
    victim_agent = FollowingVictimVulnerable(env)
    env.load_agents(n_attackers, victim_agent)
    env = AutoResetWrapper(env)
    return env

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AttackerAgent(nn.Module):
    def __init__(self, envs, num_agents, hidden_dim):
        super(AttackerAgent, self).__init__()
        # print(envs.action_space)
        self.nvec = envs.action_space.nvec
        self.critic = nn.Sequential(
            layer_init(nn.Linear((n_attackers+1)*5, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear((n_attackers+1)*5, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, self.nvec.sum()), std=0.01),
        )

    def get_value(self, x):
        x = x.flatten(start_dim=-2)
        return self.critic(x)
    
    def get_action_and_value(self, x,  action=None):
        x = x.flatten(start_dim=-2)
        logits = self.actor(x)
    
        split_logits = torch.split(logits, self.nvec.tolist(), dim=-1)
        # print("split logits shape: {}".format(split_logits.shape))
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action, logprob.sum(0), entropy.sum(0), self.critic(x), logits
    
    def get_action_info(self, x, action=None):
        x = x.flatten(start_dim=-2)
        logits = self.actor(x)

        split_logits = torch.split(logits, self.nvec.tolist(), dim=-1)
        # print("split logits shape: {}".format(split_logits.shape))
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action, logprob.sum(0), entropy.sum(0)
    
    def get_kl(self, x, b_logits):
        x = x.flatten(start_dim=-2)
        logits = self.actor(x)
        split_logits = torch.split(logits, self.nvec.tolist(), dim=-1)
        b_split_logits = torch.split(b_logits, self.nvec.tolist(), dim=-1)
        probs = [torch.nn.functional.softmax(logits_, dim=-1) for logits_ in split_logits]
        b_probs = [torch.nn.functional.softmax(b_logits_, dim=-1) for b_logits_ in b_split_logits]

        return torch.stack([categorical_kl(probs[i], b_probs[i]).mean() for i in range(len(split_logits))]).mean()

if __name__ == "__main__":
    seed = 5
    PATH = "trpo\\trpo_penalty_back3_cost15_seed_" + str(seed) + "_" + str(n_attackers) + ".pt"
    # PATH = "trpo\\trpo_vulnerable_seed_" + str(seed) + "_" + str(n_attackers) + ".pt"
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda")

    envs = config_env()

    agent=torch.load(PATH, map_location=device).to(device)
    agent.eval()

    num_of_eps = 0
    
    while True:
        num_of_eps += 1
        terminated = False
        truncated = False
        obs, info = envs.reset()
        obs = torch.Tensor(obs).to(device)
        while not terminated or truncated:
            action, *_ = agent.get_action_and_value(obs)
            obs, reward, terminated, truncated, info = envs.step(tuple(action.cpu().numpy()))
            # print(tuple(action.cpu().numpy()))
            obs = torch.Tensor(obs).to(device)
            envs.render()
            time.sleep(0.1)
        print("number of episodes: ", num_of_eps)
        if num_of_eps == 2000:
            break
    print(PATH)