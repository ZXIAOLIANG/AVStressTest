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
from gym_wrapper import MultiAgentAutoResetWrapper
from util import *
from conjugate_gradient import cg
from cost import cost_function

n_attackers = 4
    
def config_env():
    env = gym.make("highway-perfect-target-fast-v0", render_mode="rgb_array")	
    env.configure({"observation": {	
                        "type": "MultiAgentObservation",	
                        "observation_config": {	
                            "type": "AttackerKinematics",	
                            "see_behind": True,	
                            "vehicles_count": n_attackers+1	
                        }	
                    },	
                   "attacker_num": n_attackers, 	
                   "controlled_vehicles": n_attackers,	
                   "close_vehicle_cost": 20,
                #    "time_penalty": -0.1/n_attackers,
                   "time_penalty": 0.0,	
                   "attacker_collide_each_other_reward": -2.5,	
                   "vicitm_collision_reward": 10.0/n_attackers,	
                   "randomize_starting_position": False,
                   "constraint_env": True,
                   "vis": False,
                   "testing": True})
    env.reset()
    victim_agent = FollowingVictimVulnerable(env)
    env.load_agents(n_attackers, victim_agent)
    env = MultiAgentAutoResetWrapper(env)
    return env

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class AttackerAgent(nn.Module):
    def __init__(self, envs, num_agents, hidden_dim):
        super(AttackerAgent, self).__init__()
        # print(envs.action_space)
        self.critic = nn.Sequential(
            layer_init(nn.Linear((n_attackers+1)*5, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.cost_critic = nn.Sequential(
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
            layer_init(nn.Linear(hidden_dim, 5), std=0.01),
        )
        

    def get_value(self, x):
        x = x.flatten(start_dim=-2)
        return self.critic(x)
    
    def get_cost_value(self, x):
        x = x.flatten(start_dim=-2)
        return self.cost_critic(x)

    def get_action_and_value(self, x, centralized_x,  action=None):
        x = x.flatten(start_dim=-2)
        centralized_x = centralized_x.flatten(start_dim=-2)
        logits = self.actor(x)
    
        dist = Categorical(logits=logits)	
        if action is None:	
            action = dist.sample()	
        return action, dist.log_prob(action), dist.entropy(), self.critic(centralized_x), logits, self.cost_critic(x)
    
    def get_action_info(self, x, action=None):
        x = x.flatten(start_dim=-2)
        logits = self.actor(x)
    
        dist = Categorical(logits=logits)	
        if action is None:	
            action = dist.sample()	
        return action, dist.log_prob(action), dist.entropy()
    def get_kl(self, x, b_logits):
        x = x.flatten(start_dim=-2)
        logits = self.actor(x)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        b_probs = torch.nn.functional.softmax(b_logits, dim=-1)
        kl = categorical_kl(probs, b_probs).mean()
        return kl


if __name__ == "__main__":
    seed = 5
    PATHS = []
    # folder_path = "saved_models\\macpo_perfect_victim_middle_seed_" + str(seed) + "_" + str(n_attackers)
    # folder_path = "saved_models\\macpo_perfect_victim_middle_seed_" + str(seed) + "_" + str(n_attackers) + "_100k"
    # folder_path = "saved_models\\macpo_perfect_victim_middle_seed_" + str(seed) + "_" + str(n_attackers) + "_time"
    
    # folder_path = "thesis_models_new\\macpo_one_critic_perfect_victim_middle_test_seed_" + str(seed) + "_" + str(n_attackers)
    # folder_path = "thesis_models_new\\macpo_one_critic_perfect_victim_vulnerable_combined_middle_seed_" + str(seed) + "_" + str(n_attackers)
    # folder_path = "thesis_models_new\\macpo_one_critic_perfect_victim_vulnerable_combined_middle_seed_5"  + "_" + str(n_attackers)
    # folder_path = "thesis_models_new\\macpo_one_critic_perfect_victim_back3_middle_cost20_seed_" + str(seed) + "_" + str(n_attackers)
    folder_path = "thesis_models_diff_start\\macpo_one_critic_perfect_victim_vulnerable_start1_rand_cost20_seed_5"  + "_" + str(n_attackers)
    # folder_path = "thesis_models_diff_start2\\macpo_one_critic_perfect_victim_vulnerable_start1_rand_cost5_seed_5"  + "_" + str(n_attackers)
    

    for i in range(n_attackers):
        PATHS.append(folder_path +"\\agent" + str(i) + ".pt")
    

    device = torch.device("cuda")

   
    envs = config_env()

    agents = []
    for i in range(n_attackers):
        agents.append(torch.load(PATHS[i], map_location=device).to(device))
        agents[i].eval()

    num_of_eps = 0
    while True:
        num_of_eps += 1
        terminated = [False for _ in range(n_attackers)]
        truncated = False
        obs, info = envs.reset()
        obs = torch.Tensor(obs).to(device)
        while not all(terminated) or truncated:
            actions_to_take = [0 for _ in range(n_attackers)]
            for i in range(n_attackers):
                action, *_ = agents[i].get_action_info(obs[i])
                actions_to_take[i] = action.cpu().item()
            obs, reward, terminated, truncated, info = envs.step(tuple(actions_to_take))
            # print(tuple(action.cpu().numpy()))
            obs = torch.Tensor(obs).to(device)
            # envs.render()
            # time.sleep(0.1)
        print("number of episodes: ", num_of_eps)
        if num_of_eps == 2000:
            break
        