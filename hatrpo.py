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
EPS = 1e-8

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--env-name", type=str, default="highway-perfect-target-fast-v0",
        help="the name of this experiment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=5,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=200000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=1000,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--cvf-coef", type=float, default=0.5,
        help="coefficient of the cost value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--damping", type=float, default=1e-1,
        help="the damping value for fisher vector product")
    parser.add_argument("--max-kl", type=float, default=0.05,
        help="the trust region for KL divergence")
    parser.add_argument("--max-constraint", type=float, default=0.1,
        help="the maximum constraint (d_k)")
    parser.add_argument("--constraint-annealing-factor", type=float, default=1e-6,
        help="the annealing factor of constraint")
    parser.add_argument("--line-search-fraction", type=float, default=0.5,
        help="line search fraction")
    parser.add_argument("--value-update-epochs", type=int, default=10,
        help="the number of epochs to update the value functions")
    parser.add_argument("--fraction-coef", type=float, default=0.5,
        help="the fraction coefficient")
    parser.add_argument("--ls-step", type=int, default=10,
        help="maximum number of line search steps")
    
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args

def config_env():
    env = gym.make(args.env_name, render_mode="rgb_array")	
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
                   "time_penalty": -0.1/n_attackers,
                   "time_penalty": 0.0,	
                   "attacker_collide_each_other_reward": -2.5,	
                   "vicitm_collision_reward": 10.0/n_attackers,	
                   "randomize_starting_position": False,
                   "constraint_env": True,
                   "vis": False,
                   "testing": False,
                   "victim_lane_id": 1})
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
    

    def get_action_and_value(self, x, centralized_x,  action=None):
        x = x.flatten(start_dim=-2)
        centralized_x = centralized_x.flatten(start_dim=-2)
        logits = self.actor(x)
    
        dist = Categorical(logits=logits)	
        if action is None:	
            action = dist.sample()	
        return action, dist.log_prob(action), dist.entropy(), self.critic(centralized_x), logits
    
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
    args = parse_args()
    PATHS = []
    folder_path = "saved_models/matrpo_newer_target_seed_" + str(args.seed) + "_" + str(n_attackers)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    for i in range(n_attackers):
        PATHS.append(folder_path +"/agent" + str(i) + ".pt")
    env_name = "highway_custom_matrpo_newer_target_" + str(n_attackers)
    run_name = f"{env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/matrpo/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(args.device)

    envs = config_env()
    agents = []
    critic_optimizers = []
    for i in range(n_attackers):
        agents.append(AttackerAgent(envs, n_attackers, 128).to(device))
        critic_optimizers.append(optim.Adam(agents[i].critic.parameters(), lr=args.learning_rate, eps=1e-5))

    obs = torch.zeros((n_attackers, args.num_steps, args.num_envs) + (n_attackers+1,5)).to(device)
    centralized_obs = torch.zeros((args.num_steps, args.num_envs) + (n_attackers+1,5)).to(device)
    actions = torch.zeros((n_attackers, args.num_steps, args.num_envs) + ()).to(device)
    logprobs = torch.zeros((n_attackers, args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((n_attackers, args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((n_attackers, args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((n_attackers, args.num_steps, args.num_envs)).to(device)
    saved_logits = torch.zeros((n_attackers, args.num_steps, args.num_envs) + (5, )).to(device)

    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(n_attackers).to(device)
    num_updates = args.total_timesteps // args.batch_size
    '''
    collect samples
    '''
    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            for a in range(n_attackers):
                critic_optimizers[a].param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            actions_to_take = [0 for _ in range(n_attackers)]
            centralized_obs[step] = next_obs[0]
            for i in range(n_attackers):
                obs[i, step] = next_obs[i]
                dones[i, step] = next_done[i]
                
                with torch.no_grad():
                    action, logprob, _, value, logits_ = agents[i].get_action_and_value(next_obs[i], centralized_obs[step])
                    values[i, step] = value.flatten()
                    saved_logits[i, step] = logits_
                actions[i, step] = action
                # print("action: {}".format(action))
                logprobs[i, step] = logprob
                actions_to_take[i] = action.item()
            next_obs, reward, done, truncated, info = envs.step(tuple(actions_to_take))
            
            for i in range(n_attackers):
                rewards[i, step] = torch.tensor(reward[i]).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            
            if 'final_info' in info.keys():
                item = info['final_info']
                print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
           

        '''
        estimate advantages
        '''
        with torch.no_grad():
            next_values = []
            for i in range(n_attackers):
                next_values.append(agents[i].get_value(next_obs[0]))
            if args.gae:
                # estimate advantages
                advantages = torch.zeros_like(rewards).to(device)
                for i in range(n_attackers):
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done[i]
                            nextvalues = next_values[i]
                        else:
                            nextnonterminal = 1.0 - dones[i, t + 1]
                            nextvalues = values[i, t + 1]
                        delta = rewards[:, t].mean() + args.gamma * nextvalues * nextnonterminal - values[i, t]
                        advantages[i, t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    '''
                # use a centralized dones to calculate advantages 
                advantages = torch.zeros((args.num_steps, args.num_envs)).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = any(1.0 - next_done)
                        nextvalues = next_value
                    else:
                        nextnonterminal = any(1.0 - dones[:, t + 1])
                        nextvalues = values[t + 1]
                    # print("value of different agents: ", values[:, t])
                    delta = rewards[:, t].mean() + args.gamma * nextvalues * nextnonterminal - values[t]
                    # print("*********")
                    # print(delta)
                    # print(advantages.shape)
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    '''
                returns = advantages + values

        b_obs = obs.squeeze()
        b_centralized_obs = centralized_obs.squeeze()
        b_logprobs = logprobs.squeeze()
        b_actions = actions.squeeze()
        b_advantages = advantages.squeeze()
        b_returns = returns.squeeze()
        b_values = values.squeeze()
        b_logits = saved_logits.squeeze()
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                M = torch.ones_like(b_actions.long()[i, mb_inds]).to(device)
                for i in np.random.permutation(n_attackers):
                    mb_advantages = b_advantages[i, mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    _, old_logprobs_M, *_ = agents[i].get_action_info(b_obs[i, mb_inds], b_actions.long()[i, mb_inds])
                    def get_loss(volatile=False):
                        with torch.set_grad_enabled(not volatile):
                            _, newlogprob, *_ = agents[i].get_action_info(b_obs[i, mb_inds], b_actions.long()[i, mb_inds])
                            action_loss = -mb_advantages * torch.exp(newlogprob - b_logprobs[i, mb_inds])*M
                            return action_loss.mean()
                        
                    def fisher_vector_product(y):
                        kl = agents[i].get_kl(b_obs[i, mb_inds], b_logits[i, mb_inds])
                        grads = torch.autograd.grad(kl, agents[i].actor.parameters(), create_graph=True)
                        flat_grads = torch.cat([g.view(-1) for g in grads])

                        inner_prod = flat_grads.t() @ y  # different results due to numerical precision and
                        # exploiting GPU parallelism by reduction in operations
                        grads = torch.autograd.grad(inner_prod, agents[i].actor.parameters())
                        flat_grads = torch.cat([g.reshape(-1) for g in grads]).data
                        return flat_grads + y * args.damping 

                    loss = get_loss()
                    grads = torch.autograd.grad(loss, agents[i].actor.parameters())
                    loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach() #g  
                    grad_norm = False # TODO: shall we set this to True?
                    if grad_norm == True:
                        loss_grad = loss_grad/torch.norm(loss_grad)
                    stepdir = cg(fisher_vector_product, -loss_grad, 10) #(H^-1)*g   
                    if grad_norm == True:
                        stepdir = stepdir/torch.norm(stepdir)
                    
                    q = (stepdir * fisher_vector_product(stepdir)).sum() #g^T.H^-1.g
                    lam = torch.sqrt((q / (2 * args.max_kl)))
                    x = (1. / (lam + EPS)) * stepdir
                    fraction = args.line_search_fraction
                    # line search
                    prev_params = get_flat_params_from(agents[i].actor)
                    prev_loss = loss.clone().detach()
                    expected_improve = -torch.dot(x, loss_grad)
                    expected_improve = expected_improve.clone().detach()
                    flag = False
                    fraction_coef = args.fraction_coef
                    for l in range(args.ls_step):
                        x_norm = torch.norm(x)
                        if x_norm > 0.5:
                            x = x * 0.5 / x_norm

                        new_params = prev_params + fraction_coef * (fraction**l) * x # TODO: should this be + or -?
                        set_flat_params_to(new_params, agents[i].actor)
                        try:
                            new_loss = get_loss(True)
                        except:
                            print("network exploded!!!!!!!!!!!!!!!")
                            break
                        kl = agents[i].get_kl(b_obs[i, mb_inds], b_logits[i, mb_inds])
                        print("objective improvement: ", new_loss- prev_loss)

                        # see https: // en.wikipedia.org / wiki / Backtracking_line_search
                        if ((kl <= args.max_kl) and (new_loss < prev_loss) and torch.isfinite(new_loss)):
                            flag = True
                            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                            print("line search successful")
                            break
                        expected_improve *= fraction

                    if not flag:
                        # line search failed
                        print("line search failed")
                        set_flat_params_to(prev_params, agents[i].actor)
                    _, new_logprobs_M, *_ = agents[i].get_action_info(b_obs[i, mb_inds], b_actions.long()[i, mb_inds])
                    M = M * torch.exp(new_logprobs_M - old_logprobs_M).detach()

                    for _ in range(args.value_update_epochs):
                        # Value loss
                        newvalue = agents[i].get_value(b_centralized_obs[mb_inds]).view(-1)
                        if args.clip_vloss:
                            v_loss_unclipped = (newvalue - b_returns[i, mb_inds]) ** 2
                            v_clipped = b_values[i, mb_inds] + torch.clamp(
                                newvalue - b_values[i, mb_inds],
                                -args.clip_coef,
                                args.clip_coef,
                            )
                            v_loss_clipped = (v_clipped - b_returns[i, mb_inds]) ** 2
                            v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                            v_loss = 0.5 * v_loss_max.mean()
                        else:
                            v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                        critic_optimizers[i].zero_grad()
                        (v_loss * args.vf_coef).backward()
                        nn.utils.clip_grad_norm_(agents[i].critic.parameters(), args.max_grad_norm)
                        critic_optimizers[i].step()
                        
        writer.add_scalar("charts/learning_rate", critic_optimizers[i].param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    for i in range(n_attackers):
        torch.save(agents[i], PATHS[i])
    envs.close()
    writer.close()