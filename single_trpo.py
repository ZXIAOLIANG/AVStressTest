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
                        "type": "Kinematics",
                        "see_behind": True,
                        "vehicles_count": n_attackers+1
                    },
                   "attacker_num": n_attackers, 
                   "controlled_vehicles": n_attackers,
                   "time_penalty": 0.0,	
                   "close_vehicle_cost": 20,
                   "randomize_starting_position": False,
                   "constraint_env": True,
                   "vis": False,
                   "victim_index": 2,
                   "victim_lane_id": 1,
                   "testing": False})
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
    args = parse_args()
    PATH = "trpo\\trpo_penalty_back3_cost20_seed_" + str(args.seed) + "_" + str(n_attackers) + ".pt"
    env_name = "trpo_penalty_back3_cost20_seed_" + str(args.seed) + "_" + str(n_attackers)
    run_name = f"{env_name}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/trpo/{run_name}")
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
    agent = AttackerAgent(envs, n_attackers, 128).to(device)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs) + (n_attackers+1,5)).to(device)
    
    actions = torch.zeros((args.num_steps, args.num_envs) + (envs.action_space.shape)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)  
    saved_logits = torch.zeros((args.num_steps, args.num_envs) + (envs.action_space.nvec.sum(), )).to(device)
    
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    '''
    collect samples
    '''
    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            critic_optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs

            obs[step] = next_obs
            dones[step] = next_done
            
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, logits_ = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
                saved_logits[step] = logits_
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, done, truncated, info = envs.step(tuple(action.cpu().numpy()))
            
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([done]).to(device)

            if 'final_info' in info.keys():
                item = info['final_info']
                print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                writer.add_scalar("charts/episodic_cost", item["episode"]["c"], global_step)
           
        '''
        estimate advantages
        '''
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            if args.gae:
                # estimate advantages
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam    
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        b_obs = obs.reshape((-1,) + (n_attackers+1,5))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_logits = saved_logits.reshape((-1,) + (envs.action_space.nvec.sum(), ))

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                        
                
                def get_loss(volatile=False):
                    with torch.set_grad_enabled(not volatile):
                        _, newlogprob, *_ = agent.get_action_info(b_obs[mb_inds], b_actions.long()[mb_inds].T)
                        action_loss = -mb_advantages * torch.exp(newlogprob - b_logprobs[mb_inds])
                        return action_loss.mean()
                    
                def fisher_vector_product(y):
                    kl = agent.get_kl(b_obs[mb_inds], b_logits[mb_inds])
                    grads = torch.autograd.grad(kl, agent.actor.parameters(), create_graph=True)
                    flat_grads = torch.cat([g.view(-1) for g in grads])

                    inner_prod = flat_grads.t() @ y  # different results due to numerical precision and
                    # exploiting GPU parallelism by reduction in operations
                    grads = torch.autograd.grad(inner_prod, agent.actor.parameters())
                    flat_grads = torch.cat([g.reshape(-1) for g in grads]).data
                    return flat_grads + y * args.damping 

                loss = get_loss()
                grads = torch.autograd.grad(loss, agent.actor.parameters())
                loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach() #g  
                grad_norm = False # TODO: shall we set this to True?
                if grad_norm == True:
                    loss_grad = loss_grad/torch.norm(loss_grad)
                stepdir = cg(fisher_vector_product, -loss_grad, 10) #(H^-1)*g   
                if grad_norm == True:
                    stepdir = stepdir/torch.norm(stepdir)
                    
                # Define q
                q = -loss_grad.dot(stepdir) #g^T.H^-1.g
                print("q: {}".format(q))

                lam = torch.sqrt(
                            (q / (2 * args.max_kl)))

                fraction = args.line_search_fraction
                loss_improve = 0
                x = (1. / (lam + EPS)) * stepdir
                    
                # line search
                prev_params = get_flat_params_from(agent.actor)
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

                    set_flat_params_to(new_params, agent.actor)
                    try:
                        new_loss = get_loss(True)
                    except:
                        print("network exploded!!!!!!!!!!!!!!!")
                        break
                    kl = agent.get_kl(b_obs[mb_inds], b_logits[mb_inds])
                    print("objective improvement: ", new_loss- prev_loss)
                    if ((kl <= args.max_kl) and (new_loss < prev_loss)
                            ):
                        flag = True
                        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                        print("line search successful")
                        break
                    expected_improve *= fraction

                if not flag:
                    # line search failed
                    print("line search failed")
                    set_flat_params_to(prev_params, agent.actor)

                for _ in range(args.value_update_epochs):
                    # Value loss
                    newvalue = agent.get_value(b_obs[mb_inds]).view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                    critic_optimizer.zero_grad()
                    (v_loss * args.vf_coef).backward()
                    nn.utils.clip_grad_norm_(agent.critic.parameters(), args.max_grad_norm)
                    critic_optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", critic_optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", loss.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    torch.save(agent, PATH)
    envs.close()
    writer.close()