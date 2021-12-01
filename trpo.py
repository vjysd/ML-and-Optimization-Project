# Machine Learning and Optimization (Fall 2021)
# Group G8
# Trusted Region Policy Optimization (TRPO): https://arxiv.org/abs/1502.05477

# Import required libraries
import os
import sys
import gym
import random
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from collections import namedtuple, deque

# Config parameters for environment and training. Here we have set it up for OpenAI's CartPole-v1 environemnet.
env_name = 'CartPole-v1'
gamma = 0.99
goal_score = 200
log_interval = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_kl = 0.01

# Trsusted Region Policy Optimization class
class TRPO(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(TRPO, self).__init__()
        self.t = 0
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc_1 = nn.Linear(num_inputs, 128)
        self.fc_2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = torch.relu(self.fc_1(input))
        policy = F.softmax(self.fc_2(x))

        return policy

    @classmethod
    def train_model(cls, net, transitions):
        states, actions, rewards, masks = transitions.state, transitions.action, transitions.reward, transitions.mask

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)

        returns = torch.zeros_like(rewards)

        running_return = 0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            returns[t] = running_return

        policy = net(states)
        policy = policy.view(-1, net.num_outputs)
        policy_action = (policy * actions.detach()).sum(dim=1)

        old_policy = net(states).detach()
        old_policy = old_policy.view(-1, net.num_outputs)
        old_policy_action = (old_policy * actions.detach()).sum(dim=1)

        surrogate_loss = ((policy_action / old_policy_action) * returns).mean()

        surrogate_loss_grad = torch.autograd.grad(surrogate_loss, net.parameters())
        surrogate_loss_grad = flat_grad(surrogate_loss_grad)

        step_dir = conjugate_gradient(net, states, surrogate_loss_grad.data)

        params = flat_params(net)
        shs = (step_dir * fisher_vector_product(net, states, step_dir)).sum(0, keepdim=True)
        step_size = torch.sqrt((2 * max_kl) / shs)[0]
        full_step = step_size * step_dir

        fraction = 1.0
        for _ in range(10):
            new_params = params + fraction * full_step
            update_model(net, new_params)
            policy = net(states)
            policy = policy.view(-1, net.num_outputs)
            policy_action = (policy * actions.detach()).sum(dim=1)
            surrogate_loss = ((policy_action / old_policy_action) * returns).mean()

            kl = kl_divergence(policy, old_policy)
            kl = kl.mean()

            if kl < max_kl:
                break
            fraction = fraction * 0.5

        return -surrogate_loss

    def get_action(self, input):
        policy = self.forward(input)
        policy = policy[0].data.numpy()
            
        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        return action

def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten

def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten

def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten

def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length

def kl_divergence(policy, old_policy):
    kl = old_policy * torch.log(old_policy / policy)

    kl = kl.sum(1, keepdim=True)
    return kl

def fisher_vector_product(net, states, p, cg_damp=0.1):
    policy = net(states)
    old_policy = net(states).detach()
    kl = kl_divergence(policy, old_policy)
    kl = kl.mean()
    kl_grad = torch.autograd.grad(kl, net.parameters(), create_graph=True) # create_graph is True if we need higher order derivative products
    kl_grad = flat_grad(kl_grad)

    kl_grad_p = (kl_grad * p.detach()).sum()
    kl_hessian_p = torch.autograd.grad(kl_grad_p, net.parameters())
    kl_hessian_p = flat_hessian(kl_hessian_p)

    return kl_hessian_p + cg_damp * p.detach()


def conjugate_gradient(net, states, loss_grad, n_step=10, residual_tol=1e-10):
    x = torch.zeros(loss_grad.size())
    r = loss_grad.clone()
    p = loss_grad.clone()
    r_dot_r = torch.dot(r, r)

    for i in range(n_step):
        A_dot_p = fisher_vector_product(net, states, p)
        alpha = r_dot_r / torch.dot(p, A_dot_p)
        x += alpha * p
        r -= alpha * A_dot_p
        new_r_dot_r = torch.dot(r,r)
        betta = new_r_dot_r / r_dot_r
        p = r + betta * p
        r_dot_r = new_r_dot_r
        if r_dot_r < residual_tol:
            break
    return x

# Transition tuple that keeps track of state, next_state, action and reward
Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))

class Memory(object):
    def __init__(self):
        self.memory = deque()

    def push(self, state, next_state, action, reward, mask):
        self.memory.append(Transition(state, next_state, action, reward, mask))

    def sample(self):
        memory = self.memory
        return Transition(*zip(*memory)) 

    def __len__(self):
        return len(self.memory)

def main():
    env = gym.make(env_name)
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    net = TRPO(num_inputs, num_actions)
    writer = SummaryWriter('logs')

    net.to(device)
    net.train()
    running_score = 0
    steps = 0
    loss = 0
    for e in range(30000):
        done = False
        memory = Memory()

        score = 0
        state = env.reset()
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        while not done:
            steps += 1

            action = net.get_action(state)
            next_state, reward, done, _ = env.step(action)

            next_state = torch.Tensor(next_state)
            next_state = next_state.unsqueeze(0)

            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1

            action_one_hot = torch.zeros(2)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, mask)

            score += reward
            state = next_state

        loss = TRPO.train_model(net, memory.sample())

        score = score if score == 500.0 else score + 1
        running_score = 0.99 * running_score + 0.01 * score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f}'.format(
                e, running_score))
            writer.add_scalar('log/score', float(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

        if running_score > goal_score:
            break


if __name__=="__main__":
    main()
