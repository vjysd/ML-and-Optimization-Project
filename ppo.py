# Machine Learning and Optimization (Fall 2021)
# Group G8
# Proximal Policy Optimization (PPO): https://arxiv.org/abs/1707.06347

# Import required libraries
import os
import sys
import gym
import random
import warnings
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from tensorboardX import SummaryWriter
from collections import namedtuple, deque

# Config parameters for environment and training. Here we have set it up for OpenAI's CartPole-v1 environemnet.
env_name = 'CartPole-v1'
gamma = 0.99
lr = 0.001
goal_score = 200
log_interval = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lambda_gae = 0.96
epsilon_clip = 0.2
ciritic_coefficient = 0.5
entropy_coefficient = 0.01
batch_size = 8
epoch_k = 10

# Proximal Policy Optimization class
class PPO(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(PPO, self).__init__()
        self.t = 0
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc = nn.Linear(num_inputs, 128)
        self.fc_actor = nn.Linear(128, num_outputs)
        self.fc_critic = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, input):
        x = torch.relu(self.fc(input))
        policy = F.softmax(self.fc_actor(x), dim=-1)
        value = self.fc_critic(x)
        return policy, value

    @classmethod
    def get_gae(self, values, rewards, masks):
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)

        running_return = 0
        previous_value = 0
        running_advantage = 0

        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return * masks[t]
            running_tderror = rewards[t] + gamma * previous_value * masks[t] - values.data[t]
            running_advantage = running_tderror + (gamma * lambda_gae) * running_advantage * masks[t]

            returns[t] = running_return
            previous_value = values.data[t]
            advantages[t] = running_advantage

        return returns, advantages

    @classmethod
    def train_model(cls, net, transitions, optimizer):
        states, actions, rewards, masks = transitions.state, transitions.action, transitions.reward, transitions.mask

        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.Tensor(rewards)
        masks = torch.Tensor(masks)
        
        old_policies, old_values = net(states)
        old_policies = old_policies.view(-1, net.num_outputs).detach()
        returns, advantages = net.get_gae(old_values.view(-1).detach(), rewards, masks)

        batch_maker = BatchMaker(states, actions, returns, advantages, old_policies)
        for _ in range(epoch_k):
            for _ in range(len(states) // batch_size):
                states_sample, actions_sample, returns_sample, advantages_sample, old_policies_sample = batch_maker.sample()
                
                policies, values = net(states_sample)
                values = values.view(-1)
                policies = policies.view(-1, net.num_outputs)

                ratios = ((policies / old_policies_sample) * actions_sample.detach()).sum(dim=1)
                
                
                clipped_ratios = torch.clamp(ratios, min=1.0-epsilon_clip, max=1.0+epsilon_clip)

                actor_loss = -torch.min(ratios * advantages_sample,
                                        clipped_ratios * advantages_sample).sum()

                critic_loss = (returns_sample.detach() - values).pow(2).sum()

                policy_entropy = (torch.log(policies) * policies).sum(1, keepdim=True).mean()

                loss = actor_loss + ciritic_coefficient * critic_loss - entropy_coefficient * policy_entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        return loss

    def get_action(self, input):
        policy, _ = self.forward(input)
        
        policy = policy[0].data.numpy()
        action = np.random.choice(self.num_outputs, 1, p=policy)[0]
        
        return action

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

class BatchMaker():
    def __init__(self, states, actions, returns, advantages, old_policies):
        self.states = states
        self.actions = actions
        self.returns = returns
        self.advantages = advantages
        self.old_policies = old_policies
    
    def sample(self):
        sample_indexes = random.sample(range(len(self.states)), batch_size)
        states_sample = self.states[sample_indexes]
        actions_sample = self.actions[sample_indexes]
        retruns_sample = self.returns[sample_indexes]
        advantages_sample = self.advantages[sample_indexes]
        old_policies_sample = self.old_policies[sample_indexes]
        
        return states_sample, actions_sample, retruns_sample, advantages_sample, old_policies_sample


def main():
    env = gym.make(env_name)
    #env.render()
    env.seed(500)
    torch.manual_seed(500)

    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    net = PPO(num_inputs, num_actions)

    optimizer = optim.Adam(net.parameters(), lr=lr)
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

        loss = PPO.train_model(net, memory.sample(), optimizer)

        score = score if score == 500.0 else score + 1
        if running_score == 0:
            running_score = score
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
