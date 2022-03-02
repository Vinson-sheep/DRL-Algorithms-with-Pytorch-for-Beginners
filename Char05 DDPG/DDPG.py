import os
import numpy as np
import copy
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from buffer import ReplayBuffer

'''
Deep Deterministic Policy Gradients (DDPG)
Original paper:
    CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING https://arxiv.org/abs/1509.02971
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
env_name = "Pendulum-v1"
tau = 0.01
epsilon = 0.8
epsilon_decay = 0.9999
actor_lr = 3e-4
critic_lr = 3e-4
discount = 0.99
buffer_size = 10000
batch_size = 128
max_episode = 40000
max_step_size = 500
seed = 1

render = True
load = False

env = gym.make(env_name)

def envAction(action):

    low = env.action_space.low
    high = env.action_space.high
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)

    return action

# Set seeds
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.l3.weight.data.uniform_(init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a))

        return a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        sa = torch.cat((state, action), 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)

        return q


class DDPG:

    def __init__(self):
        super(DDPG, self).__init__()

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.buffer = ReplayBuffer(buffer_size, batch_size)

        self.num_training = 1

        self.writer = SummaryWriter('./log')

        os.makedirs('./model/', exist_ok=True)

    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def put(self, *transition):
        state, action, reward, next_state, done = transition
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action = torch.FloatTensor(action).to(device).unsqueeze(0)
        Q = self.critic(state, action).detach()
        self.buffer.add(transition)

        return Q.cpu().item()

    def update(self):

        if not self.buffer.sample_available():
            return

        state, action, reward, next_state, done = self.buffer.sample()

        # state = (state - self.buffer.state_mean())/(self.buffer.state_std() + 1e-7)
        # next_state = (next_state - self.buffer.state_mean())/(self.buffer.state_std() + 1e-6)
        # reward = reward / (self.buffer.reward_std() + 1e-6)

        state = torch.tensor(state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.float).to(device)
        reward = torch.tensor(reward, dtype=torch.float).view(batch_size, -1).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.float).to(device).view(batch_size, -1).to(device)

        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * discount * target_Q

        # Get current Q estimates
        current_Q = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_training)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor losse
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_training)

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.num_training += 1

    def save(self):
        torch.save(self.actor.state_dict(), './model/actor.pth')
        torch.save(self.critic.state_dict(), './model/critic.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        torch.load(self.actor.state_dict(), './model/actor.pth')
        torch.load(self.critic.state_dict(), './model/critic.pth')
        print("====================================")
        print("Model has been loaded...")
        print("====================================")


if __name__ == '__main__':
    agent = DDPG()
    state = env.reset()

    if load:
        agent.load()
    if render:
        env.render()

    print("====================================")
    print("Collection Experience...")
    print("====================================")

    total_step = 0

    for episode in range(max_episode):

        total_reward = 0
        state = env.reset()

        for step in range(max_step_size):

            total_step += 1

            action = agent.act(state)

            if epsilon > np.random.random():
                action = (action + np.random.normal(0, 0.2, size=action_dim)).clip(-1, 1)

            next_state, reward, done, _ = env.step(envAction(action))

            # reward trick of BipedalWalker-v3
            # if reward == -100:
            #     reward = -1

            if render:
                env.render()

            agent.put(state, action, reward, next_state, done)

            agent.update()

            total_reward += reward

            state = next_state

            epsilon = max(epsilon_decay*epsilon, 0.10)
            agent.writer.add_scalar('Other/epsilon', epsilon, global_step=total_step)

            if done:
                break

        if episode % 10 == 0:
            agent.save()

        agent.writer.add_scalar('Other/total_reward', total_reward, global_step=episode)
