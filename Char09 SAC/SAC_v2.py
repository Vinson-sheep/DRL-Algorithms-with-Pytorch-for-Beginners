import os
import numpy as np
import copy
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from buffer import ReplayBuffer

'''
Soft actor critic with automated temperature
Original paper:
    Soft Actor-Critic Algorithms and Applications https://arxiv.org/pdf/1812.05905.pdf
    Learning to Walk via Deep Reinforcement Learning https://arxiv.org/pdf/1812.11103.pdf
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
env_name = "BipedalWalker-v3"
tau = 0.01
actor_lr = 3e-4
Q_net_lr = 3e-4
alpha_lr = 3e-4
discount = 0.99
init_temperature = 0.1
buffer_size = 20000
batch_size = 1024
actor_update_frequency = 1
Q_net_target_update_frequency = 1
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

        self.mu_head = nn.Linear(256, action_dim)
        self.mu_head.weight.data.uniform_(init_w, init_w)
        self.mu_head.bias.data.uniform_(-init_w, init_w)

        self.log_std_head = nn.Linear(256, action_dim)
        self.log_std_head.weight.data.uniform_(-init_w, init_w)
        self.log_std_head.bias.data.uniform_(-init_w, init_w)

        self.min_log_std = -20
        self.max_log_std = 2

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mu = self.mu_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = log_std.exp()
        return mu, std

    def evaluate(self, state, epsilon=1e-6):
        mu, std = self.forward(state)
        dist = Normal(0, 1)
        z = dist.sample(mu.shape).to(device)
        # reparameterization trick
        action = torch.tanh(mu + std*z)
        log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)
        log_prob -= (2*(np.log(2) - action - F.softplus(-2* action))).sum(axis=1, keepdim=True)
        return action, log_prob, z, mu, std

class Q_net(nn.Module):

    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(Q_net, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
        self.l6.weight.data.uniform_(-init_w, init_w)
        self.l6.bias.data.uniform_(-init_w, init_w)

    def forward(self, s, a):
        sa = torch.cat((s, a), 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)

        return q1, q2

    def Q1(self, s, a):
        sa = torch.cat((s, a), 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        return q1


class SAC:

    def __init__(self):
        super(SAC, self).__init__()

        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.Q_net = Q_net(state_dim, action_dim).to(device)
        self.Q_net_target = copy.deepcopy(self.Q_net)
        self.Q_net_optimizer = optim.Adam(self.Q_net.parameters(), lr=Q_net_lr)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)

        # set target entropy to -|A|
        self.target_entropy = -action_dim

        self.buffer = ReplayBuffer(buffer_size, batch_size)

        self.num_training = 1

        self.writer = SummaryWriter('./log')

        os.makedirs('./model/', exist_ok=True)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, state):
        state = torch.FloatTensor(np.array(state).reshape(1, -1)).to(device)
        with torch.no_grad():
            mu, std = self.actor(state)
            dist = Normal(mu, std)
            z = dist.sample()
            action = torch.tanh(z)
        return action.flatten().cpu().numpy()

    def put(self, *transition):
        state, action, reward, next_state, done = transition
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action = torch.FloatTensor(action).to(device).unsqueeze(0)
        Q = self.Q_net.Q1(state, action).detach()
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
        reward = torch.tensor(reward, dtype=torch.float).view(batch_size, -1).to(device) /10
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.float).to(device).view(batch_size, -1).to(device)

        reward_scale = 10

        # alpha

        current_Q1, current_Q2 = self.Q_net(state, action)
        new_action, log_prob, _, _, _ = self.actor.evaluate(state)
        new_next_action, next_log_prob, _, _, _ = self.actor.evaluate(next_state)
        reward = reward_scale * (reward - reward.mean(dim=0)) / (reward.std(dim=0) + 1e-6) # normalize with batch mean and std; plus a small number to prevent numerical problem

        if self.num_training % actor_update_frequency == 0:

            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.writer.add_scalar('Loss/alpha_loss', alpha_loss, global_step=self.num_training)
            self.writer.add_scalar('Loss/alpha', self.alpha, global_step=self.num_training)

            self.log_alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

        # Q_net

        target_Q1, target_Q2 = self.Q_net_target(next_state, new_next_action)
        target_Q = reward + (1 - done) * discount * torch.min(target_Q1, target_Q2)
        Q_net_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.writer.add_scalar('Loss/Q_loss', Q_net_loss, global_step=self.num_training)

        self.Q_net_optimizer.zero_grad()
        Q_net_loss.backward()
        self.Q_net_optimizer.step()

        # actor
        if self.num_training % actor_update_frequency == 0:
            new_Q1, new_Q2 = self.Q_net(state, new_action)
            new_Q = torch.min(new_Q1, new_Q2)
            actor_loss = (self.alpha * log_prob - new_Q).mean()
            self.writer.add_scalar('Loss/pi_loss', actor_loss, global_step=self.num_training)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # soft update
        if self.num_training % Q_net_target_update_frequency == 0:
            for target_param, param in zip(self.Q_net_target.parameters(), self.Q_net.parameters()):
                target_param.data.copy_(target_param * (1 - tau) + param * tau)

        self.num_training += 1

    def save(self):
        torch.save(self.actor.state_dict(), './model/actor.pth')
        torch.save(self.Q_net.state_dict(), './model/Q_net.pth')
        torch.save(self.log_alpha, './model/log_alpha.pth')

        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        torch.load(self.actor.state_dict(), './model/actor.pth')
        torch.load(self.Q_net.state_dict(), './model/Q_net.pth')
        torch.load(self.log_alpha, './model/log_alpha.pth')
        print("====================================")
        print("Model has been loaded...")
        print("====================================")

if __name__ == '__main__':
    agent = SAC()
    state = env.reset()

    if load:
        agent.load()
    if render:
        env.render()

    print("====================================")
    print("Collection Experience...")
    print("====================================")

    for episode in range(max_episode):

        total_reward = 0
        state = env.reset()

        for step in range(max_step_size):

            action = agent.act(state)
            next_state, reward, done, _ = env.step(envAction(action))

            # reward trick of BipedalWalker-v3
            if reward == -100:
                reward = -1

            if render:
                env.render()

            agent.put(state, action, reward, next_state, done)

            agent.update()

            total_reward += reward

            state = next_state

            if done:
                break

        if episode % 10 == 0:
            agent.save()

        agent.writer.add_scalar('total_reward', total_reward, global_step=episode)
