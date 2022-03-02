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
    Reinforcement Learning with Deep Energy-Based Policies https://arxiv.org/abs/1702.08165
    Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor 
    https://arxiv.org/pdf/1801.01290.pdf
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
env_name = "Pendulum-v1"
tau = 0.01
gradient_steps = 1
actor_lr = 3e-4
V_net_lr = 3e-4
Q_net_lr = 3e-4
discount = 0.99
alpha = 0.2
buffer_size = 10000
batch_size = 128
max_episode = 40000
max_step_size = 300
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
        self.mu_head.weight.data.uniform_(-init_w, init_w)
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

    def evaluate(self, state):
        mu, std = self.forward(state)
        dist = Normal(0, 1)
        z = dist.sample(mu.shape).to(device)
        # reparameterization trick
        action = torch.tanh(mu + std*z)
        ## origin implementation
        # log_prob = dist.log_prob(z) - torch.log(1 - sample_action.pow(2) + 1e-7)
        # log_prob = log_prob.sum(dim=1, keepdim=True)
        ## optimized implementation
        # see https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/core.py
        log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)
        log_prob -= (2*(np.log(2) - action - F.softplus(-2* action))).sum(axis=1, keepdim=True)
        return action, log_prob, z, mu, std


class V_net(nn.Module):

    def __init__(self, state_dim, init_w=3e-3):
        super(V_net, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)

        self.l3 = nn.Linear(256, 1)
        self.l3.weight.data.uniform_(-init_w, init_w)
        self.l3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


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

        self.V_net = V_net(state_dim).to(device)
        self.V_net_target = copy.deepcopy(self.V_net)
        self.V_net_optimizer = optim.Adam(self.V_net.parameters(), lr=V_net_lr)

        self.Q_net = Q_net(state_dim, action_dim).to(device)
        self.Q_net_optimizer = optim.Adam(self.Q_net.parameters(), lr=Q_net_lr)

        self.buffer = ReplayBuffer(buffer_size, batch_size)

        self.num_training = 1

        self.writer = SummaryWriter('./log')

        os.makedirs('./model/', exist_ok=True)

    def act(self, state):
        state = torch.FloatTensor(np.array(state).reshape(1, -1)).to(device)
        with torch.no_grad():
            mu, std = self.actor(state)
            dist = Normal(0, 1)
            z = dist.sample(mu.shape).to(device)
            action = torch.tanh(mu + std*z)
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

        state = torch.tensor(state, dtype=torch.float).to(device)
        action = torch.tensor(action, dtype=torch.float).to(device)
        reward = torch.tensor(reward, dtype=torch.float).view(batch_size, 1).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.float).to(device).view(batch_size, -1).to(device)

        target_V = self.V_net_target(next_state)
        target_Q = reward + (1 - done) * discount * target_V

        current_V = self.V_net(state)
        current_Q1, current_Q2 = self.Q_net(state, action)

        sample_action, log_prob, _, _, _ = self.actor.evaluate(state)

        new_Q1, new_Q2 = self.Q_net(state, sample_action)
        excepted_new_Q = torch.min(new_Q1, new_Q2)
        next_value = excepted_new_Q - alpha * log_prob

        V_loss = F.mse_loss(current_V, next_value.detach())  # J_V
        V_loss = V_loss.mean()

        Q_loss = F.mse_loss(current_Q1, target_Q.detach()) + F.mse_loss(current_Q2, target_Q.detach()) # J_Q
        Q_loss = Q_loss.mean()

        log_policy_target = excepted_new_Q - current_V

        pi_loss = alpha * log_prob * (alpha * log_prob - log_policy_target).detach()
        # pi_loss = alpha * log_prob - excepted_new_Q
        pi_loss = pi_loss.mean()

        self.writer.add_scalar('Loss/V_loss', V_loss, global_step=self.num_training)
        self.writer.add_scalar('Loss/Q_loss', Q_loss, global_step=self.num_training)
        self.writer.add_scalar('Loss/pi_loss', pi_loss, global_step=self.num_training)

        # mini batch gradient descent
        self.V_net_optimizer.zero_grad()
        V_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.V_net.parameters(), 0.5)
        self.V_net_optimizer.step()

        self.Q_net_optimizer.zero_grad()
        Q_loss.backward(retain_graph = True)
        nn.utils.clip_grad_norm_(self.Q_net.parameters(), 0.5)
        self.Q_net_optimizer.step()

        self.actor_optimizer.zero_grad()
        pi_loss.backward(retain_graph = True)
        nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        # soft update
        for target_param, param in zip(self.V_net_target.parameters(), self.V_net.parameters()):
            target_param.data.copy_(target_param * (1 - tau) + param * tau)

        self.num_training += 1

    def save(self):
        torch.save(self.actor.state_dict(), './model/actor.pth')
        torch.save(self.V_net.state_dict(), './model/V_net.pth')
        torch.save(self.Q_net.state_dict(), './model/Q_net.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        torch.load(self.actor.state_dict(), './model/actor.pth')
        torch.load(self.V_net.state_dict(), './model/V_net.pth')
        torch.load(self.Q_net.state_dict(), './model/Q_net.pth')
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
