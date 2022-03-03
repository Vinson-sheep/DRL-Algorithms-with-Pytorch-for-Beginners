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
from torch.distributions import Categorical

'''
Double Deep Q Network with Dueling Network
Dueling Network Architectures for Deep Reinforcement Learning
Original paper:
    Dueling Network Architectures for Deep Reinforcement Learning https://www.webofscience.com/wos/alldb/full-record/WOS:000684193702009
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parameters
env_name = "CartPole-v1"
tau = 0.01
learning_rate = 1e-3
discount = 0.99
buffer_size = 10000
batch_size = 256
max_episode = 40000
max_step_size = 500
seed = 1

render = True
load = False

env = gym.make(env_name)

# Set seeds
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

class DuelingNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, init_w=3e-3):
        super(DuelingNetwork, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.value_stream = nn.Linear(64, 1)
        self.advantage_stream = nn.Linear(64, action_dim)

        self.value_stream.weight.data.uniform_(-init_w, init_w)
        self.value_stream.bias.data.uniform_(-init_w, init_w)
        self.advantage_stream.weight.data.uniform_(-init_w, init_w)
        self.advantage_stream.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)

        return values + (advantages - advantages.mean())

class DDQN:

    def __init__(self):
        super(DDQN, self).__init__()

        self.model = DuelingNetwork(state_dim, action_dim).to(device)
        self.model_target = copy.deepcopy(self.model)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.buffer = ReplayBuffer(buffer_size, batch_size)

        self.num_training = 1

        self.writer = SummaryWriter('./log')

        os.makedirs('./model/', exist_ok=True)

    def act(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        with torch.no_grad():
            Q_value = self.model(state)
            action_prob = torch.softmax(Q_value, dim=1)
            dist = Categorical(action_prob)
            action = dist.sample()
        return action.cpu().numpy().flatten()[0]

    def put(self, *transition):
        state, action, reward, next_state, done = transition
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action = torch.LongTensor([action]).to(device).unsqueeze(0)
        Q = (self.model(state).gather(1, action))
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
        action = torch.tensor(action, dtype=torch.long).to(device).view(-1, 1)
        reward = torch.tensor(reward, dtype=torch.float).to(device).view(-1, 1)
        next_state = torch.tensor(next_state, dtype=torch.float).to(device)
        done = torch.tensor(done, dtype=torch.float).to(device).view(-1, 1)

        curr_Q = self.model(state).gather(1, action)
        next_Q = self.model_target(next_state)
        max_next_Q = torch.max(next_Q, dim=1, keepdim=True)[0]
        expected_Q = reward + (1 - done) * discount * max_next_Q

        loss = F.mse_loss(curr_Q, expected_Q.detach())
        self.writer.add_scalar('loss', loss, global_step=self.num_training)

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

        # Update the frozen target models
        for param, target_param in zip(self.model.parameters(), self.model_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        self.num_training += 1

    def save(self):
        torch.save(self.model.state_dict(), './model/model.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        torch.load(self.model.state_dict(), './model/model.pth')
        print("====================================")
        print("Model has been loaded...")
        print("====================================")


if __name__ == '__main__':
    agent = DDQN()
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

            next_state, reward, done, _ = env.step(action)

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

        agent.writer.add_scalar('Other/total_reward', total_reward, global_step=episode)
