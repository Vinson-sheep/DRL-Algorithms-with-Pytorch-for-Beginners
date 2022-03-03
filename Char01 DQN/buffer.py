from collections import deque
import numpy as np

class ReplayBuffer:

    def __init__(self, buffer_size, batch_size):
        """

        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.state_buffer = deque(maxlen=self.buffer_size)
        self.next_state_buffer = deque(maxlen=self.buffer_size)
        self.reward_buffer = deque(maxlen=self.buffer_size)
        self.action_buffer = deque(maxlen=self.buffer_size)
        self.done_buffer = deque(maxlen=self.buffer_size)

    def add(self, transition):
        (state, action, reward, next_state, done) = transition
        self.state_buffer.append(state)
        self.next_state_buffer.append(next_state)
        self.reward_buffer.append(reward)
        self.action_buffer.append(action)
        self.done_buffer.append(done)

    def sample(self):
        indices = np.random.choice(self.size(), self.batch_size, replace=False)
        state = np.array([(self.state_buffer[i]) for i in indices])
        next_state = np.array([(self.next_state_buffer[i]) for i in indices])
        reward = np.array([(self.reward_buffer[i]) for i in indices])
        action = np.array([(self.action_buffer[i]) for i in indices])
        done = np.array([(self.done_buffer[i]) for i in indices])
        return state, action, reward, next_state, done

    def size(self):
        return len(self.state_buffer)

    def state_mean(self):
        return np.array(self.state_buffer).mean(dim=1)

    def state_std(self):
        return np.array(self.state_buffer).std(dim=1)

    def reward_std(self):
        return np.array(self.reward_buffer).std()

    def sample_available(self):
        return self.size() >= self.batch_size



