import numpy as np
from torch import tensor
import torch

from utils.Utils import to_numpy


class Transition:
    def __init__(self, device, state, action, next_state, reward, done):
        self.state = to_numpy(state).reshape(-1)
        self.action = to_numpy(action).reshape(-1)
        self.next_state = to_numpy(next_state).reshape(-1)
        self.reward = np.array([reward], dtype=np.float32)
        self.not_done = np.array([not done])

    def get(self, i):
        return [self.state, self.action, self.next_state, self.reward, self.not_done][i]


class MemoryReplay:
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.index = 0
        self.device = device
        self.memory = []

    def push(self, state, action, next_state, reward, done):
        element = Transition(self.device, state, action, next_state, reward, done)

        if len(self.memory) < self.capacity:
            self.memory.append(element)
        else:
            self.memory[self.index] = element

        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch = np.random.choice(self.memory, batch_size)

        return [torch.cat([torch.tensor([trans.get(i)], device=self.device)
                           for trans in batch]) for i in range(5)]

    def __len__(self):
        return len(self.memory)
