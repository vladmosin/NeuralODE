import numpy as np
from torch import tensor
import torch

class Transition:
    def __init__(self, device, state, action, next_state, reward, done):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = torch.tensor([[reward]], device=device, dtype=torch.float64)
        self.not_done = torch.tensor([[not done]], device=device)

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

        return [torch.cat([trans.get(i) for trans in batch]) for i in range(5)]

    def __len__(self):
        return len(self.memory)
