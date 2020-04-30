import math
from random import random, randint
from torch import tensor
import torch

from enums.Exploration import Exploration


class ActionSelector:
    def __init__(self,
                 device,
                 action_space,
                 eps_decay=2000,
                 start_eps=0.9,
                 end_eps=0.05):
        self.action_space = action_space.n
        self.eps_decay = eps_decay
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.steps = 0
        self.device = device

    def select_action(self, model, state, with_eps_greedy=Exploration.ON):
        eps = self.end_eps + (self.start_eps - self.end_eps) * math.exp(-self.steps / self.eps_decay)
        if with_eps_greedy:
            self.steps += 1

        if eps < random() or with_eps_greedy == Exploration.OFF:
            with torch.no_grad():
                return model(state).max(1)[1].view(1, 1)
        else:
            return tensor([[randint(0, self.action_space - 1)]], device=self.device)


class NoisedSelector:
    def __init__(self,
                 device,
                 action_space,
                 start_eps,
                 end_eps,
                 eps_decay):
        self.low = tensor(action_space.low[0], device=device)
        self.high = tensor(action_space.high[0], device=device)
        self.action_dim = action_space.shape[0]
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.eps = start_eps
        self.eps_decay = eps_decay

    def select_action(self, model, state, with_eps_greedy=True):
        action = model(state).view(-1, 1)
        noise = self.eps * (random() - 0.5)

        if with_eps_greedy:
            return (action + noise).clamp(self.low, self.high)
        else:
            return action


class InverseNoisedSelector:
    def __init__(self,
                 device,
                 action_space,
                 start_eps,
                 end_eps,
                 eps_decay):
        self.low = tensor(action_space.low[0], device=device)
        self.high = tensor(action_space.high[0], device=device)
        self.action_dim = action_space.shape[0]
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.eps = start_eps
        self.eps_decay = eps_decay

    def select_action(self, model, state, with_eps_greedy=True):
        action = model.best_action(state.view(1, -1))
        noise = self.eps * (random() - 0.5)

        if with_eps_greedy:
            return (action + noise).clamp(self.low, self.high)
        else:
            return action
