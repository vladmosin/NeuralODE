import gym
import numpy as np

from math import e, cos, sin


class ActionSpace:
    def __init__(self):
        self.low = np.array([-5])
        self.high = np.array([5])
        self.shape = np.array([2])


class SpiralEnv(gym.Env):
    def __init__(self):
        self.t = 0
        self.inc = 0.01
        self.steps = 0

        self.fx = lambda t: e ** t * (cos(t) - sin(t))
        self.fy = lambda t: e ** t * (cos(t) + sin(t))

        self.observation_space = np.zeros(2)
        self.action_space = ActionSpace()

    def step(self, action):
        self.steps += 1
        self.t += self.inc

        next_state = np.array([self.fx(self.t), self.fy(self.t)])

        reward = np.mean((action - next_state) ** 2) / 10
        done = self.steps == 100

        return next_state, -reward, done, None

    def reset(self):
        self.t = 0
        self.steps = 0
        return np.array([self.fx(self.t), self.fy(self.t)])

    def render(self, mode='human'):
        pass