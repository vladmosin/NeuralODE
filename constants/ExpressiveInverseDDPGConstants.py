import copy

import torch

from torch import nn

from agents.CommonAgent import CommonAgent
from agents.ExpressiveInverseAgent import ExpressiveInverseAgent
from constants.ExpressiveInverseConfig import ExpressiveInverseConfig
from enums.AgentType import AgentType
from utils.ActionSelector import NoisedSelector, InverseNoisedSelector
from utils.MemoryReplay import MemoryReplay
from utils.RewardConverter import MountainCarConverter


class ExpressiveInverseDDPGConstants:
    def __init__(self, device, env):
        self.device = device
        self.env = env
        self.gamma = 0.999
        self.tau = 0.05
        self.agent_type = AgentType.ExpressiveInverse
        self.num_episodes = 300
        self.start_eps = 0.9
        self.end_eps = 0.005
        self.eps_decay = 1000
        self.transformed_state_dim = 400
        self.memory_size = 20000
        self.critic_lr = 1e-4
        self.batch_size = 256
        self.t = 1.0

        self.expressive_config = ExpressiveInverseConfig(
            env=env, transformed_state=self.transformed_state_dim, t=self.t
        )

    def get_models(self):
        model = ExpressiveInverseAgent(config=self.expressive_config, device=self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.critic_lr)

        target_model = copy.deepcopy(model)

        return model, target_model, optimizer

    def get_action_selector(self):
        return InverseNoisedSelector(action_space=self.env.action_space,
                                     device=self.device,
                                     end_eps=self.end_eps,
                                     start_eps=self.start_eps,
                                     eps_decay=self.eps_decay)

    def get_reward_converter(self):
        return MountainCarConverter()

    def get_memory(self):
        return MemoryReplay(device=self.device, capacity=self.memory_size)