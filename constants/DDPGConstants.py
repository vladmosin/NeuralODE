import copy

import torch

from agents.CommonAgent import CommonAgent
from enums.AgentType import AgentType
from utils.ActionSelector import NoisedSelector
from utils.MemoryReplay import MemoryReplay
from utils.RewardConverter import MountainCarConverter


class DDPGConstants:
    def __init__(self, device, env):
        self.device = device
        self.env = env
        self.gamma = 0.999
        self.tau = 0.05
        self.agent_type = AgentType.Common

        self.num_episodes = 300
        self.start_eps = 0.9
        self.end_eps = 0.005
        self.eps_decay = 1000
        self.neuron_number = 128
        self.memory_size = 20000
        self.critic_lr = 1e-4
        self.actor_lr = 1e-4
        self.batch_size = 64

    def get_models(self):
        state_space_dim = self.env.observation_space.shape[0]
        action_space_dim = self.env.action_space.shape[0]

        actor = CommonAgent(
            device=self.device,
            neuron_number=self.neuron_number,
            input_dim=state_space_dim,
            output_dim=action_space_dim
        ).double()

        critic = CommonAgent(
            device=self.device,
            neuron_number=self.neuron_number,
            input_dim=state_space_dim + action_space_dim,
            output_dim=1
        ).double()

        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.actor_lr)

        target_actor = copy.deepcopy(actor)
        target_critic = copy.deepcopy(critic)

        full_critic = critic
        full_critic.target = target_critic

        full_actor = actor
        full_actor.target = target_actor

        return full_actor, full_critic, actor_optimizer, critic_optimizer

    def get_action_selector(self):
        return NoisedSelector(action_space=self.env.action_space,
                              device=self.device,
                              end_eps=self.end_eps,
                              start_eps=self.start_eps,
                              eps_decay=self.eps_decay)

    def get_reward_converter(self):
        return MountainCarConverter()

    def get_memory(self):
        return MemoryReplay(device=self.device, capacity=self.memory_size)
