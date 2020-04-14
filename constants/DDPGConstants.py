import copy

import torch

from torch import nn

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
        self.critic_lr = 1e-5
        self.actor_lr = 1e-5
        self.batch_size = 64

    def get_models(self):
        state_space_dim = self.env.observation_space.shape[0]
        action_space_dim = self.env.action_space.shape[0]

        actor = DDPGActor(
            obs_size=state_space_dim,
            act_size=action_space_dim
        ).double().to(device=self.device)

        critic = DDPGCritic(
            obs_size=state_space_dim,
            act_size=action_space_dim
        ).double().to(device=self.device)

        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.critic_lr)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.actor_lr)

        target_actor = copy.deepcopy(actor)
        target_critic = copy.deepcopy(critic)

        return actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer

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


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, state, action):
        obs = self.obs_net(state)
        return self.out_net(torch.cat([obs, action], dim=1))
