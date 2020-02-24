import torch

from agents.CommonAgent import CommonAgent


# ------------ Constants -----------#
from utils.ActionSelector import NoisedSelector
from utils.MemoryReplay import MemoryReplay
from utils.RewardConverter import MountainCarConverter

START_EPS = 0.9
END_EPS = 0.005
EPS_DECAY = 2000

MEMORY_SIZE = 10000
LR = 0.0003
BATCH_SIZE = 128
GAMMA = 0.999
NEURON_NUMBER = 64

CRITIC_LR = 1e-5
ACTOR_LR = 1e-5
TAU = 0.05
#####################################


class DDPGConstants:
    def __init__(self, device, env):
        self.device = device
        self.env = env
        self.gamma = GAMMA
        self.tau = TAU

    def get_models(self):
        state_space_dim = self.env.observation_space.shape[0]
        action_space_dim = self.env.action_space.shape[0]

        actor = CommonAgent(
            device=self.device,
            neuron_number=NEURON_NUMBER,
            input_dim=state_space_dim,
            output_dim=action_space_dim
        )

        critic = CommonAgent(
            device=self.device,
            neuron_number=NEURON_NUMBER,
            input_dim=state_space_dim + action_space_dim,
            output_dim=1
        )

        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LR)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=CRITIC_LR)

        return actor, critic, actor_optimizer, critic_optimizer

    def get_action_selector(self):
        return NoisedSelector(action_space=self.env.action_space,
                              device=self.device,
                              end_eps=END_EPS,
                              start_eps=START_EPS,
                              eps_decay=EPS_DECAY)

    def get_reward_converter(self):
        return MountainCarConverter()

    def get_memory(self):
        return MemoryReplay(device=self.device, capacity=MEMORY_SIZE)
