import torch

from agents.CommonAgent import CommonAgent
from enums.AgentType import AgentType
from utils.ActionSelector import ActionSelector
from utils.MemoryReplay import MemoryReplay
from utils.RewardConverter import CartPoleConverter


# ------------ Constants -----------#
START_EPS = 0.9
END_EPS = 0.05
EPS_DECAY = 1000

MEMORY_SIZE = 10000
LR = 0.001
BATCH_SIZE = 64
GAMMA = 0.999
NEURON_NUMBER = 64
#####################################


class DQNConstants:
    def __init__(self, device, env):
        self.num_episodes = 1000
        self.test_episodes = 10
        self.batch_size = 128
        self.target_update = 10
        self.gamma = 0.99
        self.agent_type = AgentType.Common
        self.device = device
        self.env = env

    # Should add if expression on AgentType
    def get_models(self):
        state_space_dim = self.env.observation_space.shape[0]
        action_space_dim = self.env.action_space.n

        policy_net = CommonAgent(
            device=self.device,
            neuron_number=NEURON_NUMBER,
            input_dim=state_space_dim,
            output_dim=action_space_dim
        )

        target_net = CommonAgent(
            device=self.device,
            neuron_number=NEURON_NUMBER,
            input_dim=state_space_dim,
            output_dim=action_space_dim
        )

        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)

        return policy_net, target_net, optimizer

    def get_memory(self):
        return MemoryReplay(device=self.device, capacity=MEMORY_SIZE)

    def get_action_selector(self):
        return ActionSelector(
            device=self.device,
            action_space=self.env.action_space,
            end_eps=END_EPS,
            start_eps=START_EPS,
            eps_decay=EPS_DECAY
        )

    # Should add if on different environments
    def get_reward_converter(self):
        return CartPoleConverter()