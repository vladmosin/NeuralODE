import torch

from agents.CommonAgent import CommonAgent
from enums.AgentType import AgentType
from utils.ActionSelector import ActionSelector
from utils.MemoryReplay import MemoryReplay
from utils.RewardConverter import CartPoleConverter


class DQNConstants:
    def __init__(self, device, env):
        self.num_episodes = 1000
        self.test_episodes = 10
        self.batch_size = 64
        self.target_update = 5
        self.gamma = 0.999
        self.agent_type = AgentType.Common
        self.device = device
        self.env = env

        self.neuron_number = 64
        self.start_eps = 0.9
        self.end_eps = 0.05
        self.eps_decay = 1000
        self.memory_size = 20000
        self.lr = 1e-3

    # Should add if expression on AgentType
    def get_models(self):
        state_space_dim = self.env.observation_space.shape[0]
        action_space_dim = self.env.action_space.n

        policy_net = CommonAgent(
            device=self.device,
            neuron_number=self.neuron_number,
            input_dim=state_space_dim,
            output_dim=action_space_dim
        )

        target_net = CommonAgent(
            device=self.device,
            neuron_number=self.neuron_number,
            input_dim=state_space_dim,
            output_dim=action_space_dim
        )

        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.lr)

        return policy_net, target_net, optimizer

    def get_memory(self):
        return MemoryReplay(device=self.device, capacity=self.memory_size)

    def get_action_selector(self):
        return ActionSelector(
            device=self.device,
            action_space=self.env.action_space,
            end_eps=self.end_eps,
            start_eps=self.start_eps,
            eps_decay=self.eps_decay
        )

    # Should add if on different environments
    def get_reward_converter(self):
        return CartPoleConverter()