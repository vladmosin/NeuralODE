import torch
import numpy as np

from agents.BlockODEAgent import BlockODEAgent
from agents.CommonAgent import CommonAgent
from agents.ODEAgent import ODEAgent
from enums.AgentType import AgentType
from utils.ActionSelector import ActionSelector
from utils.MemoryReplay import MemoryReplay
from utils.RewardConverter import CartPoleConverter


class DQNConstants:
    def __init__(self, device, env, batch_size=64, lr=1e-3,
                 neuron_number=64, num_episodes=1000,
                 gamma=0.999, memory_size=20000,
                 target_update=5, eps_decay=1000, t=1.0):
        self.num_episodes = num_episodes
        self.test_episodes = 10
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.agent_type = AgentType.Common
        self.device = device
        self.env = env

        self.neuron_number = neuron_number
        self.start_eps = 0.9
        self.end_eps = 0.05
        self.eps_decay = eps_decay
        self.memory_size = memory_size
        self.lr = lr
        self.t = t

    # Should add if expression on AgentType
    def get_models(self):
        state_space_dim = self.env.observation_space.shape[0]
        action_space_dim = self.env.action_space.n

        if self.agent_type == AgentType.ODE:
            net = ODEAgent
        elif self.agent_type == AgentType.BlockODE:
            net = BlockODEAgent
        else:
            net = CommonAgent

        policy_net = net(
            device=self.device,
            neuron_number=self.neuron_number,
            input_dim=state_space_dim,
            output_dim=action_space_dim,
            t=self.t
        )

        target_net = net(
            device=self.device,
            neuron_number=self.neuron_number,
            input_dim=state_space_dim,
            output_dim=action_space_dim,
            t=self.t
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

    def __str__(self):
        params = self.__dict__
        str_params = ["DQN Config"]

        for param in params:
            if param not in ["device", "env", "agent_type"]:
                str_params.append('{}={}'.format(param, params[param]))

        return "\n".join(str_params)

    def gen_configs_list(self):
        # testing_params = ['batch_size', 'lr', 'neuron_number']
        return [self.gen_config(i) for i in range(3 ** 3)]

    def gen_config(self, code):
        muls = (np.array([code // 9, (code % 9) // 3, code % 3]) + 1) / 2
        batch_size = self.batch_size
        neuron_number = self.neuron_number
        lr = self.lr

        return DQNConstants(device=self.device, env=self.env, batch_size=int(batch_size * muls[0]),
                            neuron_number=int(neuron_number * muls[1]), lr=lr * muls[2])

