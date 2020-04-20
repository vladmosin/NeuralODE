from torchdiffeq import odeint

from constants.InverseAgentConfig import InverseAgentConfig, ValueBlockConfig, StateTransformationBlockConfig
from torch import nn
import torch
import torch.nn.functional as F
from torch.distributions import Normal


class ValueBlock(nn.Module):
    def __init__(self, config: ValueBlockConfig):
        super(ValueBlock, self).__init__()
        self.blocks = config.create()

    def forward(self, state):
        for block in self.blocks:
            state = F.relu(block(state))
        return state


class StateTransformationBlock(nn.Module):
    def __init__(self, config: StateTransformationBlockConfig):
        super(StateTransformationBlock, self).__init__()
        self.blocks = config.create()

    def forward(self, state):
        for block in self.blocks:
            state = F.relu(block(state))
        return state


class GradientInverseNet(nn.Module):
    def __init__(self, inverse_blocks, state_dim):
        super(GradientInverseNet, self).__init__()
        self.inverse_blocks = inverse_blocks
        self.state_dim = state_dim
        self.state = None

    def forward(self, t, action):
        for block in self.inverse_blocks:
            action = block(action=action, state=self.state)
        return action

    def inverse(self, action, state):
        for block in reversed(self.inverse_blocks):
            action = block.inverse(action=action, state=state)

        return action


class InverseNet(nn.Module):
    def __init__(self, inverse_blocks, state_dim, device):
        super(InverseNet, self).__init__()
        self.gradient_net = GradientInverseNet(inverse_blocks, state_dim).to(device=device, dtype=torch.float64)
        self.times = torch.tensor([0, 1], dtype=torch.float64, device=device)

    def forward(self, state, action):
        self.gradient_net.state = state
        return odeint(self.gradient_net, action, self.times)[1]

    def inverse_grad(self, state, action):
        return self.gradient_net.inverse(action=action, state=state)


class InverseAgent(nn.Module):
    def __init__(self, config: InverseAgentConfig, env):
        super(InverseAgent, self).__init__()
        inverse_blocks = config.create_inverse_blocks()

        self.device = config.device
        self.inverse_net = InverseNet(inverse_blocks, config.inverse_block_config.state_dim,
                                      device=config.device).to(dtype=torch.float64)
        self.value_block = ValueBlock(config.value_block_config).to(device=config.device, dtype=torch.float64)
        self.state_transformation_block = StateTransformationBlock(
            config.state_transformation_config).to(device=config.device, dtype=torch.float64)
        self.last_layer = nn.Linear(in_features=config.inverse_block_config.action_dim,
                                    out_features=1).to(device=config.device, dtype=torch.float64)
        self.random = Normal(torch.tensor([0]).double(), torch.tensor([1]).double())
        self.env = env
        self.normal_distribution_dim = config.normal_distribution_dim
        self.action_dim = config.action_dim

    def forward(self, state, action):
        rand = self.random.sample((action.shape[0], self.normal_distribution_dim)).reshape((action.shape[0], self.normal_distribution_dim)).double()
        action = torch.cat([action, rand], dim=1)
        state = self.state_transformation_block(state)
        advantage = self.inverse_net(state=state, action=action)
        advantage = self.last_layer(F.relu(advantage))
        value = self.value_block(state)

        return value + advantage

    def find_best_action(self, state):
        zeroes = torch.zeros((state.shape[0], self.action_dim + self.normal_distribution_dim),
                             device=self.device, dtype=torch.float64)
        low = torch.tensor(self.env.action_space.low[0], dtype=torch.float64)
        high = torch.tensor(self.env.action_space.high[0], dtype=torch.float64)
        with torch.no_grad():
            state = self.state_transformation_block(state)
            return self.inverse_net.inverse_grad(state, zeroes).clamp(low, high)[:, :self.action_dim]
