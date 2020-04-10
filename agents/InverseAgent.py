from torchdiffeq import odeint

from constants.InverseAgentConfig import InverseAgentConfig, ValueBlockConfig, StateTransformationBlockConfig
from torch import nn
import torch
import torch.nn.functional as F


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

    def forward(self, x):
        state = x[:, :self.state_dim]
        action = x[:, self.state_dim:]

        for block in self.inverse_blocks:
            action = block(action=action, state=state)

        return action

    def inverse(self, action, state):
        for block in reversed(self.inverse_blocks):
            action = block.inverse(action=action, state=state)

        return action


class InverseNet(nn.Module):
    def __init__(self, inverse_blocks, state_dim, device):
        super(InverseNet, self).__init__()
        self.gradient_net = GradientInverseNet(inverse_blocks, state_dim).to(device=device)
        self.times = torch.tensor([0, 1], dtype=torch.double, device=device)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return odeint(self.gradient_net, x, self.times)[1]

    def inverse_grad(self, state, action):
        return self.gradient_net.inverse(action=action, state=state)


class InverseAgent:
    def __init__(self, config: InverseAgentConfig):
        inverse_blocks = config.create_inverse_blocks()

        self.inverse_net = InverseNet(inverse_blocks, config.inverse_block_config.state_dim, device=config.device)
        self.value_block = ValueBlock(config.value_block_config).to(device=config.device)
        self.state_transformation_block = StateTransformationBlock(
            config.state_transformation_config).to(device=config.device)
        self.last_layer = nn.Linear(in_features=config.inverse_block_config.action_dim, out_features=1)

    def forward(self, state, action):
        state = self.state_transformation_block(state)
        advantage = self.inverse_net(state=state, action=action)
        advantage = self.last_layer(F.relu(advantage))
        value = self.value_block(state)

        return value, advantage
