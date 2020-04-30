from torch import nn
import torch

from constants.ExpressiveInverseConfig import ExpressiveInverseConfig
from invertable_nets import LinearInvertibleModule, LeakyReluIM

from torchdiffeq import odeint


class ExpressiveInverseAgent(nn.Module):
    def __init__(self, config: ExpressiveInverseConfig, device):
        super(ExpressiveInverseAgent, self).__init__()
        self.state_transform = nn.Sequential(
            nn.Linear(config.state_dim, config.transformed_state),
            nn.LayerNorm(config.transformed_state),
            nn.ReLU()
        ).to(device=device).double()

        self.action_dim = config.action_dim
        self.device = device

        self.relu = LeakyReluIM().to(device=device).double()
        self.inverse_net = ExpressiveInverseBlock(action_dim=config.action_dim, device=device,
                                                  state_dim=config.transformed_state, t=config.t).double()
        self.last_layer = nn.Linear(config.action_dim, 1).to(device=device).double()

    def forward(self, state, action):
        state = self.state_transform(state)
        state = self.relu(self.inverse_net(state, action))
        return self.last_layer(state)

    def best_action(self, state):
        state = self.state_transform(state)
        return self.inverse_net.best_action(state)


class ExpressiveInverseBlock(nn.Module):
    def __init__(self, action_dim, state_dim, t, device):
        super(ExpressiveInverseBlock, self).__init__()
        self.times = torch.tensor([0, t], dtype=torch.double, device=device)
        self.device = device
        self.action_dim = action_dim
        self.grad = ExpressiveGradient(action_dim=action_dim, state_dim=state_dim, device=device).double()

    def forward(self, state, action):
        return odeint(self.model, torch.cat([action, state]), self.times, method="euler")[1]

    def best_action(self, state):
        action = torch.zeros((state.shape[0], self.action_dim)).to(device=self.device, dtype=torch.double)
        return self.grad.inverse(state=state, action=action)


class ExpressiveGradient(nn.Module):
    def __init__(self, action_dim, state_dim, device):
        super(ExpressiveGradient, self).__init__()

        self.state_transform = nn.Sequential(
            nn.Linear(state_dim, action_dim),
            nn.LayerNorm(action_dim),
            nn.ReLU()
        ).double().to(device=device)

        self.action_dim = action_dim
        self.relu = LeakyReluIM()

        self.action_layer = LinearInvertibleModule(action_dim).to(device=device).double()

    def forward(self, t, x):
        action = x[:, :self.action_dim]
        state = x[:, self.action_dim:]

        return self.state_transform(state) + self.relu(self.action_layer(action))

    def inverse(self, state, action):
        state = self.state_transform(state)
        return self.action_layer.backward(self.relu.backward(action + state))