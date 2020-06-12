from torch import nn
from torchdiffeq import odeint
import torch


class ODEFunction(nn.Module):
    def __init__(self,
                 device,
                 neuron_number=32):
        super(ODEFunction, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(neuron_number, neuron_number)
        )

    def forward(self, t, x: torch.Tensor):
        return self.model(x)


class ODEBlock(nn.Module):
    def __init__(self, device, neuron_number, t):
        super(ODEBlock, self).__init__()
        self.times = torch.tensor([0.0, t], dtype=torch.float32)
        self.model = ODEFunction(device, neuron_number)

    def forward(self, x):
        return odeint(self.model, x, self.times, method="euler")[1] - x


class BlockODEAgent(nn.Module):
    def __init__(self,
                 device,
                 neuron_number=16,
                 input_dim=4,
                 output_dim=2,
                 t=1.0):
        super(BlockODEAgent, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, neuron_number),
            nn.ReLU(),
            ODEBlock(neuron_number, neuron_number, t),
            nn.ReLU(),
            nn.Linear(neuron_number, output_dim)
        ).to(device)

    def forward(self, x: torch.Tensor):
        return self.model(x)
