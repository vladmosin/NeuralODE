from torch import nn
from torchdiffeq import odeint
from torch import tensor
import torch


class ODEFunction(nn.Module):
    def __init__(self,
                 device,
                 neuron_number=32):
        super(ODEFunction, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(neuron_number, neuron_number).double()
        )

    def forward(self, t, x: torch.Tensor):
        return self.model(x)


class ODEBlock(nn.Module):
    def __init__(self, device, neuron_number):
        super(ODEBlock, self).__init__()
        self.times = torch.tensor([0, 1], dtype=torch.double)
        self.model = ODEFunction(device, neuron_number)

    def forward(self, x):
        return odeint(self.model, x, self.times)[1]


class BlockODEAgent(nn.Module):
    def __init__(self,
                 device,
                 neuron_number=16,
                 input_dim=4,
                 output_dim=2):
        super(BlockODEAgent, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, neuron_number),
            nn.ReLU(),
            ODEBlock(neuron_number, neuron_number),
            nn.ReLU(),
            nn.Linear(neuron_number, output_dim)
        ).to(device).double()

    def forward(self, x: torch.Tensor):
        return self.model(x)