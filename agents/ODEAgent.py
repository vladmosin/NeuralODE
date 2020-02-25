from torch import nn
from torchdiffeq import odeint
from torch import tensor
import torch


class ODEFunction(nn.Module):
    def __init__(self,
                 device,
                 neuron_number=32,
                 input_dim=4
                 ):
        super(ODEFunction, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, neuron_number),
            nn.ReLU(),
            nn.Linear(neuron_number, neuron_number),
            nn.ReLU(),
            nn.Linear(neuron_number, input_dim)
        ).to(device).double()

    def forward(self, t, x: torch.Tensor):
        return self.model(x)


class ODEAgent(nn.Module):
    def __init__(self,
                 device,
                 neuron_number=32,
                 input_dim=4,
                 output_dim=2,
                 time=1):
        super(ODEAgent, self).__init__()

        self.time = tensor([0, time], dtype=torch.float64, device=device)
        self.output_dim = output_dim
        self.device = device

        self.gradient = ODEFunction(device=device,
                                    neuron_number=neuron_number,
                                    input_dim=input_dim)

        self.last = nn.Linear(input_dim, output_dim).double()

    def forward(self, x: torch.Tensor):
        transformed = odeint(self.gradient, x, self.time)[1]
        return self.last(transformed)