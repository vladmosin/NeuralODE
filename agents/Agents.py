from torch import nn
from torchdiffeq import odeint_adjoint as odeint
from torch import tensor
import torch


class CommonAgent(nn.Module):
    def __init__(self, device, neuron_number=32, input_dim=4, output_dim=2):
        super(CommonAgent, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, neuron_number),
            nn.ReLU(),
            nn.Linear(neuron_number, neuron_number),
            nn.ReLU(),
            nn.Linear(neuron_number, output_dim)
        ).to(device).double()

    def forward(self, x):
        return self.model.forward(x)


# What is an initial value? Just zero?
# Problems with different dimensions of h(t + 1) and h(t)
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

        self.model = nn.Sequential(
            nn.Linear(input_dim, neuron_number),
            nn.ReLU(),
            nn.Linear(neuron_number, neuron_number),
            nn.ReLU(),
            nn.Linear(neuron_number, output_dim)
        ).to(device)

    def forward(self, x: torch.Tensor):
        predicted = self.model(x)
        return odeint(predicted, torch.zeros((x.shape[0], self.output_dim),
                                             dtype=torch.float64, device=self.device), self.time)[1]


class BlockODEAgent(nn.Module):
    def __init__(self,
                 device,
                 neuron_number=16,
                 block_number=2,
                 input_dim=4,
                 output_dim=2,
                 time=1):
        super(BlockODEAgent, self).__init__()

        self.time = tensor([0, time], dtype=torch.float64, device=device)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.neuron_number = neuron_number
        self.device = device

        self.blocks = [self.create_block(i == 0, i == block_number - 1) for i in range(block_number)]

    def create_block(self, is_first, is_last):
        if is_first:
            input_dim = self.input_dim
        else:
            input_dim = self.output_dim
        if is_last:
            return nn.Linear(input_dim, self.output_dim)
        else:
            return nn.Sequential(
                nn.Linear(input_dim, self.neuron_number),
                nn.ReLU()
            )

    def forward(self, x: torch.Tensor):
        pass