from torch import nn


class CommonAgent(nn.Module):
    def __init__(self, device, neuron_number=32, input_dim=4, output_dim=2, t=1.0):
        super(CommonAgent, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, neuron_number),
            nn.ReLU(),
            nn.Linear(neuron_number, neuron_number),
            nn.ReLU(),
            nn.Linear(neuron_number, output_dim)
        ).to(device)

    def forward(self, x):
        return self.model.forward(x)