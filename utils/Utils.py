import copy

import torch
from torch import nn


def backprop(loss: torch.Tensor, model: torch.nn.Module, optimizer):
    optimizer.zero_grad()
    loss.backward()
#    for i, param in enumerate(model.parameters()):
#        print(param.shape)
#        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def soft_update_backprop(loss: torch.Tensor, model, optimizer, tau):
    net, target_net = model
    backprop(loss, net, optimizer)

    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def to_tensor(x, device):
    return torch.tensor([x], device=device, dtype=torch.float32)


def dense_net(input_dim, output_dim, neuron_number, block_num, device):
    if block_num == 1:
        return nn.Linear(in_features=input_dim, out_features=output_dim).to(
            device=device, dtype=torch.float64
        )

    blocks = [nn.Linear(in_features=input_dim, out_features=neuron_number).to(
            device=device, dtype=torch.float64
        )]
    for _ in range(1, block_num):
        blocks.append(nn.Linear(in_features=neuron_number, out_features=neuron_number).to(
            device=device, dtype=torch.float64
        ))

    blocks.append(nn.Linear(in_features=neuron_number, out_features=output_dim).to(
            device=device, dtype=torch.float64
        ))
    return blocks


def to_numpy(x):
    return x.cpu().detach().numpy()
