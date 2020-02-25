import copy

import torch


def backprop(loss: torch.Tensor, model: torch.nn.Module, optimizer):
    optimizer.zero_grad()
    loss.backward()
    for i, param in enumerate(model.parameters()):
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def soft_update_backprop(loss: torch.Tensor, model: torch.nn.Module, optimizer, tau):
    backprop(loss, model, optimizer)

    for target_param, param in zip(model.target.parameters(), model.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def to_tensor(x, device):
    return torch.tensor([x], device=device, dtype=torch.float64)