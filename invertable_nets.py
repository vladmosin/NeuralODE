import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import abc


def atanh(x):
    return 0.5*(torch.log(1 + x + 1e-20) - torch.log(1 - x + 1e-20))  # 1e-20 will affect only if x is exactly 1 or -1 due to float precision


def _get_subnet(input_size, conditional_size, hidden_size, output_size=None):
    if output_size is None:
        output_size = input_size
    return nn.Sequential(
        torch.nn.Linear(input_size + conditional_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size)
    )


def _get_subnet_sigmoid(input_size, conditional_size, hidden_size, output_size=None):
    if output_size is None:
        output_size = input_size
    return nn.Sequential(
        torch.nn.Linear(input_size + conditional_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size),
        torch.nn.Sigmoid()  # Required for training stability
    )


def _get_subnet_tanh(input_size, conditional_size, hidden_size, output_size=None):
    if output_size is None:
        output_size = input_size
    return nn.Sequential(
        torch.nn.Linear(input_size + conditional_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size),
        torch.nn.Tanh()  # Required for training stability
    )


def _soft_sign(input):
    input = input - torch.log(-torch.log(torch.rand_like(input) * (1 - 1e-20) + 1e-20) + 1e-20) # Add noise so that values will be non-zero almost surely
    return (torch.sign(input) - input).detach() + input


class InvertibleModule(nn.Module):
    def __init__(self, input_size, condition_size=None, hidden_size=None):
        super().__init__()
        self.input_size = input_size
        self.condition_size = condition_size
        self.hidden_size = hidden_size

    def forward(self, x, condition=None):
        return self._forward(x, condition)[0]

    def backward(self, x, condition=None):
        return self._backward(x, condition)[0]

    @abc.abstractmethod
    def _forward(self, x, condition=None):
        pass

    @abc.abstractmethod
    def _backward(self, x, condition=None):
        pass


class LeakyReluIM(InvertibleModule):
    def _forward(self, x, condition=None):
        mask = (x > 0).float()
        return mask * x + (1 - mask) * x * 0.01, np.log(0.01) * (1 - mask).sum(dim=-1)

    def _backward(self, x, condition=None):
        mask = (x > 0).float()
        return mask * x + (1 - mask) * x / 0.01, np.log(1 / 0.01) * (1 - mask).sum(dim=-1)

    def __init__(self, input_size=None, hidden_size=None):
        super().__init__(input_size, hidden_size)


class TanhIM(InvertibleModule):
    def _forward(self, x, condition=None):
        return torch.tanh(x), 2 * torch.log(torch.cosh(x)).sum(dim=-1)

    def _backward(self, x, condition=None):
        atanh = 0.5*(torch.log(1 + x + 1e-20) - torch.log(1 - x + 1e-20))  # 1e-20 will affect only if x is exactly 1 or -1 due to float precision
        log_det_jacobian = (torch.log(torch.abs(-x)) - torch.log((1 + x + 1e-20) * (1 - x + 1e-20))).sum(dim=-1)
        return atanh(x), log_det_jacobian

    def __init__(self, input_size=None, hidden_size=None):
        super().__init__(input_size, hidden_size)


class LinearInvertibleModule(InvertibleModule):
    def __init__(self, input_size, hidden_size=None):
        super().__init__(input_size, hidden_size)
        mask = torch.tensor([[(0 if j < i else 1) for j in range(input_size)] for i in range(input_size)]).float()
        self.u_mask = nn.Parameter(data=mask, requires_grad=False)
        self.eye = nn.Parameter(data=torch.eye(input_size).float(), requires_grad=False)

        self.bias = nn.Parameter(torch.Tensor(input_size).float())
        self.bias.data.uniform_(-1/input_size**0.5, 1/input_size**0.5)

        self.m = nn.Parameter(torch.Tensor(input_size, input_size).float())
        self.m.data.uniform_(-0.1/input_size**0.5, 0.1/input_size**0.5)

        self.v_1 = nn.Parameter(torch.Tensor(input_size).float())
        self.v_1.data.uniform_(-1/input_size**0.5, 1/input_size**0.5)
        self.v_2 = nn.Parameter(torch.Tensor(input_size).float())
        self.v_2.data.uniform_(-1/input_size**0.5, 1/input_size**0.5)

    def _forward(self, x, condition=None):
        h_1 = self.eye - 2*self.v_1.ger(self.v_1) / self.v_1.matmul(self.v_1)
        h_2 = self.eye - 2*self.v_2.ger(self.v_2) / self.v_2.matmul(self.v_2)
        q = h_1.matmul(h_2)
        r = self.u_mask * self.m
        a = q.matmul(r)
        return x.matmul(a) + self.bias.unsqueeze(0).expand_as(x), r.diag().abs().log().unsqueeze(0).expand_as(x).sum(-1)

    def _backward(self, x, condition=None):
        h_1 = self.eye - 2*self.v_1.ger(self.v_1) / self.v_1.matmul(self.v_1)
        h_2 = self.eye - 2*self.v_2.ger(self.v_2) / self.v_2.matmul(self.v_2)
        q_inv = h_1.matmul(h_2).inverse()
        r_inv = (self.u_mask * self.m).inverse()
        a_inv = r_inv.matmul(q_inv)
        return (x - self.bias.unsqueeze(0).expand_as(x)).matmul(a_inv), r_inv.diag().abs().log().unsqueeze(0).expand_as(x).sum(-1)


class ConditionalLinearIM(InvertibleModule):
    def __init__(self, input_size, condition_size, hidden_size):
        super().__init__(input_size, condition_size, hidden_size)
        # Condition
        self.condition = _get_subnet(condition_size, 0, hidden_size, input_size * input_size)

        # Parameters
        mask = torch.tensor([[(0 if j < i else 1) for j in range(input_size)] for i in range(input_size)]).float()
        self.u_mask = nn.Parameter(data=mask, requires_grad=False)
        self.eye = nn.Parameter(data=torch.eye(input_size).float(), requires_grad=False)

        self.bias = nn.Parameter(torch.Tensor(input_size).float())
        self.bias.data.uniform_(-1 / input_size ** 0.5, 1 / input_size ** 0.5)

        self.m = nn.Parameter(torch.Tensor(input_size, input_size).float())
        self.m.data.uniform_(-0.1 / input_size ** 0.5, 0.1 / input_size ** 0.5)

        self.v_1 = nn.Parameter(torch.Tensor(input_size).float())
        self.v_1.data.uniform_(-1 / input_size ** 0.5, 1 / input_size ** 0.5)
        self.v_2 = nn.Parameter(torch.Tensor(input_size).float())
        self.v_2.data.uniform_(-1 / input_size ** 0.5, 1 / input_size ** 0.5)

    def _forward(self, x, condition=None):
        c = self.condition(condition).view(-1, self.input_size, self.input_size)

        h_1 = self.eye - 2 * self.v_1.ger(self.v_1) / self.v_1.matmul(self.v_1)
        h_2 = self.eye - 2 * self.v_2.ger(self.v_2) / self.v_2.matmul(self.v_2)
        q = h_1.matmul(h_2)
        r = self.u_mask * self.m.matmul(c)
        a = q.matmul(r)
        return torch.bmm(a, x.unsqueeze(-1)).view(*x.size()) + self.bias.unsqueeze(0).expand_as(x), \
               r.diagonal(dim1=-1, dim2=-2).abs().log().sum(-1)

    def _backward(self, x, condition=None):
        c = self.condition(condition).view(-1, self.input_size, self.input_size)

        h_1 = self.eye - 2 * self.v_1.ger(self.v_1) / self.v_1.matmul(self.v_1)
        h_2 = self.eye - 2 * self.v_2.ger(self.v_2) / self.v_2.matmul(self.v_2)
        q_inv = h_1.matmul(h_2).inverse()
        r_inv = (self.u_mask * self.m.matmul(c)).inverse()
        a_inv = r_inv.matmul(q_inv)
        return torch.bmm(a_inv, (x - self.bias.unsqueeze(0).expand_as(x)).unsqueeze(-1)).view(*x.size()), \
               r_inv.diagonal(dim1=-1, dim2=-2).abs().log().sum(-1)


class ConditionalSignIM(InvertibleModule):
    def __init__(self, input_size, condition_size, hidden_size):
        super().__init__(input_size, condition_size, hidden_size)
        mask = torch.zeros(input_size).float()
        mask[np.random.choice(input_size, input_size // 2)] = 1.0
        self.mask = nn.Parameter(data=mask, requires_grad=False)
        self.net_a = _get_subnet_tanh(input_size, condition_size, hidden_size)
        self.net_b = _get_subnet_tanh(input_size, condition_size, hidden_size)

    def _forward(self, x, condition):
        mask = self.mask.unsqueeze(0).expand_as(x)
        mask_inv = (1.0 - self.mask).unsqueeze(0).expand_as(x)
        a = _soft_sign(self.net_a(torch.cat([mask_inv*x, condition], dim=-1)))
        x = x * mask_inv + mask * x * a
        b = _soft_sign(self.net_b(torch.cat([mask*x, condition], dim=-1)))
        x = x * mask + mask_inv * x * b
        return x, torch.zeros_like(x).sum(-1)

    def _backward(self, x, condition):
        mask = self.mask.unsqueeze(0).expand_as(x)
        mask_inv = (1.0 - self.mask).unsqueeze(0).expand_as(x)
        b = _soft_sign(self.net_b(torch.cat([mask*x, condition], dim=-1)))
        x = x * mask + mask_inv * x * b
        a = _soft_sign(self.net_a(torch.cat([mask_inv*x, condition], dim=-1)))
        x = x * mask_inv + mask * x * a
        return x, torch.zeros_like(x).sum(-1)


class ConditionalSquaredIM(InvertibleModule):
    def __init__(self, input_size, condition_size, hidden_size):
        super().__init__(input_size, condition_size, hidden_size)
        mask = torch.zeros(input_size).float()
        mask[np.random.choice(input_size, input_size // 2)] = 1.0
        self.mask = nn.Parameter(data=mask, requires_grad=False)
        self.net_a = _get_subnet_tanh(input_size, condition_size, hidden_size)
        self.net_b = _get_subnet_tanh(input_size, condition_size, hidden_size)

    def _forward(self, x, condition):
        mask = self.mask.unsqueeze(0).expand_as(x)
        mask_inv = (1.0 - self.mask).unsqueeze(0).expand_as(x)
        a = self.net_a(torch.cat([mask_inv*x, condition], dim=-1)) / 2
        x = x * mask_inv + mask * (x**2 * a + x - a)
        b = self.net_b(torch.cat([mask*x, condition], dim=-1)) / 2
        x = x * mask + mask_inv * (x**2 * b + x - b)
        return x, None  # TODO: Jacobian

    def _backward(self, x, condition):
        mask = self.mask.unsqueeze(0).expand_as(x)
        mask_inv = (1.0 - self.mask).unsqueeze(0).expand_as(x)
        b = self.net_b(torch.cat([mask*x, condition], dim=-1))
        x = x * mask + mask_inv * (-1 + (1 + b * (b - 2 * x))**0.5) / b  # TODO: Handle b == 0
        a = self.net_a(torch.cat([mask_inv*x, condition], dim=-1))
        x = x * mask_inv + mask * (-1 + (1 + a * (a - 2 * x))**0.5) / a  # TODO: Handle a == 0
        return x, None  # TODO: Jacobian


class ConditionalFracIM(InvertibleModule):
    def __init__(self, input_size, condition_size, hidden_size):
        super().__init__(input_size, condition_size, hidden_size)
        mask = torch.zeros(input_size).float()
        mask[np.random.choice(input_size, input_size // 2)] = 1.0
        self.mask = nn.Parameter(data=mask, requires_grad=False)
        self.net_a = _get_subnet_tanh(input_size, condition_size, hidden_size)
        self.net_b = _get_subnet_tanh(input_size, condition_size, hidden_size)

    def _forward(self, x, condition):
        mask = self.mask.unsqueeze(0).expand_as(x)
        mask_inv = (1.0 - self.mask).unsqueeze(0).expand_as(x)
        a = self.net_a(torch.cat([mask_inv*x, condition], dim=-1))
        x = x * mask_inv + mask * (x + a) / (a * x + 1)
        b = self.net_b(torch.cat([mask*x, condition], dim=-1))
        x = x * mask + mask_inv * (x + b) / (b * x + 1)
        return x, None  # TODO: Jacobian

    def _backward(self, x, condition):
        mask = self.mask.unsqueeze(0).expand_as(x)
        mask_inv = (1.0 - self.mask).unsqueeze(0).expand_as(x)
        b = self.net_b(torch.cat([mask*x, condition], dim=-1))
        x = x * mask + mask_inv * (x - b) / (1 - b * x)
        a = self.net_a(torch.cat([mask_inv*x, condition], dim=-1))
        x = x * mask_inv + mask * (x - a) / (1 - a * x)
        return x, None  # TODO: Jacobian


class ConditionalBiasedProdIM(InvertibleModule):
    def __init__(self, input_size, condition_size, hidden_size):
        super().__init__(input_size, condition_size, hidden_size)
        mask = torch.zeros(input_size).float()
        mask[np.random.choice(input_size, input_size // 2)] = 1.0
        self.mask = nn.Parameter(data=mask, requires_grad=False)
        self.net_a_s = _get_subnet_tanh(input_size, condition_size, hidden_size)
        self.net_b_s = _get_subnet_tanh(input_size, condition_size, hidden_size)
        self.net_a_t = _get_subnet(input_size, condition_size, hidden_size)
        self.net_b_t = _get_subnet(input_size, condition_size, hidden_size)

    def _forward(self, x, condition):
        mask = self.mask.unsqueeze(0).expand_as(x)
        mask_inv = (1.0 - self.mask).unsqueeze(0).expand_as(x)
        s = self.net_a_s(torch.cat([mask_inv*x, condition], dim=-1))
        t = self.net_a_t(torch.cat([mask_inv*x, condition], dim=-1))
        x = x * mask_inv + mask * (x * torch.exp(s) + t)
        s = self.net_b_s(torch.cat([mask*x, condition], dim=-1))
        t = self.net_b_t(torch.cat([mask*x, condition], dim=-1))
        x = x * mask + mask_inv * (x * torch.exp(s) + t)
        return x, None  # TODO: Jacobian

    def _backward(self, x, condition):
        mask = self.mask.unsqueeze(0).expand_as(x)
        mask_inv = (1.0 - self.mask).unsqueeze(0).expand_as(x)
        s = self.net_b_s(torch.cat([mask*x, condition], dim=-1))
        t = self.net_b_t(torch.cat([mask*x, condition], dim=-1))
        x = x * mask + mask_inv * ((x - t) / torch.exp(s))
        s = self.net_a_s(torch.cat([mask_inv*x, condition], dim=-1))
        t = self.net_a_t(torch.cat([mask_inv*x, condition], dim=-1))
        x = x * mask_inv + mask * ((x - t) / torch.exp(s))
        return x, None  # TODO: Jacobian


class SequentialInvertibleNet(InvertibleModule):
    def _forward(self, x, condition=None):
        jacobians = torch.zeros_like(x).sum(-1)
        for m in self.m:
            x, j = m._forward(x, condition)
            jacobians += j
        return x, jacobians

    def _backward(self, x, condition=None):
        jacobians = torch.zeros_like(x).sum(-1)
        for m in reversed(self.m):
            x, j = m._backward(x, condition)
            jacobians += j
        return x, jacobians

    def __init__(self, *modules):
        super().__init__(None)
        self.m = list(modules)


class InvertibleSequential(InvertibleModule):
    def _forward(self, x, condition=None):
        jacobians = torch.zeros_like(x).sum(-1)
        for m in self.m:
            x, j = m._forward(x, condition)
            jacobians += j
        return x, jacobians

    def _backward(self, x, condition=None):
        jacobians = torch.zeros_like(x).sum(-1)
        for m in reversed(self.m):
            x, j = m._backward(x, condition)
            jacobians += j
        return x, jacobians

    def __init__(self, *modules):
        super().__init__(None)
        self.m = torch.nn.ModuleList(modules)


class ConditionalThreeDirectionalNet(nn.Module):
    def __init__(self, z_to_latent, x_to_latent, y_to_latent):
        super().__init__()
        self.z_to_latent = z_to_latent
        self.x_to_latent = x_to_latent
        self.y_to_latent = y_to_latent

    def xy_to_z(self, x, y, condition):
        x_latent, x_jacobian = self.x_to_latent._forward(x, condition)
        y_latent, y_jacobian = self.y_to_latent._forward(y, condition)
        z, z_jacobian = self.z_to_latent._backward(x_latent + y_latent, condition)
        return z, (x_jacobian, y_jacobian, z_jacobian)

    def xz_to_y(self, x, z, condition):
        x_latent, x_jacobian = self.x_to_latent._forward(x, condition)
        z_latent, z_jacobian = self.z_to_latent._forward(z, condition)
        y, y_jacobian = self.z_to_latent._backward(x_latent - z_latent, condition)
        return y, (x_jacobian, y_jacobian, z_jacobian)

    def yz_to_x(self, y, z, condition):
        y_latent, y_jacobian = self.y_to_latent._forward(y, condition)
        z_latent, z_jacobian = self.z_to_latent._forward(z, condition)
        x, x_jacobian = self.x_to_latent._backward(y_latent - z_latent, condition)
        return x, (x_jacobian, y_jacobian, z_jacobian)
