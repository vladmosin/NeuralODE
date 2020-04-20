import torch
import torch.nn.functional as F
from torch import nn

from constants.InverseBlockConfig import SBlockConfig, TBlockConfig, InverseBlockConfig


class InverseBlock(nn.Module):
    def __init__(self, inverse_config: InverseBlockConfig):
        super(InverseBlock, self).__init__()

        self.sblock = SBlock(inverse_config.sblock_config).double()
        self.tblock = TBlock(inverse_config.tblock_config).double()
        self.mask = inverse_config.create_mask()
        self.reversed_mask = [not e for e in self.mask]
        self.inverse_config = inverse_config
        self.device = inverse_config.device

    def forward(self, action, state):
        not_trained_action_part = action[:, self.mask]
        trained_action_part = action[:, self.reversed_mask]

        exp_part = self.sblock(trained_action_part, state)
        lin_part = self.tblock(trained_action_part, state)

        return torch.cat([trained_action_part, not_trained_action_part * torch.exp(exp_part) + lin_part], dim=1)

    def inverse(self, action, state):
        trained_dim = self.inverse_config.action_dim - self.inverse_config.not_trained_action
        res = torch.zeros(action.shape, dtype=torch.float64)
        trained_action_part = action[:, :trained_dim]
        not_trained_action_part = action[:, trained_dim:]

        with torch.no_grad():
            exp_part = self.sblock(trained_action_part, state)
            lin_part = self.tblock(trained_action_part, state)

        not_trained_action_part = (not_trained_action_part - lin_part) * torch.exp(-exp_part)
        res[:, self.mask] = not_trained_action_part
        res[:, self.reversed_mask] = trained_action_part

        return res


class SBlock(nn.Module):
    def __init__(self, sblock_config: SBlockConfig):
        super(SBlock, self).__init__()
        self.blocks = sblock_config.create()

    def forward(self, action, state):
        x = torch.cat([action, state], dim=1)
        for block in self.blocks:
            x = F.relu(block(x))
        return x


class TBlock(nn.Module):
    def __init__(self, tblock_config: TBlockConfig):
        super(TBlock, self).__init__()
        self.blocks = tblock_config.create()

    def forward(self, action, state):
        x = torch.cat([action, state], dim=1)
        for block in self.blocks:
            x = F.relu(block(x))
        return x

"""
if __name__ == "__main__":
    sblock_config = SBlockConfig(input_dim=10, output_dim=4, neuron_number=32, block_num=3)
    tblock_config = TBlockConfig(input_dim=10, output_dim=4, neuron_number=32, block_num=3)

    inverse_config = InverseBlockConfig(sblock_config=sblock_config, tblock_config=tblock_config,
                                        action_dim=7, state_dim=7, not_trained_action=4)

    block = InverseBlock(inverse_config=inverse_config)
    action = torch.randn(100, 7)
    state = torch.randn(100, 7)

    ans = block.forward(action, state)
    inverted = block.inverse(ans, state)

    print(torch.norm(action - inverted))
"""