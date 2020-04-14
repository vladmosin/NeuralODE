from random import shuffle
from utils.Utils import dense_net


class SBlockConfig:
    def __init__(self, input_dim, output_dim, neuron_number, block_num):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neuron_number = neuron_number
        self.block_num = block_num

    def create(self):
        return dense_net(input_dim=self.input_dim, output_dim=self.output_dim,
                         neuron_number=self.neuron_number, block_num=self.block_num)


class TBlockConfig:
    def __init__(self, input_dim, output_dim, neuron_number, block_num):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neuron_number = neuron_number
        self.block_num = block_num

    def create(self):
        return dense_net(input_dim=self.input_dim, output_dim=self.output_dim,
                         neuron_number=self.neuron_number, block_num=self.block_num)


class InverseBlockConfig:
    def __init__(self, sblock_config: SBlockConfig, tblock_config: TBlockConfig,
                 action_dim, state_dim, not_trained_action):
        self.sblock_config = sblock_config
        self.tblock_config = tblock_config

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.not_trained_action = not_trained_action
        self.check_consistency()

    def create_mask(self):
        mask = [False] * self.action_dim
        indexes = list(range(self.action_dim))
        shuffle(indexes)
        indexes = indexes[:self.not_trained_action]
        for index in indexes:
            mask[index] = True
        return mask

    def check_consistency(self):
        assert self.action_dim + self.state_dim - self.not_trained_action == self.sblock_config.input_dim
        assert self.not_trained_action == self.sblock_config.output_dim
        assert self.sblock_config.input_dim == self.tblock_config.input_dim
        assert self.sblock_config.output_dim == self.tblock_config.output_dim
