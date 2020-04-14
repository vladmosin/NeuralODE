from agents.InverseBlock import InverseBlock
from constants.InverseBlockConfig import InverseBlockConfig
from utils.Utils import dense_net


class ValueBlockConfig:
    def __init__(self, input_dim, neuron_number, block_num):
        self.input_dim = input_dim
        self.neuron_number = neuron_number
        self.block_num = block_num

    def create(self):
        return dense_net(input_dim=self.input_dim, output_dim=1,
                         neuron_number=self.neuron_number, block_num=self.block_num)


class StateTransformationBlockConfig:
    def __init__(self, input_dim, output_dim, neuron_number, block_num):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neuron_number = neuron_number
        self.block_num = block_num

    def create(self):
        return dense_net(input_dim=self.input_dim, output_dim=self.output_dim,
                         neuron_number=self.neuron_number, block_num=self.block_num)


class InverseAgentConfig:
    def __init__(self, inverse_block_number,
                 inverse_block_config: InverseBlockConfig,
                 state_transformation_config: StateTransformationBlockConfig,
                 value_block_config: ValueBlockConfig, device,
                 action_dim, normal_distribution_dim):

        self.inverse_block_number = inverse_block_number
        self.inverse_block_config = inverse_block_config
        self.state_transformation_config = state_transformation_config
        self.value_block_config = value_block_config
        self.device = device
        self.action_dim = action_dim
        self.normal_distribution_dim = normal_distribution_dim


    def create_inverse_blocks(self):
        return [InverseBlock(self.inverse_block_config) for _ in range(self.inverse_block_number)]
