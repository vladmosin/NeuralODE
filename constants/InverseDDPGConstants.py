import copy

import torch

from torch import nn

from agents.InverseAgent import InverseAgent
from constants.InverseAgentConfig import ValueBlockConfig, StateTransformationBlockConfig, InverseAgentConfig
from constants.InverseBlockConfig import SBlockConfig, TBlockConfig, InverseBlockConfig
from enums.AgentType import AgentType
from utils.ActionSelector import NoisedSelector
from utils.MemoryReplay import MemoryReplay
from utils.RewardConverter import MountainCarConverter


class InverseDDPGConstants:
    def __init__(self, device, env):
        self.device = device
        self.env = env
        self.gamma = 0.999
        self.tau = 0.05

        self.inverse_block_number = 2
        self.transformed_state_dim = 300
        self.state_transformation_neuron_number = 300
        self.state_transformation_block_number = 2

        self.value_block_neuron_number = 300
        self.value_block_block_number = 2

        self.num_episodes = 300
        self.start_eps = 0.9
        self.end_eps = 0.005
        self.eps_decay = 1000
        self.neuron_number = 128
        self.memory_size = 20000
        self.lr = 1e-5
        self.batch_size = 64

        self.state_space_dim = self.env.observation_space.shape[0]
        self.action_space_dim = self.env.action_space.shape[0]

        value_block_config = ValueBlockConfig(input_dim=self.transformed_state_dim,
                                                   neuron_number=self.value_block_neuron_number,
                                                   block_num=self.value_block_block_number)
        state_transformation_config = StateTransformationBlockConfig(input_dim=self.state_space_dim,
                                                                          output_dim=self.transformed_state_dim,
                                                                          neuron_number=self.state_transformation_neuron_number,
                                                                          block_num=self.state_transformation_block_number)
        self.normal_distribution_dim = 10
        self.not_trained_action = (self.normal_distribution_dim + self.action_space_dim) // 2
        self.sblock_neuron_number = 200
        self.sblock_block_num = 2
        self.tblock_neuron_number = 200
        self.tblock_block_num = 2

        blocks_input_dim = self.transformed_state_dim + self.action_space_dim + self.normal_distribution_dim - self.not_trained_action
        blocks_output_dim = self.not_trained_action

        sblock_config = SBlockConfig(input_dim=blocks_input_dim,
                                     output_dim=blocks_output_dim,
                                     neuron_number=self.sblock_neuron_number,
                                     block_num=self.sblock_block_num)

        tblock_config = TBlockConfig(input_dim=blocks_input_dim,
                                     output_dim=blocks_output_dim,
                                     neuron_number=self.tblock_neuron_number,
                                     block_num=self.tblock_block_num)

        inverse_block_config = InverseBlockConfig(sblock_config=sblock_config,
                                                  tblock_config=tblock_config,
                                                  action_dim=self.action_space_dim + self.normal_distribution_dim,
                                                  state_dim=self.transformed_state_dim,
                                                  not_trained_action=self.not_trained_action)

        self.inverse_agent_config = InverseAgentConfig(inverse_block_number=self.inverse_block_number,
                                                       inverse_block_config=inverse_block_config,
                                                       state_transformation_config=state_transformation_config,
                                                       value_block_config=value_block_config,
                                                       device=self.device,
                                                       action_dim=self.action_space_dim,
                                                       normal_distribution_dim=self.normal_distribution_dim)

    def get_models(self):
        model = InverseAgent(config=self.inverse_agent_config, env=self.env)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        target_model = copy.deepcopy(model)

        return model, target_model, optimizer

    def get_action_selector(self):
        return NoisedSelector(action_space=self.env.action_space,
                              device=self.device,
                              end_eps=self.end_eps,
                              start_eps=self.start_eps,
                              eps_decay=self.eps_decay)

    def get_reward_converter(self):
        return MountainCarConverter()

    def get_memory(self):
        return MemoryReplay(device=self.device, capacity=self.memory_size)
