from datetime import datetime
import pandas as pd
from pathlib import Path
from tensorboardX import SummaryWriter
import torch
import numpy as np

"""
Writes rewards to file located in data/
Next folder is csv or tensorboard 
Next folder is algorithm (DQN, DDPG, ...)
Next folder is type of distance between to test measurements (Episode, OptimizeStep)
Next folder is Exploration or Exploit 
Next folder is distance between two measurements
Last folder is a type of agent neural network (Common, NeuralODE, NeuralBlockODE)
Filename is a date and time of experiment
"""


class Logger:
    def __init__(self, algorithm, distance_type, exploration,
                 agent_type, distance, start_distance=1):
        self.rewards = []
        self.algorithm = algorithm
        self.distance_type = distance_type.value
        self.distance = distance
        self.start_distance = start_distance
        self.agent_type = agent_type.value
        self.exploration = exploration.value
        self.losses = []
        self.current_losses = []

    # reward is 1-dimensional tensor
    def add(self, reward):
        self.rewards.append(reward)

    def add_loss(self, loss):
        self.current_losses.append(loss.item())

    def add_average_loss(self):
        self.losses.append(np.mean(self.current_losses))

    def save_losses(self, folder):
        real_folder = "../losses/{}".format(folder)
        Path(real_folder).mkdir(parents=True, exist_ok=True)

        tb_writer = SummaryWriter(real_folder)
        for i, loss in enumerate(self.losses):
            tb_writer.add_scalar("loss", loss, i)

        tb_writer.close()


    # rewards is a two-dimensional list
    def to_csv(self):
        distances, csv_rewards, indexes = [], [], []
        current_distance = self.start_distance
        directory, filename = self.file_path("csv")

        for i, epoch_rewards in enumerate(self.rewards):
            csv_rewards += epoch_rewards
            distances += [current_distance] * len(epoch_rewards)
            indexes += [i + 1] * len(epoch_rewards)

            current_distance += self.distance

        data = pd.DataFrame.from_dict({'reward': csv_rewards, 'distance': distances, 'index': indexes})

        Path(directory).mkdir(parents=True, exist_ok=True)
        data.to_csv("{}/{}".format(directory, filename))

    # log type: csv or tensorboard
    def file_path(self, log_type):
        filename = "{}.csv".format(datetime.now().strftime('%Y-%m-%d_%H-%M'))
        directory = "../data/{}/{}/{}/{}/{}/{}".format(
            log_type, self.algorithm, self.agent_type,
            self.exploration, self.distance_type, self.distance)

        return directory, filename

    def to_tensorboard(self):
        directory, filename = self.file_path("tensorboard")
        experiment_directory = filename[:-4]  # remove extension
        tb_writer = SummaryWriter("{}/{}".format(directory, experiment_directory))

        for i, reward in enumerate(self.rewards):
            reward = np.array(reward)
            tb_writer.add_scalar("mean", reward.mean(), i)
            tb_writer.add_scalar("std", reward.std(), i)

        tb_writer.close()
