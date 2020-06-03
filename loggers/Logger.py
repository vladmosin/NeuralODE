from datetime import datetime
import pandas as pd
from pathlib import Path
from tensorboardX import SummaryWriter
import torch
import numpy as np

class Logger:
    def __init__(self, config, tester):
        self.rewards = []
        self.config = config
        self.tester = tester
        self.losses = []
        self.current_losses = []

    # reward is 1-dimensional tensor
    def add(self, reward):
        self.rewards.append(reward)

    def add_loss(self, loss):
        self.current_losses.append(loss.item())

    def add_average_loss(self):
        self.losses.append(np.mean(self.current_losses))

    def save_losses(self, directory):
        real_folder = "{}".format(directory)

        tb_writer = SummaryWriter(real_folder)
        for i, loss in enumerate(self.losses):
            tb_writer.add_scalar("loss", loss, i)

        tb_writer.close()

    # rewards is a two-dimensional list
    def to_csv(self, directory):
        csv_rewards, indexes = [], []

        for i, epoch_rewards in enumerate(self.rewards):
            csv_rewards += epoch_rewards
            indexes += [i + 1] * len(epoch_rewards)

        data = pd.DataFrame.from_dict({'reward': csv_rewards, 'index': indexes})

        Path(directory).mkdir(parents=True, exist_ok=True)
        data.to_csv("{}/log.csv".format(directory))

    def to_tensorboard(self, directory):
        tb_writer = SummaryWriter("{}".format(directory))

        for i, reward in enumerate(self.rewards):
            reward = np.array(reward)
            tb_writer.add_scalar("mean", reward.mean(), i)
            tb_writer.add_scalar("std", reward.std(), i)

        tb_writer.close()

    def save_config(self, directory):
        f = open("{}/config.txt".format(directory), "w")
        f.write(str(self.config))
        f.write(str(self.tester))

    def save_all(self):
        cur_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        directory = "experiments1/{}/{}".format(self.config.agent_type.value, cur_time)
        Path(directory).mkdir(parents=True, exist_ok=True)

        self.save_losses(directory)
        self.to_tensorboard(directory)
        self.to_csv(directory)
        self.save_config(directory)

