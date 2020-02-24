from datetime import datetime
import pandas as pd
from pathlib import Path
from tensorboardX import SummaryWriter

"""
Writes rewards to file located in data/csv/
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

    # reward is 1-dimensional tensor
    def add(self, reward):
        self.rewards.append(reward)

    # rewards is a two-dimensional list
    def to_csv(self):
        distances, csv_rewards, indexes = [], [], []
        current_distance = self.start_distance
        directory, filename = self.file_path()

        for i, epoch_rewards in enumerate(self.rewards):
            csv_rewards += epoch_rewards
            distances += [current_distance] * len(epoch_rewards)
            indexes += [i + 1] * len(epoch_rewards)

            current_distance += self.distance

        data = pd.DataFrame.from_dict({'reward': csv_rewards, 'distance': distances, 'index': indexes})

        Path(directory).mkdir(parents=True, exist_ok=True)
        data.to_csv("{}/{}".format(directory, filename))

    def file_path(self):
        filename = "{}.csv".format(datetime.now().strftime('%Y-%m-%d_%H-%M'))
        directory = "../data/csv/{}/{}/{}/{}/{}".format(
            self.algorithm, self.agent_type, self.exploration, self.distance_type, self.distance)

        return directory, filename

    def to_tensorboard(self):
        pass