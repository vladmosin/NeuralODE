import argparse
import sys
from os import listdir
import pandas as pd
import numpy as np
import plotly.graph_objects as go


def find_equals_experiments(path, time_from="2020-03-27_00-00-00", time_to="2020-03-29_00-00-00"):
    paths_to_csv = {}
    for experiment in listdir(path):
        if time_from <= experiment <= time_to:
            config = read_config(path, experiment)
            if config not in paths_to_csv:
                paths_to_csv[config] = []

            paths_to_csv[config].append("{}/{}".format(path, experiment))

    return paths_to_csv


def read_config(path, experiment):
    filepath = "{}/{}/config.txt".format(path, experiment)
    f = open(filepath, experiment)
    lines = f.readlines()
    f.close()
    return lines


def draw_graph(configs, path_to_csv):
    fig = go.Figure()
    for config in configs:
        num_episodes = get_num_episodes(config)
        rewards = get_rewards(config, path_to_csv)
        indexes = list(range(1, num_episodes, num_episodes // len(rewards)))
        fig.add_trace(go.Scatter(x=indexes, y=rewards, name=get_name_from_config(config)))
    fig.show()


def get_name_from_config(config):
    params = ['lr', 'batch_size', 'gamma', 'neuron_number']
    lines = []
    for line in config:
        parts = line.split('=')
        if len(parts) == 2 and parts[0] in params:
            lines.append(line)

    return " ".join(lines)


def get_rewards(config, paths_to_csv):
    paths = paths_to_csv[config]
    rewards = []
    for path in paths:
        data = pd.read_csv("{}/log.csv".format(path))
        rewards.append(summarize_rewards(data))

    return np.array(rewards).mean(axis=0)


def get_num_episodes(config):
    for line in config:
        line = line.trim()
        parts = line.split('=')
        if len(parts) == 2 and parts[0] == 'num_episodes':
            return int(parts[1])


def summarize_rewards(data: pd.DataFrame):
    data = data[['index', 'reward']]
    rewards = []
    index_from, index_to = data['index'].min(), data['index'].max()
    for i in range(index_from, index_to + 1):
        reward = data[data['index'] == i]['reward'].mean()
        rewards.append(reward)

    return rewards


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", default="../experiments/BlockODE")
    arg_parser.add_argument("--date_from", default="2020-03-27")
    arg_parser.add_argument("--date_to", default="2020-03-29")

    args = arg_parser.parse_args(sys.argv)
    path = args.path
    time_from = args.date_from
    time_to = args.date_to

    if time_from.split('-') == 3:
        time_from = time_from + "_00-00-00"
    if time_to.split('-') == 3:
        time_to = time_to + "_00-00-00"

    paths_to_csv = find_equals_experiments(path=path, time_from=time_from, time_to=time_to)
    configs = paths_to_csv.keys()

    draw_graph(configs=configs, path_to_csv=paths_to_csv)