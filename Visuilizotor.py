import argparse
import sys
from datetime import datetime
from os import listdir
from pathlib import Path

import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def find_equals_experiments(path, time_from="2020-03-27_00-00-00", time_to="2020-03-29_00-00-00"):
    paths_to_csv = {}
    for experiment in listdir(path):
        if time_from <= experiment <= time_to:
            config = read_config(path, experiment)
            str_config = config_to_string(config)
            if str_config not in paths_to_csv:
                paths_to_csv[str_config] = []

            paths_to_csv[str_config].append("{}/{}".format(path, experiment))

    return paths_to_csv


def read_config(path, experiment):
    filepath = "{}/{}/config.txt".format(path, experiment)
    f = open(filepath, "r")
    lines = f.readlines()
    f.close()
    return lines


def config_to_string(config):
    return "\n".join(config)


def config_from_string(line):
    return line.split("\n")


def draw_graph(str_configs, path_to_csv, on_one_plot=6, cols=2):
    n = len(str_configs)
    rows = max(n // (cols * on_one_plot), 1)
    figs = [[go.Figure() for _ in range(cols)] for _ in range(rows)]
    row, col = 0, 0
    for str_config in str_configs:
        num_episodes = get_num_episodes(str_config)
        rewards = get_rewards(str_config, path_to_csv)
        indexes = list(range(1, num_episodes, num_episodes // len(rewards)))
        figs[row][col].add_trace(go.Scatter(x=indexes, y=rewards,
                                 name=get_name_from_config(str_config)))
        row, col = next_row_col((row, col), rows, cols)
        save_figures(figs)


def save_figures(figs):
    i = 1
    directory = "graphics/experiment{}".format(datetime.now().strftime('%Y-%m-%d_%H-%M'))
    Path(directory).mkdir(parents=True, exist_ok=True)

    dashboard = open(f"{directory}/DASHBOARD.html", 'w')
    dashboard.write("<html><head></head><body>" + "\n")
    for row_figs in figs:
        for fig in row_figs:
            plotly.offline.plot(fig, filename=f"{directory}/Chart_{i}.html", auto_open=False)
            dashboard.write(
                "  <object data=\"Chart_{}.html\" width=\"1200\" height=\"500\"></object>".format(i) + "\n"
            )
            i += 1

    dashboard.write("</body></html>")
    dashboard.close()


def next_row_col(row_col, rows, cols):
    row, col = row_col
    if col < cols - 1:
        return row, col + 1
    elif row < rows - 1:
        return row + 1, 0
    else:
        return 0, 0


def get_name_from_config(str_config):
    params = ['lr', 'batch_size', 'gamma', 'neuron_number']
    lines = []
    config = config_from_string(str_config)
    for line in config:
        parts = line.split('=')
        if len(parts) == 2 and parts[0] in params:
            lines.append(line)

    return " ".join(lines)


def get_rewards(str_config, paths_to_csv):
    paths = paths_to_csv[str_config]
    rewards = []
    for path in paths:
        data = pd.read_csv("{}/log.csv".format(path))
        rewards.append(summarize_rewards(data))

    return np.array(rewards).mean(axis=0)


def get_num_episodes(str_config):
    for line in config_from_string(str_config):
        line = line.strip()
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
    arg_parser.add_argument("--path", default="../DataNeuralODE/Data/experiments/BlockODE")
    arg_parser.add_argument("--date_from", default="2020-03-27")
    arg_parser.add_argument("--date_to", default="2020-03-29")

    args = arg_parser.parse_args(sys.argv[1:])
    path = args.path
    time_from = args.date_from
    time_to = args.date_to

    if time_from.split('-') == 3:
        time_from = time_from + "_00-00-00"
    if time_to.split('-') == 3:
        time_to = time_to + "_00-00-00"

    paths_to_csv = find_equals_experiments(path=path, time_from=time_from, time_to=time_to)
    str_configs = paths_to_csv.keys()
    draw_graph(str_configs=str_configs, path_to_csv=paths_to_csv)