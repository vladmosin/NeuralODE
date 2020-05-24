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


def find_equals_experiments(path, time_from, time_to, params):
    paths_to_csv_ode = {}
    paths_to_csv_common = {}

    common_path = "{}/Common".format(path)
    ode_path = "{}/BlockODE".format(path)

    fill_paths_to_csv(ode_path, paths_to_csv_ode, params, False)
    fill_paths_to_csv(common_path, paths_to_csv_common, params, True)

    return paths_to_csv_ode, paths_to_csv_common


def fill_paths_to_csv(path, paths_to_csv, params, common):
    for experiment in listdir(path):
        if time_from <= experiment <= time_to:
            config = read_config(path, experiment)

            if not config_satisfies(config):
                continue

            str_config = config_to_string(config, params, common)
            if str_config not in paths_to_csv:
                paths_to_csv[str_config] = []

            paths_to_csv[str_config].append("{}/{}".format(path, experiment))


def config_satisfies(config):
    for line in config:
        parts = line.split("=")
        if parts[0] == "neuron_number":
            if int(parts[1]) != 16:
                return False

    return True


def read_config(path, experiment):
    filepath = "{}/{}/config.txt".format(path, experiment)
    f = open(filepath, "r")
    lines = f.readlines()
    f.close()
    return lines


def config_to_string(config, params, common):
    observed_params = []
    for line in config:
        parts = line.split("=")
        if common and parts[0] == 't':
            continue
        elif parts[0] in params:
            observed_params.append(line)

    return ", ".join(observed_params)


def config_from_string(line):
    return line.split("\n")


def draw_graph(path_to_csv_common, path_to_csv_ode, filename):
    fig = go.Figure()
    add_rewards(paths_to_csv_common, fig, 1000)
    add_rewards(paths_to_csv_ode, fig, 1000, prefix="ODE: ")

    plotly.offline.plot(fig, filename=filename)


def add_rewards(path_to_csv, fig, num_episodes, prefix=""):
    for str_config in path_to_csv:
        rewards = get_rewards(str_config, path_to_csv)
        indexes = list(range(1, num_episodes, num_episodes // len(rewards)))
        fig.add_trace(go.Scatter(x=indexes, y=rewards, name=prefix + str_config))


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

    return smooth(np.array(rewards).mean(axis=0))


def smooth(l):
    res = [(l[0] + l[1]) / 2]
    for i in range(1, len(l) - 1):
        res.append((l[i - 1] + l[i] + l[i + 1]) / 3)
    res.append((l[-1] + l[-2]) / 2)

    return res


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
    for i in range(int(index_from), int(index_to) + 1):
        reward = data[data['index'] == i]['reward'].mean()
        rewards.append(reward)

    return rewards


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", default="../DataNeuralODE/Data/experiments2")
    arg_parser.add_argument("--date_from", default="2020-05-01")
    arg_parser.add_argument("--date_to", default="2020-05-29")
    arg_parser.add_argument("--params", action='append')

    args = arg_parser.parse_args(sys.argv[1:])
    path = args.path
    time_from = args.date_from
    time_to = args.date_to
    params = args.params

    if time_from.split('-') == 3:
        time_from = time_from + "_00-00-00"
    if time_to.split('-') == 3:
        time_to = time_to + "_00-00-00"

    Path("results").mkdir(parents=True, exist_ok=True)
    filename = "results/experiments{}".format(len(listdir("results")))

    paths_to_csv_ode, paths_to_csv_common = find_equals_experiments(
        path=path, time_from=time_from, time_to=time_to, params=params
    )
    draw_graph(path_to_csv_common=paths_to_csv_common, path_to_csv_ode=paths_to_csv_ode, filename=filename)