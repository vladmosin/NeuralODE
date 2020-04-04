from pathlib import Path

TASKS_PER_RUN = 24


def generate_configs(repetitions=4):
    f = open("settup", "r")
    names, values, lengths = [], [], []
    for line in f.readlines():
        param_info = line.split()
        name, value = param_info[0], param_info[1:]
        names.append(name)
        values.append(value)
        lengths.append(len(value) - 1)

    f.close()

    indexes = [0] * len(names)
    lines = []
    while indexes is not None:
        line = gen_commandline_args(indexes, names, values)
        for _ in range(repetitions):
            lines.append(line)
        indexes = next_elem(indexes, lengths)

    batch_number = (len(lines) + TASKS_PER_RUN - 1) // TASKS_PER_RUN
    for i in range(batch_number):
        gen_run_script(lines, i)


def gen_run_script(lines, batch_index):
    Path("run").mkdir(parents=True, exist_ok=True)
    Path("arguments").mkdir(parents=True, exist_ok=True)
    filename = "arguments/{}".format(batch_index + 1)
    f = open(filename, "w")
    for line in lines[batch_index * TASKS_PER_RUN:(batch_index + 1) * TASKS_PER_RUN]:
        f.write(line + "\n")
    f.close()

    f = open("run/run{}.sh".format(batch_index + 1), "w")
    f.write("#!/bin/bash\n\n")
    f.write("module add singularity hpcx/hpcx-cuda-ompi\n")
    f.write("singularity exec container.sif python NeuralODE/DQN.py --path NeuralODE/" + filename)
    f.close()


def gen_commandline_args(indexes, names, values):
    commandline_args = []
    for i, index in enumerate(indexes):
        commandline_args.append("--{} {}".format(names[i], values[i][index]))

    return " ".join(commandline_args)


def next_elem(indexes, lengths):
    assert len(indexes) == len(lengths)

    if indexes[-1] == lengths[-1]:
        if len(indexes) == 1:
            return None
        prev_indexes = next_elem(indexes[:-1], lengths[:-1])
        if prev_indexes is None:
            return None
        else:
            prev_indexes.append(0)
            return prev_indexes
    else:
        indexes[-1] += 1
        return indexes


if __name__ == "__main__":
    generate_configs()
