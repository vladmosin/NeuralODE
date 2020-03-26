from pathlib import Path


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

    for i, line in enumerate(lines):
        gen_run_script(line, i)


def gen_run_script(line, index):
    Path("run").mkdir(parents=True, exist_ok=True)
    f = open("run/run{}.sh".format(index + 1), "w")
    f.write("#!/bin/bash\n\n")
    f.write("module add singularity hpcx/hpcx-cuda-ompi\n")
    f.write("singularity exec container.sif python DQN.py " + line)
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
