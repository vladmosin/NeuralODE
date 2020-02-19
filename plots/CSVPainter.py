import pandas as pd
import matplotlib.pyplot as plt


def draw_plot(path):
    data = pd.read_csv(path)[['reward', 'distance']]
    print(data.groupby('distance').mean())
    data.groupby('distance').mean().plot()
    plt.show()


draw_plot('../data/csv/DQN/Common/Exploit/Episode/20/2020-02-19_00-41.csv')
