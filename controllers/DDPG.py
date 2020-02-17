from itertools import count

import gym
import torch
from torch.optim import Adam
from torch.nn.functional import smooth_l1_loss
from tqdm import tqdm

from Agents import CommonAgent
from ActionSelector import NoisedSelector
from utils.MemoryReplay import MemoryReplay
from utils.RewardConverter import CartPoleConverter


env = gym.make('MountainCarContinuous-v0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------- Constants --------
GAMMA = 0.99
START_EPS = 0.5
END_EPS = 0.002
BATCH_SIZE = 256
ACTOR_LR = 1e-5
CRITIC_LR = 1e-5
# -----------------------------

action_space_dim = env.action_space.n
state_space_dim = env.observation_space.shape[0]


action_selector = NoisedSelector(
    device=device,
    action_space=env.action_space,
    start_eps=START_EPS,
    end_eps=END_EPS,
    eps_decay=
)