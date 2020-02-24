import gym
import torch
from torch.nn.functional import mse_loss

from constants.DDPGConstants import DDPGConstants
from utils.Utils import soft_update_backprop

env = gym.make('MountainCarContinuous-v0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------- Constants --------
GAMMA = 0.99
TAU = 0.05
BATCH_SIZE = 256
ACTOR_LR = 1e-5
CRITIC_LR = 1e-5
MAX_EPISODES = 35
MAX_TIMESTAMPS = 500
SIGMA = 0.6
START_EPS = 0.1
END_EPS = 1
EPS_DECAY = 2000

CAPACITY = 50000
NEURON_NUMBER = 100
# -----------------------------


def ddpg_update():
    states, actions, rewards, next_states, not_dones = memory.sample()

    expected_profit = critic(next_states, actor(next_states))
    expected_profit = (rewards + not_dones * ddpg_config.gamma * expected_profit).detach()
    current_profit = critic(states, actions)

    critic_loss = mse_loss(current_profit, expected_profit)
    soft_update_backprop(critic_loss, critic, critic_optimizer, ddpg_config.tau)

    actor_loss = - critic(states, actor(states)).mean()
    soft_update_backprop(actor_loss, actor, actor_optimizer, ddpg_config.tau)


if __name__ == '__main__':
    ddpg_config = DDPGConstants(device=device, env=env)

    actor, critic, actor_optimizer, critic_optimizer = ddpg_config.get_models()
    action_selector = ddpg_config.get_action_selector()
    reward_converter = ddpg_config.get_reward_converter()
    memory = ddpg_config.get_memory()

