import copy
from itertools import count

import gym
import torch
from torch.nn.functional import mse_loss
from tqdm import tqdm

from constants.DDPGConstants import DDPGConstants
from enums.DistanceType import DistanceType
from enums.Exploration import Exploration
from envs.spiral import SpiralEnv
from utils.Tester import Tester
from utils.Utils import soft_update_backprop, to_tensor, to_numpy

env = SpiralEnv()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


TEST_EPISODES = 3


def optimize_model():
    if len(memory) < ddpg_config.batch_size:
        return

    states, actions, next_states, rewards, not_dones = memory.sample(ddpg_config.batch_size)

    expected_profit = target_critic(next_states, target_actor(next_states))
    expected_profit = (rewards + not_dones.float() * ddpg_config.gamma * expected_profit)
    expected_profit = expected_profit.detach()

    current_profit = critic(states, actions)

    critic_loss = mse_loss(current_profit, expected_profit)
    soft_update_backprop(critic_loss, (critic, target_critic), critic_optimizer, ddpg_config.tau)

    actor_loss = - critic(states, actor(states)).mean()
    soft_update_backprop(actor_loss, (actor, target_actor), actor_optimizer, ddpg_config.tau)


def train():
    for i in tqdm(range(ddpg_config.num_episodes)):
        episode_reward = 0
        state = to_tensor(env.reset(), device=device)

        for t in count():
            action = action_selector.select_action(model=actor, state=state, with_eps_greedy=Exploration.ON)
            next_state, reward, done, _ = env.step(to_numpy(action)[0])
            episode_reward += reward

            if done:
                reward = reward_converter.convert_reward(reward_converter, t)
            else:
                reward = reward_converter.convert_reward(state, reward)

            next_state = to_tensor(next_state, device=device)
            memory.push(state, action, next_state, reward, done)
            state = next_state
            optimize_model()

            ticks_counter.step(DistanceType.BY_OPTIMIZER_STEP)
            if ticks_counter.test_time():
                ticks_counter.reset()
                logger.add(tester.test())

            if done:
                break

        ticks_counter.step(DistanceType.BY_EPISODE)
        if ticks_counter.test_time():
            ticks_counter.reset()
            logger.add(tester.test())
        print(episode_reward)


if __name__ == '__main__':
    ddpg_config = DDPGConstants(device=device, env=env)

    actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer = ddpg_config.get_models()
    action_selector = ddpg_config.get_action_selector()
    reward_converter = ddpg_config.get_reward_converter()
    memory = ddpg_config.get_memory()

    tester = Tester(action_selector=action_selector,
                    device=device,
                    env=env,
                    model=actor,
                    test_episodes=TEST_EPISODES,
                    algorithm='DQN',
                    distance=5,
                    distance_type=DistanceType.BY_EPISODE,
                    first_test=1,
                    agent_type=ddpg_config.agent_type,
                    exploration=Exploration.OFF)

    logger = tester.create_csv_logger(ddpg_config)
    ticks_counter = tester.create_ticks_counter()
    train()
    logger.to_csv()
    logger.to_tensorboard()