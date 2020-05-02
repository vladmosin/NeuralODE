import copy
from itertools import count

import gym
import torch
from torch.nn.functional import mse_loss
from tqdm import tqdm

from constants.DDPGConstants import DDPGConstants
from constants.ExpressiveInverseDDPGConstants import ExpressiveInverseDDPGConstants
from enums.DistanceType import DistanceType
from enums.Exploration import Exploration
from utils.Tester import Tester
from utils.Utils import soft_update_backprop, to_tensor

env = gym.make('MountainCarContinuous-v0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


TEST_EPISODES = 3


def optimize_model():
    if len(memory) < ei_ddpg_config.batch_size:
        return

    states, actions, next_states, rewards, not_dones = memory.sample(ei_ddpg_config.batch_size)

    expected_profit = target_model(state=next_states, action=target_model.best_action(next_states))
    expected_profit = (rewards + not_dones.float() * ei_ddpg_config.gamma * expected_profit)
    expected_profit = expected_profit.detach()

    current_profit = model(states, actions)

    loss = mse_loss(current_profit, expected_profit)
    soft_update_backprop(loss, (model, target_model), optimizer, ei_ddpg_config.tau)


def train():
    for i in tqdm(range(ei_ddpg_config.num_episodes)):
        episode_reward = 0
        state = to_tensor(env.reset(), device=device)

        for t in count():
            action = action_selector.select_action(model=model, state=state, with_eps_greedy=Exploration.ON)
            next_state, reward, done,_ = env.step([action.item()])
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
                #logger.add(tester.test())

            if done:
                break

        ticks_counter.step(DistanceType.BY_EPISODE)
        if ticks_counter.test_time():
            ticks_counter.reset()
            #logger.add(tester.test())
        print(episode_reward)


if __name__ == '__main__':
    ei_ddpg_config = ExpressiveInverseDDPGConstants(device=device, env=env)

    model, target_model, optimizer = ei_ddpg_config.get_models()
    action_selector = ei_ddpg_config.get_action_selector()
    reward_converter = ei_ddpg_config.get_reward_converter()
    memory = ei_ddpg_config.get_memory()

    tester = Tester(action_selector=action_selector,
                    device=device,
                    env=env,
                    model=model,
                    test_episodes=TEST_EPISODES,
                    algorithm='EI_DDPG',
                    distance=5,
                    distance_type=DistanceType.BY_EPISODE,
                    first_test=1,
                    agent_type=ei_ddpg_config.agent_type,
                    exploration=Exploration.OFF)

    #logger = tester.create_csv_logger()
    ticks_counter = tester.create_ticks_counter()
    train()
    #logger.to_csv()
    #logger.to_tensorboard()