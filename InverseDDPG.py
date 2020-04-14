from itertools import count

import argparse
import sys
import gym
import torch
import torch.nn.functional as F
from tqdm import tqdm

from constants.DQNConstants import DQNConstants
from constants.InverseDDPGConstants import InverseDDPGConstants
from enums.DistanceType import DistanceType
from enums.Exploration import Exploration
from utils.Tester import Tester
from utils.Utils import backprop, to_tensor, soft_update_backprop

from multiprocessing import Process

TEST_EPISODES = 3


class GlobalVariables:
    def __init__(self, memory, config: InverseDDPGConstants, net,
                 target_net, logger, optimizer,
                 env, action_selector, reward_converter,
                 ticks_counter, tester, device):
        self.memory = memory
        self.config = config
        self.net = net
        self.target_net = target_net
        self.logger = logger
        self.optimizer = optimizer
        self.env = env
        self.action_selector = action_selector
        self.reward_converter = reward_converter
        self.ticks_counter = ticks_counter
        self.tester = tester
        self.device = device


def optimize_model(gv: GlobalVariables):
    if len(gv.memory) < gv.config.batch_size:
        return

    states, actions, next_states, rewards, not_done = gv.memory.sample(gv.config.batch_size)
    not_done = not_done.view(-1)

    expected_profit = gv.target_net(states=next_states, actions=gv.target_net.find_best_action(states))
    expected_profit = (rewards + not_done.double() * gv.config.gamma * expected_profit)
    expected_profit = expected_profit.detach()

    profit = gv.net(states=states, actions=gv.net.find_best_action())

    loss = F.mse_loss(profit, expected_profit)
    soft_update_backprop(loss, (gv.net, gv.target_net), gv.optimizer, gv.config.tau)


def train(gv: GlobalVariables):
    for i in tqdm(range(gv.config.num_episodes)):
        episode_reward = 0
        state = to_tensor(gv.env.reset(), device=gv.device)

        for t in count():
            action = gv.action_selector.select_action(model=gv.net, state=state, with_eps_greedy=Exploration.ON)
            next_state, reward, done, _ = gv.env.step([action.item()])
            episode_reward += reward

            if done:
                reward = gv.reward_converter.final_reward(reward, t)
            else:
                reward = gv.reward_converter.convert_reward(state, reward)

            next_state = to_tensor(next_state, device=gv.device)
            gv.memory.push(state, action, next_state, reward, done)
            state = next_state
            optimize_model(gv)

            gv.ticks_counter.step(DistanceType.BY_OPTIMIZER_STEP)
            if gv.ticks_counter.test_time():
                gv.ticks_counter.reset()
                gv.logger.add(gv.tester.test())

            if done:
                break

        if len(gv.logger.current_losses) > 0:
            gv.logger.add_average_loss()

        gv.ticks_counter.step(DistanceType.BY_EPISODE)
        if gv.ticks_counter.test_time():
            gv.ticks_counter.reset()
            gv.logger.add(gv.tester.test())


def runner(config: InverseDDPGConstants, env_name):
    env = gym.make(env_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_env = gym.make(env_name)

    net, target_net, optimizer = config.get_models()

    memory = config.get_memory()
    reward_converter = config.get_reward_converter()

    action_selector = config.get_action_selector()

    tester = Tester(action_selector=action_selector,
                    device=device,
                    env=test_env,
                    model=net,
                    test_episodes=TEST_EPISODES,
                    algorithm='InverseDDPG',
                    distance=10,
                    distance_type=DistanceType.BY_EPISODE,
                    first_test=1,
                    exploration=Exploration.OFF)

    logger = tester.create_csv_logger(config)
    ticks_counter = tester.create_ticks_counter()

    gv = GlobalVariables(
        memory=memory, config=config,
        net=net, target_net=target_net,
        logger=logger, optimizer=optimizer,
        env=env, action_selector=action_selector,
        reward_converter=reward_converter,
        ticks_counter=ticks_counter, tester=tester,
        device=device
    )

    train(gv)
    logger.save_all()


def init_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--batch_size", type=int, default=64)
    arg_parser.add_argument("--lr", type=float, default=1e-3)
    arg_parser.add_argument("--neuron_number", type=int, default=64)
    arg_parser.add_argument("--num_episodes", type=int, default=1000)
    arg_parser.add_argument("--gamma", type=float, default=0.999)
    arg_parser.add_argument("--memory_size", type=int, default=20000)
    arg_parser.add_argument("--target_update", type=int, default=5)
    arg_parser.add_argument("--eps_decay", type=int, default=1000)
    arg_parser.add_argument("--env_name", default='CartPole-v1')

    return arg_parser


def create_dqn_config(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make(args.env_name)
    return DQNConstants(
        device=device, env=env,
        batch_size=args.batch_size, lr=args.lr,
        neuron_number=args.neuron_number, num_episodes=args.num_episodes,
        gamma=args.gamma, memory_size=args.memory_size,
        target_update=args.target_update, eps_decay=args.eps_decay
    )


def parse_sys_args(sys_args):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", default="arguments/1")

    args = arg_parser.parse_args(sys_args)
    return args.path


if __name__ == "__main__":
    env = gym.make('MountainCarContinuous-v0')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = InverseDDPGConstants(env=env, device=device)

    runner(config=config, env_name='MountainCarContinuous-v0')
