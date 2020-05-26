from itertools import count

import argparse
import sys
import gym
import torch
import torch.nn.functional as F
from tqdm import tqdm

from constants.DQNConstants import DQNConstants
from enums.DistanceType import DistanceType
from enums.Exploration import Exploration
from utils.Tester import Tester
from utils.Utils import backprop, to_tensor

from multiprocessing import Process

TEST_EPISODES = 10


class GlobalVariables:
    def __init__(self, memory, dqn_config, policy_net,
                 target_net, logger, optimizer,
                 env, action_selector, reward_converter,
                 ticks_counter, tester, device):
        self.memory = memory
        self.dqn_config = dqn_config
        self.policy_net = policy_net
        self.target_net = target_net
        self.logger = logger
        self.optimizer = optimizer
        self.env = env
        self.action_selector = action_selector
        self.reward_converter = reward_converter
        self.ticks_counter = ticks_counter
        self.tester = tester
        self.device = device


def optimize_model(gv):
    if len(gv.memory) < gv.dqn_config.batch_size:
        return

    states, actions, next_states, rewards, done = gv.memory.sample(gv.dqn_config.batch_size)
    done = done.view(-1)

    profit = gv.policy_net(states).gather(1, actions)
    next_profit = torch.zeros(gv.dqn_config.batch_size, device=gv.device, dtype=torch.float32)
    next_profit[done] = gv.target_net(next_states[done]).max(1)[0].detach()
    expected_reward = (next_profit.unsqueeze(1) * gv.dqn_config.gamma) + rewards

    loss = F.smooth_l1_loss(profit, expected_reward)
    gv.logger.add_loss(loss)
    backprop(loss, gv.policy_net, gv.optimizer)


def train(gv):
    for i in tqdm(range(gv.dqn_config.num_episodes)):
        episode_reward = 0
        state = to_tensor(gv.env.reset(), device=gv.device)

        for t in count():
            action = gv.action_selector.select_action(model=gv.policy_net, state=state, with_eps_greedy=Exploration.ON)
            next_state, reward, done, _ = gv.env.step(action.item())
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

        if i % gv.dqn_config.target_update == 0:
            gv.target_net.load_state_dict(gv.policy_net.state_dict())

        if len(gv.logger.current_losses) > 0:
            gv.logger.add_average_loss()

        gv.ticks_counter.step(DistanceType.BY_EPISODE)
        if gv.ticks_counter.test_time():
            gv.ticks_counter.reset()
            gv.logger.add(gv.tester.test())


def runner(dqn_config, env_name):
    env = gym.make(env_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_env = gym.make(env_name)

    policy_net, target_net, optimizer = dqn_config.get_models()

    memory = dqn_config.get_memory()
    reward_converter = dqn_config.get_reward_converter()

    action_selector = dqn_config.get_action_selector()

    tester = Tester(action_selector=action_selector,
                    device=device,
                    env=test_env,
                    model=policy_net,
                    test_episodes=TEST_EPISODES,
                    algorithm='DQN',
                    distance=20,
                    distance_type=DistanceType.BY_EPISODE,
                    first_test=1,
                    agent_type=dqn_config.agent_type,
                    exploration=Exploration.OFF)

    logger = tester.create_csv_logger(dqn_config)
    ticks_counter = tester.create_ticks_counter()

    gv = GlobalVariables(
        memory=memory, dqn_config=dqn_config,
        policy_net=policy_net, target_net = target_net,
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
    arg_parser.add_argument("--batch_size", type=int, default=256)
    arg_parser.add_argument("--lr", type=float, default=1e-3)
    arg_parser.add_argument("--neuron_number", type=int, default=40)
    arg_parser.add_argument("--num_episodes", type=int, default=1000)
    arg_parser.add_argument("--gamma", type=float, default=0.998)
    arg_parser.add_argument("--memory_size", type=int, default=20000)
    arg_parser.add_argument("--target_update", type=int, default=10)
    arg_parser.add_argument("--eps_decay", type=int, default=1000)
    arg_parser.add_argument("--env_name", default='MountainCar-v0')
    arg_parser.add_argument("--t", type=float, default=1.0)

    return arg_parser


def create_dqn_config(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = gym.make(args.env_name)
    return DQNConstants(
        device=device, env=env,
        batch_size=args.batch_size, lr=args.lr,
        neuron_number=args.neuron_number, num_episodes=args.num_episodes,
        gamma=args.gamma, memory_size=args.memory_size,
        target_update=args.target_update, eps_decay=args.eps_decay,
        t=args.t
    )


def parse_sys_args(sys_args):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--path", default="arguments/1")

    args = arg_parser.parse_args(sys_args)
    return args.path


if __name__ == "__main__":
    path = parse_sys_args(sys.argv[1:])
    f = open(path, "r")
    arguments = f.readlines()
    parser = init_parser()

    processes = []

    for line in arguments[:1]:
        args = parser.parse_args(line.split())
        dqn_config = create_dqn_config(args)

        #process = Process(target=runner, args=(dqn_config, args.env_name))
        #process.start()
        #processes.append(process)

        runner(dqn_config, args.env_name)

    #for process in processes:
        #process.join()