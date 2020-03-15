from itertools import count

import gym
import torch
import torch.nn.functional as F
from tqdm import tqdm

from constants.DQNConstants import DQNConstants
from enums.DistanceType import DistanceType
from enums.Exploration import Exploration
from utils.Tester import Tester
from utils.Utils import backprop, to_tensor

env_name = 'CartPole-v1'


env = gym.make(env_name)
test_env = gym.make(env_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEST_EPISODES = 10


def optimize_model():
    if len(memory) < dqn_config.batch_size:
        return

    states, actions, next_states, rewards, done = memory.sample(dqn_config.batch_size)
    done = done.view(-1)

    profit = policy_net(states).gather(1, actions)
    next_profit = torch.zeros(dqn_config.batch_size, device=device, dtype=torch.float64)
    next_profit[done] = target_net(next_states[done]).max(1)[0].detach()
    expected_reward = (next_profit.unsqueeze(1) * dqn_config.gamma) + rewards

    loss = F.smooth_l1_loss(profit, expected_reward)
    logger.add_loss(loss)
    backprop(loss, policy_net, optimizer)


def train():
    for i in tqdm(range(dqn_config.num_episodes)):
        episode_reward = 0
        state = to_tensor(env.reset(), device=device)

        for t in count():
            action = action_selector.select_action(model=policy_net, state=state, with_eps_greedy=Exploration.ON)
            next_state, reward, done, _ = env.step(action.item())
            episode_reward += reward

            if done:
                reward = reward_converter.final_reward(reward, t)
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

        if i % dqn_config.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if len(logger.current_losses) > 0:
            logger.add_average_loss()

        ticks_counter.step(DistanceType.BY_EPISODE)
        if ticks_counter.test_time():
            ticks_counter.reset()
            logger.add(tester.test())


if __name__ == '__main__':
    action_space_dim = env.action_space.n
    state_space_dim = env.observation_space.shape[0]

    dqn_configs = DQNConstants(device=device, env=env).gen_configs_list()
    for dqn_config in dqn_configs:
        policy_net, target_net, optimizer = dqn_config.get_models()

        memory = dqn_config.get_memory()
        reward_converter = dqn_config.get_reward_converter()

        action_selector = dqn_config.get_action_selector()

        tester = Tester(action_selector=action_selector,
                        device=device,
                        env=env,
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

        train()
        logger.save_all()