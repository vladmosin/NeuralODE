from itertools import count

import gym
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from agents.CommonAgent import CommonAgent
from agents.ODEAgent import ODEAgent
from loggers.CSVLogger import CSVLogger
from loggers.TicksCounter import TicksCounter
from utils.ActionSelector import ActionSelector
from utils.MemoryReplay import MemoryReplay
from utils.RewardConverter import CartPoleConverter

from enums.DistanceType import DistanceType
from enums.Exploration import Exploration


env_name = 'CartPole-v1'


env = gym.make(env_name)
test_env = gym.make(env_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------- Constants --------
START_EPS = 0.9
END_EPS = 0.05
EPS_DECAY = 2000

MEMORY_SIZE = 10000
LR = 0.0003
BATCH_SIZE = 128
GAMMA = 0.99
NEURON_NUMBER = 32

NUM_EPISODES = 1000
TARGET_UPDATE = 10
TEST_EPISODES = 10
# -----------------------------


def backprop(loss: torch.Tensor):
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    states, actions, next_states, rewards, done = memory.sample(BATCH_SIZE)
    done = done.view(-1)

    profit = policy_net(states).gather(1, actions)
    next_profit = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float64)
    next_profit[done] = target_net(next_states[done]).max(1)[0].detach()
    expected_reward = (next_profit.unsqueeze(1) * GAMMA) + rewards

    loss = F.smooth_l1_loss(profit, expected_reward)
    backprop(loss)


reward_converter = CartPoleConverter()
test_rewards = []


def train():
    for i in tqdm(range(NUM_EPISODES)):
        episode_reward = 0
        state = to_tensor(env.reset())

        for t in count():
            action = action_selector.select_action(model=policy_net, state=state, with_eps_greedy=Exploration.ON)
            next_state, reward, done, _ = env.step(action.item())
            episode_reward += reward

            if done:
                reward = reward_converter.final_reward(reward, t)
            else:
                reward = reward_converter.convert_reward(state, reward)

            next_state = to_tensor(next_state)
            memory.push(state, action, next_state, reward, done)
            state = next_state
            optimize_model()

            ticks_counter.step(DistanceType.BY_OPTIMIZER_STEP)
            if ticks_counter.test_time():
                ticks_counter.reset()
                logger.add(test(EXPLORATION))

            if done:
                break

        if i % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        ticks_counter.step(DistanceType.BY_EPISODE)
        if ticks_counter.test_time():
            ticks_counter.reset()
            logger.add(test(EXPLORATION))


def to_tensor(x):
    return torch.tensor([x], device=device, dtype=torch.float64)


def test(with_eps_greedy):
    rewards = []
    for i in range(TEST_EPISODES):
        episode_reward = 0
        state = to_tensor(test_env.reset())

        for t in count():
            action = action_selector.select_action(
                model=policy_net, state=state, with_eps_greedy=with_eps_greedy)
            next_state, reward, done, _ = test_env.step(action.item())
            episode_reward += reward

            state = to_tensor(next_state)
            if done:
                break

        rewards.append(episode_reward)

    print(rewards)

    return rewards


# -------- Log constants ------ #
AGENT_TYPE = 'ODE'  # Common, ODE, BlockODE
ALGORITHM = 'DQN'  # DQN, DDPG
DISTANCE = 20
DISTANCE_TYPE = DistanceType.BY_EPISODE  # Episode, OptimizeStep
EXPLORATION = Exploration.OFF  # Exploration, Exploit
FIRST_TEST = 1  # First test model after this number of updates
#################################


def create_csv_logger():
    return CSVLogger(agent_type=AGENT_TYPE,
                     algorithm=ALGORITHM,
                     distance=DISTANCE,
                     distance_type=DISTANCE_TYPE,
                     exploration=EXPLORATION,
                     start_distance=FIRST_TEST)


def create_ticks_counter():
    return TicksCounter(steps=DISTANCE, type_=DISTANCE_TYPE, start_steps=FIRST_TEST)


if __name__ == '__main__':
    action_space_dim = env.action_space.n
    state_space_dim = env.observation_space.shape[0]

    logger = create_csv_logger()
    ticks_counter = create_ticks_counter()

    policy_net = ODEAgent(
        device=device,
        neuron_number=NEURON_NUMBER,
        input_dim=state_space_dim,
        output_dim=action_space_dim
    )

    target_net = ODEAgent(
        device=device,
        neuron_number=NEURON_NUMBER,
        input_dim=state_space_dim,
        output_dim=action_space_dim
    )

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = MemoryReplay(device=device, capacity=MEMORY_SIZE)

    action_selector = ActionSelector(
        device=device,
        action_space=env.action_space,
        end_eps=END_EPS,
        start_eps=START_EPS,
        eps_decay=EPS_DECAY
    )

    train()
    logger.to_csv()
