from itertools import count

import gym
import torch
from torch.optim import Adam
from torch.nn.functional import smooth_l1_loss
from tqdm import tqdm
import numpy as np

from Agents import CommonAgent
from ActionSelector import ActionSelector
from utils.MemoryReplay import MemoryReplay
from utils.RewardConverter import CartPoleConverter


env = gym.make('CartPole-v1')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ------- Constants --------
START_EPS = 0.9
END_EPS = 0.05
EPS_DECAY = 2000

MEMORY_SIZE = 10000
LR = 0.003
BATCH_SIZE = 128
GAMMA = 0.99
NEURON_NUMBER = 32

NUM_EPISODES = 1000
TEST_FREQUENCY = 50
TARGET_UPDATE = 10

# -----------------------------
action_space_dim = env.action_space.n
state_space_dim = env.observation_space.shape[0]


action_selector = ActionSelector(
    device=device,
    action_space=env.action_space,
    end_eps=END_EPS,
    start_eps=START_EPS,
    eps_decay=EPS_DECAY
)

memory = MemoryReplay(
    device=device,
    capacity=MEMORY_SIZE
)


policy_net = CommonAgent(
    device=device,
    neuron_number=NEURON_NUMBER,
    input_dim=state_space_dim,
    output_dim=action_space_dim
)

target_net = CommonAgent(
    device=device,
    neuron_number=NEURON_NUMBER,
    input_dim=state_space_dim,
    output_dim=action_space_dim
)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = Adam(policy_net.parameters(), lr=LR)


def backprop(loss: torch.Tensor):
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    states, actions, next_states, rewards, not_done = memory.sample(BATCH_SIZE)
    current_profit = policy_net(states).gather(1, actions)

    not_done = not_done.view(-1)

    next_profit = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float64)

    next_profit[not_done] = policy_net(next_states[not_done]).max(1)[0].detach()

    expected_profit = next_profit.view(-1, 1) * GAMMA + rewards
    loss = smooth_l1_loss(current_profit, expected_profit)
    print(loss)

    backprop(loss)


reward_converter = CartPoleConverter()
test_rewards = []
for i in range(NUM_EPISODES):
    episode_reward = 0
    state = torch.tensor([env.reset()], device=device, dtype=torch.float64)
    is_train_episode = i % TEST_FREQUENCY != 0

    for t in count():
        action = action_selector.select_action(model=policy_net, state=state, with_eps_greedy=is_train_episode)
        next_state, reward, done, _ = env.step(action.item())
        episode_reward += reward

        if done:
            print(t)
            reward = reward_converter.final_reward(reward, t)
        else:
            reward = reward_converter.convert_reward(state, reward)

        next_state = torch.tensor([next_state], device=device, dtype=torch.float64)
        memory.push(state, action, next_state, reward, done)
        state = next_state
        optimize_model()

        if done:
            break

    if i % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if not is_train_episode:
        print("Test episode {} out of {}: {}".format(
            i // TEST_FREQUENCY, NUM_EPISODES // TEST_FREQUENCY, int(episode_reward)
        ))
