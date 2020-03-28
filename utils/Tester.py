from itertools import count

from enums.AgentType import AgentType
from enums.DistanceType import DistanceType
from enums.Exploration import Exploration
from loggers.Logger import Logger
from loggers.TicksCounter import TicksCounter
from utils.Utils import to_tensor


class Tester:
    def __init__(self, env,
                 test_episodes, device,
                 action_selector, model,
                 algorithm='DQN', agent_type=AgentType.Common,
                 distance=20, distance_type=DistanceType.BY_EPISODE,
                 exploration=Exploration.OFF, first_test=1):

        self.env = env
        self.device = device
        self.test_episodes = test_episodes
        self.action_selector = action_selector
        self.model = model
        self.exploration = exploration
        self.algorithm = algorithm
        self.agent_type = agent_type
        self.distance=distance
        self.distance_type = distance_type
        self.first_test = first_test

    def create_csv_logger(self, config):
        return Logger(config=config, tester=self)

    def create_ticks_counter(self):
        return TicksCounter(steps=self.distance,
                            type_=self.distance_type,
                            start_steps=self.first_test)

    def test(self):
        rewards = []
        for i in range(self.test_episodes):
            episode_reward = 0
            state = to_tensor(self.env.reset(), device=self.device)

            for t in count():
                action = self.action_selector.select_action(
                    model=self.model, state=state, with_eps_greedy=self.exploration)
                next_state, reward, done, _ = self.env.step(action.item())
                episode_reward += reward

                state = to_tensor(next_state, self.device)
                if done:
                    break

            rewards.append(episode_reward)

        #print(rewards)

        return rewards

    def __str__(self):
        return "\nTest config\n" + \
                "env={}\n".format(self.env.unwrapped.spec.id) + \
                "test_episodes={}\n".format(self.test_episodes) + \
                "exploration={}\n".format(self.exploration.value) + \
                "algorithm={}\n".format(self.algorithm) + \
                "agent_type={}\n".format(self.agent_type.value) + \
                "distance={}\n".format(self.distance) + \
                "distance_type={}\n".format(self.distance_type.value) + \
                "first_test={}\n".format(self.first_test)