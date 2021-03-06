class CartPoleConverter:
    def convert_reward(self, state, reward):
        return reward

    def final_reward(self, reward, iteration):
        return -200 if iteration != 499 else 0

class MountainCarConverter:
    def convert_reward(self, state, reward):
        return reward + 10 * abs(state[0][1])

    def final_reward(self, reward, iteration):
        return reward