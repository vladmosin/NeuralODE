class CartPoleConverter:
    def convert_reward(self, state, reward):
        return reward

    def final_reward(self, reward, iteration):
        return -200 if iteration != 499 else 0
