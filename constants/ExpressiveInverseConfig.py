class ExpressiveInverseConfig:
    def __init__(self, env, transformed_state=400, t=1.0):
        self.transformed_state = transformed_state
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.t = t