from enums.DistanceType import DistanceType


class TicksCounter:
    def __init__(self, type_: DistanceType, steps, start_steps=1):
        self.type_ = type_
        self.steps = steps
        self.left = start_steps

    def step(self, type_: DistanceType):
        if type_ == self.type_:
            self.left -= 1

    def test_time(self):
        return self.left == 0

    def reset(self):
        self.left = self.steps