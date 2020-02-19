from enum import Enum


class DistanceType(Enum):
    BY_EPISODE = 'Episode'
    BY_OPTIMIZER_STEP = 'Optimizer step'
