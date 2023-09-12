from abc import ABC, abstractmethod
from typing import List
from .fibs import Fibonacci
from .wave_node import Pivot


class ElliottWaveBase(ABC):
    def __init__(self, pivots: List[Pivot]):
        self.pivots = pivots
        self.fibs = Fibonacci()

    @abstractmethod
    def analyze(self):
        pass

    @abstractmethod
    def validate_pivots(self):
        pass

    @abstractmethod
    def find(self):
        pass

    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def add(self):
        pass

    @abstractmethod
    def discard(self):
        pass
