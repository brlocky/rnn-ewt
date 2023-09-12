import pandas as pd
from typing import List
from .elliott_wave_base import ElliottWaveBase
from ..waves import CorrectiveWave, ImpulsiveWave, Wave1
from .wave_node import Pivot, WaveNode


class WaveValidator:
    def __init__(self, pivots: List[Pivot]):
        self.pivots = pivots

    def validate(self):
        args = {
            'pivots': self.pivots,
        }
        self.waves: List[ElliottWaveBase] = [
            # CorrectiveWave(**args),
            # ImpulsiveWave(**args),
            Wave1(**args),
        ]

        waves: List[WaveNode] = []
        for wave in self.waves:
            result = wave.validate_pivots()
            if result is not None:
                waves.append(result)

        return waves
