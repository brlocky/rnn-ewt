

from ..core.elliott_wave_base import ElliottWaveBase


class ImpulsiveWave(ElliottWaveBase):
    def validate_pivots(self):
        print('ImpulsiveWave validate_pivots', self.pivots)
        return None

    def analyze(self):
        # Implementation specific to CorrectiveWave
        pass

    def find(self):
        pass

    def validate(self):
        pass

    def add(self):
        pass

    def discard(self):
        pass
