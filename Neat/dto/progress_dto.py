class ProgressDto:
    def __init__(self, generation: int, population: int, epoc: int, status: str):
        self.generation: int = generation
        self.population: int = population
        self.epoc: int = epoc
        self.status: str = status
