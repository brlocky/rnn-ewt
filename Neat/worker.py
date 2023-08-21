
# UI imports
from PyQt6.QtCore import QObject, pyqtSignal
from sympy import false

# Custom widgets
from evolution import Evolution


class Worker(QObject):
    finished = pyqtSignal()
    dataSent = pyqtSignal(object)
    progress = pyqtSignal(object)
    console = pyqtSignal(object)
    running = false

    def setup(
        self,
        file_name,
        num_generations: int,
        population_size: int,
        num_epocs: int,
        mutation_factor: float,
        mutation_strength: float,
        mutation_probability: float,
        top_performance: float,
    ):
        self.file_name = file_name
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_epocs = num_epocs
        self.mutation_factor = mutation_factor
        self.mutation_strength = mutation_strength
        self.mutation_probability = mutation_probability
        self.top_performance = top_performance

    def run(self):
        try:
            self.running = True
            self.neat = Evolution(
                self.file_name,
                self.num_generations,
                self.population_size,
                self.num_epocs,
                self.mutation_factor,
                self.mutation_strength,
                self.mutation_probability,
                self.top_performance,
                self.progress,
                self.console,
                self.isAbort
            )

            best_model = self.neat.run()
            self.dataSent.emit(best_model)
            self.finished.emit()
        except Exception as e:
            print("Error during evolution:", e)

    def stop(self):
        self.running = False

    def isAbort(self):
        return not self.running
