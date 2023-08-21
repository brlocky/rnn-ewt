
from abc import ABC, abstractmethod

import torch


class IEvolutionModel(ABC):

    def __init__(self, file_name: str = None, generation=0, id=0, mutation_factor=0.0, mutation_strength=0.0, mutation_probability=0.0):

        self.file_name = file_name
        self.generation = generation
        self.id = id
        self.model = None
        self.mutation_factor = mutation_factor
        self.mutation_strength = mutation_strength
        self.mutation_probability = mutation_probability

        """ open_trade = 0
        capital = 0
        profit = 0
 """
        if self.file_name:
            self.load_model()
        else:
            self.create_model()

    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def create_children(self, generation: int, id: int):
        pass

    @abstractmethod
    def get_data(self):
        pass

    def load_model(self):
        self.model = self.create_model()
        try:
            # Load the state dictionary from a file
            self.model.load_state_dict(torch.load(self.file_name))
            # print(f"Model from file: {self.file_name} ")
        except Exception as e:
            print(f"Error loading model from file: {self.file_name} using new model")

    def save_model(self, file_name):
        torch.save(self.model.state_dict(), file_name)
