
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from evolution_model import EvolutionModel
from i_evolution_mode import IEvolutionModel

from dto.progress_dto import ProgressDto


class Evolution:
    def __init__(
        self,
        model_file,
        num_generations,
        population_size,
        num_epocs,
        mutation_factor,
        mutation_strength,
        mutation_probability,
        top_performance,
        eventEmitter,
        consoleEmitter,
        isAbortPooling
    ):

        # Configuration
        self.num_generations = num_generations
        self.population_size = population_size
        self.num_epocs = num_epocs
        self.mutation_factor = mutation_factor
        self.mutation_strength = mutation_strength
        self.mutation_probability = mutation_probability
        self.top_performance = top_performance

        # Events
        self.eventEmitter = eventEmitter
        self.consoleEmitter = consoleEmitter
        self.isAbortPooling = isAbortPooling

        # Counters
        self.child_count = 0
        self.current_generation = 0
        self.current_population = 0
        self.current_epoch = 0
        self.status = 'waiting'

        # Send Event to updat UI
        self.emitProgress()

        # Assign device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.print(f"Using device: {self.device}")

        # Load initial model
        self.model_file = model_file
        self.emodel = EvolutionModel(
            model_file,
            self.current_generation,
            self.child_count,
            self.mutation_factor,
            self.mutation_strength,
            self.mutation_probability
        )
        # Load model into device
        self.emodel.model.to(self.device)

        self.criterion = nn.MSELoss()  # Use Mean Squared Error for regression
        # self.optimizer = optim.Adam(model.parameters(), lr=self.mutation_factor)

        self.population = []

    # Emit event
    def emitProgress(self):
        data = ProgressDto(
            self.current_generation + 1,
            self.current_population,
            self.current_epoch + 1,
            self.status
        )
        self.eventEmitter.emit(data)

    def print(self, txt: str):
        # print('>> ', txt)
        self.consoleEmitter.emit(txt)

    # Train Model
    def train_model(self, emodel: IEvolutionModel, epocs, X, y):
        model = emodel.model

        # Set model to training mode
        model.train()

        # model.train()  # Set model to training mode
        self.optimizer = optim.Adam(model.parameters(), lr=self.mutation_factor)
        self.current_epoch = 0
        for self.current_epoch in range(epocs):
            self.optimizer.zero_grad()  # clear the gradients
            outputs = model(X)  # compute the model output

            # Ensure gradients are enabled for the backward pass
            with torch.enable_grad():
                loss = self.criterion(outputs, y)  # Calculate loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update model weights

            # Apply gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            self.emitProgress()

            # Thread abort Signals
            if self.isAbortPooling():
                self.num_epocs = 0
                self.status = 'AbortPooling'
                self.emitProgress()
                break

    # Run model Evaluation

    def test_model(self, emodel, X, y):
        model = emodel.model

        # Calculate validation loss
        # Set the model to evaluation mode
        model.eval()
        with torch.no_grad():
            val_outputs = model(X)
            val_loss = self.criterion(val_outputs, y)

        print(f"Val Loss: {val_loss.item():.4f},  {val_loss.item()}")

        # model.eval()  # Set the model to evaluation mode
        return val_loss.item()

    def create_echild(self, emodel: IEvolutionModel, generation: int):
        self.child_count += 1
        echild = emodel.create_children(generation, self.child_count)
        echild.model.to(self.device)
        return echild

    # Create initial population
    def create_initial_population(self, emodel: IEvolutionModel):
        self.population = []
        # Include the parent model in the population
        self.population.append(emodel)
        for _ in range(self.population_size - 1):
            new_child = self.create_echild(self.emodel, 0)

            # Append the child model to the population
            self.population.append(new_child)

    # Create following population
    def create_next_generation_population(self, scores, generation: int):
        # Select top-performing models
        # Ensure at least 1 model is selected
        num_selected = max(1, int(self.top_performance * self.population_size))

        # negative values, closer to 0 is better
        selected_indices = np.argsort(scores)[:num_selected]
        sorted_scores = np.array(scores)[selected_indices]
        # self.print(f"Best Performance Scores: {sorted_scores}")

        # Move top performance models to new population
        new_population = [self.population[i] for i in selected_indices]

        self.print("Top Models")
        for idx, e in enumerate(new_population):
            self.print(
                f"#{e.id} - Gen: {e.generation} - Score: {sorted_scores[idx]}")

        while len(new_population) < self.population_size:
            # for i in range(self.population_size):
            parentIndex = np.random.choice(selected_indices)
            new_population.append(self.create_echild(
                self.population[parentIndex], generation))

        return new_population

    def get_best_model(self):
        # Evolution loop
        for self.current_generation in range(self.num_generations):
            self.status = 'Training'
            self.emitProgress()
            self.print(
                f'{self.status} - {self.current_generation + 1} / {self.num_generations} - {len(self.population)} / {self.population_size}')

            self.current_population = 0
            # Training loop
            for emodel in self.population:
                X, y, X_val, y_val = emodel.get_data(self.device)
                self.train_model(emodel, self.num_epocs, X, y)
                self.current_population += 1
                self.emitProgress()
                if self.num_epocs == 0:
                    break

            if self.num_epocs == 0:
                break

            self.status = 'Testing'
            self.emitProgress()

            # Evaluate fitness for each model
            fitness_scores = [self.test_model(
                emodel, X_val, y_val) for emodel in self.population]

            self.population = self.create_next_generation_population(
                fitness_scores, self.current_generation + 1)
            self.print("-------------------------")

        if self.num_epocs == 0:
            return None

        # Select the best model from the final population
        selected_indices = np.argsort(fitness_scores)
        best_model = self.population[selected_indices[0]]
        return best_model

    def run(self):
        # Initialize the population with random model architectures
        self.create_initial_population(self.emodel)
        best_model = self.get_best_model()

        if best_model is None:
            return None

        self.status = 'Completed'
        self.emitProgress()
        self.print('Best Model found')

        # best_model.save_model(self.model_file)

        return best_model
