
from ui.components.input_slider_widget import InputSlider
from ui.components.input_text_widget import InputText
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QPushButton, QHBoxLayout, QVBoxLayout)


def add_inputs(self, layout):
    # Model Name input

    model_name_layout = QVBoxLayout()
    self.input_text_widget = InputText(
        label="Model Name", default_value=self.model_name, callback=self.update_name)
    model_name_layout.addWidget(self.input_text_widget)

    model_buttons_layout = QHBoxLayout()
    # self.model_load_button = QPushButton("Load")
    # self.model_load_button.clicked.connect(self.load_model)
    # model_buttons_layout.addWidget(self.model_load_button)

    self.model_save_button = QPushButton("Save")
    self.model_save_button.clicked.connect(self.save_model)
    self.model_save_button.setEnabled(False)
    model_buttons_layout.addWidget(self.model_save_button)

    # Add the buttons layout to the name layout
    model_name_layout.addLayout(model_buttons_layout)

    layout.addLayout(model_name_layout)

    # Population Size
    generation_input = InputSlider("Generations", self.num_generations,
                                   min_value=1, max_value=100, callback=self.update_num_generations_size)
    layout.addWidget(generation_input)

    # Population Size
    population_input = InputSlider("Population", self.population_size,
                                   min_value=1, max_value=100, callback=self.update_population_size)
    layout.addWidget(population_input)

    # Number of Epochs
    num_epochs_input = InputSlider("Epochs", self.num_epochs,
                                   min_value=25, max_value=10000, step=25, callback=self.update_num_epochs)
    layout.addWidget(num_epochs_input)

    # Mutation Factor
    mut_factor_input = InputText(
        label="Mutation Factor", default_value=str(self.mutation_factor), callback=self.update_mutation_factor)
    layout.addWidget(mut_factor_input)

    # Mutation Strength
    mutation_strength_input = InputText(
        label="Mutation Strength", default_value=str(self.mutation_strength), callback=self.update_mutation_strength)
    layout.addWidget(mutation_strength_input)

    # Mutation Probability
    mut_probability_input = InputText(
        label="Mutation Probability", default_value=str(self.mutation_probability), callback=self.update_mutation_probability)
    layout.addWidget(mut_probability_input)

    # Top Performance
    top_performance_input = InputText(
        label="Top Performance", default_value=str(self.top_performance), callback=self.update_top_performance)
    layout.addWidget(top_performance_input)

    # Buttons section
    buttons_layout = QHBoxLayout()
    self.run_button = QPushButton("Run Evolution")
    self.run_button.clicked.connect(self.run_neat)
    buttons_layout.addWidget(self.run_button)

    self.stop_button = QPushButton("Stop")
    self.stop_button.clicked.connect(self.stop_neat)
    buttons_layout.addWidget(self.stop_button)

    layout.addLayout(buttons_layout)
