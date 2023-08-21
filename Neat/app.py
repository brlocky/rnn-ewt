import sys
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QTextCursor
from PyQt6.QtWidgets import (QApplication, QMainWindow,
                             QFrame, QSplitter, QTextEdit, QVBoxLayout, QWidget)
from evolution_model import EvolutionModel
from i_evolution_mode import IEvolutionModel
from dto.progress_dto import ProgressDto
from ui.status_bar import status_bar

from ui.inputs import add_inputs
from worker import Worker


class MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Settings
        self.model_name = 'model.pth'
        self.num_generations = 3
        self.population_size = 10
        self.num_epochs = 250
        self.mutation_factor = 0.0001
        self.mutation_strength = 0.00001
        self.mutation_probability = 0.2
        self.top_performance = 0.5

        self.console = []

        # Control Variables
        self.running = False
        self.loaded_model: IEvolutionModel() = None

        # UI
        self.setWindowTitle("Conda Environment Deleter")

        self.setGeometry(400, 200, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Create a splitter to divide the UI into sections
        splitter = QSplitter()
        main_layout.addWidget(splitter)

        # Set initial sizes for the sections
        splitter.setSizes([100, 500])

        # Left section (Input section)
        input_frame = QFrame(splitter)
        input_layout = QVBoxLayout(input_frame)

        add_inputs(self, input_layout)

        # Right section (Console)
        console_frame = QFrame(splitter)
        console_layout = QVBoxLayout(console_frame)
        self.console_widget = QTextEdit()
        console_layout.addWidget(self.console_widget)

        # Bottom section (Statu Bar)
        status_bar(self)

        self.show()

    def update_name(self, n: str):
        self.model_name = n

    def update_population_size(self, n: int):
        self.population_size = n

    def update_num_generations_size(self, n: int):
        self.num_generations = n

    def update_num_epochs(self, n: int):
        self.num_epochs = n

    def update_mutation_factor(self, n: str):
        self.mutation_factor = float(n)

    def update_mutation_strength(self, n: str):
        self.mutation_strength = float(n)

    def update_mutation_probability(self, n: str):
        self.mutation_probability = float(n)

    def update_top_performance(self, n: str):
        self.top_performance = float(n)

    def update_status_bar(self, currentProgress):
        self.status_bar_state.setText(f"{currentProgress.status}")
        self.status_bar_label1.setText(
            f"{currentProgress.generation}/{self.num_generations}")
        self.status_bar_label2.setText(
            f"{currentProgress.population}/{self.population_size}")
        self.status_bar_label3.setText(f"{currentProgress.epoc}/{self.num_epochs}")

    def report_progress(self, progress: ProgressDto):
        self.update_status_bar(progress)

    def print_console(self, txt: str):
        self.console.append(txt)

        self.console_widget.clear()
        self.console_widget.setPlainText('\n'.join(self.console))

        cursor = self.console_widget.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.console_widget.setTextCursor(cursor)

    def training_completed(self, res):

        self.running = False
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.model_save_button.setEnabled(False)

        if isinstance(res, IEvolutionModel):
            self.loaded_model = res
            self.model_save_button.setEnabled(True)
        else:
            self.loaded_model = None

    def save_model(self):
        print('Saving model')
        if self.loaded_model:
            self.loaded_model.save_model(self.model_name)

    def stop_neat(self):
        self.worker.stop()

    def run_neat(self):
        # Clear Console
        self.console = []
        self.console_widget.setPlainText('Loading...')

        self.thread = QThread()

        self.worker = Worker()
        self.worker.setup(self.model_name,
                          self.num_generations,
                          self.population_size,
                          self.num_epochs,
                          self.mutation_factor,
                          self.mutation_strength,
                          self.mutation_probability,
                          self.top_performance,
                          )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.dataSent.connect(self.training_completed)
        self.worker.progress.connect(self.report_progress)
        self.worker.console.connect(self.print_console)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)

        self.thread.start()

        self.lock_ui()

    def lock_ui(self):
        self.running = True
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.model_save_button.setEnabled(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
