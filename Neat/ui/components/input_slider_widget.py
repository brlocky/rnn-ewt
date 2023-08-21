from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QVBoxLayout, QLabel, QSlider, QWidget


class InputSlider(QWidget):
    def __init__(self, label, value, min_value=None, max_value=None, step=1, callback=None):
        super().__init__()

        self.label = QLabel(f"{label}: {value}")  # Initialize label with value
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.callback = callback

        if min_value is not None:
            self.slider.setMinimum(min_value)
        if max_value is not None:
            self.slider.setMaximum(max_value)

        self.slider.setValue(value)
        self.slider.setSingleStep(step)  # Set the step when using arrow keys
        self.slider.setPageStep(step)   # Set the step when dragging the slider

        self.slider.valueChanged.connect(self.value_changed)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def value_changed(self, value):
        if self.callback:
            self.callback(value)

        self.label.setText(f"{self.label.text().split(':')[0]}: {value}")
