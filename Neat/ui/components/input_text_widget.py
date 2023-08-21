from PyQt6.QtWidgets import QVBoxLayout, QLabel, QLineEdit, QWidget


class InputText(QWidget):
    def __init__(self, label, default_value="", callback=None):
        super().__init__()

        self.label = QLabel(f"{label}:")
        self.line_edit = QLineEdit()
        self.callback = callback

        self.line_edit.setText(default_value)
        self.line_edit.textChanged.connect(self.value_changed)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.line_edit)
        self.setLayout(layout)

    def value_changed(self, text):
        if self.callback:
            self.callback(text)
