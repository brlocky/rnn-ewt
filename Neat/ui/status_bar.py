from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


def status_bar2(self):
    self.status_bar = self.statusBar()

    # Add labels to the status bar
    self.status_bar_state = QLabel("")
    self.status_bar.addPermanentWidget(self.status_bar_state)

    self.status_bar.addPermanentWidget(QLabel(" | "))

    self.status_bar_label1 = QLabel("")
    self.status_bar.addPermanentWidget(self.status_bar_label1)

    self.status_bar.addPermanentWidget(QLabel(" | "))

    self.status_bar_label2 = QLabel("")
    self.status_bar.addPermanentWidget(self.status_bar_label2)

    self.status_bar.addPermanentWidget(QLabel(" | "))

    self.status_bar_label3 = QLabel("")
    self.status_bar.addPermanentWidget(self.status_bar_label3)


def status_bar(self):
    self.status_bar = self.statusBar()

    def create_metric(label, value=0):
        vbox = QVBoxLayout()
        vbox.addWidget(QLabel(label))
        valueLabel = QLabel(str(value))  # Initialize with a default value
        valueLabel.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        vbox.addWidget(valueLabel)

        widget = QWidget()
        widget.setLayout(vbox)
        return widget, valueLabel

    widget, self.status_bar_state = create_metric('Status')
    self.status_bar.addPermanentWidget(widget)

    widget, self.status_bar_label1 = create_metric('Gen')
    self.status_bar.addPermanentWidget(widget)

    widget, self.status_bar_label2 = create_metric('Pop')
    self.status_bar.addPermanentWidget(widget)

    widget, self.status_bar_label3 = create_metric('Epochs')
    self.status_bar.addPermanentWidget(widget)
