from PyQt6.QtWidgets import (
    QApplication,
    QLabel,
    QRadioButton,
    QButtonGroup,
    QLineEdit,
    QGridLayout,
    QPushButton,
    QWidget,
)
from PyQt6.QtGui import QFont, QIntValidator, QDoubleValidator, QMovie
from PyQt6.QtCore import QThread, pyqtSignal
import sys
import math
from penalties import run_with_args_from_ui


class OptimizationThread(QThread):
    finished = pyqtSignal(object)

    def __init__(self, values):
        super().__init__()
        self.values = values

    def run(self):
        result = run_with_args_from_ui(self.values)
        self.finished.emit(result)


class InterfaceGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Swarm Optimization UI")
        self.setMinimumSize(600, 400)

        font = QFont("JetBrainsMono Nerd Font", 10)
        self.setFont(font)

        self.layout = QGridLayout()
        self.layout.setHorizontalSpacing(40)  # distance between columns
        self.setLayout(self.layout)
        self.columns = 3
        self.positions = [0] * self.columns
        self.current_column = 0

        self.opt_alg = self.labeled_radiobutton(
            "Optimization",
            [("PSO", "PSO"), ("FO-VPPSO", "FO-VPPSO")],
            default="FO-VPPSO",
        )
        self.place_widget(QLabel(""))
        self.terrain_mode = self.labeled_radiobutton(
            "Terrain Mode",
            [("Static", "Static"), ("Random", "Random")],
            default="Static",
        )

        self.place_widget(QLabel(""))

        self.spinner = QLabel()
        self.spinner_movie = QMovie(
            "./ui_images/loading-gif.gif"
        )  # any spinner.gif file
        self.spinner.setMovie(self.spinner_movie)
        self.spinner.setFixedSize(90, 90)
        self.spinner.setScaledContents(True)
        self.spinner.setVisible(False)
        self.place_widget(self.spinner)

        self.next_column()

        self.num_uavs = self.labeled_int_entry("Number of UAVs", "3")
        self.waypoints_per_uav = self.labeled_int_entry("Waypoints per UAV", "4")
        self.max_iter = self.labeled_int_entry("Max iterations", "300")
        self.swarm_size = self.labeled_int_entry("Swarm size", "200")
        self.a_max = self.labeled_float_entry("Max altitude (A_max)", "3.0")
        self.p_co = self.labeled_float_entry("Collision cost (P_co)", "1000.0")
        self.q = self.labeled_float_entry("Terrain cost (Q)", "100.0")

        self.next_column()

        self.epsilon = self.labeled_float_entry("Mountain penalty (ε)", "1000000.0")
        self.d_safe = self.labeled_float_entry("Safe distance (d_safe)", "5.0")
        self.l_min = self.labeled_float_entry("Min leg length (l_min)", "1.0")
        self.l_max = self.labeled_float_entry("Max leg length (l_max)", "30.0")
        self.psi_max = self.labeled_float_entry("Max yaw angle (ψ_max°)", "90.0")
        self.theta_max = self.labeled_float_entry(
            "Max pitch angle (θ_max)", str(round(math.pi / 8, 3))
        )

        self.run_button = QPushButton("Run Optimization")
        self.run_button.clicked.connect(self.run_optimization)

        self.place_widget(QLabel(""))
        self.place_widget(self.run_button)

    def place_widget(self, widget):
        col = self.current_column
        row = self.positions[col]
        self.layout.addWidget(widget, row, col)
        self.positions[col] += 1

    def next_column(self):
        self.current_column = (self.current_column + 1) % self.columns

    def labeled_int_entry(self, label, default):
        self.place_widget(QLabel(label))
        field = QLineEdit()
        field.setText(default)
        field.setValidator(QIntValidator())
        self.place_widget(field)
        return field

    def labeled_float_entry(self, label, default):
        self.place_widget(QLabel(label))
        field = QLineEdit()
        field.setText(default)
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        field.setValidator(validator)
        self.place_widget(field)
        return field

    def labeled_radiobutton(self, label, options, default):
        self.place_widget(QLabel(label))
        group = QButtonGroup(self)
        for text, value in options:
            btn = QRadioButton(text)
            group.addButton(btn)
            btn.setProperty("value", value)
            self.place_widget(btn)
            if value == default:
                btn.setChecked(True)
        return group

    def run_optimization(self):
        self.run_button.setEnabled(False)
        self.spinner_movie.start()
        self.spinner.setVisible(True)

        values = {
            "opt_alg": self.opt_alg.checkedButton().property("value"),
            "terrain_mode": self.terrain_mode.checkedButton().property("value"),
            "num_uavs": int(self.num_uavs.text()),
            "waypoints_per_uav": int(self.waypoints_per_uav.text()),
            "max_iter": int(self.max_iter.text()),
            "swarm_size": int(self.swarm_size.text()),
            "a_max": float(self.a_max.text()),
            "p_co": float(self.p_co.text()),
            "q": float(self.q.text()),
            "epsilon": float(self.epsilon.text()),
            "d_safe": float(self.d_safe.text()),
            "l_min": float(self.l_min.text()),
            "l_max": float(self.l_max.text()),
            "psi_max": math.radians(float(self.psi_max.text())),
            "theta_max": float(self.theta_max.text()),
        }

        self.optim_thread = OptimizationThread(values)
        self.optim_thread.finished.connect(self.optimization_done)
        self.optim_thread.start()

    def optimization_done(self):
        self.run_button.setEnabled(True)
        self.spinner_movie.stop()
        self.spinner.setVisible(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InterfaceGUI()
    window.show()
    sys.exit(app.exec())
