"""
DAREG Configuration GUI

PyQt5 interface for interactive configuration of registration parameters.
Launch with: python -m DAREG.main_motion --gui
"""

import sys
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
        QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
        QMessageBox, QTabWidget, QScrollArea, QFrame, QSizePolicy
    )
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont
    PYQT_VERSION = 5
except ImportError:
    try:
        from PyQt6.QtWidgets import (
            QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
            QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
            QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QFileDialog,
            QMessageBox, QTabWidget, QScrollArea, QFrame, QSizePolicy
        )
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QFont
        PYQT_VERSION = 6
    except ImportError:
        raise ImportError(
            "PyQt5 or PyQt6 is required for the GUI.\n"
            "Install with: pip install PyQt5"
        )

from ..config import (
    load_config, default_config, load_preset, list_available_presets,
    generate_output_name, RegistrationConfig
)


class FilePickerWidget(QWidget):
    """Reusable file picker with label, text field, and browse button"""

    def __init__(self, label: str, file_filter: str = "All Files (*)",
                 is_directory: bool = False, parent=None):
        super().__init__(parent)
        self.file_filter = file_filter
        self.is_directory = is_directory

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel(label)
        self.label.setMinimumWidth(100)
        layout.addWidget(self.label)

        self.line_edit = QLineEdit()
        self.line_edit.setMinimumWidth(300)
        layout.addWidget(self.line_edit, 1)

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse)
        layout.addWidget(self.browse_btn)

    def _browse(self):
        if self.is_directory:
            path = QFileDialog.getExistingDirectory(self, "Select Directory")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", "", self.file_filter)

        if path:
            self.line_edit.setText(path)

    def text(self) -> str:
        return self.line_edit.text()

    def setText(self, text: str):
        self.line_edit.setText(text)


class CollapsibleGroupBox(QGroupBox):
    """Group box that can be collapsed/expanded"""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setCheckable(True)
        self.setChecked(False)
        self.toggled.connect(self._on_toggle)
        self._content_widget = None

    def setContentWidget(self, widget: QWidget):
        self._content_widget = widget
        layout = QVBoxLayout(self)
        layout.addWidget(widget)
        self._on_toggle(self.isChecked())

    def _on_toggle(self, checked: bool):
        if self._content_widget:
            self._content_widget.setVisible(checked)


class ConfigurationGUI(QMainWindow):
    """Main configuration GUI window"""

    def __init__(self, initial_config: Optional[Dict] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DAREG Motion Registration - Configuration")
        self.setMinimumSize(700, 600)

        self._result = None
        self._output_path = None

        # Load default config
        self._config = default_config()

        # Create main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # Create scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content_widget = QWidget()
        self._main_layout = QVBoxLayout(content_widget)
        self._main_layout.setSpacing(10)

        self._create_input_section()
        self._create_output_section()
        self._create_basic_settings_section()
        self._create_advanced_settings_section()
        self._create_button_section()

        scroll.setWidget(content_widget)

        layout = QVBoxLayout(main_widget)
        layout.addWidget(scroll)

        # Populate from initial config if provided
        if initial_config:
            self._populate_from_dict(initial_config)

    def _create_input_section(self):
        """Create input files section"""
        group = QGroupBox("Input Files")
        layout = QVBoxLayout(group)

        self.image4d_picker = FilePickerWidget(
            "4D Image*:",
            "NIfTI Files (*.nii *.nii.gz);;All Files (*)"
        )
        layout.addWidget(self.image4d_picker)

        self.static_picker = FilePickerWidget(
            "Static Image:",
            "NIfTI Files (*.nii *.nii.gz);;All Files (*)"
        )
        layout.addWidget(self.static_picker)

        self.seg_picker = FilePickerWidget(
            "Segmentation:",
            "NIfTI Files (*.nii *.nii.gz);;All Files (*)"
        )
        layout.addWidget(self.seg_picker)

        self._main_layout.addWidget(group)

    def _create_output_section(self):
        """Create output settings section"""
        group = QGroupBox("Output Settings")
        layout = QGridLayout(group)

        self.output_picker = FilePickerWidget("Output Dir:", is_directory=True)
        self.output_picker.setText("./motion_output")
        layout.addWidget(self.output_picker, 0, 0, 1, 3)

        self.auto_name_cb = QCheckBox("Auto-generate output folder name from config")
        layout.addWidget(self.auto_name_cb, 1, 0, 1, 3)

        self._main_layout.addWidget(group)

    def _create_basic_settings_section(self):
        """Create basic registration settings section"""
        group = QGroupBox("Registration Settings")
        layout = QGridLayout(group)

        # Model selection
        layout.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "rigid",
            "rigid+affine",
            "rigid+affine+ffd",
            "rigid+affine+svffd"
        ])
        self.model_combo.setCurrentText("rigid+affine+ffd")
        layout.addWidget(self.model_combo, 0, 1)

        # Device selection
        layout.addWidget(QLabel("Device:"), 0, 2)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda", "mps", "auto"])
        self.device_combo.setCurrentText("cpu")
        layout.addWidget(self.device_combo, 0, 3)

        # Preset selection
        layout.addWidget(QLabel("Preset:"), 1, 0)
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("default")
        for preset in list_available_presets():
            self.preset_combo.addItem(preset)
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        layout.addWidget(self.preset_combo, 1, 1)

        # Load preset button
        self.load_preset_btn = QPushButton("Load Preset")
        self.load_preset_btn.clicked.connect(self._load_selected_preset)
        layout.addWidget(self.load_preset_btn, 1, 2)

        # Frame selection
        layout.addWidget(QLabel("Start Frame:"), 2, 0)
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, 1000)
        self.start_frame_spin.setValue(0)
        layout.addWidget(self.start_frame_spin, 2, 1)

        layout.addWidget(QLabel("Num Frames:"), 2, 2)
        self.num_frames_spin = QSpinBox()
        self.num_frames_spin.setRange(0, 1000)
        self.num_frames_spin.setValue(0)
        self.num_frames_spin.setSpecialValueText("All")
        layout.addWidget(self.num_frames_spin, 2, 3)

        # Pipeline options
        self.skip_alignment_cb = QCheckBox("Skip alignment (static already aligned)")
        layout.addWidget(self.skip_alignment_cb, 3, 0, 1, 2)

        self.skip_refinement_cb = QCheckBox("Skip refinement (faster)")
        layout.addWidget(self.skip_refinement_cb, 3, 2, 1, 2)

        self._main_layout.addWidget(group)

    def _create_advanced_settings_section(self):
        """Create advanced settings section with tabs"""
        group = CollapsibleGroupBox("Advanced Settings")

        content = QWidget()
        content_layout = QVBoxLayout(content)

        tabs = QTabWidget()

        # FFD Tab
        ffd_tab = self._create_ffd_tab()
        tabs.addTab(ffd_tab, "FFD")

        # SVFFD Tab
        svffd_tab = self._create_svffd_tab()
        tabs.addTab(svffd_tab, "SVFFD")

        # Similarity Tab
        similarity_tab = self._create_similarity_tab()
        tabs.addTab(similarity_tab, "Similarity")

        content_layout.addWidget(tabs)
        group.setContentWidget(content)

        self._main_layout.addWidget(group)

    def _create_ffd_tab(self) -> QWidget:
        """Create FFD parameters tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        row = 0
        layout.addWidget(QLabel("Control Point Spacing (mm):"), row, 0)
        self.ffd_spacing_spin = QSpinBox()
        self.ffd_spacing_spin.setRange(1, 20)
        self.ffd_spacing_spin.setValue(4)
        layout.addWidget(self.ffd_spacing_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("Pyramid Levels:"), row, 0)
        self.ffd_levels_spin = QSpinBox()
        self.ffd_levels_spin.setRange(1, 6)
        self.ffd_levels_spin.setValue(4)
        layout.addWidget(self.ffd_levels_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("Iterations per Level:"), row, 0)
        self.ffd_iterations_edit = QLineEdit("100, 100, 100, 100")
        layout.addWidget(self.ffd_iterations_edit, row, 1)

        row += 1
        layout.addWidget(QLabel("Learning Rates:"), row, 0)
        self.ffd_lr_edit = QLineEdit("0.01, 0.01, 0.01, 0.01")
        layout.addWidget(self.ffd_lr_edit, row, 1)

        row += 1
        layout.addWidget(QLabel("Bending Weight:"), row, 0)
        self.ffd_bending_spin = QDoubleSpinBox()
        self.ffd_bending_spin.setRange(0.0, 0.1)
        self.ffd_bending_spin.setDecimals(6)
        self.ffd_bending_spin.setSingleStep(0.0001)
        self.ffd_bending_spin.setValue(0.001)
        layout.addWidget(self.ffd_bending_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("Diffusion Weight:"), row, 0)
        self.ffd_diffusion_spin = QDoubleSpinBox()
        self.ffd_diffusion_spin.setRange(0.0, 0.01)
        self.ffd_diffusion_spin.setDecimals(6)
        self.ffd_diffusion_spin.setSingleStep(0.0001)
        self.ffd_diffusion_spin.setValue(0.0005)
        layout.addWidget(self.ffd_diffusion_spin, row, 1)

        layout.setRowStretch(row + 1, 1)
        return widget

    def _create_svffd_tab(self) -> QWidget:
        """Create SVFFD parameters tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        row = 0
        layout.addWidget(QLabel("Control Point Spacing (mm):"), row, 0)
        self.svffd_spacing_spin = QSpinBox()
        self.svffd_spacing_spin.setRange(1, 20)
        self.svffd_spacing_spin.setValue(4)
        layout.addWidget(self.svffd_spacing_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("Pyramid Levels:"), row, 0)
        self.svffd_levels_spin = QSpinBox()
        self.svffd_levels_spin.setRange(1, 6)
        self.svffd_levels_spin.setValue(4)
        layout.addWidget(self.svffd_levels_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("Iterations per Level:"), row, 0)
        self.svffd_iterations_edit = QLineEdit("100, 100, 100, 100")
        layout.addWidget(self.svffd_iterations_edit, row, 1)

        row += 1
        layout.addWidget(QLabel("Learning Rates:"), row, 0)
        self.svffd_lr_edit = QLineEdit("0.005, 0.005, 0.005, 0.005")
        layout.addWidget(self.svffd_lr_edit, row, 1)

        row += 1
        layout.addWidget(QLabel("Integration Steps:"), row, 0)
        self.svffd_steps_spin = QSpinBox()
        self.svffd_steps_spin.setRange(1, 10)
        self.svffd_steps_spin.setValue(5)
        layout.addWidget(self.svffd_steps_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("Bending Weight:"), row, 0)
        self.svffd_bending_spin = QDoubleSpinBox()
        self.svffd_bending_spin.setRange(0.0, 0.1)
        self.svffd_bending_spin.setDecimals(6)
        self.svffd_bending_spin.setSingleStep(0.0001)
        self.svffd_bending_spin.setValue(0.0005)
        layout.addWidget(self.svffd_bending_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("Diffusion Weight:"), row, 0)
        self.svffd_diffusion_spin = QDoubleSpinBox()
        self.svffd_diffusion_spin.setRange(0.0, 0.01)
        self.svffd_diffusion_spin.setDecimals(6)
        self.svffd_diffusion_spin.setSingleStep(0.0001)
        self.svffd_diffusion_spin.setValue(0.00025)
        layout.addWidget(self.svffd_diffusion_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("Velocity Smoothing (mm):"), row, 0)
        self.svffd_smoothing_spin = QDoubleSpinBox()
        self.svffd_smoothing_spin.setRange(0.0, 5.0)
        self.svffd_smoothing_spin.setDecimals(3)
        self.svffd_smoothing_spin.setSingleStep(0.1)
        self.svffd_smoothing_spin.setValue(0.0)
        layout.addWidget(self.svffd_smoothing_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("Laplacian Weight:"), row, 0)
        self.svffd_laplacian_spin = QDoubleSpinBox()
        self.svffd_laplacian_spin.setRange(0.0, 0.01)
        self.svffd_laplacian_spin.setDecimals(6)
        self.svffd_laplacian_spin.setSingleStep(0.0001)
        self.svffd_laplacian_spin.setValue(0.0)
        layout.addWidget(self.svffd_laplacian_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("Jacobian Penalty:"), row, 0)
        self.svffd_jacobian_spin = QDoubleSpinBox()
        self.svffd_jacobian_spin.setRange(0.0, 0.1)
        self.svffd_jacobian_spin.setDecimals(6)
        self.svffd_jacobian_spin.setSingleStep(0.001)
        self.svffd_jacobian_spin.setValue(0.0)
        layout.addWidget(self.svffd_jacobian_spin, row, 1)

        layout.setRowStretch(row + 1, 1)
        return widget

    def _create_similarity_tab(self) -> QWidget:
        """Create similarity settings tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        row = 0
        layout.addWidget(QLabel("Metric:"), row, 0)
        self.similarity_metric_combo = QComboBox()
        self.similarity_metric_combo.addItems(["nmi", "ncc", "ssd"])
        self.similarity_metric_combo.setCurrentText("nmi")
        layout.addWidget(self.similarity_metric_combo, row, 1)

        row += 1
        layout.addWidget(QLabel("Histogram Bins:"), row, 0)
        self.similarity_bins_spin = QSpinBox()
        self.similarity_bins_spin.setRange(16, 256)
        self.similarity_bins_spin.setValue(64)
        layout.addWidget(self.similarity_bins_spin, row, 1)

        row += 1
        layout.addWidget(QLabel("Foreground Threshold:"), row, 0)
        self.similarity_threshold_spin = QDoubleSpinBox()
        self.similarity_threshold_spin.setRange(0.0, 1.0)
        self.similarity_threshold_spin.setDecimals(4)
        self.similarity_threshold_spin.setSingleStep(0.01)
        self.similarity_threshold_spin.setValue(0.01)
        layout.addWidget(self.similarity_threshold_spin, row, 1)

        layout.setRowStretch(row + 1, 1)
        return widget

    def _create_button_section(self):
        """Create button row"""
        layout = QHBoxLayout()
        layout.addStretch()

        self.save_config_btn = QPushButton("Save Config")
        self.save_config_btn.clicked.connect(self._save_config)
        layout.addWidget(self.save_config_btn)

        self.run_btn = QPushButton("Save && Run")
        self.run_btn.setDefault(True)
        font = self.run_btn.font()
        font.setBold(True)
        self.run_btn.setFont(font)
        self.run_btn.clicked.connect(self._save_and_run)
        layout.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.close)
        layout.addWidget(self.cancel_btn)

        self._main_layout.addLayout(layout)

    def _on_preset_changed(self, preset_name: str):
        """Handle preset selection change"""
        pass  # Just updates combo, actual load is on button click

    def _load_selected_preset(self):
        """Load the selected preset and populate form"""
        preset_name = self.preset_combo.currentText()
        if preset_name == "default":
            config = default_config()
        else:
            try:
                config = load_config(preset=preset_name)
            except FileNotFoundError as e:
                QMessageBox.warning(self, "Preset Error", str(e))
                return

        self._populate_from_config(config)
        QMessageBox.information(self, "Preset Loaded", f"Loaded preset: {preset_name}")

    def _populate_from_config(self, config: RegistrationConfig):
        """Populate form fields from config object"""
        # Pipeline settings
        self.model_combo.setCurrentText(config.pipeline.model)
        self.device_combo.setCurrentText(config.pipeline.device)

        # FFD settings
        self.ffd_spacing_spin.setValue(config.ffd.control_point_spacing)
        self.ffd_levels_spin.setValue(config.ffd.pyramid_levels)
        self.ffd_iterations_edit.setText(
            ", ".join(map(str, config.ffd.iterations_per_level))
        )
        self.ffd_lr_edit.setText(
            ", ".join(map(str, config.ffd.learning_rates_per_level))
        )
        self.ffd_bending_spin.setValue(config.ffd.regularization.bending_weight)
        self.ffd_diffusion_spin.setValue(config.ffd.regularization.diffusion_weight)

        # SVFFD settings
        self.svffd_spacing_spin.setValue(config.svffd.control_point_spacing)
        self.svffd_levels_spin.setValue(config.svffd.pyramid_levels)
        self.svffd_iterations_edit.setText(
            ", ".join(map(str, config.svffd.iterations_per_level))
        )
        self.svffd_lr_edit.setText(
            ", ".join(map(str, config.svffd.learning_rates_per_level))
        )
        self.svffd_steps_spin.setValue(config.svffd.integration_steps)
        self.svffd_bending_spin.setValue(config.svffd.regularization.bending_weight)
        self.svffd_diffusion_spin.setValue(config.svffd.regularization.diffusion_weight)
        self.svffd_smoothing_spin.setValue(config.svffd.regularization.velocity_smoothing_sigma)
        self.svffd_laplacian_spin.setValue(config.svffd.regularization.laplacian_weight)
        self.svffd_jacobian_spin.setValue(config.svffd.regularization.jacobian_penalty)

        # Similarity settings
        self.similarity_metric_combo.setCurrentText(config.similarity.metric)
        self.similarity_bins_spin.setValue(config.similarity.num_bins)
        self.similarity_threshold_spin.setValue(config.similarity.foreground_threshold)

    def _populate_from_dict(self, config_dict: Dict):
        """Populate form from config dictionary (JSON format)"""
        # Input files
        if "inputs" in config_dict:
            inputs = config_dict["inputs"]
            if "image4d" in inputs:
                self.image4d_picker.setText(inputs["image4d"])
            if "static" in inputs:
                self.static_picker.setText(inputs["static"])
            if "seg" in inputs:
                self.seg_picker.setText(inputs["seg"])

        # Output
        if "output" in config_dict:
            output = config_dict["output"]
            if "output_dir" in output:
                self.output_picker.setText(output["output_dir"])

        # Registration model
        if "registration" in config_dict:
            reg = config_dict["registration"]
            if "model" in reg:
                self.model_combo.setCurrentText(reg["model"])

        # Device
        if "device" in config_dict:
            self.device_combo.setCurrentText(config_dict["device"])

        # Frame selection
        if "frame_selection" in config_dict:
            fs = config_dict["frame_selection"]
            if "start_frame" in fs:
                self.start_frame_spin.setValue(fs["start_frame"])
            if "num_frames" in fs and fs["num_frames"] is not None:
                self.num_frames_spin.setValue(fs["num_frames"])

        # FFD parameters
        if "ffd" in config_dict:
            ffd = config_dict["ffd"]
            if "control_point_spacing" in ffd:
                self.ffd_spacing_spin.setValue(ffd["control_point_spacing"])
            if "pyramid_levels" in ffd:
                self.ffd_levels_spin.setValue(ffd["pyramid_levels"])
            if "iterations_per_level" in ffd:
                self.ffd_iterations_edit.setText(
                    ", ".join(map(str, ffd["iterations_per_level"]))
                )
            if "learning_rates_per_level" in ffd:
                self.ffd_lr_edit.setText(
                    ", ".join(map(str, ffd["learning_rates_per_level"]))
                )
            if "regularization" in ffd:
                reg = ffd["regularization"]
                if "bending_weight" in reg:
                    self.ffd_bending_spin.setValue(reg["bending_weight"])
                if "diffusion_weight" in reg:
                    self.ffd_diffusion_spin.setValue(reg["diffusion_weight"])

        # SVFFD parameters
        if "svffd" in config_dict:
            svffd = config_dict["svffd"]
            if "control_point_spacing" in svffd:
                self.svffd_spacing_spin.setValue(svffd["control_point_spacing"])
            if "pyramid_levels" in svffd:
                self.svffd_levels_spin.setValue(svffd["pyramid_levels"])
            if "iterations_per_level" in svffd:
                self.svffd_iterations_edit.setText(
                    ", ".join(map(str, svffd["iterations_per_level"]))
                )
            if "learning_rates_per_level" in svffd:
                self.svffd_lr_edit.setText(
                    ", ".join(map(str, svffd["learning_rates_per_level"]))
                )
            if "integration_steps" in svffd:
                self.svffd_steps_spin.setValue(svffd["integration_steps"])
            if "regularization" in svffd:
                reg = svffd["regularization"]
                if "bending_weight" in reg:
                    self.svffd_bending_spin.setValue(reg["bending_weight"])
                if "diffusion_weight" in reg:
                    self.svffd_diffusion_spin.setValue(reg["diffusion_weight"])
                if "velocity_smoothing_sigma" in reg:
                    self.svffd_smoothing_spin.setValue(reg["velocity_smoothing_sigma"])
                if "laplacian_weight" in reg:
                    self.svffd_laplacian_spin.setValue(reg["laplacian_weight"])
                if "jacobian_penalty" in reg:
                    self.svffd_jacobian_spin.setValue(reg["jacobian_penalty"])

    def _parse_list(self, text: str, dtype=int):
        """Parse comma-separated list"""
        try:
            return [dtype(x.strip()) for x in text.split(",") if x.strip()]
        except ValueError:
            return []

    def build_config_dict(self) -> Dict[str, Any]:
        """Build config dictionary from form values"""
        config = {
            "inputs": {
                "image4d": self.image4d_picker.text(),
            },
            "output": {
                "output_dir": self.output_picker.text(),
            },
            "registration": {
                "model": self.model_combo.currentText(),
            },
            "frame_selection": {
                "start_frame": self.start_frame_spin.value(),
            },
            "device": self.device_combo.currentText(),
            "similarity": {
                "metric": self.similarity_metric_combo.currentText(),
                "num_bins": self.similarity_bins_spin.value(),
                "foreground_threshold": self.similarity_threshold_spin.value(),
            },
            "ffd": {
                "control_point_spacing": self.ffd_spacing_spin.value(),
                "pyramid_levels": self.ffd_levels_spin.value(),
                "iterations_per_level": self._parse_list(self.ffd_iterations_edit.text()),
                "learning_rates_per_level": self._parse_list(self.ffd_lr_edit.text(), float),
                "regularization": {
                    "bending_weight": self.ffd_bending_spin.value(),
                    "diffusion_weight": self.ffd_diffusion_spin.value(),
                },
            },
            "svffd": {
                "control_point_spacing": self.svffd_spacing_spin.value(),
                "pyramid_levels": self.svffd_levels_spin.value(),
                "iterations_per_level": self._parse_list(self.svffd_iterations_edit.text()),
                "learning_rates_per_level": self._parse_list(self.svffd_lr_edit.text(), float),
                "integration_steps": self.svffd_steps_spin.value(),
                "regularization": {
                    "bending_weight": self.svffd_bending_spin.value(),
                    "diffusion_weight": self.svffd_diffusion_spin.value(),
                    "velocity_smoothing_sigma": self.svffd_smoothing_spin.value(),
                    "laplacian_weight": self.svffd_laplacian_spin.value(),
                    "jacobian_penalty": self.svffd_jacobian_spin.value(),
                },
            },
        }

        # Optional fields
        if self.static_picker.text():
            config["inputs"]["static"] = self.static_picker.text()
        if self.seg_picker.text():
            config["inputs"]["seg"] = self.seg_picker.text()

        num_frames = self.num_frames_spin.value()
        if num_frames > 0:
            config["frame_selection"]["num_frames"] = num_frames

        if self.skip_alignment_cb.isChecked():
            config["registration"]["skip_alignment"] = True
        if self.skip_refinement_cb.isChecked():
            config["registration"]["skip_refinement"] = True

        return config

    def validate(self) -> Tuple[bool, str]:
        """Validate form inputs"""
        errors = []

        # Required: 4D image
        if not self.image4d_picker.text():
            errors.append("4D Image is required")
        elif not Path(self.image4d_picker.text()).exists():
            errors.append(f"4D Image file not found: {self.image4d_picker.text()}")

        # Optional files: check if they exist when specified
        if self.static_picker.text() and not Path(self.static_picker.text()).exists():
            errors.append(f"Static image file not found: {self.static_picker.text()}")

        if self.seg_picker.text() and not Path(self.seg_picker.text()).exists():
            errors.append(f"Segmentation file not found: {self.seg_picker.text()}")

        # Output directory
        if not self.output_picker.text():
            errors.append("Output directory is required")

        # Validate list formats
        ffd_iters = self._parse_list(self.ffd_iterations_edit.text())
        if not ffd_iters:
            errors.append("Invalid FFD iterations format (use: 100, 100, 100, 100)")

        ffd_lrs = self._parse_list(self.ffd_lr_edit.text(), float)
        if not ffd_lrs:
            errors.append("Invalid FFD learning rates format (use: 0.01, 0.01, 0.01, 0.01)")

        svffd_iters = self._parse_list(self.svffd_iterations_edit.text())
        if not svffd_iters:
            errors.append("Invalid SVFFD iterations format")

        svffd_lrs = self._parse_list(self.svffd_lr_edit.text(), float)
        if not svffd_lrs:
            errors.append("Invalid SVFFD learning rates format")

        if errors:
            return False, "\n".join(errors)
        return True, ""

    def _save_config(self):
        """Save config to JSON file without running"""
        config_dict = self.build_config_dict()

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Configuration",
            "registration_config.json",
            "JSON Files (*.json);;All Files (*)"
        )

        if path:
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
            QMessageBox.information(self, "Saved", f"Configuration saved to:\n{path}")

    def _save_and_run(self):
        """Validate, save config, and signal to run pipeline"""
        valid, error_msg = self.validate()
        if not valid:
            QMessageBox.critical(self, "Validation Error", error_msg)
            return

        config_dict = self.build_config_dict()
        self._result = config_dict

        # Determine output path
        output_dir = self.output_picker.text()
        if self.auto_name_cb.isChecked():
            # Generate auto name
            model = self.model_combo.currentText()
            ffd_type = "svffd" if "svffd" in model else "ffd"
            if ffd_type == "svffd":
                spacing = self.svffd_spacing_spin.value()
                bending = self.svffd_bending_spin.value()
                steps = self.svffd_steps_spin.value()
                auto_name = f"svffd_cp{spacing}_bend{bending}_steps{steps}"
            else:
                spacing = self.ffd_spacing_spin.value()
                bending = self.ffd_bending_spin.value()
                auto_name = f"ffd_cp{spacing}_bend{bending}"
            self._output_path = Path(f"./motion_output_{auto_name}")
        else:
            self._output_path = Path(output_dir)

        # Save config to output directory
        self._output_path.mkdir(parents=True, exist_ok=True)
        config_file = self._output_path / "gui_config.json"
        with open(config_file, "w") as f:
            json.dump(config_dict, f, indent=2)

        self.close()

    def get_result(self) -> Optional[Dict]:
        """Get the result config dict (None if cancelled)"""
        return self._result

    def get_output_path(self) -> Optional[Path]:
        """Get the determined output path"""
        return self._output_path


def launch_config_gui(initial_config: Optional[Dict] = None) -> Tuple[Optional[Dict], Optional[Path]]:
    """
    Launch configuration GUI and return config when user clicks Save & Run.

    Args:
        initial_config: Optional initial config dict to populate form

    Returns:
        Tuple of (config_dict, output_path) or (None, None) if cancelled
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    gui = ConfigurationGUI(initial_config=initial_config)
    gui.show()
    app.exec_() if PYQT_VERSION == 5 else app.exec()

    return gui.get_result(), gui.get_output_path()


if __name__ == "__main__":
    # Test GUI standalone
    config, output = launch_config_gui()
    if config:
        print(f"Config: {json.dumps(config, indent=2)}")
        print(f"Output path: {output}")
    else:
        print("Cancelled")
