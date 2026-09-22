"""Digital signal analyzer GUI (BPSK / QPSK / 16-QAM).

    python demod_analyzer.py [capture.cs8|.cs16|...]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

import dsp

I_COLOR = (80, 175, 255)
Q_COLOR = (255, 150, 60)
MARK_COLOR = (255, 60, 110)
CONST_COLOR = (80, 220, 160)

FORMAT_LABELS = {
    "int8": "IQ8 signed (cs8)",
    "uint8": "IQ8 unsigned (cu8, RTL-SDR)",
    "int16": "IQ16 signed LE (cs16)",
}
MARKER_MODES = ["Dots", "Vertical lines", "Dots + lines", "None"]


def _spin(lo, hi, val, step=1, suffix=""):
    w = QtWidgets.QSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    w.setSingleStep(step)
    w.setSuffix(suffix)
    w.setAccelerated(True)
    return w


def _dspin(lo, hi, val, step, decimals, suffix=""):
    w = QtWidgets.QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setValue(val)
    w.setSingleStep(step)
    w.setSuffix(suffix)
    w.setAccelerated(True)
    return w


def file_info_html(f: dsp.IQFile, fs: float) -> str:
    lsb = f.scale
    return (
        f"{f.n_samples:,} samples, {f.n_samples / fs:.6g} s<br>"
        f"RMS {f.rms_dbfs:.2f} dBFS (AC {f.rms_ac_dbfs:.2f} dBFS)<br>"
        f"<b>DC offset</b>: I {f.dc.real:+.5f}, Q {f.dc.imag:+.5f} FS<br>"
        f"&nbsp;&nbsp;= I {f.dc.real * lsb:+.2f}, Q {f.dc.imag * lsb:+.2f} LSB, "
        f"{f.dc_dbfs:.1f} dBFS")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Demod Analyzer")
        self.resize(1500, 950)
        self.settings = QtCore.QSettings("demod-analyzer", "demod-analyzer")

        self.f: dsp.IQFile | None = None
        self.res: dsp.DemodResult | None = None
        self._syncing = False
        self._reset_iq_view = True
        self._last_os = 1
        self._lp_sps_follows = True

        self._demod_timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self._demod_timer.timeout.connect(self.update_demod)
        self._power_timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self._power_timer.timeout.connect(self.update_power)

        self._build_plots()
        self._build_controls()
        self._build_menu()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.controls_scroll)
        splitter.addWidget(self.plots)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1080])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Open an IQ file (File > Open, Ctrl+O)")

    # ------------------------------------------------------------------ UI build

    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        act = m.addAction("&Open…")
        act.setShortcut(QtGui.QKeySequence.Open)
        act.triggered.connect(self.open_dialog)
        m.addSeparator()
        q = m.addAction("&Quit")
        q.setShortcut(QtGui.QKeySequence.Quit)
        q.triggered.connect(self.close)

    def _build_controls(self):
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(panel)

        # --- file
        g = QtWidgets.QGroupBox("File")
        fl = QtWidgets.QFormLayout(g)
        self.open_btn = QtWidgets.QPushButton("Open…")
        self.open_btn.clicked.connect(self.open_dialog)
        self.path_lbl = QtWidgets.QLabel("—")
        self.path_lbl.setWordWrap(True)
        fl.addRow(self.open_btn, self.path_lbl)
        self.fmt_combo = QtWidgets.QComboBox()
        for k, v in FORMAT_LABELS.items():
            self.fmt_combo.addItem(v, k)
        self.fmt_combo.currentIndexChanged.connect(self._format_changed)
        fl.addRow("Format", self.fmt_combo)
        self.fs_spin = _dspin(1, 1e10, float(self.settings.value("fs", 1e6)), 1000, 0, " Hz")
        self.fs_spin.setGroupSeparatorShown(True)
        self.fs_spin.valueChanged.connect(self._fs_changed)
        fl.addRow("Sample rate", self.fs_spin)
        self.info_lbl = QtWidgets.QLabel("")
        self.info_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        fl.addRow(self.info_lbl)
        self.dc_chk = QtWidgets.QCheckBox("Remove DC offset")
        self.dc_chk.setChecked(True)
        self.dc_chk.toggled.connect(self._framing_changed)
        fl.addRow(self.dc_chk)
        lay.addWidget(g)

        # --- framing
        g = QtWidgets.QGroupBox("Framing")
        fl = QtWidgets.QFormLayout(g)
        self.skip_spin = _spin(0, 2_000_000_000, 0, 1, " samples")
        self.skip_spin.valueChanged.connect(self._framing_changed)
        fl.addRow("Skip first", self.skip_spin)
        self.sps_spin = _spin(1, 100_000, int(self.settings.value("sps", 8)))
        self.sps_spin.valueChanged.connect(self._framing_changed)
        fl.addRow("Samples / symbol", self.sps_spin)
        self.from_spin = _spin(0, 0, 0)
        self.to_spin = _spin(0, 0, 0)
        self.from_spin.valueChanged.connect(self._range_changed)
        self.to_spin.valueChanged.connect(self._range_changed)
        fl.addRow("Symbol from", self.from_spin)
        fl.addRow("Symbol to (excl.)", self.to_spin)
        self.nsym_lbl = QtWidgets.QLabel("")
        fl.addRow(self.nsym_lbl)
        lay.addWidget(g)

        # --- demodulation
        g = QtWidgets.QGroupBox("Demodulation")
        fl = QtWidgets.QFormLayout(g)

        self.mod_combo = QtWidgets.QComboBox()
        for key, m in dsp.MODULATIONS.items():
            self.mod_combo.addItem(m.label, key)
        i = self.mod_combo.findData(str(self.settings.value("modulation", "bpsk")))
        self.mod_combo.setCurrentIndex(max(i, 0))
        self.mod_combo.setToolTip(
            "Constellation used by the Auto estimators, the metrics and the bit slicer.\n"
            "Apart from the blind equalizer, which it drives, it does not change the\n"
            "signal processing chain itself.")
        self.mod_combo.currentIndexChanged.connect(self._modulation_changed)
        fl.addRow("Modulation", self.mod_combo)

        self.freq_spin = _spin(-500_000, 500_000, 0, 1, " Hz")
        self.freq_spin.setToolTip(
            "Samples are multiplied by exp(j·2π·f·n/fs).\n"
            "To cancel a carrier offset of +X Hz, enter −X.")
        self.freq_spin.valueChanged.connect(self.schedule_demod)
        b = QtWidgets.QPushButton("Auto")
        self.auto_freq_btn = b
        b.setToolTip("Estimate the carrier offset over the selected range")
        b.clicked.connect(self.auto_freq)
        fl.addRow("Freq shift", self._hbox(self.freq_spin, b))
        self.carrier_lbl = QtWidgets.QLabel()
        self.carrier_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.carrier_lbl.setToolTip("Carrier frequency of the original signal relative to the "
                                    "capture centre (= −freq shift)")
        self.freq_spin.valueChanged.connect(self._update_carrier_label)
        fl.addRow("", self.carrier_lbl)
        self._update_carrier_label()

        self.phase_spin = _dspin(-180, 180, 0, 0.5, 1, "°")
        self.phase_spin.setWrapping(True)
        self.phase_spin.setMinimumWidth(66)
        self.phase_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.phase_slider.setRange(-1800, 1800)
        self.phase_spin.valueChanged.connect(self._phase_spin_changed)
        self.phase_slider.valueChanged.connect(
            lambda v: self._sync_set(self.phase_spin, v / 10.0))
        b = QtWidgets.QPushButton("Auto")
        self.auto_phase_btn = b
        b.setToolTip("Rotate the constellation onto the ideal one")
        b.clicked.connect(self.auto_phase)
        b90 = QtWidgets.QPushButton("+90°")
        b90.setToolTip("Add a quarter turn, to step through the phase ambiguity")
        b90.setFixedWidth(b90.fontMetrics().horizontalAdvance("+90°") + 20)
        b90.clicked.connect(self.rotate_90)
        fl.addRow("Phase", self._hbox(self.phase_spin, b, b90))
        fl.addRow("", self.phase_slider)

        self.os_combo = QtWidgets.QComboBox()
        self.os_combo.addItem("Off (1×)", 1)
        self.os_combo.addItem("4×", 4)
        self.os_combo.currentIndexChanged.connect(self._os_changed)
        fl.addRow("Oversampling", self.os_combo)

        self.filt_combo = QtWidgets.QComboBox()
        self.filt_combo.addItem("None", "none")
        self.filt_combo.addItem("RRC (root raised cosine)", "rrc")
        self.filt_combo.addItem("RC (raised cosine)", "rc")
        self.filt_combo.addItem("Low-pass (Kaiser sinc)", "lp")
        self.filt_combo.currentIndexChanged.connect(self._filter_changed)
        fl.addRow("Matched filter", self.filt_combo)
        self.beta_spin = _dspin(0.01, 1.0, 0.35, 0.05, 2)
        self.beta_spin.valueChanged.connect(self.schedule_demod)
        self.beta_spin.setToolTip("Excess bandwidth of the RC / RRC pulse")
        fl.addRow("RC/RRC roll-off β", self.beta_spin)
        self.span_spin = _spin(1, 64, 8, 1, " sym")
        self.span_spin.valueChanged.connect(self.schedule_demod)
        self.span_spin.setToolTip("Total length of the RC / RRC pulse in symbols")
        fl.addRow("RC/RRC span", self.span_spin)

        self.lp_sps_spin = _spin(2, 100_000, self.sps_spin.value())
        self.lp_sps_spin.setToolTip(
            "Samples per symbol of the input the cutoff is derived from: the filter is the\n"
            "anti-aliasing low-pass for decimating to 1 sample/symbol, so the cutoff is\n"
            "fs / (2·N) — half the symbol rate at the default. Follows Samples / symbol\n"
            "until you change it.")
        self.lp_sps_spin.valueChanged.connect(self._lp_sps_edited)
        fl.addRow("LP samples / symbol", self.lp_sps_spin)
        self.lp_delay_spin = _spin(1, 64, 4, 1, " sym")
        self.lp_delay_spin.setToolTip("Group delay / half span; the filter spans 2× this many symbols")
        self.lp_delay_spin.valueChanged.connect(self.schedule_demod)
        fl.addRow("LP delay", self.lp_delay_spin)
        self.lp_beta_spin = _dspin(0.0, 20.0, 5.0, 0.5, 1)
        self.lp_beta_spin.setToolTip("Kaiser window shape: 0 = no window (max ringing), higher = deeper stopband")
        self.lp_beta_spin.valueChanged.connect(self.schedule_demod)
        fl.addRow("LP Kaiser β", self.lp_beta_spin)

        self.sp_spin = _spin(0, 7, 0)
        self.sp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sp_slider.setRange(0, 7)
        self.sp_spin.valueChanged.connect(self._sp_spin_changed)
        self.sp_slider.valueChanged.connect(lambda v: self._sync_set(self.sp_spin, v))
        b = QtWidgets.QPushButton("Auto")
        b.setToolTip("Place the sampling point half a symbol away from the transitions")
        b.clicked.connect(self.auto_timing)
        fl.addRow("Sampling point", self._hbox(self.sp_spin, b))
        fl.addRow("", self.sp_slider)

        b = QtWidgets.QPushButton("Auto: frequency → timing → phase")
        b.clicked.connect(self.auto_all)
        fl.addRow(b)

        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Sunken)
        fl.addRow(line)
        self.eq_chk = QtWidgets.QCheckBox("Blind equalizer (CMA)")
        self.eq_chk.setToolTip(
            "Adaptive symbol-spaced equalizer on the sampled symbols only — one tap per\n"
            "symbol, not per sample, so the I/Q trace and the eye diagram are unchanged.\n"
            "Blind: it is driven by the constant-modulus cost, not by a known preamble.")
        self.eq_chk.toggled.connect(self._eq_changed)
        fl.addRow(self.eq_chk)
        self.eq_mu_spin = _dspin(1e-5, 1.0, 0.01, 0.005, 5)
        self.eq_mu_spin.setToolTip(
            "Adaptation step. Normalized by the signal level and the tap energy, so it\n"
            "means the same at any amplitude: small = slow but steady, large = fast but\n"
            "noisy, and too large diverges (the equalizer then reports it and passes the\n"
            "symbols through).")
        self.eq_mu_spin.valueChanged.connect(self.schedule_demod)
        fl.addRow("EQ learning rate", self.eq_mu_spin)
        self.eq_taps_spin = _spin(1, 129, 9, 2, " sym")
        self.eq_taps_spin.setToolTip("Number of taps, one per symbol — the span of the ISI it can undo")
        self.eq_taps_spin.valueChanged.connect(self._eq_taps_changed)
        fl.addRow("EQ filter length", self.eq_taps_spin)
        self.eq_look_spin = _spin(0, 128, 2, 1, " sym")
        self.eq_look_spin.setToolTip(
            "How many of the taps sit on future symbols, making the filter non-causal.\n"
            "Needed for precursor ISI (ringing that arrives before the symbol); the rest\n"
            "of the taps cover the current and past symbols. At most filter length − 1.")
        self.eq_look_spin.valueChanged.connect(self.schedule_demod)
        fl.addRow("EQ lookahead", self.eq_look_spin)
        self.eq_lbl = QtWidgets.QLabel()
        self.eq_lbl.setWordWrap(True)
        fl.addRow("", self.eq_lbl)
        lay.addWidget(g)

        # --- display
        g = QtWidgets.QGroupBox("Display")
        fl = QtWidgets.QFormLayout(g)
        self.marker_combo = QtWidgets.QComboBox()
        self.marker_combo.addItems(MARKER_MODES)
        self.marker_combo.currentIndexChanged.connect(self.redraw)
        fl.addRow("Sampling marks", self.marker_combo)
        self.const_time_chk = QtWidgets.QCheckBox("Color constellation by time")
        self.const_time_chk.toggled.connect(self.redraw)
        fl.addRow(self.const_time_chk)
        self.eye_mode_combo = QtWidgets.QComboBox()
        self.eye_mode_combo.addItems(["Lines", "Density"])
        self.eye_mode_combo.currentIndexChanged.connect(self.redraw)
        fl.addRow("Eye mode", self.eye_mode_combo)
        self.eye_width_spin = _spin(1, 4, 1, 1, " sym")
        self.eye_width_spin.valueChanged.connect(self.redraw)
        fl.addRow("Eye width", self.eye_width_spin)
        self.eye_max_spin = _spin(10, 200_000, 1000, 100)
        self.eye_max_spin.valueChanged.connect(self.redraw)
        fl.addRow("Eye max traces", self.eye_max_spin)
        self.eye_q_chk = QtWidgets.QCheckBox("Show Q in eye (lines mode)")
        self.eye_q_chk.setChecked(True)
        self.eye_q_chk.toggled.connect(self.redraw)
        fl.addRow(self.eye_q_chk)
        lay.addWidget(g)

        # --- results
        g = QtWidgets.QGroupBox("Symbols")
        vl = QtWidgets.QVBoxLayout(g)
        self.metrics_lbl = QtWidgets.QLabel("")
        self.metrics_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        vl.addWidget(self.metrics_lbl)
        self.diff_chk = QtWidgets.QCheckBox("Differential")
        self.inv_chk = QtWidgets.QCheckBox("Invert")
        self.diff_chk.toggled.connect(self._update_bits)
        self.inv_chk.toggled.connect(self._update_bits)
        copy_btn = QtWidgets.QPushButton("Copy bits")
        copy_btn.clicked.connect(self._copy_bits)
        vl.addLayout(self._hbox(self.diff_chk, self.inv_chk, copy_btn, stretch=False))
        self.bits_edit = QtWidgets.QPlainTextEdit()
        self.bits_edit.setReadOnly(True)
        self.bits_edit.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        self.bits_edit.setMinimumHeight(120)
        vl.addWidget(self.bits_edit)
        lay.addWidget(g)
        lay.addStretch(1)

        self.controls_scroll = QtWidgets.QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setWidget(panel)
        self.controls_scroll.setMinimumWidth(380)
        self._filter_changed()
        self._eq_taps_changed()
        self._modulation_changed()

    @staticmethod
    def _hbox(*widgets, stretch=True):
        h = QtWidgets.QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        for i, w in enumerate(widgets):
            h.addWidget(w, 1 if (stretch and i == 0) else 0)
        if not stretch:
            h.addStretch(1)
        return h

    def _build_plots(self):
        pg.setConfigOptions(antialias=False)
        self.plots = QtWidgets.QSplitter(QtCore.Qt.Vertical)

        # symbol power
        self.pw_plot = pg.PlotWidget(title="Symbol power (RMS over each SPS window)")
        self.pw_plot.setLabel("bottom", "symbol index")
        self.pw_plot.setLabel("left", "dBFS")
        self.pw_plot.showGrid(x=True, y=True, alpha=0.3)
        self.pw_curve = self.pw_plot.plot(pen=pg.mkPen((200, 200, 200), width=1))
        self.pw_curve.setDownsampling(auto=True, method="peak")
        self.pw_curve.setClipToView(True)
        self.region = pg.LinearRegionItem(brush=(80, 175, 255, 40))
        self.region.setZValue(10)
        self.region.sigRegionChangeFinished.connect(self._region_changed)
        self.pw_plot.addItem(self.region)
        self.plots.addWidget(self.pw_plot)

        # I/Q vs sample
        self.iq_plot = pg.PlotWidget(title="I / Q (frequency shifted, phase rotated, filtered)")
        self.iq_plot.setLabel("bottom", "sample (input rate, after skip)")
        self.iq_plot.setLabel("left", "amplitude (FS)")
        self.iq_plot.showGrid(x=True, y=True, alpha=0.3)
        self.iq_plot.addLegend(offset=(10, 10))
        self.i_curve = self.iq_plot.plot(pen=pg.mkPen(I_COLOR, width=1), name="I")
        self.q_curve = self.iq_plot.plot(pen=pg.mkPen(Q_COLOR, width=1), name="Q")
        for c in (self.i_curve, self.q_curve):
            c.setClipToView(True)
            c.setDownsampling(auto=True, method="peak")
        self.vlines = pg.PlotDataItem(pen=pg.mkPen(MARK_COLOR + (110,), width=1), connect="pairs")
        self.vlines.setClipToView(True)
        self.iq_plot.addItem(self.vlines)
        self.i_marks = pg.PlotDataItem(pen=None, symbol="o", symbolSize=5,
                                       symbolBrush=MARK_COLOR, symbolPen=None, name="sampling point")
        self.q_marks = pg.PlotDataItem(pen=None, symbol="o", symbolSize=4,
                                       symbolBrush=MARK_COLOR + (150,), symbolPen=None)
        for c in (self.i_marks, self.q_marks):
            c.setClipToView(True)
            self.iq_plot.addItem(c)
        self.plots.addWidget(self.iq_plot)

        bottom = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        # constellation
        self.const_plot = pg.PlotWidget(title="Constellation (at sampling point)")
        self.const_plot.setAspectLocked(True)
        self.const_plot.showGrid(x=True, y=True, alpha=0.3)
        self.const_plot.setLabel("bottom", "I")
        self.const_plot.setLabel("left", "Q")
        for ang in (0, 90):
            self.const_plot.addItem(pg.InfiniteLine(angle=ang, pen=pg.mkPen((120, 120, 120), width=1)))
        self.const_ref = pg.ScatterPlotItem(size=11, symbol="x", brush=None,
                                            pen=pg.mkPen((235, 235, 235, 150), width=1))
        self.const_ref.setZValue(5)
        self.const_scatter = pg.ScatterPlotItem(size=4, pen=None, brush=CONST_COLOR + (140,))
        self.const_plot.addItem(self.const_scatter)
        self.const_plot.addItem(self.const_ref)
        bottom.addWidget(self.const_plot)

        # eye
        self.eye_plot = pg.PlotWidget(title="Eye diagram (sampling point at center)")
        self.eye_plot.setLabel("bottom", "offset from sampling point (oversampled samples)")
        self.eye_plot.setLabel("left", "amplitude (FS)")
        self.eye_plot.showGrid(x=True, y=True, alpha=0.3)
        self.eye_img = pg.ImageItem()
        self.eye_img.setColorMap(pg.colormap.get("inferno"))
        self.eye_plot.addItem(self.eye_img)
        self.eye_q = self.eye_plot.plot(pen=pg.mkPen(Q_COLOR + (60,), width=1), connect="finite")
        self.eye_i = self.eye_plot.plot(pen=pg.mkPen(I_COLOR + (90,), width=1), connect="finite")
        self.eye_center = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen(MARK_COLOR, width=1))
        self.eye_plot.addItem(self.eye_center)
        bottom.addWidget(self.eye_plot)
        bottom.setSizes([500, 700])
        self.plots.addWidget(bottom)
        self.plots.setSizes([200, 330, 420])
        for p in (self.pw_plot, self.iq_plot, self.const_plot, self.eye_plot):
            for ax in ("left", "bottom"):
                p.getAxis(ax).enableAutoSIPrefix(False)

    # ------------------------------------------------------------------ file

    def open_dialog(self):
        start = str(self.settings.value("last_dir", os.getcwd()))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open IQ file", start,
            "IQ files (*.cs8 *.cu8 *.cs16 *.iq8 *.iq16 *.sc8 *.sc16 *.c8 *.c16 *.raw *.bin *.iq *.dat);;All files (*)")
        if path:
            self.settings.setValue("last_dir", os.path.dirname(path))
            self.load_file(path)

    def load_file(self, path: str, fmt: str | None = None):
        fmt = fmt or dsp.guess_format(path)
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            f = dsp.IQFile(path, fmt)
        except Exception as e:  # noqa: BLE001 - show any load error to the user
            QtWidgets.QMessageBox.critical(self, "Cannot open file", f"{path}\n\n{e}")
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.f = f
        self.fmt_combo.blockSignals(True)
        self.fmt_combo.setCurrentIndex(self.fmt_combo.findData(fmt))
        self.fmt_combo.blockSignals(False)
        self.path_lbl.setText(os.path.basename(path))
        self.path_lbl.setToolTip(path)
        self.setWindowTitle(f"Demod Analyzer — {os.path.basename(path)}")
        self.skip_spin.setMaximum(max(0, f.n_samples - 1))
        self._update_info()
        self._framing_changed(first_load=True)

    def _format_changed(self):
        if self.f is not None:
            self.load_file(self.f.path, self.fmt_combo.currentData())

    def _update_info(self):
        if self.f is not None:
            self.info_lbl.setText(file_info_html(self.f, self.fs_spin.value()))

    def _fs_changed(self):
        fs = self.fs_spin.value()
        self.settings.setValue("fs", fs)
        lim = int(min(fs / 2, 2_000_000_000))
        self.freq_spin.setRange(-lim, lim)
        self._update_info()
        self._update_nsym_label()
        self.schedule_demod()

    # ------------------------------------------------------------------ framing

    def _n_sym_total(self) -> int:
        if self.f is None:
            return 0
        return max(0, (self.f.n_samples - self.skip_spin.value()) // self.sps_spin.value())

    def _framing_changed(self, *_, first_load=False):
        self.settings.setValue("sps", self.sps_spin.value())
        if self._lp_sps_follows:
            self._syncing = True
            self.lp_sps_spin.setValue(max(2, self.sps_spin.value()))
            self._syncing = False
        n = self._n_sym_total()
        self._syncing = True
        self.from_spin.setMaximum(max(0, n - 1))
        self.to_spin.setMaximum(n)
        if first_load:
            self.from_spin.setValue(0)
            self.to_spin.setValue(min(n, 2000))
        self._syncing = False
        self._update_sp_range()
        self._update_nsym_label()
        self._reset_iq_view = True
        self._power_timer.start()
        self.schedule_demod()

    def _update_nsym_label(self):
        n = self._n_sym_total()
        sel = max(0, self.to_spin.value() - self.from_spin.value())
        rate = self.fs_spin.value() / self.sps_spin.value()
        self.nsym_lbl.setText(f"{n:,} symbols in file · {sel:,} selected<br>symbol rate {rate:,.1f} Bd")

    def _range_changed(self):
        if self._syncing:
            return
        self._sync_region()
        self._update_nsym_label()
        self._reset_iq_view = True
        self.schedule_demod()

    def _sync_region(self):
        self._syncing = True
        self.region.setRegion((self.from_spin.value(), self.to_spin.value()))
        self._syncing = False

    def _region_changed(self):
        if self._syncing:
            return
        lo, hi = self.region.getRegion()
        n = self._n_sym_total()
        lo = int(round(min(max(lo, 0), n)))
        hi = int(round(min(max(hi, 0), n)))
        self._syncing = True
        self.from_spin.setValue(lo)
        self.to_spin.setValue(max(hi, lo + 1))
        self._syncing = False
        self._range_changed()

    def update_power(self):
        if self.f is None:
            return
        t0 = time.perf_counter()
        pw = dsp.symbol_power_db(self.f, self.skip_spin.value(), self.sps_spin.value(),
                                 self.dc_chk.isChecked())
        self.pw_curve.setData(np.arange(len(pw)), pw)
        self.region.setBounds((0, len(pw)))
        self._sync_region()
        self.pw_plot.enableAutoRange()
        self.statusBar().showMessage(f"Symbol power: {len(pw):,} symbols in {time.perf_counter() - t0:.2f} s", 4000)

    # ------------------------------------------------------------------ demod params

    def _sync_set(self, widget, value):
        if self._syncing:
            return
        widget.setValue(value)

    def mod(self) -> dsp.Modulation:
        return dsp.MODULATIONS[self.mod_combo.currentData()]

    def _modulation_changed(self, *_):
        m = self.mod()
        self.settings.setValue("modulation", m.name)
        self.auto_freq_btn.setToolTip(
            f"Estimate the carrier offset over the selected range\n"
            f"(x^{m.order} spectral line, valid for |offset| < fs/{2 * m.order})")
        self.auto_phase_btn.setToolTip(
            f"Rotate the constellation onto the ideal one\n"
            f"({m.ambiguity_deg:.0f}° ambiguity remains; use +90°, Invert or Differential)")
        self.diff_chk.setEnabled(m.diff)
        if not m.diff and self.diff_chk.isChecked():
            self.diff_chk.setChecked(False)  # redraws through _update_bits
        if self.eq_chk.isChecked():
            self.schedule_demod()  # the equalizer is driven by the constellation
        else:
            self.redraw()

    def rotate_90(self):
        v = (self.phase_spin.value() + 90) % 360
        self.phase_spin.setValue(v - 360 if v > 180 else v)

    def _update_carrier_label(self, *_):
        self.carrier_lbl.setText(f"carrier at {-self.freq_spin.value():+,} Hz")

    def _phase_spin_changed(self, v):
        self._syncing = True
        self.phase_slider.setValue(int(round(v * 10)))
        self._syncing = False
        self.schedule_demod()

    def _sp_spin_changed(self, v):
        self._syncing = True
        self.sp_slider.setValue(v)
        self._syncing = False
        self.schedule_demod()

    def _update_sp_range(self):
        top = self.sps_spin.value() * self.os_combo.currentData() - 1
        self.sp_spin.setMaximum(top)
        self.sp_slider.setMaximum(top)

    def _os_changed(self):
        new = self.os_combo.currentData()
        sp = self.sp_spin.value()
        self._update_sp_range()
        # keep the same sampling instant when switching the oversampling factor
        self.sp_spin.setValue(int(sp * new / self._last_os))
        self._last_os = new
        self.schedule_demod()

    def _lp_sps_edited(self):
        if not self._syncing:
            self._lp_sps_follows = False
        self.schedule_demod()

    def _filter_changed(self):
        filt = self.filt_combo.currentData()
        self.beta_spin.setEnabled(filt in ("rrc", "rc"))
        self.span_spin.setEnabled(filt in ("rrc", "rc"))
        for w in (self.lp_sps_spin, self.lp_delay_spin, self.lp_beta_spin):
            w.setEnabled(filt == "lp")
        self.schedule_demod()

    def _eq_changed(self, *_):
        for w in (self.eq_mu_spin, self.eq_taps_spin, self.eq_look_spin):
            w.setEnabled(self.eq_chk.isChecked())
        self.schedule_demod()

    def _eq_taps_changed(self, *_):
        # the reference tap sits at the lookahead position, so it has to stay inside
        self.eq_look_spin.setMaximum(self.eq_taps_spin.value() - 1)
        self._eq_changed()

    def params(self) -> dsp.DemodParams:
        return dsp.DemodParams(
            fs=self.fs_spin.value(),
            skip=self.skip_spin.value(),
            sps=self.sps_spin.value(),
            sym_from=self.from_spin.value(),
            sym_to=self.to_spin.value(),
            remove_dc=self.dc_chk.isChecked(),
            freq_hz=float(self.freq_spin.value()),
            phase_deg=self.phase_spin.value(),
            oversample=self.os_combo.currentData(),
            filt=self.filt_combo.currentData(),
            rc_beta=self.beta_spin.value(),
            rc_span=self.span_spin.value(),
            lp_sps=self.lp_sps_spin.value(),
            lp_delay=self.lp_delay_spin.value(),
            lp_beta=self.lp_beta_spin.value(),
            sample_point=self.sp_spin.value(),
            eq_on=self.eq_chk.isChecked(),
            eq_mu=self.eq_mu_spin.value(),
            eq_taps=self.eq_taps_spin.value(),
            eq_lookahead=self.eq_look_spin.value(),
        )

    def schedule_demod(self, *_):
        self._demod_timer.start()

    # ------------------------------------------------------------------ auto

    def auto_freq(self):
        if self.f is None:
            return
        est = dsp.estimate_freq(self.f, self.params(), self.mod())
        self.freq_spin.setValue(-int(round(est)))
        self.statusBar().showMessage(f"Estimated carrier offset {est:+.1f} Hz", 5000)
        self.update_demod()

    def auto_timing(self):
        self.update_demod()
        if self.res is not None and self.res.nsym:
            self.sp_spin.setValue(dsp.estimate_timing(self.res))
            self.update_demod()

    def auto_phase(self):
        self.update_demod()
        if self.res is not None and self.res.nsym:
            ph = self.phase_spin.value() - dsp.estimate_phase_deg(self.res.sym, self.mod())
            self.phase_spin.setValue((ph + 180) % 360 - 180)
            self.update_demod()

    def auto_all(self):
        self.auto_freq()
        self.auto_timing()
        self.auto_phase()

    # ------------------------------------------------------------------ processing

    def update_demod(self):
        self._demod_timer.stop()
        p = self.params()
        if self.f is None or p.sym_to <= p.sym_from:
            self.res = None
            self.redraw()
            return
        t0 = time.perf_counter()
        self.res = dsp.demodulate(self.f, p, self.mod())
        dt = time.perf_counter() - t0
        self.redraw()
        self.statusBar().showMessage(
            f"Demodulated {self.res.nsym:,} symbols in {dt * 1000:.0f} ms "
            f"(+ drawing {1000 * (time.perf_counter() - t0 - dt):.0f} ms)", 4000)

    def redraw(self, *_):
        r = self.res
        if r is None or r.nsym == 0:
            for c in (self.i_curve, self.q_curve, self.vlines, self.i_marks, self.q_marks,
                      self.eye_i, self.eye_q):
                c.setData([], [])
            self.const_scatter.clear()
            self.const_ref.clear()
            self.eye_img.clear()
            self.metrics_lbl.setText("")
            self.eq_lbl.setText("")
            self.bits_edit.setPlainText("")
            return

        # I/Q time plot
        self.i_curve.setData(r.t, r.y.real)
        self.q_curve.setData(r.t, r.y.imag)
        ts = r.t[r.idx]
        mode = self.marker_combo.currentText()
        dots = mode in ("Dots", "Dots + lines")
        lines = mode in ("Vertical lines", "Dots + lines")
        raw = r.sym_raw if r.sym_raw is not None else r.sym
        if dots:
            self.i_marks.setData(ts, raw.real)
            self.q_marks.setData(ts, raw.imag)
        else:
            self.i_marks.setData([], [])
            self.q_marks.setData([], [])
        if lines:
            ymax = 1.2 * float(np.max(np.abs(np.r_[r.y.real, r.y.imag]))) or 1.0
            self.vlines.setData(np.repeat(ts, 2), np.tile([-ymax, ymax], len(ts)))
        else:
            self.vlines.setData([], [])
        if self._reset_iq_view:
            self._reset_iq_view = False
            show = min(r.nsym, 40) * (r.L / r.oversample)
            self.iq_plot.setXRange(r.t[0], r.t[0] + show, padding=0.02)
            self.iq_plot.enableAutoRange(axis="y")

        # constellation
        s = r.sym
        if self.const_time_chk.isChecked():
            cmap = pg.colormap.get("viridis")
            cols = cmap.map(np.linspace(0, 1, len(s)), mode="qcolor")
            self.const_scatter.setData(s.real, s.imag, brush=cols)
        else:
            self.const_scatter.setData(s.real, s.imag, brush=pg.mkBrush(CONST_COLOR + (140,)))

        # eye
        offs, seg = dsp.eye_segments(r, self.eye_width_spin.value(), self.eye_max_spin.value())
        if self.eye_mode_combo.currentText() == "Lines":
            self.eye_img.clear()
            self.eye_img.hide()
            self._set_eye_lines(self.eye_i, offs, seg.real)
            if self.eye_q_chk.isChecked():
                self._set_eye_lines(self.eye_q, offs, seg.imag)
            else:
                self.eye_q.setData([], [])
        else:
            self.eye_i.setData([], [])
            self.eye_q.setData([], [])
            ym = 1.1 * float(np.max(np.abs(seg.real))) if seg.size else 1.0
            ym = ym or 1.0
            H = dsp.eye_density(offs, seg.real, (-ym, ym))
            self.eye_img.show()
            self.eye_img.setImage(np.log1p(H), autoLevels=True)
            self.eye_img.setRect(QtCore.QRectF(offs[0], -ym, offs[-1] - offs[0], 2 * ym))
        self.eye_plot.enableAutoRange()

        # metrics + bits
        mod = self.mod()
        m = dsp.symbol_metrics(r.sym, mod)
        ref = mod.points * m.amplitude
        self.const_ref.setData(ref.real, ref.imag)
        self.metrics_lbl.setText(
            f"{m.nsym:,} symbols · {m.nsym * mod.bits_per_symbol:,} bits<br>"
            f"level {m.amplitude:.4f} FS · power {m.power_dbfs:.2f} dBFS<br>"
            f"MER {m.mer_db:.2f} dB · EVM {m.evm_pct:.2f} %<br>"
            f"residual phase {m.residual_phase_deg:+.2f}° "
            f"(mod {mod.ambiguity_deg:.0f}°)")
        self._update_eq_label(r, mod)
        self._update_bits()

    def _update_eq_label(self, r: dsp.DemodResult, mod: dsp.Modulation):
        if r.eq_w is None:
            self.eq_lbl.setText("")
            return
        if r.eq_diverged:
            self.eq_lbl.setText("<b>diverged</b> — symbols passed through; "
                                "lower the learning rate")
            return
        raw = r.sym_raw if r.sym_raw is not None else r.sym
        before = dsp.symbol_metrics(raw, mod)
        after = dsp.symbol_metrics(r.sym, mod)
        txt = f"MER {before.mer_db:.2f} → {after.mer_db:.2f} dB"
        # w[k] multiplies symbol n+lookahead-k, so reversing puts the taps in time order
        w = np.abs(r.eq_w[::-1])
        if len(w) <= 13:
            g = 20 * np.log10(w / max(w.max(), dsp.EPS) + dsp.EPS)
            txt += ("<br>taps, past→future (dB rel. peak): "
                    + " ".join(f"{v:.0f}" for v in g))
        txt += f"<br>peak tap at {int(np.argmax(w)) - (len(w) - 1 - r.eq_lookahead):+d} sym"
        self.eq_lbl.setText(txt)

    @staticmethod
    def _set_eye_lines(item, offs, seg):
        n, w = seg.shape
        if n == 0:
            item.setData([], [])
            return
        x = np.empty((n, w + 1))
        x[:, :w] = offs
        x[:, w] = np.nan
        y = np.full((n, w + 1), np.nan)
        y[:, :w] = seg
        item.setData(x.ravel(), y.ravel())

    def _bits(self) -> np.ndarray:
        if self.res is None:
            return np.zeros(0, np.uint8)
        return dsp.slice_bits(self.res.sym, self.mod(), self.diff_chk.isChecked(),
                              self.inv_chk.isChecked())

    @staticmethod
    def _format_bits(b: np.ndarray) -> str:
        s = "".join("01"[v] for v in b)
        rows = []
        for i in range(0, len(s), 32):
            row = s[i:i + 32]
            rows.append(" ".join(row[j:j + 4] for j in range(0, len(row), 4)))
        return "\n".join(rows)

    def _update_bits(self):
        b = self._bits()
        limit = 50_000
        txt = self._format_bits(b[:limit])
        if len(b) > limit:
            txt += f"\n… {len(b) - limit:,} more bits (use Copy bits)"
        self.bits_edit.setPlainText(txt)

    def _copy_bits(self):
        QtWidgets.QApplication.clipboard().setText("".join("01"[v] for v in self._bits()))
        self.statusBar().showMessage("Bits copied to clipboard", 3000)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Demod Analyzer")
    w = MainWindow()
    w.show()
    if len(sys.argv) > 1:
        w.load_file(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
