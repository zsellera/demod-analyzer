"""Generate a synthetic BPSK / QPSK / 16-QAM capture for trying out the analyzer.

Without arguments it opens a GUI:

    python make_test_signal.py

    python make_test_signal.py test.cs8   --format int8
    python make_test_signal.py test.cs16  --format int16 --mod qam16
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, fields

import numpy as np

import dsp

FORMAT_EXT = {"int8": ".cs8", "uint8": ".cu8", "int16": ".cs16"}


@dataclass
class Params:
    out: str = "test.cs8"
    format: str = "int8"
    mod: str = "bpsk"
    fs: float = 1e6
    sps: int = 8
    nsym: int = 200  # symbols per burst
    bursts: int = 1  # number of bursts, separated by `lead` noise samples
    freq: float = 0.0  # carrier offset, Hz
    phase: float = 0.0  # carrier phase, deg
    snr: float = 22.0  # Es/N0, dB
    lead: int = 200  # noise-only samples before the burst
    delay: int = 3  # extra sample delay (timing offset)
    rrc: bool = False  # RRC pulse shaping instead of rectangular
    seed: int = 1


def synthesize(p: Params) -> tuple[np.ndarray, np.ndarray]:
    """Build the complex baseband capture and the bit stream it carries."""
    rng = np.random.default_rng(p.seed)
    mod = dsp.MODULATIONS[p.mod]
    k = mod.bits_per_symbol
    words = rng.integers(0, 1 << k, p.nsym * p.bursts)
    bits = ((words[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).ravel()
    parts = [np.zeros(p.lead + p.delay, complex)]
    for burst_words in np.split(words, p.bursts):
        syms = mod.points[burst_words]
        if p.rrc:
            up = np.zeros(len(syms) * p.sps, complex)
            up[:: p.sps] = syms
            h = dsp.rrc_taps(p.sps, 0.35, 8)
            parts.append(np.convolve(up, h / np.sqrt(np.sum(h**2)) * np.sqrt(p.sps), mode="same"))
        else:
            parts.append(np.repeat(syms, p.sps))
        parts.append(np.zeros(p.lead))
    x = np.concatenate(parts)
    n = np.arange(len(x))
    x = x * np.exp(1j * (2 * np.pi * p.freq / p.fs * n + np.deg2rad(p.phase)))
    noise_sigma = np.sqrt(p.sps / 10 ** (p.snr / 10) / 2)
    x = x + noise_sigma * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return 0.4 * x / np.max(np.abs(x)) + (0.02 - 0.01j), bits  # scale and add a DC offset


def write_capture(p: Params, x: np.ndarray, bits: np.ndarray) -> None:
    """Quantize `x` into the raw IQ file and save the reference bits next to it."""
    dtype, offset, scale = dsp.FORMATS[p.format]
    info = np.iinfo(dtype)
    iq = np.empty(2 * len(x))
    iq[0::2] = x.real
    iq[1::2] = x.imag
    np.clip(np.round(iq * scale + offset), info.min, info.max).astype(dtype).tofile(p.out)
    np.save(p.out + ".bits.npy", bits)


def summary(p: Params, x: np.ndarray) -> str:
    return (f"wrote {p.out}: {len(x)} samples, {p.mod}, fs={p.fs:g}, sps={p.sps}, "
            f"skip={p.lead}, freq={p.freq:+g} Hz, phase={p.phase:g} deg, delay={p.delay}")


# ---------------------------------------------------------------------- GUI


def run_gui() -> int:
    from PySide6 import QtCore, QtGui, QtWidgets  # noqa: PLC0415 - only needed for the GUI
    import pyqtgraph as pg  # noqa: PLC0415

    from demod_analyzer import FORMAT_LABELS, _dspin, _spin

    I_COLOR = (80, 175, 255)
    Q_COLOR = (255, 150, 60)
    CONST_COLOR = (80, 220, 160)
    PREVIEW_SYMBOLS = 64  # symbols of the burst shown in the time plot
    PREVIEW_POINTS = 3000  # constellation points drawn at most

    class GeneratorWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Test Signal Generator")
            self.resize(1200, 800)
            self.settings = QtCore.QSettings("bpsk-decode", "test-signal")

            self.x = np.zeros(0, complex)
            self.bits = np.zeros(0, np.uint8)

            self._build_plots()
            self._build_controls()
            self._build_menu()

            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            splitter.addWidget(self.controls_scroll)
            splitter.addWidget(self.right)
            splitter.setStretchFactor(1, 1)
            splitter.setSizes([370, 830])
            self.setCentralWidget(splitter)
            self.statusBar().showMessage("Set the parameters and press Generate (Ctrl+G)")
            self.preview()

        # ------------------------------------------------------------ UI build

        def _build_menu(self):
            m = self.menuBar().addMenu("&File")
            act = m.addAction("&Generate")
            act.setShortcut("Ctrl+G")
            act.triggered.connect(self.generate)
            m.addSeparator()
            q = m.addAction("&Quit")
            q.setShortcut(QtGui.QKeySequence.Quit)
            q.triggered.connect(self.close)

        def _build_controls(self):
            panel = QtWidgets.QWidget()
            lay = QtWidgets.QVBoxLayout(panel)
            d = Params()

            # --- signal
            g = QtWidgets.QGroupBox("Signal")
            fl = QtWidgets.QFormLayout(g)
            self.mod_combo = QtWidgets.QComboBox()
            for k, v in dsp.MODULATIONS.items():
                self.mod_combo.addItem(v.label, k)
            self.mod_combo.setCurrentIndex(
                max(0, self.mod_combo.findData(str(self.settings.value("mod", d.mod)))))
            fl.addRow("Modulation", self.mod_combo)
            self.fs_spin = _dspin(1, 1e10, float(self.settings.value("fs", d.fs)), 1000, 0, " Hz")
            self.fs_spin.setGroupSeparatorShown(True)
            fl.addRow("Sample rate", self.fs_spin)
            self.sps_spin = _spin(1, 100_000, int(self.settings.value("sps", d.sps)))
            fl.addRow("Samples / symbol", self.sps_spin)
            self.nsym_spin = _spin(1, 100_000_000, d.nsym, 1000, " sym")
            self.nsym_spin.setGroupSeparatorShown(True)
            self.nsym_spin.setToolTip("Symbols per burst")
            fl.addRow("Symbols / burst", self.nsym_spin)
            self.bursts_spin = _spin(1, 10_000, d.bursts)
            self.bursts_spin.setToolTip("Bursts are separated by the lead-in noise")
            fl.addRow("Bursts", self.bursts_spin)
            self.rrc_chk = QtWidgets.QCheckBox("RRC pulse shaping (β = 0.35)")
            self.rrc_chk.setToolTip("Unchecked: rectangular pulses")
            self.rrc_chk.setChecked(d.rrc)
            fl.addRow(self.rrc_chk)
            lay.addWidget(g)

            # --- impairments
            g = QtWidgets.QGroupBox("Impairments")
            fl = QtWidgets.QFormLayout(g)
            self.freq_spin = _dspin(-1e9, 1e9, d.freq, 100, 1, " Hz")
            self.freq_spin.setToolTip("Carrier frequency offset")
            fl.addRow("Carrier offset", self.freq_spin)
            self.phase_spin = _dspin(-360, 360, d.phase, 1, 1, " °")
            fl.addRow("Carrier phase", self.phase_spin)
            self.snr_spin = _dspin(-20, 60, d.snr, 0.5, 1, " dB")
            self.snr_spin.setToolTip("Es/N0")
            fl.addRow("SNR", self.snr_spin)
            self.lead_spin = _spin(0, 100_000_000, d.lead, 100, " smp")
            self.lead_spin.setGroupSeparatorShown(True)
            self.lead_spin.setToolTip("Noise-only samples before each burst")
            fl.addRow("Lead-in", self.lead_spin)
            self.delay_spin = _spin(0, 100_000, d.delay, 1, " smp")
            self.delay_spin.setToolTip("Extra sample delay (timing offset)")
            fl.addRow("Delay", self.delay_spin)
            self.seed_spin = _spin(0, 2_000_000_000, d.seed)
            self.seed_spin.setToolTip("Random seed for the symbols and the noise")
            fl.addRow("Seed", self.seed_spin)
            lay.addWidget(g)

            # --- output
            g = QtWidgets.QGroupBox("Output")
            fl = QtWidgets.QFormLayout(g)
            self.fmt_combo = QtWidgets.QComboBox()
            for k, v in FORMAT_LABELS.items():
                self.fmt_combo.addItem(v, k)
            self.fmt_combo.setCurrentIndex(
                max(0, self.fmt_combo.findData(str(self.settings.value("format", d.format)))))
            self.fmt_combo.currentIndexChanged.connect(self._format_changed)
            fl.addRow("Format", self.fmt_combo)
            self.out_edit = QtWidgets.QLineEdit(str(self.settings.value("out", d.out)))
            browse = QtWidgets.QPushButton("…")
            browse.setFixedWidth(32)
            browse.clicked.connect(self.browse_out)
            fl.addRow("File", self._hbox(self.out_edit, browse))
            lay.addWidget(g)

            self.gen_btn = QtWidgets.QPushButton("Generate")
            self.gen_btn.setShortcut("Ctrl+G")
            self.gen_btn.clicked.connect(self.generate)
            lay.addWidget(self.gen_btn)
            self.info_lbl = QtWidgets.QLabel("")
            self.info_lbl.setWordWrap(True)
            self.info_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            lay.addWidget(self.info_lbl)
            lay.addStretch(1)

            self.mod_combo.currentIndexChanged.connect(self.preview)
            for w in (self.fs_spin, self.sps_spin, self.bursts_spin, self.freq_spin,
                      self.phase_spin, self.snr_spin, self.delay_spin, self.seed_spin):
                w.valueChanged.connect(self.preview)
            self.rrc_chk.toggled.connect(self.preview)

            self.controls_scroll = QtWidgets.QScrollArea()
            self.controls_scroll.setWidgetResizable(True)
            self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            self.controls_scroll.setWidget(panel)
            self.controls_scroll.setMinimumWidth(340)

        @staticmethod
        def _hbox(*widgets):
            h = QtWidgets.QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)
            for i, w in enumerate(widgets):
                h.addWidget(w, 1 if i == 0 else 0)
            return h

        def _build_plots(self):
            pg.setConfigOptions(antialias=False)
            self.right = QtWidgets.QSplitter(QtCore.Qt.Vertical)

            self.time_plot = pg.PlotWidget(
                title=f"Preview: first {PREVIEW_SYMBOLS} symbols of the burst")
            self.time_plot.setLabel("bottom", "sample")
            self.time_plot.setLabel("left", "amplitude, FS")
            self.time_plot.showGrid(x=True, y=True, alpha=0.3)
            self.time_plot.addLegend(offset=(-10, 10))
            self.i_curve = self.time_plot.plot(pen=pg.mkPen(I_COLOR, width=1), name="I")
            self.q_curve = self.time_plot.plot(pen=pg.mkPen(Q_COLOR, width=1), name="Q")
            self.right.addWidget(self.time_plot)

            self.const_plot = pg.PlotWidget(title="Preview: samples at the symbol instants")
            self.const_plot.setLabel("bottom", "I")
            self.const_plot.setLabel("left", "Q")
            self.const_plot.showGrid(x=True, y=True, alpha=0.3)
            self.const_plot.setAspectLocked(True)
            self.const_scatter = pg.ScatterPlotItem(size=3, pen=None,
                                                    brush=pg.mkBrush(*CONST_COLOR, 120))
            self.const_plot.addItem(self.const_scatter)
            self.right.addWidget(self.const_plot)
            self.right.setSizes([400, 400])

        # ------------------------------------------------------------ actions

        def params(self, nsym: int | None = None) -> Params:
            return Params(
                out=self.out_edit.text().strip(),
                format=self.fmt_combo.currentData(),
                mod=self.mod_combo.currentData(),
                fs=self.fs_spin.value(),
                sps=self.sps_spin.value(),
                nsym=self.nsym_spin.value() if nsym is None else nsym,
                bursts=self.bursts_spin.value(),
                freq=self.freq_spin.value(),
                phase=self.phase_spin.value(),
                snr=self.snr_spin.value(),
                lead=self.lead_spin.value(),
                delay=self.delay_spin.value(),
                rrc=self.rrc_chk.isChecked(),
                seed=self.seed_spin.value(),
            )

        def _format_changed(self):
            """Follow the format with the file extension, unless it was customized."""
            ext = FORMAT_EXT[self.fmt_combo.currentData()]
            base, old = os.path.splitext(self.out_edit.text().strip())
            if base and old in FORMAT_EXT.values():
                self.out_edit.setText(base + ext)

        def browse_out(self):
            fmt = self.fmt_combo.currentData()
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Write capture to", self.out_edit.text().strip(),
                f"IQ capture (*{FORMAT_EXT[fmt]});;All files (*)")
            if path:
                self.out_edit.setText(path)

        def preview(self):
            """Regenerate a short capture with the current settings and plot it."""
            p = self.params(nsym=PREVIEW_SYMBOLS + 8)
            p.bursts = 1
            p.lead = min(p.lead, 64)
            try:
                x, _ = synthesize(p)
            except (ValueError, MemoryError) as e:
                self.statusBar().showMessage(f"Preview failed: {e}", 5000)
                return
            start = p.lead + p.delay
            self.i_curve.setData(np.arange(len(x)), x.real)
            self.q_curve.setData(np.arange(len(x)), x.imag)
            self.time_plot.setXRange(start, start + PREVIEW_SYMBOLS * p.sps, padding=0.02)

            # a longer, noise-free-lead run just for the constellation cloud
            pc = self.params(nsym=min(self.nsym_spin.value(), PREVIEW_POINTS))
            pc.bursts = 1
            pc.lead = min(pc.lead, 64)
            try:
                xc, _ = synthesize(pc)
            except (ValueError, MemoryError):
                return
            sym = xc[pc.lead + pc.delay + pc.sps // 2:: pc.sps][:PREVIEW_POINTS]
            self.const_scatter.setData(sym.real, sym.imag)

        def generate(self):
            p = self.params()
            if not p.out:
                QtWidgets.QMessageBox.warning(self, "No output file",
                                              "Choose a file to write the capture to.")
                return
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                x, bits = synthesize(p)
                write_capture(p, x, bits)
            except (OSError, ValueError, MemoryError) as e:
                QtWidgets.QApplication.restoreOverrideCursor()
                QtWidgets.QMessageBox.critical(self, "Cannot write capture", f"{p.out}\n\n{e}")
                return
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            for k in ("out", "format", "mod", "fs", "sps"):
                self.settings.setValue(k, getattr(p, k))
            self.info_lbl.setText(
                f"<b>{os.path.basename(p.out)}</b><br>"
                f"{len(x):,} samples, {len(x) / p.fs:.6g} s<br>"
                f"{p.mod}, {len(bits):,} bits in {p.bursts} burst(s)<br>"
                f"Load with fs={p.fs:g}, skip={p.lead}, SPS={p.sps}<br>"
                f"Reference bits: {os.path.basename(p.out)}.bits.npy")
            self.statusBar().showMessage(f"Wrote {p.out} ({len(x):,} samples)", 5000)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Test Signal Generator")
    w = GeneratorWindow()
    w.show()
    return app.exec()


# ---------------------------------------------------------------------- CLI


def main():
    if len(sys.argv) == 1:
        sys.exit(run_gui())

    d = Params()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--format", choices=list(dsp.FORMATS), default=d.format)
    ap.add_argument("--mod", choices=list(dsp.MODULATIONS), default=d.mod)
    ap.add_argument("--fs", type=float, default=d.fs)
    ap.add_argument("--sps", type=int, default=d.sps)
    ap.add_argument("--nsym", type=int, default=d.nsym, help="symbols per burst")
    ap.add_argument("--bursts", type=int, default=d.bursts,
                    help="number of bursts, separated by --lead of noise")
    ap.add_argument("--freq", type=float, default=d.freq, help="carrier offset, Hz")
    ap.add_argument("--phase", type=float, default=d.phase, help="carrier phase, deg")
    ap.add_argument("--snr", type=float, default=d.snr, help="Es/N0, dB")
    ap.add_argument("--lead", type=int, default=d.lead,
                    help="noise-only samples before the burst")
    ap.add_argument("--delay", type=int, default=d.delay, help="extra sample delay (timing offset)")
    ap.add_argument("--rrc", action="store_true", help="RRC pulse shaping instead of rectangular")
    ap.add_argument("--seed", type=int, default=d.seed)
    a = ap.parse_args()

    p = Params(**{f.name: getattr(a, f.name) for f in fields(Params)})
    x, bits = synthesize(p)
    write_capture(p, x, bits)
    print(summary(p, x))


if __name__ == "__main__":
    main()
