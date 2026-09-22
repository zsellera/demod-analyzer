"""IQ capture splitter: cut a long capture into chunks where the RSSI exceeds a threshold.

    python iq_splitter.py [capture.cs8]
"""
from __future__ import annotations

import os
import re
import sys
import time

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

import dsp
from demod_analyzer import FORMAT_LABELS, _dspin, _spin, file_info_html

CHUNK_BRUSH = (80, 220, 160, 60)
THRESH_COLOR = (255, 60, 110)
TABLE_COLUMNS = ["#", "start sample", "samples", "start (s)", "duration (ms)",
                 "mean dBFS", "peak dBFS"]


class SplitterWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IQ Splitter")
        self.resize(1400, 900)
        self.settings = QtCore.QSettings("bpsk-decode", "iq-splitter")

        self.f: dsp.IQFile | None = None
        self.power = np.zeros(0, np.float32)
        self.bursts = np.zeros((0, 2), np.int64)
        self._syncing = False
        self._first_power = False
        self._last_sps = int(self.settings.value("sps", 8))

        self._power_timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self._power_timer.timeout.connect(self.update_power)
        self._seg_timer = QtCore.QTimer(self, singleShot=True, interval=80)
        self._seg_timer.timeout.connect(self.update_bursts)

        self._build_plots()
        self._build_controls()
        self._build_menu()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.controls_scroll)
        splitter.addWidget(self.right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([370, 1030])
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
        open_btn = QtWidgets.QPushButton("Open…")
        open_btn.clicked.connect(self.open_dialog)
        self.path_lbl = QtWidgets.QLabel("—")
        self.path_lbl.setWordWrap(True)
        fl.addRow(open_btn, self.path_lbl)
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
        self.dc_chk = QtWidgets.QCheckBox("Remove DC offset (for RSSI)")
        self.dc_chk.setToolTip("Only affects the RSSI computation; chunks are written "
                               "with the original raw samples.")
        self.dc_chk.setChecked(True)
        self.dc_chk.toggled.connect(lambda: self._power_timer.start())
        fl.addRow(self.dc_chk)
        lay.addWidget(g)

        # --- symbols / range
        g = QtWidgets.QGroupBox("Range")
        fl = QtWidgets.QFormLayout(g)
        self.sps_spin = _spin(1, 1_000_000, self._last_sps)
        self.sps_spin.valueChanged.connect(self._sps_changed)
        fl.addRow("Samples / symbol", self.sps_spin)
        self.from_spin = _spin(0, 0, 0)
        self.to_spin = _spin(0, 0, 0)
        self.from_spin.valueChanged.connect(self._range_changed)
        self.to_spin.valueChanged.connect(self._range_changed)
        fl.addRow("Symbol from", self.from_spin)
        fl.addRow("Symbol to (excl.)", self.to_spin)
        self.nsym_lbl = QtWidgets.QLabel("")
        fl.addRow(self.nsym_lbl)
        hint = QtWidgets.QLabel("The range is the visible part of the plot: zoom (wheel) "
                                "and pan (drag) to select it.")
        hint.setWordWrap(True)
        fl.addRow(hint)
        full_btn = QtWidgets.QPushButton("Full view (whole file)")
        full_btn.clicked.connect(self.full_view)
        fl.addRow(full_btn)
        lay.addWidget(g)

        # --- segmentation
        g = QtWidgets.QGroupBox("Segmentation")
        fl = QtWidgets.QFormLayout(g)
        self.thr_spin = _dspin(-200, 20, -30, 0.5, 1, " dBFS")
        self.thr_spin.setToolTip("Symbols with RSSI above this level belong to a chunk. "
                                 "You can also drag the red line on the plot.")
        self.thr_spin.valueChanged.connect(self._thr_spin_changed)
        fl.addRow("RSSI threshold", self.thr_spin)
        self.gap_spin = _spin(0, 100_000_000, 8, 1, " sym")
        self.gap_spin.setToolTip("Dips below the threshold up to this long don't split a chunk")
        fl.addRow("Merge gaps ≤", self.gap_spin)
        self.minlen_spin = _spin(1, 100_000_000, 16, 1, " sym")
        self.minlen_spin.setToolTip("Shorter bursts (before padding) are dropped")
        fl.addRow("Min length", self.minlen_spin)
        self.pad_spin = _spin(0, 100_000_000, 16, 1, " sym")
        self.pad_spin.setToolTip("Extra symbols kept before and after each burst")
        fl.addRow("Padding", self.pad_spin)
        for w in (self.gap_spin, self.minlen_spin, self.pad_spin):
            w.valueChanged.connect(lambda *_: self._seg_timer.start())
        self.seg_lbl = QtWidgets.QLabel("")
        self.seg_lbl.setWordWrap(True)
        fl.addRow(self.seg_lbl)
        lay.addWidget(g)

        # --- output
        g = QtWidgets.QGroupBox("Output")
        fl = QtWidgets.QFormLayout(g)
        self.dir_edit = QtWidgets.QLineEdit(str(self.settings.value("out_dir", "")))
        browse = QtWidgets.QPushButton("…")
        browse.setFixedWidth(32)
        browse.clicked.connect(self.browse_dir)
        fl.addRow("Directory", self._hbox(self.dir_edit, browse))
        self.prefix_edit = QtWidgets.QLineEdit("chunk")
        self.prefix_edit.textChanged.connect(self._update_name_preview)
        fl.addRow("Prefix", self.prefix_edit)
        self.name_lbl = QtWidgets.QLabel("")
        fl.addRow("", self.name_lbl)
        self.make_btn = QtWidgets.QPushButton("Make chunks")
        self.make_btn.clicked.connect(self.make_chunks)
        fl.addRow(self.make_btn)
        self.out_lbl = QtWidgets.QLabel("")
        self.out_lbl.setWordWrap(True)
        self.out_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        fl.addRow(self.out_lbl)
        lay.addWidget(g)
        lay.addStretch(1)

        self.controls_scroll = QtWidgets.QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.controls_scroll.setWidget(panel)
        self.controls_scroll.setMinimumWidth(340)
        self._update_name_preview()

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

        self.plot = pg.PlotWidget(title="RSSI per symbol (RMS over each SPS window) — "
                                        "zoom/pan to select the range to split")
        self.plot.setLabel("bottom", "symbol index")
        self.plot.setLabel("left", "dBFS")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        for ax in ("left", "bottom"):
            self.plot.getAxis(ax).enableAutoSIPrefix(False)
        vb = self.plot.getViewBox()
        vb.setMouseEnabled(x=True, y=False)
        vb.enableAutoRange(y=False)
        vb.sigXRangeChanged.connect(self._view_changed)

        self.bars = pg.BarGraphItem(x0=[0], x1=[0], y0=[0], y1=[0],
                                    brush=CHUNK_BRUSH, pen=None)
        self.bars.setZValue(-5)
        self.bars.hide()
        self.plot.addItem(self.bars)
        self.curve = self.plot.plot(pen=pg.mkPen((200, 200, 200), width=1))
        self.curve.setDownsampling(auto=True, method="peak")
        self.curve.setClipToView(True)
        self.thr_line = pg.InfiniteLine(
            angle=0, movable=True, pen=pg.mkPen(THRESH_COLOR, width=2),
            hoverPen=pg.mkPen(THRESH_COLOR, width=4),
            label="RSSI {value:.1f} dBFS", labelOpts={"position": 0.1, "color": THRESH_COLOR, "fill": (0, 0, 0, 170)})
        self.thr_line.setZValue(20)
        self.thr_line.sigPositionChanged.connect(self._thr_line_moved)
        self.plot.addItem(self.thr_line)
        self.right.addWidget(self.plot)

        self.table = QtWidgets.QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels(TABLE_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.right.addWidget(self.table)
        self.right.setSizes([600, 300])

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
        self.setWindowTitle(f"IQ Splitter — {os.path.basename(path)}")
        self.info_lbl.setText(file_info_html(f, self.fs_spin.value()))
        self.prefix_edit.setText(os.path.splitext(os.path.basename(path))[0])
        if not self.dir_edit.text().strip():
            self.dir_edit.setText(os.path.dirname(os.path.abspath(path)))
        self.out_lbl.setText("")
        self._first_power = True
        self._update_limits()
        self.update_power()

    def _format_changed(self):
        if self.f is not None:
            self.load_file(self.f.path, self.fmt_combo.currentData())

    def _fs_changed(self):
        self.settings.setValue("fs", self.fs_spin.value())
        if self.f is not None:
            self.info_lbl.setText(file_info_html(self.f, self.fs_spin.value()))
        self._update_nsym_label()
        self._seg_timer.start()

    # ------------------------------------------------------------------ range

    def _n_sym_total(self) -> int:
        return 0 if self.f is None else self.f.n_samples // self.sps_spin.value()

    def _update_limits(self):
        n = self._n_sym_total()
        self._syncing = True
        self.from_spin.setMaximum(max(0, n - 1))
        self.to_spin.setMaximum(n)
        self._syncing = False

    def _sps_changed(self):
        new = self.sps_spin.value()
        old = self._last_sps
        self._last_sps = new
        self.settings.setValue("sps", new)
        # keep the selected range at the same sample positions
        fr = self.from_spin.value() * old // new
        to = -(-self.to_spin.value() * old // new)
        self._update_limits()
        self._syncing = True
        self.from_spin.setValue(fr)
        self.to_spin.setValue(to)
        self._syncing = False
        self._power_timer.start()

    def _update_nsym_label(self):
        n = self._n_sym_total()
        sel = max(0, self.to_spin.value() - self.from_spin.value())
        sps = self.sps_spin.value()
        self.nsym_lbl.setText(f"{n:,} symbols in file · {sel:,} selected<br>"
                              f"selection {sel * sps / self.fs_spin.value():.6g} s")

    def _range_changed(self):
        """Symbol from/to edited: show exactly that range."""
        if self._syncing:
            return
        self._syncing = True
        self.plot.setXRange(self.from_spin.value(),
                            max(self.to_spin.value(), self.from_spin.value() + 1), padding=0)
        self._syncing = False
        self._update_nsym_label()
        self._seg_timer.start()

    def _view_changed(self, _vb, xr):
        """Plot zoomed/panned: the visible part becomes the selected range."""
        if self._syncing or self.f is None:
            return
        n = self._n_sym_total()
        lo = int(np.floor(min(max(xr[0], 0), n)))
        hi = int(np.ceil(min(max(xr[1], 0), n)))
        self._syncing = True
        self.from_spin.setValue(lo)
        self.to_spin.setValue(max(hi, lo + 1))
        self._syncing = False
        self._update_nsym_label()
        self._seg_timer.start()

    def full_view(self):
        self._syncing = True
        self.from_spin.setValue(0)
        self.to_spin.setValue(self._n_sym_total())
        self._syncing = False
        self._range_changed()

    # ------------------------------------------------------------------ RSSI

    def update_power(self):
        if self.f is None:
            return
        t0 = time.perf_counter()
        pw = dsp.symbol_power_db(self.f, 0, self.sps_spin.value(), self.dc_chk.isChecked())
        self.power = pw
        n = len(pw)
        if n == 0:
            self.curve.setData([], [])
            self._seg_timer.start()
            return
        self.curve.setData(np.arange(n), pw)
        # fixed Y scale; ignore the -200 dBFS floor of all-zero windows
        lo = float(np.floor(np.percentile(pw, 0.1))) - 5
        hi = float(np.ceil(pw.max())) + 3
        self._y = (lo, hi)
        vb = self.plot.getViewBox()
        vb.setLimits(xMin=0, xMax=n, yMin=lo, yMax=hi, minXRange=4)
        self.plot.setYRange(lo, hi, padding=0)
        self.thr_line.setBounds((lo, hi))
        if self._first_power:
            self._first_power = False
            p_lo, p_hi = np.percentile(pw, [10, 99.5])
            self.thr_spin.setValue(round(float(p_lo + p_hi) / 2, 1))
            self.full_view()
        else:
            self._range_changed()
        self._update_nsym_label()
        self.statusBar().showMessage(
            f"RSSI: {n:,} symbols in {time.perf_counter() - t0:.2f} s", 4000)
        self._seg_timer.start()

    def _thr_spin_changed(self, v):
        if not self._syncing:
            self._syncing = True
            self.thr_line.setValue(v)
            self._syncing = False
        self._seg_timer.start()

    def _thr_line_moved(self):
        if self._syncing:
            return
        self._syncing = True
        self.thr_spin.setValue(round(self.thr_line.value(), 1))
        self._syncing = False
        self._seg_timer.start()

    # ------------------------------------------------------------------ segmentation

    def update_bursts(self):
        self._seg_timer.stop()
        if self.f is None or len(self.power) == 0:
            self.bursts = np.zeros((0, 2), np.int64)
        else:
            self.bursts = dsp.find_bursts(
                self.power, self.thr_spin.value(), self.from_spin.value(), self.to_spin.value(),
                self.gap_spin.value(), self.minlen_spin.value(), self.pad_spin.value())
        b = self.bursts
        if len(b):
            lo, hi = self._y
            self.bars.setOpts(x0=b[:, 0], x1=b[:, 1], y0=np.full(len(b), lo),
                              y1=np.full(len(b), hi))
            self.bars.show()
        else:
            self.bars.hide()
        self._fill_table()
        sps = self.sps_spin.value()
        total = int(np.sum(b[:, 1] - b[:, 0])) * sps
        if len(b):
            lens = (b[:, 1] - b[:, 0]) * sps
            self.seg_lbl.setText(
                f"<b>{len(b):,} chunks</b>, {total:,} samples total<br>"
                f"length {lens.min():,} … {lens.max():,} samples")
        else:
            self.seg_lbl.setText("<b>no chunks</b> above the threshold in the selected range")

    def _fill_table(self):
        b = self.bursts
        sps = self.sps_spin.value()
        fs = self.fs_spin.value()
        mean, peak = dsp.burst_stats(self.power, b)
        self.table.setUpdatesEnabled(False)
        self.table.blockSignals(True)
        self.table.clearContents()
        self.table.setRowCount(len(b))
        for i, (s, e) in enumerate(b):
            a, n = int(s) * sps, int(e - s) * sps
            vals = [str(i), f"{a:,}", f"{n:,}", f"{a / fs:.6f}", f"{n / fs * 1e3:.3f}",
                    f"{mean[i]:.1f}", f"{peak[i]:.1f}"]
            for c, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                it.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
                self.table.setItem(i, c, it)
        self.table.blockSignals(False)
        self.table.setUpdatesEnabled(True)

    # ------------------------------------------------------------------ output

    def browse_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Working directory", self.dir_edit.text() or os.getcwd())
        if d:
            self.dir_edit.setText(d)

    def _ext(self) -> str:
        return dsp.chunk_ext(self.f) if self.f is not None else ".cs8"

    def _update_name_preview(self):
        prefix = self.prefix_edit.text().strip() or "…"
        self.name_lbl.setText(f"→ {prefix}-00000{self._ext()}, {prefix}-index.csv")

    def make_chunks(self):
        if self.f is None:
            return
        self.update_bursts()  # apply any pending parameter change
        directory = self.dir_edit.text().strip()
        prefix = self.prefix_edit.text().strip()
        if not directory or not prefix:
            QtWidgets.QMessageBox.warning(self, "Make chunks", "Set a directory and a prefix.")
            return
        if len(self.bursts) == 0:
            self.out_lbl.setText("Nothing written: no chunks above the threshold.")
            return
        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            names = dsp.write_chunks(self.f, self.bursts, self.sps_spin.value(), directory,
                                     prefix, self.fs_spin.value(), self.power)
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Make chunks", str(e))
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.settings.setValue("out_dir", directory)
        msg = (f"Wrote {len(names)} chunks ({names[0]} … {names[-1]}) "
               f"and {prefix}-index.csv to {directory}")
        stale = self._stale_chunks(directory, prefix, len(names))
        if stale:
            msg += (f"<br><b>Note:</b> {stale} older chunk file(s) numbered ≥ {len(names):05d} "
                    f"from a previous run are still in the directory.")
        self.out_lbl.setText(msg)
        self.statusBar().showMessage(f"Wrote {len(names)} chunks", 5000)

    def _stale_chunks(self, directory: str, prefix: str, count: int) -> int:
        pat = re.compile(re.escape(prefix) + r"-(\d{5})" + re.escape(self._ext()) + "$")
        return sum(1 for name in os.listdir(directory)
                   if (m := pat.match(name)) and int(m.group(1)) >= count)


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("IQ Splitter")
    w = SplitterWindow()
    w.show()
    if len(sys.argv) > 1:
        w.load_file(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
