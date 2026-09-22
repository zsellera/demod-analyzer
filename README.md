# Demod Analyzer

Interactive analysis/demodulation of digital signals (BPSK, QPSK, 16-QAM)
captured as interleaved IQ files. Python + PySide6 (Qt) + pyqtgraph.

## Setup / run

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python demod_analyzer.py [capture.cs8]
```

Try it on a synthetic signal:

```sh
.venv/bin/python make_test_signal.py test.cs8 --format int8     # rectangular pulses
.venv/bin/python make_test_signal.py test.cs16 --format int16 --rrc
.venv/bin/python make_test_signal.py qam.cs16 --format int16 --rrc --mod qam16
.venv/bin/python demod_analyzer.py test.cs8                      # fs=1 MHz, skip 5000, SPS 8
```

`--mod bpsk|qpsk|qam16` picks the constellation (default `bpsk`). Run
`make_test_signal.py` without arguments for a GUI with a live preview of the
signal it is about to write.

## Digital Demodulation Analyzer

Analyze BPSK/QPSK/QAM16 signals.

![analyzer](screenshots/analyzer.png)

## IQ splitter

Segment an IQ recording into smaller chunks based on RSSI

![splitter](screenshots/splitter.png)

## Test signal generator

Generate test data; available in command line as well (see above)

![generator](screenshots/generator.png)
