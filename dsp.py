"""Signal processing for the digital signal analyzer (numpy/scipy only, no UI)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import signal as sig

# name -> (numpy dtype, zero offset, full scale)
FORMATS = {
    "int8": (np.dtype(np.int8), 0.0, 128.0),
    "uint8": (np.dtype(np.uint8), 127.5, 128.0),
    "int16": (np.dtype("<i2"), 0.0, 32768.0),
}

EPS = 1e-20


def db(p):
    """Power ratio -> dB (relative to full scale when p is normalized power)."""
    return 10.0 * np.log10(np.maximum(p, EPS))


def guess_format(path: str) -> str:
    name = os.path.basename(path).lower()
    ext = os.path.splitext(name)[1]
    if ext == ".cu8":
        return "uint8"
    if ext in (".cs16", ".iq16", ".sc16", ".c16", ".ci16", ".s16"):
        return "int16"
    if ext in (".cs8", ".iq8", ".sc8", ".c8", ".ci8", ".s8"):
        return "int8"
    return "int16" if "16" in name else "int8"


class IQFile:
    """Memory-mapped interleaved I/Q file, samples normalized to +-1.0 full scale."""

    CHUNK = 1 << 22

    def __init__(self, path: str, fmt: str):
        dtype, self.offset, self.scale = FORMATS[fmt]
        raw = np.memmap(path, dtype=dtype, mode="r")
        n = raw.size // 2
        if n == 0:
            raise ValueError("file contains no complete I/Q sample")
        self.path = path
        self.fmt = fmt
        self.raw = raw[: 2 * n].reshape(n, 2)
        self.n_samples = n
        self._compute_stats()

    def _compute_stats(self):
        acc = np.zeros(2)
        pwr = 0.0
        for a in range(0, self.n_samples, self.CHUNK):
            blk = self.raw[a : a + self.CHUNK].astype(np.float64) - self.offset
            acc += blk.sum(axis=0)
            pwr += np.square(blk).sum()
        mean = acc / self.n_samples / self.scale
        self.dc = complex(mean[0], mean[1])
        self.power = pwr / self.n_samples / self.scale**2  # mean |x|^2, DC included

    @property
    def dc_dbfs(self) -> float:
        return float(db(abs(self.dc) ** 2))

    @property
    def rms_dbfs(self) -> float:
        return float(db(self.power))

    @property
    def rms_ac_dbfs(self) -> float:
        return float(db(self.power - abs(self.dc) ** 2))

    def read(self, start: int, stop: int, remove_dc: bool = True) -> np.ndarray:
        start = max(0, int(start))
        stop = min(self.n_samples, int(stop))
        if stop <= start:
            return np.zeros(0, np.complex64)
        blk = self.raw[start:stop]
        x = np.empty(stop - start, np.complex64)
        x.real = blk[:, 0]
        x.imag = blk[:, 1]
        if self.offset:
            x -= np.complex64(self.offset * (1 + 1j))
        x /= np.float32(self.scale)
        if remove_dc:
            x -= np.complex64(self.dc)
        return x


def symbol_power_db(f: IQFile, skip: int, sps: int, remove_dc: bool) -> np.ndarray:
    """dBFS of the RMS over each SPS-long window, for every symbol after `skip`."""
    n_sym = max(0, (f.n_samples - skip) // sps)
    out = np.empty(n_sym, np.float32)
    per = max(1, IQFile.CHUNK // sps)
    for k in range(0, n_sym, per):
        k2 = min(n_sym, k + per)
        x = f.read(skip + k * sps, skip + k2 * sps, remove_dc)
        p = (x.real**2 + x.imag**2).reshape(-1, sps).mean(axis=1)
        out[k:k2] = db(p)
    return out


# --- burst splitting -----------------------------------------------------------

DEFAULT_EXT = {"int8": ".cs8", "uint8": ".cu8", "int16": ".cs16"}


def _merge_close(s: np.ndarray, e: np.ndarray, gap: int):
    """Merge [s, e) intervals separated by at most `gap`."""
    if len(s) < 2:
        return s, e
    keep = (s[1:] - e[:-1]) > gap
    return np.r_[s[:1], s[1:][keep]], np.r_[e[:-1][keep], e[-1:]]


def find_bursts(power_db: np.ndarray, threshold_db: float, sym_from: int, sym_to: int,
                merge_gap: int = 8, min_len: int = 16, pad: int = 16) -> np.ndarray:
    """Symbol intervals [start, end) within [sym_from, sym_to) where RSSI > threshold.

    Dips of at most `merge_gap` symbols are bridged, bursts shorter than
    `min_len` symbols are dropped, then each burst is widened by `pad` symbols
    on both sides (bursts that overlap after padding are merged).
    """
    active = np.asarray(power_db[sym_from:sym_to]) > threshold_db
    d = np.diff(np.r_[0, active.astype(np.int8), 0])
    s = np.flatnonzero(d == 1)
    e = np.flatnonzero(d == -1)
    s, e = _merge_close(s, e, merge_gap)
    ok = (e - s) >= min_len
    s = np.maximum(s[ok] - pad, 0)
    e = np.minimum(e[ok] + pad, len(active))
    s, e = _merge_close(s, e, 0)
    return np.stack([s + sym_from, e + sym_from], axis=1)


def burst_stats(power_db: np.ndarray, bursts: np.ndarray):
    """(mean dBFS, peak dBFS) of each burst; mean is taken in the power domain."""
    mean = np.array([db(np.mean(10 ** (power_db[s:e] / 10))) for s, e in bursts])
    peak = np.array([power_db[s:e].max() for s, e in bursts])
    return mean, peak


def chunk_ext(f: IQFile) -> str:
    return os.path.splitext(f.path)[1] or DEFAULT_EXT[f.fmt]


def write_chunks(f: IQFile, bursts: np.ndarray, sps: int, directory: str, prefix: str,
                 fs: float, power_db: np.ndarray) -> list[str]:
    """Write each burst as `<prefix>-NNNNN<ext>` (original raw samples, same format)
    plus `<prefix>-index.csv`, overwriting existing files. Returns the chunk file names."""
    os.makedirs(directory, exist_ok=True)
    ext = chunk_ext(f)
    mean, peak = burst_stats(power_db, bursts)
    names = []
    rows = ["chunk,file,start_sample,n_samples,start_s,duration_s,mean_dbfs,peak_dbfs"]
    for i, (s, e) in enumerate(bursts):
        a, b = int(s) * sps, min(int(e) * sps, f.n_samples)
        name = f"{prefix}-{i:05d}{ext}"
        np.asarray(f.raw[a:b]).tofile(os.path.join(directory, name))
        names.append(name)
        rows.append(f"{i},{name},{a},{b - a},{a / fs:.9f},{(b - a) / fs:.9f},"
                    f"{mean[i]:.2f},{peak[i]:.2f}")
    with open(os.path.join(directory, f"{prefix}-index.csv"), "w") as fh:
        fh.write("\n".join(rows) + "\n")
    return names


def _pulse_grid(sps: int, beta: float, span: int) -> tuple[np.ndarray, float]:
    """Symbol-spaced time axis for an odd-length centered pulse, and clamped beta."""
    n = span * sps
    n += n % 2
    return (np.arange(n + 1) - n / 2) / sps, min(max(beta, 1e-4), 1.0)


def rc_taps(sps: int, beta: float, span: int) -> np.ndarray:
    """Raised-cosine taps, odd length (centered), unity DC gain.

    The full Nyquist pulse, i.e. rrc_taps convolved with itself. Zero ISI on its
    own, so it is the right receive filter when the transmitter did no pulse
    shaping -- but it is not a matched filter for an RRC-shaped transmitter
    (cascading it with a TX RRC gives an RRC-cubed response, not a Nyquist one).
    """
    t, b = _pulse_grid(sps, beta, span)
    h = np.empty_like(t)
    sing = np.isclose(np.abs(t), 1.0 / (2 * b))
    reg = ~sing
    tr = t[reg]
    h[reg] = np.sinc(tr) * np.cos(np.pi * b * tr) / (1 - (2 * b * tr) ** 2)
    h[sing] = np.pi / 4 * np.sinc(1.0 / (2 * b))
    return h / h.sum()


def rrc_taps(sps: int, beta: float, span: int) -> np.ndarray:
    """Root-raised-cosine taps, odd length (centered), unity DC gain."""
    t, b = _pulse_grid(sps, beta, span)
    h = np.empty_like(t)
    zero = np.isclose(t, 0.0)
    sing = np.isclose(np.abs(t), 1.0 / (4 * b))
    reg = ~(zero | sing)
    tr = t[reg]
    h[reg] = (np.sin(np.pi * tr * (1 - b)) + 4 * b * tr * np.cos(np.pi * tr * (1 + b))) / (
        np.pi * tr * (1 - (4 * b * tr) ** 2)
    )
    h[zero] = 1 - b + 4 * b / np.pi
    h[sing] = b / np.sqrt(2) * (
        (1 + 2 / np.pi) * np.sin(np.pi / (4 * b)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * b))
    )
    return h / h.sum()


def lp_taps(sps: int, cutoff_sps: float, delay: int, beta: float) -> np.ndarray:
    """Kaiser-windowed low-pass sinc, odd length (centered), unity DC gain.

    The anti-aliasing filter you would use to decimate the signal down to one
    sample per symbol, applied without actually decimating: the cutoff sits at
    fs/(2*cutoff_sps), i.e. half the symbol rate when cutoff_sps is the samples
    per symbol of the incoming signal. A sinc with that cutoff has its zero
    crossings one symbol apart, so it is itself a zero-ISI (Nyquist) pulse with
    no excess bandwidth; the Kaiser window trades stopband depth against the
    ringing caused by truncating it.

    `sps` is the samples per symbol of the rate the taps run at, so the filter
    spans 2*delay symbols and its group delay is exactly `delay` symbols.
    """
    n = 2 * max(delay, 1) * sps
    h = sig.firwin(n + 1, 1.0 / max(cutoff_sps, 1.0 + 1e-6), window=("kaiser", max(beta, 0.0)))
    return h / h.sum()


@dataclass
class DemodParams:
    fs: float
    skip: int
    sps: int
    sym_from: int
    sym_to: int  # exclusive
    remove_dc: bool = True
    freq_hz: float = 0.0
    phase_deg: float = 0.0
    oversample: int = 1
    filt: str = "none"  # "none" | "rrc" | "rc" | "lp"
    rc_beta: float = 0.35  # roll-off, shared by "rrc" and "rc"
    rc_span: int = 8  # span in symbols, shared by "rrc" and "rc"
    lp_sps: int = 8  # samples per symbol the cutoff is derived from (input rate)
    lp_delay: int = 4  # group delay / half span, in symbols
    lp_beta: float = 5.0  # Kaiser window shape
    sample_point: int = 0
    eq_on: bool = False  # blind equalizer, on the sampled symbols
    eq_mu: float = 0.01  # adaptation step (normalized, see equalize_cma)
    eq_taps: int = 9  # filter length in symbols
    eq_lookahead: int = 2  # how many of the taps sit on future symbols


@dataclass
class DemodResult:
    t: np.ndarray  # sample index (input rate, relative to skip) of each y sample
    y: np.ndarray  # processed signal at oversampled rate
    L: int  # samples per symbol at oversampled rate
    oversample: int
    idx: np.ndarray  # sampling point indices into y
    sym: np.ndarray  # y[idx], equalized when the blind equalizer is on
    sym_from: int
    sym_raw: np.ndarray | None = None  # y[idx] before equalization
    eq_w: np.ndarray | None = None  # converged equalizer taps (None when off)
    eq_lookahead: int = 0  # taps on future symbols, as the equalizer clamped it

    @property
    def eq_diverged(self) -> bool:
        return self.eq_w is not None and not bool(np.all(np.isfinite(self.eq_w)))

    @property
    def nsym(self) -> int:
        return len(self.idx)


def _mix(x: np.ndarray, n0: int, freq_hz: float, fs: float, phase_deg: float) -> np.ndarray:
    """x[n] * exp(j*(2*pi*f*n/fs + phase)), n counted from the first sample after skip."""
    n = np.arange(n0, n0 + len(x), dtype=np.float64)
    ph = 2 * np.pi * np.mod(freq_hz / fs * n, 1.0) + np.deg2rad(phase_deg)
    return x * np.exp(1j * ph).astype(np.complex64)


def demodulate(f: IQFile, p: DemodParams, mod: Modulation | None = None) -> DemodResult:
    os_ = p.oversample
    L = p.sps * os_
    a = p.skip + p.sym_from * p.sps
    nsym = max(0, min(p.sym_to - p.sym_from, (f.n_samples - a) // p.sps))
    b = a + nsym * p.sps
    if p.filt in ("rrc", "rc"):
        margin = p.rc_span * p.sps + 32
    elif p.filt == "lp":
        margin = 2 * p.lp_delay * p.sps + 32
    else:
        margin = 32
    ra = max(p.skip, a - margin)
    rb = min(f.n_samples, b + margin)

    x = f.read(ra, rb, p.remove_dc)
    x = _mix(x, ra - p.skip, p.freq_hz, p.fs, p.phase_deg)

    if p.filt in ("rrc", "rc", "lp"):
        if p.filt == "rrc":
            h = rrc_taps(L, p.rc_beta, p.rc_span)
        elif p.filt == "rc":
            h = rc_taps(L, p.rc_beta, p.rc_span)
        else:
            # cutoff_sps is scaled by os_ so the cutoff stays at the same
            # physical frequency whatever the oversampling factor
            h = lp_taps(L, p.lp_sps * os_, p.lp_delay, p.lp_beta)
        if os_ > 1:
            u = np.zeros(len(x) * os_, np.complex64)
            u[::os_] = x
            h = h * os_
        else:
            u = x
        y = sig.fftconvolve(u, h.astype(np.float32), mode="same") if len(u) else u
    elif os_ > 1:
        y = sig.resample_poly(x, os_, 1)
    else:
        y = x
    y = np.asarray(y, np.complex64)

    off = (a - ra) * os_
    y = y[off : off + nsym * L]
    t = p.sym_from * p.sps + np.arange(len(y)) / os_
    sp = min(max(p.sample_point, 0), L - 1)
    idx = np.arange(nsym) * L + sp
    sym = y[idx]
    eq_w = None
    if p.eq_on:
        # on the symbols only: y (and so the eye and the I/Q trace) is untouched
        sym, eq_w = equalize_cma(sym, mod or BPSK, p.eq_mu, p.eq_taps, p.eq_lookahead)
    return DemodResult(t=t, y=y, L=L, oversample=os_, idx=idx, sym=sym,
                       sym_from=p.sym_from, sym_raw=y[idx], eq_w=eq_w,
                       eq_lookahead=eq_shape(p.eq_taps, p.eq_lookahead)[1])


# --- modulations ---------------------------------------------------------------


def _pam_levels(nbits: int) -> np.ndarray:
    """Gray-coded PAM levels indexed by the bit word: `levels[word] = level`.

    Levels are the odd integers -(2^n - 1) … +(2^n - 1); adjacent levels differ
    in exactly one bit and the MSB is 1 over the positive half, so the one-bit
    case is the usual BPSK convention ("positive means 1").
    """
    n = 1 << nbits
    i = np.arange(n)
    lv = np.empty(n)
    lv[i ^ (i >> 1)] = 2.0 * i - (n - 1)
    return lv


@lru_cache(maxsize=None)
def _grid(i_bits: int, q_bits: int) -> tuple[np.ndarray, float]:
    """Constellation on the odd-integer grid, indexed by symbol word, and its RMS.

    The word is the I bits (MSB first) followed by the Q bits, so the axes can be
    sliced independently.
    """
    li = _pam_levels(i_bits)
    lq = _pam_levels(q_bits) if q_bits else np.zeros(1)
    p = (li[:, None] + 1j * lq[None, :]).ravel()
    return p, float(np.sqrt(np.mean(np.abs(p) ** 2)))


@dataclass(frozen=True)
class Modulation:
    name: str
    label: str
    i_bits: int  # bits carried by the in-phase axis
    q_bits: int  # bits carried by the quadrature axis (0 for BPSK)
    order: int  # rotational symmetry the carrier / phase estimators raise to
    ref_deg: float  # angle of (ideal symbol)^order, in degrees
    diff: bool  # differential decoding is meaningful (constant modulus)

    @property
    def bits_per_symbol(self) -> int:
        return self.i_bits + self.q_bits

    @property
    def points(self) -> np.ndarray:
        """Ideal constellation indexed by symbol word, normalized to unit mean power."""
        p, rms = _grid(self.i_bits, self.q_bits)
        return p / rms

    @property
    def ambiguity_deg(self) -> float:
        """Rotation the constellation is invariant under (the phase ambiguity)."""
        return 360.0 / self.order


MODULATIONS = {
    "bpsk": Modulation("bpsk", "BPSK (1 bit/symbol)", 1, 0, 2, 0.0, True),
    "qpsk": Modulation("qpsk", "QPSK (2 bits/symbol)", 1, 1, 4, 180.0, True),
    "qam16": Modulation("qam16", "16-QAM (4 bits/symbol)", 2, 2, 4, 180.0, False),
}
BPSK = MODULATIONS["bpsk"]


def _quantize(v: np.ndarray, nbits: int) -> tuple[np.ndarray, np.ndarray]:
    """Nearest Gray-coded PAM level: (word, level) for every value of `v`."""
    n = 1 << nbits
    i = np.clip(np.rint((v + (n - 1)) / 2.0), 0, n - 1).astype(np.int64)
    return i ^ (i >> 1), 2.0 * i - (n - 1)


def decide(sym: np.ndarray, mod: Modulation, gain: float) -> tuple[np.ndarray, np.ndarray]:
    """Hard decisions on `sym` for an ideal constellation of scale `gain`.

    Returns (symbol words, decided ideal symbols at the scale of `sym`). Square
    constellations are separable, so the axes are sliced independently instead of
    searching all points.
    """
    _, rms = _grid(mod.i_bits, mod.q_bits)
    g = max(abs(gain), EPS) / rms  # scale of one grid unit
    s = np.asarray(sym)
    wi, li = _quantize(s.real / g, mod.i_bits)
    if not mod.q_bits:
        return wi, li.astype(np.complex128) * g
    wq, lq = _quantize(s.imag / g, mod.q_bits)
    return (wi << mod.q_bits) | wq, (li + 1j * lq) * g


def estimate_gain(sym: np.ndarray, mod: Modulation, iters: int = 4) -> float:
    """Scale of the unit-power ideal constellation that best fits `sym`.

    Starts from the RMS and refines it by least squares against the hard
    decisions; the first guess is already exact for the constant-modulus
    constellations, whose decisions do not depend on the gain at all.
    """
    s = np.asarray(sym, np.complex128)
    if len(s) == 0:
        return 0.0
    g = float(np.sqrt(np.mean(np.abs(s) ** 2)))
    for _ in range(iters):
        _, ref = decide(s, mod, g)
        den = float(np.mean(np.abs(ref) ** 2))
        if den <= EPS:
            break
        g *= float(np.real(np.mean(s * np.conj(ref)))) / den
    return g


# --- blind equalization --------------------------------------------------------

EQ_WARMUP_PASSES = 1  # sweeps run to converge the taps before the one that is kept


def eq_shape(ntaps: int, lookahead: int) -> tuple[int, int]:
    """Usable (ntaps, lookahead): the reference tap has to stay inside the filter."""
    ntaps = max(1, int(ntaps))
    return ntaps, min(max(int(lookahead), 0), ntaps - 1)


def equalize_cma(sym: np.ndarray, mod: Modulation = BPSK, mu: float = 0.01,
                 ntaps: int = 5, lookahead: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Blind (CMA) equalizer over the sampled symbols: one tap per symbol.

    The output is y[n] = sum_k w[k]*x[n + lookahead - k], so `lookahead` of the
    taps sit on future symbols and the rest on the current and past ones -- a
    non-causal filter, which is what precursor ISI (ringing that arrives before
    the symbol) needs. w starts as a pass-through (a single unit tap at index
    `lookahead`) and is adapted with the constant-modulus (Godard, p=2) cost,
    the step normalized by the energy in the tap window:

        w += mu / ||u||^2 * (R2 - |y|^2) * y * conj(u)

    R2 = E|a|^4 / E|a|^2 of the ideal constellation. The symbols are normalized
    to the fitted constellation gain before adapting and scaled back afterwards,
    so `mu` is independent of the signal level and the output keeps the level of
    the input. CMA is phase blind: it removes ISI and amplitude distortion but
    not a constellation rotation, which stays the job of the phase control.

    The block is swept EQ_WARMUP_PASSES extra times before the sweep whose
    output is returned, so the first symbols already see converged taps while
    the filter still adapts along the block.

    Returns (equalized symbols, final taps). If the adaptation diverges -- `mu`
    too large -- the input symbols are returned unchanged and the taps come back
    non-finite.
    """
    s = np.asarray(sym, np.complex128)
    ntaps, lookahead = eq_shape(ntaps, lookahead)
    w = np.zeros(ntaps, np.complex128)
    w[lookahead] = 1.0
    g = estimate_gain(s, mod) if len(s) else 0.0
    if len(s) == 0 or ntaps == 1 or g <= EPS:
        return np.asarray(sym, np.complex64), w

    z = s / g
    p = mod.points
    r2 = float(np.mean(np.abs(p) ** 4) / np.mean(np.abs(p) ** 2))
    zp = np.zeros(len(z) + ntaps - 1, np.complex128)
    head = ntaps - 1 - lookahead
    zp[head : head + len(z)] = z
    # u[n][k] = x[n + lookahead - k]; the windows are reversed so k counts back
    u = np.lib.stride_tricks.sliding_window_view(zp, ntaps)[:, ::-1]
    uc = np.conj(u)
    # ||u[n]||^2 from the running energy, instead of re-summing the window
    cum = np.concatenate([[0.0], np.cumsum(np.abs(zp) ** 2)])
    step = mu / (EPS + cum[ntaps:] - cum[:-ntaps])

    out = np.empty(len(z), np.complex128)
    with np.errstate(over="ignore", invalid="ignore"):  # a diverging w is caught below
        for _ in range(EQ_WARMUP_PASSES + 1):
            for n in range(len(z)):
                y = complex(np.dot(w, u[n]))
                out[n] = y
                w += (step[n] * (r2 - (y.real * y.real + y.imag * y.imag)) * y) * uc[n]
            if not np.all(np.isfinite(w)):
                return np.asarray(sym, np.complex64), w
    return (out * g).astype(np.complex64), w


# --- estimators --------------------------------------------------------------


def estimate_freq(f: IQFile, p: DemodParams, mod: Modulation = BPSK,
                  max_len: int = 1 << 20) -> float:
    """Carrier offset in Hz from the M-th power spectral line (M = `mod.order`).

    Raising the signal to the M-th power strips an M-PSK modulation (and leaves a
    line for square QAM too, whose M-th moment is non-zero); the line sits at M
    times the offset, so the estimate is unambiguous for |offset| < fs/(2·M).
    """
    a = p.skip + p.sym_from * p.sps
    b = min(p.skip + p.sym_to * p.sps, a + max_len)
    x = f.read(a, b, p.remove_dc)
    if len(x) < 16:
        return 0.0
    z = (x.astype(np.complex128) ** mod.order) * np.hanning(len(x))
    nfft = 1 << int(np.ceil(np.log2(len(z) * 4)))
    P = np.abs(np.fft.fft(z, nfft))
    k = int(np.argmax(P))
    lp = np.log(P[[(k - 1) % nfft, k, (k + 1) % nfft]] + EPS)
    den = lp[0] - 2 * lp[1] + lp[2]
    delta = 0.5 * (lp[0] - lp[2]) / den if den != 0 else 0.0
    fm = (k + delta) / nfft
    if fm >= 0.5:
        fm -= 1.0
    return fm * p.fs / mod.order


def estimate_timing(res: DemodResult) -> int:
    """Sampling point (0..L-1) half a symbol away from the symbol transitions.

    Transitions are located as the circular centroid of the sample-to-sample
    change energy; unlike max-energy timing this also works for unfiltered
    rectangular (constant envelope) modulations.
    """
    if res.nsym < 2:
        return 0
    L = res.L
    d = np.zeros(res.nsym * L, np.float64)
    d[1:] = np.abs(np.diff(res.y[: res.nsym * L])) ** 2
    prof = d.reshape(res.nsym, L).mean(axis=0)
    k = np.arange(L)
    centroid = np.angle(np.sum(prof * np.exp(2j * np.pi * k / L))) * L / (2 * np.pi)
    # d[n] measures the change between n-1 and n, i.e. at n - 0.5
    return int(np.round(centroid - 0.5 + L / 2)) % L


def estimate_phase_deg(sym: np.ndarray, mod: Modulation = BPSK) -> float:
    """Residual constellation phase in degrees, modulo the phase ambiguity.

    The M-th power of the symbols concentrates at `mod.ref_deg` when the
    constellation is aligned, so the residual is the M-th of the deviation from
    it, wrapped into ±180/M degrees.
    """
    s = np.asarray(sym, np.complex128)
    if len(s) == 0:
        return 0.0
    a = np.rad2deg(np.angle(np.sum(s ** mod.order))) - mod.ref_deg
    span = mod.ambiguity_deg
    return float((a / mod.order + span / 2) % span - span / 2)


@dataclass
class Metrics:
    nsym: int
    amplitude: float  # scale of the unit-power ideal constellation, in FS
    power_dbfs: float
    mer_db: float
    evm_pct: float
    residual_phase_deg: float


def symbol_metrics(sym: np.ndarray, mod: Modulation = BPSK) -> Metrics | None:
    if len(sym) == 0:
        return None
    s = np.asarray(sym, np.complex128)
    g = estimate_gain(s, mod)
    _, ref = decide(s, mod, g)
    err = float(np.mean(np.abs(s - ref) ** 2))
    pwr = float(np.mean(np.abs(ref) ** 2))
    return Metrics(
        nsym=len(s),
        amplitude=g,
        power_dbfs=float(db(np.mean(np.abs(s) ** 2))),
        mer_db=float(db(pwr / max(err, EPS))),
        evm_pct=float(100.0 * np.sqrt(err / max(pwr, EPS))),
        residual_phase_deg=estimate_phase_deg(s, mod),
    )


def slice_bits(sym: np.ndarray, mod: Modulation = BPSK, differential: bool = False,
               invert: bool = False) -> np.ndarray:
    """Hard-decision bits, `mod.bits_per_symbol` per symbol, I bits then Q, MSB first.

    `differential` decodes the rotation from one symbol to the next instead of the
    symbols themselves, which removes the phase ambiguity entirely. The word is
    the one whose point carries that rotation away from point 0, so "no change"
    decodes as all-zeros (for BPSK: the XOR of consecutive hard bits). It only
    makes sense for the constant-modulus constellations.
    """
    s = np.asarray(sym, np.complex128)
    if differential:
        if len(s) < 2:
            return np.zeros(0, np.uint8)
        s = s[1:] * np.conj(s[:-1]) * mod.points[0]
    w, _ = decide(s, mod, estimate_gain(s, mod))
    k = mod.bits_per_symbol
    b = ((w[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).ravel()
    if invert:
        b = b ^ 1
    return b


# --- eye diagram ---------------------------------------------------------------


def eye_segments(res: DemodResult, width_syms: int, max_traces: int):
    """Windows of width_syms*L samples centered on each sampling point.

    Returns (offsets, segments) where offsets are in oversampled samples relative
    to the sampling point and segments is a (n_traces, len(offsets)) complex array.
    """
    half = (width_syms * res.L) // 2
    offs = np.arange(-half, half + 1)
    c = res.idx[(res.idx - half >= 0) & (res.idx + half < len(res.y))]
    if len(c) > max_traces:
        c = c[np.linspace(0, len(c) - 1, max_traces).astype(int)]
    if len(c) == 0:
        return offs, np.zeros((0, len(offs)), np.complex64)
    return offs, res.y[c[:, None] + offs[None, :]]


def eye_density(offs: np.ndarray, seg: np.ndarray, y_range: tuple[float, float],
                nx: int = 300, ny: int = 240, upsample: int = 4) -> np.ndarray:
    """2-D hit histogram (nx, ny) of linearly interpolated eye traces."""
    H = np.zeros((nx, ny), np.float32)
    if seg.shape[0] == 0 or len(offs) < 2:
        return H
    x0, x1 = float(offs[0]), float(offs[-1])
    xf = np.linspace(x0, x1, nx * upsample)
    pos = xf - x0
    i0 = np.clip(np.floor(pos).astype(int), 0, len(offs) - 2)
    w = (pos - i0).astype(np.float32)
    v = seg[:, i0] * (1 - w) + seg[:, i0 + 1] * w
    y0, y1 = y_range
    ix = np.clip(((xf - x0) / (x1 - x0) * nx).astype(int), 0, nx - 1)
    iy = np.floor((v - y0) / (y1 - y0) * ny).astype(int)
    ok = (iy >= 0) & (iy < ny)
    flat = (np.broadcast_to(ix, iy.shape) * ny + iy)[ok]
    H += np.bincount(flat, minlength=nx * ny).reshape(nx, ny)
    return H
