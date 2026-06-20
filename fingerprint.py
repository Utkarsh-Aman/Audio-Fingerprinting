"""
fingerprint.py — Audio Fingerprinting Engine
EE200 · Signals, Systems & Networks

Shazam-style pipeline:
    spectrogram  →  constellation (peak picking)  →  hash pairing  →  database matching
"""

import numpy as np
import librosa
from scipy.ndimage import maximum_filter
from collections import defaultdict
import os, glob, time


# ── tuneable parameters ──────────────────────────────────
SAMPLE_RATE       = 22050
N_FFT             = 4096
HOP_LENGTH        = 2048
PEAK_NEIGHBORHOOD = 10      # size of the local-max filter window
PEAK_THRESHOLD    = -50      # dB below the spectrogram max to keep
FAN_OUT           = 10       # number of target peaks per anchor
TARGET_T_MIN      = 1        # min frame gap when pairing
TARGET_T_MAX      = 50       # max frame gap when pairing (~4.6 s)
FREQ_LIMIT        = 512      # only use the first N frequency bins


# ── audio loading ────────────────────────────────────────

def load_audio(filepath, sr=SAMPLE_RATE):
    """Load an audio file, convert to mono, resample to *sr*."""
    y, _ = librosa.load(filepath, sr=sr, mono=True)
    return y


# ── spectrogram ──────────────────────────────────────────

def compute_spectrogram(y, sr=SAMPLE_RATE, n_fft=N_FFT, hop=HOP_LENGTH):
    """
    STFT magnitude spectrogram in dB (0 dB = loudest).
    Returns (S_db, freqs, times).
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    return S_db, freqs, times


# ── peak detection (constellation map) ───────────────────

def find_peaks(S_db, neighborhood=PEAK_NEIGHBORHOOD,
               threshold=PEAK_THRESHOLD, freq_limit=FREQ_LIMIT):
    """
    Local maxima in the lower-frequency portion of the spectrogram.
    Returns list of (time_frame, freq_bin) tuples.
    """
    S_cut = S_db[:freq_limit, :]

    local_max = maximum_filter(S_cut, size=neighborhood)
    mask = (S_cut == local_max) & (S_cut > threshold)

    fi, ti = np.where(mask)                       # freq indices, time indices
    return list(zip(ti.astype(int), fi.astype(int)))


# ── hash generation ──────────────────────────────────────

def generate_hashes(peaks, fan_out=FAN_OUT,
                    t_min=TARGET_T_MIN, t_max=TARGET_T_MAX):
    """
    Pair nearby peaks into hashes: (f_anchor, f_target, Δt).
    Returns [(hash_tuple, anchor_time), …].
    """
    sorted_peaks = sorted(peaks, key=lambda p: p[0])
    hashes = []
    for i, (t1, f1) in enumerate(sorted_peaks):
        count = 0
        for j in range(i + 1, len(sorted_peaks)):
            t2, f2 = sorted_peaks[j]
            dt = t2 - t1
            if dt < t_min:
                continue
            if dt > t_max:
                break
            hashes.append(((f1, f2, dt), t1))
            count += 1
            if count >= fan_out:
                break
    return hashes


# ── database building ────────────────────────────────────

def index_songs(song_dir, progress_cb=None):
    """
    Index every .mp3 in *song_dir*.
    Returns (hash_db, metadata).
        hash_db  : { hash_tuple : [(song_name, anchor_time), …] }
        metadata : { song_name : {duration, num_peaks, num_hashes, peaks} }
    """
    hash_db  = defaultdict(list)
    metadata = {}
    songs    = sorted(glob.glob(os.path.join(song_dir, "*.mp3")))

    for idx, path in enumerate(songs):
        name = os.path.splitext(os.path.basename(path))[0]
        if progress_cb:
            progress_cb(idx, len(songs), name)

        y     = load_audio(path)
        dur   = len(y) / SAMPLE_RATE
        S_db, _, _ = compute_spectrogram(y)
        peaks = find_peaks(S_db)
        hashes = generate_hashes(peaks)

        for h, t1 in hashes:
            hash_db[h].append((name, t1))

        metadata[name] = dict(
            duration=dur,
            num_peaks=len(peaks),
            num_hashes=len(hashes),
            peaks=peaks,
        )

    if progress_cb:
        progress_cb(len(songs), len(songs), "done")

    return dict(hash_db), metadata


# ── identification ───────────────────────────────────────

def identify(y, hash_db, metadata, sr=SAMPLE_RATE):
    """
    Identify a query audio array against the pre-built database.
    Returns a dict with the prediction, scores, intermediate arrays, and timings.
    """
    timings = {}

    # step 1 — spectrogram
    t0 = time.perf_counter()
    S_db, freqs, times = compute_spectrogram(y, sr=sr)
    timings["spectrogram"] = (time.perf_counter() - t0) * 1000

    # step 2 — constellation
    t0 = time.perf_counter()
    peaks = find_peaks(S_db)
    timings["constellation"] = (time.perf_counter() - t0) * 1000

    # step 3 — hashing
    t0 = time.perf_counter()
    hashes = generate_hashes(peaks)
    timings["hashing"] = (time.perf_counter() - t0) * 1000

    # step 4 — database lookup
    t0 = time.perf_counter()
    offsets = defaultdict(lambda: defaultdict(int))
    for h, tq in hashes:
        if h in hash_db:
            for song, tdb in hash_db[h]:
                offsets[song][tdb - tq] += 1
    timings["db_lookup"] = (time.perf_counter() - t0) * 1000

    # step 5 — scoring (find tallest offset-histogram spike per song)
    t0 = time.perf_counter()
    scores   = {}
    best_off = {}
    for song, hist in offsets.items():
        off = max(hist, key=hist.get)
        scores[song]   = hist[off]
        best_off[song] = dict(offset=off, histogram=dict(hist))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    timings["scoring"] = (time.perf_counter() - t0) * 1000
    timings["total"]   = sum(timings.values())

    # best match
    if ranked:
        prediction = ranked[0][0]
        score      = ranked[0][1]
        runner_up  = ranked[1][1] if len(ranked) > 1 else 0
    else:
        prediction, score, runner_up = None, 0, 0

    return dict(
        prediction   = prediction,
        score        = score,
        runner_up    = runner_up,
        ratio        = score / runner_up if runner_up else float("inf"),
        ranked       = ranked[:10],
        best_offsets = best_off,
        spectrogram  = S_db,
        freqs        = freqs,
        times        = times,
        peaks        = peaks,
        hashes       = hashes,
        timings      = timings,
    )
