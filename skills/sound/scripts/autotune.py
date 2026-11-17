#!/usr/bin/env python3
# Copyright 2026 Creative Stack Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a real-time REAPER vocal Auto-Tune + A/B comparison studio and pitch benchmark.

    # 1. verify audio interfaces, REAPER buffer latency, Graillon 3 VST3, Demucs & Seed-VC
    python3 autotune.py --check-setup

    # 2. build a complete 10-track REAPER Auto-Tune A/B project from a karaoke backing track
    python3 autotune.py --karaoke-backing out/over-each-other/backing-track.wav \\
        --video out/over-each-other/karaoke-video.mp4 \\
        --markers out/over-each-other/lyrics.md \\
        --bpm 98 --apply-live -o out/over-each-other/autotune-studio.rpp

    # 3. split a full reference song with Demucs, extract the vocal melody MIDI guide
    #    for hard MIDI pitch-locking (Graillon 3 + ReaVocode), and wire the REAPER session
    python3 autotune.py -i song.mp3 --apply-live -o out/song/autotune.rpp

    # 4. generate a synthetic badly-out-of-tune vocal demo, render all rigs (Without Auto-Tune,
    #    Stock ReaTune, Stock ReaVocode, Graillon 3 MIDI-Lock, and Seed-VC AI Singing Voice),
    #    and benchmark pitch accuracy in cents
    python3 autotune.py --synth-demo --benchmark --ai-voice auto --apply-live \\
        -o out/autotune-studio/autotune-demo.rpp

Rigs wired in every generated `.rpp` project:
  - Track 1 (`01 - Dry Mic Input`): Record-armed input (`MAINSEND 0 0`) feeding Pre-FX sends to:
  - Track 2 (`02 - Rig 0: Bypass (No Auto-Tune)`): Identical studio vocal chain (ReaGate + ReaEQ +
    Dual ReaComp + De-esser + Echo/Reverb sends), zero pitch shift — the control baseline.
  - Track 3 (`03 - Rig 1A: Stock ReaTune (Scale Snap)`): Cockos ReaTune automatic scale pitch
    correction (`20 ms` attack, exact 12-bit scale mask) + full studio vocal chain.
  - Track 4 (`04 - Rig 1B: Stock ReaVocode (MIDI Vocoder)`): 4-channel track combining Mic voice
    (Ch 1/2 Detector) with MIDI-driven ReaSynth carrier (Ch 3/4 from Track 6) for 100% locked pitch.
  - Track 5 (`05 - Rig 2: Graillon 3 MIDI-Lock`): Auburn Sounds Graillon 3 Free VST3 receiving
    both the song's scale mask AND live MIDI notes from Track 6 (`MIDI In Enabled = ON`).
  - Track 6 (`06 - Guide Melody MIDI + Synth Carrier`): Holds `<stem>.melody.mid` + `VSTi: ReaSynth`,
    routing MIDI to Track 5 (Graillon 3) and carrier audio to Track 4 Ch 3/4 (ReaVocode).
  - Tracks 7–10: Tempo-synced `Vocal Echo`, `Vocal Reverb`, `Backing Track`, and `Lyrics Video`.
"""

import argparse
import base64
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import soundlib as S  # noqa: E402
import hostenv  # noqa: E402
import session as sess  # noqa: E402
import karaoke as kar  # noqa: E402


PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

ENHARMONIC_MAP = {
    "DB": "C#",
    "EB": "D#",
    "FB": "E",
    "GB": "F#",
    "AB": "G#",
    "BB": "A#",
    "CB": "B",
    "E#": "F",
    "B#": "C",
}

SCALE_INTERVALS = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "chromatic": list(range(12)),
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
}

# Krumhansl-Schmuckler key profiles (C=0 .. B=11)
KS_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
KS_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

RIG_TRACK_NAMES = {
    "bypass": "02 - Rig 0: Bypass (No Auto-Tune)",
    "reatune": "03 - Rig 1A: Stock ReaTune (Scale Snap)",
    "reavocode": "04 - Rig 1B: Stock ReaVocode (MIDI Vocoder)",
    "graillon": "05 - Rig 2: Graillon 3 MIDI-Lock",
}


def ensure_audio_ml_venv():
    """Re-exec under `.venv/bin/python` if the current interpreter lacks `librosa` or `mido`."""
    try:
        import librosa  # noqa: F401
        import mido  # noqa: F401
        return
    except ImportError:
        pass
    venv_py = hostenv.venv_python()
    if venv_py and Path(venv_py).resolve() != Path(sys.executable).resolve():
        os.execv(str(venv_py), [str(venv_py), str(Path(__file__).resolve()), *sys.argv[1:]])
    S.die("librosa and mido are required. Install via: uv pip install --python .venv/bin/python librosa mido")


def normalize_note_name(name):
    """Normalize a note root string (`'c'`, `'Bb'`, `'F#'`, `'Eb'`) to `PITCH_CLASSES`."""
    raw = str(name or "C").strip().upper().replace("♯", "#").replace("♭", "B")
    if not raw:
        return "C"
    if raw in PITCH_CLASSES:
        return raw
    if raw in ENHARMONIC_MAP:
        return ENHARMONIC_MAP[raw]
    raise ValueError(f"unknown pitch class {name!r} (expected one of {PITCH_CLASSES})")


def scale_pitch_classes(root_name="C", mode="major"):
    """Return a 12-element list of `0`/`1` flags for `C=0 .. B=11` in `(root_name, mode)`."""
    root = normalize_note_name(root_name)
    root_idx = PITCH_CLASSES.index(root)
    mode_key = str(mode or "major").strip().lower()
    if mode_key not in SCALE_INTERVALS:
        raise ValueError(f"unknown scale mode {mode!r} (expected one of {sorted(SCALE_INTERVALS)})")
    mask = [0] * 12
    for step in SCALE_INTERVALS[mode_key]:
        mask[(root_idx + step) % 12] = 1
    return mask


def scale_note_names(root_name="C", mode="major"):
    """Return the human-readable list of allowed notes in `(root_name, mode)`."""
    mask = scale_pitch_classes(root_name, mode)
    return [PITCH_CLASSES[i] for i, flag in enumerate(mask) if flag]


def reatune_scale_bitmask(scale_mask_c_to_b):
    """Convert a 12-element `C..B` scale mask into Cockos ReaTune's 12-bit integer mask.

    Empirically verified against REAPER 7's `reatune.vst.dylib` state chunk:
    ReaTune indexes its 12 scale checkboxes starting at `A = bit 0`, `A# = bit 1`,
    `B = bit 2`, `C = bit 3`, ..., `G# = bit 11`. Thus pitch class `pc` (`C=0..B=11`)
    maps to `bit = (pc + 3) % 12`.
    """
    if len(scale_mask_c_to_b) != 12:
        raise ValueError("scale_mask_c_to_b must have 12 elements (C..B)")
    bitmask = 0
    for pc, allowed in enumerate(scale_mask_c_to_b):
        if allowed:
            bit = (pc + 3) % 12
            bitmask |= (1 << bit)
    return bitmask


def build_reatune_state_b64(scale_mask_c_to_b, attack_ms=20, window_ms=30,
                            ref_hz=440, auto_enable=True):
    """Build the base64 chunk lines for `<VST "VST: ReaTune (Cockos)" ...>`.

    ReaTune stores its Automatic Pitch Correction state in a 104-byte (`26 x uint32`)
    little-endian struct following its fixed 76-byte VST header line:
      - `words[2]` (offset 8):  Detector window ms (`30`)
      - `words[3]` (offset 12): Reference A4 frequency Hz (`440`)
      - `words[4]` (offset 16): `1` = Automatic pitch correction enabled (`0` = off)
      - `words[5]` (offset 20): 12-bit scale bitmask (`bit 0 = A .. bit 11 = G#`)
      - `words[6]` (offset 24): Attack time ms (`20` for fast modern pop correction)
    """
    header_b64 = (
        "bnRlcu5e7f4EAAAAAQAAAAAAAAACAAAAAAAAAAQAAAAAAAAACAAAAAAAAAACAAAA"
        "AQAAAAAAAAACAAAAAAAAAGgAAAABAAAAAAAQAA=="
    )
    bitmask = reatune_scale_bitmask(scale_mask_c_to_b)
    words = [
        0,
        0,
        int(window_ms),
        int(ref_hz),
        1 if auto_enable else 0,
        int(bitmask),
        int(attack_ms),
        50,
        50,
        0,
        0,
        0,
        0,
        1090519040,
        1086324736,
        0,
        1090519040,
        1,
        1,
        11,
        1,
        0,
        0,
        0,
        0,
        0,
    ]
    payload = struct.pack("<26I", *words)
    b64 = base64.b64encode(payload).decode("ascii")
    return [header_b64, b64[:128], b64[128:], "AAAQAAAA"]


def build_reavocode_state_b64(wet=1.0, dry=0.0, band_norm=0.5, invert=0.0,
                              stereo=1.0, mod_dry=0.0):
    """Build the base64 chunk lines for `<VST "VST: ReaVocode (Cockos)" ...>`.

    ReaVocode takes the Voice Detector on main inputs (Ch 1/2) and the Synth Carrier
    to be modulated on the second inputs (Ch 3/4) when `invert=0.0`.
    """
    header_b64 = (
        "ZHZlcu5e7f4EAAAAAQAAAAAAAAACAAAAAAAAAAQAAAAAAAAACAAAAAAAAAACAAAA"
        "AQAAAAAAAAACAAAAAAAAABwAAAABAAAAAAAQAA=="
    )
    payload = struct.pack(
        "<I6f",
        6,
        float(wet),
        float(dry),
        float(band_norm),
        float(invert),
        float(stereo),
        float(mod_dry),
    )
    return [header_b64, base64.b64encode(payload).decode("ascii"), "AAAQAAAA"]


def build_reasynth_state_b64(attack=0.004, release=0.010, square_mix=0.35,
                             saw_mix=0.85, triangle_mix=0.0, volume=0.707946):
    """Build the base64 chunk lines for `<VST "VSTi: ReaSynth (Cockos)" ...>`.

    A vocoder carrier needs rich upper harmonics (saw + pulse waves) so formant
    consonants and vowels articulate clearly.
    """
    header_b64 = "eXNlcu9e7f4AAAAAAgAAAAEAAAAAAAAAAgAAAAAAAABEAAAAAAAAAAAAEAA="
    payload = struct.pack(
        "<2I15f",
        3735928559,
        3735924749,
        float(attack),
        float(release),
        float(square_mix),
        float(saw_mix),
        float(triangle_mix),
        float(volume),
        0.06660444,
        0.0,
        0.5,
        1.0,
        1.0,
        0.5,
        0.0,
        0.0,
        0.0,
    )
    return [header_b64, base64.b64encode(payload).decode("ascii"), "AAAQAAAA"]


def build_graillon3_state_b64(scale_mask_c_to_b, midi_enable=True,
                              smooth_norm=0.15, inertia_norm=0.05):
    """Build the base64 chunk lines for `<VST "VST3: Graillon 3 (Auburn Sounds)" ...>`.

    Empirically verified against Graillon 3 v3.2.0's 312-byte VST3 state chunk:
    30-byte VST3 header + 66 little-endian `float32` parameters (`params[1..66]`)
    + 18-byte zero trailer.
    """
    if len(scale_mask_c_to_b) != 12:
        raise ValueError("scale_mask_c_to_b must have 12 elements (C..B)")
    header_b64 = "V6r6P+5e7f4CAAAAAQAAAAAAAAACAAAAAAAAAAIAAAABAAAAAAAAAAIAAAAAAAAAMgEAAAEAAAAAAAAA"
    prefix = bytes.fromhex("220100000100000000000b20ba9201000000000000000002030000000000")
    params = [0.0] * 66
    params[2 - 1] = 0.707946       # Wet Mix: 0.0 dB
    params[7 - 1] = 1.0            # Global Amount: 100%
    params[8 - 1] = 0.0            # Dry Mix: -inf dB
    params[9 - 1] = 0.35           # Low Cut: ~80 Hz
    params[10 - 1] = 0.707946      # Output Gain: 0.0 dB
    params[14 - 1] = 1.0           # PTM mode
    params[15 - 1] = 0.5           # Mod Octave: 0
    params[17 - 1] = 0.5           # Pitch Shift: 0.00 st
    params[18 - 1] = 1.0           # Formant Amount: 100%
    params[19 - 1] = 1.0           # Corr. Amount: 100%
    params[20 - 1] = float(smooth_norm)   # Smooth (fast snap)
    params[21 - 1] = 1.0           # Snap Max: 8.00 semitones
    params[22 - 1] = 0.504682      # Reference: 440 Hz
    params[23 - 1] = float(inertia_norm)  # Inertia
    for idx, flag in enumerate(scale_mask_c_to_b):
        params[24 - 1 + idx] = 1.0 if flag else 0.0  # Allow C .. Allow B
        params[36 - 1 + idx] = 0.5                   # Bend C .. Bend B
    params[52 - 1] = 1.0           # Pitch LR Link: yes
    params[53 - 1] = 0.5           # Formant Shift: 0.00
    params[54 - 1] = 0.5           # Pitch Engine: Graillon3
    params[56 - 1] = 1.0 if midi_enable else 0.0  # MIDI In Enabled
    params[60 - 1] = 0.5           # Doubler: 0.00
    params[62 - 1] = 1.0           # FX Enabled: yes
    params[63 - 1] = 0.0           # PTM Enabled: no (disable ring-mod)
    params[64 - 1] = 1.0           # Correction: yes
    params[65 - 1] = 1.0           # Formant: yes

    payload = prefix + struct.pack("<66f", *params) + (b"\x00" * 18)
    b64 = base64.b64encode(payload).decode("ascii")
    return [header_b64, b64[:256], b64[256:408], b64[408:]]


def hz_to_midi_float(hz):
    """Convert frequency in Hz to fractional MIDI note number (`A4 = 440 Hz -> 69.0`)."""
    val = float(hz or 0.0)
    if val <= 0.0 or math.isnan(val) or math.isinf(val):
        return None
    return 69.0 + 12.0 * math.log2(val / 440.0)


def midi_to_hz(midi_note):
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((float(midi_note) - 69.0) / 12.0))


def nearest_scale_midi(midi_float, scale_mask_c_to_b=None, min_midi=36, max_midi=84):
    """Snap a fractional MIDI value to the nearest integer MIDI note allowed by `scale_mask_c_to_b`."""
    if midi_float is None or math.isnan(midi_float):
        return None
    candidates = [
        n for n in range(int(min_midi), int(max_midi) + 1)
        if not scale_mask_c_to_b or scale_mask_c_to_b[n % 12]
    ]
    if not candidates:
        candidates = list(range(int(min_midi), int(max_midi) + 1))
    return min(candidates, key=lambda n: (abs(float(midi_float) - n), n))


def quantize_f0_to_midi_events(times, f0_hz, voiced_mask, scale_mask_c_to_b=None,
                               min_note_sec=0.08, gap_sec=0.008):
    """Quantize frame-level `(times, f0_hz, voiced_mask)` into monophonic MIDI note events.

    Returns `[(start_sec, end_sec, midi_note), ...]` with strict non-overlap
    (`end_sec <= next_start_sec - gap_sec`) so monophonic MIDI pitch-lockers
    (Auburn Sounds Graillon 3 and ReaSynth) never receive overlapping note-ons.
    """
    n_frames = min(len(times), len(f0_hz), len(voiced_mask))
    if n_frames == 0:
        return []

    frame_dt = (float(times[1]) - float(times[0])) if n_frames > 1 else 0.02
    frame_dt = max(0.005, frame_dt)

    snapped = []
    for i in range(n_frames):
        is_voiced = bool(voiced_mask[i])
        hz = float(f0_hz[i]) if f0_hz[i] is not None and not math.isnan(float(f0_hz[i])) else 0.0
        if not is_voiced or hz <= 45.0:
            snapped.append(None)
            continue
        mf = hz_to_midi_float(hz)
        snapped.append(nearest_scale_midi(mf, scale_mask_c_to_b))

    # 5-frame median smoothing over voiced frames to suppress consonant/octave spikes
    smoothed = list(snapped)
    for i in range(n_frames):
        if snapped[i] is None:
            continue
        window = [
            snapped[j] for j in range(max(0, i - 2), min(n_frames, i + 3))
            if snapped[j] is not None
        ]
        if window:
            window.sort()
            smoothed[i] = window[len(window) // 2]

    # Collapse contiguous runs into raw segments
    raw_segments = []
    cur_note = None
    cur_start = None
    last_t = None
    for i in range(n_frames):
        t = float(times[i])
        note = smoothed[i]
        if note is not None:
            if cur_note == note:
                last_t = t
            else:
                if cur_note is not None:
                    raw_segments.append([cur_start, last_t + frame_dt, int(cur_note)])
                cur_note = note
                cur_start = t
                last_t = t
        else:
            if cur_note is not None:
                raw_segments.append([cur_start, last_t + frame_dt, int(cur_note)])
                cur_note = None
                cur_start = None
    if cur_note is not None:
        raw_segments.append([cur_start, last_t + frame_dt, int(cur_note)])

    # Merge tiny micro-gaps (<= 0.045s) between identical consecutive notes
    merged = []
    for seg in raw_segments:
        if merged and merged[-1][2] == seg[2] and (seg[0] - merged[-1][1]) <= 0.045:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)

    # Filter out short blips and enforce monophonic gap_sec
    filtered = [seg for seg in merged if (seg[1] - seg[0]) >= float(min_note_sec)]
    events = []
    for idx, (s_sec, e_sec, note) in enumerate(filtered):
        if idx + 1 < len(filtered):
            next_start = filtered[idx + 1][0]
            e_sec = min(e_sec, next_start - float(gap_sec))
        if e_sec > s_sec + 0.01:
            events.append((round(float(s_sec), 4), round(float(e_sec), 4), int(note)))
    return events


def compute_pitch_benchmark_metrics(times, f0_hz, voiced_mask, scale_mask_c_to_b=None,
                                    target_events=None):
    """Measure pitch accuracy in cents against the musical scale and/or target melody.

    Returns a dictionary containing:
      - `voiced_frames`: count of detected voiced frames
      - `scale_median_abs_cents`, `scale_mean_abs_cents`, `scale_pct_within_25_cents`, `scale_pct_within_50_cents`
      - `target_median_abs_cents`, `target_mean_abs_cents`, `target_pct_within_25_cents`, `target_pct_within_50_cents`
      - `median_abs_cents`, `mean_abs_cents`, `pct_within_25_cents`, `pct_within_50_cents`
        (primary summary metric: target melody accuracy when `target_events` is provided, else scale accuracy)
    """
    n_frames = min(len(times), len(f0_hz), len(voiced_mask))
    scale_cents = []
    target_cents = []

    for i in range(n_frames):
        if not voiced_mask[i]:
            continue
        hz_val = f0_hz[i]
        if hz_val is None or math.isnan(float(hz_val)) or float(hz_val) <= 45.0:
            continue
        mf = hz_to_midi_float(float(hz_val))
        if mf is None:
            continue

        snap_scale = nearest_scale_midi(mf, scale_mask_c_to_b)
        if snap_scale is not None:
            scale_cents.append(abs(mf - snap_scale) * 100.0)

        if target_events:
            t = float(times[i])
            active_note = None
            for s_sec, e_sec, note in target_events:
                if s_sec - 1e-4 <= t <= e_sec + 1e-4:
                    active_note = note
                    break
            if active_note is None:
                best_dist = 0.025
                for s_sec, e_sec, note in target_events:
                    dist = min(abs(t - s_sec), abs(t - e_sec))
                    if dist <= best_dist:
                        best_dist = dist
                        active_note = note
            if active_note is not None:
                # Compare pitch modulo octave when an octave harmonic or vocoder carrier shifts octave,
                # while keeping exact semitone distance within [-6, +6] semitones of target pitch class.
                diff_semi = (mf - float(active_note) + 6.0) % 12.0 - 6.0
                target_cents.append(abs(diff_semi) * 100.0)

    def _summarize(arr):
        if not arr:
            return {
                "count": 0,
                "median_abs_cents": 0.0,
                "mean_abs_cents": 0.0,
                "pct_within_25_cents": 0.0,
                "pct_within_50_cents": 0.0,
            }
        s = sorted(arr)
        mid = len(s) // 2
        median = s[mid] if len(s) % 2 == 1 else 0.5 * (s[mid - 1] + s[mid])
        mean = sum(s) / len(s)
        w25 = 100.0 * sum(1 for c in s if c <= 25.0) / len(s)
        w50 = 100.0 * sum(1 for c in s if c <= 50.0) / len(s)
        return {
            "count": len(s),
            "median_abs_cents": round(median, 2),
            "mean_abs_cents": round(mean, 2),
            "pct_within_25_cents": round(w25, 1),
            "pct_within_50_cents": round(w50, 1),
        }

    sc_sum = _summarize(scale_cents)
    tg_sum = _summarize(target_cents)
    primary = tg_sum if (target_events and tg_sum["count"] > 0) else sc_sum

    return {
        "voiced_frames": sc_sum["count"],
        "median_abs_cents": primary["median_abs_cents"],
        "mean_abs_cents": primary["mean_abs_cents"],
        "pct_within_25_cents": primary["pct_within_25_cents"],
        "pct_within_50_cents": primary["pct_within_50_cents"],
        "scale_median_abs_cents": sc_sum["median_abs_cents"],
        "scale_mean_abs_cents": sc_sum["mean_abs_cents"],
        "scale_pct_within_25_cents": sc_sum["pct_within_25_cents"],
        "scale_pct_within_50_cents": sc_sum["pct_within_50_cents"],
        "target_median_abs_cents": tg_sum["median_abs_cents"],
        "target_mean_abs_cents": tg_sum["mean_abs_cents"],
        "target_pct_within_25_cents": tg_sum["pct_within_25_cents"],
        "target_pct_within_50_cents": tg_sum["pct_within_50_cents"],
    }


def detect_key_and_scale(audio_path):
    """Estimate `(root_note, mode, confidence)` from an audio file using CQT chroma + K-S profiles."""
    ensure_audio_ml_venv()
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=120.0)
    if len(y) < 2048:
        return "C", "major", 0.5
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vec = np.mean(chroma, axis=1)
    if np.sum(chroma_vec) <= 1e-8:
        return "C", "major", 0.5
    chroma_norm = (chroma_vec - np.mean(chroma_vec)) / (np.std(chroma_vec) + 1e-8)

    best_score = -999.0
    best_root = "C"
    best_mode = "major"
    for mode_name, profile in (("major", KS_MAJOR_PROFILE), ("minor", KS_MINOR_PROFILE)):
        prof = np.array(profile, dtype=float)
        prof_norm = (prof - np.mean(prof)) / (np.std(prof) + 1e-8)
        for root_idx, root_name in enumerate(PITCH_CLASSES):
            rotated = np.roll(prof_norm, root_idx)
            corr = float(np.dot(chroma_norm, rotated) / 12.0)
            if corr > best_score:
                best_score = corr
                best_root = root_name
                best_mode = mode_name
    return best_root, best_mode, round(best_score, 3)


def extract_f0_contour(audio_path, fmin_hz=75.0, fmax_hz=880.0):
    """Run `librosa.pyin` on `audio_path` and return `(times, f0_hz, voiced_mask)` lists."""
    ensure_audio_ml_venv()
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    hop_length = 256
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=float(fmin_hz), fmax=float(fmax_hz), sr=sr, hop_length=hop_length
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    f0_list = [None if (v is None or np.isnan(v)) else float(v) for v in f0]
    voiced_list = [bool(v) for v in voiced_flag]
    times_list = [float(t) for t in times]
    return times_list, f0_list, voiced_list


def write_monophonic_midi(events, midi_path, bpm=120.0, track_name="Vocal Guide Melody"):
    """Write `[(start_sec, end_sec, midi_note), ...]` to a Standard MIDI File (`.mid`).

    Appends a quiet pad event 2 bars after the last note so synths/renderers do not
    clip the final release envelope.
    """
    ensure_audio_ml_venv()
    import mido

    ticks_per_beat = 480
    tempo_us = mido.bpm2tempo(float(bpm))

    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=str(track_name), time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo_us, time=0))

    def sec_to_ticks(sec):
        return int(round(mido.second2tick(max(0.0, float(sec)), ticks_per_beat, tempo_us)))

    cur_tick = 0
    for s_sec, e_sec, note in sorted(events, key=lambda x: x[0]):
        s_tick = max(cur_tick, sec_to_ticks(s_sec))
        e_tick = max(s_tick + 12, sec_to_ticks(e_sec))
        track.append(mido.Message("note_on", channel=0, note=int(note), velocity=100, time=s_tick - cur_tick))
        track.append(mido.Message("note_off", channel=0, note=int(note), velocity=0, time=e_tick - s_tick))
        cur_tick = e_tick

    # Two silent bars of tail padding so REAPER/FluidSynth never chops the final release tail
    tail_ticks = ticks_per_beat * 8
    track.append(mido.MetaMessage("end_of_track", time=tail_ticks))
    Path(midi_path).parent.mkdir(parents=True, exist_ok=True)
    mid.save(str(midi_path))
    return Path(midi_path)


def read_midi_note_events(midi_path):
    """Read `[(start_sec, end_sec, midi_note), ...]` from an existing `.mid` file."""
    ensure_audio_ml_venv()
    import mido

    mid = mido.MidiFile(str(midi_path))
    events = []
    cur_sec = 0.0
    active = {}
    for msg in mid:
        cur_sec += float(msg.time)
        if msg.type == "note_on" and msg.velocity > 0:
            active[msg.note] = cur_sec
        elif msg.type in ("note_off", "note_on") and (msg.type == "note_off" or msg.velocity == 0):
            if msg.note in active:
                s_sec = active.pop(msg.note)
                if cur_sec > s_sec + 0.01:
                    events.append((round(s_sec, 4), round(cur_sec, 4), int(msg.note)))
    return sorted(events, key=lambda x: x[0])


def generate_synth_demo_assets(out_dir, bpm=100.0):
    """Synthesize a C-Major backing track, target vocal melody MIDI, and a badly out-of-tune vocal take.

    The synthetic singer uses a realistic glottal pulse train shaped by 5 moving vowel
    formant resonators (`/a/`, `/e/`, `/i/`, `/o/`, `/u/`) so:
      1. Pitch detectors (`librosa.pyin`, `ReaTune`, `Graillon 3`) track its $f_0$ cleanly.
      2. `ReaVocode` extracts articulate vowel formants to modulate its `ReaSynth` carrier.
      3. Every phrase is deliberately sung `+42` to `-145` cents off-target (including
         wrong out-of-scale notes `F#4` and `G#4` plus pitch sag) so the A/B difference
         between Bypass (`Rig 0`), Scale-Snap (`Rig 1A`), Vocoder (`Rig 1B`), and MIDI-Lock
         (`Rig 2`) is immediately audible and measurable.
    """
    ensure_audio_ml_venv()
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sr = S.RATE
    duration = 12.0
    n_samples = int(sr * duration)
    t = np.arange(n_samples, dtype=np.float64) / sr

    # Target melody in C Major (C4=60, D4=62, E4=64, F4=65, G4=67, A4=69)
    # alongside the badly out-of-tune sung MIDI pitch (start_midi -> end_midi drift)
    phrase_spec = [
        # (start_sec, end_sec, target_midi, sung_midi_start, sung_midi_end, vowel_formants_hz)
        (0.50, 1.95, 64, 64.42, 64.48, (730.0, 1090.0, 2440.0)),   # Target E4; sung +45c sharp
        (2.10, 3.55, 67, 66.15, 65.85, (530.0, 1840.0, 2480.0)),   # Target G4; sung near F#4 (-95c flat, out-of-key!)
        (3.70, 5.15, 69, 68.20, 67.90, (390.0, 1990.0, 2550.0)),   # Target A4; sung near G#4 (-95c flat, out-of-key!)
        (5.30, 6.75, 67, 67.44, 67.49, (570.0, 840.0, 2410.0)),    # Target G4; sung +46c sharp
        (6.90, 8.35, 65, 64.54, 64.50, (660.0, 1720.0, 2410.0)),   # Target F4; sung -48c flat
        (8.50, 9.95, 64, 62.65, 62.55, (530.0, 1840.0, 2480.0)),   # Target E4; sung -140c flat (near D4!)
        (10.10, 11.55, 60, 60.45, 60.48, (730.0, 1090.0, 2440.0)), # Target C4; sung +46c sharp
    ]

    target_events = [(s, e, m) for (s, e, m, _, _, _) in phrase_spec]
    midi_path = out_dir / "synth-demo.melody.mid"
    write_monophonic_midi(target_events, midi_path, bpm=bpm, track_name="C Major Target Vocal Melody")

    # Synthesize the out-of-tune vocal take with glottal harmonics + formant filtering
    vocal = np.zeros(n_samples, dtype=np.float64)
    for s_sec, e_sec, _target_m, sung_m0, sung_m1, (f1, f2, f3) in phrase_spec:
        i0 = int(s_sec * sr)
        i1 = min(n_samples, int(e_sec * sr))
        seg_len = i1 - i0
        if seg_len <= 0:
            continue
        seg_t = np.arange(seg_len, dtype=np.float64) / sr
        frac = np.linspace(0.0, 1.0, seg_len)
        # Pitch curve with slight natural vibrato (+/- 8 cents at 5.2 Hz) on top of the bad pitch
        inst_midi = sung_m0 + (sung_m1 - sung_m0) * frac + 0.08 * np.sin(2.0 * np.pi * 5.2 * seg_t)
        inst_hz = 440.0 * (2.0 ** ((inst_midi - 69.0) / 12.0))
        phase = 2.0 * np.pi * np.cumsum(inst_hz) / sr

        # Glottal source with 22 harmonics weighted by vowel formant resonances
        note_sig = np.zeros(seg_len, dtype=np.float64)
        mean_hz = float(np.mean(inst_hz))
        for h in range(1, 23):
            h_hz = h * mean_hz
            if h_hz >= sr * 0.45:
                break
            # Spectral tilt (-6 dB/oct) + 3 vowel formant peaks
            tilt = 1.0 / (h ** 1.15)
            formant_gain = (
                1.0 * np.exp(-0.5 * ((h_hz - f1) / 110.0) ** 2)
                + 0.75 * np.exp(-0.5 * ((h_hz - f2) / 150.0) ** 2)
                + 0.45 * np.exp(-0.5 * ((h_hz - f3) / 200.0) ** 2)
                + 0.12
            )
            note_sig += (tilt * formant_gain) * np.sin(h * phase)

        # Smooth attack/release envelope
        att = min(int(0.035 * sr), seg_len // 4)
        rel = min(int(0.055 * sr), seg_len // 4)
        env = np.ones(seg_len, dtype=np.float64)
        if att > 0:
            env[:att] = np.linspace(0.0, 1.0, att)
        if rel > 0:
            env[-rel:] = np.linspace(1.0, 0.0, rel)
        vocal[i0:i1] += note_sig * env

    peak = np.max(np.abs(vocal)) or 1.0
    vocal = (vocal / peak) * 0.45

    bad_vocal_path = out_dir / "synth-bad-vocal.wav"
    _write_wav_mono_or_stereo(bad_vocal_path, vocal, sr, stereo=False)

    # Synthesize a warm C-Major chord pad backing track (C -> G -> Am -> F -> C)
    backing_l = np.zeros(n_samples, dtype=np.float64)
    backing_r = np.zeros(n_samples, dtype=np.float64)
    chords = [
        (0.0, 3.0, [48, 55, 60, 64]),   # C major
        (3.0, 6.0, [43, 50, 55, 59]),   # G major
        (6.0, 9.0, [45, 52, 57, 60]),   # A minor
        (9.0, 12.0, [41, 48, 53, 60]),  # F major -> C
    ]
    for c_start, c_end, notes in chords:
        i0 = int(c_start * sr)
        i1 = min(n_samples, int(c_end * sr))
        seg_len = i1 - i0
        seg_t = np.arange(seg_len, dtype=np.float64) / sr
        env = np.ones(seg_len, dtype=np.float64)
        att = min(int(0.08 * sr), seg_len // 4)
        rel = min(int(0.15 * sr), seg_len // 4)
        env[:att] = np.linspace(0.0, 1.0, att)
        env[-rel:] = np.linspace(1.0, 0.0, rel)
        for idx_n, m_note in enumerate(notes):
            hz = midi_to_hz(m_note)
            sig = (
                0.65 * np.sin(2.0 * np.pi * hz * seg_t)
                + 0.22 * np.sin(2.0 * np.pi * (2.0 * hz) * seg_t)
                + 0.08 * np.sin(2.0 * np.pi * (3.0 * hz) * seg_t)
            ) * env
            pan = 0.35 + 0.30 * (idx_n % 2)
            backing_l[i0:i1] += sig * pan
            backing_r[i0:i1] += sig * (1.0 - pan)

    b_peak = max(float(np.max(np.abs(backing_l))), float(np.max(np.abs(backing_r))), 1e-6)
    backing_l = (backing_l / b_peak) * 0.25
    backing_r = (backing_r / b_peak) * 0.25
    backing_path = out_dir / "synth-backing.wav"
    _write_wav_stereo(backing_path, backing_l, backing_r, sr)

    return {
        "backing_wav": backing_path,
        "bad_vocal_wav": bad_vocal_path,
        "melody_mid": midi_path,
        "target_events": target_events,
        "key": "C",
        "scale": "major",
        "bpm": bpm,
    }


def _write_wav_mono_or_stereo(path, samples, sr, stereo=False):
    """Write float64 `[-1, 1]` numpy samples to a 16-bit PCM WAV file."""
    import numpy as np

    clipped = np.clip(samples, -0.99, 0.99)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2 if stereo else 1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        if stereo:
            interleaved = np.column_stack([pcm, pcm]).ravel()
            wf.writeframes(interleaved.tobytes())
        else:
            wf.writeframes(pcm.tobytes())


def _write_wav_stereo(path, left, right, sr):
    """Write left/right float64 `[-1, 1]` arrays to a stereo 16-bit PCM WAV file."""
    import numpy as np

    l_pcm = (np.clip(left, -0.99, 0.99) * 32767.0).astype("<i2")
    r_pcm = (np.clip(right, -0.99, 0.99) * 32767.0).astype("<i2")
    interleaved = np.column_stack([l_pcm, r_pcm]).ravel()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(interleaved.tobytes())


def _vocal_cleanup_fx_lines(indent=6, reatune_lines=None, reavocode_lines=None, graillon_lines=None):
    """Build `<FXCHAIN>` lines for a vocal rig track (`ReaGate + [Pitch FX] + ReaEQ + Dual ReaComp + De-esser`)."""
    pad = " " * indent
    gate_lib = hostenv.reaper_native_vst_lib("reagate")
    eq_lib = hostenv.reaper_native_vst_lib("reaeq")
    comp_lib = hostenv.reaper_native_vst_lib("reacomp")
    tune_lib = hostenv.reaper_native_vst_lib("reatune")
    voc_lib = hostenv.reaper_native_vst_lib("reavocode")

    lines = [
        f"{pad}<FXCHAIN",
        f"{pad}  SHOW 0",
        f"{pad}  LASTSEL 0",
        f"{pad}  DOCKED 0",
        f"{pad}  BYPASS 0 0 0",
        f'{pad}  <VST "VST: ReaGate (Cockos) - Vocal Silence Gate" {gate_lib} 0 "" 1919248244<56535472656774726561676174650000> ""',
        f"{pad}  >",
    ]
    if reatune_lines:
        lines += [
            f"{pad}  BYPASS 0 0 0",
            f'{pad}  <VST "VST: ReaTune (Cockos) - Automatic Scale Snap" {tune_lib} 0 "" 1919251566<5653547265746E72656174756E650000> ""',
            *[f"{pad}    {ln}" for ln in reatune_lines],
            f"{pad}  >",
        ]
    if reavocode_lines:
        lines += [
            f"{pad}  BYPASS 0 0 0",
            f'{pad}  <VST "VST: ReaVocode (Cockos) - MIDI Vocoder" {voc_lib} 0 "" 1919252068<56535472657664726561766F636F6465> ""',
            *[f"{pad}    {ln}" for ln in reavocode_lines],
            f"{pad}  >",
        ]
    if graillon_lines:
        lines += [
            f"{pad}  BYPASS 0 0 0",
            f'{pad}  <VST "VST3: Graillon 3 (Auburn Sounds)" "Auburn Sounds Graillon 3.vst3" 0 "" 1073392215{{0B20BA920CE0B1456E62754133317340}} ""',
            *[f"{pad}    {ln}" for ln in graillon_lines],
            f"{pad}  >",
        ]
    lines += [
        f"{pad}  BYPASS 0 0 0",
        f'{pad}  <VST "VST: ReaEQ (Cockos) - Vocal Presence & Air" {eq_lib} 0 "" 1919247729<56535472656571726561657100000000> ""',
        f"{pad}  >",
        f"{pad}  BYPASS 0 0 0",
        f'{pad}  <VST "VST: ReaComp (Cockos) - Peak Catcher (4:1)" {comp_lib} 0 "" 1919247213<5653547265636D726561636F6D700000> ""',
        f"{pad}  >",
        f"{pad}  BYPASS 0 0 0",
        f'{pad}  <VST "VST: ReaComp (Cockos) - Smooth Leveler (2.5:1)" {comp_lib} 0 "" 1919247213<5653547265636D726561636F6D700000> ""',
        f"{pad}  >",
        f"{pad}  BYPASS 0 0 0",
        f'{pad}  <JS Liteon/deesser ""',
        f"{pad}    1 1 0 6200 1.2 -25 4 1 0 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -",
        f"{pad}  >",
        f"{pad}>",
    ]
    return lines


def build_autotune_rpp(dst, plan, sample_rate=S.RATE):
    """Write the 10-track REAPER `.rpp` Auto-Tune & A/B Comparison Studio project."""
    dst = Path(dst)
    tempo = float(plan.get("bpm", 120.0))
    scale_mask = plan["scale_mask_c_to_b"]
    attack_ms = int(plan.get("reatune_attack_ms", 20))
    default_rig = plan.get("default_rig", "graillon")
    if default_rig == "graillon" and not plan.get("graillon_vst3"):
        default_rig = "reatune"

    vocal_file = Path(plan["vocal_file"]) if plan.get("vocal_file") else None
    backing_file = Path(plan["backing_file"]) if plan.get("backing_file") else None
    melody_mid = Path(plan["melody_midi"]) if plan.get("melody_midi") else None
    video_file = Path(plan["video_file"]) if plan.get("video_file") else None
    markers = plan.get("markers") or []

    duration = float(plan.get("duration", 180.0))
    echo_send_vol = sess.db_to_linear_gain(plan["echo"]["send_db"])
    rev_send_vol = sess.db_to_linear_gain(plan["reverb"]["send_db"])
    back_vol = sess.db_to_linear_gain(plan.get("backing_gain_db", -5.0))
    vocal_vol = sess.db_to_linear_gain(1.0)

    eq_lib = hostenv.reaper_native_vst_lib("reaeq")
    comp_lib = hostenv.reaper_native_vst_lib("reacomp")
    delay_lib = hostenv.reaper_native_vst_lib("readelay")
    verb_lib = hostenv.reaper_native_vst_lib("reaverbate")
    synth_lib = hostenv.reaper_native_vst_lib("reasynth")

    reatune_b64 = build_reatune_state_b64(scale_mask, attack_ms=attack_ms, auto_enable=True)
    reavocode_b64 = build_reavocode_state_b64(wet=1.0, dry=0.0, band_norm=0.5, invert=0.0, stereo=1.0, mod_dry=0.0)
    reasynth_b64 = build_reasynth_state_b64()
    has_midi = bool(melody_mid and melody_mid.is_file())
    graillon_b64 = (
        build_graillon3_state_b64(scale_mask, midi_enable=has_midi)
        if plan.get("graillon_vst3")
        else None
    )

    allowed_names = ", ".join(plan.get("scale_notes", []))
    lines = [
        '<REAPER_PROJECT 0.1 "7.0" 0',
        "  <NOTES 0 0",
        f"    |{dst.stem} — Real-Time Vocal Auto-Tune & A/B Comparison Studio",
        f"    |Key / Scale: {plan['key']} {plan['scale']} ({allowed_names}) | Tempo: {tempo:g} BPM",
        "    |",
        "    |HOW TO COMPARE WITHOUT vs. WITH AUTO-TUNE (SINGLE-MUTE TOGGLE):",
        "    |  - Track 1 ('01 - Dry Mic Input') records your raw microphone once (MAINSEND 0) and feeds Tracks 2-5.",
        "    |  - Unmute ONLY ONE of Tracks 2-5 at a time to audition each rig with identical studio EQ/Comp/Reverb:",
        "    |      * Track 2: Rig 0 — Bypass (No Auto-Tune, studio EQ/Comp/Reverb only)",
        f"    |      * Track 3: Rig 1A — Stock ReaTune (Automatic Scale Snap to {plan['key']} {plan['scale']}, {attack_ms}ms attack)",
        "    |      * Track 4: Rig 1B — Stock ReaVocode (100% pitch-locked to Track 6 MIDI Synth Carrier)",
        f"    |      * Track 5: Rig 2 — Graillon 3 Free VST3 (MIDI Pitch-Locked + {plan['key']} {plan['scale']} scale)",
        "  >",
        "  RIPPLE 0",
        '  RECORD_PATH "recordings" ""',
        f"  SAMPLERATE {int(sample_rate)} 0 0",
        f"  TEMPO {tempo:g} 4 4",
        "  MASTER_VOLUME 1 0 -1 -1 1",
        "  MASTER_FX 1",
        "  <MASTERFXLIST",
        "    SHOW 0",
        "    LASTSEL 0",
        "    DOCKED 0",
        "    BYPASS 0 0 0",
        f'    <VST "VST: ReaEQ (Cockos) - Mastering Tone" {eq_lib} 0 "" 1919247729<56535472656571726561657100000000> ""',
        "    >",
        "    BYPASS 0 0 0",
        f'    <VST "VST: ReaComp (Cockos) - Master Bus Glue" {comp_lib} 0 "" 1919247213<5653547265636D726561636F6D700000> ""',
        "    >",
        "    BYPASS 0 0 0",
        '    <JS loser/masterLimiter ""',
        "      -2.5 200 100 0 250 -1 0 - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -",
        "    >",
        "    BYPASS 0 0 0",
        '    <JS analysis/loudness_meter ""',
        "      1 0 0 2 1 1 1 1 -14 -9 1 0 0 1.8 0 0 - - - - - - - - - - - - - 0 -150 0 -150 -150 0 0 - - - - - - - - - - - - - - - - - - - - - - - - - - - -",
        "    >",
        "  >",
    ]

    for idx, (sec, label) in enumerate(markers, 1):
        safe_label = str(label).replace('"', "'")
        lines.append(f'  MARKER {idx} {float(sec):.3f} "{safe_label}" 0')

    # Track 1 (index 0): 01 - Dry Mic Input (Record Here) -> MAINSEND 0 0
    rec_arm = 0 if (vocal_file and vocal_file.is_file()) else 1
    lines += [
        "  <TRACK",
        '    NAME "01 - Dry Mic Input (Record Here)"',
        "    VOLPAN 1.000000 0 -1 -1 1",
        "    MUTESOLO 0 0 0",
        f"    REC {rec_arm} 0 1 0 0 0 0 0",
        "    NCHAN 2",
        "    FX 1",
        "    MAINSEND 0 0",
    ]
    if plan.get("automation_points"):
        lines += [
            "    <VOLENV",
            "      ACT 1 -1",
            "      VIS 1 1 1",
            "      LANEHEIGHT 56 0",
            "      ARM 0",
            "      DEFSHAPE 0 -1 -1",
            "      VOLTYPE 1",
        ]
        for pt_t, pt_gain in plan["automation_points"]:
            lines.append(f"      PT {float(pt_t):g} {float(pt_gain):.6f} 0")
        lines.append("    >")
    if vocal_file and vocal_file.is_file():
        trim_s = float(plan.get("trim_start", 0.0))
        trim_l = float(plan.get("trim_length", duration))
        lines += [
            "    <ITEM",
            f"      POSITION {trim_s:.3f}",
            f"      SOFFS {trim_s:.3f}",
            f"      LENGTH {trim_l:.3f}",
            "      LOOP 0",
            f'      NAME "{vocal_file.name}"',
            *sess.source_block(vocal_file.resolve(), trim_l, 6),
            "    >",
        ]
    lines.append("  >")

    # Track 2 (index 1): 02 - Rig 0: Bypass (No Auto-Tune)
    mute_bypass = 0 if default_rig == "bypass" else 1
    lines += [
        "  <TRACK",
        f'    NAME "{RIG_TRACK_NAMES["bypass"]}"',
        f"    VOLPAN {vocal_vol:.6f} 0 -1 -1 1",
        f"    MUTESOLO {mute_bypass} 0 0",
        "    NCHAN 2",
        "    FX 1",
        "    MAINSEND 1 0",
        "    AUXRECV 0 1 1.000000 0 0 0 0 0 0 -1:U 31 -1 ''",
        *_vocal_cleanup_fx_lines(6),
        "  >",
    ]

    # Track 3 (index 2): 03 - Rig 1A: Stock ReaTune (Scale Snap)
    mute_reatune = 0 if default_rig == "reatune" else 1
    lines += [
        "  <TRACK",
        f'    NAME "{RIG_TRACK_NAMES["reatune"]}"',
        f"    VOLPAN {vocal_vol:.6f} 0 -1 -1 1",
        f"    MUTESOLO {mute_reatune} 0 0",
        "    NCHAN 2",
        "    FX 1",
        "    MAINSEND 1 0",
        "    AUXRECV 0 1 1.000000 0 0 0 0 0 0 -1:U 31 -1 ''",
        *_vocal_cleanup_fx_lines(6, reatune_lines=reatune_b64),
        "  >",
    ]

    # Track 4 (index 3): 04 - Rig 1B: Stock ReaVocode (MIDI Vocoder)
    # Ch 1/2 = Mic Detector from Track 1 (index 0); Ch 3/4 = Synth Carrier from Track 6 (index 5)
    mute_reavocode = 0 if default_rig == "reavocode" else 1
    lines += [
        "  <TRACK",
        f'    NAME "{RIG_TRACK_NAMES["reavocode"]}"',
        f"    VOLPAN {vocal_vol:.6f} 0 -1 -1 1",
        f"    MUTESOLO {mute_reavocode} 0 0",
        "    NCHAN 4",
        "    FX 1",
        "    MAINSEND 1 0",
        "    AUXRECV 0 1 1.000000 0 0 0 0 0 0 -1:U 31 -1 ''",
        "    AUXRECV 5 0 1.000000 0 0 0 0 0 2 -1:U 31 -1 ''",
        *_vocal_cleanup_fx_lines(6, reavocode_lines=reavocode_b64),
        "  >",
    ]

    # Track 5 (index 4): 05 - Rig 2: Graillon 3 MIDI-Lock
    # Receives Pre-FX audio from Track 1 (index 0) and MIDI All->All from Track 6 (index 5).
    mute_graillon = 0 if default_rig == "graillon" else 1
    lines += [
        "  <TRACK",
        f'    NAME "{RIG_TRACK_NAMES["graillon"]}"',
        f"    VOLPAN {vocal_vol:.6f} 0 -1 -1 1",
        f"    MUTESOLO {mute_graillon} 0 0",
        "    NCHAN 2",
        "    FX 1",
        "    MAINSEND 1 0",
        "    AUXRECV 0 1 1.000000 0 0 0 0 0 0 -1:U 31 -1 ''",
        "    AUXRECV 5 0 1.000000 0 0 0 0 -1 0 -1:U 0 -1 ''",
        *_vocal_cleanup_fx_lines(6, graillon_lines=graillon_b64),
        "  >",
    ]

    # Track 6 (index 5): 06 - Guide Melody MIDI + Synth Carrier -> MAINSEND 0 0
    lines += [
        "  <TRACK",
        '    NAME "06 - Guide Melody MIDI + Synth Carrier"',
        "    VOLPAN 1.000000 0 -1 -1 1",
        "    MUTESOLO 0 0 0",
        "    NCHAN 2",
        "    FX 1",
        "    MAINSEND 0 0",
        "    <FXCHAIN",
        "      SHOW 0",
        "      LASTSEL 0",
        "      DOCKED 0",
        "      BYPASS 0 0 0",
        f'      <VST "VSTi: ReaSynth (Cockos) - Vocoder Carrier" {synth_lib} 0 "" 1919251321<5653547265737972656173796E746800> ""',
        *[f"        {ln}" for ln in reasynth_b64],
        "      >",
        "    >",
    ]
    if melody_mid and melody_mid.is_file():
        lines += [
            "    <ITEM",
            "      POSITION 0.000000",
            f"      LENGTH {duration:.6f}",
            "      LOOP 0",
            f'      NAME "{melody_mid.name}"',
            *sess.source_block(melody_mid.resolve(), duration, 6),
            "    >",
        ]
    lines.append("  >")

    # Track 7 (index 6): 07 - Vocal Echo (Stereo Delay Bus) — receives from Rigs on Tracks 2..5 (indices 1..4)
    lines += [
        "  <TRACK",
        '    NAME "07 - Vocal Echo (Stereo Delay Bus)"',
        "    VOLPAN 1.000000 0 -1 -1 1",
        "    MUTESOLO 0 0 0",
        "    FX 1",
        "    MAINSEND 1 0",
        f"    AUXRECV 1 0 {echo_send_vol:.6f} 0 0 0 0 0 0 -1:U 31 -1 ''",
        f"    AUXRECV 2 0 {echo_send_vol:.6f} 0 0 0 0 0 0 -1:U 31 -1 ''",
        f"    AUXRECV 3 0 {echo_send_vol:.6f} 0 0 0 0 0 0 -1:U 31 -1 ''",
        f"    AUXRECV 4 0 {echo_send_vol:.6f} 0 0 0 0 0 0 -1:U 31 -1 ''",
        "    <FXCHAIN",
        "      SHOW 0",
        "      LASTSEL 0",
        "      DOCKED 0",
        "      BYPASS 0 0 0",
        f'      <VST "VST: ReaDelay (Cockos) - Slap & Stereo Echo" {delay_lib} 0 "" 1919247468<5653547265646C72656164656C617900> ""',
        "      >",
        "      BYPASS 0 0 0",
        f'      <VST "VST: ReaEQ (Cockos) - Warm Echo Filter" {eq_lib} 0 "" 1919247729<56535472656571726561657100000000> ""',
        "      >",
        "    >",
        "  >",
    ]

    # Track 8 (index 7): 08 - Vocal Reverb (Lush Plate Bus) — receives from Rigs on Tracks 2..5 (indices 1..4)
    lines += [
        "  <TRACK",
        '    NAME "08 - Vocal Reverb (Lush Plate Bus)"',
        "    VOLPAN 1.000000 0 -1 -1 1",
        "    MUTESOLO 0 0 0",
        "    FX 1",
        "    MAINSEND 1 0",
        f"    AUXRECV 1 0 {rev_send_vol:.6f} 0 0 0 0 0 0 -1:U 31 -1 ''",
        f"    AUXRECV 2 0 {rev_send_vol:.6f} 0 0 0 0 0 0 -1:U 31 -1 ''",
        f"    AUXRECV 3 0 {rev_send_vol:.6f} 0 0 0 0 0 0 -1:U 31 -1 ''",
        f"    AUXRECV 4 0 {rev_send_vol:.6f} 0 0 0 0 0 0 -1:U 31 -1 ''",
        "    <FXCHAIN",
        "      SHOW 0",
        "      LASTSEL 0",
        "      DOCKED 0",
        "      BYPASS 0 0 0",
        f'      <VST "VST: ReaVerbate (Cockos) - Lush Vocal Plate" {verb_lib} 0 "" 1919252066<56535472657662726561766572626174> ""',
        "      >",
        "      BYPASS 0 0 0",
        f'      <VST "VST: ReaEQ (Cockos) - Abbey Road Reverb Curve" {eq_lib} 0 "" 1919247729<56535472656571726561657100000000> ""',
        "      >",
        "    >",
        "  >",
    ]

    # Track 9 (index 8): 09 - Backing Track (Instrumental)
    if backing_file and backing_file.is_file():
        lines += [
            "  <TRACK",
            '    NAME "09 - Backing Track (Instrumental)"',
            f"    VOLPAN {back_vol:.6f} 0 -1 -1 1",
            "    MUTESOLO 0 0 0",
            "    FX 1",
            "    MAINSEND 1 0",
            "    <FXCHAIN",
            "      SHOW 0",
            "      LASTSEL 0",
            "      DOCKED 0",
            "      BYPASS 0 0 0",
            f'      <VST "VST: ReaEQ (Cockos) - Vocal Pocket Carve" {eq_lib} 0 "" 1919247729<56535472656571726561657100000000> ""',
            "      >",
            "    >",
            "    <ITEM",
            "      POSITION 0.000000",
            f"      LENGTH {duration:.6f}",
            "      LOOP 0",
            f'      NAME "{backing_file.name}"',
            *sess.source_block(backing_file.resolve(), duration, 6),
            "    >",
            "  >",
        ]

    # Track 10 (index 9): 10 - Lyrics Video (Press Cmd+Shift+V)
    if video_file and video_file.is_file():
        lines += [
            "  <TRACK",
            '    NAME "10 - Lyrics Video (Press Cmd+Shift+V)"',
            "    VOLPAN 1.000000 0 -1 -1 1",
            "    MUTESOLO 1 0 0",
            "    MAINSEND 1 0",
            "    <ITEM",
            "      POSITION 0.000000",
            f"      LENGTH {duration:.6f}",
            "      LOOP 0",
            f'      NAME "{video_file.name}"',
            *sess.source_block(video_file.resolve(), duration, 6),
            "    >",
            "  >",
        ]

    lines.append(">")
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return duration


def build_autotune_lua(rpp_path, report_path, plan):
    """Build the `.autotune.lua` ReaScript that configures ReaEQ, ReaDelay, ReaVerbate & Graillon 3."""
    eq = plan["eq"]
    echo = plan["echo"]
    rev = plan["reverb"]
    tempo = float(plan.get("bpm", 120.0))
    musical_norm = sess.reaper_delay_musical_norm(echo["eighth_notes"])
    scale_mask = plan["scale_mask_c_to_b"]
    has_midi = bool(plan.get("melody_midi") and Path(plan["melody_midi"]).is_file())
    midi_in_val = 1.0 if has_midi else 0.0

    rpp_escaped = json.dumps(str(Path(rpp_path).resolve()))
    report_escaped = json.dumps(str(Path(report_path).resolve()))

    scale_lua_lines = []
    for pc_idx, flag in enumerate(scale_mask):
        param_idx = 24 + pc_idx
        note_label = PITCH_CLASSES[pc_idx]
        scale_lua_lines.append(
            f"    reaper.TrackFX_SetParam(gr_tr, gr_fx, {param_idx}, {float(flag):.1f}) -- Allow {note_label}"
        )
    scale_lua_block = "\n".join(scale_lua_lines)

    return f"""-- Generated by creative-stack /sound autotune.py
-- Configures all 4 vocal rigs (Bypass, ReaTune, ReaVocode, Graillon 3 MIDI-Lock) + studio buses
reaper.Main_openProject("noprompt:" .. {rpp_escaped})

local report_path = {report_escaped}

local function db_to_gain(db)
  return 10.0 ^ (db / 20.0)
end

local function set_eq(tr, fx, hpf_hz, ls_hz, ls_db, b1_hz, b1_db, b1_bw, b2_hz, b2_db, b2_bw, hs_hz, hs_db)
  if hpf_hz and hpf_hz > 0 then
    reaper.TrackFX_SetEQBandEnabled(tr, fx, 0, 0, true)
    reaper.TrackFX_SetEQParam(tr, fx, 0, 0, 0, hpf_hz, false)
    reaper.TrackFX_SetEQParam(tr, fx, 0, 0, 1, 1.0, false)
    reaper.TrackFX_SetEQParam(tr, fx, 0, 0, 2, 1.4, false)
  else
    reaper.TrackFX_SetEQBandEnabled(tr, fx, 0, 0, false)
  end

  reaper.TrackFX_SetEQBandEnabled(tr, fx, 1, 0, true)
  reaper.TrackFX_SetEQParam(tr, fx, 1, 0, 0, ls_hz or 120, false)
  reaper.TrackFX_SetEQParam(tr, fx, 1, 0, 1, db_to_gain(ls_db or 0.0), false)
  reaper.TrackFX_SetEQParam(tr, fx, 1, 0, 2, 1.0, false)

  reaper.TrackFX_SetEQBandEnabled(tr, fx, 2, 0, true)
  reaper.TrackFX_SetEQParam(tr, fx, 2, 0, 0, b1_hz or 330, false)
  reaper.TrackFX_SetEQParam(tr, fx, 2, 0, 1, db_to_gain(b1_db or 0.0), false)
  reaper.TrackFX_SetEQParam(tr, fx, 2, 0, 2, b1_bw or 1.0, false)

  reaper.TrackFX_SetEQBandEnabled(tr, fx, 2, 1, true)
  reaper.TrackFX_SetEQParam(tr, fx, 2, 1, 0, b2_hz or 3400, false)
  reaper.TrackFX_SetEQParam(tr, fx, 2, 1, 1, db_to_gain(b2_db or 0.0), false)
  reaper.TrackFX_SetEQParam(tr, fx, 2, 1, 2, b2_bw or 1.1, false)

  reaper.TrackFX_SetEQBandEnabled(tr, fx, 4, 0, true)
  reaper.TrackFX_SetEQParam(tr, fx, 4, 0, 0, hs_hz or 10000, false)
  reaper.TrackFX_SetEQParam(tr, fx, 4, 0, 1, db_to_gain(hs_db or 0.0), false)
  reaper.TrackFX_SetEQParam(tr, fx, 4, 0, 2, 1.2, false)
end

local function find_fx_by_subname(tr, needle)
  local cnt = reaper.TrackFX_GetCount(tr)
  for i = 0, cnt - 1 do
    local _, nm = reaper.TrackFX_GetFXName(tr, i, "")
    if nm and string.find(string.lower(nm), string.lower(needle), 1, true) then
      return i
    end
  end
  return -1
end

-- 1. Master Track EQ
local master_tr = reaper.GetMasterTrack(0)
set_eq(master_tr, 0, 30.0, 80.0, 0.8, 300.0, -0.8, 1.0, 3500.0, 0.5, 1.2, 12000.0, 1.0)

-- 2. Configure Track 5 (index 4): Rig 2 - Graillon 3 MIDI-Lock
local gr_tr = reaper.GetTrack(0, 4)
if gr_tr then
  local gr_fx = find_fx_by_subname(gr_tr, "graillon")
  if gr_fx >= 0 and reaper.TrackFX_GetNumParams(gr_tr, gr_fx) <= 10 then
    reaper.TrackFX_Delete(gr_tr, gr_fx)
    gr_fx = -1
  end
  if gr_fx < 0 then
    local added = reaper.TrackFX_AddByName(gr_tr, "Graillon 3", false, -1)
    if added >= 0 then
      -- Move Graillon 3 to slot 1 (immediately after ReaGate, before ReaEQ)
      reaper.TrackFX_CopyToTrack(gr_tr, added, gr_tr, 1, true)
      gr_fx = 1
    end
  end
  if gr_fx >= 0 then
    reaper.TrackFX_SetParam(gr_tr, gr_fx, 2, 0.707946)  -- Wet Mix: 0.0 dB
    reaper.TrackFX_SetParam(gr_tr, gr_fx, 7, 1.0)       -- Global Amount: 100%
    reaper.TrackFX_SetParam(gr_tr, gr_fx, 8, 0.0)       -- Dry Mix: -inf dB
    reaper.TrackFX_SetParam(gr_tr, gr_fx, 9, 0.35)      -- Low Cut: ~80 Hz
    reaper.TrackFX_SetParam(gr_tr, gr_fx, 18, 1.0)      -- Formant Amount: 100%
    reaper.TrackFX_SetParam(gr_tr, gr_fx, 19, 1.0)      -- Corr. Amount: 100%
    reaper.TrackFX_SetParam(gr_tr, gr_fx, 20, 0.12)     -- Smooth: fast pop lock
    reaper.TrackFX_SetParam(gr_tr, gr_fx, 21, 1.0)      -- Snap Max: 8.00 semitones
    reaper.TrackFX_SetParam(gr_tr, gr_fx, 23, 0.05)     -- Inertia: low
{scale_lua_block}
    reaper.TrackFX_SetParam(gr_tr, gr_fx, 54, 0.5)      -- Pitch Engine: Graillon3
    reaper.TrackFX_SetParam(gr_tr, gr_fx, 56, {midi_in_val:.1f}) -- MIDI In Enabled
    reaper.TrackFX_SetParam(gr_tr, gr_fx, 62, 1.0)      -- FX Enabled: yes
    reaper.TrackFX_SetParam(gr_tr, gr_fx, 63, 0.0)      -- PTM Enabled: no (disable ring-mod)
    reaper.TrackFX_SetParam(gr_tr, gr_fx, 64, 1.0)      -- Correction: yes
    reaper.TrackFX_SetParam(gr_tr, gr_fx, 65, 1.0)      -- Formant: yes
  end
end

-- 3. Configure identical Vocal ReaEQ curve on all 4 vocal rigs (Tracks 2..5, indices 1..4)
for tr_idx = 1, 4 do
  local tr = reaper.GetTrack(0, tr_idx)
  if tr then
    local eq_fx = find_fx_by_subname(tr, "reaeq")
    if eq_fx >= 0 then
      set_eq(tr, eq_fx, {eq['hpf_hz']}, {eq['low_shelf_hz']}, {eq['low_shelf_db']}, {eq['mud_hz']}, {eq['mud_db']}, {eq['mud_bw']}, {eq['presence_hz']}, {eq['presence_db']}, {eq['presence_bw']}, {eq['air_hz']}, {eq['air_db']})
    end
  end
end

-- 4. Track 7 (index 6): Vocal Echo Bus
local echo_tr = reaper.GetTrack(0, 6)
if echo_tr then
  reaper.TrackFX_SetParam(echo_tr, 0, 0, 0.5)
  reaper.TrackFX_SetParam(echo_tr, 0, 1, 0.0)
  reaper.TrackFX_SetParam(echo_tr, 0, 2, 1.0)
  reaper.TrackFX_SetParam(echo_tr, 0, 3, 0.0)
  reaper.TrackFX_SetParam(echo_tr, 0, 4, {musical_norm:.8f})
  reaper.TrackFX_SetParam(echo_tr, 0, 5, {echo['feedback_norm']:.4f})
  reaper.TrackFX_SetParam(echo_tr, 0, 6, 0.190)
  reaper.TrackFX_SetParam(echo_tr, 0, 7, 0.011)
  reaper.TrackFX_SetParam(echo_tr, 0, 9, 1.0)
  reaper.TrackFX_SetParam(echo_tr, 0, 10, 0.5)
  set_eq(echo_tr, 1, 220.0, 250.0, 0.0, 500.0, 0.0, 1.0, 3200.0, -2.5, 1.0, 4500.0, -8.0)
end

-- 5. Track 8 (index 7): Vocal Reverb Bus
local rev_tr = reaper.GetTrack(0, 7)
if rev_tr then
  reaper.TrackFX_SetParam(rev_tr, 0, 0, 0.5)
  reaper.TrackFX_SetParam(rev_tr, 0, 1, 0.0)
  reaper.TrackFX_SetParam(rev_tr, 0, 2, {rev['room_size']:.2f})
  reaper.TrackFX_SetParam(rev_tr, 0, 5, {rev['predelay_norm']:.3f})
  set_eq(rev_tr, 1, 220.0, 250.0, 0.0, 350.0, -2.5, 0.9, 2500.0, 0.0, 1.0, 7500.0, -6.0)
end

-- 6. Track 9 (index 8): Backing Track EQ
local back_tr = reaper.GetTrack(0, 8)
if back_tr then
  local eq_fx = find_fx_by_subname(back_tr, "reaeq")
  if eq_fx >= 0 then
    set_eq(back_tr, eq_fx, 0.0, 100.0, 0.0, 200.0, -1.0, 1.0, 3200.0, -2.2, 1.2, 10000.0, 0.0)
  end
end

local f = io.open(report_path, "w")
local function log(line)
  if f then
    local clean = string.gsub(line, "[^%g%s]", "")
    f:write(clean .. "\\n")
  end
end

local function dump_track_fx(label, tr)
  local fx_count = reaper.TrackFX_GetCount(tr)
  log(string.format("=== %s (FX count: %d) ===", label, fx_count))
  for fx = 0, fx_count - 1 do
    local _, fx_name = reaper.TrackFX_GetFXName(tr, fx, "")
    log(string.format("  [FX %d] %s", fx + 1, fx_name))
    local p_count = reaper.TrackFX_GetNumParams(tr, fx)
    for p = 0, math.min(p_count - 1, 66) do
      local _, p_name = reaper.TrackFX_GetParamName(tr, fx, p, "")
      local _, p_val = reaper.TrackFX_GetFormattedParamValue(tr, fx, p, "")
      log(string.format("    - [%02d] %-24s : %s", p, p_name, p_val))
    end
  end
end

dump_track_fx("MASTER TRACK", master_tr)
local n_tracks = reaper.CountTracks(0)
for i = 0, n_tracks - 1 do
  local tr = reaper.GetTrack(0, i)
  local _, t_name = reaper.GetTrackName(tr)
  dump_track_fx(string.format("Track %d: %s", i + 1, t_name), tr)
end

if f then f:close() end
reaper.Main_SaveProject(0, false)
"""


def toggle_rpp_rig_solo(rpp_text, active_rig_key, render_dir, render_stem,
                        sample_rate=S.RATE, include_buses_and_backing=False):
    """Transform a saved `.rpp` text so only `active_rig_key` renders to `render_dir/render_stem.wav`.

    Always keeps Track 1 (`01 - Dry Mic Input`) and Track 6 (`06 - Guide Melody MIDI + Synth Carrier`)
    unmuted (`MUTESOLO 0 0 0`) because both have `MAINSEND 0 0` and feed the vocal rigs.
    Unmutes the single chosen rig track (`bypass`, `reatune`, `reavocode`, or `graillon`)
    and mutes the other three rigs. When `include_buses_and_backing=False`, also mutes
    Tracks 7–10 (`Vocal Echo`, `Vocal Reverb`, `Backing Track`, `Lyrics Video`) so headless
    render outputs the pure processed vocal stem for `librosa.pyin` $f_0$ benchmarking.
    """
    target_track_name = RIG_TRACK_NAMES[active_rig_key]
    all_rig_names = set(RIG_TRACK_NAMES.values())

    render_header = [
        f'  RENDER_FILE "{Path(render_dir).resolve()}"',
        f'  RENDER_PATTERN "{render_stem}"',
        f"  RENDER_FMT 0 2 {int(sample_rate)}",
        "  RENDER_RANGE 1 0 0 18 1000",
        "  RENDER_STEMS 0",
        "  RENDER_ADDTOPROJ 0",
    ]

    # Strip any existing RENDER_* lines at top level and inject new ones right after SAMPLERATE/TEMPO
    cleaned_lines = []
    injected_render = False
    for line in rpp_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("RENDER_FILE ", "RENDER_PATTERN ", "RENDER_FMT ", "RENDER_RANGE ", "RENDER_STEMS ", "RENDER_ADDTOPROJ ")):
            continue
        cleaned_lines.append(line)
        if not injected_render and stripped.startswith("TEMPO "):
            cleaned_lines.extend(render_header)
            injected_render = True

    # Walk track blocks and adjust MUTESOLO
    out_lines = []
    cur_track_name = None
    for line in cleaned_lines:
        stripped = line.strip()
        m_name = re.match(r'^NAME\s+"([^"]*)"', stripped)
        if m_name:
            cur_track_name = m_name.group(1)
            out_lines.append(line)
            continue
        if stripped.startswith("MUTESOLO ") and cur_track_name is not None:
            indent = line[: len(line) - len(line.lstrip())]
            if cur_track_name in ("01 - Dry Mic Input (Record Here)", "06 - Guide Melody MIDI + Synth Carrier"):
                out_lines.append(f"{indent}MUTESOLO 0 0 0")
            elif cur_track_name in all_rig_names:
                mute_val = 0 if cur_track_name == target_track_name else 1
                out_lines.append(f"{indent}MUTESOLO {mute_val} 0 0")
            elif cur_track_name.startswith(("07 -", "08 -", "09 -")):
                mute_val = 0 if include_buses_and_backing else 1
                out_lines.append(f"{indent}MUTESOLO {mute_val} 0 0")
            elif cur_track_name.startswith("10 -"):
                out_lines.append(f"{indent}MUTESOLO 1 0 0")
            else:
                out_lines.append(line)
            continue
        out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def run_seed_vc_conversion(source_wav, target_ref_wav, out_wav, diffusion_steps=15,
                           semi_shift=0, f0_condition=True, auto_f0_adjust=True):
    """Run Plachtaa/seed-vc singing voice conversion (`f0_condition=True`) on Apple Silicon MPS / GPU / CPU."""
    svc = hostenv.seed_vc_paths()
    if not svc["ready"]:
        S.die("Seed-VC is not installed. Run: bash scripts/install/seed-vc.sh")

    target_ref = target_ref_wav
    if not target_ref or str(target_ref).strip().lower() == "auto":
        default_ref = svc["repo"] / "examples/reference/s1p1.wav"
        if not default_ref.is_file():
            refs = sorted((svc["repo"] / "examples/reference").glob("*.wav"))
            default_ref = refs[0] if refs else None
        if not default_ref:
            S.die("No reference voice WAV found for Seed-VC.")
        target_ref = default_ref

    source_wav = Path(source_wav).resolve()
    target_ref = Path(target_ref).resolve()
    out_wav = Path(out_wav).resolve()
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="cs-seedvc-out-") as tmp_out:
        cmd = [
            str(svc["python"]),
            str(svc["inference"]),
            "--source", str(source_wav),
            "--target", str(target_ref),
            "--output", str(tmp_out),
            "--diffusion-steps", str(int(diffusion_steps)),
            "--f0-condition", "True" if f0_condition else "False",
            "--auto-f0-adjust", "True" if auto_f0_adjust else "False",
            "--semi-tone-shift", str(int(semi_shift)),
            "--fp16", "True",
        ]
        print(f"🤖 running Seed-VC singing voice conversion (ref={target_ref.name}, steps={diffusion_steps}) …")
        env = dict(os.environ)
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        proc = subprocess.run(cmd, cwd=str(svc["repo"]), env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            S.die(f"Seed-VC inference failed:\n{proc.stderr or proc.stdout}")
        generated = sorted(Path(tmp_out).glob("*.wav"))
        if not generated:
            S.die(f"Seed-VC did not emit a .wav file in {tmp_out}")
        S.ffmpeg([
            "-i", str(generated[0]),
            "-ar", str(S.RATE), "-ac", "1", "-c:a", "pcm_s24le",
            str(out_wav),
        ])
    return out_wav


def ensure_reaper_cfg_args():
    """Return `['-cfgfile', ...]` when REAPER's default resource dir is read-only or lacks VST cache."""
    res_dir = hostenv.reaper_resource_dir()
    vst_ini = res_dir / "reaper-vstplugins_arm64.ini"
    if res_dir.is_dir() and os.access(res_dir, os.W_OK) and vst_ini.is_file():
        return []
    cfg_dir = Path("/tmp/cs-reaper-cfg")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    ini_path = cfg_dir / "reaper.ini"
    # Never write `vstpath=` here: an explicit `vstpath=` overrides REAPER's default
    # internal FX directory (`/Applications/REAPER.app/Contents/Plugins/FX`) and
    # causes native ReaEQ/ReaComp/ReaGate to load as 3-param offline placeholders.
    # Leaving `[reaper]` without `vstpath=` lets REAPER scan both its built-in
    # FX folder and standard system/user VST3 paths automatically.
    if not ini_path.is_file():
        ini_path.write_text("[reaper]\n", encoding="utf-8")
    else:
        txt = ini_path.read_text(encoding="utf-8", errors="replace")
        if "vstpath=" in txt or "vstpath64=" in txt:
            ini_path.write_text("[reaper]\n", encoding="utf-8")
            for stale in cfg_dir.glob("reaper-vstplugins*.ini"):
                stale.unlink(missing_ok=True)
    return ["-cfgfile", str(ini_path)]


def apply_autotune_reascript(lua_path, report_path, headless_quit=True):
    """Execute `lua_path` inside REAPER to set live FX parameters, write `report_path`, and save the `.rpp`."""
    binary = sess.resolve_reaper_binary()
    if not binary:
        S.die("reaper not found. " + hostenv.package_hint("reaper"))
    cfg_args = ensure_reaper_cfg_args()
    report_path = Path(report_path)
    before = report_path.stat().st_mtime if report_path.exists() else 0

    if headless_quit:
        quit_lua = Path(lua_path).with_name(f"_{Path(lua_path).stem}_headless.lua")
        quit_lua.write_text(
            Path(lua_path).read_text(encoding="utf-8") + "\nreaper.Main_OnCommand(40004, 0)\n",
            encoding="utf-8",
        )
        try:
            subprocess.run(
                [binary, "-newinst", "-nosplash", *cfg_args, str(quit_lua)],
                capture_output=True, text=True, timeout=45, check=False,
            )
        finally:
            quit_lua.unlink(missing_ok=True)
        return report_path.exists() and report_path.stat().st_mtime > before and report_path.stat().st_size > 0

    subprocess.Popen(
        [binary, "-nonewinst", *cfg_args, str(lua_path)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        if report_path.exists() and report_path.stat().st_mtime > before and report_path.stat().st_size > 0:
            return True
        time.sleep(0.3)
    return False


def render_with_reaper(project_rpp, dst_wav):
    """Render `project_rpp` to `dst_wav` via `REAPER -newinst -nosplash -renderproject`."""
    binary = sess.resolve_reaper_binary()
    if not binary:
        S.die("reaper not found. " + hostenv.package_hint("reaper"))
    cfg_args = ensure_reaper_cfg_args()
    dst_wav = Path(dst_wav)
    before = dst_wav.stat().st_mtime if dst_wav.exists() else 0
    subprocess.run(
        [binary, "-newinst", "-nosplash", *cfg_args, "-renderproject", str(project_rpp)],
        capture_output=True, text=True, timeout=300, check=False,
    )
    for _ in range(60):
        if dst_wav.exists() and dst_wav.stat().st_mtime > before and dst_wav.stat().st_size > 0:
            return True
        import time as _time
        _time.sleep(0.5)
    return False


def benchmark_all_rigs(rpp_path, plan, ai_voice_ref=None):
    """Render each rig from `rpp_path` via REAPER, measure `librosa.pyin` cent errors, and write `.benchmark.json`."""
    rpp_path = Path(rpp_path).resolve()
    out_dir = rpp_path.parent / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_rpp_text = rpp_path.read_text(encoding="utf-8")
    scale_mask = plan["scale_mask_c_to_b"]
    target_events = None
    if plan.get("melody_midi") and Path(plan["melody_midi"]).is_file():
        target_events = read_midi_note_events(plan["melody_midi"])

    rig_specs = [
        ("rig0_bypass", "bypass", "Rig 0 — Without Auto-Tune (Bypass Control)"),
        ("rig1a_reatune", "reatune", f"Rig 1A — Stock REAPER ReaTune ({plan['key']} {plan['scale']} Scale Snap)"),
        ("rig1b_reavocode", "reavocode", "Rig 1B — Stock REAPER ReaVocode (MIDI Synth Carrier Lock)"),
    ]
    if plan.get("graillon_vst3"):
        rig_specs.append(("rig2_graillon3", "graillon", "Rig 2 — Graillon 3 Free VST3 (MIDI Pitch-Lock + Scale)"))

    results = {}
    backing_wav = Path(plan["backing_file"]) if plan.get("backing_file") and Path(plan["backing_file"]).is_file() else None

    for slug, rig_key, label in rig_specs:
        vocal_stem_name = f"{rpp_path.stem}.{slug}.vocal"
        vocal_wav = out_dir / f"{vocal_stem_name}.wav"
        tmp_rpp = rpp_path.parent / f"_{slug}_render.rpp"
        tmp_rpp.write_text(
            toggle_rpp_rig_solo(saved_rpp_text, rig_key, out_dir, vocal_stem_name, include_buses_and_backing=False),
            encoding="utf-8",
        )
        print(f"🎧 rendering {label} via REAPER → {vocal_wav.name} …")
        if not render_with_reaper(tmp_rpp, vocal_wav):
            S.die(f"REAPER failed to render {vocal_wav}")
        tmp_rpp.unlink(missing_ok=True)

        # Also create a full studio mix preview (vocal + backing track) for human A/B listening
        mix_wav = out_dir / f"{rpp_path.stem}.{slug}.mix.wav"
        if backing_wav:
            kar.mix_karaoke_audio(
                backing_wav, None, vocal_wav, mix_wav,
                guide_db=None, sing_reverb=0.18, target_i=-14.0, target_tp=-1.0,
            )

        times, f0_hz, voiced_mask = extract_f0_contour(vocal_wav)
        pitch_metrics = compute_pitch_benchmark_metrics(
            times, f0_hz, voiced_mask, scale_mask_c_to_b=scale_mask, target_events=target_events
        )
        lufs_stats = S.measure(vocal_wav)
        results[slug] = {
            "label": label,
            "rig_key": rig_key,
            "vocal_wav": str(vocal_wav),
            "mix_wav": str(mix_wav) if mix_wav.exists() else None,
            "lufs_i": round(float(lufs_stats["i"]), 2),
            "true_peak_dbtp": round(float(lufs_stats["tp"]), 2),
            **pitch_metrics,
        }
        S.report(
            vocal_wav,
            f"median error {pitch_metrics['median_abs_cents']:.1f}c | "
            f"scale error {pitch_metrics['scale_median_abs_cents']:.1f}c | "
            f"<=25c: {pitch_metrics['pct_within_25_cents']:.1f}%",
        )

    # Optional Rig 4: Seed-VC AI Singing Voice Conversion
    if ai_voice_ref and plan.get("vocal_file") and Path(plan["vocal_file"]).is_file():
        slug = "rig4_seedvc"
        label = "Rig 4 — Seed-VC AI Singing Voice Conversion (MPS Zero-Shot Timbre + F0)"
        # Feed the pitch-corrected Graillon 3 (or ReaTune) vocal into Seed-VC so both pitch AND timbre are transformed!
        pitch_locked_src = results.get("rig2_graillon3", results.get("rig1a_reatune", {}))["vocal_wav"]
        svc_vocal_wav = out_dir / f"{rpp_path.stem}.{slug}.vocal.wav"
        run_seed_vc_conversion(pitch_locked_src, ai_voice_ref, svc_vocal_wav, diffusion_steps=15)
        mix_wav = out_dir / f"{rpp_path.stem}.{slug}.mix.wav"
        if backing_wav:
            kar.mix_karaoke_audio(
                backing_wav, None, svc_vocal_wav, mix_wav,
                guide_db=None, sing_reverb=0.18, target_i=-14.0, target_tp=-1.0,
            )
        times, f0_hz, voiced_mask = extract_f0_contour(svc_vocal_wav)
        pitch_metrics = compute_pitch_benchmark_metrics(
            times, f0_hz, voiced_mask, scale_mask_c_to_b=scale_mask, target_events=target_events
        )
        lufs_stats = S.measure(svc_vocal_wav)
        results[slug] = {
            "label": label,
            "rig_key": "seedvc",
            "vocal_wav": str(svc_vocal_wav),
            "mix_wav": str(mix_wav) if mix_wav.exists() else None,
            "lufs_i": round(float(lufs_stats["i"]), 2),
            "true_peak_dbtp": round(float(lufs_stats["tp"]), 2),
            **pitch_metrics,
        }
        S.report(
            svc_vocal_wav,
            f"median error {pitch_metrics['median_abs_cents']:.1f}c | "
            f"scale error {pitch_metrics['scale_median_abs_cents']:.1f}c | "
            f"<=25c: {pitch_metrics['pct_within_25_cents']:.1f}%",
        )

    bench_doc = {
        "project_rpp": str(rpp_path),
        "key": plan["key"],
        "scale": plan["scale"],
        "scale_notes": plan["scale_notes"],
        "bpm": plan["bpm"],
        "melody_midi": plan.get("melody_midi"),
        "rigs": results,
    }
    bench_path = rpp_path.with_name(f"{rpp_path.stem}.benchmark.json")
    bench_path.write_text(json.dumps(bench_doc, indent=2) + "\n", encoding="utf-8")
    S.report(bench_path, "objective A/B pitch & loudness benchmark")
    return bench_doc


def print_setup_check():
    """Inspect and report audio interfaces, REAPER latency settings, VST3 plugins, Demucs, and Seed-VC."""
    reaper_bin = sess.resolve_reaper_binary()
    reaper_audio = hostenv.reaper_audio_settings()
    graillon_vst3 = hostenv.find_vst3("Auburn Sounds Graillon 3") or hostenv.find_vst3("Graillon 3")
    demucs_cmd = hostenv.demucs_cmd()
    svc = hostenv.seed_vc_paths()
    devices = hostenv.audio_devices()

    report = {
        "reaper_binary": reaper_bin,
        "reaper_audio": reaper_audio,
        "graillon_vst3": str(graillon_vst3) if graillon_vst3 else None,
        "demucs": demucs_cmd,
        "seed_vc": {
            "ready": svc["ready"],
            "repo": str(svc["repo"]),
            "python": str(svc["python"]),
        },
        "audio_devices": devices,
    }

    print("=== Creative Stack — Real-Time REAPER Auto-Tune & AI Voice Studio Check ===")
    print(f"  REAPER Binary        : {reaper_bin or 'MISSING'}")
    print(f"  REAPER INI Path      : {reaper_audio['ini_path']} (configured={reaper_audio['configured']})")
    print(
        f"  REAPER Buffer Target : {reaper_audio['buffer_samples']} samples @ {reaper_audio['sample_rate']} Hz "
        f"(~{reaper_audio['estimated_roundtrip_ms']} ms round-trip; recommend 64-128 samples for live monitoring)"
    )
    print(f"  Graillon 3 Free VST3 : {graillon_vst3 or 'MISSING (run: bash scripts/install/graillon.sh)'}")
    print(f"  Demucs v4 Stem Split : {' '.join(demucs_cmd) if demucs_cmd else 'MISSING'}")
    print(f"  Seed-VC (AI Voice)   : {'READY (' + str(svc['repo']) + ')' if svc['ready'] else 'MISSING (run: bash scripts/install/seed-vc.sh)'}")
    hw_inputs = [d for d in devices if not d.get("loopback")]
    virt_inputs = [d for d in devices if d.get("loopback")]
    print(f"  Audio Inputs Found   : {len(hw_inputs)} hardware, {len(virt_inputs)} virtual/loopback")
    for dev in hw_inputs[:6]:
        print(f"    - [HW]      {dev['name']} ({dev.get('manufacturer', dev.get('backend', ''))})")
    for dev in virt_inputs[:6]:
        print(f"    - [VIRTUAL] {dev['name']} ({dev.get('manufacturer', dev.get('backend', ''))})")
    return report


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check-setup", action="store_true",
                    help="verify REAPER audio latency, Graillon 3 VST3, Demucs, Seed-VC, and audio inputs")
    ap.add_argument("--launch-live-vc", action="store_true",
                    help="launch Seed-VC's real-time voice conversion GUI (real-time-gui.py)")
    ap.add_argument("--synth-demo", action="store_true",
                    help="generate a C-Major backing track, melody MIDI, and badly out-of-tune synthetic vocal take")
    ap.add_argument("--input", "-i",
                    help="full reference song (MP3/WAV/MP4) to split via Demucs into instrumental + vocal melody MIDI")
    ap.add_argument("--reference-song",
                    help="optional full song with original vocals used to extract the melody MIDI for a karaoke backing track")
    ap.add_argument("--karaoke-backing",
                    help="existing instrumental backing WAV/MP3")
    ap.add_argument("--youtube", "-y",
                    help="YouTube karaoke URL to download (1080p MP4 + 48kHz 24-bit WAV)")
    ap.add_argument("--midi",
                    help="existing monophonic vocal melody .mid file for Rig 1B (ReaVocode) & Rig 2 (Graillon 3)")
    ap.add_argument("--vocal-take",
                    help="recorded dry vocal take WAV to place on Track 1 for A/B comparison & benchmarking")
    ap.add_argument("--key",
                    help="musical key root (e.g. C, F#, Bb); auto-detected via CQT chroma if omitted")
    ap.add_argument("--scale", default="auto",
                    choices=["auto", "major", "minor", "chromatic", "pentatonic_major", "pentatonic_minor"],
                    help="musical scale mode (default: auto-detect major/minor)")
    ap.add_argument("--bpm", type=float, default=100.0,
                    help="project tempo in BPM (default: 100)")
    ap.add_argument("--attack-ms", type=int, default=20,
                    help="ReaTune pitch correction attack time in ms (default: 20)")
    ap.add_argument("--default-rig", default="graillon",
                    choices=["bypass", "reatune", "reavocode", "graillon"],
                    help="which rig track starts unmuted in the Reaper session (default: graillon)")
    ap.add_argument("--video",
                    help="karaoke lyric video file to place on Track 10 (muted, Cmd+Shift+V)")
    ap.add_argument("--markers",
                    help="markdown or text file with timestamped lyric/section markers")
    ap.add_argument("--benchmark", action="store_true",
                    help="render all rigs via REAPER, measure f0 cent error via librosa.pyin, and emit .benchmark.json")
    ap.add_argument("--ai-voice",
                    help="also run Rig 4 (Seed-VC AI singing voice conversion) using this target reference WAV (or 'auto')")
    ap.add_argument("--apply-live", action="store_true",
                    help="execute the companion .autotune.lua ReaScript in REAPER and save all VST/VST3 states")
    ap.add_argument("--open", action="store_true",
                    help="open the finished .rpp project in REAPER")
    ap.add_argument("--output", "-o",
                    help="output .rpp project path (default: out/autotune-studio/autotune-studio.rpp)")
    args = ap.parse_args()

    if args.check_setup:
        print_setup_check()
        if not (args.synth_demo or args.input or args.karaoke_backing or args.youtube or args.launch_live_vc):
            return 0

    if args.launch_live_vc:
        svc = hostenv.seed_vc_paths()
        if not svc["ready"]:
            S.die("Seed-VC is not installed. Run: bash scripts/install/seed-vc.sh")
        print(f"🚀 launching Seed-VC real-time GUI ({svc['realtime_gui']}) …")
        env = dict(os.environ)
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        subprocess.Popen([str(svc["python"]), str(svc["realtime_gui"])], cwd=str(svc["repo"]), env=env)
        return 0

    dst = Path(args.output or "out/autotune-studio/autotune-studio.rpp")
    if dst.suffix.lower() != ".rpp":
        dst = dst.with_suffix(".rpp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    (dst.parent / "recordings").mkdir(parents=True, exist_ok=True)
    stem_base = dst.with_suffix("")

    backing_wav = Path(args.karaoke_backing).expanduser() if args.karaoke_backing else None
    vocal_take_wav = Path(args.vocal_take).expanduser() if args.vocal_take else None
    melody_mid = Path(args.midi).expanduser() if args.midi else None
    video_mp4 = Path(args.video).expanduser() if args.video else None
    ref_vocals_wav = None

    # Mode 1: Synthetic Out-of-Tune Demo
    if args.synth_demo:
        print("🎹 generating synthetic C-Major backing, target melody MIDI, and out-of-tune vocal take …")
        demo = generate_synth_demo_assets(dst.parent, bpm=args.bpm)
        backing_wav = demo["backing_wav"]
        vocal_take_wav = vocal_take_wav or demo["bad_vocal_wav"]
        melody_mid = melody_mid or demo["melody_mid"]
        args.key = args.key or demo["key"]
        if args.scale == "auto":
            args.scale = demo["scale"]

    # Mode 2: YouTube Karaoke URL
    if args.youtube:
        dl_video, dl_backing = kar.download_youtube_karaoke(args.youtube, dst.parent, stem=stem_base.name)
        video_mp4 = video_mp4 or dl_video
        backing_wav = backing_wav or dl_backing

    # Mode 3: Full song input (-i) or --reference-song -> separate stems via Demucs & extract melody MIDI
    song_for_stems = args.input or args.reference_song
    if song_for_stems:
        src_song = Path(song_for_stems).expanduser()
        if not src_song.is_file():
            S.die(f"input song not found: {src_song}")
        inst_dst = stem_base.with_name(f"{stem_base.name}.instrumental.wav")
        voc_dst = stem_base.with_name(f"{stem_base.name}.vocals.wav")
        if not (inst_dst.exists() and voc_dst.exists()):
            engine = kar.resolve_engine("auto")
            kar.separate_stems(src_song, inst_dst, voc_dst, engine)
        if not backing_wav:
            backing_wav = inst_dst
        ref_vocals_wav = voc_dst

    if not backing_wav and not vocal_take_wav:
        ap.error("provide --synth-demo, --input (-i), --karaoke-backing, --youtube (-y), or --check-setup")

    # Key & Scale detection
    if not args.key or args.scale == "auto":
        detect_src = backing_wav if (backing_wav and backing_wav.is_file()) else vocal_take_wav
        det_root, det_mode, det_conf = detect_key_and_scale(detect_src)
        key_root = normalize_note_name(args.key) if args.key else det_root
        scale_mode = det_mode if args.scale == "auto" else args.scale
        print(f"🎼 detected musical key/scale: {key_root} {scale_mode} (chroma correlation={det_conf:+.2f})")
    else:
        key_root = normalize_note_name(args.key)
        scale_mode = args.scale

    scale_mask = scale_pitch_classes(key_root, scale_mode)
    allowed_notes = scale_note_names(key_root, scale_mode)

    # Extract monophonic vocal guide MIDI if we have an isolated reference vocal stem and no --midi yet
    if not melody_mid and ref_vocals_wav and ref_vocals_wav.is_file():
        melody_mid = stem_base.with_name(f"{stem_base.name}.melody.mid")
        print(f"🎵 extracting monophonic vocal melody MIDI from {ref_vocals_wav.name} → {melody_mid.name} …")
        times, f0_hz, voiced_mask = extract_f0_contour(ref_vocals_wav)
        events = quantize_f0_to_midi_events(times, f0_hz, voiced_mask, scale_mask_c_to_b=scale_mask)
        write_monophonic_midi(events, melody_mid, bpm=args.bpm)
        S.report(melody_mid, f"{len(events)} quantized vocal guide notes")

    # Analyze vocal take (if provided) or backing track for level/EQ staging
    if vocal_take_wav and vocal_take_wav.is_file():
        prod_plan = sess.analyze_vocal_take(vocal_take_wav, backing_wav)
    else:
        back_info = S.probe(backing_wav)
        back_lufs = S.measure(backing_wav)
        back_gain_db = round(max(-8.0, min(0.0, -14.0 - back_lufs["i"])), 1)
        prod_plan = {
            "vocal_file": None,
            "duration": back_info["duration"],
            "rate": back_info["rate"],
            "channels": 2,
            "vocal_lufs": None,
            "backing_lufs": back_lufs,
            "backing_gain_db": back_gain_db,
            "trim_start": 0.0,
            "trim_length": back_info["duration"],
            "automation_points": [],
            "eq": {
                "hpf_hz": 95.0,
                "low_shelf_hz": 180.0,
                "low_shelf_db": 1.2,
                "mud_hz": 330.0,
                "mud_db": -4.0,
                "mud_bw": 0.75,
                "presence_hz": 3400.0,
                "presence_db": 4.2,
                "presence_bw": 1.1,
                "air_hz": 10500.0,
                "air_db": 3.2,
            },
            "echo": {
                "send_db": -12.5,
                "eighth_notes": 2.0,
                "feedback_norm": 0.3548,
                "lowpass_hz": 3800.0,
                "highpass_hz": 220.0,
            },
            "reverb": {
                "send_db": -10.5,
                "room_size": 0.72,
                "predelay_norm": 0.070,
            },
        }

    graillon_path = hostenv.find_vst3("Auburn Sounds Graillon 3") or hostenv.find_vst3("Graillon 3")
    markers = sess.parse_markers_file(args.markers) if args.markers else []

    plan = {
        **prod_plan,
        "key": key_root,
        "scale": scale_mode,
        "scale_notes": allowed_notes,
        "scale_mask_c_to_b": scale_mask,
        "reatune_bitmask": reatune_scale_bitmask(scale_mask),
        "reatune_attack_ms": int(args.attack_ms),
        "bpm": float(args.bpm),
        "default_rig": args.default_rig,
        "backing_file": str(backing_wav.resolve()) if (backing_wav and backing_wav.is_file()) else None,
        "melody_midi": str(melody_mid.resolve()) if (melody_mid and melody_mid.is_file()) else None,
        "video_file": str(video_mp4.resolve()) if (video_mp4 and video_mp4.is_file()) else None,
        "graillon_vst3": str(graillon_path) if graillon_path else None,
        "markers": markers,
    }

    plan_path = stem_base.with_name(f"{stem_base.name}.autotune.json")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    S.report(plan_path, f"Auto-Tune plan ({key_root} {scale_mode}: {', '.join(allowed_notes)})")

    duration = build_autotune_rpp(dst, plan, sample_rate=S.RATE)
    S.report(dst, f"10-track Auto-Tune & A/B Comparison .rpp ({duration:.1f}s)")

    lua_path = stem_base.with_name(f"{stem_base.name}.autotune.lua")
    report_path = stem_base.with_name(f"{stem_base.name}.fx-report.txt")
    lua_path.write_text(build_autotune_lua(dst, report_path, plan), encoding="utf-8")
    S.report(lua_path, "ReaScript live FX & Graillon 3 configurator")

    if args.apply_live or args.benchmark:
        print("🎛️  applying live REAPER FX parameters (ReaTune, ReaVocode, Graillon 3, ReaEQ, Echo, Reverb) …")
        if apply_autotune_reascript(lua_path, report_path, headless_quit=not args.open) and report_path.exists():
            S.report(report_path, "verified live REAPER FX & VST3 state")
        else:
            print(f"⚠ timed out waiting for REAPER FX report → {report_path}", file=sys.stderr)

    if args.benchmark:
        if not (vocal_take_wav and vocal_take_wav.is_file()):
            S.die("--benchmark requires --vocal-take <wav> or --synth-demo")
        benchmark_all_rigs(dst, plan, ai_voice_ref=args.ai_voice)
    elif args.ai_voice and vocal_take_wav and vocal_take_wav.is_file():
        svc_out = stem_base.with_name(f"{stem_base.name}.seedvc.wav")
        run_seed_vc_conversion(vocal_take_wav, args.ai_voice, svc_out)
        S.report(svc_out, "Seed-VC AI singing voice stem")

    if args.open and not (args.apply_live or args.benchmark):
        opener = "open" if hostenv.IS_MAC else "xdg-open"
        subprocess.run([opener, str(dst)], check=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
