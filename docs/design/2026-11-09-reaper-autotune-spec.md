# Specification: REAPER Multi-Rig Real-Time Auto-Tune, A/B Benchmark & Seed-VC Pipeline

- **Status:** approved
- **Date:** 2026-11-09
- **Subsystems:** `skills/sound/scripts/autotune.py`, `skills/sound/scripts/session.py`, `skills/sound/scripts/karaoke.py`, `scripts/hostenv.py`, `packages/manifest.json`, `install/`

---

## 1. Objective

Build an end-to-end, cross-platform (`macOS` + `Linux`) real-time vocal pitch-correction and A/B comparison workflow inside `creative-stack` so a user can hand over **either a full song (with vocals) or a karaoke instrumental/video**, sing live into REAPER with low-latency pitch rescue in their headphones, and immediately compare **without Auto-Tune vs. three distinct real-time Auto-Tune rigs (plus post-take/live AI voice conversion)** on the exact same vocal performance.

---

## 2. Official Source & API Grounding

* **REAPER ReaScript & Project State (`.rpp` / Lua):** [Cockos ReaScript API](https://www.reaper.fm/sdk/reascript/reascripthelp.html)
  * Live parameter control via `reaper.TrackFX_SetParam(tr, fx, param, val)` and `reaper.TrackFX_GetFormattedParamValue`.
  * `ReaEQ` band gain via `reaper.TrackFX_SetEQParam(..., paramtype=1, linear_gain, false)` using linear amplitude `10 ^ (dB / 20)` per [`session.db_to_linear_gain`](file:///Users/ksprashanth/code/github/creative-stack/skills/sound/scripts/session.py#L62-L70).
  * `ReaTune` automatic scale configuration via `reaper.GetTrackStateChunk` / `reaper.SetTrackStateChunk` + `TrackFX_SetParam`, verified by dumping active FX state to `<stem>.fx-report.txt` via `REAPER -nonewinst`.
  * Headless stem/mix rendering via `REAPER -renderproject <project.rpp>`.
* **Cockos `ReaVocode` (`reavocode.vst.dylib` / `libreavocode.so`):**
  * 4-channel track routing (`NCHAN 4`): channels `1/2` receive the microphone **Detector** signal; channels `3/4` receive the `ReaSynth` MIDI **Carrier** signal (`AUXRECV` destination channel index `2`).
* **Auburn Sounds Graillon 3 Free Edition (`VST3` / `LV2`):** [Auburn Sounds Graillon Documentation](https://www.auburnsounds.com/products/Graillon.html)
  * Cross-platform (`macOS` Universal `VST3`/`AU`, `Linux` `VST3`/`LV2`) real-time pitch shifter and corrector supporting both **Scale Snap** and **Live MIDI Target Pitch** (`Enable MIDI` + `Correction` + `Smooth`/`Inertia`).
  * Installed into user-scope plugin paths (`~/Library/Audio/Plug-Ins/VST3` on macOS, `~/.vst3` on Linux) so installation requires no system-partition mutation.
* **Pitch, Key & Melody MIDI Extraction (`librosa` + `mido`):** [Librosa `pyin` Docs](https://librosa.org/doc/latest/generated/librosa.pyin.html)
  * Key and scale (`Major` / `Minor`) estimated via `librosa.feature.chroma_cqt` correlated against Krumhansl-Schmuckler tonal profiles.
  * Monophonic reference vocal melody extracted from isolated `.vocals.wav` via `librosa.pyin` ($f_0$ + voiced probability), median-filtered to suppress vibrato/transient jitter, quantized to MIDI note numbers (`mido`), and written to `<stem>.melody.mid`.
* **Neural Stem Separation (`Demucs`):** [`karaoke.py`](file:///Users/ksprashanth/code/github/creative-stack/skills/sound/scripts/karaoke.py)
  * Uses `htdemucs --two-stems=vocals` when installed in `.venv`, falling back deterministically to bass-preserving (`< 130 Hz`) stereo `center-cut`.
* **Singing Voice Conversion — Rig 4 (`Plachtaa/seed-vc`):** [Seed-VC GitHub Repository](https://github.com/Plachtaa/seed-vc)
  * Batch/post-take Singing Voice Conversion via `inference.py --source <in.wav> --target <ref.wav> --output <dir> --diffusion-steps 40 --f0-condition True --auto-f0-adjust False --semi-tone-shift <int>`.
  * Real-time streaming experimentation via `real-time-gui.py` over Apple Silicon `MPS` + CoreAudio / Rogue Amoeba `Loopback` (`ARK.driver`).

---

## 3. Architecture & Chokepoint Design

### 3.1 Platform Chokepoints ([AGENTS.md](file:///Users/ksprashanth/code/github/creative-stack/AGENTS.md))
No `sys.platform` checks appear in skill scripts. All OS differences live in the repo's chokepoints:
1. **[`scripts/hostenv.py`](file:///Users/ksprashanth/code/github/creative-stack/scripts/hostenv.py):**
   * `vst3_dirs()` and `find_vst3(bundle_name)` -> resolves user and system VST3 directories across macOS (`~/Library/Audio/Plug-Ins/VST3`, `/Library/Audio/Plug-Ins/VST3`) and Linux (`~/.vst3`, `/usr/lib/vst3`, `/usr/local/lib/vst3`).
   * `reaper_fx_ident(plugin_key)` -> returns platform-appropriate native Cockos plugin binary names (`reatune.vst.dylib` vs `reatune.vst.so`, etc.) and VST3 identifiers.
   * `reaper_ini_path()` and `reaper_audio_settings()` -> read-only parser for `reaper.ini` reporting current audio device, sample rate, and block/buffer size for live-monitoring latency diagnostics.
   * `seed_vc_paths()` -> resolves `~/.local/share/creative-stack/seed-vc` (venv Python + pinned repo checkout).
2. **`install/graillon.sh` & `install/seed-vc.sh` (+ optional entries in [`packages/manifest.json`](file:///Users/ksprashanth/code/github/creative-stack/packages/manifest.json)):**
   * `install/graillon.sh`: idempotently downloads the official Auburn Sounds Graillon 3 Free archive, verifies non-empty payload, extracts `Auburn Sounds Graillon 3.vst3` into the user VST3 directory, clears macOS quarantine xattr on the extracted user bundle if present, and verifies presence via `hostenv.find_vst3`.
   * `install/seed-vc.sh`: idempotently provisions a `uv` Python `3.10` virtual environment under `~/.local/share/creative-stack/seed-vc/venv`, clones `https://github.com/Plachtaa/seed-vc.git` pinned to an audited commit SHA, and installs `requirements-mac.txt` (or Linux equivalent) without touching `creative-stack/.venv`.

### 3.2 `skills/sound/scripts/autotune.py` Data Flow
Every invocation writes a reviewable `<stem>.autotune.json` plan before generating `.mid`, `.rpp`, `.lua`, or rendered audio:

1. **Ingest & Stem Resolution:**
   * Accepts `--input <audio_or_video>`, `--youtube <url>`, or `--karaoke-backing <wav>` + optional `--reference-song <file_or_url>` / `--midi <file.mid>`.
   * Full songs are split into `<stem>.instrumental.wav` and `<stem>.vocals.wav` via `Demucs` (`htdemucs`).
   * Karaoke videos without lead vocals extract 48 kHz / 24-bit WAV + OCR lyric markers (`--ocr-lyrics`). When `--reference-song` is supplied, its isolated vocal stem is time-aligned to the karaoke backing via chroma cross-correlation so `<stem>.melody.mid` locks to the karaoke timeline.
2. **Key/Scale & Melody MIDI Extraction:**
   * Computes root pitch class (`C`..`B`), mode (`major` / `minor`), 12-note scale mask, and BPM.
   * When an isolated vocal stem exists, runs `librosa.pyin` (`fmin=65 Hz`, `fmax=1046 Hz`), filters unvoiced frames and `< 80 ms` micro-glitches, quantizes pitches to MIDI notes, and writes `<stem>.melody.mid`.
3. **10-Track REAPER Session + Live ReaScript (`.rpp` + `.autotune.lua`):**
   * **Track 1 — `01 - Dry Mic Input (Record Here)`**: Record-armed (`REC 1 0 1 ...`), **`MAINSEND 0 0`** (prevents raw out-of-tune voice from bleeding directly to Master while monitoring), with Pre-Fader sends (`AUXRECV` mode `3`) to Tracks 2–5. Recording once on Track 1 feeds all rigs identically on playback.
   * **Track 2 — `02 - Rig 0: Bypass (No Auto-Tune Control)`**: Muted during live recording (`MUTESOLO 1 0 0`); carries the identical studio vocal chain (`ReaGate` -> `ReaEQ` -> Dual `ReaComp` -> `JS: Liteon/deesser` -> Echo/Reverb sends) with **zero pitch correction** so A/B Solo comparison isolates pitch correction alone without loudness or reverb bias.
   * **Track 3 — `03 - Rig 1A: Stock ReaTune (Scale-Snap)`**: `ReaTune` (`Automatic pitch correction` enabled, locked to detected song key/scale, `20 ms` attack, low-latency pitch shifter) + studio vocal chain. Unmuted as live monitor when no melody MIDI exists; muted when Rig 2 is the active monitor.
   * **Track 4 — `04 - Rig 1B: Stock ReaVocode (MIDI Synth Carrier)`**: `NCHAN 4` track with `ReaVocode` (`Ch 1/2` Mic Detector + `Ch 3/4` Synth Carrier) + studio vocal chain; muted by default.
   * **Track 5 — `05 - Rig 2: Graillon 3 (MIDI-Locked)`**: `Auburn Sounds Graillon 3.vst3` configured for live MIDI pitch-tracking when `<stem>.melody.mid` is present (or Scale Snap when instrumental-only) + studio vocal chain. **Unmuted default live headphone monitor** when Graillon 3 + MIDI are present.
   * **Track 6 — `06 - Guide Melody MIDI + Synth Carrier`**: `MAINSEND 0 0`; holds `<stem>.melody.mid` item + `ReaSynth`, routing MIDI (`Ch 1`) to Track 5 (`Graillon 3`) and `ReaSynth` audio output to Track 4 `Ch 3/4` (`ReaVocode` carrier).
   * **Track 7 — `07 - Vocal Echo (Stereo Delay Bus)`**: Tempo-synced `ReaDelay` + `ReaEQ` receiving equal post-fader sends from Tracks 2–5.
   * **Track 8 — `08 - Vocal Reverb (Lush Plate Bus)`**: `ReaVerbate` + Abbey Road `ReaEQ` receiving equal post-fader sends from Tracks 2–5.
   * **Track 9 — `09 - Backing Track (Instrumental)`**: Gain-staged to `-14 LUFS` headroom + `ReaEQ` `-2.2 dB` vocal pocket at `3.2 kHz`.
   * **Track 10 — `10 - Lyrics Video (Press Cmd+Shift+V)`**: Muted video track (`MUTESOLO 1 0 0`) for synced lyric teleprompter playback.
4. **Objective A/B Benchmark (`--benchmark <take.wav>`):**
   * Places `<take.wav>` on Track 1, renders stem outputs for **Rig 0 (Bypass)**, **Rig 1A (`ReaTune`)**, **Rig 1B (`ReaVocode`)**, and **Rig 2 (`Graillon 3`)** (plus **Rig 4 (`Seed-VC`)** when `--ai-voice` is supplied) via `REAPER -renderproject`.
   * Measures $f_0$ contours via `librosa.pyin` against both the song's scale and `<stem>.melody.mid`, writing `<stem>.benchmark.json` (`median_abs_cents`, `mean_abs_cents`, `pct_within_25_cents`, `pct_within_50_cents`, `lufs`) and an HTML comparison report (`reports/` / `out/<stem>/autotune-report.html`).
5. **Rig 4 Integration (`--ai-voice <target_ref.wav>` & `--launch-live-vc`):**
   * `--ai-voice`: Runs `Seed-VC` `inference.py` (`--f0-condition True`) on the pitch-corrected Rig 2 stem (so AI timbre conversion inherits 100% corrected pitch!), resamples back to `48 kHz / 24-bit WAV`, and inserts Track `06B - Rig 4: Seed-VC AI Vocal` into the `.rpp`.
   * `--launch-live-vc`: Prints detected CoreAudio / Loopback routing tips and launches `Seed-VC` `real-time-gui.py` from `~/.local/share/creative-stack/seed-vc`.

---

## 4. Tooling, Security & Environment Boundaries

* **No ambient MCP servers assumed:** Everything executes via local deterministic Python scripts, FFmpeg, and REAPER CLI flags (`-nonewinst`, `-renderproject`).
* **Supply-chain & privilege hygiene:**
  * Zero `sudo` system-wide mutations required: `Graillon 3.vst3` installs into `~/Library/Audio/Plug-Ins/VST3` (macOS) or `~/.vst3` (Linux); `Seed-VC` installs into `~/.local/share/creative-stack/seed-vc`.
  * `Plachtaa/seed-vc` is cloned at an explicit pinned commit SHA, never a floating branch.
  * Zero secrets or API keys used or stored anywhere.

---

## 5. Explicit Non-Goals

* **Live causal correction of late vocal timing:** Real-time FX cannot play syllables the singer has not sung yet; rhythm/timing alignment remains an offline post-recording step.
* **Custom RVC model training:** Rig 4 uses `Seed-VC` zero-shot reference audio (`10–30 s` WAV) rather than multi-hour `.pth` dataset training.
* **User-scope MCP configuration edits:** Prohibited by [`AGENTS.md`](file:///Users/ksprashanth/code/github/creative-stack/AGENTS.md).

---

## 6. Acceptance Criteria

* [ ] **AC-1 (Dependencies & Plugins Ready):** `Auburn Sounds Graillon 3.vst3` verified in `~/Library/Audio/Plug-Ins/VST3` and scanned cleanly by REAPER; `demucs` + `torchaudio` verified in `.venv`; `Plachtaa/seed-vc` verified in `~/.local/share/creative-stack/seed-vc` on Apple Silicon `MPS`.
* [ ] **AC-2 (Audio & Latency Diagnostics):** `autotune.py --check-setup` reports host audio devices, Loopback/HAL drivers, REAPER buffer/sample-rate settings, VST3 discovery, Demucs status, and `Seed-VC` status.
* [ ] **AC-3 (Multi-Rig REAPER Project Generation):** Given either a full song or a karaoke track, `autotune.py` emits `<stem>.autotune.json`, `<stem>.melody.mid` (when vocal reference exists), `<stem>.rpp`, `<stem>.autotune.lua`, and verified `<stem>.fx-report.txt` with all 4 rigs wired for one-click Solo A/B comparison.
* [ ] **AC-4 (Quantitative With-vs-Without Proof):** On a deliberately out-of-tune test vocal (`50–200 cents` off-pitch + wrong-scale drift), `--benchmark` renders all rigs through REAPER and proves measurably lower cent error and higher `% in-tune frames` on Rigs `1A`, `1B`, and `2` vs. `Rig 0 (Bypass)`, captured in `<stem>.benchmark.json` and an Atelier HTML visual report.
* [ ] **AC-5 (Repo Quality Gates):** `python3 scripts/validate_skills.py`, `python3 -m unittest discover -s tests`, and `--help` standalone checks on every modified/added script exit `0`.
