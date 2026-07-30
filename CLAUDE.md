# SwaraShruti — Claude Code Context

## What this project is
SwaraShruti converts hummed or sung audio into realistic instrument playback.
Name derives from Sanskrit: Swara (a single musical note) + Shruti (pitch / that which is heard).

**Pipeline:** voice/hum input → torchcrepe pitch detection → pitch correction → MIDI construction → soundfont renderer → instrument audio output

**Two parallel goals:** build a working end-to-end product AND gain deep hands-on AI/ML experience.

**Future product vision:** web app with multi-instrument selection, animated instrument UI synced to MIDI playback, copyright infringement checking, and a multitrack DAW-style mixer.

---

## Repository
- **Remote:** `git@github-personal:kk-shetty/swarashruti.git`
- **Local:** `~/Documents/PersonalProjects/swarashruti`

---

## Stack
- Python 3.12, managed via `uv`, pinned via `.python-version`
- torchcrepe (pitch detection), librosa, pretty_midi, pyfluidsynth, soundfile
- ruff (linting + formatting), pytest (testing), pre-commit hooks
- Jira board: `swarashruti.atlassian.net` (project key: SWARA)

---

## Architecture
```
core/        — pure logic, no framework dependencies
tests/       — mirrors core/ structure
docs/adr/    — architecture decision records
scripts/     — local dev utilities (not production code)
models/      — ML model weights (gitignored)
output/      — synthesized audio output (gitignored)
debug/       — intermediate pipeline arrays for diagnosis (gitignored)
```

---

## Engineering principles
- Separation of concerns
- Reversible design
- Test-as-you-build — all functions must have tests before marking a ticket Done
- ADRs for every significant architectural decision
- Conventional commits, no space before colon (`chore:`, `feat:`, `docs:`, `fix:`, `test:`)
- Sustainable marathon pace — not speed sprints

---

## Critical environment gotchas

**Python version**
`.python-version` at project root pins Python 3.12. uv respects this automatically.
Never remove it — without it, uv defaults to the highest available Python, which breaks
older packages like torchcrepe's transitive dependencies.

**uv PyPI index**
`pyproject.toml` sets `[[tool.uv.index]]` with `default = true` pointing to `https://pypi.org/simple/`.
This overrides any global uv config (e.g. HubSpot's internal mirror on managed machines).
Do not remove the `default = true` flag.

**torchcrepe import verification**
`torchcrepe.__version__` does not exist. Verify installation with:
`import torchcrepe; print('torchcrepe imported OK')`

**Running scripts**
Scripts in `scripts/` must be run with `PYTHONPATH=.` to resolve `core/` imports:
`PYTHONPATH=. uv run python scripts/try_pipeline.py path/to/audio.m4a`

**HubSpot-managed machine only**
The following gotchas apply only when working from a HubSpot-managed laptop:
- Global `~/.config/uv/uv.toml` points to HubSpot's internal PyPI mirror — the project-level index config overrides this, but be aware of it.
- Global `~/.npmrc` routes to HubSpot's internal registry. For personal project global installs, pass `--registry https://registry.npmjs.org` as a flag.
- `~/.ssh/config` must always use `>>` to append — never `>`, which overwrites.
- `gh` CLI account switching is handled via a `cd()` function in `~/.zshrc` combined with `[includeIf "gitdir:..."]` blocks in `~/.gitconfig`.

**Directory boundaries**
- HubSpot work: `~/Documents/HubSpot/`
- Personal projects: `~/Documents/PersonalProjects/`
- Simpro: `~/Documents/Simpro/`

---

## Scrum conventions
- Story points: 1=trivial, 2=small, 3=medium, 5=large, 8=needs breakdown
- Jira statuses: To Do → In Progress → In Review → Blocked → Done
- Ticket cycle: pre-research → build → review → commit → update Jira comments
- Claude reviews completed tickets before the next one opens

---

## Completed sprints

### Sprint 0 — Project Foundation (Jul 14–28 2026) ✅
| Ticket | Title | Commit |
|--------|-------|--------|
| SWARA-6 | Git repo + .gitignore | `chore: initial repo setup` |
| SWARA-7 | pyproject.toml with uv sync verified | `chore: add pyproject.toml with project metadata` |
| SWARA-8 | Folder scaffold with .gitkeep files | `chore: scaffold project folder structure` |
| SWARA-9 | pytest configured | `chore: configure pytest` |
| SWARA-10 | ruff configured | `chore: configure ruff` |
| SWARA-11 | pre-commit hooks | `chore: add pre-commit hooks` |
| SWARA-12 | README.md | `docs: add README` |
| SWARA-13 | CHANGELOG.md | `docs: add CHANGELOG with initial 0.1.0 entry` |
| SWARA-14 | ADR-001: Python as core engine | `docs: add ADR-001 for Python core engine decision` |

### Sprint 1 — Core Pipeline: Pitch Detection (Jul 28 – Aug 11 2026) ✅
Goal: working, tested pitch detection module. Audio in → cleaned, note-mapped pitch sequence out. No MIDI yet.

| Ticket | Title | Points | Commit |
|--------|-------|--------|--------|
| SWARA-15 | Load and validate audio input | 3 | `feat: add audio input loader with 16kHz normalisation` |
| SWARA-16 | Integrate torchcrepe for pitch detection | 5 | `feat: integrate torchcrepe pitch detection` |
| SWARA-17 | Filter low-confidence frames | 2 | `feat: add periodicity-based pitch frame filter` |
| SWARA-18 | Map Hz to musical notes | 3 | `feat: add Hz to musical note mapper` |
| SWARA-19 | Write ADR-002: pitch detection library | 1 | `docs: add ADR-002 pitch detection library` |

**Key outputs:**
- `core/audio/audio_input.py` — `load_audio(path) -> np.ndarray`, `PIPELINE_SAMPLE_RATE = 16000`, `SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}`
- `core/pitch/pitch_detector.py` — `detect_pitch(audio, sr) -> dict`, returns `{time, frequency, periodicity}` via torchcrepe full model (`fmin=50.0`, `fmax=550.0`, `hop_length=160`)
- `core/pitch/pitch_filter.py` — `filter_by_periodicity(pitch_data, threshold=0.21) -> dict`, zeroes frequency below threshold; threshold 0.21 is torchcrepe authors' recommendation
- `core/pitch/note_mapper.py` — `hz_to_midi(pitch_data) -> dict`, adds `midi_note` array; silent frames map to MIDI 0

### Sprint 2 — MIDI Construction and Synthesis (Aug 11 – Aug 25 2026) ✅
Goal: full end-to-end pipeline. Pitch data → MIDI → synthesized audio output.

| Ticket | Title | Points | Commit |
|--------|-------|--------|--------|
| SWARA-20 | Note segmentation | 3 | `feat: add note segmenter` |
| SWARA-21 | MIDI builder | 2 | `feat: add MIDI builder` |
| SWARA-22 | FluidSynth synthesizer | 3 | `feat: add FluidSynth synthesizer` |
| SWARA-23 | ADR-003: soundfont strategy | 1 | `docs: add ADR-003 soundfont strategy` |
| SWARA-24 | Soundfont download script | 2 | `chore: add soundfont download script` |
| SWARA-25 | Refactor fixture to conftest | 1 | `refactor: move make_mono_wav_file fixture to conftest.py` |
| SWARA-26 | End-to-end pipeline integration | 3 | `feat: add end-to-end pipeline integration` |

**Key outputs:**
- `core/midi/note_segmenter.py` — `segment_notes(pitch_data, min_duration=0.05) -> list[dict]`, returns `[{midi_note, start_time, end_time}]`
- `core/midi/midi_builder.py` — `build_midi(note_events, tempo=120.0, velocity=100) -> PrettyMIDI`, `PIANO_PROGRAM=0`, `DEFAULT_VELOCITY=100`
- `core/synthesis/synthesizer.py` — `synthesize(midi, soundfont_path, sample_rate=44100) -> np.ndarray`, `OUTPUT_SAMPLE_RATE=44100`, `SOUNDFONT_PATH="models/FluidR3_GM.sf2"`
- `core/pipeline.py` — `run_pipeline(audio_path, soundfont_path, debug=False, debug_dir="debug") -> np.ndarray`
- `scripts/download_soundfont.sh` — fetches FluidR3_GM.sf2 (~144MB) to `models/`
- `scripts/try_pipeline.py` — local dev script: `PYTHONPATH=. uv run python scripts/try_pipeline.py path/to/audio.m4a`
- `tests/conftest.py` — `make_mono_wav_file` shared fixture
- 24 tests passing

**Real-world test findings (Twinkle Twinkle Little Star, hummed):**
- Vocal range landed in MIDI 46–60 (C3 register) — one octave below initial prediction, normal for adult male voice
- torchcrepe marked 100% of frames as voiced — breath/silence sections not gated
- 251 one-semitone jumps (13.6% of frames) — pitch jitter between adjacent semitones (50↔51, 57↔58)
- Increasing `min_duration` to 0.1s produced marginal improvement — confirms jitter must be addressed upstream of segmentation
- Fix identified: median filter on voiced MIDI frames, kernel_size=5 (50ms window)

---

## Current sprint

### Sprint 3 — Audio Quality: Pitch Smoothing (Aug 25 – Sep 8 2026)
Goal: pipeline produces clean, listenable output on real-world humming input.

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| SWARA-27 | Commit pending real-world test changes | 2 | To Do |
| SWARA-28 | Add pitch smoothing with median filter | 3 | To Do |
| SWARA-29 | Add energy-based silence gate | 3 | To Do |
| SWARA-30 | Update CLAUDE.md and README | 1 | To Do |

---

## Key domain knowledge
- CREPE architecture: 1D CNN, 360-bin softmax output, ~32–1975 Hz range
- Why classification outperforms regression for pitch estimation: avoids single-value collapse on ambiguous frames
- torchcrepe is a PyTorch reimplementation of CREPE using the same pre-trained weights — outputs are equivalent
- torchcrepe returns `periodicity` not `confidence` — same concept, different numerical values; use the term "periodicity" in code and comments
- torchcrepe does not expose `__version__` — verify installation with `import torchcrepe; print('torchcrepe imported OK')`
- torchcrepe marks all frames as voiced by default — periodicity filtering removes low-confidence frames but does not gate silence; an energy-based gate is needed upstream
- Pitch jitter root cause: torchcrepe estimates pitch every 10ms; raw Hz drifts around a semitone boundary; MIDI rounding causes adjacent-semitone oscillation on sustained notes
- PESTO (PyTorch-based): comparable accuracy to CREPE at significantly lower compute cost — natural migration path if inference speed becomes a bottleneck
- PENN: highly accurate within a specific domain but lacks CREPE's cross-domain robustness
- The web app future reinforces the offline pipeline model: browser submits audio via API, Python engine processes and returns rendered audio — no real-time DSP required on the Python side

---

## ADRs filed
| ADR | File | Decision | Status |
|-----|------|----------|--------|
| ADR-001 | `ADR-001-python-core-engine.md` | Python as core engine language | Accepted |
| ADR-002 | `ADR-002-pitch-detection-library.md` | torchcrepe over original crepe (broken PyPI package) | Accepted |
| ADR-003 | `ADR-003-soundfont-strategy.md` | FluidR3_GM via download script | Accepted |
| ADR-004 | `ADR-004-pitch-smoothing-strategy.md` | Median filter on voiced MIDI frames, kernel=5 | Accepted |

---

## How to update this file
After each sprint: update the Completed sprints table, move Current sprint to Completed, add the new sprint under Current sprint, and update ADRs filed if new ones were added.