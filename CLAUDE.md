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
- torchcrepe (pitch detection), librosa, pretty_midi, pyfluidsynth
- ruff (linting + formatting), pytest (testing), pre-commit hooks
- Jira board: `swarashruti.atlassian.net` (project key: SWARA)

---

## Architecture
```
core/        — pure logic, no framework dependencies
tests/       — mirrors core/ structure
docs/adr/    — architecture decision records
data/        — audio input/output fixtures
models/      — ML model weights
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

**SWARA-15 output:** `core/audio/audio_input.py` — `load_audio(file_path: str) -> np.ndarray`, hardcoded `PIPELINE_SAMPLE_RATE = 16000`.

**SWARA-16 output:** `core/pitch/pitch_detector.py` — `detect_pitch(audio: np.ndarray, sr: int = PIPELINE_SAMPLE_RATE) -> dict[str, np.ndarray]`, returns `frequency`/`periodicity`/`time` arrays via `torchcrepe.predict` (full model, `fmin=50.0`, `fmax=550.0`, `hop_length=160`).

**SWARA-17 output:** `core/pitch/pitch_filter.py` — `filter_by_periodicity(pitch_data, threshold: float = DEFAULT_THRESHOLD) -> dict[str, np.ndarray]`, zeroes frequency for frames below `DEFAULT_THRESHOLD = 0.21` (torchcrepe authors' recommended threshold for clean speech) rather than dropping frames.

**SWARA-18 output:** `core/pitch/note_mapper.py` — `hz_to_midi(pitch_data) -> dict[str, np.ndarray]`, adds a rounded `midi_note` array; frequency ≤ 0 maps to MIDI note 0.

**SWARA-19 output:** `docs/adr/ADR-002-pitch-detection-library.md` — documents why `torchcrepe` was chosen over the original `crepe` PyPI package (broken build metadata, abandoned since 2019, TensorFlow dependency).

---

## Current sprint

Sprint 2 not yet planned.

---

## Key domain knowledge
- CREPE architecture: 1D CNN, 360-bin softmax output, ~32–1975 Hz range
- Why classification outperforms regression for pitch estimation: avoids single-value collapse on ambiguous frames
- torchcrepe is a PyTorch reimplementation of CREPE using the same pre-trained weights — outputs are equivalent
- torchcrepe returns `periodicity` not `confidence` — same concept, different numerical values; use the term "periodicity" in code and comments
- torchcrepe does not expose `__version__` — verify installation with `import torchcrepe; print('torchcrepe imported OK')`
- PESTO (PyTorch-based): comparable accuracy to CREPE at significantly lower compute cost — natural migration path if inference speed becomes a bottleneck
- PENN: highly accurate within a specific domain but lacks CREPE's cross-domain robustness
- The web app future reinforces the offline pipeline model: browser submits audio via API, Python engine processes and returns rendered audio — no real-time DSP required on the Python side

---

## ADRs filed
| ADR | Decision | Status |
|-----|----------|--------|
| ADR-001 | Python as core engine language | Accepted |
| ADR-002 | torchcrepe as pitch detection library | Accepted |

---

## How to update this file
After each sprint: update the Completed sprints table, move Current sprint to Completed, add the new sprint under Current sprint, and update ADRs filed if new ones were added.
