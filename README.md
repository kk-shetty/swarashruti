# swarashruti

Convert hummed audio to instrument playback.

Hum a melody, get back a rendered musical performance. The pipeline handles pitch
detection from raw audio, maps detected pitches to MIDI notes, and synthesizes the
result through a chosen instrument voice.

> **Status:** early development — the scaffold is in place, core modules are being built out.

---

## Tech stack

| Layer | Library | Why |
|-------|---------|-----|
| Pitch detection | CREPE | Neural network approach; more accurate than autocorrelation on sung/hummed input |
| Audio processing | librosa | Industry standard for audio feature extraction in Python |
| MIDI construction | pretty_midi | Clean API for building and manipulating MIDI programmatically |
| Synthesis | pyfluidsynth | Renders MIDI through soundfonts for realistic instrument output |
| Package management | uv | Fast, lockfile-based, replaces pip + venv in one tool |

---

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — used for all dependency and virtualenv management

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

## Quick start

```bash
git clone git@github.com:kk-shetty/swarashruti.git
cd swarashruti

# Create virtualenv and install all dependencies (including dev deps)
uv sync

# Verify the setup by running the test suite
uv run pytest
```

That's it. No manual `pip install`, no `python -m venv`, no activation step needed — `uv run` handles the venv automatically.

---

## Development workflow

### Running tests

```bash
uv run pytest
```

### Linting and formatting

```bash
# Check for lint errors
uv run ruff check

# Auto-fix lint errors where possible
uv run ruff check --fix

# Format code
uv run ruff format

# Check formatting without writing changes
uv run ruff format --check
```

### Pre-commit hooks

The repo ships with pre-commit hooks that run `ruff check --fix` and `ruff format` on every commit.

```bash
# Install the hooks (one-time setup after cloning)
pre-commit install

# Run hooks manually against all files
pre-commit run --all-files
```

---

## Project structure

```
swarashruti/
├── core/
│   ├── audio/       # Audio I/O and preprocessing
│   ├── pitch/       # Pitch detection from audio signal
│   ├── midi/        # Pitch-to-MIDI mapping and sequencing
│   └── synthesis/   # Instrument synthesis / playback
├── api/             # External API layer
├── ui/              # User interface
├── scripts/         # Utility scripts
├── data/            # Sample audio and reference data
├── docs/
│   └── adr/         # Architecture decision records
└── tests/           # Mirrors core/ structure
    ├── audio/
    ├── pitch/
    ├── midi/
    └── synthesis/
```

---

## Roadmap

- **Phase 1** — Core pipeline: audio input → pitch detection → MIDI → synthesized output
- **Phase 2** — Web interface with instrument selection and animated playback UI
- **Phase 3** — Multitrack mixer and DAW-style controls
- **Phase 4** — Copyright infringement detection for melodies

---

## License

MIT — see [LICENSE](LICENSE).