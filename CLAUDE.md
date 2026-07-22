# SwaraShruti — Claude Code Context

## What this project is
SwaraShruti converts hummed or sung audio into realistic instrument playback.
Name derives from Sanskrit: Swara (a single musical note) + Shruti (pitch / that which is heard).

**Pipeline:** voice/hum input → CREPE pitch detection → pitch correction → MIDI construction → soundfont renderer → instrument audio output

**Two parallel goals:** build a working end-to-end product AND gain deep hands-on AI/ML experience.

**Future product vision:** web app with multi-instrument selection, animated instrument UI synced to MIDI playback, copyright infringement checking, and a multitrack DAW-style mixer.

---

## Repository
- **Remote:** `git@github-personal:kk-shetty/swarashruti.git`
- **Local:** `~/Documents/PersonalProjects/swarashruti`

---

## Stack
- Python 3.12, managed via `uv`
- CREPE (pitch detection), librosa, pretty_midi, pyfluidsynth
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

**uv + HubSpot PyPI mirror conflict**
Global `~/.config/uv/uv.toml` points to HubSpot's internal PyPI mirror. For all personal project uv commands, always pass `--index https://pypi.org/simple/` as a flag. Never modify the global config.

**npm global installs + HubSpot registry**
Global `~/.npmrc` routes to HubSpot's internal registry. For personal project global installs, pass `--registry https://registry.npmjs.org` as a flag.

**SSH config append rule**
`~/.ssh/config` must always use `>>` to append — never `>`, which overwrites. The personal GitHub SSH block was previously lost this way.

**gh CLI multi-account switching**
Automatic switching between HubSpot and personal accounts is handled via a custom `cd()` function in `~/.zshrc` that calls `gh auth switch --user` based on working directory, combined with `[includeIf "gitdir:..."]` blocks in `~/.gitconfig`.

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

---

## Current sprint

### Sprint 1 — Core Pipeline: Pitch Detection (Jul 28 – Aug 11 2026)
Goal: working, tested pitch detection module. Audio in → cleaned, note-mapped pitch sequence out. No MIDI yet.

| Ticket | Title | Points | Status |
|--------|-------|--------|--------|
| SWARA-15 | Load and validate audio input | 3 | To Do |
| SWARA-16 | Integrate CREPE for pitch detection | 5 | To Do |
| SWARA-17 | Filter low-confidence frames | 2 | To Do |
| SWARA-18 | Map Hz to musical notes | 3 | To Do |
| SWARA-19 | Write ADR-002: CREPE vs alternatives | 1 | To Do |

---

## Key domain knowledge
- CREPE architecture: 1D CNN, 360-bin softmax output, ~32–1975 Hz range
- Why classification outperforms regression for pitch estimation: avoids single-value collapse on ambiguous frames
- PESTO (PyTorch-based): comparable accuracy to CREPE at significantly lower compute cost — natural migration path if inference speed becomes a bottleneck
- PENN: highly accurate within a specific domain but lacks CREPE's cross-domain robustness
- The web app future reinforces the offline pipeline model: browser submits audio via API, Python engine processes and returns rendered audio — no real-time DSP required on the Python side

---

## ADRs filed
| ADR | Decision | Status |
|-----|----------|--------|
| ADR-001 | Python as core engine language | Accepted |

---

## How to update this file
After each sprint: update the Completed sprints table, move Current sprint to Completed, add the new sprint under Current sprint, and update ADRs filed if new ones were added.
