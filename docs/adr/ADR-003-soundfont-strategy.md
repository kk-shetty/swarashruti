# ADR-003: Soundfont Strategy

## Status
Accepted

## Date
2026-07-27

## Context
SwaraShruti's core pipeline converts hummed audio into MIDI. After pitch detection
and note mapping, the final stage renders MIDI into audible audio. This requires a
synthesis engine and a soundfont — a binary file containing sampled instrument sounds
that the engine plays back at the correct pitch and duration.

`pyfluidsynth` was selected as the synthesis engine because it is the de facto Python
binding for FluidSynth, the most widely used open-source MIDI synthesizer. FluidSynth
renders MIDI through soundfont files in the SF2 format.

This ADR resolves how the SF2 soundfont file is made available to the project — not
which synthesis engine to use.

## Options Considered

### Option A — Bundle in repository
Commit the SF2 file directly to the repo.

Suitable soundfonts small enough to bundle (e.g. TimGM6mb.sf2 at 5.7MB) produce
noticeably lower audio quality. Higher quality fonts exceed 100MB, which is
inappropriate for a git repository. Git is not designed for binary assets — even a
small binary creates permanent object store bloat and makes clones heavier over time.

### Option B — Download script
Commit a shell script that downloads the SF2 file to `models/` on first run. The
script is tracked; the binary is `.gitignore`'d.

This keeps the repository lean, supports a high-quality soundfont, and remains fully
reproducible — anyone can recover the exact binary by running the script. The only
cost is a single manual setup step after cloning.

### Option C — Git LFS
Track the binary through Git Large File Storage rather than the main object store.

LFS has per-account storage and bandwidth quotas, requires LFS to be installed on
every machine that clones, and adds infrastructure overhead. The benefit over a
download script is marginal for a single file in a personal project.

### Option D — User-provided
Document an expected path and require contributors to supply their own soundfont.

This produces the worst developer experience — anyone running the project fresh has
no documented, reproducible path to the correct file. Quality and compatibility would
vary across contributors.

## Decision
Use **Option B — download script with FluidR3_GM**.

FluidR3_GM is the standard general MIDI soundfont used across pyfluidsynth tutorials
and open-source MIDI projects. It is MIT-licensed, compatible with SwaraShruti's own
MIT license, and approximately 140MB — suitable for `models/` which is already
`.gitignore`'d. A shell script at `scripts/download_soundfont.sh` fetches it on first
run. The script is idempotent — re-running it after the file exists is a no-op.

## Consequences

**Positive**
- Repository stays lean — no binary assets in git history
- High-quality, realistic piano output from the first synthesis run
- Setup is reproducible and documented — one script, one command
- License is compatible — FluidR3_GM is MIT

**Negative / Watch points**
- Internet access required on first run after cloning
- `brew install fluid-synth` is a separate prerequisite that must be documented
  in the README — pyfluidsynth is a Python binding only and will fail silently
  if the FluidSynth C library is not present on the system
- Tests that depend on the soundfont must use `pytest.mark.skipif` to skip
  cleanly when `models/FluidR3_GM.sf2` is absent — CI should not fail on a
  missing binary
- If a higher-quality or different-licensed soundfont is needed in future, the
  download script and this ADR should both be updated
