# ADR-004: Pitch Smoothing Strategy

**Status:** Accepted  
**Date:** 2026-07-30  
**Ticket:** SWARA-28

---

## Context

Real-world testing with a hummed recording of Twinkle Twinkle Little Star
revealed severe pitch jitter in the pipeline output. Debug analysis of the
intermediate stage arrays produced the following measurements:

- 2151 total frames from torchcrepe (21.5s recording at 10ms/frame)
- 1844 voiced frames after periodicity filtering
- **251 one-semitone jumps** out of 1843 consecutive frame transitions
- **13.6% of all frame transitions** are single-semitone jitter
- Dominant oscillation patterns: MIDI 50 ↔ 51 (D3 ↔ Eb3), MIDI 57 ↔ 58 (A3 ↔ Bb3)
- Many runs of 1–2 frames (10–20ms) — far below any musically meaningful duration

Root cause: torchcrepe estimates pitch every 10ms. On a sustained hum, the
raw Hz value drifts slightly around a semitone boundary. When rounded to the
nearest integer MIDI note, frames alternate between two adjacent semitones
rather than locking to one. The result in the rendered audio is a rapid
flickering between two piano keys.

Increasing `min_duration` in the note segmenter was tested first (0.05s → 0.1s)
and produced only marginal improvement. This confirmed the jitter must be
addressed upstream of segmentation, at the MIDI note array level.

---

## Options Considered

**Option A — Raise periodicity threshold**  
Filter more frames below a higher threshold (e.g. 0.5 instead of 0.21).  
Rejected: the oscillation occurs in high-periodicity frames — the detector
is confident but jittery. Raising the threshold would lose real pitched frames
without addressing the oscillation.

**Option B — Median filter on voiced MIDI frames** ✅ Chosen  
Apply `scipy.signal.medfilt` to the integer MIDI note array, operating only
on voiced frames (midi_note > 0). A kernel of 5 frames (50ms window) looks
at each frame's neighbourhood and picks the median — noise spikes collapse,
sustained notes stay stable.

For the dominant oscillation pattern:

```
Before: 50, 51, 50, 51, 50, 50, 50
After:  50, 50, 50, 50, 50, 50, 50
```

**Option C — Median filter on raw Hz frequencies before MIDI mapping**  
Smoothing in Hz space before quantization is theoretically more correct —
it prevents the quantization boundary from being crossed repeatedly.  
Rejected for now: the jitter originates at quantization, so post-quantization
smoothing is sufficient. Hz-space smoothing adds complexity with no measurable
benefit at this stage.

**Option D — Increase min_duration in note segmenter**  
Already tested. Marginal improvement — treats the symptom downstream rather
than the cause upstream.

---

## Decision

Option B: median filter on voiced MIDI frames, `kernel_size=5` (50ms window),
applied as a new pipeline stage between `hz_to_midi` and `segment_notes`.

Silent frames (`midi_note == 0`) are excluded from the filter window to prevent
bleeding silence into voiced regions or vice versa.

---

## Consequences

- New module: `core/pitch/pitch_smoother.py` — `smooth_midi_notes(pitch_data, kernel_size=5) -> dict`
- `core/pipeline.py` gains one step between `hz_to_midi` and `segment_notes`
- `scipy` moves from `[dev]` to main `[dependencies]` in `pyproject.toml` — it is now required at runtime, not just in tests
- `kernel_size` is exposed as a parameter to allow future tuning without code changes
- If Hz-space smoothing is needed in future (e.g. for microtonal accuracy), Option C remains available as a separate upstream stage

---

## References
- Debug arrays: `debug/stage4_midi_notes.npy` (not committed — in `.gitignore`)
- torchcrepe hop_length: 160 samples at 16kHz = 10ms per frame
- Kernel size 5 = 50ms window = 5 × 10ms frames