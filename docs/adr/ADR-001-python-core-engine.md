# ADR-001: Python as Core Engine Language

## Status
Accepted

## Context
SwaraShruti's core pipeline performs pitch detection on audio input, constructs MIDI from detected notes, and renders instrument audio via soundfonts. The primary dependency driving this decision is CREPE (Convolutional REpresentation for Pitch Estimation), a deep learning model for monophonic pitch detection. The choice of language for the core engine must support this ML-heavy pipeline while remaining practical for a solo developer building an early-stage product.

This is not a real-time audio processing system. The pipeline operates offline: audio in, MIDI out, rendered audio out. Latency constraints that would disqualify Python from live DSP work do not apply here.

## Decision
Python 3.12 is the core engine language, managed via `uv`. The full pipeline — pitch detection, MIDI construction, and soundfont rendering — is implemented in Python using CREPE, librosa, pretty_midi, and pyfluidsynth.

## Alternatives Considered

**Node.js**
Node.js has a mature ecosystem for web applications but lacks serious ML and audio analysis libraries. There is no native CREPE binding, no equivalent to librosa, and no production-grade soundfont renderer. Bridging to Python via child processes would add complexity without benefit. Rejected.

**Go**
Go offers excellent concurrency and performance for systems work but has a thin ecosystem for ML inference and audio DSP. Running CREPE would require calling out to Python anyway, making Go a wrapper around the real work rather than the engine itself. Rejected.

**Pure web-based approach (WebAudio API + ONNX Runtime Web) as the core engine**
CREPE has been ported to ONNX, making browser-based inference theoretically possible. However, this conflates the core engine with the delivery mechanism. The intended product direction is a web application — but the web layer will sit on top of the Python backend via an API boundary, not replace it. Constraining the pipeline to a browser sandbox would limit ML library access, complicate local model inference, and couple architectural decisions to frontend constraints before core functionality is proven. Deferred to the frontend layer in future phases.

**C++**
The industry standard for real-time audio plugins (JUCE, iPlug2). Would offer the best raw performance but has no CREPE binding and would require either reimplementing the model from scratch or maintaining a Python subprocess for inference. The development overhead is not justified for a non-real-time offline pipeline at this stage. Rejected.

## Consequences

**Easier:**
- Direct access to the full CREPE, librosa, pretty_midi, and pyfluidsynth ecosystem with no bridging layer
- Rapid iteration — Python's interactive tooling and notebook ecosystem accelerates experimentation during early pipeline development
- Accessible codebase for the ML community; most audio ML research is published with Python examples
- Aligns naturally with the future web app architecture: the browser submits audio via an API, the Python engine processes and returns rendered audio. No real-time DSP on the Python side is required.

**Harder:**
- Python's GIL and interpreter overhead make real-time audio processing impractical if the product ever needs live in-browser instrument playback with sub-10ms latency
- Custom DSP kernels in pure Python are inefficient; any performance-sensitive signal processing must rely on NumPy/SciPy's C backends or be rewritten
- Packaging and distribution is more complex than a compiled binary, though this is mitigated by deploying the engine as a web service

**Watch list:**
- PESTO (PyTorch-based) offers comparable pitch tracking accuracy to CREPE at significantly lower computational cost. If inference speed becomes a bottleneck, PESTO is the natural migration path without leaving the Python ecosystem.
- The API boundary between the Python engine and the future web frontend should be designed early — it is the architectural seam that keeps the core engine independent of the delivery layer.