# ADR-002: Pitch Detection Library

## Status
Accepted

## Date
2026-07-24

## Context
SwaraShruti's core pipeline converts hummed audio into MIDI. The first processing step after audio loading is pitch detection — identifying the fundamental frequency (in Hz) at each moment in time. The quality and reliability of every downstream stage (pitch correction, MIDI construction, note mapping) depends entirely on this step.

The CREPE algorithm (Convolutional Representation for Pitch Estimation) was identified early as the right algorithmic approach. It uses a 1D CNN with a 360-bin softmax output to model pitch detection as a classification problem rather than regression, which produces more robust results than traditional signal-processing approaches (e.g. YIN, pyin) particularly on monophonic vocal/hummed input.

The question this ADR resolves is: which package implements CREPE?

## Options Considered

### `crepe` (PyPI)
The original TensorFlow implementation by the algorithm's authors.

- Last published: 2019 (`v0.0.16`)
- Dependency: TensorFlow 1.x/2.x
- Status: **Unusable.** The package has broken build metadata — its `setup.py` references `pkg_resources` without declaring `setuptools` as a build dependency. This causes a hard build failure under PEP 517 isolated builds (the modern default for all packaging tools including uv, pip, and Poetry). No workaround exists that doesn't require either patching the package source or permanently disabling build isolation for the entire project. Bringing in a TensorFlow dependency for a single library would also significantly inflate the dependency footprint.

### `torchcrepe` (PyPI)
A PyTorch reimplementation of the same CREPE model by Max Morrison.

- Latest version: `0.0.24` (May 2025)
- Dependency: PyTorch
- Status: **Actively maintained.** 24 releases since 2020, 216K monthly downloads, 503 GitHub stars. Installs cleanly with uv and modern Python.
- Uses the same pre-trained model weights as the original (converted via MMdnn), so pitch estimation quality is equivalent.
- Returns timestamps, frequency in Hz per frame, and a periodicity score analogous to the original confidence score.
- Supports the same `model_capacity` options: `'tiny'` and `'full'`.

## Decision
Use **`torchcrepe`**.

The original `crepe` package is not installable under any modern Python toolchain without invasive workarounds. `torchcrepe` implements the identical algorithm with the same pre-trained weights, is actively maintained, and installs cleanly. PyTorch is a well-established dependency that other planned libraries (e.g. future model work) may also use, making it a reasonable addition to the project's dependency tree.

## Consequences

**Positive**
- Clean installation with `uv add torchcrepe`
- Actively maintained; bugs and Python compatibility issues get fixed
- PyTorch is a widely understood dependency
- Viterbi decoding available as a default (more stable than the original's weighted argmax — relevant for later pitch correction stages)

**Negative / Watch points**
- `torchcrepe.predict()` signature differs slightly from original `crepe.predict()`: it requires explicit `fmin`, `fmax`, and `hop_length` arguments rather than inferring them. SWARA-16 implementation must account for this.
- The periodicity score returned by `torchcrepe` is conceptually equivalent to CREPE's confidence score but is not numerically identical. Any future documentation or comments should use the term "periodicity" for accuracy.
- PyTorch adds a non-trivial install size (~700MB+). Acceptable for a local ML pipeline; would need consideration if the project later targets a lightweight server deployment.
