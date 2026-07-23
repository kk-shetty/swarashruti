# Understanding CREPE: A Convolutional Representation for Pitch Estimation

**A simplified reference guide for the SwaraShruti project.**

Based on the original paper by Jong Wook Kim, Justin Salamon, Peter Li, and Juan Pablo Bello — ICASSP 2018 · [arXiv:1802.06182](https://arxiv.org/abs/1802.06182)

**Purpose:** This document explains the CREPE pitch detection algorithm in accessible terms, covering the foundational concepts of digital audio, convolutional neural networks, and the complete architecture and experimental results from the original paper. It serves as a reference for the SwaraShruti audio-to-instrument transcription pipeline, where CREPE is the core pitch detection engine.

---

## Table of contents

1. [Digital audio fundamentals](#1-digital-audio-fundamentals)
2. [Neural network foundations](#2-neural-network-foundations)
3. [The CREPE architecture](#3-the-crepe-architecture)
4. [Training: how CREPE learns](#4-training-how-crepe-learns)
5. [Experiments and results](#5-experiments-and-results)
6. [Discussion and implications for SwaraShruti](#6-discussion-and-implications-for-swarashruti)
7. [References](#references)

---

## 1. Digital audio fundamentals

### 1.1 What is sound?

Sound is a vibration that travels through air (or another medium) as a wave of pressure changes. When you hum a note, your vocal cords vibrate, pushing air molecules back and forth. These pressure fluctuations reach your ear, where they're interpreted as sound. A microphone does the same job electronically — it converts those pressure changes into an electrical signal that varies in voltage over time.

### 1.2 What is a WAV file?

A WAV file is one of the simplest audio file formats. It contains two parts: a **header** (metadata like sample rate, bit depth, and number of channels) and **raw sample data** (a long sequence of numbers). That's it. There's no compression, no encoding tricks — just numbers representing air pressure at each instant in time.

Each number in the sequence is called a **sample**. A sample is a single measurement of the audio signal's amplitude (pressure level) at one specific moment. In a normalised representation, samples are floating-point numbers between -1.0 and +1.0, where 0.0 is silence, +1.0 is maximum positive pressure, and -1.0 is maximum negative pressure.

A 3-second recording at 16,000 samples per second contains exactly 48,000 of these numbers. That is the entire content of the audio — nothing more.

#### Example: first 10 samples of a 440 Hz sine wave at 16 kHz

| Index | Value      | Time      |
|-------|------------|-----------|
| [0]   | +0.000000  | 0.0000 ms |
| [1]   | +0.171929  | 0.0625 ms |
| [2]   | +0.336890  | 0.1250 ms |
| [3]   | +0.488063  | 0.1875 ms |
| [4]   | +0.619398  | 0.2500 ms |
| [5]   | +0.726286  | 0.3125 ms |
| [6]   | +0.805079  | 0.3750 ms |
| [7]   | +0.853553  | 0.4375 ms |
| [8]   | +0.871168  | 0.5000 ms |
| [9]   | +0.858316  | 0.5625 ms |

Each row is one sample — one measurement of air pressure at one instant. The values trace out a sine wave: starting at 0, rising to a peak, falling through zero to a trough, and repeating.

### 1.3 Sample rate

The **sample rate** is how many measurements per second the system takes. Common sample rates include 44,100 Hz (CD quality), 48,000 Hz (professional recording), and 16,000 Hz (speech and CREPE's requirement). Higher sample rates capture more detail but produce larger files.

The **Nyquist theorem** states that to accurately capture a frequency *f*, you need at least *2f* samples per second. At 16 kHz, the highest frequency that can be faithfully represented is 8 kHz — more than sufficient for the pitch range CREPE covers (32–1975 Hz).

> **Why CREPE requires 16 kHz:** CREPE's input is exactly 1024 samples. At 16 kHz, that equals 64 milliseconds of audio. The convolutional filters were trained to recognise pitch patterns at this specific time scale. If you feed 44.1 kHz audio without resampling, 1024 samples would only cover ~23 ms — the pitch relationships the model learned would be wrong.

### 1.4 Mono vs stereo

**Mono** audio has one channel — a single sequence of samples. **Stereo** has two channels (left and right), which means two interleaved sequences of samples. CREPE expects mono input. When loading stereo audio, the two channels are **downmixed** by averaging them: `mono_sample = (left + right) / 2`. This is what librosa's `load()` function does by default with the `mono=True` parameter.

### 1.5 Time domain vs frequency domain

Audio can be represented in two fundamental ways. The **time domain** is the raw waveform — sample values plotted over time. This is what you see when you open a WAV file. The **frequency domain** (a spectrogram) shows how much energy is present at each frequency at each moment, produced by applying a mathematical transformation like the FFT (Fast Fourier Transform).

Most audio classification systems convert audio to spectrograms first. CREPE is notable for operating directly on the time-domain waveform, which preserves phase information that spectrograms discard — and phase matters for precise pitch detection.

---

## 2. Neural network foundations

### 2.1 What is a convolutional neural network (CNN)?

A convolutional neural network is a type of deep learning model that excels at finding patterns in structured data. The key operation is **convolution**: a small pattern detector called a **filter** (or kernel) slides across the input, computing how well each region matches the pattern. Each filter produces one output that highlights where its particular pattern appears.

A single convolutional layer typically has many filters (32, 64, 128, or more), each learning to detect a different pattern. Early layers detect simple, low-level features. Later layers combine those features to detect increasingly complex, high-level patterns. This progression from simple to complex is called a **feature hierarchy**.

### 2.2 The 2D CNN approach (standard audio classification)

The most common way to use CNNs with audio is to first convert the audio into a **mel-spectrogram** — a 2D image where the x-axis is time, the y-axis is frequency, and colour intensity represents magnitude. A 2D CNN then slides its filters across both time and frequency dimensions, just like it would over a photograph. This works well for tasks like speech recognition, genre detection, and environmental sound classification.

### 2.3 CREPE's 1D CNN approach

CREPE uses a **1D CNN** that operates directly on the raw waveform. Its filters slide along the time axis only, over the raw sample values. This is architecturally significant because the spectrogram transformation used by 2D CNNs is a lossy process — it discards phase information. For most audio tasks this doesn't matter, but pitch detection is exquisitely sensitive to the fine structure of the waveform.

A 440 Hz note and a 441 Hz note look almost identical on a mel-spectrogram, but their raw waveforms drift in and out of phase in a mathematically precise way. CREPE's 1D CNN can learn to detect that drift directly from the samples.

| Aspect       | Standard audio CNN (2D)                       | CREPE (1D)                                        |
|--------------|-----------------------------------------------|---------------------------------------------------|
| Input        | 2D mel-spectrogram (time × frequency)         | 1D raw waveform (samples)                         |
| Convolution  | Slides over time AND frequency                | Slides over time only                             |
| Motivation   | Works well for classification tasks           | Preserves phase info critical for pitch           |
| Tradeoff     | Loses phase, gains frequency intuition        | Keeps all waveform detail, learns from scratch    |

### 2.4 Why classification, not regression?

CREPE frames pitch detection as a **classification** problem (which of 360 bins is the pitch in?) rather than a **regression** problem (what is the exact Hz value?). This is a key design choice. With regression, the model outputs a single number and is penalised for how far off it is — but it can't express *uncertainty*. With classification over 360 bins, the model can output a spread of confidence scores, saying "I'm fairly sure it's around bin 180, but it might be 179 or 181." A weighted average then recovers a precise frequency from that distribution.

### 2.5 Key CNN concepts used in CREPE

- **Filter / Kernel:** A small pattern detector that slides across the input. Each filter learns to recognise a specific pattern during training.
- **Stride:** How many positions the filter jumps between each application. Stride 4 means the filter moves 4 samples at a time, reducing the output length.
- **Max pooling:** Takes the maximum value from a small window (e.g. 2 samples), halving the output length. This provides translation invariance — the ability to detect a pattern regardless of exactly where it appears.
- **Batch normalisation:** Re-centres and re-scales values between layers, stabilising training. Think of it as recalibrating the signal at each processing stage.
- **Dropout (0.25):** Randomly sets 25% of neurons to zero during each training step. This prevents overfitting by forcing the network not to rely on any single neuron.
- **Sigmoid activation:** Squashes each output independently to a value between 0 and 1. Unlike softmax (where outputs must sum to 1), sigmoid lets each bin vote independently.
- **Dense / fully connected layer:** Every neuron in one layer connects to every neuron in the next. Used as the final layer to map the 2048-dimensional latent vector to 360 output bins.

---

## 3. The CREPE architecture

### 3.1 Overview

CREPE takes 1024 samples of raw audio at 16 kHz (64 ms) and produces a pitch estimate in Hz. The architecture is a pipeline: six convolutional layers progressively compress the audio into a 2048-dimensional representation, which is then mapped through a dense layer to 360 output bins representing pitch classes.

> 64 ms of raw audio (1024 samples at 16 kHz) → 6 convolutional layers → 2048-dimensional latent vector → Dense + sigmoid (360 confidence scores) → Weighted average → pitch in Hz

### 3.2 Layer-by-layer breakdown

The following table shows the exact configuration of each layer as specified in the paper's Figure 1. Notice how the temporal dimension shrinks through pooling while the number of filters (channels) grows — the network trades time resolution for representational richness.

| Layer    | Filters | Kernel size | Stride | MaxPool | Output shape   |
|----------|---------|-------------|--------|---------|----------------|
| Input    | —       | —           | —      | —       | 1024 × 1       |
| Conv 1   | 1024    | 512         | 4      | 2       | 128 × 1024     |
| Conv 2   | 128     | 64          | 1      | 2       | 64 × 128       |
| Conv 3   | 128     | 64          | 1      | 2       | 32 × 128       |
| Conv 4   | 256     | 64          | 1      | 2       | 16 × 256       |
| Conv 5   | 256     | 64          | 1      | 2       | 8 × 256        |
| Conv 6   | 512     | 64          | 1      | 2       | 4 × 512        |
| Reshape  | —       | —           | —      | —       | 2048           |
| Dense    | 360     | —           | —      | —       | 360            |

**Layer 1 is special.** Its kernel size of 512 covers 32 ms of audio at 16 kHz — enough to see one full cycle of any pitch down to about 31 Hz (the lowest in CREPE's range). Stride 4 reduces computation by jumping 4 samples between filter applications.

**Where 2048 comes from:** After Layer 6, you have 4 remaining timesteps × 512 filters = 2,048 values. These are flattened into a single vector. The 2048 numbers no longer represent "what happened at each moment" — they represent "what kind of pitch content is present" in this 64 ms window.

Each convolutional layer is preceded by batch normalisation and followed by dropout with probability 0.25. The total model has approximately 22.2 million parameters.

### 3.3 The 360 output bins and the cents scale

Musical pitch is **logarithmic**, not linear. Going from 100 Hz to 200 Hz (one octave) is the same perceived jump as 200 Hz to 400 Hz (also one octave). If CREPE spaced its bins linearly in Hz, it would waste bins on high frequencies and have too few for lower ones.

Instead, CREPE uses the **cents scale**, a logarithmic pitch ruler where 100 cents equals one semitone (one piano key) and 1200 cents equals one octave.

**Equation 1:**

```
¢(f) = 1200 · log₂(f / f_ref)
```

Here `f_ref = 10 Hz` is an arbitrary reference point (the "zero" of the ruler, not a frequency CREPE detects). The 360 bins cover six octaves from C1 (32.70 Hz) to B7 (1975.5 Hz) in 20-cent steps.

Sanity check: 6 octaves × 1200 cents/octave = 7200 cents ÷ 20 cents/bin = **360 bins**. This covers the full range of singing voices and most instruments.

### 3.4 From bins to Hz: the weighted average

CREPE doesn't just pick the highest-confidence bin. It computes a **weighted average** of all 360 bin positions, weighted by their confidence scores:

**Equation 2:**

```
ĉ = Σ(ŷᵢ · ¢ᵢ) / Σ(ŷᵢ)       →       f̂ = f_ref · 2^(ĉ/1200)
```

For example, if the network outputs high confidence around 440 Hz:

| Bin | Confidence | Corresponding pitch |
|-----|------------|---------------------|
| 178 | 0.05       | ~437 Hz             |
| 179 | 0.40       | ~438.5 Hz           |
| 180 | 0.92       | ~440 Hz             |
| 181 | 0.35       | ~441.5 Hz           |
| 182 | 0.03       | ~443 Hz             |

The weighted average might give 440.3 Hz — a value between bins, which is more precise than any single bin alone. The second part of the equation converts from cents back to Hz.

---

## 4. Training: how CREPE learns

### 4.1 The Gaussian-blurred training target

The naive approach to training labels would be a **one-hot** vector: all zeros except a 1 at the bin matching the true pitch. But this creates a harsh learning signal — predicting bin 179 (off by just 20 cents, barely audible) gets the same penalty as predicting bin 50 (wildly wrong). The loss function can't distinguish near-misses from total misses.

CREPE's solution is to **Gaussian-blur** the target, creating a smooth bell curve centred on the true pitch with a standard deviation of 25 cents (about a quarter of a semitone):

**Equation 3:**

```
yᵢ = exp( −(¢ᵢ − ¢_true)² / (2 · 25²) )
```

This produces a target where the true bin has value 1.0, neighbouring bins have high values, and distant bins trail off to near zero:

| Offset from true pitch | Target value | Visual                |
|------------------------|--------------|-----------------------|
| −60 cents              | 0.0540       | █                     |
| −40 cents              | 0.2780       | █████                 |
| −20 cents              | 0.8521       | █████████████████     |
|   0 cents              | 1.0000       | ████████████████████  |
| +20 cents              | 0.8521       | █████████████████     |
| +40 cents              | 0.2780       | █████                 |
| +60 cents              | 0.0540       | █                     |

Near-correct predictions receive partial credit during training, helping the model learn smoothly.

### 4.2 Binary cross entropy loss

The **loss function** is a single number that tells the network "how wrong were you?" Every adjustment to every filter during training aims to make this number smaller.

**Equation 4:**

```
L(y, ŷ) = Σᵢ₌₁₋₃₆₀ ( −yᵢ log ŷᵢ − (1 − yᵢ) log(1 − ŷᵢ) )
```

Since CREPE uses sigmoid (not softmax), each of the 360 bins independently outputs a value between 0 and 1 — each bin is essentially an independent yes/no question: "Is the pitch near this frequency?" Binary cross entropy is the mathematically ideal loss function for such independent predictions. For each bin, if the target was 1.0 and the model predicted 0.01, the penalty is huge. If the target was 1.0 and the model predicted 0.95, the penalty is tiny.

### 4.3 Training procedure

| Parameter            | Value / description                                    |
|----------------------|--------------------------------------------------------|
| Optimiser            | ADAM with learning rate 0.0002                         |
| Batch size           | 32 examples per batch                                  |
| Epoch                | 500 batches = 16,000 training frames                   |
| Early stopping       | Stop if no improvement for 32 consecutive epochs       |
| Batch normalisation  | Applied before each convolutional layer                |
| Dropout              | 0.25 probability after each convolutional layer        |
| Framework            | Keras with TensorFlow backend                          |
| Total parameters     | ~22.2 million                                          |

**Early stopping** prevents overfitting — the model memorising training data instead of learning general patterns. The model that performs best on validation data (not training data) is the one that gets saved.

---

## 5. Experiments and results

### 5.1 Datasets

To evaluate CREPE, the authors needed datasets with **perfect ground truth** — exact pitch labels for every frame. Even human annotations contain errors, so they used synthesised audio where the pitch is controlled by the synthesis process itself.

| Dataset         | Hours | Content                                               | Role                      |
|-----------------|-------|-------------------------------------------------------|---------------------------|
| RWC-synth       | 6.16  | Simple sinusoidal synthesis, homogeneous timbre       | Easy test (baseline)      |
| MDB-stem-synth  | 15.56 | 230 tracks, 25 instruments, realistic timbre          | Hard test (generalisation)|

### 5.2 Methodology

The model was evaluated using **5-fold cross-validation** with a 60/20/20 split (train / validate / test). For MDB-stem-synth, **artist-conditional folds** ensured that tracks from the same artist never appeared in both training and test sets, preventing the model from learning to recognise specific voices rather than general pitch patterns.

**Metrics:**

- **Raw Pitch Accuracy (RPA):** Proportion of frames within 50 cents of the true pitch.
- **Raw Chroma Accuracy (RCA):** Same, but ignoring octave errors (C4 vs C5 would be wrong for RPA but correct for RCA).

### 5.3 Pitch accuracy results

At the standard 50-cent threshold:

| Dataset         | Metric | CREPE           | pYIN            | SWIPE           |
|-----------------|--------|-----------------|-----------------|-----------------|
| RWC-synth       | RPA    | 0.999 ± 0.002   | 0.990 ± 0.006   | 0.963 ± 0.023   |
| RWC-synth       | RCA    | 0.999 ± 0.002   | 0.990 ± 0.006   | 0.966 ± 0.020   |
| MDB-stem-synth  | RPA    | 0.967 ± 0.091   | 0.919 ± 0.129   | 0.925 ± 0.116   |
| MDB-stem-synth  | RCA    | 0.970 ± 0.084   | 0.936 ± 0.092   | 0.936 ± 0.100   |

At tighter thresholds, the gap widens dramatically:

| Dataset         | Threshold | CREPE           | pYIN            | SWIPE           |
|-----------------|-----------|-----------------|-----------------|-----------------|
| RWC-synth       | 50 cents  | 0.999 ± 0.002   | 0.990 ± 0.006   | 0.963 ± 0.023   |
| RWC-synth       | 25 cents  | 0.999 ± 0.003   | 0.972 ± 0.012   | 0.949 ± 0.026   |
| RWC-synth       | 10 cents  | 0.995 ± 0.004   | 0.908 ± 0.032   | 0.833 ± 0.055   |
| MDB-stem-synth  | 50 cents  | 0.967 ± 0.091   | 0.919 ± 0.129   | 0.925 ± 0.116   |
| MDB-stem-synth  | 25 cents  | 0.953 ± 0.103   | 0.890 ± 0.134   | 0.897 ± 0.127   |
| MDB-stem-synth  | 10 cents  | 0.909 ± 0.126   | 0.826 ± 0.150   | 0.816 ± 0.165   |

> **Key finding:** At 10 cents on MDB-stem-synth, CREPE achieves 90.9% accuracy versus pYIN's 82.6% — an 8+ percentage point gap. CREPE doesn't just get the right note; it gets very close to the exact frequency. This precision matters for SwaraShruti's pitch correction stage.

### 5.4 Noise robustness

The authors tested with four noise types (pub, white, pink, brown) at seven SNR levels (from clean to extremely noisy). Key findings:

- **Pub noise and white noise:** CREPE maintains highest accuracy at all noise levels.
- **Pink noise:** CREPE wins at all levels except the cleanest, where performance is nearly equal.
- **Brown noise:** The one exception — pYIN outperforms here because brown noise concentrates in low frequencies, where YIN's autocorrelation method is naturally robust.
- **Consistency:** CREPE shows lower variance (smaller error bars) across all conditions, meaning it is not just more accurate on average but more reliably accurate.

### 5.5 What the network actually learned (model analysis)

The authors analysed the first layer's 1024 convolutional filters by examining their frequency spectra — asking "what frequencies does each filter respond to?"

**On RWC-synth** (simple timbre): filters' peak frequencies are between 600–1500 Hz, even though actual pitches are 100–600 Hz. The network learned to detect **overtones** (harmonics) rather than the fundamental directly. Why? In simple synthesised audio, harmonics are a reliable pitch indicator — energy at 600, 900, and 1200 Hz implies a fundamental of 300 Hz.

**On MDB-stem-synth** (diverse timbres): filters span a wider range overlapping with the actual pitch distribution. With complex, varied timbres, harmonic structure differs across instruments, so the network needs filters that capture the fundamental frequency's periodicity directly, in addition to harmonics.

> The network discovers different analysis strategies depending on the data. No human told it to look at harmonics or fundamentals — it found the optimal approach through learning.

### 5.6 Performance by instrument

Performance varies across instruments. Notable observations:

- Performance tends to be **lower for higher-frequency instruments**.
- The **dizi** (Chinese transverse flute) performs poorly because all dizi tracks came from one artist and ended up in the same test fold — the model never saw dizi timbre during training. This is a generalisation failure.
- Instruments with timbres **similar to well-represented ones** (e.g. bass clarinet vs clarinet) perform well even with few training examples.
- The model performs well on **singing voices** (male and female singers are well-represented), which is directly relevant to SwaraShruti's humming input.

---

## 6. Discussion and implications for SwaraShruti

### 6.1 Limitations identified by the authors

**No temporal tracking:** CREPE estimates each 64 ms frame **independently**. It doesn't know what it predicted for the previous frame. In reality, pitch changes smoothly — a singer doesn't randomly jump between distant pitches. pYIN exploits temporal continuity using a Hidden Markov Model (HMM). The authors suggest adding a recurrent layer (LSTM/GRU) to create a CRNN that could learn temporal patterns.

**Invariance to non-pitch transformations:** Ideally, CREPE should give the same pitch regardless of distortion, reverb, or other effects that don't change the actual pitch. Pooling layers provide some translation invariance, but the model isn't specifically designed to ignore other transformations. The authors suggest data augmentation (training on distorted/reverberant versions) as a practical fix.

**Unseen timbres:** The dizi results show that CREPE struggles with timbres completely absent from training data. However, for timbres similar to those in the training set, generalisation is strong.

### 6.2 What this means for SwaraShruti

- **Input normalisation is critical (SWARA-15):** Audio must be resampled to 16 kHz mono before reaching CREPE. The `load_audio()` function's entire purpose is to guarantee this.
- **Confidence scores are useful:** CREPE's per-frame confidence output can be used to filter out unreliable frames (e.g. during silence, breath, or noise). Low-confidence frames should not be converted to MIDI notes.
- **Frame independence requires smoothing:** CREPE may occasionally produce "rogue" frames with wild pitch estimates. The pitch correction stage should handle this — either by applying temporal smoothing or by using confidence-weighted filtering.
- **Humming is well-covered:** Singing voices (male and female) are well-represented in CREPE's training data. The model should perform well for SwaraShruti's primary use case.
- **High precision at tight thresholds:** CREPE's 90.9% accuracy at 10 cents means the pitch estimates are precise enough for high-quality instrument synthesis without heavy correction.

### 6.3 DSP: digital signal processing

DSP (Digital Signal Processing) refers to the field of mathematically manipulating digital signals to extract information or transform them. In the audio context, DSP encompasses the hand-crafted mathematical techniques developed over decades: FFT (decomposing a signal into frequencies), filtering, autocorrelation (finding repeating patterns), and the algorithms CREPE was benchmarked against (pYIN, SWIPE). The word "hand-crafted" is key — a DSP engineer encodes their understanding of physics into explicit mathematical rules. CREPE's significance is that it outperforms these hand-crafted pipelines without being told anything about how pitch works, purely by learning from data.

---

## References

### Original paper

1. Kim, J.W., Salamon, J., Li, P., and Bello, J.P. (2018). "CREPE: A Convolutional Representation for Pitch Estimation." Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP). [arXiv:1802.06182](https://arxiv.org/abs/1802.06182).
2. CREPE GitHub repository: [github.com/marl/crepe](https://github.com/marl/crepe)

### Algorithms referenced

3. Mauch, M. and Dixon, S. (2014). "pYIN: A fundamental frequency estimator using probabilistic threshold distributions." ICASSP 2014.
4. Camacho, A. and Harris, J.G. (2008). "A sawtooth waveform inspired pitch estimator for speech and music." JASA, 124(3).
5. De Cheveigné, A. and Kawahara, H. (2002). "YIN, a fundamental frequency estimator for speech and music." JASA, 111(4).

### Architecture details

6. Riess, V. and Morrison, M. (2023). "Cross-domain Neural Pitch and Periodicity Estimation." [arXiv:2301.12258](https://arxiv.org/abs/2301.12258).

### Further learning

- CNN for Audio (YouTube playlist): [Valence Vibrations series](https://www.youtube.com/watch?v=fMqL5vckiU0&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf)
- Neural Networks (YouTube playlist): [3Blue1Brown series](https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0)
- [CNNs for Audio Feature Extraction](https://anudeepareddy-s.medium.com/convolutional-neural-networks-cnns-for-audio-data-1c41f4aac35d) (Medium)

---

*This document was prepared as part of the SwaraShruti project. All credit for the CREPE algorithm belongs to the original authors: Jong Wook Kim, Justin Salamon, Peter Li, and Juan Pablo Bello (Music and Audio Research Laboratory, NYU).*