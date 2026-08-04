import numpy as np

from core.pitch.pitch_detector import HOP_LENGTH

# RMS amplitude below this threshold is treated as silence.
# librosa normalises audio to [-1.0, 1.0]; genuine silence typically
# falls below 0.001–0.002, while soft vocal content sits above 0.003.
# Set conservatively to avoid silencing quiet recordings.
# Note: a fixed threshold is sensitive to recording level — adaptive
# thresholding (fraction of peak/mean energy) is a future improvement.
SILENCE_RMS_THRESHOLD = 0.002


def apply_silence_gate(
    pitch_data: dict[str, np.ndarray],
    audio: np.ndarray,
    hop_length: int = HOP_LENGTH,
    threshold: float = SILENCE_RMS_THRESHOLD,
) -> dict[str, np.ndarray]:
    """
    Zero out pitch and periodicity for frames where the raw audio RMS
    energy falls below threshold — these frames are silence or breath,
    not voiced audio.

    torchcrepe assigns a pitch estimate to every frame regardless of
    whether audio is present. This gate uses energy as a prior: if the
    waveform has no energy, there is no pitch.

    Parameters:
    - pitch_data (dict): Output of detect_pitch.
      Contains 'frequency', 'periodicity', 'time'.
    - audio (np.ndarray): Raw audio time series at PIPELINE_SAMPLE_RATE.
    - hop_length (int): Frame hop in samples. Must match torchcrepe's
      hop_length. Default 160 (10ms at 16kHz).
    - threshold (float): RMS amplitude threshold. Frames below this are
      silenced. Default 0.01.

    Returns:
    A new pitch_data dict with frequency and periodicity zeroed for
    silent frames. 'time' is unchanged.
    """
    n_frames = len(pitch_data["frequency"])

    # Compute per-frame RMS energy from raw audio
    # Compute per-frame RMS energy from raw audio.
    # Guard against torchcrepe returning more frames than audio has hops —
    # it pads internally, so the final frame(s) may exceed the audio length.
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        frame = audio[i * hop_length : (i + 1) * hop_length]
        if len(frame) > 0:
            rms[i] = np.sqrt(np.mean(frame**2))
        # Empty frame → rms stays 0.0 → treated as silence, which is correct

    silence_mask = rms < threshold

    frequency = pitch_data["frequency"].copy()
    periodicity = pitch_data["periodicity"].copy()

    frequency[silence_mask] = 0.0
    periodicity[silence_mask] = 0.0

    return {
        "frequency": frequency,
        "periodicity": periodicity,
        "time": pitch_data["time"],
    }
