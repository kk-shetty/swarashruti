import numpy as np
from scipy.signal import medfilt

# 5 frames × 10ms/frame (hop_length=160 at 16kHz) = 50ms smoothing window.
# Eliminates single-semitone jitter while preserving genuine note transitions.
SMOOTH_KERNEL_SIZE = 5


def smooth_midi_notes(
    pitch_data: dict[str, np.ndarray], kernel_size: int = SMOOTH_KERNEL_SIZE
) -> dict[str, np.ndarray]:
    """
    Smooth MIDI note array using a median filter to reduce pitch jitter.

    Operates only on voiced frames (midi_note > 0) to prevent zero-padding
    artefacts at silence boundaries.

    Parameters:
    pitch_data (dict): Dictionary containing 'midi_note' and other pitch arrays.
    kernel_size (int): Median filter kernel size in frames. Must be odd.
                       Default 5 = 50ms window at 10ms/frame.

    Returns:
    A new dictionary with smoothed 'midi_note' array. All other keys unchanged.
    """
    if kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size must be odd, got {kernel_size}. "
            "scipy.signal.medfilt requires an odd kernel size."
        )

    midi_notes = pitch_data["midi_note"].copy()
    voiced_mask = midi_notes > 0

    if voiced_mask.any():
        voiced_notes = midi_notes[voiced_mask].astype(float)
        if len(voiced_notes) >= kernel_size:
            smoothed = np.round(medfilt(voiced_notes, kernel_size=kernel_size)).astype(
                np.int32
            )
            midi_notes[voiced_mask] = smoothed
        # Too few voiced frames to smooth meaningfully — return as-is

    return {
        "frequency": pitch_data["frequency"],
        "periodicity": pitch_data["periodicity"],
        "time": pitch_data["time"],
        "midi_note": midi_notes,
    }
