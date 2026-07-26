import numpy as np


def hz_to_midi(pitch_data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Convert frequency in Hz to MIDI note numbers.

    Parameters:
    pitch_data (dict): Dictionary containing filtered 'frequency',
    'periodicity', and 'time'

    Returns:
    A dictionary containing:
    Original dictionary with an additional key:
    1. midi_note (np.ndarray): MIDI note numbers corresponding to the frequencies.
    """

    # Avoid log2(0) by replacing non-positive frequencies with 1.0
    safe_frequency = np.where(pitch_data["frequency"] > 0, pitch_data["frequency"], 1.0)
    # exclusively setting midi_note to 0 for frequency <= 0
    # to avoid negative infinity values in the log2 calculation.
    midi_note = np.where(
        pitch_data["frequency"] > 0, 69 + 12 * np.log2(safe_frequency / 440.0), 0
    )
    # MIDI note numbers are integers by definition,
    # So rounding to the nearest integer is appropriate.
    return {
        "frequency": pitch_data["frequency"],
        "periodicity": pitch_data["periodicity"],
        "time": pitch_data["time"],
        "midi_note": np.round(midi_note).astype(np.int32),
    }
