import numpy as np
import pretty_midi

SOUNDFONT_PATH = "models/FluidR3_GM.sf2"
OUTPUT_SAMPLE_RATE = 44100


def synthesize(
    midi: pretty_midi.PrettyMIDI,
    soundfont_path: str = SOUNDFONT_PATH,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
) -> np.ndarray:
    """
    Converts a given MIDI object into audio file which is numpy array.
    Note that OUTPUT_SAMPLE_RATE = 44100 differs from PIPELINE_SAMPLE_RATE = 16000
    This is the boundary where pipeline sample rate ends
    and playback sample rate begins.

    Parameters:
    - midi (pretty_midi.PrettyMIDI): MIDI object containing instruments and notes
    - soundfont_path (str) - path to local installation of required soundfont
    - sample_rate (int): Output audio sample rate in Hz. Defaults to 44100.

    Returns:
    audio (np.ndarray) - returns the synthesized audio file

    """

    audio = midi.fluidsynth(synthesizer=soundfont_path, fs=sample_rate)

    return audio
