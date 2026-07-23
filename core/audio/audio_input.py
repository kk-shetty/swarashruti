import os

import librosa
import numpy as np

# CREPE requires exactly 16 kHz — other rates produce wrong predictions
CREPE_SAMPLE_RATE = 16000
SUPPORTED_AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg"}


def load_audio(file_path: str) -> np.ndarray:
    """
    Load an audio file and return the audio time series.

    Parameters:
    file_path (str): Path to the audio file.

    Returns:
    y (np.ndarray): Audio time series.
    """

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    _, ext = os.path.splitext(file_path)
    if ext.lower() not in SUPPORTED_AUDIO_FORMATS:
        raise ValueError(
            f"Unsupported audio format: {ext}. "
            f"Supported formats are: {SUPPORTED_AUDIO_FORMATS}"
        )

    try:
        y, _ = librosa.load(file_path, sr=CREPE_SAMPLE_RATE, mono=True)
    except Exception as e:
        raise RuntimeError(f"Error loading audio file '{file_path}': {e}")

    if y.size == 0:
        raise ValueError(f"Audio file '{file_path}' is empty.")

    return y
