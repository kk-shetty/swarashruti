import numpy as np
import pytest
import scipy.io.wavfile as wavfile

from core.audio.audio_input import PIPELINE_SAMPLE_RATE


@pytest.fixture
def make_mono_wav_file(tmp_path):
    """
    Create a mono WAV file with configurable sample rate, duration, and frequency.
    Defaults to 16000 Hz, 1 second, and 440 Hz.
    """

    def _make_wav_file(
        sample_rate: int = PIPELINE_SAMPLE_RATE,
        duration: float = 1.0,
        frequency: float = 440.0,
    ) -> str:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        file_path = tmp_path / f"test_{sample_rate}hz.wav"
        wavfile.write(str(file_path), sample_rate, audio)
        return str(file_path)

    return _make_wav_file
