import numpy as np
import pytest

from core.audio.audio_input import PIPELINE_SAMPLE_RATE
from core.pitch.pitch_detector import FMAX, FMIN, HOP_LENGTH, detect_pitch


@pytest.fixture
def make_sine_wave():
    """
    Create a sine wave audio signal with configurable sample rate, duration,
    and frequency.
    Defaults to 16000 Hz, 1 second, and 440 Hz.
    """

    def _make_sine_wave(
        sample_rate: int = PIPELINE_SAMPLE_RATE,
        duration: float = 1.0,
        frequency: float = 440.0,
    ) -> np.ndarray:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        return audio

    return _make_sine_wave


def test_output_shapes(make_sine_wave) -> None:
    """
    Test that the output shapes of detect_pitch are as expected.
    """
    audio = make_sine_wave()
    result = detect_pitch(audio)
    assert isinstance(result, dict)
    assert "frequency" in result and "periodicity" in result and "time" in result
    assert isinstance(result["frequency"], np.ndarray)
    assert isinstance(result["periodicity"], np.ndarray)
    assert isinstance(result["time"], np.ndarray)
    assert len(result["frequency"]) == len(result["periodicity"]) == len(result["time"])


def test_frequency_range(make_sine_wave) -> None:
    """
    Test that the detected frequencies are within the expected range.
    """
    audio = make_sine_wave(frequency=440.0)
    result = detect_pitch(audio)
    assert np.all((result["frequency"] >= FMIN) & (result["frequency"] <= FMAX))


def test_periodicity_range(make_sine_wave) -> None:
    """
    Test that the detected periodicity values are within the range 0.0–1.0.
    """
    audio = make_sine_wave(frequency=440.0)
    result = detect_pitch(audio)
    assert np.all((result["periodicity"] >= 0.0) & (result["periodicity"] <= 1.0))


def test_time_values(make_sine_wave) -> None:
    """
    Test that the time values are correctly calculated
    based on the hop length and sample rate.
    """
    audio = make_sine_wave(frequency=440.0)
    result = detect_pitch(audio)
    expected_time = (
        np.arange(len(result["frequency"])) * HOP_LENGTH / PIPELINE_SAMPLE_RATE
    )
    assert np.allclose(result["time"], expected_time)
    assert result["time"][0] == 0.0
    assert np.all(np.diff(result["time"]) > 0)
