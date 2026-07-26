import numpy as np
import pytest
import scipy.io.wavfile as wavfile

from core.audio.audio_input import PIPELINE_SAMPLE_RATE, load_audio


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


@pytest.fixture
def make_stereo_wav_file(tmp_path):
    """
    Create a Stereo WAV file with configurable sample rate, duration, and frequencies.
    Defaults to 16000 Hz, 1 second, and 440 Hz left and 540 Hz right frequencies.
    """

    def _make_wav_file(
        sample_rate: int = PIPELINE_SAMPLE_RATE,
        duration: float = 1.0,
        left_frequency: float = 440.0,
        right_frequency: float = 540.0,
    ) -> str:
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio_left = np.sin(2 * np.pi * left_frequency * t).astype(np.float32)
        audio_right = np.sin(2 * np.pi * right_frequency * t).astype(np.float32)
        stereo_audio = np.column_stack((audio_left, audio_right))
        file_path = tmp_path / f"test_stereo_{sample_rate}hz.wav"
        wavfile.write(str(file_path), sample_rate, stereo_audio)
        return str(file_path)

    return _make_wav_file


def test_resamples_from_44100(make_mono_wav_file) -> None:
    """Test that a WAV file with a sample rate of 44100 Hz is resampled to 16000 Hz."""
    path = make_mono_wav_file(sample_rate=44100)
    y = load_audio(path)
    assert len(y) == PIPELINE_SAMPLE_RATE


def test_already_16000(make_mono_wav_file) -> None:
    """Test that a WAV file with a sample rate of 16000 Hz is loaded correctly."""
    path = make_mono_wav_file(sample_rate=16000)
    y = load_audio(path)
    assert len(y) == PIPELINE_SAMPLE_RATE
    assert isinstance(y, np.ndarray)


def test_stereo_to_mono(make_stereo_wav_file) -> None:
    """Test that a stereo WAV file is converted to mono correctly."""
    path = make_stereo_wav_file(sample_rate=16000)
    y = load_audio(path)
    assert len(y) == PIPELINE_SAMPLE_RATE
    assert y.ndim == 1  # mono = single dimension


def test_file_not_found() -> None:
    """Test that loading a non-existent file raises a FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_audio("non_existent_file.wav")


def test_unsupported_format(tmp_path) -> None:
    """Test that loading a file with an unsupported format raises a ValueError."""
    unsupported_file = tmp_path / "test.txt"
    unsupported_file.write_text("This is not an audio file.")
    with pytest.raises(ValueError):
        load_audio(str(unsupported_file))


def test_empty_audio_file(tmp_path) -> None:
    """Test that loading an empty audio file raises a ValueError."""
    empty_file = tmp_path / "empty.wav"
    wavfile.write(str(empty_file), PIPELINE_SAMPLE_RATE, np.array([], dtype=np.float32))
    with pytest.raises(ValueError):
        load_audio(str(empty_file))


def test_invalid_audio_file(tmp_path) -> None:
    """Test that loading a corrupted audio file raises a RuntimeError."""
    invalid_file = tmp_path / "invalid.wav"
    invalid_file.write_bytes(b"This is not a valid WAV file.")
    with pytest.raises(RuntimeError):
        load_audio(str(invalid_file))
