import numpy as np

from core.audio.silence_gate import SILENCE_RMS_THRESHOLD, apply_silence_gate

HOP = 160
SR = 16000


def _make_pitch_data(n_frames: int) -> dict:
    return {
        "frequency": np.full(n_frames, 220.0),
        "periodicity": np.full(n_frames, 0.9),
        "time": np.arange(n_frames) * HOP / SR,
    }


def _sine_audio(n_frames: int, amplitude: float = 0.1) -> np.ndarray:
    """Voiced audio — sine wave at given amplitude."""
    samples = n_frames * HOP
    t = np.linspace(0, samples / SR, samples)
    return (np.sin(2 * np.pi * 440 * t) * amplitude).astype(np.float32)


def _silent_audio(n_frames: int) -> np.ndarray:
    """Silent audio — all zeros."""
    return np.zeros(n_frames * HOP, dtype=np.float32)


def test_silent_frames_zeroed() -> None:
    """Frames with zero energy get frequency and periodicity set to 0."""
    pitch_data = _make_pitch_data(10)
    audio = _silent_audio(10)
    result = apply_silence_gate(pitch_data, audio)
    assert np.all(result["frequency"] == 0.0)
    assert np.all(result["periodicity"] == 0.0)


def test_voiced_frames_pass_through() -> None:
    """Frames with sufficient RMS energy pass through unchanged."""
    pitch_data = _make_pitch_data(10)
    audio = _sine_audio(10, amplitude=0.1)
    result = apply_silence_gate(pitch_data, audio)
    assert np.all(result["frequency"] == 220.0)
    assert np.all(result["periodicity"] == 0.9)


def test_mixed_audio_partial_gate() -> None:
    """Silent frames zeroed, voiced frames preserved in same array."""
    n = 10
    pitch_data = _make_pitch_data(n)
    audio = np.concatenate([_silent_audio(5), _sine_audio(5, amplitude=0.1)])
    result = apply_silence_gate(pitch_data, audio)
    assert np.all(result["frequency"][:5] == 0.0)
    assert np.all(result["frequency"][5:] == 220.0)


def test_time_array_unchanged() -> None:
    """Time array is never modified."""
    pitch_data = _make_pitch_data(10)
    audio = _silent_audio(10)
    result = apply_silence_gate(pitch_data, audio)
    np.testing.assert_array_equal(result["time"], pitch_data["time"])


def test_returns_correct_dict_keys() -> None:
    """Output dict contains exactly the expected keys."""
    pitch_data = _make_pitch_data(5)
    audio = _sine_audio(5)
    result = apply_silence_gate(pitch_data, audio)
    assert set(result.keys()) == {"frequency", "periodicity", "time"}


def test_does_not_mutate_input() -> None:
    """Original pitch_data is not modified."""
    pitch_data = _make_pitch_data(5)
    original_freq = pitch_data["frequency"].copy()
    audio = _silent_audio(5)
    apply_silence_gate(pitch_data, audio)
    np.testing.assert_array_equal(pitch_data["frequency"], original_freq)


def test_default_threshold_constant() -> None:
    """Default RMS threshold is 0.002 — conservative to handle quiet recordings."""
    assert SILENCE_RMS_THRESHOLD == 0.002


def test_short_audio_no_warning(recwarn) -> None:
    """Audio shorter than n_frames * hop_length does not raise RuntimeWarning."""
    n_frames = 5
    pitch_data = _make_pitch_data(n_frames)
    # Audio shorter than 5 full hops — simulates torchcrepe padding behaviour
    audio = _sine_audio(3, amplitude=0.1)
    result = apply_silence_gate(pitch_data, audio)
    assert not any(issubclass(w.category, RuntimeWarning) for w in recwarn.list)
    assert len(result["frequency"]) == n_frames
