import numpy as np
import pytest

from core.pitch.pitch_smoother import SMOOTH_KERNEL_SIZE, smooth_midi_notes


def _make_pitch_data(midi_notes: list[int]) -> dict:
    n = len(midi_notes)
    return {
        "time": np.arange(n) * 0.01,
        "frequency": np.ones(n) * 220.0,
        "periodicity": np.ones(n),
        "midi_note": np.array(midi_notes, dtype=np.int32),
    }


def test_smooths_single_semitone_jitter() -> None:
    """Alternating adjacent semitones on a sustained note collapse to one note."""
    pitch_data = _make_pitch_data([50, 51, 50, 51, 50, 50, 51, 50, 51, 50])
    result = smooth_midi_notes(pitch_data)
    assert np.all(result["midi_note"] == 50)


def test_preserves_silent_frames() -> None:
    """Silent frames (midi_note=0) remain 0 after smoothing."""
    pitch_data = _make_pitch_data([0, 0, 50, 50, 50, 0, 0])
    result = smooth_midi_notes(pitch_data)
    assert np.all(result["midi_note"][[0, 1, 5, 6]] == 0)


def test_preserves_genuine_note_change() -> None:
    """A clear step between two stable notes is preserved."""
    pitch_data = _make_pitch_data([50, 50, 50, 50, 50, 57, 57, 57, 57, 57])
    result = smooth_midi_notes(pitch_data)
    assert np.all(result["midi_note"][:5] == 50)
    assert np.all(result["midi_note"][5:] == 57)


def test_all_silent_input_does_not_raise() -> None:
    """All-silent input (no voiced frames) is handled gracefully."""
    pitch_data = _make_pitch_data([0, 0, 0, 0, 0])
    result = smooth_midi_notes(pitch_data)
    assert np.all(result["midi_note"] == 0)


def test_even_kernel_size_raises() -> None:
    """Even kernel_size raises ValueError — scipy.signal.medfilt requires odd."""
    pitch_data = _make_pitch_data([50, 50, 50, 50, 50])
    with pytest.raises(ValueError, match="kernel_size must be odd"):
        smooth_midi_notes(pitch_data, kernel_size=4)


def test_returns_all_dict_keys() -> None:
    """Output dictionary contains all expected keys."""
    pitch_data = _make_pitch_data([50, 50, 50, 50, 50])
    result = smooth_midi_notes(pitch_data)
    assert set(result.keys()) == {"time", "frequency", "periodicity", "midi_note"}


def test_default_kernel_size() -> None:
    """Default kernel size constant is 5 (50ms at 10ms/frame)."""
    assert SMOOTH_KERNEL_SIZE == 5
