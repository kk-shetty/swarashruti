import pytest

from core.midi.note_consolidator import (
    MIN_CONSOLIDATION_DURATION,
    consolidate_notes,
)


def _note(midi_note: int, start: float, end: float) -> dict:
    return {"midi_note": midi_note, "start_time": start, "end_time": end}


def test_short_note_with_predecessor_extends_predecessor() -> None:
    """Short note after a long note: predecessor's end_time absorbs it."""
    events = [_note(60, 0.0, 0.5), _note(61, 0.5, 0.6)]
    result = consolidate_notes(events)
    assert len(result) == 1
    assert result[0]["midi_note"] == 60
    assert result[0]["end_time"] == pytest.approx(0.6)


def test_short_note_with_no_predecessor_is_dropped() -> None:
    """Short note at the start of a phrase with no predecessor is dropped."""
    events = [_note(61, 0.0, 0.1)]
    result = consolidate_notes(events)
    assert result == []


def test_long_notes_pass_through_unchanged() -> None:
    """Events at or above min_duration are returned unchanged."""
    events = [_note(60, 0.0, 0.5), _note(55, 0.5, 1.2)]
    result = consolidate_notes(events)
    assert len(result) == 2
    assert result[0] == _note(60, 0.0, 0.5)
    assert result[1] == _note(55, 0.5, 1.2)


def test_empty_input_returns_empty() -> None:
    """Empty input returns empty list without error."""
    assert consolidate_notes([]) == []


def test_all_short_notes_no_predecessor_returns_empty() -> None:
    """All short notes with no long predecessor are dropped entirely."""
    events = [_note(60, 0.0, 0.1), _note(61, 0.1, 0.2), _note(62, 0.2, 0.3)]
    result = consolidate_notes(events)
    assert result == []


def test_returns_correct_dict_keys() -> None:
    """Output dicts contain exactly the expected keys."""
    events = [_note(60, 0.0, 0.5)]
    result = consolidate_notes(events)
    assert set(result[0].keys()) == {"midi_note", "start_time", "end_time"}


def test_default_min_duration_constant() -> None:
    """Default threshold is 0.15s — the identified portamento range ceiling."""
    assert MIN_CONSOLIDATION_DURATION == 0.15
