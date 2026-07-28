import numpy as np
import pytest

from core.midi.note_segmenter import segment_notes

midi_note_identical = np.array([69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69])
midi_note_two = np.array([0, 0, 69, 69, 69, 69, 69, 69, 71, 71, 71, 71, 71, 71, 71])
midi_note_silence_frame = np.array(
    [69, 69, 69, 69, 69, 0, 0, 0, 0, 0, 0, 71, 71, 71, 71, 71, 71]
)
midi_note_short_duration = np.array([69, 69, 69, 69])
midi_note_empty = np.array([])


def prepare_pitch_data(midi_note_array):
    time_array = np.arange(len(midi_note_array)) * 0.01
    frequency_array = np.ones(len(midi_note_array)) * 400
    periodicity_array = np.ones(len(midi_note_array))

    return {
        "midi_note": midi_note_array,
        "time": time_array,
        "frequency": frequency_array,
        "periodicity": periodicity_array,
    }


def test_segment_notes_identical():
    pitch_data = prepare_pitch_data(midi_note_identical)
    segments = segment_notes(pitch_data)
    assert len(segments) == 1
    assert segments[0]["midi_note"] == 69
    assert segments[0]["start_time"] == pytest.approx(0.0)
    assert segments[0]["end_time"] == pytest.approx(0.13)


def test_segment_notes_two():
    pitch_data = prepare_pitch_data(midi_note_two)
    segments = segment_notes(pitch_data)
    assert len(segments) == 2
    assert segments[0]["midi_note"] == 69
    assert segments[0]["start_time"] == pytest.approx(0.02)
    assert segments[0]["end_time"] == pytest.approx(0.08)
    assert segments[1]["midi_note"] == 71
    assert segments[1]["start_time"] == pytest.approx(0.08)
    assert segments[1]["end_time"] == pytest.approx(0.15)


def test_segment_notes_with_silence():
    pitch_data = prepare_pitch_data(midi_note_silence_frame)
    segments = segment_notes(pitch_data)
    assert len(segments) == 2
    assert segments[0]["midi_note"] == 69
    assert segments[0]["start_time"] == pytest.approx(0.0)
    assert segments[0]["end_time"] == pytest.approx(0.05)
    assert segments[1]["midi_note"] == 71
    assert segments[1]["start_time"] == pytest.approx(0.11)
    assert segments[1]["end_time"] == pytest.approx(0.17)


def test_segment_notes_short_duration():
    pitch_data = prepare_pitch_data(midi_note_short_duration)
    segments = segment_notes(pitch_data, min_duration=0.05)
    assert len(segments) == 0


def test_segment_notes_empty():
    pitch_data = prepare_pitch_data(midi_note_empty)
    segments = segment_notes(pitch_data)
    assert len(segments) == 0
