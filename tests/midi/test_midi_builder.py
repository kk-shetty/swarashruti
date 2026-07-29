import pretty_midi

from core.midi.midi_builder import build_midi

full_note_event = [
    {"midi_note": 69, "start_time": 0.02, "end_time": 0.08},
    {"midi_note": 71, "start_time": 0.08, "end_time": 0.13},
    {"midi_note": 80, "start_time": 0.16, "end_time": 0.23},
    {"midi_note": 69, "start_time": 0.30, "end_time": 0.36},
]

empty_note_event = []


def test_build_midi_full():
    midi = build_midi(full_note_event)
    assert isinstance(midi, pretty_midi.PrettyMIDI)
    instruments = midi.instruments
    assert len(instruments) == 1
    notes = midi.instruments[0].notes
    assert len(notes) == len(full_note_event)
    for note, event in zip(notes, full_note_event):
        assert note.pitch == event["midi_note"]
        assert note.start == event["start_time"]
        assert note.end == event["end_time"]


def test_build_midi_empty():
    midi = build_midi(empty_note_event)
    assert isinstance(midi, pretty_midi.PrettyMIDI)
    instruments = midi.instruments
    assert len(instruments) == 1
    notes = midi.instruments[0].notes
    assert len(notes) == 0
