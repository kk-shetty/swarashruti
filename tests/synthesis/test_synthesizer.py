import os

import numpy as np
import pytest

from core.midi.midi_builder import build_midi
from core.synthesis.synthesizer import SOUNDFONT_PATH, synthesize

# This skips all the tests if valid soundfont is not present.
pytestmark = pytest.mark.skipif(
    not os.path.exists(SOUNDFONT_PATH), reason="Soundfont doesn't exists"
)

full_note_event = [
    {"midi_note": 69, "start_time": 0.0, "end_time": 0.5},
    {"midi_note": 71, "start_time": 0.5, "end_time": 1.0},
]
empty_note_event = []


def test_synthesize_returns_ndarray():
    midi = build_midi(full_note_event)
    audio = synthesize(midi=midi)
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0


def test_synthesize_output_length():
    midi = build_midi(full_note_event)
    audio = synthesize(midi=midi)
    expected = 44100  # 1 second at 44100 Hz
    assert expected <= len(audio) <= expected * 2


def test_synthesize_empty_midi():
    midi = build_midi(empty_note_event)
    audio = synthesize(midi=midi)
    assert isinstance(audio, np.ndarray)
