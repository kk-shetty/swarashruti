import numpy as np

from core.pitch.note_mapper import hz_to_midi

# Crafting frequencies and expected MIDI notes
# from known reference values for testing
frequencies = np.array([0.0, 220.0, 261.63, 440.0, 523.25], dtype=np.float32)
expected_midi = np.array([0, 57, 60, 69, 72], dtype=np.int32)
periodicity = np.ones(len(frequencies), dtype=np.float32)
time = np.linspace(0, 0.1, len(frequencies), dtype=np.float32)

pitch_data = {
    "frequency": frequencies,
    "periodicity": periodicity,
    "time": time,
}


def test_hz_to_midi() -> None:
    """
    Test that hz_to_midi correctly converts frequencies to MIDI note numbers.
    """

    result = hz_to_midi(pitch_data)

    assert isinstance(result, dict)
    assert (
        "frequency" in result
        and "periodicity" in result
        and "time" in result
        and "midi_note" in result
    )

    # Check that the MIDI note values are as expected
    assert np.array_equal(result["midi_note"], expected_midi)
    assert result["midi_note"].dtype == np.int32
    np.testing.assert_array_equal(result["frequency"], pitch_data["frequency"])
    np.testing.assert_array_equal(result["periodicity"], pitch_data["periodicity"])
    np.testing.assert_array_equal(result["time"], pitch_data["time"])
