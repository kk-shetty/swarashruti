import pretty_midi

DEFAULT_VELOCITY = 100
DEFAULT_TEMPO = 120.0
PIANO_PROGRAM = 0


def build_midi(
    note_events: list[dict],
    tempo: float = DEFAULT_TEMPO,
    velocity: int = DEFAULT_VELOCITY,
) -> pretty_midi.PrettyMIDI:
    """
    Converts a plain pitch event dict to an actual MIDI Object.
    Default instrument is set to Piano (program number 0)
    Initial tempo is set to 120.0

    Parameters:
    note_events: A list of dicts, each containing:
        - midi_note (int): MIDI note number
        - start_time (float): Note onset in seconds
        - end_time (float): Note offset in seconds
    tempo (float): Initial tempo in BPM. Defaults to 120.0.
    velocity (int): MIDI velocity for all notes, range 0–127. Defaults to 100.

    Returns:
    A fully constructed pretty_midi object
    """

    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    piano = pretty_midi.Instrument(program=PIANO_PROGRAM)

    for event in note_events:
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=event["midi_note"],
            start=event["start_time"],
            end=event["end_time"],
        )
        piano.notes.append(note)

    pm.instruments.append(piano)

    return pm
