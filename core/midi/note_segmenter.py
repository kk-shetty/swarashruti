import numpy as np

# Reasonable minimum duration for a note for hummed vocal input.
# This is to avoid segmenting very short notes that may be noise or artifacts.
MIN_DURATION = 0.05


def segment_notes(
    pitch_data: dict[str, np.ndarray], min_duration: float = MIN_DURATION
) -> list[dict]:
    """
    Converts a frame-by-frame MIDI note array into a list of discrete note events,
    each with a start time, end time, and MIDI note number.
    Silence frames are skipped. Events shorter than min_duration are dropped.

    Parameters:
    - pitch_data (dict): A dictionary containing 'frequency', 'periodicity',
    'time', and 'midi_note' as keys, each mapping to a numpy array of values.
    - min_duration (float): The minimum duration for a note segment in seconds.

    Returns:
    A list of dicts, each containing:
    - midi_note (int): MIDI note number
    - start_time (float): Note onset in seconds
    - end_time (float): Note offset in seconds
    """

    midi_notes = pitch_data["midi_note"]
    times = pitch_data["time"]

    if len(midi_notes) == 0:
        return []

    boundaries = np.where(np.diff(midi_notes) != 0)[0] + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(midi_notes)]])

    note_segments = []
    frame_duration = times[1] - times[0]  # Assuming uniform sampling

    for start, end in zip(starts, ends):
        midi_note = int(midi_notes[start])

        if midi_note == 0:
            continue

        start_time = times[start]
        end_time = times[end - 1] + frame_duration
        duration = end_time - start_time

        if duration < min_duration:
            continue

        note_segments.append(
            {
                "midi_note": midi_note,
                "start_time": float(start_time),
                "end_time": float(end_time),
            }
        )

    return note_segments
