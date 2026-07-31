# Portamento slides in untrained vocal input typically fall in the 50-150ms range.
# Events shorter than this threshold are absorbed into their predecessor.
MIN_CONSOLIDATION_DURATION = 0.15

# Absorption should only happen when the short event is close to its predecessor
# i.e., genuine portamento.
# If the gap is large, the event is isolated noise in silence — drop it outright.
MAX_ABSORPTION_GAP = 0.20


def consolidate_notes(
    note_events: list[dict],
    min_duration: float = MIN_CONSOLIDATION_DURATION,
    max_absorption_gap: float = MAX_ABSORPTION_GAP,
) -> list[dict]:
    """
    Cleans up note event lists in two passes:

    Pass 1 — portamento absorption:
      Absorbs short events (< min_duration) into their predecessor to remove
      chromatic slides a human voice makes when transitioning between notes.
      If no predecessor exists, the short event is dropped.

    Parameters:
    - note_events (list[dict]): Output of segment_notes. Each dict contains
      'midi_note' (int), 'start_time' (float), 'end_time' (float).
    - min_duration (float): Threshold in seconds below which an event is
      considered a portamento artefact. Default 0.15s.
    - max_absorption_gap (float):  Threshold in seconds above which an event
      is considered silence and needs to be dropped

    Returns:
    A new list of note event dicts with artefacts absorbed and held notes merged.
    """
    # Pass 1 — portamento absorption
    after_portamento = []
    for event in note_events:
        duration = event["end_time"] - event["start_time"]
        if duration >= min_duration:
            after_portamento.append(dict(event))
        else:
            if after_portamento:
                gap = event["start_time"] - after_portamento[-1]["end_time"]
                if gap <= max_absorption_gap:
                    # Close enough — genuine portamento, extend predecessor
                    after_portamento[-1]["end_time"] = event["end_time"]
                # else: isolated noise in silence — drop it

    return after_portamento
