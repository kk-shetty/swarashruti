import numpy as np

from core.audio.audio_input import load_audio
from core.midi.midi_builder import build_midi
from core.midi.note_segmenter import segment_notes
from core.pitch.note_mapper import hz_to_midi
from core.pitch.pitch_detector import detect_pitch
from core.pitch.pitch_filter import filter_by_periodicity
from core.synthesis.synthesizer import SOUNDFONT_PATH, synthesize


def run_pipeline(audio_path: str, soundfont_path: str = SOUNDFONT_PATH) -> np.ndarray:
    """
    The integration function that runs the entire pipeline in below order:
    load_audio(audio_path)
        ↓ np.ndarray
    detect_pitch(audio, sr=PIPELINE_SAMPLE_RATE)
        ↓ dict {time, frequency, periodicity}
    filter_by_periodicity(pitch_data)
        ↓ dict {time, frequency, periodicity}
    hz_to_midi(pitch_data)
        ↓ dict {time, frequency, periodicity, midi_note}
    segment_notes(pitch_data)
        ↓ list[dict]
    build_midi(note_events)
        ↓ pretty_midi.PrettyMIDI
    synthesize(midi, soundfont_path)
        ↓ np.ndarray

    Parameters:
    - audio_path (str) - Path to the audio WAV file
    - soundfont_path (str) - path to local installation of required soundfont
    default is fetched from SOUNDFONT_PATH defined in core.synthesis.synthesizer
    This parameter allows for override

    Returns:
    - audio (np.ndarray): Synthesized audio at 44100 Hz
    note - this differs from PIPELINE_SAMPLE_RATE = 16000 used in earlier stages.
    """

    audio_time_series = load_audio(audio_path)
    pitch_data = detect_pitch(audio_time_series)
    filtered_pitch_data = filter_by_periodicity(pitch_data)
    mapped_notes = hz_to_midi(filtered_pitch_data)
    segmented_notes = segment_notes(mapped_notes)
    midi = build_midi(segmented_notes)
    audio = synthesize(midi=midi, soundfont_path=soundfont_path)

    return audio
