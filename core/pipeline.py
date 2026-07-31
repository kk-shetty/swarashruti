import os

import numpy as np

from core.audio.audio_input import PIPELINE_SAMPLE_RATE, load_audio
from core.midi.midi_builder import build_midi
from core.midi.note_consolidator import consolidate_notes
from core.midi.note_segmenter import segment_notes
from core.pitch.note_mapper import hz_to_midi
from core.pitch.pitch_detector import detect_pitch
from core.pitch.pitch_filter import filter_by_periodicity
from core.pitch.pitch_smoother import smooth_midi_notes
from core.synthesis.synthesizer import SOUNDFONT_PATH, synthesize


def run_pipeline(
    audio_path: str,
    soundfont_path: str = SOUNDFONT_PATH,
    debug: bool = False,
    debug_dir: str = "debug",
) -> np.ndarray:
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
    - soundfont_path (str) - path to local installation of required soundfont.
      Default is fetched from SOUNDFONT_PATH defined in core.synthesis.synthesizer.
      This parameter allows for override.
    - debug (bool) - If True, prints intermediate stage summaries and saves
      numpy arrays to debug_dir for inspection. Default False.
    - debug_dir (str) - Directory to save debug output files. Default "debug".

    Returns:
    - audio (np.ndarray): Synthesized audio at 44100 Hz.
      Note: this differs from PIPELINE_SAMPLE_RATE = 16000 used in earlier stages.
    """

    if debug:
        os.makedirs(debug_dir, exist_ok=True)
        print(f"\n=== Pipeline Debug: {audio_path} ===\n")

    # Stage 1 — load audio
    audio_time_series = load_audio(audio_path)
    if debug:
        duration = len(audio_time_series) / PIPELINE_SAMPLE_RATE
        print(
            f"[1] load_audio      : {len(audio_time_series)} samples, "
            f"{duration:.2f}s @ {PIPELINE_SAMPLE_RATE}Hz"
        )
        np.save(f"{debug_dir}/stage1_audio.npy", audio_time_series)

    # Stage 2 — detect pitch
    pitch_data = detect_pitch(audio_time_series)
    if debug:
        freqs = pitch_data["frequency"]
        voiced = freqs[freqs > 0]
        print(
            f"[2] detect_pitch    : {len(freqs)} frames total, "
            f"{len(voiced)} voiced ({100 * len(voiced) / len(freqs):.1f}%), "
            f"freq range {voiced.min():.1f}–{voiced.max():.1f} Hz"
        )
        np.save(f"{debug_dir}/stage2_frequency.npy", freqs)
        np.save(f"{debug_dir}/stage2_periodicity.npy", pitch_data["periodicity"])

    # Stage 3 — filter by periodicity
    filtered_pitch_data = filter_by_periodicity(pitch_data)
    if debug:
        freqs_filtered = filtered_pitch_data["frequency"]
        voiced_filtered = freqs_filtered[freqs_filtered > 0]
        dropped = len(voiced) - len(voiced_filtered)
        print(
            f"[3] filter          : {len(voiced_filtered)} voiced frames remain, "
            f"{dropped} dropped by periodicity threshold"
        )
        np.save(f"{debug_dir}/stage3_frequency.npy", freqs_filtered)

    # Stage 4 — Hz to MIDI
    mapped_notes = hz_to_midi(filtered_pitch_data)
    if debug:
        midi_notes = mapped_notes["midi_note"]
        voiced_notes = midi_notes[midi_notes > 0]
        unique_notes = sorted(np.unique(voiced_notes).tolist())
        print(
            f"[4] hz_to_midi      : {len(unique_notes)} "
            f"unique MIDI notes — {unique_notes}"
        )
        np.save(f"{debug_dir}/stage4_midi_notes.npy", midi_notes)

    # Stage 5 — smooth MIDI notes
    smoothed_notes = smooth_midi_notes(mapped_notes)
    if debug:
        smoothed_midi = smoothed_notes["midi_note"]
        voiced_smoothed = smoothed_midi[smoothed_midi > 0]
        unique_smoothed = sorted(np.unique(voiced_smoothed).tolist())
        print(
            f"[5] smooth_midi     : {len(unique_smoothed)} unique MIDI notes "
            f"after smoothing — {unique_smoothed}"
        )
        np.save(f"{debug_dir}/stage5_midi_smoothed.npy", smoothed_midi)

    # Stage 6 — segment note events
    segmented_notes = segment_notes(smoothed_notes)
    if debug:
        print(f"[5] segment_notes   : {len(segmented_notes)} note events")
        np.save(f"{debug_dir}/stage6_midi_segmented.npy", segmented_notes)
        # for i, note in enumerate(segmented_notes):
        #     dur = note["end_time"] - note["start_time"]
        #     print(
        #         f"    [{i:03d}] MIDI {note['midi_note']:3d}  "
        #         f"{note['start_time']:.3f}s → {note['end_time']:.3f}s  ({dur:.3f}s)"
        #     )

    # Stage 7 - consolidate note events
    consolidated_notes = consolidate_notes(segmented_notes)  # ← new
    if debug:
        print(
            f"[5.5] consolidate   : {len(segmented_notes)} → "
            f"{len(consolidated_notes)} note events after consolidation"
        )
        np.save(f"{debug_dir}/stage7_midi_consolidated.npy", consolidated_notes)

    # Stage 8 — build MIDI object
    midi = build_midi(consolidated_notes)
    if debug:
        print("[6] build_midi      : MIDI object created")

    # Stage 9 — synthesize audio
    audio = synthesize(midi=midi, soundfont_path=soundfont_path)
    if debug:
        output_duration = len(audio) / 44100
        print(
            f"[7] synthesize      : {len(audio)} "
            f"samples @ 44100Hz ({output_duration:.2f}s)"
        )
        print(f"\n=== Debug arrays saved to '{debug_dir}/' ===\n")

    return audio
