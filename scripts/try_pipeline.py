import sys
from pathlib import Path

import soundfile as sf

from core.pipeline import run_pipeline

audio_path = sys.argv[1]
output_path = "output/output_" + Path(audio_path).stem + ".wav"
audio = run_pipeline(audio_path, debug=True)
sf.write(output_path, audio, 44100)
print(f"Done - {output_path}")
