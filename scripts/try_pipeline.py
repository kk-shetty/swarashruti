import os
import sys
from datetime import datetime
from pathlib import Path

import soundfile as sf

from core.pipeline import run_pipeline

audio_path = sys.argv[1]

# Shared timestamp links output file to its debug folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Output directory — use provided path if writable, else fall back to output/
if len(sys.argv) > 2:
    requested = Path(sys.argv[2])
    try:
        requested.mkdir(parents=True, exist_ok=True)
        output_dir = requested if os.access(requested, os.W_OK) else Path("output")
    except OSError:
        output_dir = Path("output")
else:
    output_dir = Path("output")

output_dir.mkdir(parents=True, exist_ok=True)

# Debug lives inside the output directory, namespaced by run timestamp
debug_dir = output_dir / "debug" / f"run_{timestamp}"
debug_dir.mkdir(parents=True, exist_ok=True)

# Timestamped filename — no overwrites
output_path = output_dir / f"output_{Path(audio_path).stem}_{timestamp}.wav"

audio = run_pipeline(audio_path, debug=True, debug_dir=str(debug_dir))
sf.write(str(output_path), audio, 44100)
print(f"Done  - {output_path}")
print(f"Debug - {debug_dir}")
