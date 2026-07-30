import os

import numpy as np
import pytest

from core.pipeline import run_pipeline
from core.synthesis.synthesizer import SOUNDFONT_PATH

# This skips all the tests if valid soundfont is not present.
pytestmark = pytest.mark.skipif(
    not os.path.exists(SOUNDFONT_PATH), reason="Soundfont doesn't exists"
)


def test_run_pipeline_non_empty_output(make_mono_wav_file) -> None:
    """
    Test that a WAV file with a given sample rate is processed successfully
    in the pipeline and produces an non-empty numpy array.
    """
    path = make_mono_wav_file(sample_rate=44100)
    audio = run_pipeline(path)
    assert isinstance(audio, np.ndarray)
    assert len(audio) > 0
