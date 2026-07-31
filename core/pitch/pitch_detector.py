import numpy as np
import torch
import torchcrepe

from core.audio.audio_input import PIPELINE_SAMPLE_RATE

HOP_LENGTH = 160  # 10ms hop length at 16kHz

# these values cover the general adult humming range
# chosen to reduce octave errors on the kind of input this project expects
FMIN = 50.0
FMAX = 550.0


def detect_pitch(
    audio: np.ndarray, sr: int = PIPELINE_SAMPLE_RATE
) -> dict[str, np.ndarray]:
    """
    Detect pitch and periodicity from an audio time series.

    Parameters:
    audio (np.ndarray): Audio time series.
    sr (int): Sample rate of the audio time series. Default is 16000 Hz.

    Returns:
    A dictionary containing:
    1. frequency (np.ndarray):
        Detected pitch values in Hz.
    2. periodicity (np.ndarray):
        Detected periodicity value range 0.0–1.0,
        where higher means a more stable periodic signal.
    3. time (np.ndarray):
        Time values corresponding to the pitch and periodicity in seconds.
    """

    # torchcrepe expects a batched tensor (batch, samples);
    # single clip treated as batch of one
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)

    pitch, periodicity = torchcrepe.predict(
        audio=audio_tensor,
        sample_rate=sr,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        fmax=FMAX,
        model="full",
        decoder=torchcrepe.decode.viterbi,
        return_periodicity=True,
    )

    # strip the batch dimension torchcrepe added,
    # convert back to numpy for pipeline consistency
    pitch = pitch.squeeze(0).numpy()
    periodicity = periodicity.squeeze(0).numpy()

    # torchcrepe does not return timestamps;
    # reconstruct from hop position and sample rate
    time = np.arange(len(pitch)) * HOP_LENGTH / sr

    return {"frequency": pitch, "periodicity": periodicity, "time": time}
