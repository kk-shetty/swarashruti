import numpy as np

# torchcrepe periodicity is a sigmoid activation, not a probability.
# 0.21 is the threshold recommended by the torchcrepe authors for clean speech;
# tune upward for noisier input or stricter voiced-frame requirements.
DEFAULT_THRESHOLD = 0.21


def filter_by_periodicity(
    pitch_data: dict[str, np.ndarray], threshold: float = DEFAULT_THRESHOLD
) -> dict[str, np.ndarray]:
    """
    Filter pitch data based on periodicity threshold
    Sets below-threshold pitch values to 0.0, not removed.

    Parameters:
    pitch_data (dict): Dictionary containing 'frequency', 'periodicity', and 'time'
    threshold (float): Periodicity threshold for filtering

    Returns:
    A dictionary containing:
    1. frequency (np.ndarray): Filtered pitch values in Hz.
    2. periodicity (np.ndarray): Original periodicity values between 0.0 - 1.0.
    3. time (np.ndarray): Original time values in seconds.
    """

    filtered_pitch = np.where(
        pitch_data["periodicity"] >= threshold, pitch_data["frequency"], 0.0
    )

    return {
        "frequency": filtered_pitch,
        "periodicity": pitch_data["periodicity"],
        "time": pitch_data["time"],
    }
