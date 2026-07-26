import numpy as np
import pytest

from core.pitch.pitch_filter import DEFAULT_THRESHOLD, filter_by_periodicity


@pytest.fixture
def make_pitch_data():
    """
    Create a mock pitch data dictionary with configurable frequency, periodicity,
    and time arrays.
    Defaults to 10 frames of data with frequencies ranging from 50 Hz to 550 Hz,
    periodicity values ranging from 0.0 to 1.0
    Time values from 0.0 to 0.09 seconds.
    """

    def _make_pitch_data(
        num_frames: int = 10,
        freq_min: float = 50.0,
        freq_max: float = 550.0,
        periodicity_min: float = 0.0,
        periodicity_max: float = 1.0,
        time_start: float = 0.0,
        time_end: float = 0.09,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        frequency = rng.uniform(freq_min, freq_max, num_frames).astype(np.float32)
        periodicity = rng.uniform(periodicity_min, periodicity_max, num_frames).astype(
            np.float32
        )
        # guarantee at least one below-threshold frame for filter tests
        periodicity[0] = 0.10
        time = np.linspace(time_start, time_end, num_frames).astype(np.float32)
        return {"frequency": frequency, "periodicity": periodicity, "time": time}

    return _make_pitch_data


def test_filter_by_periodicity(make_pitch_data) -> None:
    """
    Test that filter_by_periodicity correctly filters pitch data
    based on the periodicity threshold.
    """

    pitch_data = make_pitch_data()
    filtered_data = filter_by_periodicity(pitch_data, DEFAULT_THRESHOLD)

    # Check that the output is a dictionary with the expected keys
    assert isinstance(filtered_data, dict)
    assert (
        "frequency" in filtered_data
        and "periodicity" in filtered_data
        and "time" in filtered_data
    )

    # Check that the frequency values below the threshold are set to 0.0
    for freq, orig_freq, periodicity in zip(
        filtered_data["frequency"],
        pitch_data["frequency"],
        filtered_data["periodicity"],
    ):
        if periodicity < DEFAULT_THRESHOLD:
            assert freq == 0.0
        else:
            assert freq == orig_freq

    # Check that the periodicity and time values remain unchanged
    np.testing.assert_array_equal(
        filtered_data["periodicity"], pitch_data["periodicity"]
    )
    np.testing.assert_array_equal(filtered_data["time"], pitch_data["time"])
