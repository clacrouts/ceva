import numpy as np
import pytest

from ceva.shoreline import clear_sky, mndwi, nearest_crossing, orient_seaward, seaward_edge

DISTANCE = np.arange(0.0, 1000.0, 10.0)


def ramp(shoreline_m: float, slope: float = 0.01) -> np.ndarray:
    """Index that is negative on land and crosses zero exactly at ``shoreline_m``."""
    return ((DISTANCE - shoreline_m) * slope).astype("float32")


def test_mndwi_values_and_undefined_cases():
    out = mndwi(np.array([0.3, 0.0, -0.05, 0.1]), np.array([0.1, 0.0, 0.2, -0.02]))
    assert out[0] == pytest.approx(0.5)
    assert np.isnan(out[1])            # no signal at all
    assert out[2] == pytest.approx(-1.0)   # negative green clipped to zero
    assert out[3] == pytest.approx(1.0)


def test_clear_sky_uses_each_sensor_rule():
    qa = np.array([[[0b01000000, 0b00001000]], [[6, 9]]], dtype="uint8")   # [time=2, transect=1, distance=2]
    ok = clear_sky(qa, np.array([True, False]), [0, 1, 2, 3, 4, 5], [0, 1, 3, 8, 9, 10, 11])
    assert ok.tolist() == [[[True, False]], [[True, False]]]


@pytest.mark.parametrize("position", [203.0, 455.5, 787.25])
def test_seaward_edge_is_sub_sample_accurate(position):
    assert seaward_edge(ramp(position), DISTANCE, 0.0, 50.0) == pytest.approx(position, abs=0.01)


def test_seaward_edge_ignores_small_offshore_land_and_inland_water():
    index = ramp(400.0)
    index[10:14] = 0.5        # a lagoon behind the beach
    index[70:72] = -0.5       # 20 m of "land" offshore: a boat, a bar crest, noise
    assert seaward_edge(index, DISTANCE, 0.0, 50.0) == pytest.approx(400.0, abs=0.01)


def test_seaward_edge_treats_a_long_emerged_bar_as_land():
    index = ramp(400.0)
    index[70:80] = -0.5       # 100 m of land offshore, longer than the minimum run
    assert 780.0 < seaward_edge(index, DISTANCE, 0.0, 50.0) < 800.0


def test_seaward_edge_needs_land_then_water():
    assert np.isnan(seaward_edge(np.full(100, -0.3, "float32"), DISTANCE, 0.0, 50.0))
    assert np.isnan(seaward_edge(np.full(100, 0.3, "float32"), DISTANCE, 0.0, 50.0))
    ends_on_land = ramp(400.0)
    ends_on_land[90:] = -0.5  # the transect is too short to reach open water
    assert np.isnan(seaward_edge(ends_on_land, DISTANCE, 0.0, 50.0))


def test_seaward_edge_does_not_interpolate_across_a_gap():
    index = ramp(400.0)
    index[38:42] = np.nan     # the crossing itself is missing
    assert np.isnan(seaward_edge(index, DISTANCE, 0.0, 50.0))


def test_seaward_edge_leading_gap_as_land_for_surveys():
    index = ramp(30.0)
    index[:2] = np.nan        # survey starts on the dune, 20 m in
    assert np.isnan(seaward_edge(index, DISTANCE, 0.0, 50.0))
    assert seaward_edge(index, DISTANCE, 0.0, 50.0, leading_gap_is_land=True) == pytest.approx(30.0, abs=0.01)


def test_seaward_edge_is_vectorised():
    stack = np.stack([ramp(300.0), ramp(600.0)])
    assert seaward_edge(stack, DISTANCE, 0.0, 50.0) == pytest.approx([300.0, 600.0], abs=0.01)


def test_nearest_crossing_picks_the_one_near_the_reference():
    index = ramp(400.0)
    index[10:14] = 0.5                                     # second land-to-water crossing near 95 m
    reference = np.array([410.0])
    assert nearest_crossing(index[None], DISTANCE, 0.0, reference, 150.0)[0] == pytest.approx(400.0, abs=0.01)
    assert np.isnan(nearest_crossing(index[None], DISTANCE, 0.0, np.array([700.0]), 150.0)[0])
    assert np.isnan(nearest_crossing(index[None], DISTANCE, 0.0, np.array([np.nan]), 150.0)[0])


def test_nearest_crossing_with_per_observation_threshold_and_validity():
    index = np.stack([ramp(400.0), ramp(400.0)])[:, None, :]             # [time=2, transect=1, distance]
    found = nearest_crossing(index, DISTANCE, np.array([[0.0], [0.5]]), np.full((2, 1), 400.0), 150.0)
    assert found[:, 0] == pytest.approx([400.0, 450.0], abs=0.01)
    cloudy = index.copy()
    cloudy[0, 0, 42:60] = np.nan                                           # most of the window is missing
    out = nearest_crossing(cloudy, DISTANCE, 0.0, np.full((2, 1), 400.0), 150.0, min_valid_fraction=0.8)
    assert np.isnan(out[0, 0]) and out[1, 0] == pytest.approx(400.0, abs=0.01)


def test_orient_seaward():
    typical = np.stack([ramp(400.0), ramp(400.0)[::-1], np.full(100, -0.4, "float32"), np.full(100, np.nan, "float32")])
    assert orient_seaward(typical).tolist() == [1, -1, 0, 0]
