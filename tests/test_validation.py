import numpy as np
import pandas as pd
import pytest

from ceva.validation.compare import error_metrics, match_in_time, metrics_table


def frame(times, values, column="x", transect="t0"):
    return pd.DataFrame({"transect_id": transect, "time": pd.to_datetime(times), column: values})


def test_survey_is_interpolated_when_bracketed_within_ten_days():
    survey = frame(["2020-01-01", "2020-01-11"], [100.0, 120.0])
    sat = frame(["2020-01-06"], [115.0], "x_corr")
    m = match_in_time(sat, survey, "x_corr")
    assert m["x_survey"].iloc[0] == pytest.approx(110.0)
    assert m["error"].iloc[0] == pytest.approx(5.0)
    assert m["gap_days"].iloc[0] == pytest.approx(5.0)


def test_nearest_survey_within_three_days_otherwise_no_match():
    survey = frame(["2020-01-01", "2020-03-01"], [100.0, 140.0])
    sat = frame(["2020-01-03", "2020-01-20", "2019-12-30", "2019-12-20"], [90.0, 90.0, 90.0, 90.0], "x_corr")
    m = match_in_time(sat, survey, "x_corr")
    assert m["time"].dt.strftime("%m-%d").tolist() == ["01-03", "12-30"]
    assert m["x_survey"].tolist() == [100.0, 100.0]


def test_transects_are_matched_separately_and_missing_ones_skipped():
    survey = pd.concat([frame(["2020-01-01", "2020-01-05"], [100.0, 100.0], transect="a"),
                        frame(["2020-01-01", "2020-01-05"], [300.0, 300.0], transect="b")])
    sat = pd.concat([frame(["2020-01-03"], [110.0], "x_corr", "a"), frame(["2020-01-03"], [290.0], "x_corr", "b"),
                     frame(["2020-01-03"], [0.0], "x_corr", "c")])
    m = match_in_time(sat, survey, "x_corr").set_index("transect_id")["error"]
    assert m.to_dict() == {"a": 10.0, "b": -10.0}
    assert match_in_time(sat[sat["transect_id"] == "c"], survey, "x_corr").empty


def test_error_metrics():
    m = error_metrics(pd.Series([-10.0, 0.0, 10.0, 20.0, np.nan]))
    assert m["n"] == 4 and m["bias"] == pytest.approx(5.0)
    assert m["rmse"] == pytest.approx(np.sqrt(150.0)) and m["mae"] == pytest.approx(10.0)
    assert m["std"] == pytest.approx(np.std([-10, 0, 10, 20], ddof=1))
    assert error_metrics(pd.Series([], dtype=float))["n"] == 0


def test_correlation_ignores_each_transects_own_offset():
    rng = np.random.default_rng(0)
    rows = []
    for transect, offset in (("a", 0.0), ("b", 500.0)):
        truth = rng.normal(0, 20, 50)
        rows.append(pd.DataFrame({"transect_id": transect, "x_survey": truth + offset, "error": -15.0, "group": "g"}))
    table = metrics_table(pd.concat(rows), ["group"])
    assert table["r"].iloc[0] == pytest.approx(1.0) and table["bias"].iloc[0] == pytest.approx(-15.0)
