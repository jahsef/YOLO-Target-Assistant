"""The perf harness's own comparison logic.

It decides what gets flagged as a regression, so a bug here means either silent
regressions or noise that trains you to ignore the report.
"""

import pytest

from tests.conftest import COMPARE_STAT, REGRESSION_PCT, _fmt_delta


def row(minimum, p50=None, name="m", n=6, unit=""):
    """A distribution row. p50 defaults to min (no run-internal jitter). Values are big
    enough to clear MIN_ABS_MS unless a test says otherwise."""
    return {"name": name, "min": minimum, "p50": minimum if p50 is None else p50,
            "mean": minimum, "n": n, "unit": unit}


def base(minimum, p50=None, name="m", n=6, unit=""):
    return {name: row(minimum, p50, name, n, unit)}


def point(value, name="m"):
    return {"name": name, "mean": value, "min": value, "p50": value, "n": 1}


class TestDelta:
    def test_compares_steady_state_not_mean(self):
        assert COMPARE_STAT == "min"

    def test_no_baseline_entry(self):
        assert _fmt_delta(row(1.0), {}) == ("—", False)

    def test_zero_baseline_is_not_a_divide_by_zero(self):
        assert _fmt_delta(row(1.0), base(0.0)) == ("—", False)

    def test_slower_is_positive(self):
        text, flag = _fmt_delta(row(1.5), base(1.0))
        assert text.startswith("+50.0%") and flag

    def test_faster_shows_a_delta_but_is_not_flagged(self):
        """Getting faster is never something to go investigate."""
        text, flag = _fmt_delta(row(0.5, unit="ms"), base(1.0, unit="ms"))
        assert text.startswith("-50.0%")
        assert not flag

    def test_small_move_is_not_flagged(self):
        _, flag = _fmt_delta(row(1.0 + REGRESSION_PCT / 2 / 100), base(1.0))
        assert not flag

    def test_mean_swings_do_not_flag_when_the_floor_held(self):
        """The real failure mode this fixes: on a contended GPU a metric's mean can
        move 60% between identical runs while its fastest trial moves ~1%."""
        r = {"name": "m", "min": 0.2699, "p50": 0.2728, "mean": 0.2750, "n": 6}
        b = {"m": {"name": "m", "min": 0.2738, "p50": 0.2740, "mean": 0.8441, "n": 6}}
        text, flag = _fmt_delta(r, b)
        assert not flag, f"floor barely moved but got flagged: {text}"

    def test_jittery_metric_needs_a_bigger_move(self):
        """A metric whose p50 sits 60% above its own floor can't support a 30% claim."""
        _, flag = _fmt_delta(row(1.3, p50=2.1), base(1.0, p50=1.6))
        assert not flag

    def test_stable_metric_flags_a_real_slowdown(self):
        _, flag = _fmt_delta(row(1.5, unit="ms"), base(1.0, unit="ms"))
        assert flag

    def test_fps_drop_is_a_regression(self):
        """Direction depends on the unit: fewer fps is worse, more ms is worse."""
        _, flag = _fmt_delta({"name": "m", "mean": 150.0, "min": 150.0, "p50": 150.0,
                              "n": 1, "unit": "fps"},
                             {"m": {"name": "m", "mean": 220.0, "min": 220.0,
                                    "p50": 220.0, "n": 1, "unit": "fps"}})
        assert flag

    def test_fps_gain_is_not_flagged(self):
        _, flag = _fmt_delta({"name": "m", "mean": 260.0, "min": 260.0, "p50": 260.0,
                              "n": 1, "unit": "fps"},
                             {"m": {"name": "m", "mean": 220.0, "min": 220.0,
                                    "p50": 220.0, "n": 1, "unit": "fps"}})
        assert not flag

    def test_point_values_compare_on_mean(self):
        text, flag = _fmt_delta(point(2.0), {"m": point(1.0)})
        assert text.startswith("+100.0%") and flag

    def test_flag_marker_is_in_the_text(self):
        text, _ = _fmt_delta(row(2.0), base(1.0))
        assert text.endswith(" !")

    def test_tiny_absolute_move_is_not_flagged(self):
        """0.3 us doubling to 0.6 us is +100% and completely irrelevant at a 4.4 ms
        frame budget. Without this floor the report is all microbenchmark noise."""
        text, flag = _fmt_delta(row(0.0006, unit="ms"), base(0.0003, unit="ms"))
        assert text.startswith("+100.0%")
        assert not flag

    def test_floor_only_applies_to_timings(self):
        """A unitless point value has no ms floor to compare against."""
        _, flag = _fmt_delta(point(0.0006), {"m": point(0.0003)})
        assert flag

    def test_real_regression_still_flags(self):
        """The change this suite exists to catch: hsv fusion being undone."""
        _, flag = _fmt_delta(row(1.18, unit="ms"), base(0.27, unit="ms"))
        assert flag


class TestRecorder:
    def test_record_summarizes_samples(self):
        from tests.conftest import PerfRecorder
        r = PerfRecorder()
        stats = r.record("x", [1.0, 2.0, 3.0], group="g", note="n")
        assert stats["n"] == 3
        assert stats["mean"] == pytest.approx(2.0)
        assert stats["p50"] == pytest.approx(2.0)
        assert stats["group"] == "g" and stats["note"] == "n"
        assert len(r.rows) == 1

    def test_record_value_is_a_point(self):
        from tests.conftest import PerfRecorder
        r = PerfRecorder()
        stats = r.record_value("fps", 200.0, unit="fps")
        assert stats["mean"] == 200.0 and stats["ci95"] == 0.0 and stats["n"] == 1
