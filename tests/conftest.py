"""Shared fixtures + the perf recording harness.

Perf philosophy: perf tests NEVER fail on a regression. Clock/thermal variance on a
gaming box makes hard thresholds useless. They record to tests/results/latest.md,
diff against tests/results/baseline.json, and flag anything that moved more than
REGRESSION_PCT and beyond its own CI95. You read the table and decide.

    pytest                       # correctness only (perf deselected by pytest.ini)
    pytest -m perf               # perf only, writes tests/results/
    pytest -m perf --update-baseline
"""

import json
from pathlib import Path

import pytest

from tests.support.bench import summarize
from tests.support.cfg import default_cfg

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
BASELINE_PATH = RESULTS_DIR / "baseline.json"
LATEST_MD = RESULTS_DIR / "latest.md"
LATEST_JSON = RESULTS_DIR / "latest.json"

REGRESSION_PCT = 25.0  # flag threshold, not a failure threshold

# Regressions are judged on the FASTEST sample, not the mean. This box shares its GPU
# with the desktop and a browser, so means swing wildly between runs while the fastest
# trial — the steady state a sustained aimbot loop reaches — barely moves. Measured
# across two back-to-back no-op runs: hsv.detect[heuristic_spam] moved -67% by mean and
# -1.4% by min. The table still shows the full distribution; only the flag uses min.
COMPARE_STAT = "min"

# Timing moves smaller than this are never flagged regardless of percentage. The frame
# budget is ~4.4 ms; a 0.3 us call that "regressed 50%" moved 0.15 us and cannot matter.
# Without this floor the report is dominated by sub-30 us CPU microbenchmarks jittering
# with interpreter state and CPU frequency, which just trains you to ignore it.
MIN_ABS_MS = 0.02

# Units where a bigger number is better. Everything else is a duration.
HIGHER_IS_BETTER = {"fps", "x"}


def pytest_addoption(parser):
    parser.addoption("--update-baseline", action="store_true",
                     help="overwrite tests/results/baseline.json with this run's numbers")


# --- capability gates ---------------------------------------------------------

_gpu_cache = {}


def gpu_available() -> bool:
    if "ok" not in _gpu_cache:
        try:
            import cupy as cp
            _gpu_cache["ok"] = cp.cuda.runtime.getDeviceCount() > 0
        except Exception:
            _gpu_cache["ok"] = False
    return _gpu_cache["ok"]


def engine_paths() -> dict:
    cfg = default_cfg()["model"]
    return {
        "base": REPO_ROOT / cfg["base_dir"] / cfg["base_filename"],
        "sr": REPO_ROOT / cfg["sr_model"],
    }


def engines_available() -> bool:
    return all(p.exists() for p in engine_paths().values())


def pytest_runtest_setup(item):
    if item.get_closest_marker("gpu") and not gpu_available():
        pytest.skip("no CUDA device / cupy unavailable")
    if item.get_closest_marker("engine") and not engines_available():
        pytest.skip("model files missing under data/models (gitignored)")


# --- fixtures -----------------------------------------------------------------

@pytest.fixture
def cfg():
    return default_cfg()


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT


# --- perf recording -----------------------------------------------------------

class PerfRecorder:
    def __init__(self):
        self.rows = []

    def record(self, name, samples_ms, unit="ms", note="", group="misc"):
        """samples_ms: iterable of per-call times in ms."""
        stats = summarize(samples_ms)
        stats.update(name=name, unit=unit, note=note, group=group)
        self.rows.append(stats)
        return stats

    def record_value(self, name, value, unit="", note="", group="misc"):
        """Single scalar (throughput, ratio) — no distribution."""
        row = {"name": name, "unit": unit, "note": note, "group": group,
               "n": 1, "mean": float(value), "std": 0.0, "ci95": 0.0,
               "min": float(value), "p50": float(value), "p95": float(value),
               "p99": float(value), "max": float(value)}
        self.rows.append(row)
        return row


@pytest.fixture(scope="session")
def perf(request):
    rec = PerfRecorder()
    request.config._perf_recorder = rec
    return rec


def _load_baseline():
    if BASELINE_PATH.exists():
        return {r["name"]: r for r in json.loads(BASELINE_PATH.read_text())}
    return {}


def _internal_spread(row):
    """How much this metric jittered WITHIN its own run, as a fraction of its floor.
    A metric that varies 50% run-internally can't support a 30% between-run claim."""
    lo = row.get(COMPARE_STAT, row["mean"])
    if not lo or row.get("n", 1) < 2:
        return 0.0
    return max(0.0, (row.get("p50", lo) - lo) / lo)


def _fmt_delta(row, base):
    b = base.get(row["name"])
    stat = COMPARE_STAT if row.get("n", 1) > 1 else "mean"
    if not b or not b.get(stat):
        return "—", False
    new, old = row.get(stat, row["mean"]), b.get(stat, b["mean"])
    pct = (new - old) / old * 100.0
    # flag only when the move is in the WORSE direction, big, bigger than the
    # run-internal jitter, and large enough in absolute terms to move the frame budget.
    # Improvements still show their delta — they just aren't worth investigating, and
    # on a contended desktop noise moves both ways, so ignoring one direction halves
    # the false positives for free.
    worse = pct < 0 if row.get("unit") in HIGHER_IS_BETTER else pct > 0
    noise = max(_internal_spread(row), _internal_spread(b)) * 100.0
    too_small = row.get("unit") == "ms" and abs(new - old) < MIN_ABS_MS
    flag = worse and abs(pct) >= REGRESSION_PCT and abs(pct) > noise and not too_small
    mark = " !" if flag else ""
    return f"{pct:+.1f}%{mark}", flag


def pytest_sessionfinish(session, exitstatus):
    rec = getattr(session.config, "_perf_recorder", None)
    if not rec or not rec.rows:
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = _load_baseline()

    import platform
    from datetime import datetime

    gpu_name = "n/a"
    if gpu_available():
        try:
            import cupy as cp
            gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        except Exception:
            pass

    lines = [
        "perf results",
        "============",
        "",
        f"run      : {datetime.now().isoformat(timespec='seconds')}",
        f"host     : {platform.node()} / {platform.machine()} / py{platform.python_version()}",
        f"gpu      : {gpu_name}",
        f"baseline : {'tests/results/baseline.json' if base else 'none (first run)'}",
        f"vs base  : compares {COMPARE_STAT} (steady state), not mean -- see tests/conftest.py",
        f"flagged  : '!' = moved >{REGRESSION_PCT:.0f}% the wrong way AND more than the metric's",
        "           own run-internal jitter. never fails a test.",
        "           sub-100us GPU metrics are launch-overhead dominated, so treat",
        "           their flags as advisory.",
        "",
    ]

    # widths sized to the content so the columns line up as plain text
    name_w = max(len(r["name"]) for r in rec.rows)
    unit_w = max(4, max(len(r["unit"]) for r in rec.rows))
    deltas = {r["name"]: _fmt_delta(r, base) for r in rec.rows}
    delta_w = max(7, max(len(d) for d, _ in deltas.values()))
    num_w = 9

    header = (f"{'metric':<{name_w}}  {'min':>{num_w}} {'mean':>{num_w}} {'p50':>{num_w}} "
              f"{'p95':>{num_w}} {'p99':>{num_w}} {'n':>5} {'unit':<{unit_w}} "
              f"{'vs base':>{delta_w}}  note")

    flagged = []
    for group in dict.fromkeys(r["group"] for r in rec.rows):
        lines += [group, "-" * len(group), "", header, "-" * len(header)]
        for row in [r for r in rec.rows if r["group"] == group]:
            delta, flag = deltas[row["name"]]
            if flag:
                flagged.append((row["name"], delta))
            if row["n"] > 1:
                stats = " ".join(f"{row[k]:>{num_w}.4f}"
                                 for k in ("min", "mean", "p50", "p95", "p99"))
            else:
                # single value, not a distribution — repeating it five times reads as noise
                blank = " " * num_w
                stats = " ".join([f"{row['mean']:>{num_w}.4f}"] + [blank] * 4)
            lines.append(
                f"{row['name']:<{name_w}}  {stats} {row['n']:>5} {row['unit']:<{unit_w}} "
                f"{delta:>{delta_w}}  {row['note']}".rstrip()
            )
        lines.append("")

    LATEST_MD.write_text("\n".join(lines), encoding="utf-8")
    LATEST_JSON.write_text(json.dumps(rec.rows, indent=1), encoding="utf-8")

    if session.config.getoption("--update-baseline"):
        BASELINE_PATH.write_text(json.dumps(rec.rows, indent=1), encoding="utf-8")

    tr = session.config.pluginmanager.get_plugin("terminalreporter")
    if tr:
        tr.write_sep("=", "perf")
        tr.write_line(f"wrote {LATEST_MD.relative_to(REPO_ROOT)} ({len(rec.rows)} metrics)")
        if flagged:
            tr.write_line(f"{len(flagged)} metric(s) moved past {REGRESSION_PCT:.0f}%:")
            for name, delta in flagged:
                tr.write_line(f"  {name}: {delta}")
        elif base:
            tr.write_line("no metric moved past threshold")
