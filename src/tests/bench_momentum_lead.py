"""Momentum-leading test harness.

Two halves, shared trajectory generator:
  - open-loop sweep: how well a plain WMA velocity estimate predicts the target
    position one input-delay (7 ms) ahead. Sweeps buffer length x lead_sens.
  - closed-loop validation: simulate the reticle chasing the target with the real
    _scale_delta curve, comparing how the lead is applied (combined / separate /
    identity) and where its velocity comes from (WMA of mouse command vs direct
    target position-diff). Metric: interception MAE (reticle vs future target pos).

All in pixel space via a constant angular->pixel factor (constant world distance,
no screen<->world transforms). Run:
    python -m src.tests.bench_momentum_lead
"""
import sys
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.aimbot.data_parsing.targetselector import RingBuffer2D

# --- sim constants (tunable) --------------------------------------------------
FPS = 144
DT = 1.0 / FPS
N_FRAMES = 600                 # ~4 s
FOV_DEG = 80
DETECTION_W = 640
PX_PER_DEG = DETECTION_W / FOV_DEG   # 8 px/deg
DELAY_S = 0.007 # 7ms for 144fps is technically perfect
BASE_SEED = 1234
ZIG_PERIOD = 20                # frames between vy sign flips
CIRC_PERIOD = 90               # frames per revolution
MOMENTUM = 0.50                 # SGD-style inertia: no instant velocity change

# preset tables
SPEEDS = {"slow": 30.0, "med": 150.0, "fast": 400.0}     # deg/s mean
MOVEMENTS = ("linear", "zigzag", "circular")
RANDOMNESS = {"none": 0.0, "med": 0.33, "high": 1.0}     # sd = k * speed_mean

# open-loop model sweep
FRAME_BUFFERS = (8, 32, 128)
LEAD_SENS = (0.05, 0.2, 0.5, 1.0)
WARMUP = max(FRAME_BUFFERS)    # skip buffer-fill region so big buffers aren't penalized

# closed-loop params (mousemover defaults)
OVERALL_SENS = 0.45
SENS_SCALING = 0.33
MAX_DELTAS = 100
E1_LEAD_SENS = 0.2
E1_FRAME_BUFFER = 8
E2_SCENARIO = ("med", "linear", "none")
E2_SENS = (0.25, 0.45, 0.8)
# none: no lead. combined: lead in setpoint, scaled with offset (current production).
# separate: scale offset and lead independently. identity: lead applied raw/unscaled.
# posdiff: lead from direct target position-diff velocity (not the mouse-command WMA).
COMPOSE_MODES = ("none", "combined", "separate", "identity", "posdiff")


# --- open-loop models ---------------------------------------------------------
class WMAMomentumModel:
    """Linearly-weighted moving average of observed per-frame movement, used as a
    velocity estimate for leading. Mirrors TargetSelector.lead_target's WMA core."""

    def __init__(self, frame_buffer: int, lead_sens: float, fps: int):
        self.name = f"wma_fb{frame_buffer}_ls{lead_sens}"
        self.cap = frame_buffer
        self.lead_sens = lead_sens
        self.fps = fps
        self.buf = RingBuffer2D(frame_buffer)
        w = np.arange(frame_buffer, 0, -1, dtype=np.float64)  # [n, n-1, ..., 1] newest-first
        self.w = w / w.sum()

    def observe(self, dx: float, dy: float):
        self.buf.push(dx, dy)

    def predict_lead(self, delay_s: float):
        a = self.buf.ordered()                      # (cap, 2) newest-first
        vx = float((a[:, 0] * self.w).sum())        # px/frame
        vy = float((a[:, 1] * self.w).sum())
        # px/frame * frames/s * s * sens -> px
        return vx * self.fps * delay_s * self.lead_sens, vy * self.fps * delay_s * self.lead_sens


class NoLeadBaseline:
    """Predicts zero lead -> MAE equals the target's displacement over the delay,
    i.e. the residual error you'd carry with pure-P / no leading."""

    name = "no_lead"

    def observe(self, dx, dy):
        pass

    def predict_lead(self, delay_s):
        return 0.0, 0.0


def build_models():
    models = [NoLeadBaseline()]
    for fb in FRAME_BUFFERS:
        for ls in LEAD_SENS:
            models.append(WMAMomentumModel(fb, ls, FPS))
    return models


# --- trajectory generation ----------------------------------------------------
def _direction(pattern: str, t: int) -> np.ndarray:
    """Unit direction of the desired velocity at frame t."""
    if pattern == "linear":
        return np.array([1.0, 0.0])
    if pattern == "zigzag":
        vy = 1.0 if (t // ZIG_PERIOD) % 2 == 0 else -1.0
        d = np.array([1.0, vy])
        return d / np.linalg.norm(d)
    if pattern == "circular":
        w = 2.0 * np.pi / CIRC_PERIOD
        return np.array([np.cos(w * t), np.sin(w * t)])
    raise ValueError(pattern)


def make_trajectory(speed_deg_s: float, pattern: str, k_rand: float, seed: int):
    """Return (pos, disp): pos (N,2) absolute pixel position, disp (N,2) per-frame
    displacement (disp[0] = 0). Target has SGD-momentum inertia."""
    rng = np.random.default_rng(seed)
    speed_px_s = speed_deg_s * PX_PER_DEG
    sd_px_s = k_rand * speed_px_s

    pos = np.zeros((N_FRAMES, 2), dtype=np.float64)
    disp = np.zeros((N_FRAMES, 2), dtype=np.float64)
    v = np.zeros(2, dtype=np.float64)   # px/s
    for t in range(1, N_FRAMES):
        desired = speed_px_s * _direction(pattern, t)
        noisy = desired + rng.normal(0.0, sd_px_s, size=2) if sd_px_s > 0 else desired
        v = MOMENTUM * v + (1.0 - MOMENTUM) * noisy
        step = v * DT
        disp[t] = step
        pos[t] = pos[t - 1] + step
    return pos, disp


def interp_pos(pos: np.ndarray, time_s: float) -> np.ndarray:
    """Linear-interpolated absolute position at continuous time_s (delay-exact)."""
    f = time_s / DT
    lo = int(np.floor(f))
    hi = lo + 1
    frac = f - lo
    hi = min(hi, N_FRAMES - 1)
    return pos[lo] * (1.0 - frac) + pos[hi] * frac


# --- scaling (ported from mousemover) -----------------------------------------
def scale_delta(delta: float, overall_sens: float, sens_scaling: float, max_deltas: float) -> float:
    """Ported verbatim from MouseMover._scale_delta (mousemover.py:63)."""
    x = abs(delta) / max_deltas
    blend = 1 - math.exp(-x * 2.5)
    return overall_sens * delta * (1.0 + (sens_scaling - 1.0) * blend)


def _S(vec: np.ndarray, overall_sens: float) -> np.ndarray:
    return np.array([scale_delta(vec[0], overall_sens, SENS_SCALING, MAX_DELTAS),
                     scale_delta(vec[1], overall_sens, SENS_SCALING, MAX_DELTAS)])


# --- open-loop runner ---------------------------------------------------------
def run_scenario(pos, disp, models) -> dict:
    """Returns {model_name: MAE} for one trajectory (open-loop prediction)."""
    last_t = int(np.floor((N_FRAMES - 1) - DELAY_S / DT))
    for m in models:
        if hasattr(m, "buf"):
            m.buf.buf[:] = 0.0
            m.buf.idx = 0

    errs = {m.name: [] for m in models}
    for t in range(N_FRAMES):
        for m in models:
            m.observe(disp[t, 0], disp[t, 1])
        if t < WARMUP or t > last_t:
            continue
        true_pos = interp_pos(pos, t * DT + DELAY_S)
        for m in models:
            lx, ly = m.predict_lead(DELAY_S)
            pred = pos[t] + np.array([lx, ly])
            errs[m.name].append(float(np.linalg.norm(pred - true_pos)))
    return {name: float(np.mean(e)) for name, e in errs.items()}


# --- closed-loop runner -------------------------------------------------------
def run_closed_loop(pos: np.ndarray, lead_sens: float, overall_sens: float,
                    compose: str, frame_buffer: int) -> float:
    """Simulate the reticle chasing target `pos`; return interception MAE (px) vs the
    future target position pos[t + DELAY_S].

    Lead application:
      none      no lead; move = S(offset).
      combined  lead in setpoint, scaled with offset (current production): S(offset + lead).
      separate  scale offset and lead independently: S(offset) + S(lead).
      identity  lead applied raw/unscaled: S(offset) + lead.
    Velocity source (what the buffer holds):
      combined/separate/identity  the raw mouse command (offset + lead).
      posdiff                     the direct target position-diff (pos[t]-pos[t-1]) = true
                                  screen velocity, applied raw/identity (S(offset) + lead).
    """
    buf = RingBuffer2D(frame_buffer)
    w = np.arange(frame_buffer, 0, -1, dtype=np.float64)
    w = (w / w.sum()).reshape(-1, 1)

    reticle = pos[0].copy()
    last_t = int(np.floor((N_FRAMES - 1) - DELAY_S / DT))
    errs = []
    for t in range(N_FRAMES):
        offset = pos[t] - reticle
        wma = (buf.ordered() * w).sum(axis=0)             # velocity proxy from prior frames
        lead = np.zeros(2) if compose == "none" else wma * FPS * DELAY_S * lead_sens

        if compose == "none":
            move = _S(offset, overall_sens)
        elif compose == "combined":
            move = _S(offset + lead, overall_sens)
        elif compose == "separate":
            move = _S(offset, overall_sens) + _S(lead, overall_sens)
        elif compose in ("identity", "posdiff"):
            move = _S(offset, overall_sens) + lead         # lead raw, no scaling
        else:
            raise ValueError(compose)

        if compose == "posdiff":
            disp = pos[t] - pos[t - 1] if t > 0 else np.zeros(2)  # direct target velocity
            buf.push(disp[0], disp[1])
        else:
            buf.push(offset[0] + lead[0], offset[1] + lead[1])    # raw mouse command
        reticle = reticle + move
        if WARMUP <= t <= last_t:
            true_future = interp_pos(pos, t * DT + DELAY_S)
            errs.append(float(np.linalg.norm(true_future - reticle)))
    return float(np.mean(errs))


# --- printing -----------------------------------------------------------------
def _print_rows(title: str, rows: list[tuple[str, float]]):
    print(f"\n{title}")
    print(f"  {'model':<18}  {'MAE (px)':>10}")
    print(f"  {'-' * 18}  {'-' * 10}")
    for name, mae in rows:
        print(f"  {name:<18}  {mae:>10.3f}")


def _print_matrix(title: str, header, rows):
    print(f"\n{title}")
    cols = "  ".join(f"{h:>10}" for h in header)
    print(f"  {'scenario':<26}  {cols}")
    print(f"  {'-'*26}  {'  '.join('-'*10 for _ in header)}")
    for label, vals in rows:
        cells = "  ".join(f"{v:>10.3f}" for v in vals)
        print(f"  {label:<26}  {cells}")


# --- experiments --------------------------------------------------------------
def open_loop_sweep():
    print(f"\n### open-loop sweep | FPS={FPS} delay={DELAY_S*1000:.1f}ms "
          f"N={N_FRAMES} warmup={WARMUP} px_per_deg={PX_PER_DEG:.2f}")
    overall = {}
    idx = 0
    for sp_name, sp in SPEEDS.items():
        for mv in MOVEMENTS:
            for rd_name, rd_k in RANDOMNESS.items():
                models = build_models()
                pos, disp = make_trajectory(sp, mv, rd_k, BASE_SEED + idx)
                result = run_scenario(pos, disp, models)
                for name, mae in result.items():
                    overall.setdefault(name, []).append(mae)
                idx += 1
    overall_rows = sorted(((n, float(np.mean(v))) for n, v in overall.items()), key=lambda kv: kv[1])
    _print_rows("open-loop overall_all  [mean MAE across 27 scenarios, best->worst]", overall_rows)


def closed_loop_e1():
    print(f"\n### closed-loop E1: lead application & source | lead_sens={E1_LEAD_SENS} "
          f"fb={E1_FRAME_BUFFER} sens={OVERALL_SENS}")
    overall = {m: [] for m in COMPOSE_MODES}
    rows = []
    idx = 0
    for sp_name, sp in SPEEDS.items():
        for mv in MOVEMENTS:
            for rd_name, rd_k in RANDOMNESS.items():
                pos, _ = make_trajectory(sp, mv, rd_k, BASE_SEED + idx)
                vals = [run_closed_loop(pos, E1_LEAD_SENS, OVERALL_SENS, m, E1_FRAME_BUFFER)
                        for m in COMPOSE_MODES]
                for m, v in zip(COMPOSE_MODES, vals):
                    overall[m].append(v)
                rows.append((f"{sp_name}|{mv}|{rd_name}", vals))
                idx += 1
    _print_matrix("E1 per-scenario interception MAE (px)", COMPOSE_MODES, rows)
    _print_matrix("E1 overall_all", COMPOSE_MODES,
                  [("MEAN across 27", [float(np.mean(overall[m])) for m in COMPOSE_MODES])])


def closed_loop_e2():
    sp_name, mv, rd_name = E2_SCENARIO
    pos, _ = make_trajectory(SPEEDS[sp_name], mv, RANDOMNESS[rd_name], BASE_SEED + 99)
    print(f"\n### closed-loop E2: sens-invariance | scenario={sp_name}|{mv}|{rd_name} "
          f"lead_sens={E1_LEAD_SENS} fb={E1_FRAME_BUFFER}")
    print("  leading_fraction = 1 - MAE_lead / MAE_none  (higher = lead cancels more lag)")
    e2_modes = ("combined", "separate", "identity", "posdiff")
    rows = []
    for s in E2_SENS:
        none = run_closed_loop(pos, E1_LEAD_SENS, s, "none", E1_FRAME_BUFFER)
        vals = [1.0 - run_closed_loop(pos, E1_LEAD_SENS, s, m, E1_FRAME_BUFFER) / none
                for m in e2_modes]
        rows.append((f"overall_sens={s}", vals))
    _print_matrix("E2 leading fraction vs base sens", e2_modes, rows)


def main():
    open_loop_sweep()
    closed_loop_e1()
    closed_loop_e2()


if __name__ == "__main__":
    main()
