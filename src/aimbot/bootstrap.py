"""Construction helpers for Aimbot.

Each factory builds and returns exactly one component; Aimbot.__init__
orchestrates and assigns. This module must never import aimbot.py.
"""

from argparse import Namespace

# can replace with bettercam just no cupy support; pass nvidia_gpu=False on create for bettercam
import betterercam
from screeninfo import get_monitors

from .engine.hsv_crosshair import HSVCrosshairDetector
from .input import inputdetector, mousemover
from .utils.utils import log


def validate_targeting_config(cfg: dict) -> None:
    """Raises ValueError if the targeting configuration is invalid."""
    ts = cfg['targeting_settings']
    lead_target = ts['lead_target']
    predict_drop = ts['predict_drop']
    model_xh = ts['model_predict_crosshair']
    hsv_xh = ts['hsv_settings']['enabled']

    if lead_target and not predict_drop:
        raise ValueError(
            "Invalid targeting configuration: 'lead_target' requires 'predict_drop' to be enabled. "
            "Please enable 'predict_drop' in your config or disable 'lead_target'."
        )

    if model_xh and hsv_xh:
        raise ValueError(
            "Invalid targeting configuration: 'model_predict_crosshair' and 'hsv_settings.enabled' "
            "are mutually exclusive. Enable exactly one (or neither) in your config."
        )

    if hsv_xh:
        scheme = ts['hsv_settings']['voting_scheme']
        if scheme not in HSVCrosshairDetector.VOTING_SCHEMES:
            raise ValueError(
                f"Invalid 'hsv_settings.voting_scheme': {scheme!r}. Must be one of {list(HSVCrosshairDetector.VOTING_SCHEMES)}."
            )


def get_screen_dims(cfg: dict) -> tuple[int, int]:
    """(screen_x, screen_y) of the configured monitor."""
    monitor_idx = cfg['display_settings']['monitor_idx']
    monitor = get_monitors()[monitor_idx]
    log(f'LOOKING AT MONITOR: {monitor_idx}', "INFO")
    log(f'MONITOR DIMS: {monitor.width} x {monitor.height}', "INFO")
    return monitor.width, monitor.height


def create_camera(screen_xy: tuple[int, int], base_hw_capture: tuple[int, int]):
    """betterercam capture of a fixed region centered on screen at base_hw_capture size.
    base_model consumes the full frame; the sr model consumes a sub-slice.
    Note: ultralytics .pt inference assumes BGR input; engines here are trained on RGB."""
    screen_x, screen_y = screen_xy
    base_x_offset = (screen_x - base_hw_capture[1]) // 2
    base_y_offset = (screen_y - base_hw_capture[0]) // 2
    log(f'screen_x: {screen_x}', "DEBUG")
    log(f'base_hw_capture: {base_hw_capture}', "DEBUG")

    base_region = (
        base_x_offset,
        base_y_offset,
        screen_x - base_x_offset,
        screen_y - base_y_offset,
    )
    return betterercam.create(region=base_region, output_color='RGB', max_buffer_len=2, nvidia_gpu=True)


def create_mousemover(cfg: dict) -> mousemover.MouseMover:
    """Threaded variant if input_settings.separate_mouse_thread, else the direct one.
    Started here so callers get something already running; Aimbot.cleanup stops it."""
    sens_cfg = cfg['sensitivity_settings']
    args = (
        sens_cfg['overall_sens'],
        sens_cfg['sens_scaling'],
        sens_cfg['max_deltas'],
        sens_cfg['jitter_strength'],
        sens_cfg['overshoot_strength'],
        sens_cfg['overshoot_chance'],
    )
    input_cfg = cfg['input_settings']
    if not input_cfg.get('separate_mouse_thread', False):
        return mousemover.MouseMover(*args)

    mt = input_cfg['mouse_thread_config']
    mover = mousemover.ThreadedMouseMover(
        *args,
        poll_hz=mt['poll_hz'],
        drain_alpha=mt['drain_alpha'],
        min_step_px=mt['min_step_px'],
        output_alpha=mt['output_alpha'],
    )
    mover.start()
    return mover


def create_inputdetector(cfg: dict) -> inputdetector.InputDetector:
    """Constructs the detector AND starts its listeners."""
    detector = inputdetector.InputDetector(cfg['input_settings']['toggle_hotkey'])
    detector.start_input_detection()
    return detector


def create_tracker(cfg: dict, model_ext: str):
    """BYTETracker per cfg tracker_settings.tracker_impl, conforming to the
    update / multi_predict / get_active_tracks_with_lifetime contract
    (see engine.tracker_adapter)."""
    #if engine is running just going to assume 144 is the target frame rate
    #if pt model is running its probably debug screen so 30
    target_frame_rate = 144 if model_ext == ".engine" else 30
    args = Namespace(
        track_high_thresh=0.65,
        track_low_thresh=0.4,
        track_buffer=20, #track_buffer -> time = track_buffer/30 so 20/30 = 0.66 seconds until lost
        fuse_score=0.5,
        match_thresh=0.6,
        new_track_thresh=0.65
    )

    tracker_impl = cfg['tracker_settings']['tracker_impl']
    allowed_impls = ['ultralytics', 'ultralytics_vectorized', 'cpp']
    assert tracker_impl in allowed_impls, f"expected tracker_settings.tracker_impl to be in {allowed_impls}, got {tracker_impl}"

    if tracker_impl == "ultralytics":
        from ultralytics.trackers.byte_tracker import BYTETracker
    elif tracker_impl == "ultralytics_vectorized":
        from c_bytetracker.trackers.byte_tracker import BYTETracker
    elif tracker_impl == "cpp":
        from c_bytetracker.cpp_tracker import CppBYTETracker as BYTETracker

    tracker = BYTETracker(args, frame_rate=target_frame_rate)
    # stock ultralytics returns (M,8) with no lifetime cols and wants a results object,
    # not a raw (N,6) array — wrap it to match the loop's contract. vectorized/cpp are native.
    if tracker_impl == "ultralytics":
        from .engine.tracker_adapter import UltralyticsAdapter
        tracker = UltralyticsAdapter(tracker)
    return tracker
