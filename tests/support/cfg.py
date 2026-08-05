"""Canonical config for tests.

Deliberately NOT config/cfg.json — that file is the user's live tuning surface and
changes constantly. Tests pin behavior against a fixed dict; test_config_schema.py
separately asserts the live cfg still has every key the code reads.
"""

import copy

_CFG = {
    "logging": {"logging_level": "INFO", "print_fps": False},
    "input_settings": {
        "toggle_hotkey": "",
        "right_click_toggle": True,
        "separate_mouse_thread": False,
        "on_click_pull_down_px": 0,
        "mouse_thread_config": {"poll_hz": 1000, "drain_alpha": 0.2, "min_step_px": 1.0, "output_alpha": 0.5},
    },
    "other": {"inactive_throttle_ms": 10, "async_pipeline": False},
    "display_settings": {"monitor_idx": 0},
    "model": {
        "base_dir": "data/models/pf_1550img_11s/weights",
        "base_filename": "640x640_stripped.engine",
        "scan_sr_bundle": "",
        "sr_model": "data/models/sr/sr_model_b1_engine.pt",
        "sr_hysteresis_frames": 2,
        "pt_hw_capture": [320, 320],
        "conf_threshold": 0.25,
        "union_nms_iou": 0.25,
    },
    "sensitivity_settings": {
        "max_deltas": 256,
        "overall_sens": 0.64,
        "sens_scaling": 0.0,
        "jitter_strength": 0.0,
        "overshoot_strength": 0.0,
        "overshoot_chance": 0.0,
    },
    "targeting_settings": {
        "target_cls_id": 0,
        "crosshair_cls_id": 2,
        "head_toggle": True,
        "model_predict_crosshair": False,
        "hsv_settings": {
            "enabled": True,
            "center_crop": [240, 240],
            "voting_scheme": "heuristic_spam",
            "bypass_tracker": True,
        },
        "predict_drop": True,
        "lead_target": True,
        "prioritize_oldest": False,
        "zoom": 1.2,
        "projectile_velocity": 2870.0,
        "zoom_interpolation_frames": 69,
        "base_head_offset": 0.35,
        "min_frames_to_target": 2,
    },
    "tracker_settings": {"tracker_impl": "cpp"},
    "gui_settings": {
        "opencv_render": False,
        "dpg_overlay": False,
        "only_render_overlay_non_ads": True,
    },
}


def default_cfg() -> dict:
    """Fresh deep copy so a test mutating its cfg can't leak into another."""
    return copy.deepcopy(_CFG)


def key_paths(d: dict, prefix: str = "") -> set[str]:
    """Flatten to dotted key paths, skipping the _doc*/IMPORTANT* comment keys."""
    out = set()
    for k, v in d.items():
        if k.startswith("_doc") or k.startswith("IMPORTANT"):
            continue
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out |= key_paths(v, path + ".")
        else:
            out.add(path)
    return out
