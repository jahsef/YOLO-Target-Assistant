import json
from pathlib import Path
from argparse import Namespace

class ConfigLoader:
    def __init__(self, config_path):
        self.cfg = self._load_config_file(config_path)
        self._load_other_config()
        self._load_aim_config()

    def _load_config_file(self, config_path):
        with open(config_path) as f:
            return json.load(f)
    
    def _load_other_config(self):
        #logging
        self.debug = self.cfg['logging']['debug']
        self.is_fps_tracked = self.cfg['logging']['fps']
        self.inactive_throttle_ms = self.cfg['other']['inactive_throttle_ms']
        #input settings
        self.toggle_hotkey = self.cfg['input_settings']['toggle_hotkey']
        self.right_click_toggle = self.cfg['input_settings']['right_click_toggle']
        
    def _load_aim_config(self):
        #sens settings
        self.overall_sens = self.cfg['sensitivity_settings']['overall_sens']
        self.sens_scaling = self.cfg['sensitivity_settings']['sens_scaling']
        self.max_deltas = self.cfg['sensitivity_settings']['max_deltas']
        self.jitter_strength = self.cfg['sensitivity_settings']['jitter_strength']
        self.overshoot_strength = self.cfg['sensitivity_settings']['overshoot_strength']
        self.overshoot_chance = self.cfg['sensitivity_settings']['overshoot_chance']
        #targeting/ bullet prediction settings
        self.target_cls_id = self.cfg['targeting_settings']['target_cls_id']
        self.crosshair_cls_id = self.cfg['targeting_settings']['crosshair_cls_id']
        self.head_toggle = self.cfg['targeting_settings']['head_toggle']
        self.predict_drop = self.cfg['targeting_settings']['predict_drop']
        self.predict_crosshair = self.cfg['targeting_settings']['predict_crosshair']
        self.zoom = self.cfg['targeting_settings']['zoom']
        self.projectile_velocity = self.cfg['targeting_settings']['projectile_velocity']
        self.base_head_offset = self.cfg['targeting_settings']['base_head_offset']
        self.fov = self.cfg['targeting_settings']['fov']

    def get_tracker_args(self, target_frame_rate):
        return Namespace(
            track_high_thresh=self.cfg['tracker_settings']['track_high_thresh'],
            track_low_thresh=self.cfg['tracker_settings']['track_low_thresh'],
            track_buffer=int(target_frame_rate * self.cfg['tracker_settings']['track_buffer_multiplier']),
            fuse_score=self.cfg['tracker_settings']['fuse_score'],
            match_thresh=self.cfg['tracker_settings']['match_thresh'],
            new_track_thresh=self.cfg['tracker_settings']['new_track_thresh']
        )
