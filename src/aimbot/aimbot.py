
import time
import sys
from ultralytics.utils.ops import xyxy2xywh
import numpy as np
import json



from . import bootstrap
from .data_parsing import targetselector
from .engine.detection_pipeline import DetectionPipeline
from .engine.tracker_adapter import crosshair_rows_to_tracked
from .gui import gui_manager
from .utils import fpstracker

import traceback
import argparse
from .utils.utils import log
import logging

class Aimbot:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.cfg = json.load(f)
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s',
            force=True
        )
        # set our app logger to the configured level (root stays INFO to suppress betterercam debug spam)
        logging.getLogger('aimbot').setLevel(logging.getLevelNamesMapping()[self.cfg['logging']['logging_level']])
        log("Aimbot: Initializing...", "INFO")
        bootstrap.validate_targeting_config(self.cfg)
        self.pipeline = DetectionPipeline(self.cfg)
        self.base_hw_capture = self.pipeline.base_hw_capture
        log("DetectionPipeline initialized", "INFO")
        self.screen_x, self.screen_y = bootstrap.get_screen_dims(self.cfg)
        self.mousemover = bootstrap.create_mousemover(self.cfg)
        self.inputdetector = bootstrap.create_inputdetector(self.cfg)
        log("input initialized", "INFO")

        gui_cfg = self.cfg['gui_settings']
        if gui_cfg['opencv_render'] or gui_cfg['dpg_overlay']:
            self.gui_manager = gui_manager.GUI_Manager(config = self.cfg,hw_capture = self.base_hw_capture)
            log("GUI_Manager initialized", "INFO")
        else:
            # None skips the per-frame render call entirely
            self.gui_manager = None
            log("GUI disabled (no renderers enabled)", "INFO")


        self.fps_tracker = fpstracker.FPSTracker()
        self.camera = bootstrap.create_camera((self.screen_x, self.screen_y), self.base_hw_capture)
        log("camera initialized", "INFO")

        self.target_selector = targetselector.TargetSelector(
            cfg=self.cfg,
            detection_window_dim=self.base_hw_capture,
            screen_hw=(self.screen_y, self.screen_x),
            fps_tracker = self.fps_tracker
        )
        log("target_selector initialized", "INFO")
        self.tracker = bootstrap.create_tracker(self.cfg, self.pipeline.base_model.model_ext)
        log("tracker initialized", "INFO")

        self._frame_count: int = 0

    def main(self):
        log("Entering main loop", "INFO")
        try:
            while True:
                aimbot_active = (self.inputdetector.is_toggled and (self.inputdetector.is_rmb_pressed)) or not self.cfg['input_settings']['right_click_toggle']

                if not aimbot_active:
                    #throttling so when scanning we dont use all resources
                    time.sleep(self.cfg['other']['inactive_throttle_ms'] / 1000)

                frame = self.camera.grab()
                #capture lib sometimes may return none
                if frame is None:
                    continue

                results, bypass_crosshair_rows = self.pipeline.run(
                    frame,
                    ads=self.inputdetector.is_rmb_pressed,
                    locked=self.target_selector._prev_detection,
                    locked_lifetime=self.target_selector._prev_detection_lifetime,
                )

                results[:,0:4] = xyxy2xywh(results[:,0:4])
                self.tracker.update(results) # expects (N, 6) [x, y, w, h, conf, cls]
                self.tracker.multi_predict(tracks = None) # ultralytics expects stracks, our custom impl uses internal state (tracks arg unused)
                tracked_detections = self.tracker.get_active_tracks_with_lifetime() # returns (M, 10) [x1,y1,x2,y2,track_id,score,cls,idx,start_frame,last_frame]
                if bypass_crosshair_rows.shape[0]:
                    tracked_detections = np.concatenate([tracked_detections, crosshair_rows_to_tracked(bypass_crosshair_rows)], axis=0)
                # refresh routing state every frame from the freshest tracker output, independent of
                # whether aimbot is firing. without this, precision_sr can get stuck if the small target
                # is lost or replaced — see TargetSelector.update_prev_detection for the full reason.
                self.target_selector.update_prev_detection(tracked_detections)
                self._frame_count += 1

                # update tracker max_time_lost with real fps every 60 frames
                if self._frame_count % 60 == 0 and len(self.fps_tracker.buffer) == self.fps_tracker.fps_buffer_len:
                    real_fps = self.fps_tracker.get_fps()
                    self.tracker.max_time_lost = int(real_fps / 30.0 * self.tracker.args.track_buffer)

                raw_deltas = (0,0)
                scaled_deltas = (0,0)
                if self.inputdetector.is_rmb_pressed:
                    self.target_selector.update_zoom_interpolation()#while right clicking we interpolate zoom till final zoom level
                else:
                    self.target_selector.reset_zoom()

                if aimbot_active and len(tracked_detections) > 0:
                    raw_deltas, scaled_deltas = self.aimbot(tracked_detections)

                if self.cfg['targeting_settings']['lead_target']:
                    self.target_selector.update_movement_buffer(raw_deltas)

                self.fps_tracker.update()
                if self.cfg['logging']['print_fps']:
                    self.fps_tracker.print_fps()

                if self.gui_manager:
                    self.gui_manager.render(frame = frame,
                                            tracked_detections = tracked_detections,
                                            is_rmb_pressed= self.inputdetector.is_rmb_pressed,
                                            raw_deltas = raw_deltas,
                                            scaled_deltas = scaled_deltas)

        except KeyboardInterrupt:
            log("\nShutting down...", "INFO")
            self.cleanup()
            sys.exit(0)  # Clean exit

        except Exception as e:
            log(f"Fatal error: {traceback.format_exc()}", "ERROR")
            self.cleanup()
            sys.exit(1)

    def aimbot(self, detections: np.ndarray):
        """
        Args:
            detections: (n, 10) array [x1, y1, x2, y2, track_id, score, cls, idx, start_frame, last_frame]
                        Typically from BYTETracker (BetterBYTETracker lib).get_active_tracks_with_lifetime()
        """
        min_age = self.cfg['targeting_settings']['min_frames_to_target']
        lifetimes = detections[:, 9] - detections[:, 8]  # last_frame - start_frame
        filtered_detections = detections[lifetimes >= min_age]

        raw_deltas = (0,0)
        scaled_deltas = (0,0)
        if len(filtered_detections) > 0:
            raw_deltas = self.target_selector.get_deltas(filtered_detections)
            if raw_deltas != (0,0):
                scaled_deltas = self.mousemover.move_mouse_humanized(raw_deltas[0],raw_deltas[1])
        return raw_deltas, scaled_deltas


    def cleanup(self):
        # Python GC + process exit reclaim GPU memory and module state on their own.
        # Only two handles are worth releasing explicitly:
        #   - camera.release(): betterercam holds OS capture handles (DXGI/nvidia),
        #     releasing avoids leaving them dangling if we ever do an in-process restart.
        #   - gui_manager.cleanup(): destroys cv2 windows / DPG context cleanly.
        log("STARTING CLEANUP", "INFO")
        try:
            if getattr(self, 'camera', None):
                log('Releasing camera', "INFO")
                self.camera.release()
                self.camera = None
            if getattr(self, 'gui_manager', None):
                log('Cleaning up GUI', "INFO")
                self.gui_manager.cleanup()
                self.gui_manager = None
        except Exception as e:
            log(f"Cleanup error: {e}", "ERROR")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11 Aimbot")
    parser.add_argument('--config', type=str, default='config/cfg.json', help='Path to the configuration file')
    args = parser.parse_args()
    log("About to create Aimbot instance and run main", "INFO")
    Aimbot(args.config).main()
