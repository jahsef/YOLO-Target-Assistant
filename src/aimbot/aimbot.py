
import time
import sys
import threading
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


class _Slot:
    """Latest-wins handoff between pipeline stages.

    The producer never blocks and never queues: a frame that arrives while the
    previous one is still unread REPLACES it. Queueing would be strictly worse here â€”
    a backlog only lets you aim at where the target used to be. `dropped` counts how
    often the consumer was too slow, which is the useful signal.
    """

    def __init__(self):
        self._cv = threading.Condition()
        self._item = None
        self._closed = False
        self.dropped = 0

    def put(self, item):
        with self._cv:
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self._cv.notify_all()

    def get(self, timeout=0.25):
        """Returns the newest item, or None on timeout/close."""
        with self._cv:
            while self._item is None and not self._closed:
                if not self._cv.wait(timeout):
                    return None
            item, self._item = self._item, None
            self._cv.notify_all()
            return item

    def wait_free(self, timeout=0.25) -> bool:
        """Block until nothing is sitting unread. Lets a producer skip work whose
        result would only be dropped."""
        with self._cv:
            while self._item is not None and not self._closed:
                if not self._cv.wait(timeout):
                    return False
            return True

    def close(self):
        with self._cv:
            self._closed = True
            self._cv.notify_all()


class Aimbot:
    EMA_ALPHA = 0.2       # smoothing for the stage-duration estimates async capture times against
    CAPTURE_LEAD_S = 5e-4  # grab this much early; being late stalls the GPU stage, being early only costs staleness

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
        # guards the (_prev_detection, _prev_detection_lifetime) pair, which the detect
        # stage reads and the aim stage writes. only contended in async mode.
        self._route_lock = threading.Lock()
        self._stop = threading.Event()
        self._worker_error = None
        # rolling stage durations, seconds. 0 means "no estimate yet", which makes
        # async capture grab immediately until it has measured a couple of frames.
        self._detect_ema = 0.0
        self._grab_ema = 0.0
        self._starve_ema = 0.0
        self._lmb_was_down = False

    # --- pipeline stages ------------------------------------------------------
    # main() runs these back to back on one thread; main_async() runs them on three.
    # Keep them side-effect-compatible so the two loops stay behaviorally identical.

    def _is_active(self) -> bool:
        return (self.inputdetector.is_toggled and self.inputdetector.is_rmb_pressed) \
            or not self.cfg['input_settings']['right_click_toggle']

    def _stage_capture(self, own_pixels: bool):
        """Grab a frame. own_pixels copies it because betterercam cycles a small ring
        of buffers (max_buffer_len=2) â€” without a copy the capture thread would
        overwrite pixels the detect stage is still reading."""
        frame = self.camera.grab()
        # capture lib sometimes may return none
        if frame is None:
            return None
        return frame.copy() if own_pixels else frame

    def _stage_detect(self, frame):
        """GPU half: SR routing + inference + HSV. Returns (results_xywh, bypass_rows)."""
        with self._route_lock:
            locked = self.target_selector._prev_detection
            locked_lifetime = self.target_selector._prev_detection_lifetime
        results, bypass_crosshair_rows = self.pipeline.run(
            frame,
            ads=self.inputdetector.is_rmb_pressed,
            locked=locked,
            locked_lifetime=locked_lifetime,
        )
        results[:, 0:4] = xyxy2xywh(results[:, 0:4])
        return results, bypass_crosshair_rows

    def _stage_track(self, results, bypass_crosshair_rows):
        self.tracker.update(results)  # expects (N, 6) [x, y, w, h, conf, cls]
        self.tracker.multi_predict(tracks=None)  # ultralytics expects stracks, our custom impl uses internal state (tracks arg unused)
        tracked_detections = self.tracker.get_active_tracks_with_lifetime()  # returns (M, 10) [x1,y1,x2,y2,track_id,score,cls,idx,start_frame,last_frame]
        if bypass_crosshair_rows.shape[0]:
            tracked_detections = np.concatenate([tracked_detections, crosshair_rows_to_tracked(bypass_crosshair_rows)], axis=0)
        return tracked_detections

    def _stage_aim(self, tracked_detections, aimbot_active):
        # refresh routing state every frame from the freshest tracker output, independent of
        # whether aimbot is firing. without this, the sr path can get stuck if the small target
        # is lost or replaced â€” see TargetSelector.update_prev_detection for the full reason.
        with self._route_lock:
            self.target_selector.update_prev_detection(tracked_detections)
        self._frame_count += 1

        # update tracker max_time_lost with real fps every 60 frames
        if self._frame_count % 60 == 0 and len(self.fps_tracker.buffer) == self.fps_tracker.fps_buffer_len:
            real_fps = self.fps_tracker.get_fps()
            self.tracker.max_time_lost = int(real_fps / 30.0 * self.tracker.args.track_buffer)

        raw_deltas = (0, 0)
        scaled_deltas = (0, 0)
        if self.inputdetector.is_rmb_pressed:
            self.target_selector.update_zoom_interpolation()  # while right clicking we interpolate zoom till final zoom level
        else:
            self.target_selector.reset_zoom()

        if aimbot_active and len(tracked_detections) > 0:
            raw_deltas, scaled_deltas = self.aimbot(tracked_detections)

        if self.cfg['targeting_settings']['lead_target']:
            self.target_selector.update_movement_buffer(raw_deltas)

        self._apply_recoil_pull()

        self.fps_tracker.update()
        if self.cfg['logging']['print_fps']:
            self.fps_tracker.print_fps()
        return raw_deltas, scaled_deltas

    def _apply_recoil_pull(self) -> None:
        """Pull down once per LMB press while ADS. Rising edge only — level-triggered
        would pull every frame the button is held. Spread out by the mouse thread if it
        is on, otherwise it goes out as one step.

        The edge is tracked even on frames that don't pull, so clicking without ADS and
        then holding both doesn't fire a pull without a fresh click.
        """
        px = self.cfg['input_settings'].get('on_click_pull_down_px', 0)
        if not px:
            return
        down = self.inputdetector.is_lmb_pressed
        if (down and not self._lmb_was_down
                and self.inputdetector.is_toggled
                and self.inputdetector.is_rmb_pressed):
            self.mousemover.pull(0.0, float(px))
        self._lmb_was_down = down

    def _stage_render(self, frame, tracked_detections, raw_deltas, scaled_deltas):
        if self.gui_manager:
            self.gui_manager.render(frame=frame,
                                    tracked_detections=tracked_detections,
                                    is_rmb_pressed=self.inputdetector.is_rmb_pressed,
                                    raw_deltas=raw_deltas,
                                    scaled_deltas=scaled_deltas)

    # --- loops ----------------------------------------------------------------

    def main(self):
        if self.cfg['other'].get('async_pipeline', False):
            return self.main_async()
        return self.main_serial()

    def main_serial(self):
        log("Entering main loop (serial)", "INFO")
        try:
            while True:
                aimbot_active = self._is_active()
                if not aimbot_active:
                    # throttling so when scanning we dont use all resources
                    time.sleep(self.cfg['other']['inactive_throttle_ms'] / 1000)

                frame = self._stage_capture(own_pixels=False)
                if frame is None:
                    continue

                results, bypass = self._stage_detect(frame)
                tracked_detections = self._stage_track(results, bypass)
                raw_deltas, scaled_deltas = self._stage_aim(tracked_detections, aimbot_active)
                self._stage_render(frame, tracked_detections, raw_deltas, scaled_deltas)

        except KeyboardInterrupt:
            log("\nShutting down...", "INFO")
            self.cleanup()
            sys.exit(0)  # Clean exit

        except Exception:
            log(f"Fatal error: {traceback.format_exc()}", "ERROR")
            self.cleanup()
            sys.exit(1)

    def main_async(self):
        """Three-stage pipeline: capture | detect (GPU) | aim (CPU).

        All GPU work stays on the detect thread on purpose â€” a TensorRT execution
        context is not safe to drive from two threads, and keeping one owner also keeps
        kernel ordering predictable. Capture and the CPU tail are what actually overlap.

        What this is worth depends almost entirely on what camera.grab() costs, since
        the loop is otherwise GPU-bound (~4 ms of TRT against a ~0.2 ms CPU tail). With
        a cheap grab it's ~+6% fps for +0.1 ms p50; the more grab costs, the more there
        is to overlap (~+26% at 1.7 ms, ~+46% at 4 ms). Serial is the default.
        """
        log("Entering main loop (async, 3-stage)", "INFO")
        frames = _Slot()
        detections = _Slot()
        threads = [
            threading.Thread(target=self._capture_loop, args=(frames,), name="capture", daemon=True),
            threading.Thread(target=self._detect_loop, args=(frames, detections), name="detect", daemon=True),
        ]
        try:
            for t in threads:
                t.start()
            self._aim_loop(detections)  # runs on the main thread so Ctrl+C lands here
        except KeyboardInterrupt:
            log("\nShutting down...", "INFO")
        except Exception:
            log(f"Fatal error: {traceback.format_exc()}", "ERROR")
        finally:
            self._stop.set()
            frames.close()
            detections.close()
            for t in threads:
                t.join(timeout=1.0)
            self.cleanup()
        if self._worker_error is not None:
            log(f"Worker thread died: {self._worker_error}", "ERROR")
            sys.exit(1)
        sys.exit(0)

    def _guarded(self, fn):
        """Run a worker loop, recording the first failure and stopping everything."""
        try:
            fn()
        except Exception:
            self._worker_error = traceback.format_exc()
            log(f"Fatal error in {threading.current_thread().name}: {self._worker_error}", "ERROR")
            self._stop.set()

    def _capture_loop(self, frames: _Slot):
        """Grabs just-in-time: wait for the slot to drain, then hold off until the
        detect stage is about to come free, so the grab lands as late as possible.

        Free-running instead would be worse in both directions. Capture at 600 Hz into
        a ~200 Hz pipeline means two thirds of the grabs are thrown away, and each one
        still cost GPU and GIL time that detect wanted. Grabbing the moment the slot
        drains avoids the waste but the frame then ages through the whole GPU stage
        before anyone looks at it. Waiting costs nothing and every ms spent waiting is
        a ms of staleness not added to the frame.
        """
        def body():
            while not self._stop.is_set():
                if not self._is_active():
                    time.sleep(self.cfg['other']['inactive_throttle_ms'] / 1000)
                if not frames.wait_free():
                    continue
                # Only worth waiting if the PIPELINE is the bottleneck. When the camera
                # is the slow one we already spend time polling for a frame, and
                # sleeping on top of that just misses frames â€” subtract that starve
                # time so the lag collapses to zero as the camera becomes the limit.
                lag = (self._detect_ema - self._grab_ema - self._starve_ema
                       - self.CAPTURE_LEAD_S)
                if lag > 0:
                    time.sleep(lag)
                # betterercam returns None until the next frame is ready; keep asking
                frame = None
                t_start = time.perf_counter()
                while frame is None and not self._stop.is_set():
                    t0 = time.perf_counter()
                    frame = self._stage_capture(own_pixels=True)
                    if frame is not None:
                        self._grab_ema += self.EMA_ALPHA * ((time.perf_counter() - t0) - self._grab_ema)
                if frame is not None:
                    starved = max(0.0, (time.perf_counter() - t_start) - self._grab_ema)
                    self._starve_ema += self.EMA_ALPHA * (starved - self._starve_ema)
                    frames.put(frame)
        self._guarded(body)

    def _detect_loop(self, frames: _Slot, detections: _Slot):
        def body():
            while not self._stop.is_set():
                frame = frames.get()
                if frame is None:
                    continue
                t0 = time.perf_counter()
                results, bypass = self._stage_detect(frame)
                # capture times its grab against this
                self._detect_ema += self.EMA_ALPHA * ((time.perf_counter() - t0) - self._detect_ema)
                detections.put((frame, results, bypass))
        self._guarded(body)

    def _aim_loop(self, detections: _Slot):
        while not self._stop.is_set():
            item = detections.get()
            if item is None:
                continue
            frame, results, bypass = item
            tracked_detections = self._stage_track(results, bypass)
            raw_deltas, scaled_deltas = self._stage_aim(tracked_detections, self._is_active())
            self._stage_render(frame, tracked_detections, raw_deltas, scaled_deltas)

    def aimbot(self, detections: np.ndarray):
        """
        Args:
            detections: (n, 10) array [x1, y1, x2, y2, track_id, score, cls, idx, start_frame, last_frame]
                        Typically from BYTETracker (BetterBYTETracker lib).get_active_tracks_with_lifetime()
        """
        min_age = self.cfg['targeting_settings']['min_frames_to_target']
        lifetimes = detections[:, 9] - detections[:, 8]  # last_frame - start_frame
        # min_age gates TRACK stability. rows that skipped the tracker (track_id -1, see
        # hsv_settings.bypass_tracker) have no lifetime to speak of â€” they are direct
        # per-frame measurements, and the reticle is one of them. gating those would drop
        # the crosshair before _get_crosshair ever sees it, silently falling back to the
        # window center.
        untracked = detections[:, 4] < 0
        filtered_detections = detections[(lifetimes >= min_age) | untracked]

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
            if getattr(self, 'inputdetector', None):
                self.inputdetector.stop()
            mover = getattr(self, 'mousemover', None)
            if mover is not None and hasattr(mover, 'stop'):
                log('Stopping mouse thread', "INFO")
                mover.stop()
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
