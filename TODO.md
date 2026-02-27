
minimum frames to target (currently have the config for this, but doesnt do anything in code) (lower prio tbh)
    n frames before lock, reactivated tracks should not have this requirement
    or just n frames before right clicking (this solves the zoom issue close range, but see other zoom thing)

base zoom + other zoom modes, we can define multiple zoom modes (very nice for usability)
    linear warmup over n frames (user defined too) to full zoom, makes it so the aimbot doesnt flick above a target with high zoom mode as soon as you right click

could add ema momentum (bypass the wma target leading entirely since that uses uncertainty scaling) useful for when a stationary target is going like +-3 pixels. actually might not be that useful since i use tracker to smooth out those stuff anyway. idk
use corner L shapes or crosses for targets rather than full boxes possibly?

  targetselector.py:334: converts a 48-element deque to numpy every frame. Could replace the deque with a pre-allocated (48, 2) numpy ring buffer and a write index. Probably sub-millisecond but it's a pointless allocation on every    
  aiming frame.

tracker from scratch (but actually follow through this time) (would need to do on gpu to make it worth)