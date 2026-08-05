"""Compare pynput's hook against GetAsyncKeyState for right-mouse state, live.

Answers one question: after you physically release RMB, how long does each source keep
reporting 'pressed'? Run it, hold and release right click a bunch of times (in game if
you can, that's the case that matters), then Ctrl+C.

    python -m src.tests.diag_rmb

Reports the lag between the hook seeing a release and the polled state seeing it. A
consistently positive lag on release is the aimbot's "still moving after I let go".
"""

import statistics
import sys
import threading
import time

import win32api
import win32con

try:
    from pynput import mouse
except ImportError:
    print("pynput not installed; install it just for this diagnostic")
    sys.exit(1)

DOWN_BIT = 0x8000
POLL_HZ = 2000

hook_events = []   # (t, pressed)
poll_events = []   # (t, pressed)
stop = threading.Event()


def on_click(x, y, button, pressed):
    if button == mouse.Button.right:
        hook_events.append((time.perf_counter(), pressed))


def poll_loop():
    period = 1.0 / POLL_HZ
    last = None
    while not stop.is_set():
        now_down = bool(win32api.GetAsyncKeyState(win32con.VK_RBUTTON) & DOWN_BIT)
        if now_down != last:
            poll_events.append((time.perf_counter(), now_down))
            last = now_down
        time.sleep(period)


def pair_up(target_state):
    """Match each hook transition to the next poll transition of the same state."""
    lags = []
    pi = 0
    for ht, hs in hook_events:
        if hs != target_state:
            continue
        while pi < len(poll_events) and poll_events[pi][0] < ht - 0.05:
            pi += 1
        for j in range(pi, len(poll_events)):
            pt, ps = poll_events[j]
            if ps == target_state and pt >= ht - 0.05:
                lags.append((pt - ht) * 1e3)
                break
    return lags


def summarize(label, lags):
    if not lags:
        print(f"  {label}: no paired events")
        return
    print(f"  {label}: n={len(lags)}  mean {statistics.mean(lags):+7.2f} ms  "
          f"median {statistics.median(lags):+7.2f}  max {max(lags):+7.2f}  min {min(lags):+7.2f}")


def main():
    listener = mouse.Listener(on_click=on_click)
    listener.start()
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()

    print(f"polling GetAsyncKeyState at {POLL_HZ} Hz alongside the pynput hook.")
    print("hold and release right click ~10 times, then Ctrl+C.\n")
    try:
        while True:
            time.sleep(0.5)
            print(f"\r  hook transitions {len(hook_events):3d}   "
                  f"polled transitions {len(poll_events):3d}", end="")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        listener.stop()
        t.join(timeout=1)

    print("\n\nlag of GetAsyncKeyState behind the pynput hook (positive = polled is later):")
    summarize("on press  ", pair_up(True))
    summarize("on release", pair_up(False))

    stuck = [l for l in pair_up(False) if l > 20.0]
    print()
    if stuck:
        print(f"  {len(stuck)} release(s) where the polled state stayed DOWN >20ms "
              f"(worst {max(stuck):.1f} ms).")
        print("  that is enough to keep the aimbot firing after you let go.")
    else:
        print("  no release lag worth worrying about -- the polled state tracks the")
        print("  hook closely, so a late mouse move is coming from somewhere else.")


if __name__ == "__main__":
    main()
