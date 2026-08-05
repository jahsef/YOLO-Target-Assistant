"""HSV mask kernels vs an independent numpy reference.

The reference mirrors the CUDA integer math exactly, including C truncation-toward-
zero on the hue divides (numpy // floors, which differs for negative numerators).
Exact equality is the bar — any disagreement is a real semantic drift.
"""

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from src.aimbot.engine.hsv_crosshair import (  # noqa: E402
    _HSV_COLOR_CENTER,
    _HSV_COLOR_RANGE,
    _HSV_S_MAX,
    _HSV_S_MIN,
    _HSV_V_MAX,
    _HSV_V_MIN,
    _HS_RATIO_THRESHOLD,
    cupy_red_mask,
    fused_red_mask_suppress,
)
from tests.support.fakes import crosshair_frame, nametag_bar  # noqa: E402

pytestmark = pytest.mark.gpu

HSV_ARGS = dict(color_center=_HSV_COLOR_CENTER, color_range=_HSV_COLOR_RANGE,
                s_min=_HSV_S_MIN, v_min=_HSV_V_MIN, s_max=_HSV_S_MAX, v_max=_HSV_V_MAX)


def _trunc_div(num, den):
    """C integer division: truncates toward zero. numpy // floors."""
    return np.fix(num / den).astype(np.int32)


def ref_red_mask(rgb, color_center, color_range, s_min, v_min, s_max=255, v_max=255):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)

    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    v = mx
    s = np.where(mx == 0, 0, _trunc_div(diff * 255, np.maximum(mx, 1)))

    safe = np.maximum(diff, 1)
    h = np.where(mx == r, _trunc_div(30 * (g - b), safe),
                 np.where(mx == g, 60 + _trunc_div(30 * (b - r), safe),
                          120 + _trunc_div(30 * (r - g), safe)))
    h = np.where(h < 0, h + 180, h)

    d = np.abs(h - color_center)
    d = np.where(d > 90, 180 - d, d)

    return ((v >= v_min) & (v <= v_max) & (s >= s_min) & (s <= s_max)
            & (diff != 0) & (2 * d <= color_range))


def ref_row_suppress(mask, ratio=_HS_RATIO_THRESHOLD):
    """row_sum[y] > ratio * col_sum[x] -> kill. float32 compare, matching the kernel."""
    row_sum = mask.sum(axis=1).astype(np.float32)
    col_sum = mask.sum(axis=0).astype(np.float32)
    kill = row_sum[:, None] > (np.float32(ratio) * col_sum[None, :])
    return mask & ~kill


def _mixed_frame(h=240, w=240, seed=0):
    """Noise + reticle + nametag bar: exercises every branch of the hue math."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    frame = crosshair_frame(h, w, h // 2, w // 2, seed=seed, noise=False) | (frame // 4)
    nametag_bar(frame, h // 2 - 30, w // 2)
    return frame


class TestCupyRedMask:
    def test_matches_reference_on_full_random_gamut(self):
        """Every (r,g,b) branch, including hue wraparound and diff==0 greys."""
        rng = np.random.default_rng(7)
        frame = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        frame[0, :64] = 128  # force some diff == 0 pixels
        got = cp.asnumpy(cupy_red_mask(cp.asarray(frame), **HSV_ARGS))
        np.testing.assert_array_equal(got, ref_red_mask(frame, **HSV_ARGS))

    def test_matches_reference_on_realistic_frame(self):
        frame = _mixed_frame()
        got = cp.asnumpy(cupy_red_mask(cp.asarray(frame), **HSV_ARGS))
        np.testing.assert_array_equal(got, ref_red_mask(frame, **HSV_ARGS))

    def test_exhaustive_over_a_hue_sweep(self):
        """Sweep every hue at the reticle's saturation/value so boundary rounding
        can't drift silently."""
        import colorsys
        rows = []
        for hue_deg in range(360):
            r, g, b = colorsys.hsv_to_rgb(hue_deg / 360.0, 0.94, 0.87)
            rows.append([int(r * 255), int(g * 255), int(b * 255)])
        frame = np.asarray(rows, dtype=np.uint8).reshape(360, 1, 3)
        got = cp.asnumpy(cupy_red_mask(cp.asarray(frame), **HSV_ARGS))
        np.testing.assert_array_equal(got, ref_red_mask(frame, **HSV_ARGS))

    def test_nametag_reds_pass_the_mask_too(self):
        """The three colors in hsv_crosshair.py's header comment. The reticle passes,
        but so do BOTH nametag reds — s lands at 164/165 against s_min=160. The color
        gate alone does not separate them, which is exactly why row-suppression +
        opening + density downsampling exist downstream. Pinned so a future threshold
        tweak that 'fixes' this is a visible decision."""
        px = np.asarray([[[222, 40, 14], [184, 75, 65], [208, 84, 73]]], dtype=np.uint8)
        got = cp.asnumpy(cupy_red_mask(cp.asarray(px), **HSV_ARGS))
        assert got[0, 0], "game reticle red must survive the mask"
        assert got[0, 1] and got[0, 2], "nametag reds currently survive the color gate"

    def test_batch_dim_round_trips(self):
        frame = _mixed_frame(64, 64)
        single = cupy_red_mask(cp.asarray(frame), **HSV_ARGS)
        batched = cupy_red_mask(cp.asarray(frame[None]), **HSV_ARGS)
        assert single.shape == (64, 64)
        assert batched.shape == (1, 64, 64)
        np.testing.assert_array_equal(cp.asnumpy(single), cp.asnumpy(batched)[0])

    def test_returns_bool(self):
        out = cupy_red_mask(cp.asarray(_mixed_frame(32, 32)), **HSV_ARGS)
        assert out.dtype == cp.bool_


class TestFusedMaskSuppress:
    def test_matches_mask_then_suppress_reference(self):
        frame = _mixed_frame()
        got = cp.asnumpy(fused_red_mask_suppress(cp.asarray(frame), **HSV_ARGS,
                                                 ratio=_HS_RATIO_THRESHOLD))
        want = ref_row_suppress(ref_red_mask(frame, **HSV_ARGS), _HS_RATIO_THRESHOLD)
        np.testing.assert_array_equal(got, want)

    def test_suppression_kills_a_wide_thin_bar(self):
        """The nametag case the heuristic exists for: a 1px-tall, 200px-wide red bar
        has huge row sums and tiny column sums, so every pixel of it dies."""
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        frame[100, 20:220] = (222, 40, 14)
        raw = cp.asnumpy(cupy_red_mask(cp.asarray(frame), **HSV_ARGS))
        fused = cp.asnumpy(fused_red_mask_suppress(cp.asarray(frame), **HSV_ARGS))
        assert raw.sum() == 200
        assert fused.sum() == 0

    def test_suppression_spares_a_compact_reticle(self):
        frame = crosshair_frame(240, 240, 120, 120, noise=False)
        fused = cp.asnumpy(fused_red_mask_suppress(cp.asarray(frame), **HSV_ARGS))
        assert fused.sum() > 0, "a symmetric plus must survive row-suppression"

    def test_strided_view_equals_contiguous_copy(self):
        """The pitch fast-path consumes a slice of a bigger frame without copying.
        A wrong pitch silently reads the wrong pixels — this is the guard."""
        full = _mixed_frame(480, 480, seed=3)
        view = cp.asarray(full)[100:340, 60:300]
        assert not view.flags.c_contiguous
        got = cp.asnumpy(fused_red_mask_suppress(view, **HSV_ARGS))
        want = cp.asnumpy(fused_red_mask_suppress(cp.ascontiguousarray(view), **HSV_ARGS))
        np.testing.assert_array_equal(got, want)

    def test_negative_stride_view_falls_back_correctly(self):
        """A channel-flipped (BGR->RGB) view can't be described by a pitch, so the
        function must copy rather than misread."""
        full = _mixed_frame(240, 240, seed=4)
        flipped = cp.asarray(full[..., ::-1])
        got = cp.asnumpy(fused_red_mask_suppress(flipped, **HSV_ARGS))
        want = ref_row_suppress(ref_red_mask(full[..., ::-1].copy(), **HSV_ARGS))
        np.testing.assert_array_equal(got, want)

    def test_writes_into_caller_buffer(self):
        frame = _mixed_frame(240, 240)
        buf = cp.empty((240, 240), dtype=cp.uint8)
        out = fused_red_mask_suppress(cp.asarray(frame), **HSV_ARGS, out=buf)
        assert out.data.ptr == buf.data.ptr, "hot path must not allocate"

    def test_empty_mask_frame_is_all_false(self):
        frame = np.zeros((240, 240, 3), dtype=np.uint8)
        frame[..., 2] = 255  # pure blue
        got = cp.asnumpy(fused_red_mask_suppress(cp.asarray(frame), **HSV_ARGS))
        assert got.sum() == 0
