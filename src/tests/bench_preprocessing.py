"""
Preprocessing benchmark: Ultralytics vs PR candidate paths

Tests:
  1. Ultralytics     - exact current pipeline (numpy ops on CPU -> GPU transfer)
  2. Imm. to device  - Ultralytics numpy ops + non_blocking transfer (stack vs unsqueeze)
  3. Torch PR        - PR candidate: transfer first, BGR->RGB + layout on GPU

Run from repo root:
    python -m src.tests.bench_preprocessing
"""

import time
import numpy as np
import cupy as cp

import torch


WARMUP        = 200
N_FRAMES      = 512     # frames timed per benchmark
BATCH_SIZE    = 16       # batch size for batched section
SIZES         = [(640, 640)]


def make_fake_frame_np(h, w):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

def make_fake_frame_torchgpu(h, w):
    return torch.zeros(size = (h, w, 3), dtype=torch.uint8,device = torch.device("cuda"))

def make_fake_frame_cupy(h, w):
    a = make_fake_frame_torchgpu(h,w)
    return cp.asarray(a)

def timer(fn, warmup, n_frames, gpu_sync=True):
    for _ in range(warmup):
        fn()
    if gpu_sync:
        torch.cuda.synchronize()

    samples = np.empty(n_frames)
    for i in range(n_frames):
        if gpu_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if gpu_sync:
            torch.cuda.synchronize()
        samples[i] = (time.perf_counter() - t0) * 1e6

    mean = samples.mean()
    std  = samples.std(ddof=1)
    ci95 = 1.96 * std / np.sqrt(n_frames)
    return mean, std, ci95


def ultralytics_preprocess(im, device):
    """Exact Ultralytics pipeline: numpy ops on CPU then transfer."""
    im = np.stack(im)
    im = im[..., ::-1].transpose((0, 3, 1, 2))      # BGR->RGB, BHWC->BCHW
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device).float() / 255.0
    return im

def ultralytics_preprocess2(im, device):
    """Exact Ultralytics pipeline: numpy ops on CPU then transfer."""
    im = np.stack(im)
    im = torch.from_numpy(im).to(device, non_blocking=True)
    im = im.permute(0, 3, 1, 2).flip(1).contiguous().float().div_(255.0)     # BGR->RGB, BHWC->BCHW
    return im

def ultralytics_preprocess3(im:list[np.ndarray], device):
    """Exact Ultralytics pipeline: numpy ops on CPU then transfer."""
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor and len(im) == 1:
        im = im[0][np.newaxis,...] # gets only img then adds batch dim
    else:
        im = np.stack(im)
    im = torch.from_numpy(im).to(device, non_blocking=True)
    im = im.permute(0, 3, 1, 2).flip(1).contiguous().float().div_(255.0)     # BGR->RGB, BHWC->BCHW
    return im

def ultralytics_preprocess4(im:list[np.ndarray], device):
    """Exact Ultralytics pipeline: numpy ops on CPU then transfer."""
    not_tensor = not isinstance(im, torch.Tensor)

    if not_tensor:
        if len(im) == 1:
            im = torch.from_numpy(im[0]).to(device, non_blocking=True).unsqueeze(0)
        else:
            im = np.stack(im)
            im = torch.from_numpy(im).to(device, non_blocking=True)
    
    im = im.permute(0, 3, 1, 2).flip(1).contiguous().float().div_(255.0)     # BGR->RGB, BHWC->BCHW
    return im

def ultralytics_immediate_to_device_stack(frame_np, device):
    """Ultralytics exact code + non_blocking transfer (np.stack batch dim)."""
    im = np.stack([frame_np])
    im = im[..., ::-1].transpose((0, 3, 1, 2))
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device, non_blocking=True).float() / 255.0
    return im


def ultralytics_immediate_to_device_unsqueeze(frame_np, device):
    """Ultralytics exact code + non_blocking transfer (unsqueeze batch dim)."""
    im = np.ascontiguousarray(frame_np[..., ::-1].transpose((2, 0, 1)))
    im = torch.from_numpy(im).unsqueeze(0).to(device, non_blocking=True).float() / 255.0
    return im


def torch_pr_preprocess(frame_np, device):
    """PR candidate: transfer to device immediately, BGR->RGB + layout on GPU."""
    im = torch.from_numpy(frame_np).to(device).unsqueeze(0)
    im = im.permute(0, 3, 1, 2).flip(1).contiguous()
    im = im.float().div_(255.0)
    return im

def torch_pr_nonblock_preprocess(frame_np, device):
    """PR candidate: transfer to device immediately, BGR->RGB + layout on GPU."""
    im = torch.from_numpy(frame_np).to(device,non_blocking=True).unsqueeze(0)
    im = im.permute(0, 3, 1, 2).flip(1).contiguous()
    im = im.float().div_(255.0)
    return im

def torch_pr_nonblock_preprocess2(frame_np, device):
    """PR candidate: transfer to device immediately, BGR->RGB + layout on GPU."""
    im = torch.from_numpy(frame_np).to(device,non_blocking=True)
    im = im.permute(2,0,1).unsqueeze(0).flip(1).contiguous()
    im = im.float().div_(255.0)
    return im

def torch_pr_nonblock_view(frame_np, device):
    """PR nonblock - torch view ops + flip + contiguous."""
    im = torch.from_numpy(frame_np).to(device, non_blocking=True)
    im = im.permute(2, 0, 1).unsqueeze(0).flip(1).contiguous()
    im = im.float().div_(255.0)
    return im

def torch_pr_nonblock_view2(frame_np, device):
    """PR nonblock - torch view ops + flip + contiguous."""
    im = torch.from_numpy(frame_np).to(device, non_blocking=True)
    im = im.permute(2, 0, 1).flip(0).unsqueeze(0).contiguous()
    im = im.float().div_(255.0)
    return im

def torch_pr_nonblock_view3(frame_np, device):
    """PR nonblock - torch view ops + flip + contiguous."""
    im = torch.from_numpy(frame_np).to(device, non_blocking=True)
    im = im.permute(2, 0, 1).flip(0).unsqueeze(0).contiguous()
    im = im.float()
    im /= 255.0
    return im

def torch_pr_nonblock_view_fp16(frame_np, device):
    """PR nonblock - torch view ops + flip + contiguous."""
    im = torch.from_numpy(frame_np).to(device, non_blocking=True)
    im = im.permute(2, 0, 1).unsqueeze(0).flip(1).contiguous()
    im = im.half().div_(255.0)
    return im


def torch_nobgr(frame_np, device):
    """no bgr to rgb conversion"""
    im = torch.from_numpy(frame_np).to(device, non_blocking=True)
    im = im.permute(2, 0, 1).unsqueeze(0).contiguous()
    im = im.float().div_(255.0)
    return im

def torch_nobgr2(frame_torch, device):
    """no bgr to rgb conversion, assumes source frame is on gpu already"""
    frame_torch = frame_torch.permute(2, 0, 1).unsqueeze(0).contiguous()
    frame_torch = frame_torch.float().div_(255.0)
    return frame_torch


def cupy_nobgr(frame_cp:cp.array, device):
    """no bgr to rgb conversion, assumes source frame is on gpu already"""
    # print(type(frame_cp))
    frame_cp = cp.ascontiguousarray(frame_cp.transpose(2, 0, 1)[..., cp.newaxis], dtype = cp.float32)
    frame_cp /= 255.0
    return frame_cp

def cupy_nobgr2(frame_cp:cp.array, device):
    """no bgr to rgb conversion, assumes source frame is on gpu already"""
    # print(type(frame_cp))
    frame_cp = cp.ascontiguousarray(frame_cp.transpose(2, 0, 1)[..., cp.newaxis], dtype = cp.float32)
    cp.true_divide(frame_cp, 255.0, out = frame_cp)
    return frame_cp

def cupy_nobgr3(frame_cp:cp.array, device):
    """no bgr to rgb conversion, assumes source frame is on gpu already"""
    # print(type(frame_cp))
    frame_cp = frame_cp.transpose(2, 0, 1)[cp.newaxis,...]
    frame_cp = cp.ascontiguousarray(frame_cp, dtype = cp.float16)
    cp.true_divide(frame_cp, 255.0, out = frame_cp, dtype = cp.float32)
    
    return frame_cp

def cupy_nobgr4(frame_cp:cp.array, device):
    """no bgr to rgb conversion, assumes source frame is on gpu already"""
    # print(type(frame_cp))
    frame_cp = frame_cp.transpose(2, 0, 1)[cp.newaxis,...]
    frame_cp = cp.ascontiguousarray(frame_cp, dtype = cp.float16)
    cp.true_divide(frame_cp, 255.0, out = frame_cp)
    return frame_cp

def cupy_nobgr5(frame_cp:cp.array, device):
    """no bgr to rgb conversion, assumes source frame is on gpu already"""

    frame_cp = frame_cp.transpose(2, 0, 1)[cp.newaxis,...]
    frame_cp = cp.ascontiguousarray(frame_cp, dtype = cp.float16)
    cp.true_divide(frame_cp, 255.0, out = frame_cp, dtype = cp.float16)
    return frame_cp


# ── fused CuPy kernel: HWC uint8 -> NCHW float32 / 255 in a single pass ──────
# (matches model.py:_preprocess_cp output; no BGR->RGB flip)
_PREPROCESS_KERNEL = cp.RawKernel(r"""
extern "C" __global__
void hwc_u8_to_nchw_f32_div255(const unsigned char* __restrict__ src,
                                float* __restrict__ dst,
                                const int H, const int W) {
    // dst layout: (1, 3, H, W) contiguous -> dst[c*H*W + y*W + x]
    // src layout: (H, W, 3)    contiguous -> src[y*W*3 + x*3 + c]
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z;  // 0,1,2
    if (x >= W || y >= H) return;

    unsigned char v = src[(y * W + x) * 3 + c];
    dst[c * H * W + y * W + x] = (float)v * (1.0f / 255.0f);
}
""", "hwc_u8_to_nchw_f32_div255")


def fused_preprocess(frame: cp.ndarray, device=None) -> cp.ndarray:
    """frame: (H, W, 3) cp.uint8. Returns (1, 3, H, W) cp.float32. Single kernel pass."""
    H, W, _ = frame.shape
    src = cp.ascontiguousarray(frame)  # no-op if already contiguous
    out = cp.empty((1, 3, H, W), dtype=cp.float32)
    block = (32, 8, 1)
    grid = ((W + block[0] - 1) // block[0],
            (H + block[1] - 1) // block[1],
            3)
    _PREPROCESS_KERNEL(grid, block, (src, out, np.int32(H), np.int32(W)))
    return out


def cupy_pipeline_preprocess(frame: cp.ndarray, device=None) -> cp.ndarray:
    """Stock CuPy path (matches model.py:_preprocess_cp): transpose/cast/div, three kernels."""
    frame = frame.transpose(2, 0, 1)[cp.newaxis, ...]
    frame = cp.ascontiguousarray(frame, dtype=cp.float32)
    cp.true_divide(frame, 255.0, out=frame, dtype=cp.float32)
    return frame


def bench(label, fn, n_frames, gpu_sync, baseline_mean=None):
    mean, std, ci95 = timer(fn, WARMUP, n_frames, gpu_sync)
    speedup = f"  {baseline_mean/mean:.2f}x" if baseline_mean is not None else ""
    print(f"  {label:<34}: {mean:7.2f} ±{std:6.2f} µs  CI95=[{mean-ci95:.2f}, {mean+ci95:.2f}]{speedup}")
    return mean


def run(h, w, device):
    gpu = device.type == "cuda"
    frame_np = make_fake_frame_np(h, w)
    frame_torchgpu = make_fake_frame_torchgpu(h, w)
    frame_cp = make_fake_frame_cupy(h, w)

    # batched inputs
    batch_np = [make_fake_frame_np(h, w) for _ in range(BATCH_SIZE)]
    batch_cp = [make_fake_frame_cupy(h, w) for _ in range(BATCH_SIZE)]

    # ── single frame ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {h}x{w}  |  {device}  |  batch=1  |  warmup={WARMUP}  frames={N_FRAMES}")
    print(f"{'='*72}")

    ult = bench("Ultralytics",                          lambda: ultralytics_preprocess([frame_np], device),  N_FRAMES, gpu)
    bench("ultralytics_preprocess2 (tensor ops change)",       lambda: ultralytics_preprocess2([frame_np], device), N_FRAMES, gpu, ult)
    # bench("ultralytics_preprocess3 (added single frame case)",       lambda: ultralytics_preprocess3([frame_np], device),     N_FRAMES, gpu, ult)
    bench("ultralytics_preprocess4 (added single frame case)",       lambda: ultralytics_preprocess4([frame_np], device), N_FRAMES, gpu, ult)
    bench("torch_pr_nonblock_preprocess (doesnt handle all cases)",       lambda: torch_pr_nonblock_preprocess(frame_np, device),     N_FRAMES, gpu, ult)
    bench("torch_pr_nonblock_preprocess2",       lambda: torch_pr_nonblock_preprocess2(frame_np, device),     N_FRAMES, gpu, ult)
    bench("torch_pr_nonblock_view",       lambda: torch_pr_nonblock_view(frame_np, device),     N_FRAMES, gpu, ult)
    bench("torch_pr_nonblock_view2",       lambda: torch_pr_nonblock_view2(frame_np, device),     N_FRAMES, gpu, ult)
    bench("torch_pr_nonblock_view3",       lambda: torch_pr_nonblock_view3(frame_np, device),     N_FRAMES, gpu, ult)
    if gpu:
        bench("cupy_nobgr5",                            lambda: cupy_nobgr5(frame_cp, device),               N_FRAMES, gpu, ult)
        bench("cupy_pipeline (transpose/cast/div)",     lambda: cupy_pipeline_preprocess(frame_cp),          N_FRAMES, gpu, ult)
        bench("fused RawKernel (HWC->NCHW/255)",         lambda: fused_preprocess(frame_cp),                  N_FRAMES, gpu, ult)

    # ── batched ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {h}x{w}  |  {device}  |  batch={BATCH_SIZE}  |  warmup={WARMUP}  frames={N_FRAMES}")
    print(f"{'='*72}")

    ult_b = bench("Ultralytics",                        lambda: ultralytics_preprocess(batch_np, device),    N_FRAMES, gpu)
    bench("ultralytics_preprocess2 (tensor ops change)",       lambda: ultralytics_preprocess2(batch_np, device),   N_FRAMES, gpu, ult_b)
    bench("ultralytics_preprocess4 (added single frame case)",       lambda: ultralytics_preprocess4(batch_np, device),   N_FRAMES, gpu, ult_b)
    
if __name__ == "__main__":
    print("Preprocessing benchmark")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    for h, w in SIZES:
        run(h, w, torch.device("cuda"))
    # for h, w in SIZES:
    #     run(h, w, torch.device("cpu"))
