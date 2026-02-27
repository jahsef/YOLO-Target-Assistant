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
N_SAMPLES     = 512     # number of independent sprints
N_FRAMES      = 64      # iterations per sprint (GPU)
CPU_FRACTION  = 1.0     # CPU sprint length = CPU_FRACTION * N_FRAMES
BATCH_SIZE    = 4       # batch size for batched section
SIZES         = [(640, 640)]


def make_fake_frame_np(h, w):
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

def make_fake_frame_torchgpu(h, w):
    return torch.zeros(size = (h, w, 3), dtype=torch.uint8,device = torch.device("cuda"))

def make_fake_frame_cupy(h, w):
    a = make_fake_frame_torchgpu(h,w)
    return cp.asarray(a)

def timer(fn, warmup, n_samples, iters, gpu_sync=True):
    for _ in range(warmup):
        fn()
    if gpu_sync:
        torch.cuda.synchronize()

    sprint_means = np.empty(n_samples)
    for s in range(n_samples):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        if gpu_sync:
            torch.cuda.synchronize()
        sprint_means[s] = (time.perf_counter() - t0) / iters * 1e6

    mean = sprint_means.mean()
    std  = sprint_means.std(ddof=1)
    ci95 = 1.96 * std / np.sqrt(n_samples)
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


def bench(label, fn, iters, n_samples, gpu_sync, baseline_mean=None):
    mean, std, ci95 = timer(fn, WARMUP, n_samples, iters, gpu_sync)
    speedup = f"  {baseline_mean/mean:.2f}x" if baseline_mean is not None else ""
    print(f"  {label:<34}: {mean:7.2f} ±{std:6.2f} µs  CI95=[{mean-ci95:.2f}, {mean+ci95:.2f}]{speedup}")
    return mean


def run(h, w, device):
    gpu = device.type == "cuda"
    iters = N_FRAMES if gpu else max(1, int(N_FRAMES * CPU_FRACTION))
    frame_np = make_fake_frame_np(h, w)
    frame_torchgpu = make_fake_frame_torchgpu(h, w)
    frame_cp = make_fake_frame_cupy(h, w)

    # batched inputs
    batch_np = [make_fake_frame_np(h, w) for _ in range(BATCH_SIZE)]
    batch_cp = [make_fake_frame_cupy(h, w) for _ in range(BATCH_SIZE)]

    # ── single frame ──────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {h}x{w}  |  {device}  |  batch=1  |  warmup={WARMUP}  samples={N_SAMPLES}  iters/sample={iters}")
    print(f"{'='*72}")
    
    ult = bench("Ultralytics",                          lambda: ultralytics_preprocess([frame_np], device),  iters, N_SAMPLES, gpu)
    bench("ultralytics_preprocess2 (tensor ops change)",       lambda: ultralytics_preprocess2([frame_np], device), iters, N_SAMPLES, gpu, ult)
    # bench("ultralytics_preprocess3 (added single frame case)",       lambda: ultralytics_preprocess3([frame_np], device),     iters, N_SAMPLES, gpu, ult)
    bench("ultralytics_preprocess4 (added single frame case)",       lambda: ultralytics_preprocess4([frame_np], device), iters, N_SAMPLES, gpu, ult)
    # bench("torch_pr_nonblock_preprocess (doesnt handle all cases)",       lambda: torch_pr_nonblock_preprocess(frame_np, device),     iters, N_SAMPLES, gpu, ult)
    # bench("torch_pr_nonblock_preprocess2",       lambda: torch_pr_nonblock_preprocess2(frame_np, device),     iters, N_SAMPLES, gpu, ult)
    # bench("torch_pr_nonblock_view",       lambda: torch_pr_nonblock_view(frame_np, device),     iters, N_SAMPLES, gpu, ult)
    # bench("torch_pr_nonblock_view2",       lambda: torch_pr_nonblock_view2(frame_np, device),     iters, N_SAMPLES, gpu, ult)
    # bench("torch_pr_nonblock_view3",       lambda: torch_pr_nonblock_view3(frame_np, device),     iters, N_SAMPLES, gpu, ult)
    # if gpu:
    #     bench("cupy_nobgr5",                            lambda: cupy_nobgr5(frame_cp, device),               iters, N_SAMPLES, gpu, ult)

    # ── batched ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  {h}x{w}  |  {device}  |  batch={BATCH_SIZE}  |  warmup={WARMUP}  samples={N_SAMPLES}  iters/sample={iters}")
    print(f"{'='*72}")

    ult_b = bench("Ultralytics",                        lambda: ultralytics_preprocess(batch_np, device),    iters, N_SAMPLES, gpu)
    bench("ultralytics_preprocess2 (tensor ops change)",       lambda: ultralytics_preprocess2(batch_np, device),   iters, N_SAMPLES, gpu, ult_b)
    bench("ultralytics_preprocess4 (added single frame case)",       lambda: ultralytics_preprocess4(batch_np, device),   iters, N_SAMPLES, gpu, ult_b)
    
if __name__ == "__main__":
    print("Preprocessing benchmark")
    print(f"CUDA: {torch.cuda.get_device_name(0)}")
    for h, w in SIZES:
        run(h, w, torch.device("cuda"))
    for h, w in SIZES:
        run(h, w, torch.device("cpu"))
