import os
import tensorrt as trt
import numpy as np
import cupy as cp
import torch

from torchvision.ops import batched_nms


class TensorRT_Engine:
    def __init__(self, engine_file_path: str = None, conf_threshold: float = 0.0, verbose: bool = False, nms: bool = True, iou_threshold: float = 0.7, max_det: int = 300, engine_bytes: bytes = None):
        """_summary_

        Args:
            engine_file_path (str): path to a serialized .engine file. mutually exclusive with engine_bytes.
            engine_bytes (bytes): raw serialized engine (e.g. unpacked from a bundle). mutually exclusive with engine_file_path.
            conf_threshold (float): _description_
            verbose (bool, optional): _description_. Defaults to False.
            nms (bool, optional): applies nms if engine does not already have it (heuristic from shape). defaults to True
        """
        if (engine_file_path is None) == (engine_bytes is None):
            raise ValueError("provide exactly one of engine_file_path or engine_bytes")

        self.TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine(engine_file_path, engine_bytes)
        self.context = self.engine.create_execution_context()

        #now explicitly defines input instead of assuming first idx is input tensor
        self.input_tensor_name = None
        self.output_tensor_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_tensor_name = name
            else:
                self.output_tensor_name = name

        self.output_ptr = 0
        self.conf_threshold = conf_threshold
        self.input_shape = self.engine.get_tensor_shape(self.input_tensor_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_tensor_name)

        self.imgsz = self.input_shape[2:]

        self.iou_threshold = iou_threshold
        self.max_det = max_det
        # baked_nms: shape heuristic, independent of user intent
        self.baked_nms = (self.output_shape[1] == 300 and self.output_shape[2] == 6)
        # nms: should we run external NMS? user intent, but skip if it's already baked
        self.nms = nms and not self.baked_nms
        # for external NMS only: output is (B, 4+nc, anchors); derive nc once
        self.nc = int(self.output_shape[1]) - 4 if self.nms else None

        self.input_dtype = self.engine.get_tensor_dtype(self.input_tensor_name)
        self.output_dtype = self.engine.get_tensor_dtype(self.output_tensor_name)

        self._alloc_output_tensor()


    def _load_engine(self, engine_file_path: str, engine_bytes: bytes) -> trt.ICudaEngine:
        if engine_bytes is None:
            with open(engine_file_path, "rb") as f:
                engine_bytes = f.read()
        runtime = trt.Runtime(self.TRT_LOGGER)
        return runtime.deserialize_cuda_engine(engine_bytes)

    def inference_cp(self, src: cp.ndarray) -> cp.ndarray:
        """
        src must be contiguous BCHW, batch size 1.

        returns parsed results without batch dim and conf thresholded
        """

        self.forward(src)

        if self.nms:
            return self._external_nms_cp()
        return self._parse_cp_results()

    def forward(self, src: cp.ndarray) -> cp.ndarray:
        """
        no post processing / nms (unless baked in)
        """
        self.context.execute_v2(bindings=[src.data.ptr, self.output_ptr])
        return self.output_buffer

    def _external_nms_cp(self):
        """Class-aware NMS for engines that don't have it baked in.

        Input:  output_buffer shape (B, 4+nc, anchors), boxes in xywh (cx,cy,w,h).
        Output: (M, 6) cupy array, columns [x1, y1, x2, y2, conf, cls]; M <= max_det.
        """
        # zero-copy bridge cupy -> torch (same GPU memory)
        out_t = torch.from_dlpack(self.output_buffer)[0]  # (4+nc, anchors)
        pred = out_t.transpose(0, 1)  # (anchors, 4+nc)
        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4:]
        scores, classes = cls_scores.max(dim=-1)

        mask = scores > self.conf_threshold
        boxes_xywh, scores, classes = boxes_xywh[mask], scores[mask], classes[mask]
        if boxes_xywh.numel() == 0:
            return cp.empty((0, 6), dtype=cp.float32)

        cx, cy, w, h = boxes_xywh.unbind(-1)
        boxes_xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

        keep = batched_nms(boxes_xyxy, scores, classes, self.iou_threshold)[: self.max_det]
        out = torch.cat([
            boxes_xyxy[keep],
            scores[keep, None],
            classes[keep, None].to(boxes_xyxy.dtype),
        ], dim=-1)
        return cp.from_dlpack(out)

    def _parse_cp_results(self):
        """
        Returns:
            literally just conf thresholded lol. returns (B, num_dets, 6) with x1,y1,x2,y2,conf,cls_id if nms is applied. otherwise B,4+nc,num_voters
        """
        return self.output_buffer[self.output_buffer[:, :, 4] > self.conf_threshold]

    def _alloc_output_tensor(self):
        self.output_buffer = cp.empty(self.output_shape, dtype=trt.nptype(self.output_dtype))
        self.output_ptr = self.output_buffer.data.ptr

    def __del__(self):
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine


if __name__ == '__main__':
    #python -m src.aimbot.engine.tensorrt_engine
    import cv2
    import time
    import logging
    logging.basicConfig(level="INFO")
    torch.cuda.empty_cache()
    cwd = os.getcwd()
    base_dir = "data/models/pf_1550img_11s/weights"
    engine_name = "640x640_stripped.engine"
    model_path = os.path.join(cwd, base_dir, engine_name)
    imgsz = (640, 640)
    model = TensorRT_Engine(engine_file_path=model_path, conf_threshold=0, verbose=True)

    img_path = os.path.join(cwd, 'data/datasets/pf_1550img/images/train/frame13.png')
    img = cv2.imread(img_path)
    img = cv2.resize(img, imgsz[::-1])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    img /= 255.0
    np_img = np.ascontiguousarray(img)

    cp_img = cp.asarray(np_img, dtype=cp.float32)
    output = model.inference_cp(cp_img)
    print(f"Output shape: {output.shape}")

    N_POOL = 64
    WARMUP = 64
    ITERATIONS = 64
    INFERENCES = 64

    pool = [cp.ascontiguousarray(cp.asarray(np_img)) for _ in range(N_POOL)]

    for i in range(WARMUP):
        output = model.inference_cp(pool[i % N_POOL])

    sprint_means = np.empty(ITERATIONS)
    for i in range(ITERATIONS):
        start = time.perf_counter_ns()
        for j in range(INFERENCES):
            results = model.inference_cp(pool[j % N_POOL])
        sprint_means[i] = (time.perf_counter_ns() - start) / INFERENCES / 1e3

    mean = sprint_means.mean()
    std = sprint_means.std(ddof=1)
    ci95 = 1.96 * std / np.sqrt(ITERATIONS)
    print(f"inference: {mean:.2f} ±{std:.2f} µs  CI95=[{mean-ci95:.2f}, {mean+ci95:.2f}]  ({1e6/mean:.1f} inf/s)")
