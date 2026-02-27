import os
import tensorrt as trt
# import pycuda.autoinit#auto init cuda mem context
#init context in here bad, use default context
import numpy as np
import cupy as cp
# import pycuda.cuda as cuda
from ..utils.utils import log



class TensorRT_Engine:
    def __init__(self,engine_file_path:str, conf_threshold:float,verbose:bool = False):
        
        self.TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine(engine_file_path)
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

        # self.stream = cp.cuda.Stream()
        self.output_ptr = 0
        self.conf_threshold = conf_threshold
        self.input_shape = self.engine.get_tensor_shape(self.input_tensor_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_tensor_name)
        self.imgsz = self.input_shape[2:]
        log(f'engine imgsz: {self.imgsz}', "INFO")
        self.input_dtype = self.engine.get_tensor_dtype(self.input_tensor_name)
        self.output_dtype = self.engine.get_tensor_dtype(self.output_tensor_name)

        self._alloc_output_tensor()
        
        
        

        
    def _load_engine(self, engine_file_path: str) -> trt.ICudaEngine:
        log(f'Reading engine file from: {engine_file_path}', "INFO")
        with open(engine_file_path, "rb") as f:
            runtime = trt.Runtime(self.TRT_LOGGER)
            return runtime.deserialize_cuda_engine(f.read())

    def inference_cp(self, src: cp.ndarray) -> cp.ndarray:
        """
        src must be contiguous BCHW, batch size 1.

        returns parsed results without batch dim and conf thresholded
        """
        # expected_dtype = cp.dtype(trt.nptype(self.input_dtype))
        # if not (src.flags.c_contiguous and src.shape == self.input_shape and src.dtype == expected_dtype):
        #     src = cp.ascontiguousarray(src.astype(expected_dtype))
        # self.context.set_tensor_address(self.input_tensor_name, src.data.ptr)
        # execute_async_v3 benchmarks the same speed as the deprecated execute_v2, API choice doesn't matter
        self.context.execute_v2(bindings = [src.data.ptr, self.output_ptr])
        # self.context.execute_async_v3(self.stream.ptr)
        # deviceSynchronize is ~16% faster than self.stream.synchronize() in benchmarks,
        # CuPy stream sync has nontrivial overhead vs raw device sync
        # cp.cuda.runtime.deviceSynchronize()
        return self._parse_cp_results()

    def _parse_cp_results(self):
        #x1,y1,x2,y2,conf,cls_id
        removed_batch_dim = self.output_buffer.reshape(self.output_shape[1:])
        #removing batch dim retains contiguity if input is contiguous (already is after preprocessing)
        filtered_results = removed_batch_dim[removed_batch_dim[:, 4] > self.conf_threshold]
        return filtered_results
    
    def _alloc_output_tensor(self):
        self.output_buffer = cp.empty(self.output_shape, dtype=trt.nptype(self.output_dtype))
        self.output_ptr = self.output_buffer.data.ptr
        # self.context.set_tensor_address(self.output_tensor_name, self.output_ptr)
        
    def __del__(self):
        # Explicit cleanup for CUDA resources
        if hasattr(self, 'context'):
            del self.context
        if hasattr(self, 'engine'):
            del self.engine
        # if hasattr(self, 'stream'):
        #     self.stream.synchronize()
        #     del self.stream  
                   
if __name__ == '__main__':
    
    #python -m src.aimbot.engine.tensorrt_engine
    import torch
    import cv2
    import time
    import logging
    logging.basicConfig(level = "INFO")
    torch.cuda.empty_cache()
    cwd = os.getcwd()
    base_dir = "data/models/pf_1550img_11s/weights"
    engine_name = "640x640_stripped.engine"
    model_path = os.path.join(cwd, base_dir, engine_name)
    imgsz = (640,640)
    model = TensorRT_Engine(model_path,conf_threshold= 0, verbose = True)
    
    img_path = os.path.join(cwd, 'data/datasets/pf_1550img/images/train/frame13.png')
    img = cv2.imread(img_path)
    img = cv2.resize(img, imgsz[::-1]) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = np.expand_dims(img, axis=0).astype(np.float32)#add batch dim
    img /= 255.0 
    np_img = np.ascontiguousarray(img)
    
    cp_img = cp.asarray(np_img, dtype = cp.float32)
    log(f'cp shape: {cp_img.shape}', "INFO")
    output = model.inference_cp(cp_img)
    log(f"Output shape: {output.shape}", "INFO")
    batch_idx = 0
    detection_idx = 0
    log(f'output: {output}', "INFO")

    N_POOL     = 1
    WARMUP     = 1
    ITERATIONS = 2
    INFERENCES = 1
    
    # 64 independently allocated CuPy buffers — round-robin to bust GPU L2 cache
    pool = [cp.ascontiguousarray(cp.asarray(np_img)) for _ in range(N_POOL)]

    for i in range(WARMUP):
        output = model.inference_cp(pool[i % N_POOL])
    # cp.cuda.runtime.deviceSynchronize()

    sprint_means = np.empty(ITERATIONS)
    for i in range(ITERATIONS):
        start = time.perf_counter_ns()
        for j in range(INFERENCES):
            results = model.inference_cp(pool[j % N_POOL])
        # cp.cuda.runtime.deviceSynchronize()
        sprint_means[i] = (time.perf_counter_ns() - start) / INFERENCES / 1e3

    mean = sprint_means.mean()
    std  = sprint_means.std(ddof=1)
    ci95 = 1.96 * std / np.sqrt(ITERATIONS)
    log(f"inference: {mean:.2f} ±{std:.2f} µs  CI95=[{mean-ci95:.2f}, {mean+ci95:.2f}]  ({1e6/mean:.1f} inf/s)", "INFO")

