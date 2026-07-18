from pathlib import Path
from . import tensorrt_engine
import cupy as cp
import torch
import numpy as np

_PREPROCESS_KERNEL_CP = cp.RawKernel(r"""
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

class Model:
    def __init__(self,model_path:Path,hw_capture:tuple[int,int],conf_threshold:float = 0.25):
        """_summary_

        Args:
            model_path (Path): path to your model lol
            hw_capture (tuple[int,int]): THIS IS ONLY USED FOR NON ENGINE MODELS, ENGINE MODELS DETERMINE IMGSZ AUTOMATICALLY BASED ON EXPORT SETTINGS
            conf_threshold (float): confidence threshold applied by either backend

        Returns:
            model object
        """
        self.model = None
        self.conf_threshold = conf_threshold
        self._load_model(model_path=model_path,hw_capture=hw_capture)

    def _load_model(self, model_path: Path, hw_capture:tuple[int,int]):
        self.model_ext = model_path.suffix
        if self.model_ext == '.engine':
            self.model = tensorrt_engine.TensorRT_Engine(engine_file_path= model_path, conf_threshold= self.conf_threshold,verbose = False)
            self.hw_capture = self.model.imgsz
            if self.model == None:
                raise Exception("tensorrt engine did not load correctly")
        elif self.model_ext == '.pt':
            from ultralytics import YOLO
            self.hw_capture = hw_capture
            self.model = YOLO(model = model_path)
        else:
            raise Exception(f'not supported file format: {self.model_ext} <- file format should be here lol')
    
    def inference(self,src:cp.ndarray) -> any:
        """
        Args:
            src (cp.ndarray): source image in CuPy array, should be hwc
        
        Returns:
            CPU numpy array of results (n,[x1,y1,x2,y2,conf,cls_id]) where n is bounding box index
            
        """
        

        if self.model_ext == '.engine':
            #Torch/CuPy/.... array of results (n,[x1,y1,x2,y2,conf,cls_id]) where n is bounding box index
            results = cp.asnumpy(self.model.inference_cp(self._preprocess_cp(src)))
        elif self.model_ext == '.pt':
            # ultralytics returns list[Results]; .boxes.data is (n, 6) [x1,y1,x2,y2,conf,cls]
            results = self._inference_torch(self._preprocess_torch(src))[0].boxes.data.cpu().numpy()
        else:
            raise Exception('big no no happened this should never execute, model was probably not loaded correctly')
        return results

    def _preprocess_cp(self, frame: cp.ndarray) -> cp.ndarray:
        """frame: (H, W, 3) cp.uint8. Returns (1, 3, H, W) cp.float32."""
        assert frame.dtype == cp.uint8 and frame.ndim == 3 and frame.shape[2] == 3
        H, W, _ = frame.shape
        src = cp.ascontiguousarray(frame)  # no-op if already contiguous
        out = cp.empty((1, 3, H, W), dtype=cp.float32)
        block = (32, 8, 1)
        grid = ((W + block[0] - 1) // block[0],
                (H + block[1] - 1) // block[1],
                3)
        _PREPROCESS_KERNEL_CP(grid, block, (src, out, np.int32(H), np.int32(W)))
        return out
    
    def _preprocess_torch(self, frame: cp.ndarray) -> torch.Tensor:
        return torch.as_tensor(self._preprocess_cp(frame))

    
    @torch.inference_mode()
    def _inference_torch(self,source:torch.Tensor) -> list:
        results = self.model(source=source,
            conf = self.conf_threshold,
            imgsz=self.hw_capture,
            verbose = False
        )

        return results

if __name__ == '__main__':
    #python -m src.aimbot.engine.model
    import sys
    import numpy as np
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    ENGINE_PATH = Path("data/models/pf_1550img_11s/weights/640x640_stripped.engine")

    model = Model(model_path=ENGINE_PATH, hw_capture=(640, 640))
    H, W = model.hw_capture

    frame_cp = cp.asarray(np.random.randint(0, 255, (H, W, 3), dtype=np.uint8))

    result = model.inference(frame_cp)
    print(f"output shape: {result.shape}")
    print(f"output: {result}")