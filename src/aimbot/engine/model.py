from pathlib import Path
from . import tensorrt_engine
import cupy as cp
import torch
import numpy as np

class Model:
    def __init__(self,model_path:Path,hw_capture:tuple[int,int]):
        """_summary_

        Args:
            model_path (Path): path to your model lol
            hw_capture (tuple[int,int]): THIS IS ONLY USED FOR NON ENGINE MODELS, ENGINE MODELS DETERMINE IMGSZ AUTOMATICALLY BASED ON EXPORT SETTINGS

        Returns:
            model object
        """
        self.model = None
        self._load_model(model_path=model_path,hw_capture=hw_capture)

    def _parse_results(self, results: object) -> np.ndarray:
        """
        Args:
            results (object): results from model inference (cp, np, torch, or ultralytics Results list)

        Returns:
            np.ndarray: (n, 6) CPU float32 array [x1, y1, x2, y2, conf, cls_id]
        """

        if type(results) is list:
            # print('the correct thing runs now')
            return results[0].boxes.data.cpu().numpy().astype(np.float32)

        if len(results) == 0:
            return np.empty((0, 6), dtype=np.float32)

        if type(results) is not torch.Tensor:
            results = torch.as_tensor(results)

        return results.to('cpu', dtype=torch.float32).numpy()

    def _load_model(self, model_path: Path, hw_capture:tuple[int,int]):
        self.model_ext = model_path.suffix
        if self.model_ext == '.engine':
            self.model = tensorrt_engine.TensorRT_Engine(engine_file_path= model_path, conf_threshold= .25,verbose = False)
            self.hw_capture = self.model.imgsz
            if self.model == None:
                raise Exception("tensorrt engine did not load correctly")
        elif self.model_ext == '.pt':
            from ultralytics import YOLO
            self.hw_capture = hw_capture
            self.model = YOLO(model = model_path)
        else:
            raise Exception(f'not supported file format: {self.model_ext} <- file format should be here lol')

    def inference(self,src:cp.ndarray) -> np.ndarray:
        """
        Args:
            src (cp.ndarray): source image in CuPy array, should be hwc

        Returns:
            np.ndarray: (n, 6) CPU float32 array [x1, y1, x2, y2, conf, cls_id]

        """

        if self.model_ext == '.engine':
            #Torch/CuPy/.... array of results (n,[x1,y1,x2,y2,conf,cls_id]) where n is bounding box index
            results = self.model.inference_cp(self._preprocess_cp(src))
        elif self.model_ext == '.pt':
            results =  self._inference_torch(self._preprocess_torch(src))
        else:
            raise Exception('big no no happened this should never execute, model was probably not loaded correctly')

        return self._parse_results(results)


    def _preprocess_cp(self, frame: cp.ndarray) -> cp.ndarray:
        frame = frame.transpose(2, 0, 1)[cp.newaxis, ...]
        frame = cp.ascontiguousarray(frame, dtype=cp.float32)
        cp.true_divide(frame, 255.0, out=frame, dtype=cp.float32)
        return frame

    def _preprocess_torch(self, frame: cp.ndarray) -> torch.Tensor:
        return torch.as_tensor(self._preprocess_cp(frame))


    @torch.inference_mode()
    def _inference_torch(self,source:torch.Tensor):
        results = self.model(source=source,
            conf = .25,
            imgsz=self.hw_capture,
            verbose = True
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
