from pathlib import Path
from . import tensorrt_engine
from ultralytics import YOLO
import cupy as cp
import torch
from ultralytics.engine.results import Boxes


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
        self.empty_boxes = Boxes(boxes=torch.empty((0, 6)), orig_shape=self.hw_capture)
    
    def parse_results_into_ultralytics_boxes(self,results: object) -> Boxes:
        """_summary_

        Args:
            results (object): results from model inference (cp,np etc.)

        Returns:
            Boxes: results in Ultralytics Boxes format, used for Ultralytics BYTETracker
        """

        #need to convert into boxes to pass into the ultralytics BYTETracker
        if len(results) == 0:#xyxy, conf, cls, smth else?
            return self.empty_boxes
        
        if type(results) is not torch.Tensor:
            results = torch.as_tensor(results)
            
        converted_boxes = Boxes(
            boxes=results,
            orig_shape=self.hw_capture
        )
        return converted_boxes 
    
    def _load_model(self, model_path: Path, hw_capture:tuple[int,int]):
        self.model_ext = model_path.suffix
        if self.model_ext == '.engine':
            self.model = tensorrt_engine.TensorRT_Engine(engine_file_path= model_path, conf_threshold= .25,verbose = False)
            self.hw_capture = self.model.imgsz
            if self.model == None:
                raise Exception("tensorrt engine did not load correctly")
        elif self.model_ext == '.pt':
            self.hw_capture = hw_capture
            self.model = YOLO(model = model_path)
        else:
            raise Exception(f'not supported file format: {self.model_ext} <- file format should be here lol')
        
    def inference(self,src:cp.ndarray) -> any:
        """
        Args:
            src (cp.ndarray): source image in CuPy array, should be hwc
        
        Returns:
            Torch/CuPy/.... array of results (n,[x1,y1,x2,y2,conf,cls_id]) where n is bounding box index
        """
        if self.model_ext == '.engine':
            return self._inference_tensorrt(self._preprocess_cp(src))
        elif self.model_ext == '.pt':
            return self._inference_torch(self._preprocess_torch(src))
        else:
            raise Exception('big no no happened this should never execute, model was probably not loaded correctly')
     
    def _preprocess_frame(self,frame:cp.ndarray) -> cp.ndarray:
        """_summary_

        Args:
            frame (cp.ndarray): _description_

        Returns:
            cp.ndarray: normalized bchw format
        """
        bchw = frame.transpose(2, 0, 1)[cp.newaxis, ...]
        float_frame = bchw.astype(cp.float32, copy=False)#engine expects float 32 unless i export it differently
        float_frame /= 255.0 #/= is inplace, / creates a new cp arr
        return float_frame
    
    def _preprocess_cp(self,frame: cp.ndarray) -> cp.ndarray:
        return cp.ascontiguousarray(self._preprocess_frame(frame))
    
    def _preprocess_torch(self,frame: cp.ndarray) -> torch.Tensor:
        return torch.as_tensor(self._preprocess_frame(frame)).contiguous() 
      
    def _inference_tensorrt(self,src:cp.ndarray) -> cp.ndarray:
        return self.model.inference_cp(src = src)

    
    @torch.inference_mode()
    def _inference_torch(self,source:torch.Tensor) -> torch.Tensor:
        results = self.model(source=source,
            conf = .25,
            imgsz=self.hw_capture,
            verbose = False
        )
        return results