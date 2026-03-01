from ultralytics import YOLO
import os
import pickle
import torch
import time
import yaml

DATA_YAML_PATH = 'data/datasets/pf_1550img/data.yaml'
EPOCHS = 100
BATCH_SIZE = 24
NBS = 24
IMGSZ = 640
MODEL_NAME = "yolo26s.pt"
PROJECT_DIR = 'data/models'
RUN_NAME = 'pf_1550img_26s_val_2'
SAVE_PERIOD = 20
WARMUP_EPOCHS = 3
LR0 = 5e-5
WEIGHT_DECAY = 1e-3
PATIENCE = 12

def main():
    torch.cuda.empty_cache()
        
    # Define the paths to the dataset and YAML file
    cwd = os.getcwd()
    data_yaml_path = os.path.join(cwd,DATA_YAML_PATH)  # Update with your data.yaml path
    print(data_yaml_path)
    epochs = EPOCHS  # Adjust the number of epochs as needed
    batch = BATCH_SIZE # Adjust based on your GPU memory
    nbs = NBS #nominal/effective batch size, updates gradients every 4 iterations (24/6)
    imgsz = IMGSZ  # Image size for training 
    
    # base_dir = 'models/pf_1550img_11s/weights'
    # model_name = "best.pt"
    # model_path = os.path.join(os.getcwd(), base_dir, model_name)
    # model = YOLO(model_path)
    model = YOLO(MODEL_NAME)  # Use the YOLOv8 pre-trained weights or your own
    model.train(
        data=data_yaml_path,  # Path to your dataset configuration file
        epochs=epochs,  # Number of training epochs
        batch=batch,  # Batch size for training
        imgsz=imgsz,  # Image size for training
        project=PROJECT_DIR,  # Directory to save the tra  ining results
        resume = False,
        device = 0,
        name=RUN_NAME,  # Directory name for the run
        exist_ok=True,  # Overwrite the existing directory if necessary
        cache = 'disk',#takes in bool or string ig what
        patience = PATIENCE,#0 = no early stopping
        rect = False, #may need to experiment with this, default False
        save_period = SAVE_PERIOD,
        fraction = 1,#fraction of train
        plots = True,#generates some more plots
        cos_lr = True,#cos learning rate scheduler, convergence more stable
        profile = True,
        optimizer = 'adamw',
        amp = True,
        pretrained = True,
        warmup_epochs=WARMUP_EPOCHS,
        nbs=nbs,    
        lr0 = LR0,
        augment = True,
        deterministic = False,
        weight_decay = WEIGHT_DECAY,#default 5e-4
        # degrees = 5,
        # shear = 5,
        # scale = .4,
        # perspective = 1e-4
    )

    model.save()
    
if __name__ == '__main__':
    main()