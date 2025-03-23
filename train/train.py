from ultralytics import YOLO
import os
import pickle
import torch
import time
import yaml
def main():
    torch.cuda.empty_cache()
        
    # Define the paths to the dataset and YAML file
    cwd = os.getcwd()
    data_yaml_path = os.path.join(cwd,'train//datasets/EFPS_4000img//data.yaml')  # Update with your data.yaml path

    epochs = 100  # Adjust the number of epochs as needed
    batch = 11  # Adjust based on your GPU memory
    nbs = 33 #nominal/effective batch size, updates gradients every 4 iterations (24/6)
    imgsz = 1440  # Image size for training
    # Load the YOLOv8 model (pretrained or custom)
    model = YOLO("yolo11n.pt")  # Use the YOLOv8 pre-trained weights or your own
    # model = YOLO(os.path.join(cwd,r'runs\train\EFPS_4000img_11s_retrain_1440p_batch6_epoch200\weights\epoch55.pt'))
    # Train the model
    
    model.train(
        data=data_yaml_path,  # Path to your dataset configuration file
        epochs=epochs,  # Number of training epochs
        batch=batch,  # Batch size for training
        imgsz=imgsz,  # Image size for training
        project='runs/train',  # Directory to save the tra  ining results
        resume = False,
        device = 0,
        name='EFPS_4000img_11n_1440p_batch11_epoch100',  # Directory name for the run
        exist_ok=True,  # Overwrite the existing directory if necessary
        cache = 'disk',#takes in bool or string ig what
        patience = 16,#0 = no early stopping
        rect = False, #may need to experiment with this, default False
        save_period = 5,#save every 10 epochs
        fraction = 1,#fraction of train
        plots = True,#generates some more plots
        cos_lr = True,#cos learning rate scheduler, convergence more stable
        profile = True,
        optimizer = 'adamw',
        amp = True,
        pretrained = True,
        warmup_epochs=5,
        nbs=nbs,    
        lr0 = 1e-4,
        augment = True
    )

    model.save()
    
if __name__ == '__main__':
    main()