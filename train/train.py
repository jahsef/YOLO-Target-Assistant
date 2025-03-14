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

    epochs = 200  # Adjust the number of epochs as needed
    batch = 3  # Adjust based on your GPU memory
    nbs = 12 #nominal/effective batch size, updates gradients every 4 iterations (24/6)
    imgsz = 1440  # Image size for training

    # Load the YOLOv8 model (pretrained or custom)
    # model = YOLO("yolo11m.pt")  # Use the YOLOv8 pre-trained weights or your own
    # model = YOLO(os.path.join(os.getcwd(),"runs/train/\EFPS_3000image_realtrain_1440x1440_100epoch_batch6_11s/weights/best.pt"))
    # model = YOLO(os.path.join(os.getcwd(),"runs/train/EFPS_3000image_11m_1440p_batch3_epoch200/weights/epoch115.pt"))
    model = YOLO(os.path.join(cwd,r'runs\train\EFPS_4000img_11m_1440p_batch6_epoch200\weights\epoch20.pt'))
    # model = YOLO(os.path.join(cwd, "runs/train/EFPS_1863transferfrom1400_1440x1440_10epoch_batch6_11s/weights/best.pt"))
    # Train the model
    
    model.train(
        data=data_yaml_path,  # Path to your dataset configuration file
        epochs=epochs,  # Number of training epochs
        batch=batch,  # Batch size for training
        imgsz=imgsz,  # Image size for training
        project='runs/train',  # Directory to save the tra  ining results
        resume = True,
        device = 0,
        name='EFPS_4000img_11m_1440p_batch6_epoch200',  # Directory name for the run
        exist_ok=True,  # Overwrite the existing directory if necessary
        cache = 'disk',#takes in bool or string ig what
        patience = 20,#0 = no early stopping
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
        lr0 = 0.001,
        augment = True,
        dropout = .03
    )

    model.save()
    
if __name__ == '__main__':
    main()