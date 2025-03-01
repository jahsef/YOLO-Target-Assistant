from ultralytics import YOLO
import os
import pickle
def main():
        
    # Define the paths to the dataset and YAML file
    cwd = os.getcwd()
    data_yaml_path = os.path.join(cwd,'train//split_dataset//data.yaml')  # Update with your data.yaml path
    epochs = 100  # Adjust the number of epochs as needed
    batch = 1  # Adjust based on your GPU memory
    imgsz = 2560  # Image size for training

    # Load the YOLOv8 model (pretrained or custom)
    model = YOLO("yolo11n.pt")  # Use the YOLOv8 pre-trained weights or your own

    # Train the model
    model.train(
        data=data_yaml_path,  # Path to your dataset configuration file
        epochs=epochs,  # Number of training epochs
        batch=batch,  # Batch size for training
        imgsz=imgsz,  # Image size for training
        project='runs/train',  # Directory to save the training results
        name='2560x2560_batch1_11n',  # Directory name for the run
        exist_ok=True  # Overwrite the existing directory if necessary
    )

    model.save()
    

print("Training complete!")

if __name__ == "__main__":
    main()