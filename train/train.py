from ultralytics import YOLO
import os
def main():
        
    # Define the paths to the dataset and YAML file
    cwd = os.getcwd()
    data_yaml_path = os.path.join(cwd,'train//data.yaml')  # Update with your data.yaml path
    weights_path = os.path.join(cwd,'yolo11s.pt')  # You can use a pretrained model like yolov8n.pt
    epochs = 50  # Adjust the number of epochs as needed
    batch = 5  # Adjust based on your GPU memory
    imgsz = 1440  # Image size for training

    # Load the YOLOv8 model (pretrained or custom)
    model = YOLO(weights_path)  # Use the YOLOv8 pre-trained weights or your own

    # Train the model
    model.train(
        data=data_yaml_path,  # Path to your dataset configuration file
        epochs=epochs,  # Number of training epochs
        batch=batch,  # Batch size for training
        imgsz=imgsz,  # Image size for training
        project='runs/train',  # Directory to save the training results
        name='train_run',  # Directory name for the run
        exist_ok=True  # Overwrite the existing directory if necessary
    )

    # Optionally, you can save the best weights after training
    model.save()

print("Training complete!")

if __name__ == "__main__":
    main()