import os
from ultralytics import YOLO

# load a pretrained YOLOv8n classification model
model = YOLO("yolov8n-cls.pt") 

# Get the full path of the dataset folder
full_path = os.path.abspath('dataset')

# Train the model using the specified data, for 50 epochs, and with an image size of 64x64
results = model.train(data=full_path, epochs=20, imgsz=64)