import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO  
import numpy as np  

def main(image_path,model_path='./runs/classify/train/weights/last.pt'):
    # Load the pre-trained YOLO model
    model = YOLO(model_path)

    # Perform prediction on the specific image
    results = model(image_path)

    # Extract class names from the results
    names_dict = results[0].names

    # Extract predicted probabilities from the results and convert to a list
    probs = results[0].probs.data.tolist()

    # Print the predicted class based on the highest probability
    predicted_label = names_dict[np.argmax(probs)]

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure()
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Image label: ' + predicted_label)
    plt.show()


# Example usage
if __name__ == "__main__":

    main()

