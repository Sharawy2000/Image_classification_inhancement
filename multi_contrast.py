import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import run_threads

def contrast_algorithm(images, contrast_factor=1.5):
    images = images.astype(np.float64)
    # Calculate the mean intensity for each image
    mean_intensity = np.mean(images, axis=(1, 2), keepdims=True)
    # Adjust the contrast for each image
    enhanced_images = (images - mean_intensity) * contrast_factor + mean_intensity
    # Clip the values to the valid intensity range [0, 255]
    enhanced_images = np.clip(enhanced_images, 0, 255)

    return enhanced_images.astype(np.uint8)

def main(path,contrast_factor=1.5,thread_numbers=5):
    input_path = np.array(Image.open(path))
    image = input_path.astype(np.uint8)
    # Apply the contrast algorithm in parallel
    contrasted_image=run_threads.thread_num(contrast_algorithm,contrast_factor,image,thread_numbers)
    plt.imsave("Results/enhanced_img.jpg", contrasted_image)

    return contrasted_image

if __name__ == "__main__":
    main(path)