# from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from PIL import Image, ImageFilter



def contrast_algorithm(images, contrast_factor=1.5):

    images = images.astype(np.float64)

    # Calculate the mean intensity for each image
    mean_intensity = np.mean(images, axis=(1, 2), keepdims=True)

    # Adjust the contrast for each image
    enhanced_images = (images - mean_intensity) * contrast_factor + mean_intensity

    # Clip the values to the valid intensity range [0, 255]
    enhanced_images = np.clip(enhanced_images, 0, 255)

    return enhanced_images.astype(np.uint8)

def main(path,num_processes=5):

    input_path = np.array(Image.open(path))
    image = input_path.astype(np.uint8)

    # Split the image into threads for parallel processing
    height, width = image.shape[:2]
    part_height = height // num_processes

    image_threads = [image[i * part_height: (i + 1) * part_height, :] for i in range(num_processes)]

    # Create a pool of processes
    with Pool(processes=num_processes) as pool:
        # Apply the contrast_algorithm to each image thread in parallel
        contrasted_threads = pool.starmap(contrast_algorithm, [(thread, 1.5) for thread in image_threads])

    # Concatenate the contrasted threads parts to reconstruct the final image
    contrasted_image = np.concatenate(contrasted_threads, axis=0)
    contrasted_image = contrasted_image.astype(np.uint8)
    plt.imsave("Results/enhanced_img.jpg", contrasted_image)

    return contrasted_image

if __name__ == "__main__":

    # Apply the enhancement algorithm and save the result
    main(path)

