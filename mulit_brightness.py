import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from PIL import Image, ImageFilter


# Increase the brightness of the images

def brightness_algorithm(image, brightness_value):
    image = image.astype(np.float64)
    image += brightness_value
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

def main(path='Images/image.jpeg',num_processes=5):
    input_path = np.array(Image.open(path))
    image = input_path.astype(np.uint8)

    # Split the image into parts for parallel processing
    height, width = image.shape[:2]
    part_height = height // num_processes

    image_parts = [image[i * part_height: (i + 1) * part_height, :] for i in range(num_processes)]

    # Create a pool of processes
    with Pool(processes=num_processes) as pool:
        # Apply the increase_contrast to each image part in parallel
        brightened_parts = pool.starmap(brightness_algorithm, [(part, 100) for part in image_parts])

    # Concatenate the contrasted parts to reconstruct the final image
    brightened_image = np.concatenate(brightened_parts, axis=0)
    brightened_image = brightened_image.astype(np.uint8)
    plt.imsave("Results/enhanced_img.jpg", brightened_image)

    return brightened_image

if __name__ == '__main__':

    main(path)