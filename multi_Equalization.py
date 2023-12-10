import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

def apply_gaussian_blur(image_part, blur_radius):
    """
    Apply a Gaussian blur filter to a portion of an image.

    Parameters:
    - image_part: NumPy array representing a portion of the image.
    - blur_radius: The radius of the Gaussian blur filter.

    Returns:
    - Portion of the image with applied Gaussian blur.
    """
    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(image_part)

    # Apply the Gaussian blur filter to the image part
    blurred_part = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Convert the PIL Image back to a NumPy array
    blurred_part = np.array(blurred_part)

    return blurred_part.astype(np.uint8)

def apply_gaussian_blur_parallel(image=np.array(Image.open('Images/img.jpg')), blur_radius=5, num_processes=5):
    """
    Apply a Gaussian blur filter to an image using multiprocessing.

    Parameters:
    - image: NumPy array representing the image.
    - blur_radius: The radius of the Gaussian blur filter.
    - num_processes: Number of processes to use for parallelization.

    Returns:
    - Image with applied Gaussian blur.
    """
    image = image.astype(np.uint8)

    # Split the image into parts for parallel processing
    height, width = image.shape[:2]
    part_height = height // num_processes

    image_parts = [image[i * part_height : (i + 1) * part_height, :] for i in range(num_processes)]

    # Create a pool of processes
    with Pool(processes=num_processes) as pool:
        # Apply the Gaussian blur filter to each image part in parallel
        blurred_parts = pool.starmap(apply_gaussian_blur, [(part, blur_radius) for part in image_parts])

    # Concatenate the blurred parts to reconstruct the final image
    blurred_image = np.concatenate(blurred_parts, axis=0)
    blurred_image=blurred_image.astype(np.uint8)
    plt.imsave("Results/blurred_image.jpg", blurred_image)


    return blurred_image

# Load an example image (replace 'example_image.jpg' with your image path)


if __name__ == '__main__':
    # original_image = 'Images/img.jpg'
    # original_image = np.array(Image.open(image_path))

    # Apply Gaussian blur to the image using multiprocessing
    blurred_image = apply_gaussian_blur_parallel(original_image)

import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Increase the brightness of the images

def increase_brightness(image, brightness_value):
    image = image.astype(np.float64)
    image += brightness_value
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

def main(path='Images/image.jpeg'):

    # Read the input images
    image = mpimg.imread(path)

    # Create a pool of worker processes
    pool = mp.Pool(3)

    # Start processing the images in parallel
    brightened_images = pool.starmap(increase_brightness, [(image, 150)])

    # Display the original images in the first row

    plt.imsave("Results/enhanced_img.jpg", brightened_images[0])

    return brightened_images[0]

if __name__ == '__main__':

    main(path)
