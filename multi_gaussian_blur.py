import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

def apply_gaussian_blur(image_part, blur_radius):

    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(image_part)

    # Apply the Gaussian blur filter to the image part
    blurred_part = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Convert the PIL Image back to a NumPy array
    blurred_part = np.array(blurred_part)

    return blurred_part.astype(np.uint8)

def main(path, blur_radius=5, num_processes=5):
    
    image=np.array(Image.open(path))
    image = image.astype(np.uint8)

    # Split the image into threads for parallel processing
    height, width = image.shape[:2]
    part_height = height // num_processes

    image_threads = [image[i * part_height : (i + 1) * part_height, :] for i in range(num_processes)]

    # Create a pool of processes
    with Pool(processes=num_processes) as pool:
        # Apply the Gaussian blur filter to each image part in parallel
        blurred_threads = pool.starmap(apply_gaussian_blur, [(thread, blur_radius) for thread in image_threads])

    # Concatenate the blurred parts to reconstruct the final image
    blurred_image = np.concatenate(blurred_threads, axis=0)
    blurred_image = blurred_image.astype(np.uint8)
    plt.imsave("Results/enhanced_img.jpg", blurred_image)


    return blurred_image


if __name__ == '__main__':
    
    # Apply Gaussian blur to the image using multiprocessing
    blurred_image = main(original_image)

