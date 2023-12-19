import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import run_threads

def apply_gaussian_blur(image, blur_radius):

    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(image)

    # Apply the Gaussian blur filter to the image part
    blurred_img = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Convert the PIL Image back to a NumPy array
    blurred_img = np.array(blurred_img)

    return blurred_img.astype(np.uint8)

def main(path, blur_radius=2.5, thread_numbers=5):
    
    image=np.array(Image.open(path))
    image = image.astype(np.uint8)

    blurred_image = run_threads.thread_num(apply_gaussian_blur, blur_radius, image, thread_numbers)

    plt.imsave("Results/enhanced_img.jpg", blurred_image)


    return blurred_image


if __name__ == '__main__':
    
    # Apply Gaussian blur to the image using multiprocessing
    main(path)

