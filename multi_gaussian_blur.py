import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import parallel_code

def apply_gaussian_blur(image, blur_radius):
    # Convert the NumPy array to a PIL Image
    pil_image = Image.fromarray(image)
    # Apply the Gaussian blur filter to the image part
    blurred_img = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    # Convert the PIL Image back to a NumPy array
    blurred_img = np.array(blurred_img)

    return blurred_img.astype(np.uint8)

def main(path, blur_radius=2, thread_numbers=5):
    image=np.array(Image.open(path))
    image = image.astype(np.uint8)
    # Apply the Gaussian blur algorithm in parallel
    blurred_image = parallel_code.thread_num(apply_gaussian_blur, blur_radius, image, thread_numbers)
    plt.imsave("Results/enhanced_img.jpg", blurred_image)

    return blurred_image


if __name__ == '__main__':
    main(path)