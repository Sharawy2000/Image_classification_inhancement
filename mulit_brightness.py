import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import parallel_code

def brightness_algorithm(image, brightness_value):
    image = image.astype(np.float64)
    # Increase the brightness of the images
    image += brightness_value
    image = np.clip(image, 0, 255)

    return image.astype(np.uint8)

def main(path='Images/image.jpeg',brightness_value=100,thread_numbers=5):
    input_path = np.array(Image.open(path))
    image = input_path.astype(np.uint8)
    brightened_image=parallel_code.thread_num(brightness_algorithm,brightness_value,image,thread_numbers)
    # Image.fromarray(brightened_image).save("Results/enhanced_img.jpg")
    plt.imsave("Results/enhanced_img.jpg", brightened_image)

    return brightened_image

if __name__ == '__main__':
    main(path)
