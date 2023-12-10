import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time


# Increase the brightness of the images

def increase_brightness(image, brightness_value):
    image = image.astype(np.float64)
    image += brightness_value
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)

def main(path='Images/image.jpeg'):
    start_time=time.time()

    # Read the input images
    image = mpimg.imread(path)

    # Create a pool of worker processes
    pool = mp.Pool(5)

    # Start processing the images in parallel
    brightened_images = pool.starmap(increase_brightness, [(image, 100)])

    # Display the original images in the first row

    plt.imsave("Results/enhanced_img.jpg", brightened_images[0])
    end_time = time.time()

    total = end_time - start_time
    print(f"time: {total:.3f} seconds")

    return brightened_images[0]

if __name__ == '__main__':

    main(path='Images/image.jpeg')