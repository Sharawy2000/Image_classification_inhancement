import numpy as np
import matplotlib.pyplot as plt
import threading

def thread_num(algorithm,algo_factor,image,thread_numbers):

    height, width = image.shape[:2]
    part_height = height // thread_numbers

    # Split the image into threads parts for parallel processing
    image_threads = [image[i * part_height: (i + 1) * part_height, :] for i in range(thread_numbers)]

    # collect the output of thread for each part of image
    results = []

    # Function to apply brightness_algorithm to each image thread
    def process_thread(thread_idx, thread):
        result = algorithm(thread, algo_factor)
        results.append((thread_idx, result))

    # Create a list to store thread objects
    threads = []

    # Apply the brightness_algorithm to each image thread in parallel
    for i, thread in enumerate(image_threads):

        #create threads depend on image_parts
        t = threading.Thread(target=process_thread, args=(i, thread))

        #add it in thread to save
        threads.append(t)

        #run it
        t.start()

    # Wait for all threads to finish
    for t in threads:
        t.join()

    # Obtain the sorted indices based on the row indices
    sorted_indices = np.argsort([idx for idx, _ in results])

    # Extract the results in the correct order
    final_results = [results[i][1] for i in sorted_indices]

    print("Finished threads")

    brightened_image = np.concatenate(final_results, axis=0)
    brightened_image = brightened_image.astype(np.uint8)

    return brightened_image

