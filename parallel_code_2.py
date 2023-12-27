import numpy as np
import threading

def thread_code(algorithm,algo_factor,image,thread_numbers):
    # Divide image into equal-sized height segments
    height = image.shape[0]
    part_height = height // 5

    # Split the image into threads parts for parallel processing
    image_threads = [image[i * part_height: (i + 1) * part_height, :] for i in range(5)]


    # collect the output of thread for each part of image
    results = []

    # Function to apply brightness_algorithm to each image thread
    def process_thread(thread_idx, thread):
        result = algorithm(thread, algo_factor)
        results.append((thread_idx, result))

    # Create a list to store thread objects

    # Method 1 : fixed 5 threads for 5 parts of image

    # Apply the brightness_algorithm to each image thread in parallel

    thread_1 = threading.Thread(target=process_thread, args=(0, image_threads[0]))

    thread_2 = threading.Thread(target=process_thread, args=(1, image_threads[1]))

    thread_3 = threading.Thread(target=process_thread, args=(2, image_threads[2]))

    thread_4 = threading.Thread(target=process_thread, args=(3, image_threads[3]))

    thread_5 = threading.Thread(target=process_thread, args=(4, image_threads[4]))

    thread_1.start()
    thread_2.start()
    thread_3.start()
    thread_4.start()
    thread_5.start()

    thread_1.join()
    thread_2.join()
    thread_3.join()
    thread_4.join()
    thread_5.join()


    # Obtain the sorted indices based on the row indices
    sorted_indices = np.argsort([idx for idx, _ in results])

    # Extract the results in the correct order
    final_results = [results[i][1] for i in sorted_indices]

    print("Finished threads")

    Edited_image = np.concatenate(final_results, axis=0)
    Edited_image = Edited_image.astype(np.uint8)

    return Edited_image