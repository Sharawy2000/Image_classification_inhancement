import numpy as np
from concurrent.futures import ThreadPoolExecutor

def thread_num(algorithm, algo_factor, image, thread_numbers):
    height = image.shape[0]
    part_height = height // thread_numbers
    image_threads = [image[i * part_height: (i + 1) * part_height, :] for i in range(thread_numbers)]

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(algorithm, thread, algo_factor) for thread in image_threads]
        results = [result.result() for result in futures]

    sorted_indices = np.argsort([idx for idx, _ in enumerate(results)])
    final_results = [results[i] for i in sorted_indices]

    Edited_image = np.concatenate(final_results, axis=0)
    Edited_image = Edited_image.astype(np.uint8)

    return Edited_image
