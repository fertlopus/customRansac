from line_detection.detector.custom_ransac import RANSAC
import matplotlib.pyplot as plt
import cv2
import numpy as np


def main():
    img = "./data/original/edged_image.png"
    image_edged = plt.imread(img)[:, :, 0]
    best_a, best_b = RANSAC.fit_RANSAC(image_edged, epochs=200, sigma=2)
    starting_point, ending_point = RANSAC.generate_extends(best_a, best_b, image_edged.shape[0], image_edged.shape[1])
    E2 = np.zeros((image_edged.shape[0], image_edged.shape[1], 3), dtype=np.uint8)
    E3 = np.array(255 * image_edged, dtype=np.uint8)
    E2[:, :, 0] = E3
    E2[:, :, 1] = E3
    E2[:, :, 2] = E3
    I = cv2.line(E2, starting_point, ending_point, (0, 0, 255), 1)
    plt.figure(figsize=(15, 10))
    plt.imsave('./data/processed/ransac.png', I[:, :, ::-1])
    print('Done. Recognized Image saved under provided path.')
    return None


if __name__ == "__main__":
    main()