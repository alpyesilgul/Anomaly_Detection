from imutils import paths
import numpy as np
import cv2


def quantify_image(image, bins):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins,
                        [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist


def load_dataset(dataset_path, bins):
    images = list(paths.list_images(dataset_path))
    data = []
    for image in images:
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = quantify_image(image, bins)
        data.append(features)
    return np.array(data)
