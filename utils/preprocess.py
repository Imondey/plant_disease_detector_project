import os
import cv2
import numpy as np

def load_images(folder, size=(64, 64)):
    images = []
    labels = []
    for label in os.listdir(folder):
        print(f"Loading category: {label}")
        path = os.path.join(folder, label)
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, size)
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)
