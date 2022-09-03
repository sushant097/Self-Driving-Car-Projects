import cv2
import os
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
from itertools import islice


data_folder_path = "driving_dataset"
train_file = os.path.join(data_folder_path, 'data.txt')
print(train_file)


def preprocess(img):
    return cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (100, 100))


def return_data():

    X = []
    y = []
    features = []

    print("Loading...")
    with open(train_file) as fp:
        for line in islice(fp, None):
            path, angle = line.strip().split()
            full_path = os.path.join(data_folder_path, path)
            X.append(full_path)
            # With angles from -pi to pi which avoid rescaling the atan in the model
            y.append(float(angle) * math.pi / 180)

    for i in range(len(X)):
        img = plt.imread(X[i])
        features.append(preprocess(img))
    
    features = np.array(features).astype('float32')
    labels = np.array(y).astype('float32')

    with open("features", "wb") as f:
        pickle.dump(features, f, protocol=4)
    with open("labels", "wb") as f:
        pickle.dump(labels, f, protocol=4)


return_data()