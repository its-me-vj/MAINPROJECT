import os
import numpy as np
import scipy.io as sio
import cv2
import matplotlib.pyplot as plt

dataset_path = "dataset/part_A_final/train_data"

def load_image_gt(image_path, gt_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mat = sio.loadmat(gt_path)
    points = mat["image_info"][0, 0][0, 0][0]

    return image, points

image_file = "IMG_1.jpg"
gt_file = "GT_IMG_1.mat"

image, points = load_image_gt(f"{dataset_path}/images/{image_file}", f"{dataset_path}/ground_truth/{gt_file}")

plt.imshow(image)
plt.scatter(points[:, 0], points[:, 1], color="red", s=5)
plt.title("Image with Ground Truth Points")
plt.show()
