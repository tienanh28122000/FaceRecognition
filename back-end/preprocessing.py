from cv2 import IMREAD_GRAYSCALE, log
from flask import Flask, request, jsonify
import base64
import cv2
import os
import numpy as np
import math
from PIL import Image
import io
import glob

from torch import float32

train_folder = "../dataset/train-data"
test_folder = "../dataset/test-data/All/"
FACE_CASCADE = "../haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(FACE_CASCADE)


for folder_name in os.listdir(train_folder):
            for image_name in os.listdir(os.path.join(train_folder, folder_name)):
                image = cv2.imread(os.path.join(train_folder, folder_name, image_name), IMREAD_GRAYSCALE)
                faces = faceCascade.detectMultiScale(
                    image,
                    scaleFactor=1.3,
                    minNeighbors=5,
                    minSize=(10, 10),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

                for (x, y, w, h) in faces:
                    image = image[y:y + h, x:x + w]

                image = cv2.resize(image, (100, 100))
                cv2.imwrite(os.path.join(train_folder, folder_name, image_name), image)
