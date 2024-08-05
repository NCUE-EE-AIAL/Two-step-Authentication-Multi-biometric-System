import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import face_detect_crop
import cv2

model = load_model('face_model_vggface.h5')

#path = input("Enter the Path of Image:")

vc = cv2.VideoCapture(1)
if vc.isOpened():
  err, frame = vc.read()

#img = cv2.imread(path)
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cv2.imwrite("test.png", img)
img_croped = face_detect_crop.face_crop(img, 'path')


img_array = np.array([img_croped])
img_pridect = model.predict(img_array)
if img_pridect[0][np.argmax(img_pridect)] > 0.75:
    print("Match Found to:",np.argmax(img_pridect))
    print("With Accuracy: ", img_pridect[0][np.argmax(img_pridect)])
else:
    print("No Match Found")