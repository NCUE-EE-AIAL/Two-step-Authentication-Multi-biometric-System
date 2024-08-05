import face_detect_crop
import pandas as pd
import os
import cv2

data = pd.read_csv("dataset.csv")

#image Input and Output directory
input_parent_path = "/Users/aryansmac/Documents/PROJECT/TEEP/Biometric_Authentication/Aryan_work/dataset"
output_parent_path = "/Users/aryansmac/Documents/PROJECT/TEEP/Biometric_Authentication/Aryan_work/output_dataset"

def face_crop_filter(data, input_parent_path, output_parent_path):
    for i in range(len(data)):
        input_path = os.path.join(input_parent_path, data["label"][i])
        input_path = os.path.join(input_path, data["image_id"][i])
        try:
            img = cv2.imread(input_path)
            croped_img = face_detect_crop.face_crop(img, input_path)
            if croped_img is not None:
                output_path = os.path.join(output_parent_path, data["label"][i])
                #creates Sub Folder if not present
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                output_path = os.path.join(output_path, data["image_id"][i])
                cv2.imwrite(output_path, croped_img)

        except:
            continue

#calling the function to create filter the face images
face_crop_filter(data, input_parent_path, output_parent_path)