from mtcnn.mtcnn import MTCNN
import pandas as pd
import os
import cv2

data = pd.read_csv("dataset.csv")

#image Input and Output directory
input_parent_path = "/Users/aryansmac/Documents/PROJECT/TEEP/Biometric_Authentication/Aryan_work/dataset"
output_parent_path = "/Users/aryansmac/Documents/PROJECT/TEEP/Biometric_Authentication/Aryan_work/output_dataset"

def face_crop(img, path):
    detector = MTCNN()
    face_detect = detector.detect_faces(img)
    print(path)
    # bounding_box = [face['box'] for face in face_detect][0]

    if face_detect:
        for face in face_detect:
            x, y, width, height = face['box']

        # introducing padding to get squared (n*n) pixel image
        padding = (max(width, height) - min(width, height))
        x = x - (padding // 2)
        width = width + (padding)

        cropped_face = img[y:y + height, x:x + width]
        cropped_face = cv2.resize(cropped_face, (224, 224), interpolation=cv2.INTER_AREA)
        return cropped_face
    else:
        return


def face_crop_filter(data, input_parent_path, output_parent_path):
    for i in range(len(data)):
        input_path = os.path.join(input_parent_path, data["label"][i])
        input_path = os.path.join(input_path, data["image_id"][i])
        try:
            img = cv2.imread(input_path)
            croped_img = face_crop(img, input_path)
            if croped_img is not None:
                output_path = os.path.join(output_parent_path, data["label"][i])
                #creates Sub Folder if not present
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                output_path = os.path.join(output_path, data["image_id"][i])
                cv2.imwrite(output_path, croped_img)

        except:
            continue

# test
#img = cv2.cvtColor(cv2.imread("demo.JPG"), cv2.COLOR_BGR2RGB)
#img_test = cv2.imread("frame_30.jpg")
#img_test = face_crop(img_test, "IMG_4127.jpg")
#cv2.imwrite('cropped_face7.jpg', img_test)

#calling the function to create filter the face images
face_crop_filter(data, input_parent_path, output_parent_path)