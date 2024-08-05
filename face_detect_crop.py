from mtcnn.mtcnn import MTCNN
import cv2

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
        x = x - (padding//2)
        width = width + (padding)
    
        cropped_face = img[y:y+height, x:x+width]
        cropped_face = cv2.resize(cropped_face, (224, 224), interpolation=cv2.INTER_AREA)
        return cropped_face
    else:
        return
    
#img = cv2.cvtColor(cv2.imread("demo.JPG"), cv2.COLOR_BGR2RGB)
#img_test = cv2.imread("frame_30.jpg")
#img_test = face_crop(img_test, "IMG_4127.jpg")
#cv2.imwrite('cropped_face7.jpg', img_test)