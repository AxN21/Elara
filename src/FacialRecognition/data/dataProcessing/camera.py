import uuid
import cv2
import os

# GET POSITIVE AND ANCHOR IMAGES FROM THE WEBCAM
ANC_PATH = "/home/axn/Desktop/Python projects/Elara/src/FacialRecognition/data/anchor"
POS_PATH = "/home/axn/Desktop/Python projects/Elara/src/FacialRecognition/data/positive"
VER_DATA = '/home/axn/Desktop/Python projects/Elara/src/FacialRecognition/application_data/verification_images'
# connection to the webcam
cap = cv2.VideoCapture('/dev/video0')
while cap.isOpened():
    ret, frame = cap.read()
    frame = frame[120:120+250,200:200+250, :]

    # Collect Anchors 
    if cv2.waitKey(1) & 0XFF == ord('v'):
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
    # Collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)

    cv2.imshow('Image Collection', frame)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break
cap.release() 
cv2.destroyAllWindows()
