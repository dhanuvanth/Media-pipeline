import cv2
import mediapipe as mp
import time
import numpy as np
from global_var import screenResolution, posePointsColor, videoCam
from global_var import rectangleScale as rs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d','--debug', help='Check debug mode', required=False)
args = vars(parser.parse_args())

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
cap = cv2.VideoCapture(videoCam)

pTime = 0
while True:
    
    success, img = cap.read()
    img = cv2.resize(img, screenResolution)
    imgClone = np.zeros(img.shape,dtype=np.uint8)
    imgClone.fill(255) # or img[:] = 255
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            # print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            if args["debug"]:
                cv2.rectangle(img, (cx - rs[0], cy + rs[1]), (cx + rs[0], cy - rs[1]), posePointsColor, cv2.FILLED)
           

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (40, 40), cv2.FONT_HERSHEY_PLAIN, 3,
            (255, 0, 250), 3)
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break