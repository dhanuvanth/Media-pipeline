import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture('video.mp4')
pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = cv2.resize(img,(640,480))
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) !=0:
        #print(lmList[14])
        [cv2.rectangle(img, (lmList[i][1]-10, lmList[i][2]+10), (lmList[i][1]+10, lmList[i][2]-10), (0, 0, 255), cv2.FILLED) for i in range(len(lmList))]

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)