import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    _,img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLMS in results.multi_hand_landmarks:
            for id, lms in enumerate(handLMS.landmark):
                h, w, c = img.shape
                cx, cy = int(lms.x * w), int(lms.y * h)
                # print(id, cx, cy)
                if id == 4:
                    cv2.circle(img, (cx,cy), 15, (0,255,255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLMS, mpHand.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(20,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,250),2)    

    cv2.imshow("Output", img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break