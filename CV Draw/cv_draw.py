import cv2
import numpy as np
import hand_tracking_module as htm

imgCanvas = np.zeros((480, 640, 3), np.uint8)
xp, yp = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(3, 480)

hand = htm.handDetection(detectionConf=0.85)

# video format (XVID is more preferable. MJPG results in high size video. X264 gives very small size video)
four_cc = cv2.VideoWriter_fourcc(*'XVID')
# videoWriter("output name", format, frames, size)
out = cv2.VideoWriter('video_stream.mp4', four_cc, 24.0, (640,480))

while True:

    # Import the images
    result, img = cap.read()
    if  result != True:
        break
    img = cv2.flip(img, 1)

    # find Fingers
    img = hand.findHands(img, draw=True)
    lmsList = hand.findPosition(img, draw=False)

    # Check which finger is up
    if len(lmsList) != 0:
        # print(lmsList)

        x1, y1 = lmsList[8][1:]
        x2, y2 = lmsList[12][1:]

        finger = hand.fingersUp()
        # print(finger)

        if finger[1] and finger[2] == 0:
            cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            cv2.line(imgCanvas, (xp, yp), (x1, y1),
                     color=(255, 0, 0), thickness=15)

            xp, yp = x1, y1
        elif finger[1] and finger[2] and finger[3] and finger[4]:
            cv2.circle(img, (x1, y1), 50, (0, 0, 0), cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            cv2.line(imgCanvas, (xp, yp), (x1, y1),
                     color=(0, 0, 0), thickness=80)
            xp, yp = x1, y1
            # print("Erase mode")
        else:
            xp, yp = 0, 0
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imginv = cv2.threshold(imgGray, 10, 255, cv2.THRESH_BINARY_INV)
    imginv = cv2.cvtColor(imginv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imginv)
    img = cv2.bitwise_or(img, imgCanvas)

     # write the video
    out.write(img)

    cv2.imshow("Output", img)
    # cv2.imshow("Canvas", imgGray)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
