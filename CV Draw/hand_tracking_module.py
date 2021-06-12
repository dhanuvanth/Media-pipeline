import cv2
import mediapipe as mp
import time

class handDetection():
    def __init__(self, mode=False, maxhands=2, detectionConf=0.5, trackConf=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHand = mp.solutions.hands
        self.hands = self.mpHand.Hands(self.mode,self.maxhands,self.detectionConf,self.trackConf)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4,8,12,16,20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mpHand.HAND_CONNECTIONS)
        
        return img

    def findPosition(self, img, hand=0, draw=True):

        self.lmsList = []
        if self.results.multi_hand_landmarks:
            hands = self.results.multi_hand_landmarks[hand]
            for id, lms in enumerate(hands.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lms.x * w), int(lms.y * h)
                    # print(id, cx, cy)
                    self.lmsList.append([id,cx,cy])
                    if draw:
                        cv2.circle(img, (cx,cy), 15, (0,255,255), cv2.FILLED)
        
        return self.lmsList

    def fingersUp(self):
        fingers = []

        if self.lmsList[self.tipIds[0]][1] < self.lmsList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if self.lmsList[self.tipIds[id]][2] < self.lmsList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detection = handDetection()

    while True:
        _,img = cap.read()

        # Detect Hands
        img = detection.findHands(img)

        # Landmark points list
        lmsList = detection.findPosition(img)
        if len(lmsList) != 0:
            print(lmsList[1])

        # Frame rate
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img,str(int(fps)),(20,40),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,250),2)

        cv2.imshow("Output", img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()


if __name__ == "__main__":
    main()