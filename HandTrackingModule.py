import time
import cv2
import mediapipe as mp



class HandDetector():
    def __init__(self, mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 1,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image=imgRGB)
        #print(self.results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        self.lmposition = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmposition.append([id,cx,cy])
                #print(cx, cy)
                if draw:
                    if id == 8:
                        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        return self.lmposition

    def fingersUp(self):
        fingers = []

        if self.lmposition[self.tipIds[0]][1]>self.lmposition[self.tipIds[0]-1][1]:
            fingers.append(0) #reversing due to inversion of image
        else:
            fingers.append(1)

        for id in range(1,5):
            if self.lmposition[self.tipIds[id]][2] < self.lmposition[self.tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers



def main():
    pTime = 0
    cTime = 0
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmPosition = detector.findPosition(img)
        if len(lmPosition) !=0:
            print(lmPosition[8])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, "FPS:" + str(int(fps)), (5, 15), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 2)

        cv2.imshow("image", img)
        cv2.waitKey(1)





if __name__ == "__main__":
    main()
