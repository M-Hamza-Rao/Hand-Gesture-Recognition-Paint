import time
import cv2
import HandTrackingModule as htm

pTime = 0
cTime = 0
detector = htm.HandDetector()
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=False)
    lmPosition = detector.findPosition(img, draw=True)
    if len(lmPosition) !=0:
        print(lmPosition[8])
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, "FPS:" + str(int(fps)), (5, 15), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 255), 2)

    cv2.imshow("image", img)
    cv2.waitKey(1)

