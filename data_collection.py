import time
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
counter=0
offset = 20
imgSize=300

folder="Data/Z"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize,imgSize,3), np.uint8)*255
        # Ensure values don't go out of bounds
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        imgCropShape = imgCrop.shape

        aspectRatio=h/w;
        if aspectRatio>1:
            k=imgSize/h #constant
            wCal = math.ceil(k*w) #width calculation with ceiling function always
            imgResize= cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape= imgResize.shape
            wGap = math.ceil((imgSize-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize
        else:
            k = imgSize / w  # constant
            hCal = math.ceil(k * h)  # width calculation with ceiling function always
            imgResize = cv2.resize(imgCrop, (imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap+hCal , :] = imgResize


        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite",imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)

