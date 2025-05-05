import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import tensorflow as tf
# from tensorflow.keras.models import load_model  # Not needed for TFLite

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="Model/model_unquant.tflite")
interpreter.allocate_tensors()

# Load labels
labels = open("Model/labels.txt").read().splitlines()

counter = 0
offset = 20
imgSize = 300
labels = ["1","2","3","4","5","6","7","8","9","10"]

folder = "Data/C"

# ✨ Define getPrediction function
def getPrediction(imgWhite, interpreter, labels):
    input_data = np.expand_dims(cv2.resize(imgWhite, (224, 224)).astype(np.float32), axis=0) / 255.0
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    index = int(np.argmax(output))
    prediction = labels[index]
    return prediction, index

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure values don't go out of bounds
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # ✨ Get prediction and index
        prediction, index = getPrediction(imgWhite, interpreter, labels)
        print(prediction, index)

        # Draw output
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, prediction, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
