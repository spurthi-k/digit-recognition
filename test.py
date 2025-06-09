# Load TFLite model
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
from transformers import AutoProcessor, AutoModelForImageClassification
import torch
from PIL import Image

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-classification", model="prithivMLmods/Alphabet-Sign-Language-Detection")

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load labels
labels = open('Model/labels.txt').read().splitlines()

counter = 0
offset = 20
imgSize = 300

folder = "Data/C"

def getPrediction(imgWhite, processor, model):
    # Convert OpenCV image (BGR) to RGB
    img_rgb = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    inputs = processor(images=img_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    prediction = model.config.id2label[predicted_class_idx]
    return prediction, predicted_class_idx

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
        # Get prediction using the pipeline
        img_rgb = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Pass the PIL image to the pipeline
        outputs = pipe(img_pil)
        prediction = outputs[0]['label']
        index = outputs[0]['score']
        print(prediction)

        # Draw output
        cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, prediction, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)