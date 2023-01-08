import easyocr
import cv2
import matplotlib.pyplot as plot
import numpy as np

db = 'Mosic, ABC'
cap = cv2.VideoCapture(0)
reader = easyocr.Reader(['en'])

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()

    for detection in reader.readtext(frame):
        text = detection[1]
        if text in db:
            posTL = tuple(detection[0][0])
            posBR = tuple(detection[0][2])
            frame = cv2.rectangle(frame, (int(posTL[0]), int(posTL[1])), (int(posBR[0]), int(posBR[1])), (0, 255, 0), 3)
            frame = cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('webcam', frame)
    if cv2.waitKey(1) == ord('q'):
        break  # 按下 q 鍵停止

cap.release()
cv2.destroyAllWindows()

# 解決tuple的小數型態造成rectangle crash問題
# https://stackoverflow.com/questions/67921192/5bad-argument-in-function-rectangle-cant-parse-pt1-sequence-item-wit
