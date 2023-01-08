import easyocr
import cv2
from utils import image_resize

db = 'Mosaic, Glasses, Stash, Raw'  # 預先設置接收的Mode
mode = ''
cap = cv2.VideoCapture(0)  # 取得鏡頭
reader = easyocr.Reader(['en'])  # 宣告EasyOCR的TestReader
# 宣告Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
eyes_cascade = cv2.CascadeClassifier('cascades/third-party/frontalEyes35x16.xml')
nose_cascade = cv2.CascadeClassifier('cascades/third-party/Nose18x15.xml')

if not cap.isOpened():  # 判斷鏡頭是否可以打開
    print("Cannot open camera.")
    exit()
while True:  # 鏡頭Frame迴圈
    ret, frame = cap.read()
    if not ret:  # 判斷是否有讀到Frame
        print("Cannot receive frame.")
        break

    for detection in reader.readtext(frame):  # 讀取每一偵的文字
        text = detection[1]
        if text in db:
            mode = text
            posTL = tuple(detection[0][0])  # 取得檢測文字的左上座標
            posBR = tuple(detection[0][2])  # 取得檢測文字的右下座標
            frame = cv2.rectangle(frame, (int(posTL[0]), int(posTL[1])), (int(posBR[0]), int(posBR[1])), (0, 255, 0), 3)
            frame = cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 影像轉灰階
    for (x, y, w, h) in face_cascade.detectMultiScale(gray):  # 偵測人臉
        if mode == 'Mosaic':
            roi_face = frame[y:y + h, x:x + w]  # 拉出方框
            level = 10  # 模糊程度
            mh = int(h / level)
            mw = int(w / level)
            # 縮放圖片達成馬賽克效果
            roi_face = cv2.resize(roi_face, (mw, mh), interpolation=cv2.INTER_LINEAR)
            roi_face = cv2.resize(roi_face, (w, h), interpolation=cv2.INTER_NEAREST)
            frame[y:y + h, x:x + w] = roi_face  # 將原影像位置改為馬賽克

        if mode == 'Glasses':
            glasses = cv2.imread("images/glasses.png", -1)  # 取得去背墨鏡圖, -1改為BGRA色彩空間(A=Alpla通道, 透明度)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)  # 將整個Frame也轉為BGRA色彩空間
            roi_gray = gray[y:y + h, x:x + h]  # 拉出灰階影像臉部方框
            roi_raw = frame[y:y + h, x:x + h]  # 拉出原影像臉部方框
            for (ex, ey, ew, eh) in eyes_cascade.detectMultiScale(roi_gray):  # 由人臉再偵測眼部
                roi_eyes = roi_gray[ey: ey + eh, ex: ex + ew]  # 拉出灰階影像眼部方框
                resize_glasses = image_resize(glasses.copy(), width=ew)  # 將墨鏡圖的Size隨偵測到的眼部縮小
                # 以下將影像逐格取代為墨鏡圖
                gw, gh, gc = resize_glasses.shape
                for i in range(0, gw):
                    for j in range(0, gh):
                        if resize_glasses[i, j][3] != 0:  # 防止將方框全部填滿
                            roi_raw[ey + i, ex + j] = resize_glasses[i, j]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # 將整個Frame轉回BGR色彩空間

        if mode == 'Stash':
            mustache = cv2.imread('images/mustache.png', -1)  # 取得去背墨鏡圖, -1改為BGRA色彩空間(A=Alpla通道, 透明度)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)  # 將整個Frame也轉為BGRA色彩空間
            roi_gray = gray[y:y + h, x:x + h]  # 拉出灰階影像臉部方框
            roi_raw = frame[y:y + h, x:x + h]  # 拉出原影像臉部方框
            for (nx, ny, nw, nh) in nose_cascade.detectMultiScale(roi_gray):  # 由人臉再偵測鼻部
                roi_nose = roi_gray[ny: ny + nh, nx: nx + nw]  # 拉出灰階影像鼻部方框
                resize_mustache = image_resize(mustache.copy(), width=nw)  # 將鬍子圖的Size隨偵測到的鼻部縮小
                # 以下將影像逐格取代為鬍子圖
                mw, mh, mc = resize_mustache.shape
                for i in range(0, mw):
                    for j in range(0, mh):
                        if resize_mustache[i, j][3] != 0:  # 防止將方框全部填滿
                            roi_raw[ny + int(nh / 2.0) + i, nx + j] = resize_mustache[i, j]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # 將整個Frame轉回BGR色彩空間

    cv2.imshow('webcam', frame)  # 顯示Frame
    if cv2.waitKey(1) == ord('q'):  # 按下'q'鍵停止
        break

cap.release()
cv2.destroyAllWindows()

# 注1: 解決tuple的小數型態造成rectangle crash問題
# https://stackoverflow.com/questions/67921192/5bad-argument-in-function-rectangle-cant-parse-pt1-sequence-item-wit