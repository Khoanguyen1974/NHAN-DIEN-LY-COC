import cv2 
import cvzone
import math
from ultralytics import YOLO


cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

model = YOLO(r"D:\bandau\dataset\runs\detect\train\weights\best.pt")

className = ['ly coc']

while True:
    success, img = cap.read()
    results = model(img, stream= True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Lam cai hop:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (244, 0, 244), 3)            # Ti le du doan:
            conf = math.ceil((box.conf[0] * 100)) / 100
            rong, cao = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, rong, cao))
            # Ti le du doan:
            conf = math.ceil((box.conf[0] * 100)) / 100
            # classNAme
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{className[cls]} {conf}', (max(0, x1), max(35, y1)))   


    cv2.imshow("Image", img)
    cv2.waitKey(1)