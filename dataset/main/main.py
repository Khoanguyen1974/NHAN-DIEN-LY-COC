import cv2 
import cvzone
import math
from ultralytics import YOLO

cap = cv2.VideoCapture(0)  # Camera index 2, có thể cần thay đổi nếu camera không hoạt động

model = YOLO(r"D:\bandau\runs\detect\train9\weights\best.pt")

className = ['ly coc']

while True:
    success, img = cap.read()
    if not success:  # Kiểm tra xem camera có đọc được khung hình không
        print("Không thể đọc từ camera.")
        break

    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Tính độ tin cậy
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Chỉ xử lý nếu độ tin cậy >= 0.9 (90%)
            if conf >= 0.85:
                # Lấy tọa độ hộp
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # Tính kích thước hộp
                rong, cao = x2 - x1, y2 - y1
                # Vẽ hộp góc bằng cvzone
                cvzone.cornerRect(img, (x1, y1, rong, cao))
                # Lấy chỉ số lớp
                cls = int(box.cls[0])
                # Hiển thị tên lớp và độ tin cậy
                cvzone.putTextRect(img, f'{className[cls]} {conf}', (max(0, x1), max(35, y1)))
                print(f"Phát hiện: {className[cls]} với độ tin cậy {conf}")

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Thoát khi nhấn 'q'
        break

cap.release()
cv2.destroyAllWindows()