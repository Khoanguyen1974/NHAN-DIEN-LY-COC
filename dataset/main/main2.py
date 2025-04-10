import cv2
from ultralytics import YOLO

# Danh sách các tên lớp (điền đúng theo mô hình của bạn)
class_names = ['ly coc']  # Ví dụ có 3 lớp

# Tải mô hình YOLO đã huấn luyện
model = YOLO(r"D:\bandau\runs\detect\train9\weights\best.pt")  # Đường dẫn tới mô hình .pt

# Đọc ảnh bất kỳ
img = cv2.imread(r"dataset/dataset/train/images/ly (19).jpg")

# Nhận diện ảnh
results = model(img)

# Hiển thị kết quả
for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].int().tolist()
        confidence = box.conf[0].item()
        class_id = int(box.cls[0].item())  # Đổi về int để tra tên lớp

        # Ghi nhãn: tên lớp + độ tin cậy
        label = f'{class_names[class_id]} ({confidence:.2f})'
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Hiển thị ảnh
cv2.imshow("Detected Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
