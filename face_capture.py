import cv2
import os

name = input("Enter your name: ").strip()
save_dir = os.path.join("faces", name)
os.makedirs(save_dir, exist_ok=True)

cam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    r"C:\Users\sajja\OneDrive\Desktop\Smart_Door_Lock\haarcascade_frontalface_default.xml"
)

count = 0

print("📸 Capturing face images... Press 'q' to stop")

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{save_dir}/{count}.jpg", face_img)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Capture Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 200:
        break

cam.release()
cv2.destroyAllWindows()
print(f"✅ Captured {count} images for {name}")
