import cv2
import numpy as np
import os
import pickle
from datetime import datetime

# Paths
FACES_DIR = "faces"
MODEL_PATH = "lbph_model.yml"
LABELS_PATH = "labels.pkl"
UNKNOWN_DIR = "unknown_faces"

# Create unknown_faces folder if not exists
if not os.path.exists(UNKNOWN_DIR):
    os.makedirs(UNKNOWN_DIR)

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# Load labels mapping
with open(LABELS_PATH, "rb") as f:
    labels = pickle.load(f)

# Load Haar cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
if face_cascade.empty():
    raise IOError("Cannot load Haarcascade XML file!")

# Open webcam
cap = cv2.VideoCapture(0)
face_size = (200, 200)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)


    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, face_size)

        # Predict
        label_id, confidence = recognizer.predict(roi_gray)
        similarity = max(0, min(100, 100 - confidence))  # 0-100 scale

        # Determine name
        if similarity < 60:
            name = "Unknown"
            # Save unknown face with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(os.path.join(UNKNOWN_DIR, f"unknown_{timestamp}.jpg"), roi_gray)
        else:
            name = labels.get(label_id, "Unknown")

        # Draw rectangle and text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({int(similarity)}%)", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Smart Door Lock", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
