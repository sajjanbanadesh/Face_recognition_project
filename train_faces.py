import cv2
import numpy as np
import os
import pickle

# Folder where face images are stored
FACES_DIR = "faces"
face_size = (200, 200)  # Fixed size for all images

# Create LBPH recognizer (requires opencv-contrib-python)
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces_data = []
labels = []

label_id = 0
label_dict = {}  # Maps label_id -> person_name

# Loop through each person folder
for person_name in os.listdir(FACES_DIR):
    person_path = os.path.join(FACES_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    label_dict[label_id] = person_name

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print("Cannot read:", img_path)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, face_size)
        faces_data.append(gray)
        labels.append(label_id)

    label_id += 1

# Convert to numpy arrays
faces_data = np.array(faces_data)
labels = np.array(labels)

# Train LBPH recognizer
recognizer.train(faces_data, labels)
recognizer.save("lbph_model.yml")
print("Training completed successfully!")

# Save label mapping
with open("labels.pkl", "wb") as f:
    pickle.dump(label_dict, f)
print("Label mapping saved as labels.pkl")
