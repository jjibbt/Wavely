from pathlib import Path
import sys
APP_ROOT = Path(sys.executable).resolve().parent.parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
import cv2, json, os, numpy as np
from wavely_paths import CONFIG_DIR, FACES_DIR, prepare_user_data

prepare_user_data()
ROOT = str(APP_ROOT)
FACES = str(FACES_DIR)
MODEL = str(CONFIG_DIR / "face_model.yml")
LABELS = str(CONFIG_DIR / "face_labels.json")
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
faces, labels = [], []
people = sorted(folder for folder in os.listdir(FACES) if os.path.isdir(os.path.join(FACES, folder)))
print("WAVELY MULTI-PERSON FACE TRAINING\n===============================")
for label, person in enumerate(people):
    used = 0
    folder = os.path.join(FACES, person)
    for filename in sorted(os.listdir(folder)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        if filename.lower().startswith("video_"):
            crops = [image]
        else:
            found = detector.detectMultiScale(image, 1.1, 5, minSize=(70, 70))
            if len(found) == 0:
                continue
            x, y, w, h = max(found, key=lambda item: item[2] * item[3])
            crops = [image[y:y+h, x:x+w]]
        for crop in crops:
            crop = cv2.equalizeHist(cv2.resize(crop, (200, 200)))
            faces.extend((crop, cv2.flip(crop, 1)))
            labels.extend((label, label))
            used += 1
    print(f"{person}: {used} face images used")
if not faces:
    raise SystemExit("No usable face images were found.")
recognizer.train(faces, np.array(labels, dtype=np.int32))
recognizer.write(MODEL)
with open(LABELS, "w", encoding="utf-8") as file:
    json.dump({str(index): person for index, person in enumerate(people)}, file, indent=2)
print(f"\nTraining complete: {len(faces)} samples across {len(people)} people.")
