import cv2
import numpy as np
import onnxruntime as ort
import os

# ==============================
# Load ONNX models
# ==============================
detector = ort.InferenceSession(
    "scrfd_person_2.5g.onnx",
    providers=["CPUExecutionProvider"]
)

recognizer = ort.InferenceSession(
    "mbf.onnx",
    providers=["CPUExecutionProvider"]
)

# ==============================
# Load known faces
# ==============================
known_embeddings = {}
for file in os.listdir("known_faces"):
    if file.endswith(".npy"):
        name = file.replace(".npy", "")
        known_embeddings[name] = np.load(f"known_faces/{file}")

print(f"[INFO] Loaded {len(known_embeddings)} known faces")

# ==============================
# Utility functions
# ==============================
def preprocess_face(face):
    face = cv2.resize(face, (112, 112))
    face = face.astype(np.float32) / 255.0
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)
    return face

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ==============================
# IP Camera
# ==============================
cap = cv2.VideoCapture("rtsp://user:pass@IP:PORT/stream")

THRESHOLD = 0.45   # recognition threshold

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    img = cv2.resize(frame, (640, 480))

    blob = cv2.dnn.blobFromImage(
        img, 1.0 / 128, (640, 640),
        (127.5, 127.5, 127.5), swapRB=True
    )

    outputs = detector.run(None, {detector.get_inputs()[0].name: blob})

    boxes = outputs[0][0]

    for box in boxes:
        conf = box[4]
        if conf < 0.6:
            continue

        x1, y1, x2, y2 = map(int, box[:4])
        face = img[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face_input = preprocess_face(face)
        embedding = recognizer.run(None, {
            recognizer.get_inputs()[0].name: face_input
        })[0][0]

        best_match = "Unknown"
        best_score = 0

        for name, ref_emb in known_embeddings.items():
            score = cosine_similarity(embedding, ref_emb)
            if score > best_score:
                best_score = score
                best_match = name

        if best_score < THRESHOLD:
            best_match = "Unknown"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            f"{best_match} ({best_score:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
