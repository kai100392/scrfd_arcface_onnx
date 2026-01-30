import cv2
import numpy as np
import time
import os

# Paths to model files
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"

# Load MobileNet SSD
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# Object classes MobileNet SSD can detect
CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# IP Camera stream (change this)
IP_CAMERA_URL = "rtsp://username:password@IP:PORT/stream"

cap = cv2.VideoCapture(IP_CAMERA_URL)

prev_person_boxes = []
movement_threshold = 20   # pixels
confidence_threshold = 0.5

def play_alarm():
    # Simple beep (Linux)
    os.system("aplay /usr/share/sounds/alsa/Front_Center.wav")

print("[INFO] Starting person detection...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Camera not reachable")
        break

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    net.setInput(blob)
    detections = net.forward()

    current_person_boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            label = CLASSES[idx]

            if label == "person":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                current_person_boxes.append((x1, y1, x2, y2))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Person {confidence:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 2)

    # Check movement
    movement_detected = False

    if prev_person_boxes and current_person_boxes:
        for (px1, py1, px2, py2), (cx1, cy1, cx2, cy2) in zip(prev_person_boxes, current_person_boxes):
            if abs(cx1 - px1) > movement_threshold or abs(cy1 - py1) > movement_threshold:
                movement_detected = True
                break

    if movement_detected:
        print("[ALERT] Person movement detected!")
        play_alarm()
        cv2.putText(frame, "ALARM!",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0, 0, 255), 3)

    prev_person_boxes = current_person_boxes.copy()

    cv2.imshow("Person Alarm System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
