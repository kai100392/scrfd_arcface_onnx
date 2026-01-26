import cv2
import numpy as np
import time

# Optional: GPIO import if running on hardware
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# ============= USER SETTINGS =============
RTSP_URLS = [
    "rtsp://user:pass@192.168.1.10:554/stream1",
    # Add more (up to 20) in a list
]

MODEL_PATH = "yolov8n.onnx"
CONF_THRESHOLD = 0.4
AREA_THRESHOLD = 20000
CONFIRM_FRAMES = 3

# GPIO settings (if using hardware)
ALARM_PIN = 18

# If running on a machine without GPIO (PC), this will simply skip.
if GPIO_AVAILABLE:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(ALARM_PIN, GPIO.OUT)
    GPIO.output(ALARM_PIN, GPIO.LOW)

def trigger_alarm(on=True):
    if GPIO_AVAILABLE:
        GPIO.output(ALARM_PIN, GPIO.HIGH if on else GPIO.LOW)
    else:
        print("🔔 Alarm ON!" if on else "Alarm OFF")

# ============ LOAD MODEL ===============
net = cv2.dnn.readNetFromONNX(MODEL_PATH)

# Store state per camera
cam_states = [{"count": 0} for _ in RTSP_URLS]

# ============ OPEN STREAMS =============
caps = [cv2.VideoCapture(url, cv2.CAP_FFMPEG) for url in RTSP_URLS]

# Check opened
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"❌ Camera {i} failed to open")

print("▶️ Running detection...")

while True:
    for idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize to YOLO input size
        frame_resized = cv2.resize(frame, (640, 360))
        H, W = frame_resized.shape[:2]

        # Prepare ONNX input
        blob = cv2.dnn.blobFromImage(
            frame_resized, 1/255.0, (640, 640),
            swapRB=True, crop=False
        )
        net.setInput(blob)
        output = net.forward()[0]  # model outputs single array

        detected_person = False

        # Iterate detections
        for det in output:
            conf = det[4]
            if conf < CONF_THRESHOLD:
                continue

            class_scores = det[5:]
            class_id = int(np.argmax(class_scores))

            # person class id is 0
            if class_id != 0:
                continue

            # YOLOv8 center/w/h outputs
            cx, cy, w, h = det[:4]
            x1 = int((cx - w/2) * W)
            y1 = int((cy - h/2) * H)
            x2 = int((cx + w/2) * W)
            y2 = int((cy + h/2) * H)

            area = (x2 - x1) * (y2 - y1)

            # Only count person if area above threshold
            if area > AREA_THRESHOLD:
                detected_person = True
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_resized, f"Area: {area}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Confirmation logic
        state = cam_states[idx]
        if detected_person:
            state["count"] += 1
        else:
            state["count"] = 0
            trigger_alarm(False)

        if state["count"] >= CONFIRM_FRAMES:
            trigger_alarm(True)
            cv2.putText(frame_resized, "ALARM!", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        # Show (optional)
        cv2.imshow(f"Cam {idx}", frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
if GPIO_AVAILABLE:
    GPIO.cleanup()
