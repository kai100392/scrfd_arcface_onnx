import cv2
import numpy as np
import onnxruntime as ort

recognizer = ort.InferenceSession("mbf.onnx")

cap = cv2.VideoCapture(0)
name = input("Enter person name: ")

while True:
    ret, frame = cap.read()
    cv2.imshow("Capture Face", frame)

    if cv2.waitKey(1) & 0xFF == ord("s"):
        face = cv2.resize(frame, (112, 112))
        face = face.astype(np.float32) / 255.0
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)

        embedding = recognizer.run(None, {
            recognizer.get_inputs()[0].name: face
        })[0][0]

        np.save(f"known_faces/{name}.npy", embedding)
        print("[INFO] Face registered")
        break

cap.release()
cv2.destroyAllWindows()
