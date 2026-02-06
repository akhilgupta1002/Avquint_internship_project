import cv2
import numpy as np
from ultralytics import YOLO

# ============================
# SETTINGS
# ============================
MODEL_PATH = "yolo11m-seg-custom.pt"

EYE_CAM_INDEX = 1
FRONT_CAM_INDEX = 2

FRAME_W, FRAME_H = 640, 480

# Front camera projection scale (tune this)
SCALE_X = 2.5
SCALE_Y = 2.5

# ============================
# LOAD MODEL
# ============================
model = YOLO(MODEL_PATH)

# ============================
# CALIBRATION STORAGE
# ============================
eye_calib = {
    "center": None,
    "left": None,
    "right": None,
    "up": None,
    "down": None
}

eye_calibrated = False
front_calibrated = False
eye_center_ref = None  # reference pupil position for front cam

# ============================
# CAMERAS
# ============================
eye_cap = cv2.VideoCapture(EYE_CAM_INDEX)
front_cap = cv2.VideoCapture(FRONT_CAM_INDEX)

eye_cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
eye_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

front_cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
front_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

print("\n==== CONTROLS ====")
print("1 : Look CENTER")
print("2 : Look LEFT")
print("3 : Look RIGHT")
print("4 : Look UP")
print("5 : Look DOWN")
print("C : Calibrate front camera (look straight)")
print("Q : Quit\n")

# ============================
# MAIN LOOP
# ============================
while True:
    ret_eye, eye_frame = eye_cap.read()
    ret_front, front_frame = front_cap.read()

    if not ret_eye or not ret_front:
        print("Camera read failed")
        break

    eye_frame = cv2.resize(eye_frame, (FRAME_W, FRAME_H))
    front_frame = cv2.resize(front_frame, (FRAME_W, FRAME_H))

    pupil = None

    # ---------- EYE TRACKING ----------
    results = model.predict(eye_frame, imgsz=320, conf=0.4, verbose=False)

    if results[0].boxes and len(results[0].boxes) > 0:
        x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        pupil = (cx, cy)

        cv2.rectangle(eye_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(eye_frame, pupil, 5, (0, 0, 255), -1)

    cv2.putText(
        eye_frame,
        "1:CENTER 2:L 3:R 4:U 5:D  C:FRONT CAL  Q:QUIT",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow("Eye Camera", eye_frame)

    key = cv2.waitKey(1) & 0xFF

    # ---------- EYE CALIBRATION ----------
    if pupil:
        if key == ord('1'):
            eye_calib["center"] = pupil
            print("Eye Center:", pupil)
        elif key == ord('2'):
            eye_calib["left"] = pupil
            print("Eye Left:", pupil)
        elif key == ord('3'):
            eye_calib["right"] = pupil
            print("Eye Right:", pupil)
        elif key == ord('4'):
            eye_calib["up"] = pupil
            print("Eye Up:", pupil)
        elif key == ord('5'):
            eye_calib["down"] = pupil
            print("Eye Down:", pupil)

    if all(eye_calib.values()):
        eye_calibrated = True

    # ---------- FRONT CAMERA CALIBRATION ----------
    if key == ord('c') and pupil and eye_calibrated:
        eye_center_ref = pupil
        front_calibrated = True
        print("Front camera calibrated at eye position:", pupil)

    # ---------- GAZE → FRONT CAMERA ----------
    if front_calibrated and pupil:
        dx = pupil[0] - eye_center_ref[0]
        dy = pupil[1] - eye_center_ref[1]

        u = int(FRAME_W // 2 + dx * SCALE_X)
        v = int(FRAME_H // 2 + dy * SCALE_Y)

        u = np.clip(u, 0, FRAME_W - 1)
        v = np.clip(v, 0, FRAME_H - 1)

        # Draw gaze crosshair
        cv2.circle(front_frame, (u, v), 12, (0, 0, 255), 3)
        cv2.line(front_frame, (u - 20, v), (u + 20, v), (0, 0, 255), 2)
        cv2.line(front_frame, (u, v - 20), (u, v + 20), (0, 0, 255), 2)

    else:
        # Show center marker before calibration
        cv2.circle(front_frame, (FRAME_W // 2, FRAME_H // 2), 6, (255, 0, 0), -1)

    cv2.imshow("Front Camera (Gaze)", front_frame)

    if key == ord('q'):
        break

# ============================
# CLEANUP
# ============================
eye_cap.release()
front_cap.release()
cv2.destroyAllWindows()
