import cv2
import dlib
import numpy as np
import pyautogui

# Hàm tính trung điểm
def midpoint(p1, p2):
    return (int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2))

# Hàm tính EAR (Eye Aspect Ratio)
def calculate_ear(eye_points, landmarks):
    points = [landmarks.part(point) for point in eye_points]
    p1, p2, p3, p4, p5, p6 = points
    vertical_1 = np.linalg.norm((p2.x - p6.x, p2.y - p6.y))
    vertical_2 = np.linalg.norm((p3.x - p5.x, p3.y - p5.y))
    horizontal = np.linalg.norm((p1.x - p4.x, p1.y - p4.y))
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# Hàm xác định hướng nhìn
def get_eye_direction(eye_points, landmarks, frame, label):
    points = []
    for point in eye_points:
        x, y = landmarks.part(point).x, landmarks.part(point).y
        points.append((x, y))

    # Lấy các điểm chính của mắt
    left_corner = points[0]
    right_corner = points[3]
    top_center = midpoint(points[1], points[2])
    bottom_center = midpoint(points[4], points[5])
    eye_center = midpoint(left_corner, right_corner)

    # Vẽ vùng mắt và tâm mắt
    hull = cv2.convexHull(np.array(points))
    cv2.polylines(frame, [hull], True, (255, 0, 0), 2)
    cv2.circle(frame, eye_center, 4, (0, 0, 255), -1)

    # Dựa vào vị trí tâm mắt
    horizontal_ratio = (eye_center[0] - left_corner[0]) / (right_corner[0] - left_corner[0])
    vertical_ratio = (eye_center[1] - top_center[1]) / (bottom_center[1] - top_center[1])

    # Xác định hướng nhìn
    if horizontal_ratio < 0.4:
        direction = "Looking Left"
    elif horizontal_ratio > 0.6:
        direction = "Looking Right"
    elif vertical_ratio < 0.4:
        direction = "Looking Up"
    elif vertical_ratio > 0.6:
        direction = "Looking Down"
    else:
        direction = "Looking Straight"

    # Hiển thị hướng nhìn
    cv2.putText(frame, f"{label}: {direction}", 
                (left_corner[0], left_corner[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return direction

# Khởi tạo bộ phát hiện và dự đoán
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Vùng mắt trái và mắt phải
left_eye_region = [36, 37, 38, 39, 40, 41]
right_eye_region = [42, 43, 44, 45, 46, 47]

# Ngưỡng EAR để phát hiện nháy mắt
EAR_THRESHOLD = 0.25
BLINK_FRAMES = 3
left_blink_count = 0
right_blink_count = 0

# Khởi động camera
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Phát hiện hướng mắt
        left_eye_direction = get_eye_direction(left_eye_region, landmarks, frame, "Left Eye")
        right_eye_direction = get_eye_direction(right_eye_region, landmarks, frame, "Right Eye")

        # Tính EAR
        left_ear = calculate_ear(left_eye_region, landmarks)
        right_ear = calculate_ear(right_eye_region, landmarks)

        # Phát hiện nháy mắt
        if left_ear < EAR_THRESHOLD:
            left_blink_count += 1
        else:
            if left_blink_count >= BLINK_FRAMES:
                pyautogui.click(button='left')
            left_blink_count = 0

        if right_ear < EAR_THRESHOLD:
            right_blink_count += 1
        else:
            if right_blink_count >= BLINK_FRAMES:
                pyautogui.click(button='right')
            right_blink_count = 0

        # Hiển thị EAR
        cv2.putText(frame, f"Left EAR: {left_ear:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Right EAR: {right_ear:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Eye Tracking with Blinking Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
