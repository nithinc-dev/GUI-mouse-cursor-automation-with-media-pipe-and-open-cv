import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize OpenCV
cap = cv2.VideoCapture(0)

screen_width, screen_height = pyautogui.size()
cam_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
cam_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Initialize timers and states
left_click_timer = None
right_click_timer = None
hold_timer = None
left_double_click_timer = None
holding = False


def get_distance(landmark1, landmark2):
    return ((landmark1.x - landmark2.x) ** 2 + (landmark1.y - landmark2.y) ** 2) ** 0.5


def detect_gesture(hand_landmarks):
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    distance_index_thumb = get_distance(index_finger_tip, thumb_tip)
    distance_middle_thumb = get_distance(middle_finger_tip, thumb_tip)
    distance_ring_thumb = get_distance(ring_finger_tip, thumb_tip)
    distance_pinky_thumb = get_distance(pinky_tip, thumb_tip)

    distance_index_pinky = get_distance(index_finger_tip, pinky_tip)
    distance_thumb_pinky = get_distance(thumb_tip, pinky_tip)

    return distance_index_thumb, distance_middle_thumb, distance_ring_thumb, distance_pinky_thumb, distance_index_pinky, distance_thumb_pinky


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x = int(index_finger_tip.x * screen_width)
            y = int(index_finger_tip.y * screen_height)

            # Move the mouse
            pyautogui.moveTo(x, y)

            (distance_index_thumb, distance_middle_thumb, distance_ring_thumb, distance_pinky_thumb,
             distance_index_pinky, distance_thumb_pinky) = detect_gesture(hand_landmarks)

            # Left click (thumb and index finger close for 1 second)
            if distance_index_thumb < 0.05:
                if left_click_timer is None:
                    left_click_timer = time.time()
                elif time.time() - left_click_timer > 0.5 and not holding:
                    pyautogui.click()
                    left_click_timer = None
                    hold_timer = time.time()
            else:
                left_click_timer = None

            # Right click (thumb and middle finger close for 1 second)
            if distance_middle_thumb < 0.05:
                if right_click_timer is None:
                    right_click_timer = time.time()
                elif time.time() - right_click_timer > 0.5:
                    pyautogui.click(button='right')
                    right_click_timer = None
            else:
                right_click_timer = None

            # Hold (all fingers close to thumb)
            if distance_index_thumb < 0.05 and distance_middle_thumb < 0.05 and distance_ring_thumb < 0.05 and distance_pinky_thumb < 0.05:
                if hold_timer is not None and time.time() - hold_timer > 1:
                    pyautogui.mouseDown()
                    holding = True
            else:
                if holding:
                    pyautogui.mouseUp()
                    holding = False
                hold_timer = None

                # Double left click (thumb and ring finger close twice quickly)
                if distance_ring_thumb < 0.05:
                    if left_double_click_timer is not None and time.time() - left_double_click_timer < 0.5:
                        pyautogui.doubleClick()
                        left_double_click_timer = None
                    else:
                        left_double_click_timer = time.time()
                else:
                    left_double_click_timer = None

            # Scroll up (index finger tip and pinky tip close together)
            if distance_index_pinky < 0.05:
                pyautogui.scroll(100)

            # Scroll down (thumb tip and pinky tip close together)
            if distance_thumb_pinky < 0.05:
                pyautogui.scroll(-100)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit on pressing ESC
        break

cap.release()
cv2.destroyAllWindows()
