import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- TASKS API DRAWING UTILITIES ---
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles
mp_hands_connections = mp.tasks.vision.HandLandmarksConnections

# --- 1. SETUP GESTURE RECOGNIZER ---
# Ensure you have 'gesture_recognizer.task' in the same folder as this script
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=lambda result, output_image, timestamp_ms: update_result(result)
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Global storage for AI results
latest_result = None

def update_result(result):
    global latest_result
    latest_result = result

# --- 2. CAMERA SETUP ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    
    # Prepare image for MediaPipe
    timestamp_ms = int(time.time() * 1000)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Send to AI
    recognizer.recognize_async(mp_image, timestamp_ms)

    # --- 3. DRAW AND PRINT RESULTS ---
    if latest_result and latest_result.gestures and latest_result.hand_landmarks:
        #Display the detected gesture name on the frame
        gesture_name = latest_result.gestures[0][0].category_name
        cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw the hand skeleton
        for hand_landmarks in latest_result.hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands_connections.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    cv2.imshow('Camera Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
recognizer.close()
cv2.destroyAllWindows()