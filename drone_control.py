import os
import urllib.request
import time
import cv2
import numpy as np
import tensorflow as tf
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from djitellopy import Tello

# Import official MediaPipe Tasks API drawing components directly
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import HandLandmarksConnections
from mediapipe.tasks.python.vision import drawing_styles

MODEL_PATH = 'gesture_coordinate_model.keras'
LABEL_PATH = 'label_classes.pkl'
TASK_FILE = 'hand_landmarker.task'

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    print("❌ ERROR: Custom model or labels missing. Run 'train_coordinate_model.py' first.")
    exit()

if not os.path.exists(TASK_FILE):
    print(f"📥 Downloading '{TASK_FILE}'...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    try:
        urllib.request.urlretrieve(url, TASK_FILE)
        print("✅ Download finished.")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        exit()

# Load local weights and classes mapping
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, 'rb') as f:
    classes = pickle.load(f)

base_options = python.BaseOptions(model_asset_path=TASK_FILE)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

tello = Tello()
tello.connect()
print(f"🔋 Tello Battery Status: {tello.get_battery()}%")

tello.streamon()
frame_read = tello.get_frame_read()

# WARNING: Only uncomment these flight triggers in clear, wide-open environments
# tello.takeoff()
# tello.move_up(50)

print("🚁 Flight stream live. Press 'q' to disconnect and land.")
last_command_time = time.time()

while True:
    frame = frame_read.frame
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            
            # Use official Tasks API drawing utilities for standard aesthetic rendering
            drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style()
            )

            # Translation Invariance
            wrist = hand_landmarks[0]
            raw_points = [[lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z] for lm in hand_landmarks]
            
            # Scale Invariance
            distances = [np.linalg.norm(pt) for pt in raw_points]
            max_dist = max(distances)
            if max_dist == 0:
                max_dist = 1.0
            
            # Normalize and flatten
            normalized_points = [[pt[0]/max_dist, pt[1]/max_dist, pt[2]/max_dist] for pt in raw_points]
            flat_coords = np.array(normalized_points).flatten().reshape(1, -1)

            prediction = model.predict(flat_coords, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx]
            predicted_label = classes[class_idx]

            if confidence > 0.85:
                cv2.putText(frame, f"CMD: {predicted_label} ({confidence*100:.1f}%)", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # Command Throttle: Limit flight inputs to once every 1.5 seconds
                if time.time() - last_command_time > 1.5:
                    
                    if predicted_label == 'Orbit_CW':
                        print("Flight Command Issued: Orbiting")
                        tello.send_rc_control(50, 0, 0, 50) 
                        
                    elif predicted_label == 'Land_Emergency':
                        print("Flight Command Issued: Landing Drone")
                        tello.land()
                        
                    elif predicted_label == 'Hover':
                        print("Flight Command Issued: Hovering")
                        tello.send_rc_control(0, 0, 0, 0)
                        
                    last_command_time = time.time()

    cv2.imshow('Drone View', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        tello.land()
        break

# Clean termination
tello.streamoff()
cv2.destroyAllWindows()