#Extract Landmarker points from hand for training

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import csv
import math
import urllib.request

# import mediapipe drawing utilities for hand skeleton rendering
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import HandLandmarksConnections
from mediapipe.tasks.python.vision import drawing_styles

# ==============================================================================
# FILE PATHS & SETUP CONFIGURATION
# ==============================================================================

# Output file where recorded gesture coordinate dataset will be saved
CSV_FILE_NAME = "hand_dataset.csv"

# Get the directory of the current script, then point to the central models folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True) # Ensure models directory exists

# MediaPipe Hand Landmarker Task file path (used for 3D hand skeleton tracking)
TASK_MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")

# incase cannot find locally.
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# To keep the setup fully automated, we download the required Tasks file if it doesn't exist
if not os.path.exists(TASK_MODEL_PATH):
    print("=" * 60)
    print(f"Model file '{TASK_MODEL_PATH}' not found locally.")
    print("Downloading the official MediaPipe Hand Landmarker Task asset...")
    print("Please wait, this will only happen once...")
    try:
        urllib.request.urlretrieve(MODEL_URL, TASK_MODEL_PATH)
        print("Download complete. Proceeding with detector setup...")
        print("=" * 60)
    except Exception as e:
        print(f"ERROR: Failed to download model file. Details: {e}")
        exit()


# ==============================================================================
# MEDIAPIPE TASKS DETECTOR INITIALIZATION
# ==============================================================================

# Set up options for the modern MediaPipe Tasks pipeline
# base_options = python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
# options = vision.HandLandmarkerOptions(
#     base_options=base_options,
#     num_hands=1,                         # Single hand focus
#     min_hand_detection_confidence=0.7    # High threshold for high training confidence
# )
base_options = python.BaseOptions(model_asset_path=TASK_MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,                          # Single hand focus to keep control clean
    min_hand_detection_confidence=0.7,    # High threshold for high training confidence
    min_hand_presence_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE # Synchronous image mode for precise frame processing
)

# Initialize the detector
detector = vision.HandLandmarker.create_from_options(options)

# Hand joints connection list drawing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]


# ==============================================================================
# DATASET HEADER & CSV INITIALIZATION
# ==============================================================================

# 21 hand landmarks, each containing 3 coordinates (x, y, z)
header = ['label']
for i in range(21):
    header.extend([f'x{i}', f'y{i}', f'z{i}'])

# If CSV doesn't exist, create it with the header
file_exists = os.path.exists(CSV_FILE_NAME)
if not file_exists:
    with open(CSV_FILE_NAME, mode='w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(header)
    print(f"Created new coordinate dataset file: '{CSV_FILE_NAME}'")
else:
    print(f"Existing dataset found. New records will append to '{CSV_FILE_NAME}'")


# ==============================================================================
# WEBCAM INTERACTION & USER PROMPT
# ==============================================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam. Make sure your camera is connected and not in use by another app.")
    exit()

print("\n" + "="*50)
print("Gesture Coordinate Collector (MediaPipe Tasks API)")
print("="*50)
current_label = input("Enter the name of the gesture you want to record: ").strip()
while not current_label:
    current_label = input("Label cannot be empty. Enter gesture name: ").strip()

recorded_count = 0
is_recording = False
was_r_pressed = False

# Memory buffer to hold frames before writing to disk in batches (prevents I/O lag)
data_buffer = []

print(f"\nTarget gesture set to: '{current_label}'")
print("\nCOMMANDS:")
print("  - Press 'R' : Toggle recording on/off")
print("  - Press 'N' : Change the target gesture label")
print("  - Press 'Q' : Quit and save")
print("="*50 + "\n")

def flush_buffer():
    """Flushes buffered coordinate rows to the CSV file to prevent frame stutters."""
    if not data_buffer:
        return
    with open(CSV_FILE_NAME, mode='a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(data_buffer)
    data_buffer.clear()


# ==============================================================================
# MAIN REAL-TIME COLLECTION LOOP
# ==============================================================================

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Mirror the frame for better interaction with screen
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # MediaPipe Tasks expects RGB images
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # Run hand landmarker synchronously
    results = detector.detect(mp_image)

    # Track if we are currently recording this frame
    key = cv2.waitKey(1) & 0xFF

    # logic to handle keyboard inputs
    if key == ord('q') or key == ord('Q'):
        # Flush any remaining frames in buffer before exiting
        flush_buffer()
        break
    elif key == ord('n') or key == ord('N'):
        # Flush memory buffer to CSV file before switching label
        flush_buffer()
        # Switch to a new class
        print(f"\nFinished recording '{current_label}'. Captured {recorded_count} frames.")
        new_label = input("Enter the new gesture name: ").strip()
        if new_label:
            current_label = new_label
            recorded_count = 0
            print(f"Target gesture changed to: '{current_label}'")

    # Toggle recording when 'R' is pressed (not while held)
    is_r_key = key == ord('r') or key == ord('R')
    if is_r_key and not was_r_pressed:
        is_recording = not is_recording
        if not is_recording:
            # Save data to disk immediately when recording pauses
            flush_buffer()
            print("Recording disabled and data saved to CSV.")
        else:
            print("Recording enabled.")
    was_r_pressed = is_r_key

    # Process hand landmark results
    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            # Use official Tasks API drawing utilities to draw hand skeleton on the frame
            drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style()
            )

            # Extract and Normalize Coordinates
            if is_recording:
                # Translation Invariance: Center coordinates relative to the wrist (landmark 0) 
                # (normalized so position of gesture in frame doesn't get taken into account.)
                wrist_x = hand_landmarks[0].x
                wrist_y = hand_landmarks[0].y
                wrist_z = hand_landmarks[0].z

                temp_coords = []
                for landmark in hand_landmarks:
                    temp_coords.append([
                        landmark.x - wrist_x,
                        landmark.y - wrist_y,
                        landmark.z - wrist_z
                    ])

                # Scale Invariance: Divide by the maximum Euclidean span of the hand 
                # so model will learn from hand geometry instead of percieving size.
                max_dist = 0.0
                for coord in temp_coords:
                    dist = math.sqrt(coord[0]**2 + coord[1]**2 + coord[2]**2)
                    if dist > max_dist:
                        max_dist = dist

                if max_dist == 0:
                    max_dist = 1.0  # Safe fallback to prevent division by zero

                # Construct clean training row: [label, x0, y0, z0, ..., x20, y20, z20]
                row = [current_label]
                for coord in temp_coords:
                    row.extend([coord[0] / max_dist, coord[1] / max_dist, coord[2] / max_dist])

                # Append to buffer in memory
                data_buffer.append(row)
                recorded_count += 1

    # Render on-screen HUD status overlay
    status_text = "RECORDING" if is_recording else "IDLE"
    status_color = (0, 0, 255) if is_recording else (0, 255, 0)
    
    cv2.putText(frame, f"Gesture: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {status_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, f"Saved Frames: {recorded_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "PRESS 'R' TO RECORD | PRESS 'N' TO CHANGE CLASS | 'Q' TO QUIT", (10, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Show the interactive frame window
    cv2.imshow("Gesture Coordinate Collector (Tasks API)", frame)

# ==============================================================================
# 6. SHUTDOWN & CLEANUP
# ==============================================================================

cap.release()
cv2.destroyAllWindows()
detector.close()

print("\n" + "="*50)
print("     COLLECTION RUN COMPLETED")
print("="*50)
print(f"Data successfully appended to: '{CSV_FILE_NAME}'")
print("You are now ready to run 'train_coordinate_model.py' to train your TensorFlow model!")
print("="*50 + "\n")