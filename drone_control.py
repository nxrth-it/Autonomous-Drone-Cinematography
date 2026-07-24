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


MODEL_PATH = 'models/gesture_coordinate_model.keras'
LABEL_PATH = 'label_classes.pkl' #this file is to correspond the model's integer index to string classes.

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    print("ERROR: Custom model or labels missing. Run 'train_coordinate_model.py' first.")
    exit()



# Load local model and labels
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, 'rb') as f:
    classes = pickle.load(f)



#SETUP MEDIAPIPE GESTURE RECOGNIZER ---
base_options = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE #not using live stream mode since processing it as 1 frame at a time.
    #result_callback=lambda result, output_image, timestamp_ms: update_result(result)
)
recognizer = vision.GestureRecognizer.create_from_options(options)

#=========================================================================



tello = Tello()
tello.connect()
print(f"Tello Battery Status: {tello.get_battery()}%")

tello.streamon()
frame_read = tello.get_frame_read()

# WARNING: Only uncomment these flight triggers in clear, wide-open environments
# tello.takeoff()
# tello.move_up(50)

print("Flight stream live. Press 'q' to disconnect and land.")
last_command_time = time.time()

while True:
    frame = frame_read.frame
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp_ms = int(time.time() * 1000)


   
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = recognizer.recognize(mp_image)
    if result.hand_landmarks:
         #--- REMOVE ONCE TESTING IS COMPLETE --- (DRAWING HAND LANDMARKS)
        for hand_landmarks in result.hand_landmarks:
            drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style()
            )
    #----------------------------------------------------------------------
            #Normalizing and setting up data for the custom gesture recognition model
            # Translation Invariance: Shift coordinates relative to wrist origin (normalized so position of gesture in frame doesn't get taken into account.)
            wrist = hand_landmarks[0]
            raw_points = []
            for lm in hand_landmarks:
                raw_points.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
            
            # Scale Invariance: Calculate max coordinate reach and normalize size so model will look at hand geometry instead of percieving size.
            distances = [np.linalg.norm(pt) for pt in raw_points]
            max_dist = max(distances)
            if max_dist == 0: #prevent division by zero if all points are at the wrist
                max_dist = 1.0
            
            # Normalize and structure into flat feature vector
            normalized_points = [[pt[0]/max_dist, pt[1]/max_dist, pt[2]/max_dist] for pt in raw_points]
            flat_coords = np.array(normalized_points).flatten().reshape(1, -1)


    #----------------------------------------------------------------------------


            #Predictions
            prediction = model.predict(flat_coords, verbose=0)
            class_idx = np.argmax(prediction) #return indices of largest value (since axis is None by default, it returns the index of the max value in the flattened array)
            confidence = prediction[0][class_idx] 
            predicted_label = classes[class_idx]

            if confidence > 0.85 and predicted_label != "undefined": #only act on gestures that the model is confident about and are known gestures
                text = f"{predicted_label} ({confidence*100:.1f}%)"
                cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)


                if predicted_label == "point_down" and time.time() - last_command_time > 1:
                    #tello.move_down(30)
                    print("Command: Move Down")
                    last_command_time = time.time()

                elif predicted_label == "three_fingers" and time.time() - last_command_time > 1:
                    #follow the person function call
                    last_command_time = time.time()
                    print("Command: Follow Person")
                    pass

                elif predicted_label == "flat_palm" and time.time() - last_command_time > 1:
                    #tello.land()
                    last_command_time = time.time()
                    print("Command: Land")

                elif predicted_label == "L_sign" and time.time() - last_command_time > 1:
                    #orbit function call
                    print("Command: Orbit")
                    last_command_time = time.time()

                elif predicted_label == "ok_sign" and time.time() - last_command_time > 1:
                    #tello.takeoff()
                    print("Command: Takeoff")
                    last_command_time = time.time()

            elif result.gestures and result.hand_landmarks: #if custom hasn't detected a gesture, check if mp has a valid gesture
                #mediapipe gesture recognition
                gesture_name = result.gestures[0][0].category_name
                cv2.putText(frame, f"Gesture: {gesture_name}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if gesture_name == "Closed_Fist" and time.time() - last_command_time > 1:
                    print("Command: Closed Fist Action")
                    last_command_time = time.time()

                elif gesture_name == "Open_Palm" and time.time() - last_command_time > 1:
                    print("Command: Open Palm Action")
                    last_command_time = time.time()

                elif gesture_name == "Pointing_Up" and time.time() - last_command_time > 1:
                    print("Command: Pointing Up Action")
                    last_command_time = time.time()

                elif gesture_name == "Thumb_Down" and time.time() - last_command_time > 1:
                    print("Command: Thumb Down Action")
                    last_command_time = time.time()

                elif gesture_name == "Thumb_Up" and time.time() - last_command_time > 1:
                    print("Command: Thumb Up Action (Takeoff)")
                    last_command_time = time.time()

                elif gesture_name == "Victory" and time.time() - last_command_time > 1:
                    print("Command: Victory Action")
                    last_command_time = time.time()

                elif gesture_name == "ILoveYou" and time.time() - last_command_time > 1:
                    print("Command: I Love You Action")
                    last_command_time = time.time()

            else:
                cv2.putText(frame, "Undefined Gesture", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            



    cv2.imshow('Drone View', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        tello.land()
        break

# Clean termination
tello.streamoff()
cv2.destroyAllWindows()