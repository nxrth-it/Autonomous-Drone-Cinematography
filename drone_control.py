#Current Latest Stable Version for some reason. To see latest version, unstable look at timeline July 29 2026, 9:11 pm


#Proposed Image Cleaning to improve mediapipe line drawing:
# # 1. Sharpen the drone frame to kill the Wi-Fi compression blur
# kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# sharpened_frame = cv2.filter2D(bgr_frame, -1, kernel)

# # 2. Boost the contrast and brightness (histogram equalization)
# # This mimics the webcam's automatic indoor lighting adjustment
# lab = cv2.cvtColor(sharpened_frame, cv2.COLOR_BGR2LAB)
# l_channel, a, b = cv2.split(lab)
# clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
# cl = clahe.apply(l_channel)
# limg = cv2.merge((cl, a, b))
# final_processed_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

# # NOW pass 'final_processed_frame' to your gesture detector instead of the raw drone frame!





import os
import time
import cv2
import numpy as np
import tensorflow as tf
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from djitellopy import Tello
from drone_tello import DroneTello as drone
from funcs import *
import threading



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
    classes = pickle.load(f) #map model's numeric output to text labels



#SETUP MEDIAPIPE GESTURE RECOGNIZER ---
base_options = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE #not using live stream mode since processing it as 1 frame at a time.
    #result_callback=lambda result, output_image, timestamp_ms: update_result(result)
)
recognizer = vision.GestureRecognizer.create_from_options(options)

#=========================================================================

d = drone(enable_mission_pad=False, show_cam=False)
d.streamon()
time.sleep(3.5)
frame_read = d.get_frame_read()


print("Flight stream live. Press 'q' to disconnect and land.")
last_command_time = time.time()
prev_post = None

while True:
    frame = frame_read.frame
    print(frame.shape)

        #Skip invalid/empty initial frames from Tello
    if frame is None or frame.size == 0:
        time.sleep(0.01)
        continue

    native_h, native_w, _ = frame.shape
    target_w = 900
    target_h = int(target_w * native_h / native_w)
    frame = cv2.resize(frame, (target_w, target_h))

    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #convert to BGR only for cv2.imshow()
    rgb_frame = frame #mediapipe raw rgb frame
    timestamp_ms = int(time.time() * 1000)


   
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = recognizer.recognize(mp_image)

    is_actively_swiping = False

    if result.hand_landmarks:
        #loop through landmarks
        for hand_landmarks in result.hand_landmarks:
            #--- REMOVE ONCE TESTING IS COMPLETE --- (DRAWING HAND LANDMARKS)
            drawing_utils.draw_landmarks(
                display_frame,
                hand_landmarks,
                HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style()
            )
    #----------------------------------------------------------------------
            #Normalizing and setting up data for the custom gesture recognition model
            # Translation Invariance: Shift coordinates relative to wrist origin (normalized so position of gesture in frame doesn't get taken into account.)
            wrist = hand_landmarks[0]

            index_tip = hand_landmarks[8]
            curr_pos = (index_tip.x, index_tip.y)

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
            prediction = model(flat_coords, training=False).numpy() #turn off training and verbose to increase speed. Verbose is output to terminal and training is graph drawing.
            #changed from model.predict to model as model is faster. Also added .numpy() in order to change it into a numpy array.
            class_idx = np.argmax(prediction) #return indices of largest value (since axis is None by default, it returns the index of the max value in the flattened array)
            confidence = prediction[0][class_idx] 
            predicted_label = classes[class_idx]

            if confidence > 0.85 and predicted_label != "undefined": #only act on gestures that the model is confident about and are known gestures
                text = f"{predicted_label} ({confidence*100:.1f}%)"
                cv2.putText(display_frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)


                if predicted_label == "point_down":
                    last_command_time = time.time()
                    command(d.move_down, "Move Down", 30)

                elif predicted_label == "three_fingers":
                    #follow the person function call
                    last_command_time = time.time()
                    print("Command: Follow Person")
                    pass

                elif predicted_label == "flat_palm":
                    #tello.land()
                    last_command_time = time.time()
                    print("Command: Land")
                    command(d.land, "Landing")

                elif predicted_label == "L_sign":
                    #orbit function call
                    print("Command: Orbit")
                    last_command_time = time.time()

                elif predicted_label == "ok_sign":
                    #tello.takeoff()
                    print("Command: Takeoff")
                    last_command_time = time.time()
                    command(d.takeoff, "Takeoff")

            elif result.gestures and result.hand_landmarks: #if custom hasn't detected a gesture, check if mp has a valid gesture
                #mediapipe gesture recognition
                gesture_name = result.gestures[0][0].category_name
                print(f"DEBUG: MP GESTURE: {gesture_name}, score: {result.gestures[0][0].score:.2f}")
                cv2.putText(display_frame, f"Gesture: {gesture_name}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if gesture_name == "Closed_Fist":
                    print("Command: Closed Fist Action")
                    last_command_time = time.time()

                elif gesture_name == "Open_Palm":
                    print("Command: Open Palm Action")
                    last_command_time = time.time()

                elif gesture_name == "Pointing_Up":
                    print("Command: Pointing Up Action")
                    last_command_time = time.time()
                    is_actively_swiping = True
                    prev_post = swipe_control(d, curr_pos, prev_post, threshold=0.065, rc_speed=30)



                elif gesture_name == "Thumb_Down":
                    print("Command: Thumb Down Action")
                    last_command_time = time.time()
                    d.send_rc_control(0,0,0,0)

                elif gesture_name == "Thumb_Up":
                    print("Command: Thumb Up Action (Takeoff)")
                    last_command_time = time.time()

                elif gesture_name == "Victory":
                    print("Command: Victory Action")
                    last_command_time = time.time()

                elif gesture_name == "ILoveYou":
                    print("Command: I Love You Action")
                    last_command_time = time.time()
                else:
                    print("MP: Unmatched Gesture")
                    print(f"DEBUG: {gesture_name}, score: {result.gestures[0][0].score:.3f}")

            else:
                cv2.putText(display_frame, "Undefined Gesture", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(display_frame, "Undefined Gesture", (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 

    if not is_actively_swiping and prev_post is not None:
        print("Safety: Stop vector issued.")
        d.send_rc_control(0, 0, 0, 0)
        prev_post = None  # Resets anchor clean




    cv2.imshow('Drone View', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        d.land()
        break

# Clean termination
d.streamoff()
cv2.destroyAllWindows()