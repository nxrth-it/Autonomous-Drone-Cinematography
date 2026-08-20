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
from follow import PersonFollower, PID



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
    running_mode=vision.RunningMode.VIDEO #not using live stream mode since processing it as 1 frame at a time.
    #result_callback=lambda result, output_image, timestamp_ms: update_result(result)
)
recognizer = vision.GestureRecognizer.create_from_options(options)

#=========================================================================
#Initialize drone and start video stream
d = drone(enable_mission_pad=False, show_cam=False)
d.set_video_resolution(Tello.RESOLUTION_720P)
d.set_video_bitrate(Tello.BITRATE_3MBPS)
d.streamon()
time.sleep(3.5)
frame_read = d.get_frame_read()


###########################################################
#Follow Person Setup
# YAW AXIS ONLY. max_forward/max_updown become each PID's
# output_limit, so 0 clamps those axes to zero no matter what the controller
# computes - safer than remembering not to wire them, because it cannot be
# bypassed. Raise them one at a time once yaw is tuned and trusted.
follower = PersonFollower(model_path="models/yolo11n.pt",
                          max_forward=100, max_updown=0)

#Not sure, but enabling up down may be causing drone to stabilize more backwards during follow mode.
follow_mode = False
toggle_follow = False
three_fingers_prev = False




print("Flight stream live. Press 'q' to disconnect and land.")
last_command_time = time.time()
swipe_lost_frames = 0
swipe_anchor = None   # fingertip-relative-to-wrist position when the gesture began

#initialize cooldown for photo taking
last_photo_time = 0

#initialize video taking
vid_writer = None
thumb_up_prev = False

#create CUDA context and upload model weights to GPU memory. (Faster usage later + no lag)
#np.zeroes = array with 675 rows 900 columns with 3 values per pixel
#use uint8 (unsigned 8bit integer) for pixel format
follower.model.predict(np.zeros((675, 900, 3), dtype=np.uint8), imgsz=416, verbose=False)

while True:
    frame = frame_read.frame
   # print(frame.shape)

        #Skip invalid/empty initial frames from Tello
    if frame is None or frame.size == 0:
        time.sleep(0.01)
        continue

    native_h, native_w, _ = frame.shape
    target_w = 900
    target_h = int(target_w * native_h / native_w)
    frame = cv2.resize(frame, (target_w, target_h))

    #MediaPipe
    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) #convert to BGR only for cv2.imshow()

    # Record BEFORE anything is drawn on display_frame, so the footage is clean
    # - no landmark skeletons, boxes or status text baked into the video. Also
    # avoids needing a .copy(), which would be a 1.8MB memcpy every frame.
    # Must run at top level every frame: putting it inside a gesture branch
    # would only record on frames where that gesture happened to be recognised.
    if vid_writer is not None:
        vid_writer.write(display_frame)
    rgb_frame = frame #mediapipe raw rgb frame
    timestamp_ms = int(time.time() * 1000)

   
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    #recognize video
    result = recognizer.recognize_for_video(mp_image, timestamp_ms)

    is_actively_swiping = False
    # Every frame build up ONE set of RC velocities and send a single
    # 'rc' command at the bottom of the loop. The Tello's 'rc' carries all
    # four channels at once, so two separate send_rc_control() calls would
    # mean the second silently overwrites the first.
    rc_left_right = 0
    rc_forward_back = 0
    rc_up_down = 0
    rc_yaw = 0
    hand_center_x = None      # wrist x  -> drives the yaw centring
    finger_rel_pos = None     # tip-wrist -> drives the swipe
    is_thumb_up = False

    toggle_follow = False  # Reset toggle each frame; only a single frame of the gesture should trigger it
    allowed_f_gestures = ("three_fingers", "Closed_Fist", "Victory", "Open_Palm")


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

            # Where the hand sits in the frame. Only the yaw uses this, and
            # the wrist is a deliberately steady reference point - it barely
            # moves while the finger is flicking, so the drone tracks the person
            # rather than chasing their fingertip.
            hand_center_x = wrist.x

            # Where the fingertip sits RELATIVE TO THE PERSON'S OWN WRIST. Measured
            # against their wrist instead of against the frame, so the drone
            # yawing (which slides their whole hand sideways through the frame)
            # cannot fake a swipe or cancel a real one.
            finger_rel_pos = (index_tip.x - wrist.x, index_tip.y - wrist.y)

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

            if confidence > 0.85 and predicted_label.lower() != "undefined": #only act on gestures that the model is confident about and are known gestures
                text = f"{predicted_label} ({confidence*100:.1f}%)"
                cv2.putText(display_frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                if follow_mode and predicted_label not in allowed_f_gestures:
                    cv2.putText(display_frame, "Movement Gestures are locked while the drone follows you.", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    #Prevent other commands besides 3 fingers to run

                elif predicted_label == "point_down":
                    last_command_time = time.time()
                     #disable follow mode if it was enabled
                    rc_up_down = -30
                    



                elif predicted_label == "three_fingers":
                    #follow the person function call
                    last_command_time = time.time()
                    print("Command: Follow Person")
                    toggle_follow = True



                   #add frame counter to switch follow_mode off after a certain number of frames 
                elif predicted_label == "flat_palm":
                    #tello.land()
                    last_command_time = time.time()
                    print("Command: Land")
                    follower.reset()  # Reset the follower state when landing
                    
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

                # Must test gesture_name, not predicted_label - this is the
                # MediaPipe branch, and predicted_label belongs to the custom
                # model. It also has to be the HEAD of the chain (everything
                # below is now elif) or it only draws text while every gesture
                # underneath still runs.
                if follow_mode and gesture_name not in allowed_f_gestures:
                    cv2.putText(display_frame, "Movement Gestures are locked while the drone follows you.", (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                elif gesture_name == "Closed_Fist":
                    print("Command: Closed Fist Action")
                    last_command_time = time.time()
                    # Closed fist disengages follow mode. Safe to set directly:
                    # this branch only runs when the custom model did NOT fire, so
                    # toggle_follow is still False and the rising-edge check below
                    # cannot switch follow mode back on in the same frame.
                    if follow_mode:
                        follow_mode = False
                        follower.follow_state = "land"
                        follower.reset()
                        print("Follow mode: False (cancelled by closed fist)")


                elif gesture_name == "Open_Palm":
                   # print("Command: Start taking video")
                    last_command_time = time.time()
                    


                elif gesture_name == "Pointing_Up":
                    print("Command: Pointing Up Action")
                    last_command_time = time.time()
                    
                    is_actively_swiping = True

                    # First frame of this gesture: remember where the
                    # fingertip started. Every later frame is measured
                    # against that fixed point, so holding a steady offset
                    # keeps producing movement instead of resetting to zero.
                    if swipe_anchor is None:
                        swipe_anchor = finger_rel_pos

                    # Calculate only - the yaw still has to be mixed in below.
                    rc_left_right, rc_forward_back, rc_up_down = swipe_control(
                        finger_rel_pos, swipe_anchor, threshold=0.025, rc_speed=30
                    )



                elif gesture_name == "Thumb_Down":
                    print("Command: Thumb Down Action")
                    last_command_time = time.time()
                    
                    #d.send_rc_control(0,0,0,0)

                elif gesture_name == "Thumb_Up":
                    print("Command: Thumb Up Action")
                    # Only report that the gesture was seen. The toggle and the
                    # frame writing both happen at top level - this branch does
                    # not run on every frame, so neither can live here.
                    is_thumb_up = True
                    last_command_time = time.time()

                elif gesture_name == "Victory":
                    last_command_time = time.time()

                    if time.time() - last_photo_time >= 3:
                        print("Command: Take Picture. Say Cheese!")
                        t = time.strftime("%H%M$S") + ".jpg"
                        d.capture(filename=t)
                        last_photo_time = time.time()
                    #don't disable follow mode to take picture during following

                elif gesture_name == "ILoveYou":
                    print("Command: I Love You Action")
                    
                    last_command_time = time.time()
                    
                # else:
                #     print("MP: Unmatched Gesture")
                #     print(f"DEBUG: {gesture_name}, score: {result.gestures[0][0].score:.3f}")

            else:
                cv2.putText(display_frame, "Undefined Gesture", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

           
    else:
        cv2.putText(display_frame, "Undefined Gesture", (10, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 


    if is_thumb_up and not thumb_up_prev:
        if vid_writer is None:
            fc = cv2.VideoWriter_fourcc(*'mp4v')
            # Size must come from the frame actually written, and VideoWriter
            # takes (width, height) - reversed from shape.
            vh, vw = display_frame.shape[:2]
            name = time.strftime("%Y%m%d_%H%M%S") + ".mp4"
            vid_writer = cv2.VideoWriter(name, fc, 20, (vw, vh), True)

            print(f"Recording Video: {name}")

        else:
            vid_writer.release()
            vid_writer = None
            print("Recording Stopped")

    # Unconditional - must run EVERY frame, not only when an edge fires. Left
    # inside the block above, prev sticks at True after the first toggle and no
    # further rising edge can ever be detected: record once, never stop.
    thumb_up_prev = is_thumb_up
                                

    # --- add yaw then send exactly one command 
    if is_actively_swiping:
        # Only auto-rotate while person is actively driving it, so it doesn't
        # quietly spin on the spot when person is not paying attention to it.
        rc_yaw = yaw_centering(hand_center_x, deadzone=0.12, max_yaw_speed=40)
        swipe_lost_frames = 0
    else:
        # Gesture gone (or never there) - remove anchor so the next swipe
        # starts fresh from wherever the finger is then, rather than being
        # measured against a stale point from several seconds ago.
        swipe_lost_frames += 1
        if swipe_lost_frames >= 10:
            swipe_anchor = None 



    # --- follow mode toggle (rising edge) 
    # Only flip when the gesture was ABSENT last frame and is PRESENT this
    # frame, so holding three fingers toggles once instead of every frame.
    if toggle_follow and not three_fingers_prev:
        follow_mode = not follow_mode
        follower.reset()          # clean tracker/PID state on every switch
        print(f"Follow mode: {follow_mode}")

    # Unconditional - must run every frame, including frames with no gesture
    # and no hand at all. This is what records a 'no', which re-arms the check
    # so the next appearance counts as a fresh edge.
    three_fingers_prev = toggle_follow

        #Check if follow mode has been activated
    if follow_mode:
        # update() now always returns the same five things, so this unpack is
        # safe on every path - including "no person detected".
        yaw_cor, lr_cor, fb_cor, ud_cor, box_coords = follower.update(display_frame)

        # Assign, do NOT send. There is exactly one send_rc_control per frame,
        # at the bottom of the loop - a second call here would be overwritten by
        # it microseconds later, and would also bypass the is_busy() guard.
        # This sits OUTSIDE the box check on purpose: during a search there is
        # no box, but yaw_cor carries the sweep that has to reach the drone.
        rc_left_right   = lr_cor
        rc_forward_back = fb_cor
        rc_up_down      = ud_cor
        rc_yaw          = yaw_cor

        if box_coords is not None:
            x1, y1, x2, y2 = box_coords
            cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(display_frame, f"FOLLOWING id={follower.locked_id}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        else:
            cv2.putText(display_frame, f"FOLLOW: {follower.follow_state}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if follower.follow_state == "land":
            # Clear the velocities so the single send at the bottom of the loop
            # issues a stop rather than the last search sweep.
            rc_left_right, rc_forward_back, rc_up_down, rc_yaw = 0, 0, 0, 0
            follow_mode = False          # stop steering BEFORE it descends
            # Threaded wrapper, not d.land() directly - a bare land() blocks the
            # loop for several seconds, freezing the video and killing q and f.
            # is_busy() then holds off the rc stream while it descends.
            command(d.land, "Landing")
            print("Follow mode: False (landing)")
            follower.reset()  # Reset the follower state when landing





    # Skip sending while a blocking command (takeoff / land / move_down) is
    # still running in its thread - a continuous rc stream would fight that
    # command and cancel the manoeuvre halfway through.
    if not is_busy():
        d.send_rc_control(rc_left_right, rc_forward_back, rc_up_down, rc_yaw)




    cv2.imshow('Drone View', display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        # Finalise any recording in progress. Without release() the file has no
        # completed header and will not play back at all.
        if vid_writer is not None:
            vid_writer.release()
            vid_writer = None
            print("Recording Stopped (quit)")

        d.send_rc_control(0, 0, 0, 0)  # Stop movement before landing
        d.land()
        break

    elif key == ord('f'):
        # Manual follow-mode toggle. This is the only off-switch that still works
        # at filming distance - MediaPipe cannot resolve a hand from far enough
        # away to recognise the three-finger gesture, so never rely on the gesture
        # alone to stop an autonomous mode.
        follow_mode = not follow_mode
        follower.reset()
        print(f"Follow mode: {follow_mode} (keyboard)")

# Clean termination
d.streamoff()
cv2.destroyAllWindows()