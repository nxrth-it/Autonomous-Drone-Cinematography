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

#Follow mode toggle setup
follow_mode = False
toggle_follow = False
three_fingers_prev = False
follow_lost_frames = 0

#Orbit toggle
prev_orbit = False
is_orbit = False
orbit_lost_frames = 0

#Dronie toggle
prev_dronie = False
is_dronie = False
dronie_lost_frames = 0
stall_frames = 0
# Set when the follower enters orbit, cleared when it leaves. The stall check
# needs it to know how long the orbit has been running.
orbit_start_time = None

# Landing is the one command that cannot be undone in the air, so it is the
# only gesture that must be HELD rather than tapped. Wall-clock, not frames -
# "3 seconds" has to mean 3 seconds regardless of how fast the loop runs.
flat_palm_start = None
flat_palm_lost_frames = 0

# --- automatic abort limits ----------------------------------------------
# Checked every frame and acted on straight away. The Tello does protect
# itself eventually, but its warning is one line in a terminal that scrolls
# past several times a second - so the program stops outright instead, and
# the reason is the last thing left on screen.
BATTERY_ABORT = 15     # percent, abort at or below
# 85 was too low. Flight footage from 3 Sep shows this airframe idling at
# 78C and creeping to 80C over ten minutes - only 5C of headroom, so a longer
# session or a warmer room would have aborted a perfectly healthy flight.
TEMP_ABORT    = 90     # degrees C, abort at or above (hottest onboard sensor)
abort_reason  = None



print("Flight stream live. Press 'q' to disconnect and land.")
swipe_lost_frames = 0
swipe_anchor = None   # fingertip-relative-to-wrist position when the gesture began

#initialize cooldown for photo taking
last_photo_time = 0

#initialize video taking
vid_writer = None
thumb_up_prev = False
thumb_up_lost_frames = 0


gesture_name = None

# --- loop rate meter -------------------------------------------------------
# perf_counter rather than time(): it is monotonic and higher resolution, so a
# system clock adjustment mid-flight cannot produce a negative interval.
# fps_smooth starts at 0 purely as a "no reading yet" marker - the first real
# sample seeds it directly rather than letting it ramp up from zero.
last_loop_time = time.perf_counter()
fps_smooth = 0.0

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

    # --- abort on low battery / high temperature --------------------------
    # Both read the cached state packet rather than sending a command, so this
    # costs nothing per frame. Wrapped because get_state_field raises when a
    # field has not arrived yet, and a dropped packet must not kill the loop.
    try:
        battery = d.get_battery()
        temperature = d.get_highest_temperature()
    except Exception:
        battery, temperature = None, None

    if battery is not None and battery <= BATTERY_ABORT:
        abort_reason = f"BATTERY {battery}% - AT OR BELOW {BATTERY_ABORT}% LIMIT"
    elif temperature is not None and temperature >= TEMP_ABORT:
        abort_reason = f"TEMPERATURE {temperature}C - AT OR ABOVE {TEMP_ABORT}C LIMIT"

    if abort_reason:
        # Deliberately loud. A single print would be lost inside the gesture
        # debug stream, which is the whole reason this check exists.
        print("\n" + "=" * 62)
        print("  ABORTING FLIGHT:", abort_reason)
        print("=" * 62 + "\n")

        cv2.putText(display_frame, "ABORTING", (40, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)
        cv2.putText(display_frame, abort_reason, (40, 355),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Drone View', display_frame)
        cv2.waitKey(2000)          # hold the reason on screen before the window closes

        if vid_writer is not None:
            vid_writer.release()   # without release() the file has no header and will not play
            vid_writer = None

        # Land rather than simply exiting - dropping out of the loop would
        # leave the aircraft airborne until its own failsafe times out.
        # Wrapped because land() raises if the drone is already on the ground.
        try:
            d.send_rc_control(0, 0, 0, 0)
            d.land()
        except Exception as e:
            print(f"Land during abort failed: {e}")
        break

    rgb_frame = frame #mediapipe raw rgb frame
    timestamp_ms = int(time.time() * 1000)

   
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    #recognize video
    result = recognizer.recognize_for_video(mp_image, timestamp_ms)

    #Reset every frame
    is_actively_swiping = False
    is_orbit = False
    is_dronie = False
    is_flat_palm = False
    # Every frame build up ONE set of RC velocities and send a single
    # rc command at the bottom of the loop. The Tello's rc carries all
    # four channels at once, so two separate send_rc_control() calls would
    # mean the second  overwrites the first.
    rc_left_right = 0
    rc_forward_back = 0
    rc_up_down = 0
    rc_yaw = 0
    hand_center_x = None      # wrist x  -> drives the yaw centring
    finger_rel_pos = None     # tip-wrist -> drives the swipe
    is_thumb_up = False


    toggle_follow = False  # Reset toggle each frame
    allowed_f_gestures = ("three_fingers", "Closed_Fist", "Victory", "Open_Palm", "L_sign")


    if result.hand_landmarks:
        #loop through landmarks
        for hand_landmarks in result.hand_landmarks:
            #(DRAWING HAND LANDMARKS)
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
                     #disable follow mode if it was enabled
                    rc_up_down = -30
                    



                elif predicted_label == "three_fingers":
                    #follow the person function call
                    #print("Command: Follow Person")
                    toggle_follow = True



                elif predicted_label == "flat_palm":
                    # Only report the gesture. The 3 second hold is timed at top
                    # level, because this branch does not run on every frame -
                    # a timer living here would be reset by any frame where the
                    # custom model dipped below its confidence gate.
                    is_flat_palm = True

                elif predicted_label == "L_sign":
                    #orbit function call
                    #print("Command: Orbit")
                    is_orbit = True

                    

                elif predicted_label == "ok_sign":
                    #tello.takeoff()
                    print("Command: Takeoff")
                    
                    command(d.takeoff, "Takeoff")

            elif result.gestures and result.hand_landmarks: #if custom hasn't detected a gesture, check if mp has a valid gesture
                #mediapipe gesture recognition
                gesture_name = result.gestures[0][0].category_name
                #print(f"DEBUG: MP GESTURE: {gesture_name}, score: {result.gestures[0][0].score:.2f}")
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
                    # Closed fist disengages follow mode. Safe to set directly:
                    # this branch only runs when the custom model did NOT fire, so
                    # toggle_follow is still False and the rising-edge check below
                    # cannot switch follow mode back on in the same frame.
                    if follow_mode:
                        follow_mode = False
                        follower.reset()
                        print("Follow mode: False (cancelled by closed fist)")


                elif gesture_name == "Open_Palm":
                    #DRONIE!
                    # Only report the gesture. The toggle happens at top level,
                    # because this branch does not run on every frame.
                    #print("Command: Dronie")
                    is_dronie = True



                elif gesture_name == "Pointing_Up":
                    #print("Command: Pointing Up Action")   # fired every frame - blocked the loop

                    is_actively_swiping = True

                    # First frame of this gesture: remember where the
                    # fingertip started. Every later frame is measured
                    # against that fixed point, so holding a steady offset
                    # keeps producing movement instead of resetting to zero.
                    #
                    # The score gate matters more than it looks. MediaPipe
                    # starts reporting Pointing_Up at around 0.5 while the hand
                    # is still forming the gesture, and settles near 0.92 once
                    # the finger is fully extended. Anchoring on that first
                    # half-made frame stores a bent finger as "centre", so the
                    # finger simply finishing its extension afterwards reads as
                    # a large upward swipe which then wins the dy-vs-dx test
                    # on every frame and pins up_down for the whole hold.
                    # Waiting for a confident frame anchors the settled pose.
                    # swipe_control returns zeros while the anchor is still
                    # None, so nothing moves until there is a good one.
                    if swipe_anchor is None and result.gestures[0][0].score >= 0.6:
                        swipe_anchor = finger_rel_pos
                        print(f"Swipe anchor set at score {result.gestures[0][0].score:.2f}")

                    # The threshold MUST be scaled by the hand's apparent size.
                    # finger_rel_pos is in frame-normalised units, so the hand
                    # shrinks as the person backs away - a fixed number silently
                    # demands a bigger and bigger tilt with distance. At a raw
                    # 0.09 the swipe needs a 37 degree tilt at 1m and becomes
                    # mathematically impossible past about 1.6m, which is why it
                    # was recognising the gesture and then doing nothing.
                    # Scaling by the anchor's own length cancels distance out,
                    # so the trigger is a fixed ANGLE of wrist tilt at any range.
                    anchor_scale = np.hypot(swipe_anchor[0], swipe_anchor[1]) if swipe_anchor else 0.2

                    # Calculate only - the yaw still has to be mixed in below.
                    rc_left_right, rc_forward_back, rc_up_down = swipe_control(
                        finger_rel_pos, swipe_anchor, threshold=0.09 * anchor_scale, rc_speed=30
                    )
                    # Tighter arc: correct sooner (smaller deadzone) and harder
                    # (higher ceiling), so the drone rotates to keep the person
                    # framed instead of translating out around them.
                    rc_yaw = yaw_centering(hand_center_x, deadzone=0.06, max_yaw_speed=60)
                    



                elif gesture_name == "Thumb_Down":
                    pass
                    #print("Command: Thumb Down Action")
                    
                    #d.send_rc_control(0,0,0,0)

                elif gesture_name == "Thumb_Up":
                    #print("Command: Thumb Up Action")
                    # Only report that the gesture was seen. The toggle and the
                    # frame writing both happen at top level - this branch does
                    # not run on every frame, so neither can live here.
                    is_thumb_up = True

                elif gesture_name == "Victory":

                    if time.time() - last_photo_time >= 3:
                        #print("Command: Take Picture. Say Cheese!")
                        t = "photo_" + timestamp() + ".jpg"
                        d.capture(filename=t)
                        last_photo_time = time.time()
                    #don't disable follow mode to take picture during following

                elif gesture_name == "ILoveYou":
                    #print("Command: I Love You Action")
                    pass
                    
                    
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
            os.makedirs("videos", exist_ok=True)
            name = "videos/video_" + timestamp() + ".mp4"
            vid_writer = cv2.VideoWriter(name, fc, 20, (vw, vh), True)

            print(f"Recording Video: {name}")

        else:
            vid_writer.release()
            vid_writer = None
            print("Recording Stopped")

    #Period where gesture is maintained (prevent activation twice due to lag or frame skip)
    if is_thumb_up:
        thumb_up_lost_frames = 0
        thumb_up_prev = True
    else:
        thumb_up_lost_frames += 1
        if thumb_up_lost_frames >= 8:
            thumb_up_prev = False
    
                                
    #print("is actively swiping", is_actively_swiping, "gesture name", gesture_name)
    # --- add yaw then send exactly one command 
    if is_actively_swiping:
        # Only auto-rotate while person is actively driving it, so it doesn't
        # quietly spin on the spot when person is not paying attention to it.
        swipe_lost_frames = 0
        pass
    else:
        # Gesture gone. remove anchor so the next swipe
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

    #only clear after 8 frames without the detection, prevent false triggering
    if toggle_follow:
        follow_lost_frames = 0
        three_fingers_prev = True
    else:
        follow_lost_frames += 1
        if follow_lost_frames >= 8:
            three_fingers_prev = False



    if is_orbit and not prev_orbit:
        if not follow_mode:
            follow_mode = True
            follower.reset()
            follower.mode = "orbit"


        elif follower.mode == "orbit":
            follower.mode = "follow"

        else:
            follower.mode = "orbit"
    #grace period while toggling
    if is_orbit:
        orbit_lost_frames = 0
        prev_orbit = True
    else:
        orbit_lost_frames += 1
        if orbit_lost_frames >= 8:
            prev_orbit = False


    # --- dronie toggle (rising edge) --------------------------------------
    # Unlike orbit, this is not a state to sit in - it is a scripted move that
    # ends itself after dronie_duration and returns mode to "follow". So the
    # gesture only ever STARTS it; there is nothing to toggle off.
    if is_dronie and not prev_dronie:
        if not follow_mode:
            follow_mode = True
            follower.reset()             # sets mode back to "follow"
            follower.mode = "dronie"     # so this must come after
        elif follower.mode != "dronie":
            # Already following or orbiting - dronie takes over. Clearing the
            # timestamp guarantees the ramp starts at t=0 rather than resuming
            # a stale clock from a previous run.
            follower.dronie_start_time = None
            follower.mode = "dronie"

    #grace period while toggling
    if is_dronie:
        dronie_lost_frames = 0
        prev_dronie = True
    else:
        dronie_lost_frames += 1
        if dronie_lost_frames >= 8:
            prev_dronie = False


    # --- land: flat palm HELD for 3 seconds --------------------------------
    # Every other gesture fires on a rising edge, which is fine for something
    # reversible. Landing is not - a single misread frame used to put the drone
    # on the ground mid-shot. Requiring a sustained hold means a one-frame
    # flicker can no longer reach the motors.
    #
    # The 8 frame tolerance mirrors the other gestures: the model dropping out
    # for a fraction of a second must not restart the count, or the timer can
    # never finish while the hand is at filming distance.
    if is_flat_palm:
        flat_palm_lost_frames = 0
        if flat_palm_start is None:
            flat_palm_start = time.time()
            print("Landing in 3s - keep holding flat palm")

        held = time.time() - flat_palm_start
        cv2.putText(display_frame, f"LANDING IN {max(0.0, 3 - held):.1f}s", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        if held >= 3:
            print("Command: Land")
            follow_mode = False        # stop steering before it descends
            follower.reset()
            command(d.land, "Landing")
            flat_palm_start = None
    else:
        flat_palm_lost_frames += 1
        if flat_palm_lost_frames >= 8:
            flat_palm_start = None     # gesture genuinely released - start over

        #Check if follow mode has been activated
    if follow_mode:
        # update() now always returns the same five things, so this unpack is
        # safe on every path - including "no person detected".
        yaw_cor, lr_cor, fb_cor, ud_cor, box_coords = follower.update(display_frame)

        # Assign do not send. There is exactly one send_rc_control per frame,
        # at the bottom of the loop - a second call here would be overwritten by
        # it microseconds later, and would also bypass the is_busy() guard.
        # This sits outside the box check on purpose during a search there is
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
            # Losing the target no longer lands the drone. Landing is now a
            # deliberate act only - flat palm held for 3 seconds - so a failed
            # search hands control back and HOVERS instead of putting the
            # aircraft on whatever happens to be underneath it.
            rc_left_right, rc_forward_back, rc_up_down, rc_yaw = 0, 0, 0, 0
            follow_mode = False
            print("Follow mode: False (target lost - hovering)")
            follower.reset()


    #safety to prevent drone from continuously crashing into objects while orbiting
    #Prevent speed commands that are very close to 0cm/s from triggering
    #
    # SPIN-UP GRACE. The check compares a command sent this instant against a
    # speed the aircraft cannot possibly have reached yet. The Tello has to
    # build attitude before it translates at all, and vgy only refreshes at
    # about 10Hz, so a perfectly healthy orbit genuinely reports 0-4 for its
    # first several frames. With the counter free-running that reached 10
    # after roughly 500ms - every attempt, identically - and cancelled the
    # orbit before the drone had finished accelerating.
    # Only judge an orbit once it has had time to actually get moving.
    if follower.mode == "orbit":
        if orbit_start_time is None:
            orbit_start_time = time.time()
    else:
        orbit_start_time = None

    orbit_settled = orbit_start_time is not None and (time.time() - orbit_start_time) > 1.5

    if follow_mode and orbit_settled and abs(rc_left_right) > 15 and abs(d.get_speed_y()) < 5:
        stall_frames += 1
    else:
        stall_frames = 0
    if stall_frames >= 10:
        print("Drone blocked - stopping command")
        rc_left_right = rc_forward_back = rc_up_down = rc_yaw = 0
        follow_mode = False
        follower.reset()
        stall_frames = 0



    # Skip sending while a blocking command (takeoff / land / move_down) is
    # still running in its thread - a continuous rc stream would fight that
    # command and cancel the manoeuvre halfway through.
    if not is_busy():
        d.send_rc_control(rc_left_right, rc_forward_back, rc_up_down, rc_yaw)




    # --- capture status overlay -------------------------------------------
    # Drawn last, so it lands AFTER the video write at the top of the loop -
    # the indicator must never end up baked into the recorded footage.
    fh, fw = display_frame.shape[:2]

    if vid_writer is not None:
        # Blinking, so it reads as live rather than as a static label. The dot
        # toggles twice a second: int(t*2) flips parity every 0.5s.
        if int(time.time() * 2) % 2 == 0:
            cv2.circle(display_frame, (fw - 148, 34), 11, (0, 0, 255), -1)
        cv2.putText(display_frame, "REC", (fw - 128, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Photo has no lasting state to show, so the confirmation is timed off the
    # same stamp the 3 second cooldown already uses - no extra variable needed.
    if time.time() - last_photo_time < 1.5:
        cv2.putText(display_frame, "PHOTO SAVED", (fw - 260, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    if battery is not None:
        cv2.putText(display_frame, f"BAT {battery}%  {temperature}C", (20, fh - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # --- loop rate ---------------------------------------------------------
    # Measured here, at the same point every iteration, so the interval spans
    # exactly one full pass of the loop. Frames skipped by the "invalid frame"
    # continue near the top fold into the NEXT reading, which is correct - a
    # stall should show up as a low number rather than being hidden.
    now_perf = time.perf_counter()
    loop_dt = now_perf - last_loop_time
    last_loop_time = now_perf

    # Guarded divide: two iterations inside a single clock tick would give 0.
    if loop_dt > 0:
        inst_fps = 1.0 / loop_dt
        # Smoothed, because raw per-frame FPS swings far too much to read while
        # flying. Seed on the first sample so it does not ramp from zero.
        fps_smooth = inst_fps if fps_smooth == 0.0 else (fps_smooth * 0.9 + inst_fps * 0.1)

    # Colour-coded so the number can be read at a glance instead of parsed:
    # green is healthy, amber is degraded, red means tracking is starved.
    if fps_smooth >= 20:
        fps_colour = (120, 220, 120)
    elif fps_smooth >= 12:
        fps_colour = (80, 200, 240)
    else:
        fps_colour = (80, 80, 255)

    cv2.putText(display_frame, f"{fps_smooth:.0f} FPS", (20, fh - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_colour, 2)

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