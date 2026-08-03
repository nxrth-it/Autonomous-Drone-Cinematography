import time
import numpy as np
from ultralytics import YOLO
import cv2

class PID:
    def __init__(self, kp, ki, kd, output_limit=100, integral_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit

        self.reset()

    def reset(self):
        #Wipe the controller's memory. Call whenever the target is lost or follow mode is (re)entered, so no value accumulation."""
        self._integral = 0.0
        self._prev_error = None


    def update(self, error, dt):
        # Guard against zero/negative dt (two frames in one clock tick) -
        # the D term divides by it and would blow up.
        if dt <= 0:
            dt = 1e-3

        p_term = self.kp * error

        self._integral += error * dt
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit, self._integral))
        i_term = self.ki * self._integral

        if self._prev_error is None:
            # First sample: no previous error exists, so a derivative would be
            # meaningless (and enormous). Skip it exactly once.
            d_term = 0.0
        else:
            d_term = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        output = p_term + i_term + d_term
        return max(-self.output_limit, min(self.output_limit, output))




class PersonFollower:
    """
    YOLO11 person tracking + three PID loops behind a single
    'give me a frame, get back RC velocities' interface.
    """

    def __init__(self, model_path="yolo11n.pt",
                 target_height_frac=0.40,
                 max_yaw=40, max_forward=20, max_updown=15,
                 lost_limit=15):
        # 'n' = nano, the smallest YOLO11. On an RTX 4050 this is a few
        # milliseconds a frame. A bigger model buys accuracy you don't need
        # for 'where is the person in front of me'.
        self.model = YOLO(model_path)

        # How tall the person's box SHOULD be as a fraction of frame height.
        # This IS your distance setpoint: bigger number = drone sits closer.
        # 0.40 keeps a sensible gap; raise it only once tuning is trustworthy.
        self.target_height_frac = target_height_frac

        # How many consecutive frames we tolerate losing the target before
        # giving up entirely.
        self.lost_limit = lost_limit

        # One PID per axis. These gains are STARTING POINTS ONLY - see the
        # tuning section. Note ki=0.0: we deliberately start with PD only.
        self.pid_yaw = PID(kp=45.0, ki=0.0, kd=4.0, output_limit=max_yaw)
        self.pid_fwd = PID(kp=40.0, ki=0.0, kd=3.0, output_limit=max_forward)
        self.pid_ud  = PID(kp=30.0, ki=0.0, kd=2.0, output_limit=max_updown)

        self.locked_id = None
        self.lost_frames = 0
        self._last_time = None

    def reset(self):
        self.pid_yaw.reset()
        self.pid_fwd.reset()
        self.pid_ud.reset()
        self.lost_frames = 0

    def update(self, frame_bgr):
        results = self.model.track(frame_bgr, persist=True, classes=[0], verbose=False, imgsz=416)
        results = results[0]
        boxes = results.boxes

        if boxes is None or len(boxes) == 0:
            #will need to reset the PID controllers if the person disappears from frame to prevent integral windup and derivative spikes when the person reappears.
            return None # no person detected
        
        else:
            box_coords = boxes.xyxy.cpu().numpy()
            return box_coords[0]  # x1, y1, x2, y2
