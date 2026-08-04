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

    def __init__(self, model_path="models/yolo11n.pt",
                 target_height_frac=0.40,
                 max_yaw=40, max_forward=20, max_updown=15,
                 lost_limit=30):
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

        self.search_start = None  # Timestamp when search mode started
        self.follow_state = "search"  # Initial state is search mode

    def reset(self):
        # Return the follower to the same state __init__ leaves it in, so a
        # reset genuinely starts clean. Clearing locked_id is what allows a new
        # target to be acquired; clearing _last_time stops the next update()
        # from computing a huge dt spanning the whole time follow mode was off.
        self.pid_yaw.reset()
        self.pid_fwd.reset()
        self.pid_ud.reset()
        self.lost_frames = 0
        self.locked_id = None
        self._last_time = None
        self.search_start = None
        self.follow_state = "search"



    def update(self, frame_bgr):
        # Declared up front so EVERY path returns the same five things. There
        # is exactly one return, at the bottom - branches only assign.
        yaw_cor = 0
        left_right = 0
        forward_back = 0
        up_down = 0
        follow_box = None  #Prevent UnboundLocalError if not assigned.

        # --- 1. DETECT FIRST ------------------------------------------------
        # Nothing about the state can be decided until we know whether OUR
        # target is in this frame. Deciding first meant the state logic had to
        # guess, which is how it ended up sweeping at 20 yaw while looking
        # straight at the person.
        results = self.model.track(frame_bgr, persist=True, classes=[0], verbose=False, imgsz=416)
        results = results[0]
        boxes = results.boxes

        if boxes is not None and boxes.is_track and len(boxes) > 0:
            box_coords = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)

            if self.locked_id is None:
                # First lock of this session: take the LARGEST box. Biggest
                # box = closest person = almost always whoever just engaged
                # follow mode, rather than a bystander further away.
                areas = (box_coords[:, 2] - box_coords[:, 0]) * (box_coords[:, 3] - box_coords[:, 1])
                self.locked_id = ids[int(np.argmax(areas))]

            # Only ever accept OUR id. If a stranger is detected and our target
            # is not, follow_box stays None - that counts as lost, which is the
            # whole point of locking.
            for i in range(len(ids)):
                if ids[i] == self.locked_id:
                    follow_box = box_coords[i]  # x1, y1, x2, y2

        # --- 2. THEN decide the state, from what was actually found ---------
        if follow_box is not None and self.follow_state != "land":
            # Target visible. Back to following and clear both loss counters.
            # (Once "land" is decided it is final until reset() - a one-frame
            # flicker of detection must not abort a landing already underway.)
            self.follow_state = "follow"
            self.lost_frames = 0
            self.search_start = None

            # PID work goes here in Phase C. Velocities stay 0 until then.

        elif follow_box is None and self.follow_state != "land":
            # Target not visible this frame.
            self.lost_frames += 1

            # lost_limit tolerates brief dropouts - YOLO missing a frame or two
            # is routine and must not kick off a 7 second search.
            if self.lost_frames >= self.lost_limit:

                if self.search_start is None:
                    # First frame of the search: start the clock ONCE. Using
                    # "is search_start unset" rather than the state name means
                    # this is correct no matter what state we came from.
                    self.follow_state = "search"
                    self.search_start = time.time()

                    # Release the lock. The tracker only keeps a lost track's
                    # id alive for track_buffer (30) frames and has ReID off,
                    # so anyone who walks back into shot comes back with a NEW
                    # id. Holding the old one means matching against a ghost -
                    # the search could never succeed. Searching already means
                    # the original lock is gone, so let acquisition re-lock.
                    self.locked_id = None
                    print("Target lost - searching.")

                elif time.time() - self.search_start > 7:
                    self.follow_state = "land"
                    print("No person detected for 7 seconds. Landing the drone.")

                if self.follow_state == "search":
                    yaw_cor = 20  # sweep to look for them

        return yaw_cor, left_right, forward_back, up_down, follow_box

#Testing YOLO11 person tracking - careful: target reaquiring gets the person with the biggest box.
if __name__ == "__main__":
    # Example usage
    follower = PersonFollower()

    cap = cv2.VideoCapture(0)  # Use the first camera



    while True:
        ret, frame = cap.read()
        if not ret:
            break

        yaw_cor, left_right, forward_back, up_down, box_coords = follower.update(frame)

        if box_coords is not None:
            x1, y1, x2, y2 = box_coords
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"Following id={follower.locked_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Person Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # State readout - watching this is how you debug the state machine.
        cv2.putText(frame, f"{follower.follow_state}  yaw={yaw_cor}  lost={follower.lost_frames}",
                    (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Person Follower", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Simulates toggling follow mode off/on, so you can test that a
            # reset re-acquires a lock cleanly.
            follower.reset()
            print("reset")

    cap.release()
    cv2.destroyAllWindows()
