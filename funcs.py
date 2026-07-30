import time
import threading


is_command_busy = False
last_completion_time = 0
#create function to manage timing of commands and keep them in thread for faster processing and prevent crashes.
def command(command_func, name, *args, **kwargs):
    global is_command_busy, last_completion_time

    #if the drone is already carrying out a command, ignore new gesture triggers
    if is_command_busy or (time.time() - last_completion_time < 1):
        return

    def work():
        global is_command_busy, last_completion_time
        is_command_busy = True
        print(f"Command: {name}")

        try:
            command_func(*args, **kwargs)
            #time.sleep(0.05) #small delay before next command
        except Exception as e:
            print(f"Command error ({name}): {e}")
        finally:
            is_command_busy = False
            last_completion_time = time.time()

    threading.Thread(target=work, daemon=True).start()
    return True



def swipe_control(drone, curr_pos, anchor_pos, threshold=0.065, rc_speed=30):
    """
    Anchors the fingertip position once when the gesture starts, then treats
    displacement from that fixed anchor as a continuous directional command
    (like a joystick), instead of comparing adjacent frames.
    """
    # First frame seeing the gesture: anchor position, hold hover
    if anchor_pos is None or curr_pos is None:
        drone.send_rc_control(0, 0, 0, 0)
        return curr_pos

    dx = curr_pos[0] - anchor_pos[0]
    dy = curr_pos[1] - anchor_pos[1]

    left_right = 0
    forward_backward = 0
    up_down = 0
    yaw = 0

    # Horizontal swipe (X-axis)
    if abs(dx) > threshold and abs(dx) > abs(dy):
        if dx > 0:
            left_right = -rc_speed    # Swipe Right -> Roll Left
        else:
            left_right = rc_speed   # Swipe Left -> Roll Right

    # Vertical swipe (Y-axis: Y increases downward in MediaPipe)
    elif abs(dy) > threshold and abs(dy) > abs(dx):
        if dy < 0:
            up_down = rc_speed       # Swipe Up -> Throttle Up
        else:
            up_down = -rc_speed      # Swipe Down -> Throttle Down

    # Send relative speed (left_right, forward_backward, up_down, yaw)
    drone.send_rc_control(left_right, forward_backward, up_down, yaw)

    # Anchor returned unchanged. Stays fixed for whole gesture.
    return anchor_pos