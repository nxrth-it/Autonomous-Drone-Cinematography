import time
import threading


is_command_busy = False
#create function to manage timing of commands and keep them in thread for faster processing and prevent crashes.
def command(command_func, name, *args, **kwargs):
    global is_command_busy

    #if the drone is already carrying out a command, ignore new gesture triggers
    if is_command_busy:
        return

    def work():
        global is_command_busy
        is_command_busy = True
        print(f"Command: {name}")

        try:
            command_func(*args, **kwargs)
        except Exception as e:
            print(f"Command error ({name}): {e}")
        finally:
            is_command_busy = False

    threading.Thread(target=work, daemon=True).start()
    return True



def swipe_control(drone, curr_pos, prev_pos, threshold=0.065, rc_speed=30):
    """
    Calculates swipe direction from prev_pos to curr_pos and sends RC control.
    Returns the new prev_pos to carry over to the next frame.
    """
    # First frame seeing the gesture: anchor position, hold hover
    if prev_pos is None or curr_pos is None:
        drone.send_rc_control(0, 0, 0, 0)
        return curr_pos

    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]

    left_right = 0
    forward_backward = 0
    up_down = 0
    yaw = 0

    # Horizontal swipe (X-axis)
    if abs(dx) > threshold and abs(dx) > abs(dy):
        if dx > 0:
            left_right = rc_speed    # Swipe Right -> Roll Right
        else:
            left_right = -rc_speed   # Swipe Left -> Roll Left

    # Vertical swipe (Y-axis: Y increases downward in MediaPipe)
    elif abs(dy) > threshold and abs(dy) > abs(dx):
        if dy < 0:
            up_down = rc_speed       # Swipe Up -> Throttle Up
        else:
            up_down = -rc_speed      # Swipe Down -> Throttle Down

    # Send relative speed (left_right, forward_backward, up_down, yaw)
    drone.send_rc_control(left_right, forward_backward, up_down, yaw)

    # Return curr_pos to become prev_pos in the next frame
    return curr_pos