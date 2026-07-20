import sys
from drone_tello import DroneTello as drone

print("Starting drone script...", flush=True)
d = drone(enable_mission_pad=False, show_cam=True)

if not getattr(d, "is_connected", False):
    raise SystemExit("Drone connection failed. Turn on the Tello and connect to its Wi-Fi network before running this script.")

# Showcase of basic Movement
d.takeoff()
d.move_up(30)
d.move_forward(50)
d.move_left(20)
d.move_right(20)
d.move_back(50)
d.move_down(30)
d.rotate_clockwise(90)        # Rotate 90 degrees clockwise
d.rotate_counter_clockwise(90) # Rotate 90 degrees counter-clockwise

d.capture("test.jpg")

d.land()
d.cleanup()
