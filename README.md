# Gesture-Controlled Autonomous Tello Drone

Hand-gesture control and autonomous person-following for the DJI Tello, built for filming without a handheld controller. The drone's camera feed is processed on a laptop: hand gestures are recognised with MediaPipe and a custom TensorFlow classifier, people are detected and tracked with YOLO11, and the resulting commands are sent back to the drone over Wi-Fi.

> **Warning:** the Tello has no obstacle sensors. In follow, orbit and dronie modes it will fly into anything in its path. Fly in open space and keep the `q` key within reach.

---

## Features

- **Gesture flight control** — take off, land, descend, climb and move sideways using hand gestures only.
- **Person following** — locks onto the person who activated it and keeps them centred and at a set distance.
- **Orbit** — circles the subject while keeping them in frame.
- **Dronie** — flies backwards and upwards at increasing speed for 5 seconds, then resumes following.
- **Photo and video capture** — triggered by gesture and saved to disk with timestamped filenames.
- **On-screen status** — recording indicator, photo confirmation, battery, temperature and frame rate.
- **Safety cut-offs** — automatic landing and shutdown on low battery or high temperature.

---

## Gestures

When more than one hand is visible, only the **higher hand** is read. A hand resting at your side is ignored.

### Custom model

| Gesture | Action |
| :--- | :--- |
| `ok_sign` | Take off |
| `flat_palm` | Land — must be held for 3 seconds; a countdown is shown on screen |
| `point_down` | Descend while held |
| `three_fingers` | Toggle person following on/off |
| `L_sign` | Toggle orbit (starts following if it isn't already on) |

### MediaPipe

| Gesture | Action |
| :--- | :--- |
| `Pointing_Up` | Manual sideways control: tilt the finger left or right to move the drone. The drone rotates to keep you centred while moving. |
| Both hands `Pointing_Up`, raised | Climb while held |
| `Open_Palm` | Dronie |
| `Thumb_Up` | Start / stop video recording |
| `Victory` | Take a photo (3-second cooldown) |
| `Closed_Fist` | Cancel following, orbit or dronie and hover |

While following, manual movement gestures are locked out so they can't conflict with the tracking controller. Only `three_fingers`, `Closed_Fist`, `Victory`, `Open_Palm` and `L_sign` are accepted.

### Keyboard

| Key | Action |
| :--- | :--- |
| `f` | Toggle following on/off |
| `q` | Stop, land and quit |

At filming distance the camera often can't resolve hand gestures reliably, so the keyboard is the dependable way to stop autonomous modes.

---

## How it works

Each frame from the drone goes through the same loop:

1. **Hand recognition.** MediaPipe extracts 21 hand landmarks. The custom classifier runs first; if it isn't confident (below 0.84) or returns `undefined`, MediaPipe's built-in gesture recogniser is used instead.
2. **Person tracking** (follow modes only). YOLO11n detects people and BoT-SORT keeps a consistent ID, so the drone stays on the person it locked onto rather than switching to whoever else walks into frame.
3. **Control.** Position and size of the tracked person's bounding box are turned into yaw, forward/backward and vertical errors, each driven by its own PID controller. Box height is used as a distance estimate.
4. **Output.** All four movement channels are combined into a single `rc` command per frame.

If the tracked person is lost for 30 frames, the drone turns in the direction they were last seen. After 7 seconds without finding them it stops following and hovers in place.

---

## Safety behaviour

| Condition | Response |
| :--- | :--- |
| Battery at or below 15% | Lands and exits, with the reason shown on screen |
| Temperature at or above 90°C | Lands and exits, with the reason shown on screen |
| Orbit blocked (commanded sideways but not moving) | Stops orbiting and hovers |
| Target lost for 7 seconds | Stops following and hovers — does not land |
| Landing gesture | Requires a 3-second hold, so a single misread frame can't land the drone |

The thresholds are set near the top of `drone_control.py` (`BATTERY_ABORT`, `TEMP_ABORT`).

---

## Requirements

- Windows (only Windows has been tested)
- Python 3.13
- An NVIDIA GPU with CUDA is strongly recommended. YOLO on CPU is too slow for following.
- **Keep the laptop plugged in.** On battery power, Windows throttles the GPU and frame rate drops sharply.
- A DJI Tello or DJI RoboMaster TT

## Installation

```bash
git clone https://github.com/AthithIttiphakorn/Drone---HPQ-Project.git
cd Drone---HPQ-Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

The extra index URL is needed for the CUDA build of PyTorch. To confirm the GPU is being used:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Running

1. Turn on the drone and connect the laptop to its Wi-Fi network.
2. Run:
   ```bash
   python drone_control.py
   ```
3. Wait for `Flight stream live` in the terminal and the video window to appear, then use `ok_sign` to take off.

Photos are saved to `pictures/` and videos to `videos/`.

---

## Retraining the gesture model

The custom classifier is a small neural network trained on hand-landmark coordinates, not images. Coordinates are made relative to the wrist and scaled to a fixed size, so hand position and distance from the camera don't affect the prediction.

Run both scripts from the project root:

1. `gesture_training/extract.py` — collects labelled landmark samples into `hand_dataset.csv`.
2. `gesture_training/modeltrain.py` — trains the model and writes `label_classes.pkl` and `gesture_coordinate_model.keras`.
3. Move `gesture_coordinate_model.keras` into `models/`. The program loads the model from there, so skipping this step leaves the old model in use.

---

## Project structure

```
drone_control.py        main loop: video, gesture handling, flight commands, on-screen display
follow.py               PID controller and PersonFollower (tracking, follow, orbit, dronie)
funcs.py                swipe control, yaw centring, threaded command helper
drone_tello.py          drone connection and helper methods
gesture_training/       data collection and model training scripts
models/                 gesture classifier, MediaPipe task files, YOLO11 weights
hand_dataset.csv        gesture training data
label_classes.pkl       class labels for the gesture classifier
```

---

## Known limitations

- **No obstacle avoidance.** The Tello cannot sense obstacles in any direction it moves.
- **Vertical tracking is disabled in follow mode.** Distance is estimated from the person's height in frame, and climbing makes them appear smaller, which the controller reads as them moving away.
- **Orbit is only partially stable.** It can drift or stop early, particularly near walls.
- **Distance estimates are less reliable up close**, where the person's body gets cut off at the bottom of the frame.
- **Gesture recognition degrades with range and hand angle.** Gestures work best within a few metres with the hand facing the camera.

---

Enquiries: northitti@gmail.com
