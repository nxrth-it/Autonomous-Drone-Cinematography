# Autonomous Gesture-Controlled Tello Drone Software

A real-time, low-latency autonomous flight control system for the **DJI Tello** drone. This project combines a **custom TensorFlow model** trained on hand-landmark spatial geometry with **MediaPipe's Gesture Recognizer Tasks API** to provide precise, real-time gesture piloting over live video stream.

---

## ✨ Key Features
- **Film using Gestures**: use hand gestures to record and stop video.
- **Maneuvers using Gestures**: Execute maneuvers using only gestures.
- **Screenless Control**: Control your drone's movement without using a controller or device. Use just your hands!
- **Automatic Following**: Use a gesture to get your drone to follow you around!
- **⚠️ Tello Drones do not contain any sideways/backwards  sensors and is not able to avoid collision with objects!**
---

## ⚙️ Supported Gestures & Flight Controls

### Custom TensorFlow Model Gestures
| Gesture | Command Trigger | Description |
| :--- | :--- | :--- |
| **`ok_sign`** | **Takeoff** | Initiates autonomous takeoff. |
| **`flat_palm`** | **Land** | Triggers automated landing sequence after 3 seconds of holding the gesture |
| **`point_down`** | **Move Down** | Command drone to descend in height. |
| **`L_sign`** | **Orbit Mode** | Triggers automated orbital maneuver around subject. |

| **`Open_Palm`** | **Dronie Shot** | Drone backs away from you at an increasing pace to do a "dronie shot". |
| **`three_fingers`** | **Follow Person** | Activates vision-based person tracking. Drone will follow you around autonomously|

### MediaPipe Fallback Gestures
| Gesture | Default Reaction |
| :--- | :--- |
| **`Point_Up`** | **Manual Control** | Slide the drone to your left and right by just swiping your finger sideways in the air. |
| **`Closed_Fist`** | Stop current Operation |
| **`Victory`** / **`ILoveYou`** | Take photo and video |

### 📦 Prerequisites

- Python 3.9 – 3.11
- A connected **DJI Tello Drone or DJI Robomaster TT Drone** over Wi-Fi
- **⚠️ This version of this program currently only supports Windows.**

### 🚀 Installation
**Guide Coming Soon. For now, please see requirements.txt for package installation.**

### PLEASE DO NOT USE THIS CODE WITHOUT PERMISSION.
**Any enquiries please email me at northitti@gmail.com
