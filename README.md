# Autonomous Gesture-Controlled Tello Drone Software

A real-time, low-latency autonomous flight control system for the **DJI Tello** drone. This project combines a **custom TensorFlow model** trained on hand-landmark spatial geometry with **MediaPipe's Gesture Recognizer Tasks API** to provide precise, real-time gesture piloting over live video stream.

---

## ✨ Key Features
- **Film using Gestures**: use hand gestures to record and stop video.
- **Maneuvers using Gestures**: Execute maneuvers using only gestures.
- **Automatic Following**: Use a gesture to get your drone to follow you around!
---

## ⚙️ Supported Gestures & Flight Controls

### Custom TensorFlow Model Gestures
| Gesture | Command Trigger | Description |
| :--- | :--- | :--- |
| **`ok_sign`** | **Takeoff** | Initiates autonomous takeoff. |
| **`flat_palm`** | **Land** | Triggers automated landing sequence. |
| **`point_down`** | **Move Down** | Command drone to descend in height. |
| **`L_sign`** | **Orbit Mode** | Triggers automated orbital maneuver around subject. |
| **`three_fingers`** | **Follow Person** | Activates vision-based person tracking. |

### MediaPipe Fallback Gestures
| Gesture | Default Reaction |
| :--- | :--- |
| **`Thumb_Up`** | Backup Takeoff trigger |
| **`Thumb_Down`** / **`Closed_Fist`** | Auxiliary system triggers |
| **`Victory`** / **`ILoveYou`** | Custom macro function triggers |

### 📦 Prerequisites

- Python 3.9 – 3.11
- A connected **DJI Tello Drone** over Wi-Fi
- **⚠️ This version of this program only supports Windows. A mobile app for this program may be considered in the future!**

### 🚀 Installation
**Guide Coming Soon. For now, please see requirements.txt for package installation.**
