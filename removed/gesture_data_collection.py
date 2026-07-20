import os
import cv2

#Collect data by taking picutres

dataset_dir = "custom_gesture_dataset"
gestures = "point_down", "three_fingers", "flat_palm", "L_sign", "ok_sign"

for gesture in gestures:
    os.makedirs(os.path.join(dataset_dir, gesture), exist_ok=True)

current_gesture = gestures[0]
count = 0

cap = cv2.VideoCapture(0)

print(f"Recording {current_gesture} gesture. Press 's' to save, 'd' for next gesture, 'q' to quit.")

while True:
    ret, frame = cap.read()

    cv2.putText(frame, f"Record: {current_gesture} | Saved: {count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        #save gesture
        filename = os.path.join(dataset_dir, current_gesture, f"{count}.jpg")
        cv2.imwrite(filename, frame)
        count += 1
        print(f"Saved {filename}")

    elif key == ord('d'):
        index = (gestures.index(current_gesture) + 1) % len(gestures) #neat way to prevent Index out of range error.
        current_gesture = gestures[index]
        count = 0
        print(f"Switched to {current_gesture} gesture.")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()