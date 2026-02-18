import numpy as np
import cv2 
import mediapipe as mp 
import csv 
import os 

mp_hands = mp.solutions.hands 
mp_draw = mp.solutions.drawing_utils 
hands = mp_hands.Hands( 
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7) 

cap = cv2.VideoCapture(0) 

file_name = "hand_detect.csv" 
file_exists = os.path.isfile(file_name)

while True: 
    ret, frame = cap.read() 
    if not ret: 
        break 

    frame = cv2.flip(frame, 1) 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    results = hands.process(rgb) 
    h, w,_ = frame.shape 

    if results.multi_hand_landmarks: 
        for hand_landmarks in results.multi_hand_landmarks: 
            mp_draw.draw_landmarks( 
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS) 
            
            data = [] 

            wrist = hand_landmarks.landmark[0]
            base_x, base_y, base_z = wrist.x, wrist.y, wrist.z

            for lm in hand_landmarks.landmark: 
                data.append(lm.x-base_x) 
                data.append(lm.y-base_y) 
                data.append(lm.z-base_z) 

    key = cv2.waitKey(1) & 0xFF 

    label = "i" 
                
    if key == ord('s'): 
        with open(file_name, 'a', newline='') as f: 
            writer = csv.writer(f) 
            writer.writerow(data+[label]) 
            print("Saved") 
    cv2.imshow("Collecting data", frame) 
    if key == ord('q'): 
        break 
cap.release() 
cv2.destroyAllWindows() 
            

data = np.loadtxt("hand_detect.csv", delimiter=",", dtype=str)

print("Dataset loaded successfully")
print("Shape:", data.shape)

# Split into features and labels
X = data[:, :-1].astype(float)  # first 63 columns
y = data[:, -1]                 # last column

unique, counts = np.unique(y, return_counts=True)

print("\nSamples per label:")
for label, count in zip(unique, counts):
    print(label, ":", count)

print("Ntuli")
print(X.shape)
print(y.shape)
