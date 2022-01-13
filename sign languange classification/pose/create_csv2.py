from matplotlib import pyplot as plt
import time
import mediapipe as mp
import cv2
import csv
import os
import numpy as np
import pandas as pd
import glob

mp_holistic = mp.solutions.holistic 
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils 


cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            holistic_results = holistic.process(image)
            hand_results = hands.process(image)
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            mp_drawing.draw_landmarks(image, holistic_results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=1, circle_radius=2),mp_drawing.DrawingSpec(color=(245,66,230), thickness=1, circle_radius=2))

            mp_drawing.draw_landmarks(image, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            if hand_results.multi_hand_landmarks:
                for num, hand in enumerate(hand_results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),)


            try:
                face = holistic_results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten() if  holistic_results.pose_landmarks else np.zeros(468*3))

                pose = holistic_results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten() if  holistic_results.pose_landmarks else np.zeros(33*4))

                Hand = hand.landmark
                hand_row = list(np.array([[landmark.x, landmark.y, landmark.z,  landmark.visibility] for landmark in Hand]).flatten() if hand.landmark else np.zeros(21*4))


            except:
                pass

            cv2.imshow('Raw Webcam Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


        
num_coords = len(holistic_results.face_landmarks.landmark)+len(holistic_results.pose_landmarks.landmark)+len(hand.landmark)


landmarks = ['class']
for val in range(1, num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]


with open('pose_coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

cap.release()
cv2.destroyAllWindows()
