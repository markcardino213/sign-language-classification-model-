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

class_name = "hello"

path = glob.glob("C:/Users/markc/OneDrive/Desktop/pose/collectedimages/hello/*.jpg")


for file in path:
    image = cv2.imread(file)


    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        with mp_hands.Hands(max_num_hands=2, min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as hands:
            
            holistic_results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            hand_results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            mp_drawing.draw_landmarks(image, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            if hand_results.multi_hand_landmarks:
                for num, hand in enumerate(hand_results.multi_hand_landmarks):
                        mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),)


            try:
                pose = holistic_results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten()if  holistic_results.pose_landmarks else np.zeros(33*3))

                Hand = hand.landmark
                hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, ] for landmark in Hand]).flatten()) if hand.landmark else list(np.zeros(21*3))


                row = pose_row+hand_row
                row.insert(0, class_name)
                    
                with open('pose_coords.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row) 

            except:
                pass

            cv2.imshow('Webcam Feed', image)
            cv2.waitKey(10)


cv2.destroyAllWindows()



