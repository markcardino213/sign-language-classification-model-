from matplotlib import pyplot as plt
import time
import mediapipe as mp
import cv2
import csv
import os
import numpy as np
import pandas as pd
import pickle 


mp_holistic = mp.solutions.holistic 
mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils 

with open('pose2.pkl', 'rb') as f:
    model = pickle.load(f)

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
            

            mp_drawing.draw_landmarks(image, holistic_results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2))

            if hand_results.multi_hand_landmarks:
                for num, hand in enumerate(hand_results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),)


            try:
                pose = holistic_results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten() if  holistic_results.pose_landmarks else np.zeros(33*3))

                Hand = hand.landmark
                hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility ] for landmark in Hand]).flatten() if hand.landmark else np.zeros(21*3))


                row =pose_row+hand_row

                X = pd.DataFrame([row])
                sign_language_class = model.predict(X)[0]
                sign_language_prob = model.predict_proba(X)[0]
                print(sign_language_class)

                cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, sign_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'PROB   ', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (266, 266, 255), 1, cv2.LINE_AA)
                cv2.putText(image, str(round(sign_language_prob[np.argmax(sign_language_prob)],2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


            except:
                pass

            cv2.imshow('Raw Webcam Feed', image)
            #cv2.waitKey(100)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
