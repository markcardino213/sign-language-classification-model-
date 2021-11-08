import mediapipe as mp
import cv2

import csv
import os
import numpy as np

import pandas as pd
import pickle 

mp_drawing = mp.solutions.drawing_utils #Drawing Helpers
mp_hands = mp.solutions.hands #mediapipe solution

with open('hand_sign_language.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)
#imagecls = cv2.imread('B_test.jpg', -0) 

with mp_hands.Hands(max_num_hands=1, min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False        
        results = hands.process(image)
        image.flags.writeable = True   
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #results = hands.process(image)

        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),)

        # Export coordinates
        try:
            Hand = hand.landmark
            #print(Hand)
            hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, ] for landmark in Hand]).flatten()) if hand.landmark else np.zeros(21*4)

            row = hand_row

            X = pd.DataFrame([row])
            sign_language_class = model.predict(X)[0]
            sign_language_prob = model.predict_proba(X)[0]
            print(sign_language_class)

        except:
            pass

        cv2.imshow('Webcam Feed', image)
        #cv2.waitKet(3000)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cv2.destroyAllWindows()



