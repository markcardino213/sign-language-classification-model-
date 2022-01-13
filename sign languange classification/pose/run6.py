import mediapipe as mp
import cv2

import csv
import os
import numpy as np

import pandas as pd
import pickle 
import glob


mp_drawing = mp.solutions.drawing_utils #Drawing Helpers
mp_hands = mp.solutions.hands #mediapipe solution

with open('signlanguage.pkl', 'rb') as f:
    model = pickle.load(f)

path = glob.glob("C:/Users/markc/OneDrive/Desktop/thesis/asl alphabet/asl alphabet/asl_dataset/z/*.jpeg")
#path = glob.glob("C:/Users/markc/OneDrive/Desktop/thesis/asl alphabet/asl alphabet/archive/l/*.jpg")
#path = glob.glob("C:/Users/markc/OneDrive/Desktop/pose/collectedimages/yes/*.jpg")

for file in path:
    image = cv2.imread(file)


    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as hands:

            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                #print(results.multi_hand_landmarks)
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4), mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),)

            try:
                Hand = hand.landmark
                #print(Hand)
                hand_row = list(np.array([[landmark.x, landmark.y] for landmark in Hand]).flatten()) if hand.landmark else np.zeros(21*2)

                row = hand_row

                X = pd.DataFrame([row])
                sign_language_class = model.predict(X)[0]
                sign_language_prob = model.predict_proba(X)[0]
                print(sign_language_class)

                #cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                #cv2.putText(image, sign_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                #cv2.putText(image, 'PROB   ', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (266, 266, 255), 1, cv2.LINE_AA)
                #cv2.putText(image, str(round(sign_language_prob[np.argmax(sign_language_prob)],2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


            except:
                pass


            cv2.imshow('Test Data', image)
            cv2.waitKey(100)



cv2.destroyAllWindows()


