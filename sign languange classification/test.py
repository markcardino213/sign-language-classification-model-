import mediapipe as mp
import cv2
import csv
import os
import numpy as np
import pandas as pd
import glob


mp_drawing = mp.solutions.drawing_utils #Drawing Helpers
mp_hands = mp.solutions.hands #mediapipe solution

class_name = "asdad!"

path = glob.glob("C:/Users/markc/OneDrive/Desktop/thesis/asl alphabet/asl alphabet/asl_dataset/a/*.jpeg")


for file in path:
    image = cv2.imread(file)


    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as hands:

            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_hand_landmarks:
                print(results.multi_hand_landmarks)
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=1, circle_radius=4),
                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=1, circle_radius=2),)

            # Export coordinates
            try:
                Hand = hand.landmark
                print(Hand)
                hand_row = list(np.array([[landmark.x, landmark.y, landmark.z, ] for landmark in Hand]).flatten()) if Hand else np.zeros(21*4)

                row = hand_row

                row.insert(0, class_name)
                # Export to CSV
                with open('hand_coordinates.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row) 

            except:
                pass

            cv2.imshow('Webcam Feed', image)
            cv2.waitKey(10)


cv2.destroyAllWindows()



