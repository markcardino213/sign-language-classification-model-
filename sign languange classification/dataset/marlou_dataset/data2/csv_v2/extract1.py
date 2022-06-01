import cv2
import csv
import numpy as np
import mediapipe as mp
import glob

mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

class_name = "you"

path = glob.glob("C:/Users/markc/OneDrive/Desktop/thesis_final/marlou_dataset/you-marlou/*.jpg")

def mediapipe_detection(image, model):     
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                             


for file in path:
    image = cv2.imread(file)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image, results = mediapipe_detection(image, holistic)
            draw_landmarks(image, results)

            def extract_keypoints(results):
                pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
                lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
                rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
                return np.concatenate([pose, lh, rh])

            result_test = extract_keypoints(results)
            row = list(result_test)
            row.insert(0, class_name)

            print(row)
            with open('you.csv', mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(row)

    cv2.imshow('Webcam Feed', image)
    cv2.waitKey(10)

cv2.destroyAllWindows()
