import mediapipe as mp
import cv2
import csv
import os
import numpy as np
import pandas as pd
import glob
import pickle 


with open('signlanguage-03.pkl', 'rb') as f:
#with open('fingerspell-00.pkl', 'rb') as f:
    model = pickle.load(f)
    
mp_holistic = mp.solutions.holistic  
mp_drawing = mp.solutions.drawing_utils 

path = glob.glob("C:/Users/markc/OneDrive/Desktop/thesis_final/ALL_CSV/asl_fingerspell_test/letters for test/letters for test/.test/*.jpg")
#path = glob.glob("C:/Users/markc/OneDrive/Desktop/thesis_final/kyle_dataset/collectedimages/collectedimages/again/*.jpg")
#path = glob.glob("C:/Users/markc/OneDrive/Desktop/thesis_final/kyle_dataset/collectedimages/collectedimages/again/*.jpg")
#path = glob.glob("C:/Users/markc/OneDrive/Desktop/thesis_final/new_method/Images/workspace/images/collectedimages/hello-marlou/*.jpg")
#path = glob.glob("C:/Users/markc/OneDrive/Desktop/thesis_final/ALL_CSV/video/video capture (mid)/video capture (mid)/again.mp4")
#path = glob.glob("C:/Users/markc/OneDrive/Desktop/thesis/asl alphabet/asl alphabet/archive/asl-alphabet-test/A/*.jpg")
#path = glob.glob("C:/Users/markc/OneDrive/Desktop/thesis_final/ALL_CSV/asl_fingerspell_test/letters for test/letters for test/.test/*.jpg")

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


            X = pd.DataFrame([row])
            sign_language_class = model.predict(X)[0]
            sign_language_prob = model.predict_proba(X)[0]
            print(sign_language_class)
            mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.plot_landmarks(results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.plot_landmarks(results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


            cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, sign_language_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'PROB   ', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(image, str(round(sign_language_prob[np.argmax(sign_language_prob)],2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)




            cv2.imshow('ASL alphhabet', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


cv2.destroyAllWindows()



