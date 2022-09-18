# American-sign-language-classification-model-
A classification model utilizing MediaPipe as a tracking algorithm and feature extractor and various machine learning algorithms focusing in Word Level American Sign Language (WALS)
![fig4 2](https://user-images.githubusercontent.com/60088090/171367944-dd477fb1-d585-4084-af82-db5b9c9fc09d.png)

pip install mediapipe opencv-python pandas numpy scikit-learn
# Collect dataset: 
  •go dataset folder and run Collect image.ipynb
# Extracting keypoints:
  •go to python folder\n
  •create a .csv file every time you will extract a new class of dataset by running create_csv_v2.1.py
  •run create_csv_v2.1.py after every created .csv
# Training
  •merge first all csv (all letters for letters or all words for words) into one csv by running merge_csv.py.
  •open train.py and change the 4.csv depending on the name of the csv you want to train first.
  •rename the .pkl file depending on the dataset trained to avoid confusion
  •run train.py
# Testing
  •change the .pkl file first based on the name of the .pkl file in train.py
  •change the directory depending on where your test dataset your gonna use.
  •run test_run1.py 
# Real-time testing
  •change the .pkl file first based on the name of the .pkl file in train.py
  •run live_capture.py
