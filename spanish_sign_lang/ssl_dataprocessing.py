import mediapipe as mp
import cv2 as cv
import os
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data = []
labels = []

datadir = 'Data_SSL'

for letter_dir in os.listdir(datadir):
    for img_path in os.listdir(os.path.join(datadir, letter_dir)):
        landmarks = []
        img = cv.imread(os.path.join(datadir, letter_dir, img_path))
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    landmarks.append(x)
                    landmarks.append(y)
            data.append(landmarks)
            labels.append(letter_dir)

f = open('data_SSL.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
