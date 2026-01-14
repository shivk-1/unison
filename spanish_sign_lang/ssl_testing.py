import cv2 as cv
import mediapipe as mp
import pickle
import numpy as np
import warnings

import concurrent.futures

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

model_dict = pickle.load(open('model_SSL.pkl', 'rb'))
model = model_dict['model']

video_capture = cv.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

labels_dict = {
    "A": "A", "B": "B", "C": "C", "D": "D", "E": "E", "F": "F", "G": "G", "H": "H", "I": "I", "J": "J",
    "K": "K", "L": "L", "M": "M", "N": "N", "O": "O", "P": "P", "Q": "Q", "R": "R", "S": "S", "T": "T",
    "U": "U", "V": "V", "W": "W", "X": "X", "Y": "Y", "Z": "Z"
}

while True:
    landmark_data = []
    x_coords = []
    y_coords = []
    success, frame = video_capture.read()
    if not success:
        break

    height, width, _ = frame.shape

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,
                                      hand_landmarks,
                                      mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                landmark_data.append(x)
                landmark_data.append(y)
                x_coords.append(x)
                y_coords.append(y)

        x1 = int(min(x_coords) * width)
        y1 = int(min(y_coords) * height)
        x2 = int(max(x_coords) * width)
        y2 = int(max(y_coords) * height)

        if len(landmark_data) == 42:
            landmark_data.extend([0] * 42)
        elif len(landmark_data) > 84:
            landmark_data = landmark_data[:84]

        prediction = model.predict([np.asarray(landmark_data)])
        predicted_character = labels_dict[str(prediction[0])]

        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=4)
        cv.putText(frame, predicted_character, (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv.LINE_AA)

    cv.imshow('frame', frame)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

video_capture.release()
cv.destroyAllWindows()
