import cv2
import time
import threading
import tensorflow as tf
import numpy as np
import math
from ultralytics import YOLO

def camera():
    global cap
    global frame_2
    global camera_start

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame_2 = cap.read()
        camera_start = True
        if not ret:
            break

def head_count_cam():
    global yolo_count
    global center_points_prev_frames
    global frame_count
    global frame_2
    global stop

    while not stop:
        frame = frame_2.copy()
        frame_count += 1

        center_points = []
        model = YOLO('yolov8n.pt') 
        
        results = model(frame, show=False, stream=True, verbose=False)

        for r in results:
            # arr.append(r)
            boxes = r.boxes
            for box in boxes:
                if int(box.cls) == 0:
                    if box.conf < 0.5:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1 , y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cx = int((x1 + x2)/2)
                    cy = int((y1 + y2)/2)
                    center_points.append((cx, cy))

        if center_points_prev_frames == []:
            yolo_count = max(yolo_count, len(center_points))

        else:
            for i in center_points:
                unique = True
                for k in center_points_prev_frames:
                    for j in k:
                        dist = math.hypot(j[0] - i[0], j[1] - i[1])
                        if dist < 60:
                            unique = False
                            break
                            
                    if unique == False:
                        break

                if unique == True:
                    yolo_count += 1

        cnt = len(center_points)

        if frame_count < 50:
            center_points_prev_frames.append(center_points)
        else:
            try:
                center_points_prev_frames[frame_count%49] = center_points
            except:
                print("frame yolo_count", frame_count)
                print("len", len(center_points_prev_frames))
            
center_points_prev_frames = []
yolo_count = 0
frame_count = 0
stop = False

camera_start = False
threading.Thread(target=camera).start()
while True:
    if camera_start:
        break
    time.sleep(0.5)
    print("waiting for camera to start")

threading.Thread(target=head_count_cam).start()

threshold = 0.6 # Threshold for face verification
surprise = 2.3 # 2.3
netural = 2.3 # 2.8
sad = 0.5 # 0.45
happy = 2.1 # 2.1

age_model = tf.keras.models.load_model(r"C:\Users\Laptop-06\Downloads\model_1\model\model_4.h5")
gender_model = tf.keras.models.load_model(r"C:\Users\Laptop-06\Downloads\model_1\model\gender__model_6.h5")
emotion_model = tf.keras.models.load_model(r"C:\Users\Laptop-06\Downloads\model_1\model\4_emotion_2.h5")
face_cascade = cv2.CascadeClassifier(r"C:\Users\Laptop-06\Downloads\model_1\model\haarcascade_frontalface_default.xml")
group = ['1-10', '11-17', '18-24', '25-30', '31-38', '39-46', '47-55', '55+']
multiplier=[1.0,   1.0,     0.75,    1.0,     0.6,     1.15,     1.0,   1.0] # model_5.h5
emotions = ['surprise', 'neutral', 'sad', 'happy']

while True:
    frame = frame_2.copy()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 11)

    for (x, y, w, h) in faces:
        img = frame[y:y+h, x:x+w]

        # emotion prediction
        img_emotion = cv2.resize(img, (48, 48))
        img_emotion = cv2.cvtColor(img_emotion, cv2.COLOR_BGR2GRAY)
        img_emotion = np.array(img_emotion).reshape(-1, 48, 48, 1)
        emotion = emotion_model.predict(img_emotion, verbose=0)
        emotion[0][3]  = emotion[0][3] * happy # happy
        emotion[0][0]  = emotion[0][0] * surprise # surprise
        emotion[0][1]  = emotion[0][1] * netural # neutral
        emotion[0][2]  = emotion[0][2] * sad # sad
        emotion = emotion[0].argmax()
        emotion = emotions[emotion]
        # result['emotion'] = emotions[emotion]

        # age prediction
        img_age = cv2.resize(img, (50, 50))
        img_age = np.array(img_age).reshape(-1, 50, 50, 3)
        age = age_model.predict(img_age, verbose=0)
        age = age[0].argmax()
        age = group[age]
        # result['age'] = group[age]

        # gender prediction
        img_gender = cv2.resize(img, (100, 100))
        img_gender = np.array(img_gender).reshape(-1, 100, 100, 3)
        gender = gender_model.predict(img_gender, verbose=0)[0][0]
        if gender > 0.5:
            gender = 'Female'
            # result['ismale'] = False
        else:
            gender = 'Male'
            # result['ismale'] = True

        # show the result on the screen
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Emotion: {emotion}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(frame, f'Age: {age}', (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.putText(frame, f'Gender: {gender}', (x, y-70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.putText(frame, f'Count: {yolo_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop = True
        break

cap.release()
cv2.destroyAllWindows()