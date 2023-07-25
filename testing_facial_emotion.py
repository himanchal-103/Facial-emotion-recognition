import cv2
import numpy as np
from keras.models import load_model

emotion_dict={0:'Neutral' ,1:'Contempt', 2:'Disgust', 3:'Fear', 4:'Sadness', 5:'Anger', 6:'Happiness', 7:'Surprise'}

emotion_model = load_model('facial_emotion_recognition_model.h5')

# start the default webcam feed
# cap = cv2.VideoCapture(0)

# detecting expression on recorded video
cap = cv2.VideoCapture("hapiness_face_reaction.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    # Find haarcascade to draw bounding box around the face
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

    # Detecting the faces available on camera 
    num_faces = face_detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    #preprocessing the faces available on camera
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (100, 100)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
