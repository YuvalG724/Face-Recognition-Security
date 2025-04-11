import cv2 as cv
import numpy as np
import os
import pickle
import requests
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.metrics.pairwise import cosine_similarity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
THRESHOLD = 0.5
REQUIRED_FRAMES = 10
ESP32_URL = "http://<ESP IP>/unlock"
facenet = FaceNet()
normalizer = Normalizer(norm='l2')
haarcascade = cv.CascadeClassifier("") #Haar cascade file
faces_embeddings = np.load("") #Face Embeddings file
X = normalizer.transform(faces_embeddings['arr_0'])
Y = faces_embeddings['arr_1']
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
recognized_count = 0
last_detected_name = ""
model=pickle.load(open("","rb"))#Svm file
def get_best_match(embedding, threshold=THRESHOLD):
    sims = cosine_similarity(X, embedding)
    best_idx = np.argmax(sims)
    best_score = sims[best_idx][0]
    if best_score > threshold:
        name = encoder.inverse_transform([Y_encoded[best_idx]])[0]
        return name, best_score
    return "Unknown", best_score
cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = haarcascade.detectMultiScale(gray, 1.3, 5)
    for x,y,w,h in faces:
        face_img = rgb_img[y:y + h, x:x + w]
        face_img = cv.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)
        embedding = facenet.embeddings(face_img)
        normalized_embedding = normalizer.transform(embedding)
        final_name, score = get_best_match(normalized_embedding)
        confidence=model.decision_function(normalized_embedding)
        if (np.max(confidence[0]))>0.8 and score>0.6:
            if final_name != "Unknown":
                if final_name == last_detected_name:
                    recognized_count += 1
                else:
                    recognized_count = 1
                    last_detected_name = final_name
            else:
                recognized_count = 0
                last_detected_name = ""
            if recognized_count >= REQUIRED_FRAMES and final_name in [""]:
                print(f"Unlocking for {final_name}")
                requests.get(ESP32_URL)
                recognized_count = 0
                last_detected_name = ""
            else:
                print("Access Denied!")
        else:
            final_name="Unknown"
            print("Access Denied!")
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, final_name, (x, y - 10),
        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    cv.imshow('Face Recognition', frame)
    if cv.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv.destroyAllWindows()
