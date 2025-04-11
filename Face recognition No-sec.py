#import
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
#initialize
THRESHOLD=0.5
required_frames=10
recognized_count=0
last_detected_name=""
REQUIRED_FRAMES = 10
normalizer = Normalizer(norm='l2')
facenet=FaceNet()
faces_embeddings=np.load("C:/Users/yuval/OneDrive/Desktop/faces_embeddings_done_4classes.npz")
X=normalizer.transform(faces_embeddings['arr_0'])
Y=faces_embeddings['arr_1']
encoder=LabelEncoder()
encoder.fit(Y.astype(str))
Y_encoded = encoder.fit_transform(Y)
haarcascade=cv.CascadeClassifier("C:/Users/yuval/OneDrive/Desktop/haarcascade_frontalface_default.xml")
model=pickle.load(open("C:/Users/yuval/OneDrive/Desktop/svm_model_160x160.pkl",'rb'))
#while loop
def get_best_match(embedding, threshold=THRESHOLD):
    sims = cosine_similarity(X, embedding)
    best_idx = np.argmax(sims)
    best_score = sims[best_idx][0]
    if best_score > threshold:
        name = encoder.inverse_transform([Y_encoded[best_idx]])[0]
        return name, best_score
    return "Unknown", best_score
cap=cv.VideoCapture(0)
while cap.isOpened():
    ret,frame=cap.read()
    if not ret:
        break
    rgb_img=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces=haarcascade.detectMultiScale(gray,1.3,5)
    for x,y,w,h in faces:
        img=rgb_img[y:y+h,x:x+w]
        img=cv.resize(img,(160,160))
        img=np.expand_dims(img,axis=0)
        ypred=facenet.embeddings(img)
        normalized_embedding=normalizer.transform(ypred)
        final_name, score = get_best_match(normalized_embedding)
        face_name=model.predict(normalized_embedding)
        confidence=model.decision_function(normalized_embedding)
        if np.max(confidence[0])>0.8 and score>0.6 :
            if final_name!="Unknown":
                if final_name==last_detected_name:
                    recognized_count+=1
                else:
                    last_detected_name=final_name
                    recognized_count+=1
            else:
                recognized_count=0
                last_detected_name=""
            if recognized_count >= REQUIRED_FRAMES:
                recognized_count=0
                last_detected_name=""
        else:
            final_name="Unknown"
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
        cv.putText(frame,str(final_name),(x,y-10),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3,cv.LINE_AA)
    cv.imshow('Face Recognititon: ',frame)
    if cv.waitKey(1)& ord('g')==27:
        break
cap.release()
cap.destroyAllWindows()