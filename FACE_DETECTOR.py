import cv2
from keras.models import model_from_json
import numpy as np

# Load the pre-trained model
json_file = open("emotionsdetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotionsdetector.h5")

# Load the face cascade classifier
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Define emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Emotion-related questions
emotion_questions = {
    'angry': 'Why are you feeling angry?',
    'disgust': 'What is causing you to feel disgusted?',
    'fear': 'What are you afraid of?',
    'happy': 'What is making you happy?',
    'neutral': 'How are you feeling right now?',
    'sad': 'What is making you feel sad?',
    'surprise': 'What surprised you?'
}

# Open webcam feed
webcam = cv2.VideoCapture(0)

while True:
    ret, im = webcam.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (p, q, r, s) in faces:
        image = gray[q:q + s, p:p + r]
        cv2.rectangle(im, (p, q), (p + r, q + s), (255, 0, 0), 2)
        image = cv2.resize(image, (48, 48))
        img = extract_features(image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        
        # Display emotion prediction and question
        cv2.putText(im, f'Emotion: {prediction_label}', (p - 10, q - 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255))
        question = emotion_questions.get(prediction_label, 'How are you feeling?')
        cv2.putText(im, question, (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (0, 0, 255))
        
    cv2.imshow("Output", im)
    
    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
