from email_alert import send_email
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load emotion labels and model
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = load_model('model/emotion_model.h5')

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)
last_sent = 0  # Email cooldown timer (timestamp)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y + h, x:x + w]
        try:
            roi = cv2.resize(roi, (48, 48))
        except:
            continue  # Skip face if resize fails

        roi = roi.astype('float32') / 255.0
        roi = np.expand_dims(roi, axis=0)[..., np.newaxis]

        prediction = model.predict(roi)
        predicted_emotion = emotion_labels[np.argmax(prediction)]

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        print(f"Detected Emotion: {predicted_emotion}")

        # Send email alert for "Sad" with 5-minute cooldown
        if predicted_emotion.lower() == "sad" and (time.time() - last_sent > 300):
            try:
                send_email(
                    subject="Sad Emotion Detected",
                    body="The system detected a sad expression.",
                    to_email="batch2025project@gmail.com"
                )
                print("Email sent.")
                last_sent = time.time()
            except Exception as e:
                print(f"Email failed: {e}")

    # Display the result
    cv2.imshow('Live Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
