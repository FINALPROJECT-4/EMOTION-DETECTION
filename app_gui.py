import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model

# Load model
model = load_model("model/emotion_model.h5")

# Emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Preprocessing function
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    face = cv2.resize(image, (48, 48))
    face = face.reshape(1, 48, 48, 1) / 255.0
    return face

# Predict emotion
def predict_emotion():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    try:
        face = preprocess_image(file_path)
        prediction = model.predict(face)
        emotion = emotions[np.argmax(prediction)]

        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)

        panel.config(image=img)
        panel.image = img
        result_label.config(text=f"Predicted Emotion: {emotion}", fg="blue")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict emotion:\n{e}")

# Tkinter window
root = tk.Tk()
root.title("Emotion Detection GUI")

frame = tk.Frame(root)
frame.pack(padx=20, pady=20)

btn = tk.Button(frame, text="Choose Image", command=predict_emotion)
btn.pack()

panel = tk.Label(frame)
panel.pack(pady=10)

result_label = tk.Label(frame, text="", font=("Helvetica", 14))
result_label.pack()

root.mainloop()
