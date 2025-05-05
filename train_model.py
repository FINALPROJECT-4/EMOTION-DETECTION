import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('data/fer2013.csv')
pixels = df['pixels'].tolist()
faces = np.array([np.fromstring(p, sep=' ') for p in pixels], dtype='float32')
faces = faces.reshape((-1, 48, 48, 1)) / 255.0
emotions = to_categorical(df['emotion'])

model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(faces, emotions, epochs=15, batch_size=64, validation_split=0.1)

model.save('model/emotion_model.h5')