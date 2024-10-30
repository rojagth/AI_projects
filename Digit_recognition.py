import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

path = "mnist.npz"

with np.load(path) as data:
    print("Arrays in the .npz file:", data.files)
    for array_name in data.files:
        print(f"{array_name}: {data[array_name].shape}")

    X_train, y_train = data['x_train'], data['y_train']
    X_test, y_test = data['x_test'], data['y_test']
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.show()

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten

X_train = X_train / 255.0
X_test = X_test / 255.0


model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),  
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)

import numpy as np
import matplotlib.pyplot as plt


image = X_test[2]


image = np.expand_dims(image, axis=0)


prediction = model.predict(image)
print(prediction)

# Since prediction is a 2D array (1, 10), access the first element (index 0)
probabilities = prediction[0]
print(probabilities)


for i in range(10):
    print(f"The probability of {i} is {round(probabilities[i],5)}")


predicted_class = np.argmax(probabilities)

print(f"Predicted class: {predicted_class}")
plt.imshow(X_test[0], cmap='gray')
plt.title(f"Predicted: {predicted_class}")
plt.axis('off')
plt.show()

