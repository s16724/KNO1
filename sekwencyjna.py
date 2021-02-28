import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense

fashion_mnist = tf.keras.datasets.fashion_mnist
# przygotowujemy zestawy z powyższego zbioru
(training_pictures, training_labels), (test_pictures, test_labels) = fashion_mnist.load_data()

X_training = training_pictures
X_test = test_pictures
y_training = training_labels
y_test = test_labels

# dzielimy zestawy na 255 przez co uzyskamy większą skuteczność uczenia
training_pictures = training_pictures / 255
test_pictures = test_pictures / 255
# budowa sieci sekwencyjnej
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))



model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# trenowanie modelu
history = model.fit(training_pictures, training_labels, validation_data=(X_test, y_test), epochs=8)

# ocena dokładności
test_loss, test_accuracy = model.evaluate(test_pictures, test_labels)
print("test_loss ", test_loss)
print("test_accuracy ", test_accuracy)
# przewidywanie
prediction = model.predict(test_pictures)
np.set_printoptions(suppress=True)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
