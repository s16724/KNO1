import matplotlib.pyplot as plt
import tensorflow as tf


fashion_mnist = tf.keras.datasets.fashion_mnist
# przygotowujemy zestawy z powyższego zbioru
(training_clothes, training_labels), (test_clothes, test_labels) = fashion_mnist.load_data()

X_training = training_clothes
X_test = test_clothes
y_training = training_labels
y_test = test_labels

# dzielimy zestawy na 255 przez co uzyskamy większą skuteczność uczenia
training_clothes, test_clothes = training_clothes / 255, test_clothes / 255

# przygotowanie danych do procesowania


X_training = X_training.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
print(X_training.shape)

X_training_svc = X_training.reshape(60000, 784)
X_test_scv = X_test.reshape(10000, 784)

x, y = X_training_svc[:10000], y_training[:10000]
x_test_svc_to = X_test_scv[:1000]
y_test_svc_to = y_test[:1000]

y_training_one_hot = tf.keras.utils.to_categorical(y_training)  # to_categorical <-- lepsze dopasowanie do sieci
y_test_one_hot = tf.keras.utils.to_categorical(y_test)
print(y_training_one_hot[0])

#Budowa modelu oraz jego kompilacja

CNN = tf.keras.Sequential()
CNN.add(tf.keras.layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
CNN.add(tf.keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
CNN.add(tf.keras.layers.Flatten())
CNN.add(tf.keras.layers.Dense(10, activation='softmax'))

# kompilowanie modelu

CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["acc"])

hist = CNN.fit(X_training, y_training_one_hot, validation_data=(X_test, y_test_one_hot), epochs=5)


plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()




