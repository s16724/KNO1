import tensorflow as tf
from sklearn import svm
from sklearn.metrics import accuracy_score

fashion_mnist = tf.keras.datasets.fashion_mnist
print(fashion_mnist)
# przygotowujemy zestawy z powyższego zbioru
(training_clothes, training_labels), (test_clothes, test_labels) = fashion_mnist.load_data()

X_training = training_clothes
X_test = test_clothes
y_training = training_labels
y_test = test_labels

# dzielimy zestawy na 255 przez co uzyskamy większą skuteczność uczenia
training_clothes, test_clothes = training_clothes / 255, test_clothes / 255

#rozpoznawania ubrań korzystając z klasyfikatora Support Vector Machine

X_training_svc = X_training.reshape(60000, 784)# obróbka danych na 2-klasowe pasujące do modelu SVM
X_test_scv = X_test.reshape(10000, 784)
x, y = X_training_svc[:10000], y_training[:10000]
x_test_svc_to = X_test_scv[:1000]
y_test_svc_to = y_test[:1000]

svc = svm.SVC(C=100, gamma=0.001)
svc.fit(x, y)

svc_prediction = svc.predict(x_test_svc_to)# przewidywanie ubrań oraz ocena dokładności
print('Accuracy SVM:', accuracy_score(y_test_svc_to, svc_prediction))
