import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt


training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")
target_data = np.array([[0],[1],[1],[0]], "float32")


model = tf.keras.Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error',optimizer='adam', metrics=['binary_accuracy','mse','accuracy'])

class WeightHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(WeightHistory, self).__init__()
        self.layer0_history = []
        self.layer1_history = []

    def on_epoch_end(self, epoch, logs=None):
        self.layer0_history.append(model.layers[0].get_weights()[0].flatten())
        self.layer1_history.append(model.layers[1].get_weights()[0].flatten())
    
weight_history_callback = WeightHistory()
history = history = model.fit(training_data, target_data, epochs=5000, verbose=0,callbacks=[weight_history_callback])


num_epochs = len(weight_history_callback.layer0_history)

num_neurons_l0 = len(weight_history_callback.layer0_history[0])
for neuron in range(num_neurons_l0):
    plt.plot(range(num_epochs),[i[neuron] for i in weight_history_callback.layer0_history])
plt.title('Wagi pomiędzy warstwą wejściową, a ukrytą')
plt.xlabel('Epoki')
plt.ylabel('Waga')
plt.show()

num_neurons_l1 = len(weight_history_callback.layer1_history[0])
for neuron in range(num_neurons_l1):
    plt.plot(range(num_epochs),[i[neuron] for i in weight_history_callback.layer1_history])
plt.title('Wagi pomiędzy warstwą ukrytą, a wejściową')
plt.xlabel('Epoki')
plt.ylabel('Wagi')
plt.show()

plt.plot(history.history['mse'])
plt.title('Błąd średniokwadratowy (MSE)')
plt.ylabel('Błąd średniokwadratowy')
plt.xlabel('Epoki')
plt.show()

plt.plot(history.history['loss'])
plt.title('Błąd klasyfikacji')
plt.ylabel('Błąd klasyfikacji')
plt.xlabel('Epoki')
plt.show()

