import tensorflow as tf
import sklearn
from sklearn.model_selection import train_test_split
from keras import models, layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NR_OF_NOTES = 16
NR_OF_EPOCHS = 50
BATCH_SIZE = 20

data = np.load('dataset.npy')
#print(np.shape(data))
train_data, test_data = train_test_split(data, test_size=0.2, train_size=0.8, shuffle=True)



train_x_data, train_y_data = [], []

for samples in train_data:
    train_x_data.append(samples[0])
    train_y_data.append(samples[1])
#print(np.shape(train_x_data))

#print(np.shape(test_data))
test_x_data, test_y_data = [], []
for samples in test_data:
    test_x_data.append(samples[0])
    test_y_data.append(samples[1])
#print(np.shape(test_x_data))

def create_model(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(85, activation='relu'))
    model.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    model.add(layers.Reshape(input_shape))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_x_data, train_y_data, epochs=100, batch_size = 1, shuffle=True, verbose=1)
    model.summary()
    return model


def calculate_accuracy(model, test_x, test_y):
    total = 0
    correct = 0
    for (x, y) in zip(test_x, test_y):
        total += 1
        prediction = model.predict(x)
        prediction = (prediction > 0.5).astype(int)
        if prediction == test_y:
            correct += 1
    return correct / total


def main():
    print(np.shape(train_x_data))
    input_shape = (4, 4)
    model = create_model(input_shape)
    accuracy = calculate_accuracy(model, test_x_data, test_y_data)
    print(accuracy)


if __name__ == '__main__':
    main()

