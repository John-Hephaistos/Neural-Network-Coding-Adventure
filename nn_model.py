import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self._layers = nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.Sigmoid(),
            torch.nn.Linear(256, output_size)
        )
        self.double()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        self.loss_function = torch.nn.MSELoss()

    def train_model(self, epochs, train_x_data, train_y_data):


        for epoch in range(epochs):
            for X, y in zip(train_x_data, train_y_data):
                X = torch.tensor(X, dtype=torch.double)  # Ensure input is of type double
                y = torch.tensor(y, dtype=torch.double)
                output = self.forward(X)
                loss = self.loss_function(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print(f"Epoch: {epoch + 1}; Loss: {loss.item()}")

    def test_model(self, test_x_data, test_y_data):
        total_steps = len(test_x_data)
        correct = 0
        for X, y in zip(test_x_data, test_y_data):
            X = torch.tensor(X, dtype=torch.double)  # Ensure input is of type double
            y = torch.tensor(y, dtype=torch.int)
            prediction = (self.forward(torch.tensor(X, dtype=torch.double))).to(torch.int)
            if torch.equal(prediction, y):
                correct += 1
        print(correct / total_steps)


    def forward(self, model_input):
        return self._layers(model_input)



def main():


    data = np.load('dataset.npy')

    train_data, test_data = train_test_split(data, test_size=0.2, train_size=0.8, shuffle=True)

    train_x_data, train_y_data = [], []

    for samples in train_data:
        train_x_data.append(samples[0].flatten())
        train_y_data.append(samples[1].flatten())
    # print(np.shape(train_x_data))

    # print(np.shape(test_data))
    test_x_data, test_y_data = [], []
    for samples in test_data:
        test_x_data.append(samples[0].flatten())
        test_y_data.append(samples[1].flatten())

    #train_x_data = tuple(test_x_data)


    print((test_x_data[1]))



    model = MLP(16, 16)
    print(model)
    model.train()
    model.train_model(10, train_x_data, train_y_data)
    model.test_model(test_x_data, test_y_data)
    #prediction = (model.forward(torch.tensor(test_x_data[1], dtype=torch.double))).to(torch.int)
    #print(prediction)

if __name__ == '__main__':
    main()
