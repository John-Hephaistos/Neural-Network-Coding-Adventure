import numpy as np
import torch
import torch.nn as nn
import sklearn
from torch.utils.data import Subset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
import itertools
from EarlyStopping import EarlyStopping


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self._layers = nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 664),
            torch.nn.ReLU(),
            torch.nn.Linear(664, 664),
            torch.nn.ReLU(),
            torch.nn.Linear(664, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_size),
            torch.nn.Softmax()
        )
        self.double()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss_function = torch.nn.BCELoss()

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


class LSTM(nn.Module):
    def __init__(self, hidden_dim, num_layers, input_dim=4, output_dim=4):
        super(LSTM, self).__init__()
        
        # Parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Initialize cell state with zeros
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        
        # Pass through fully connected layer
        out = self.fc(out)
        
        return out
    
    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001, device='cpu'):
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.train()

        # Training
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for i, (inputs, targets) in enumerate(train_loader):
                outputs = self(inputs.to(device))
                loss = criterion(outputs, targets.to(device))

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')

    def validate(self, val_loader, criterion=nn.BCEWithLogitsLoss(), device='cpu'):
        self.eval()
        val_loss = 0.0
        count = 0
        early_stopper = EarlyStopping()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self(inputs.to(device))
                loss = criterion(outputs, targets.to(device))

                if early_stopper.check(self, loss) is False:
                    val_loss += loss.item()
                    count += 1
                else:
                    print("Stopped EARLYY")
                    break
        return val_loss / count

def cross_validation(model, dataset, batch_size, num_folds=5, num_epochs=10, learning_rate=0.001, device='cpu'):
    fold_size = len(dataset) // num_folds
    fold_losses = []

    for fold in range(num_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size

        # Create train and validation subsets
        train_subset = Subset(
            dataset,
            list(range(0, val_start)) + list(range(val_end, len(dataset)))
        )
        val_subset = Subset(dataset, list(range(val_start, val_end)))

        # Train and validation data loaders
        train_loader = torch.utils.data.DataLoader(
            dataset=train_subset,
            batch_size=batch_size,
            shuffle=True
        )


        val_loader = torch.utils.data.DataLoader(
            dataset=val_subset,
            batch_size=batch_size,
            shuffle=False
        )

        # New model for this fold, copy parameters
        fold_model = LSTM(
            input_dim=model.input_dim,
            hidden_dim=model.hidden_dim,
            num_layers=model.num_layers,
            output_dim=model.output_dim
        )
        fold_model.to(device)

        # Train the model
        fold_model.train_model(
            train_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device
        )

        # Evaluate the model on the validation set
        val_loss = fold_model.validate(val_loader, device=device)
        fold_losses.append(val_loss)

    return fold_losses

def grid_search(dataset, parameter_grid, device):
    best_params = None
    best_loss = float('inf')

    # Generate all combinations of parameters
    param_combinations = list(itertools.product(*parameter_grid.values()))

    for params in param_combinations:
        hidden_dim, num_layers, learning_rate, batch_size, num_epochs, num_folds = params
        # Create a model and set parameters
        model = LSTM(hidden_dim, num_layers)
        model.to(device)
        model.hidden_dim, model.num_layers = hidden_dim, num_layers

        # Perform cross-validation with the current parameters
        fold_losses = cross_validation(
            model,
            dataset,
            batch_size=batch_size,
            num_folds=num_folds,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device
        )

        # Compute the average validation loss
        avg_loss = np.mean(fold_losses)

        # Check if loss is better
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params = params

    return best_params, best_loss


def threshold_predictions(predictions, threshold = 0.8):
    '''
    Generate a numpy array with thresholded predicitons
    1 if >= threshold, 0 otherwise, per note
    '''
    thresholded_predictions = (predictions >= threshold).float()
    return thresholded_predictions.cpu().numpy()

def main():
    data = np.load('dataset.npy')

    # train_data, test_data = train_test_split(data, test_size=0.2, train_size=0.8, shuffle=True)

    # train_x_data, train_y_data = [], []
    # for samples in train_data:
    #     train_x_data.append(samples[0].flatten())
    #     train_y_data.append(samples[1].flatten())
    # # print(np.shape(train_x_data))

    # # print(np.shape(test_data))
    # test_x_data, test_y_data = [], []
    # for samples in test_data:
    #     test_x_data.append(samples[0].flatten())
    #     test_y_data.append(samples[1].flatten())

    #train_x_data = tuple(test_x_data)


    # model = MLP(16, 16)
    # print(model)
    # model.train()
    # model.train_model(30, train_x_data, train_y_data)
    # model.test_model(test_x_data, test_y_data)
    #prediction = (model.forward(torch.tensor(test_x_data[1], dtype=torch.double))).to(torch.int)
    #print(prediction)


    x_data, y_data = [], []
    for samples in data:
        x_data.append(samples[0])
        y_data.append(samples[1])

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_data, dtype=torch.float32),
        torch.tensor(y_data, dtype=torch.float32)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parameter_grid = {
        'hidden_dim': [32, 64, 128],
        'num_layers': [1, 2, 3],
        'learning_rate': [0.001, 0.01],
        'batch_size': [64, 128, 256],
        'num_epochs': [100],
        'num_folds': [10],
    }

    best_params, best_loss = grid_search(
        dataset,
        parameter_grid,
        device
    )

    print(best_loss)
    print(best_params)

    # test_x_data = torch.tensor(test_x_data, dtype=torch.float32).to(device)

    # # Generate predictions
    # model.eval()  # Set the model to evaluation mode
    # with torch.no_grad():  # No need to track gradients during inference
    #     test_predictions = model(test_x_data)
    

    # predictions_thresh = threshold_predictions(test_predictions)
    # np.save('test_pred.npy', predictions_thresh[0])
    # print(predictions_thresh.shape)

if __name__ == '__main__':
    main()

