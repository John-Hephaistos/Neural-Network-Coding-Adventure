import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Subset
import torch.optim as optim
import itertools
from EarlyStopping import EarlyStopping
from sound_helpers import generate_track

class LSTM(nn.Module):
    def __init__(self, hidden_dim, num_layers, dropout=0.5, input_dim=4, output_dim=4):
        super(LSTM, self).__init__()
        
        # Parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
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
        epoch_losses = []
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
            epoch_losses.append(epoch_loss / len(train_loader))

            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}')
            # Plotting the loss

            plt.figure()
            plt.plot(epoch_losses, label='Training loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss over Epochs')
            plt.legend()
            plt.grid(True)
            plt.savefig("plot_loss.png")
            plt.close()

    def validate(self, val_loader, criterion=nn.BCEWithLogitsLoss(), device='cpu'):
        self.eval()
        val_loss = 0.0
        count = 0
        early_stopper = EarlyStopping()
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = self(inputs.to(device))
                loss = criterion(outputs, targets.to(device))
                val_loss += loss.item()
                count += 1
        return val_loss / count

    def generate(self, input_sequence, length=16, threshold=.5, device='cpu'):
        output = []
        for _ in range(length):
            with torch.no_grad():
                generated = self.forward(input_sequence.to(device))
                output.append(generated)
                input_sequence = generated

        output = torch.cat(output, dim=0)  # Concatenate along the batch dimension
        output = output.view(-1, output.size(-1))  # Flatten the first two dimensions

        thresholded_tensor = torch.where(output >= threshold, torch.tensor(1), torch.tensor(0))

        # Convert tensor to NumPy array
        return thresholded_tensor.cpu().numpy()


def cross_validation(model, dataset, batch_size, num_folds=5, num_epochs=10, learning_rate=0.001, dropout=0.5, device='cpu'):
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
            output_dim=model.output_dim,
            dropout=dropout,
        )
        fold_model.to(device)

        # Train the model
        fold_model.train_model(
            train_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
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
        hidden_dim, num_layers, learning_rate, dropout, batch_size, num_epochs, num_folds = params
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
            dropout=dropout,
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

def xy_split_dataset(data):
    x_data, y_data = [], []

    for samples in data:
        x_data.append(samples[0])
        y_data.append(samples[1])

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(x_data, dtype=torch.float32),
        torch.tensor(y_data, dtype=torch.float32)
    )

    return dataset

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parameter_grid = {
        'hidden_dim': [64, 128],
        'num_layers': [2, 3],
        'learning_rate': [0.001, 0.01],
        'dropout': [0.3, 0.5],
        'batch_size': [64],
        'num_epochs': [100],
        'num_folds': [10],
    }

    train_dataset = xy_split_dataset(data=np.load('train_dataset.npy'))

    best_params, best_loss = grid_search(
        train_dataset,
        parameter_grid,
        device
    )

    print(f'\nBest model loss: {best_loss}')
    print(f'Best parameters: {best_params}\n')

    # Train the 'final' model
    hidden_dim, num_layers, learning_rate, dropout, batch_size, num_epochs, _ = best_params
    best_model = LSTM(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    best_model.train_model(
        train_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device
    )

    filepath = "model_weights.pth"
    torch.save(best_model.state_dict(), filepath)


    val_dataset = xy_split_dataset(
        data=np.load('val_dataset.npy'),
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    validation_loss = best_model.validate(val_loader=val_loader, device=device)
    print(f'Final model validation loss" {validation_loss}')

    # Generate 2 tracks:
    for i in range(2):
        generated_track = best_model.generate(val_dataset[i][0].unsqueeze(0))
        generate_track(
            bars=generated_track,
            bpm=140,
            sounds_dir='sounds',
            filename=f'track_{i}',
        )


if __name__ == '__main__':
    main()