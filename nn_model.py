import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
import torch.optim as optim
import itertools
from EarlyStopping import EarlyStopping
from sound_helpers import generate_track
import sounddevice as sd
import soundfile as sf


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
    """
     best_params, best_loss = grid_search(
        dataset,
        parameter_grid,
        device
    )

    print(best_loss)
    print(best_params)
    """


    ### Train the 'final' model
    # hidden_dim = 64, num_layers = 2, lr = 0.01, batch_size = 64, num_epochs = 100, num_folds = 10
    """
    model = LSTM(hidden_dim=64, num_layers=2)
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True
    )
    #model.train_model(train_loader,100, 0.01)
    filepath = "model_weights.pth"
    torch.save(model.state_dict(), filepath)
    
    """
    filepath = "model_weights.pth"
    model = LSTM(hidden_dim=64, num_layers=2)
    model.eval()
    model.load_state_dict(torch.load(filepath))

    example_input = torch.tensor([[1, 1, 1, 1]], dtype=torch.float32)  # Shape: (1, 4)
    example_input = example_input.unsqueeze(0)  # Shape: (1, 1, 4)

    output = (model.forward(example_input))
    predictions_thresh = threshold_predictions(output)
    print(predictions_thresh)

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=True
    )

    output = model.generate(test_loader)
    print(output[0][0])
    for i in range(10):
        for j in range(10):
            generate_track(
                bars=output[i][j],
                bpm=140,
                sounds_dir='sounds'
            )
            audio, samplerate = sf.read("track.wav")
            sd.play(audio, samplerate)
            sd.wait()

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

