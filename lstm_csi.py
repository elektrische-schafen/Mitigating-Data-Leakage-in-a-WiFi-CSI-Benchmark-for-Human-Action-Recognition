import torch
import torch.nn as nn
from lstm_config import *
from load_csi_har import *
from utils import *
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)  # Get the class with the highest score
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test data: {accuracy:.2f}%')


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Accumulate the loss
            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")


class SequenceClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len, dropout_prob):
        super(SequenceClassifier, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # Dropout layer after LSTM
        self.dropout_lstm = nn.Dropout(p=dropout_prob)
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size*seq_len, 100)  # first dense layer
        self.fc2 = nn.Linear(100, output_size)            # second dense layer
        # Dropout layer after the first dense layer
        self.dropout_fc = nn.Dropout(p=dropout_prob)
        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_size)
        # Flatten the output for the dense layers
        lstm_out = lstm_out.reshape(lstm_out.size(0), -1)  # shape: (batch_size, hidden_size * seq_len)
        # Apply dropout after the LSTM
        lstm_out = self.dropout_lstm(lstm_out)
        # Pass through fully connected layers
        x = self.relu(self.fc1(lstm_out))  # First dense layer with ReLU
        x = self.dropout_fc(x)
        x = self.fc2(x)  # Second dense layer (output layer)
        return x


if __name__=="__main__":
    cfg = get_cfg()

    # Instantiate the model
    model = SequenceClassifier(input_size=cfg.input_size, hidden_size=cfg.hidden_size, output_size=cfg.output_size, seq_len=cfg.seq_len, dropout_prob=cfg.dropout_prob)

    # Print model summary
    print(model)

    dummy = False
    if dummy==True:
        # Example data (replace with your actual data)
        train_data = torch.randn(1000, cfg.seq_len, cfg.input_size)  # 1000 sequences, each of length 52 with 10 features
        train_labels = torch.randint(0, cfg.output_size, (1000,))  # Classification (0, 1, ..., ) for each sequence

        test_data = torch.randn(200, cfg.seq_len, cfg.input_size)  # 200 sequences in the test set
        test_labels = torch.randint(0, cfg.output_size, (200,))  # Classification for each sequence in the test set
    else:
        model_config = CSIModelConfig(win_len=300, step=50, thrshd=0.6)

        if cfg.wrt_humans==True:
            train_tuple, test_tuple = model_config.preprocessing(cfg.csi_har, save=True, wrt=True)

            x_lie_down, y_lie_down, x_fall, y_fall, x_bend, y_bend, x_run, y_run, x_sitdown, y_sitdown, x_standup, y_standup, x_walk, y_walk = train_tuple
            train_data, train_labels = train_valid_split((x_lie_down, x_fall, x_bend, x_run, x_sitdown, x_standup, x_walk), train_portion=1.0, seed=200)

            x_lie_down, y_lie_down, x_fall, y_fall, x_bend, y_bend, x_run, y_run, x_sitdown, y_sitdown, x_standup, y_standup, x_walk, y_walk = test_tuple
            test_data, test_labels = train_valid_split((x_lie_down, x_fall, x_bend, x_run, x_sitdown, x_standup, x_walk), train_portion=1.0, seed=200)

        else:
            numpy_tuple = model_config.preprocessing(cfg.csi_har, save=True, wrt=False)

            x_lie_down, y_lie_down, x_fall, y_fall, x_bend, y_bend, x_run, y_run, x_sitdown, y_sitdown, x_standup, y_standup, x_walk, y_walk = numpy_tuple
            train_data, train_labels, test_data, test_labels = train_valid_split((x_lie_down, x_fall, x_bend, x_run, x_sitdown, x_standup, x_walk), train_portion=cfg.train_portion, seed=200)

        train_data = torch.from_numpy(train_data).float()
        test_data  = torch.from_numpy(test_data).float()

        train_labels = torch.from_numpy( np.argmax(train_labels, axis=1) ).long()
        test_labels = torch.from_numpy(np.argmax(test_labels, axis=1)).long()

    # Create datasets
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Suitable for classification tasks
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer with learning rate 0.001

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=cfg.num_epochs)
    # Evaluate the model
    evaluate_model(model, test_loader)

    predicted_labels, true_labels = get_predictions_and_labels(model, test_loader)

    conf_mat = get_confusion_matrix(true_labels, predicted_labels)
    print(conf_mat)
    np.savetxt("confusion.txt", conf_mat, fmt="%.1f", delimiter=',')