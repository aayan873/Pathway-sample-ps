import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1=64, hidden_size2=32, dropout=0.2):
        super().__init__()
        # First LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers=2, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers=1, batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size2, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # First LSTM
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        
        # Second LSTM
        out, _ = self.lstm2(out)
        
        # Take the last time step
        out = out[:, -1, :]
        
        # Output layer with sigmoid
        out = self.fc(out)
        out = self.sigmoid(out)
        return out
