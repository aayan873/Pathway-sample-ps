import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from utils import add_features, create_sequences
from model import LSTMClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)

window_size = 30
batch_size = 128
epochs = 50
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using Device = {device}")

# Loading and preparing Data
train_df = pd.read_csv('Bitcoin_19_11_2013-18_01_2024_train_data.csv', sep=';')
train_df = add_features(train_df)

features = ['close','volume','range','returns','MA5','MA10','MA20','volatility10','RSI14']

scaler = StandardScaler()
scaler.fit(train_df[features])
X_train_scaled = scaler.transform(train_df[features])

y_train = (train_df['returns'] > 0).astype(int).values

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, window_size)

X_train_tensor = torch.tensor(X_train_seq, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_seq, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# LSTM Model
input_size = len(features)
model = LSTMClassifier(input_size)
model = model.to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    for Xb, yb in loop:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())
    print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.6f}")

# Saving the model
torch.save(model.state_dict(), "lstm_model.pth")

# Saving scalers so as to scale test data
joblib.dump(scaler, 'scaler.pkl')

print("Training is complete")
