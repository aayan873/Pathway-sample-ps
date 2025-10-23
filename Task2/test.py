import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import add_features, create_sequences
from torch.utils.data import TensorDataset, DataLoader
from model import LSTMClassifier
import joblib

window_size = 30
batch_size = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# loadding scalers
loaded_scaler = joblib.load('scaler.pkl')

# loading test data
test_df = pd.read_csv('Bitcoin_01_01_2025-21_10_2025_test_data.csv', sep=';')
train_df = pd.read_csv('Bitcoin_19_11_2013-18_01_2024_train_data.csv', sep=';')

train_df = add_features(train_df)
test_df = add_features(test_df)

features = ['close','volume','range','returns','MA5','MA10','MA20','volatility10','RSI14']

X_test_scaled = loaded_scaler.transform(test_df[features])
y_test  = (test_df['returns']  > 0).astype(int).values

# creating sequences
X_test_seq,  y_test_seq  = create_sequences(X_test_scaled,  y_test,  window_size)

X_test_tensor  = torch.tensor(X_test_seq,  dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test_seq,  dtype=torch.float32).unsqueeze(1)

test_dataset  = TensorDataset(X_test_tensor,  y_test_tensor)

test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

#load Model
input_size = len(features)
model = LSTMClassifier(input_size)
model.load_state_dict(torch.load("lstm_model.pth"))
model = model.to(device)
model.eval()


# prediction
probs = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        X_batch = X_batch.to(device)
        p = model(X_batch)  # sigmoid probabilities
        probs.extend(p.cpu().numpy().flatten())

probs = np.array(probs)
train_next_returns = train_df['returns'].values

test_close = test_df['close'].values

pos_mask = train_next_returns > 0
neg_mask = ~pos_mask
mu_pos = train_next_returns[pos_mask].mean() if pos_mask.any() else  train_next_returns.mean()
mu_neg = train_next_returns[neg_mask].mean() if neg_mask.any() else  train_next_returns.mean()

exp_next_ret = probs * mu_pos + (1.0 - probs) * mu_neg  # shape (N_test_seq,)
idx_last_in_win = np.arange(window_size - 1, window_size - 1 + len(exp_next_ret))
idx_next = idx_last_in_win + 1

last_prices = test_close[idx_last_in_win]
actual_next_close = test_close[idx_next]
pred_next_close = last_prices * (1.0 + exp_next_ret)

# directional accuracy from probabilities (threshold 0.5) 
direction_pred = (probs > 0.5).astype(int)
direction_actual = y_test_seq[:len(direction_pred)].astype(int)
directional_accuracy = (direction_pred == direction_actual).mean() * 100.0
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

# plotting predicted vs actual next close
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))
plt.plot(actual_next_close, label='Actual Next Close', linewidth=2, alpha=0.8)
plt.plot(pred_next_close, label='Predicted Next Close', linewidth=1.8, alpha=0.8)
plt.title(f'Predicted vs Actual Next Close (Test)\nDirectional Accuracy: {directional_accuracy:.2f}%', fontsize=14)
plt.xlabel('Time Step (Aligned to Next Close)', fontsize=12)
plt.ylabel('Price', fontsize=12)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()