import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def compute_RSI(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def add_features(df):

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)

    df['returns'] = df['close'].pct_change()
    df['MA5'] = df['close'].rolling(5).mean()
    df['MA10'] = df['close'].rolling(10).mean()
    df['MA20'] = df['close'].rolling(20).mean()
    df['range'] = df['close'] - df['close'].shift(1)
    df['volatility10'] = df['returns'].rolling(10).std()
    df['RSI14'] = compute_RSI(df['close'], 14)
    df.fillna(0, inplace=True)
    return df

def create_sequences(X, y, window):
    sequences, labels = [], []
    for i in range(len(X) - window):
        sequences.append(X[i:i+window])
        labels.append(y[i+window])
    return np.array(sequences), np.array(labels)
