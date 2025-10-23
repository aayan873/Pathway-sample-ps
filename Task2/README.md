# Bitcoin Price Prediction with LSTM

Directional Bitcoin price prediction using a two-layer LSTM with technical indicators.

## Overview

This project predicts Bitcoin price movements (up/down) using historical price data and technical indicators. The model achieves **54.68% directional accuracy** on test data through a probability-weighted expected return methodology.

## Details

- **Two-Layer LSTM Architecture:** 64→32 hidden units with dropout regularization
- **Technical Indicators:** MA5, MA10, MA20, RSI14, 10-day volatility, price range, returns
- **Zero Data Leakage:** Strict temporal ordering prevents future information leakage
- **Binary Classification:** Predicts probability of upward/downward price movement

## Project Structure

```
task2/
├── Bitcoin_01_01_2025-21_10_2025_test_data.csv  # test data
├── Bitcoin_19_11_2013-31_12_2024_train_data.csv # train data
├── image.png  # Image of the prediction
├── lstm_model.pth  # Trained model weights (generated)
├── model.py  # LSTM architecture
├── requirements.txt  # Python dependencies
├── scaler.pkl  # StandardScaler (generated)
├── test.py  # evaluation and prediction
├── train.py  # training script
├── utils.py  # feature engineering utilities
└── README.md  # This file
```

## Dataset

The model uses Bitcoin historical price data:
- **Training:** November 19, 2013 – December 31, 2024
- **Testing:** January 1, 2025 – October 21, 2025
- **Raw Features:** close, volume
- **Engineered Features:** MA5, MA10, MA20, RSI14, returns, range, volatility10


Data files: 
- `Bitcoin_19_11_2013-31_12_2024_train_data.csv`
- `Bitcoin_01_01_2025-21_10_2025_test_data.csv`

### Install dependencies
```pip install -r requirements.txt```

## Usage

### 1. Training the Model

```python train.py```

**What happens:**
- Loads training data
- Engineers technical indicators (MA5, MA10, MA20, RSI14, volatility)
- Creates 30-day sequences
- Trains LSTM for 50 epochs with batch size 128
- Saves model to `lstm_model.pth` and scaler to `scaler.pkl`


### 2. Testing and Prediction

```python test.py```

**What happens:**
- Loads trained model and scaler
- Generates predictions on test data
- Calculates directional accuracy
- Displays predicted vs actual price plot
- Shows directional accuracy: **54.68%**

## Model Architecture
```
Input: [batch_size, 30 timesteps, 9 features]
↓
LSTM Layer 1: 64 units, 2 internal layers
↓
Dropout: 20%
↓
LSTM Layer 2: 32 units
↓
Dense Output: 1 unit (sigmoid activation)
↓
Output: Probability of upward movement​
```
## Feature Engineering

The model uses 9 engineered features:

1. **Close Price:** Current closing price
2. **Volume:** Trading volume
3. **Range:** Daily price range (current close - previous close)
4. **Returns:** Percentage change in price
5. **MA5:** 5-day moving average
6. **MA10:** 10-day moving average
7. **MA20:** 20-day moving average
8. **Volatility10:** 10-day rolling standard deviation of returns
9. **RSI14:** 14-day Relative Strength Index

## Prediction Methodology

The model uses **probability-weighted expected return** to convert binary classification outputs to price predictions:

1. Model outputs probability `p` of upward movement
2. Expected return calculated as: `E[R] = p × μ_pos + (1-p) × μ_neg`
   - μ_pos = mean positive return from training data
   - μ_neg = mean negative return from training data
3. Next price predicted as: `Predicted Close = Current Close × (1 + E[R])`

This methodology follows standard financial forecasting practices for probability-weighted returns.

## Results

- **Directional Accuracy:** 54.68%
- **Model captures major trend movements**
- **Maintains temporal integrity** (no future data leakage)


## References

1. [Advanced Stock Market Prediction Using LSTM Networks](https://arxiv.org/html/2505.05325v1)
2. [Moving Average Technical Indicator - Investopedia](https://www.investopedia.com/terms/m/movingaverage.asp)
3. [Expected Return Formula - Investopedia](https://www.investopedia.com/terms/e/expectedreturn.asp)
