import yfinance as yf
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ----------------------------
# Feature Engineering Functions
# ----------------------------
def add_technical_indicators(df):
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(window=14).mean() /
                                 df['Close'].diff().clip(upper=0).abs().rolling(window=14).mean()))
    df['Bollinger_Upper'] = df['Close'].rolling(14).mean() + 2 * df['Close'].rolling(14).std()
    df['Bollinger_Lower'] = df['Close'].rolling(14).mean() - 2 * df['Close'].rolling(14).std()
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()
    return df

def add_target_label(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df

def clean_features(df):
    df['Close_t-1'] = df['Close'].shift(1)
    df['RSI_14_t-1'] = df['RSI_14'].shift(1)
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    return df.dropna()

# ----------------------------
# Load & Process Data
# ----------------------------
df = yf.download("AAPL", start="2022-01-01", end="2023-12-31")
df.reset_index(inplace=True)
df = add_technical_indicators(df)
df = add_target_label(df)
df = clean_features(df)

X = df.drop(columns=["Date", "Target", "Signal_Line"], errors="ignore")
y = df["Target"]

# ----------------------------
# Time Series Cross-Validation
# ----------------------------
tscv = TimeSeriesSplit(n_splits=5)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 6],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.6, 0.8],
    "gamma": [0, 0.1]
}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="f1",
    cv=tscv,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X, y)

# ----------------------------
# Evaluate on Holdout (Last 20%)
# ----------------------------
best_xgb = grid_search.best_estimator_

print("✅ Best Parameters:", grid_search.best_params_)

split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

best_xgb.fit(X_train, y_train)
y_pred = best_xgb.predict(X_test)

print("\n📊 Final Evaluation on Holdout Set:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

# ----------------------------
# Visualize Buy/Sell Predictions
# ----------------------------
df_test = df.iloc[split_idx:].copy()
df_test['Predicted'] = y_pred

plt.figure(figsize=(14, 6))
plt.plot(df_test['Date'], df_test['Close'], label='Close Price')
plt.scatter(df_test[df_test['Predicted'] == 1]['Date'], df_test[df_test['Predicted'] == 1]['Close'],
            color='green', marker='^', label='Buy Signal')
plt.scatter(df_test[df_test['Predicted'] == 0]['Date'], df_test[df_test['Predicted'] == 0]['Close'],
            color='red', marker='v', label='Sell Signal')
plt.title("📈 XGBoost (TimeSeriesCV) Buy/Sell Predictions for AAPL")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
