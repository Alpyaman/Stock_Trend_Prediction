import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ----------------------------
# Feature Engineering Helpers
# ----------------------------
def add_technical_indicators(df):
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    df['RSI_14'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).rolling(window=14).mean() /
                                 df['Close'].diff().clip(upper=0).abs().rolling(window=14).mean()))
    df['Bollinger_Bands_Upper'] = df['Close'].rolling(window=14).mean() + 2 * df['Close'].rolling(window=14).std()
    df['Bollinger_Bands_Lower'] = df['Close'].rolling(window=14).mean() - 2 * df['Close'].rolling(window=14).std()
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def add_target_label(df):
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    return df

def clean_features_for_modeling(df):
    return df.dropna()

# ----------------------------
# Load & Prepare Data
# ----------------------------
df = yf.download("AAPL", start="2022-01-01", end="2023-12-31")
df.reset_index(inplace=True)

df = add_technical_indicators(df)
df = add_target_label(df)

# Additional Features
df['Close_t-1'] = df['Close'].shift(1)
df['RSI_14_t-1'] = df['RSI_14'].shift(1)
df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month

df = clean_features_for_modeling(df)

# ----------------------------
# Train-Test Split
# ----------------------------
X = df.drop(columns=["Date", "Target", "Signal_Line"], errors='ignore')
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2, random_state=42)

# Scale for LR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# Model Training
# ----------------------------
lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

lr.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Ensemble
ensemble = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
    voting='soft'
)
ensemble.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
y_pred = ensemble.predict(X_test)
print("📊 Final Evaluation on Holdout Set:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

# ----------------------------
# Plotting Buy/Sell Signal
# ----------------------------
df_test = df.iloc[-len(y_test):].copy()
df_test['Predicted'] = y_pred

plt.figure(figsize=(14, 6))
plt.plot(df_test['Date'], df_test['Close'], label='Close Price')
plt.scatter(df_test[df_test['Predicted'] == 1]['Date'], df_test[df_test['Predicted'] == 1]['Close'],
            color='green', label='Buy Signal (↑)', marker='^')
plt.scatter(df_test[df_test['Predicted'] == 0]['Date'], df_test[df_test['Predicted'] == 0]['Close'],
            color='red', label='Sell Signal (↓)', marker='v')
plt.title("📈 Ensemble Model Buy/Sell Predictions for AAPL")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
