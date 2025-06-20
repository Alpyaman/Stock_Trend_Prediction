import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch 2 years of historical stock data for Apple
ticker = "AAPL"
df = yf.download(ticker, start="2022-01-01", end="2023-12-31")


def add_technical_indicators(df):
    df = df.copy()

    # 1. SMA and EMA
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    df['EMA_14'] = df['Close'].ewm(span=14, adjust=False).mean()
    df['Close_t-1'] = df['Close'].shift(1)
    df['RSI_14_t-1'] = df['RSI_14'].shift(1)
    df['MACD_Hist'] = df['MACD'] - df['MACD_signal']
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month

    
    # 2. RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 3. MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 4. Bollinger Bands
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['Bollinger_Bands_Upper'] = sma20 + (std20 * 2)
    df['Bollinger_Bands_Lower'] = sma20 - (std20 * 2)

    return df

def add_target_label(df):
    df = df.copy()
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df.dropna(inplace=True)
    return df

def clean_features_for_modeling(df):
    df = df.copy()
    
    # Drop any row with NaNs (from indicators or labeling)
    df.dropna(inplace=True)

    # Optional: Reset index for clean training
    df.reset_index(drop=True, inplace=True)

    return df

def quick_eda(df):
    # Plot class distribution
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Target', palette='Set2', ax=ax)
    ax.set_title("Target Class Distribution (Up = 1, Down = 0)")
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.show()

    # Optional: Display indicator correlations
    numeric_cols = df.select_dtypes(include='number').drop(columns=['Target'], errors='ignore')
    corr = numeric_cols.corrwith(df['Target']).sort_values(ascending=False)

    print("\n🔍 Correlation of Features with Target:")
    print(corr.head(10).round(3))
    return df
