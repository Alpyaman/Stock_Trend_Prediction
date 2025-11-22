"""
Feature engineering module for technical indicators and transformations.
"""
 
import logging
from typing import Optional
import pandas as pd
import numpy as np
from .config import Config
  
logger = logging.getLogger(__name__)
 
 
class FeatureEngineer:
    """Class for creating technical indicators and features."""
 
    def __init__(self, config: Optional[Config] = None):
        """Initialize FeatureEngineer.
 
        Args:
            config: Configuration object. If None, uses default Config.
        """
        self.config = config or Config()
        logger.info("Initialized FeatureEngineer")
 
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple Moving Average (SMA) and Exponential Moving Average (EMA).
 
        Args:
            df: DataFrame with 'Close' column
 
        Returns:
            DataFrame with added SMA and EMA columns
        """
        df = df.copy()
        df['SMA_14'] = df['Close'].rolling(window=self.config.sma_window).mean()
        df['EMA_14'] = df['Close'].ewm(span=self.config.ema_window, adjust=False).mean()
 
        logger.debug(f"Added SMA_{self.config.sma_window} and EMA_{self.config.ema_window}")
        return df
 
    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Relative Strength Index (RSI).
 
        Args:
            df: DataFrame with 'Close' column
 
        Returns:
            DataFrame with added RSI column
        """
        df = df.copy()
        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
 
        avg_gain = gain.rolling(window=self.config.rsi_window).mean()
        avg_loss = loss.rolling(window=self.config.rsi_window).mean()
 
        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['RSI_14'] = 100 - (100 / (1 + rs))
 
        logger.debug(f"Added RSI_{self.config.rsi_window}")
        return df
 
    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Moving Average Convergence Divergence (MACD) and signal line.
 
        Args:
            df: DataFrame with 'Close' column
 
        Returns:
            DataFrame with added MACD, signal line, and histogram columns
        """
        df = df.copy()
        ema_fast = df['Close'].ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=self.config.macd_slow, adjust=False).mean()
 
        df['MACD'] = ema_fast - ema_slow
        df['MACD_signal'] = df['MACD'].ewm(span=self.config.macd_signal, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_signal']
 
        logger.debug(f"Added MACD ({self.config.macd_fast}/{self.config.macd_slow}/{self.config.macd_signal})")
        return df
 
    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands.
 
        Args:
            df: DataFrame with 'Close' column
 
        Returns:
            DataFrame with added upper and lower Bollinger Band columns
        """
        df = df.copy()
        sma = df['Close'].rolling(window=self.config.bollinger_window).mean()
        std = df['Close'].rolling(window=self.config.bollinger_window).std()
 
        df['Bollinger_Bands_Upper'] = sma + (std * self.config.bollinger_std)
        df['Bollinger_Bands_Lower'] = sma - (std * self.config.bollinger_std)
 
        logger.debug(f"Added Bollinger Bands (window={self.config.bollinger_window})")
        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features.
 
        Args:
            df: DataFrame with price and indicator columns
 
        Returns:
            DataFrame with added lag features
        """
        df = df.copy()
 
        # More comprehensive lag features
        df['Close_t-1'] = df['Close'].shift(1)
        df['Close_t-2'] = df['Close'].shift(2)
        df['Close_t-3'] = df['Close'].shift(3)
        df['Close_t-5'] = df['Close'].shift(5)
 
        if 'RSI_14' in df.columns:
            df['RSI_14_t-1'] = df['RSI_14'].shift(1)
 
        if 'Volume' in df.columns:
            df['Volume_t-1'] = df['Volume'].shift(1)
 
        logger.debug("Added lag features")
        return df

    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features.
 
        Args:
            df: DataFrame with 'Close' column
 
        Returns:
            DataFrame with added momentum features
        """
        df = df.copy()
 
        # Rate of change (ROC) over different periods
        df['ROC_5'] = df['Close'].pct_change(periods=5) * 100
        df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
        df['ROC_20'] = df['Close'].pct_change(periods=20) * 100
 
        # Price momentum
        df['Momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['Momentum_10'] = df['Close'] - df['Close'].shift(10)
 
        logger.debug("Added momentum features")
        return df
 
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features.
 
        Args:
            df: DataFrame with OHLC columns
 
        Returns:
            DataFrame with added volatility features
        """
        df = df.copy()
 
        # True Range and ATR
        df['TR'] = df[['High', 'Low', 'Close']].apply(
            lambda x: max(x['High'] - x['Low'],
                         abs(x['High'] - x['Close']),
                         abs(x['Low'] - x['Close'])),
            axis=1
        )
        df['ATR_14'] = df['TR'].rolling(window=14).mean()
 
        # Historical volatility
        df['Volatility_10'] = df['Close'].pct_change().rolling(window=10).std() * 100
        df['Volatility_20'] = df['Close'].pct_change().rolling(window=20).std() * 100
 
        # Intraday price range
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close'] * 100
 
        logger.debug("Added volatility features")
        return df
 
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features.
 
        Args:
            df: DataFrame with 'Volume' column
 
        Returns:
            DataFrame with added volume features
        """
        df = df.copy()
 
        # Volume moving averages
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
 
        # Volume ratio
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
 
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
 
        # Volume-Price Trend
        df['VPT'] = ((df['Close'].diff() / df['Close'].shift(1)) * df['Volume']).fillna(0).cumsum()
 
        logger.debug("Added volume features")
        return df
 
    def add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features.
 
        Args:
            df: DataFrame with 'Close' column
 
        Returns:
            DataFrame with added statistical features
        """
        df = df.copy()
 
        # Rolling mean and std
        df['Rolling_Mean_10'] = df['Close'].rolling(window=10).mean()
        df['Rolling_Std_10'] = df['Close'].rolling(window=10).std()
 
        # Distance from moving averages
        df['Distance_from_SMA'] = (df['Close'] - df['SMA_14']) / df['SMA_14'] * 100
        df['Distance_from_EMA'] = (df['Close'] - df['EMA_14']) / df['EMA_14'] * 100
 
        # Bollinger Band position
        bb_range = df['Bollinger_Bands_Upper'] - df['Bollinger_Bands_Lower']
        df['BB_Position'] = (df['Close'] - df['Bollinger_Bands_Lower']) / bb_range * 100
 
        # High-Low ratio
        df['HL_Ratio'] = df['High'] / df['Low']
 
        logger.debug("Added statistical features")
        return df
    
    def add_stochastic_oscillator(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Stochastic Oscillator (%K and %D).
 
        Args:
            df: DataFrame with High, Low, and Close columns
            window: Lookback window for stochastic calculation
 
        Returns:
            DataFrame with added Stochastic columns
        """
        df = df.copy()
 
        # Calculate %K
        low_min = df['Low'].rolling(window=window).min()
        high_max = df['High'].rolling(window=window).max()
        df['Stochastic_K'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
 
        # Calculate %D (3-period SMA of %K)
        df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
 
        logger.debug(f"Added Stochastic Oscillator (window={window})")
        return df
 
    def add_williams_r(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Williams %R indicator.
 
        Args:
            df: DataFrame with High, Low, and Close columns
            window: Lookback window
 
        Returns:
            DataFrame with added Williams %R column
        """
        df = df.copy()
 
        high_max = df['High'].rolling(window=window).max()
        low_min = df['Low'].rolling(window=window).min()
        df['Williams_R'] = -100 * (high_max - df['Close']) / (high_max - low_min)
 
        logger.debug(f"Added Williams %R (window={window})")
        return df
 
    def add_cci(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Add Commodity Channel Index (CCI).
 
        Args:
            df: DataFrame with High, Low, and Close columns
            window: Lookback window
 
        Returns:
            DataFrame with added CCI column
        """
        df = df.copy()
 
        # Typical Price
        tp = (df['High'] + df['Low'] + df['Close']) / 3
 
        # Simple Moving Average of TP
        sma_tp = tp.rolling(window=window).mean()
 
        # Mean Absolute Deviation
        mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
 
        # CCI calculation
        df['CCI'] = (tp - sma_tp) / (0.015 * mad)
 
        logger.debug(f"Added CCI (window={window})")
        return df
 
    def add_adx(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Average Directional Index (ADX) and directional indicators.
 
        Args:
            df: DataFrame with High, Low, and Close columns
            window: Lookback window
 
        Returns:
            DataFrame with added ADX, +DI, and -DI columns
        """
        df = df.copy()
 
        # Calculate True Range
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
 
        # Calculate directional movement
        high_diff = df['High'].diff()
        low_diff = -df['Low'].diff()
 
        plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
 
        # Smooth the values
        atr = tr.rolling(window=window).mean()
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / atr)
 
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        df['ADX'] = dx.rolling(window=window).mean()
        df['Plus_DI'] = plus_di
        df['Minus_DI'] = minus_di
 
        logger.debug(f"Added ADX (window={window})")
        return df
 
    def add_ichimoku_cloud(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Ichimoku Cloud indicators.
 
        Args:
            df: DataFrame with High, Low, and Close columns
 
        Returns:
            DataFrame with added Ichimoku indicators
        """
        df = df.copy()
 
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        period9_high = df['High'].rolling(window=9).max()
        period9_low = df['Low'].rolling(window=9).min()
        df['Tenkan_sen'] = (period9_high + period9_low) / 2
 
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        period26_high = df['High'].rolling(window=26).max()
        period26_low = df['Low'].rolling(window=26).min()
        df['Kijun_sen'] = (period26_high + period26_low) / 2
 
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
        df['Senkou_Span_A'] = ((df['Tenkan_sen'] + df['Kijun_sen']) / 2).shift(26)
 
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
        period52_high = df['High'].rolling(window=52).max()
        period52_low = df['Low'].rolling(window=52).min()
        df['Senkou_Span_B'] = ((period52_high + period52_low) / 2).shift(26)
 
        # Chikou Span (Lagging Span): Close shifted back 26 periods
        df['Chikou_Span'] = df['Close'].shift(-26)
 
        logger.debug("Added Ichimoku Cloud indicators")
        return df
 
    def add_money_flow_index(self, df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Add Money Flow Index (MFI).
 
        Args:
            df: DataFrame with High, Low, Close, and Volume columns
            window: Lookback window
 
        Returns:
            DataFrame with added MFI column
        """
        df = df.copy()
 
        # Typical Price
        tp = (df['High'] + df['Low'] + df['Close']) / 3
 
        # Money Flow
        mf = tp * df['Volume']
 
        # Positive and Negative Money Flow
        mf_pos = mf.where(tp > tp.shift(1), 0)
        mf_neg = mf.where(tp < tp.shift(1), 0)
 
        # Money Flow Ratio
        mf_pos_sum = mf_pos.rolling(window=window).sum()
        mf_neg_sum = mf_neg.rolling(window=window).sum()
        mf_ratio = mf_pos_sum / mf_neg_sum
 
        # Money Flow Index
        df['MFI'] = 100 - (100 / (1 + mf_ratio))
 
        logger.debug(f"Added MFI (window={window})")
        return df


    def add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features.
 
        Args:
            df: DataFrame with 'Date' column
 
        Returns:
            DataFrame with added calendar features
        """
        df = df.copy()
 
        if 'Date' not in df.columns:
            logger.warning("No 'Date' column found, skipping calendar features")
            return df
 
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
 
        logger.debug("Added calendar features")
        return df
 
    def add_all_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators and features.
 
        This is the main method that applies all feature engineering steps
        in the correct order.
 
        Args:
            df: DataFrame with OHLCV data
 
        Returns:
            DataFrame with all technical indicators added
        """
        logger.info("Adding all technical indicators")
 
        # Apply basic indicators first
        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
 
        # Add advanced indicators
        df = self.add_stochastic_oscillator(df)
        df = self.add_williams_r(df)
        df = self.add_cci(df)
        df = self.add_adx(df)
        df = self.add_ichimoku_cloud(df)
        df = self.add_money_flow_index(df)

        # Add derived features
        df = self.add_volatility_features(df)
        df = self.add_volume_features(df)
        df = self.add_statistical_features(df)
 
        logger.info(f"Added {len(df.columns)} features total")
        return df
 
    def add_target_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target label for next-day price movement.
 
        Target = 1 if next day's close > today's close, else 0
 
        Args:
            df: DataFrame with 'Close' column
 
        Returns:
            DataFrame with added 'Target' column
        """
        df = df.copy()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
 
        logger.debug("Added target label")
        return df
 
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with NaN values and reset index.
 
        Args:
            df: DataFrame to clean
 
        Returns:
            Cleaned DataFrame
        """
        initial_rows = len(df)
        df = df.dropna()
        df = df.reset_index(drop=True)
 
        rows_removed = initial_rows - len(df)
        logger.info(f"Removed {rows_removed} rows with NaN values. {len(df)} rows remaining.")
 
        return df
 
    def prepare_features(self, df: pd.DataFrame, include_target: bool = True) -> pd.DataFrame:
        """Complete feature preparation pipeline.
 
        Args:
            df: Raw DataFrame with OHLCV data
            include_target: Whether to add target label
 
        Returns:
            DataFrame ready for modeling
        """
        logger.info("Starting feature preparation pipeline")
 
        df = self.add_all_technical_indicators(df)
 
        if include_target:
            df = self.add_target_label(df)
 
        df = self.clean_data(df)
 
        logger.info("Feature preparation complete")
        return df
 
    def get_feature_list(self, df: pd.DataFrame, exclude_target: bool = True) -> list:
        """Get list of feature column names.
 
        Args:
            df: DataFrame with features
            exclude_target: Whether to exclude target column
 
        Returns:
            List of feature column names
        """
        exclude_cols = self.config.columns_to_drop.copy()
 
        if exclude_target and 'Target' in df.columns:
            exclude_cols.append('Target')
 
        # Handle both regular column names and MultiIndex tuples

        def should_exclude(col):
            # If column is a tuple (MultiIndex), check the first element
            col_name = col[0] if isinstance(col, tuple) else col
            return col_name in exclude_cols or col in exclude_cols
 
        features = [col for col in df.columns if not should_exclude(col)]

        logger.debug(f"Feature list: {features}")
        return features