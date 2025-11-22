"""
Data exploration and visualization script.
 
This script fetches data, adds technical indicators, and generates
exploratory visualizations.
"""
 
import sys
from pathlib import Path
 
# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
 
import matplotlib.pyplot as plt
import seaborn as sns
from src.stock_prediction import Config, DataFetcher, FeatureEngineer
from src.stock_prediction.utils import setup_logging
 
 
def plot_class_distribution(df, save_path="Target Class Distribution.png"):
    """Plot target class distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x='Target', palette='Set2', ax=ax)
    ax.set_title("Target Class Distribution (Up = 1, Down = 0)")
    ax.set_xlabel("Target")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved class distribution to {save_path}")
    plt.close()
 
 
def plot_feature_correlations(df, top_n=10):
    """Plot correlation of features with target."""
    numeric_cols = df.select_dtypes(include='number').drop(columns=['Target'], errors='ignore')
    corr = numeric_cols.corrwith(df['Target']).sort_values(ascending=False)
 
    print("\n🔍 Top Features Correlated with Target:")
    print(corr.head(top_n).round(3))
    print("\n🔍 Bottom Features Correlated with Target:")
    print(corr.tail(top_n).round(3))
 
 
def plot_price_with_indicators(df, ticker="AAPL"):
    """Plot price with technical indicators."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
 
    # Price with Bollinger Bands
    axes[0].plot(df['Date'], df['Close'], label='Close Price', linewidth=2)
    if 'Bollinger_Bands_Upper' in df.columns:
        axes[0].plot(df['Date'], df['Bollinger_Bands_Upper'],
                    label='BB Upper', linestyle='--', alpha=0.7)
        axes[0].plot(df['Date'], df['Bollinger_Bands_Lower'],
                    label='BB Lower', linestyle='--', alpha=0.7)
        axes[0].fill_between(df['Date'], df['Bollinger_Bands_Lower'],
                            df['Bollinger_Bands_Upper'], alpha=0.1)
    axes[0].set_ylabel('Price ($)')
    axes[0].set_title(f'{ticker} Price with Bollinger Bands')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
 
    # RSI
    if 'RSI_14' in df.columns:
        axes[1].plot(df['Date'], df['RSI_14'], label='RSI', color='purple')
        axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        axes[1].set_ylabel('RSI')
        axes[1].set_title('Relative Strength Index (RSI)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
 
    # MACD
    if 'MACD' in df.columns and 'MACD_signal' in df.columns:
        axes[2].plot(df['Date'], df['MACD'], label='MACD', linewidth=2)
        axes[2].plot(df['Date'], df['MACD_signal'], label='Signal', linewidth=2)
        if 'MACD_Hist' in df.columns:
            axes[2].bar(df['Date'], df['MACD_Hist'], label='Histogram',
                       alpha=0.3, width=1)
        axes[2].set_ylabel('MACD')
        axes[2].set_xlabel('Date')
        axes[2].set_title('MACD Indicator')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig('Technical_Indicators_Overview.png', dpi=100, bbox_inches='tight')
    print("Saved technical indicators plot to Technical_Indicators_Overview.png")
    plt.close()
 
 
def main():
    """Main execution function."""
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("Starting data exploration")
 
    # Initialize configuration
    config = Config()
    logger.info(f"Exploring data for {config.ticker}")
 
    # Fetch data
    data_fetcher = DataFetcher(config)
    df = data_fetcher.fetch_data()
    data_fetcher.validate_data(df)
 
    # Get data summary
    summary = data_fetcher.get_data_summary(df)
    print("\n📊 Data Summary:")
    print(f"Rows: {summary['rows']}")
    print(f"Date Range: {summary['date_range']}")
    print(f"Price Range: ${summary['price_range']['min']:.2f} - ${summary['price_range']['max']:.2f}")
    print(f"Average Price: ${summary['price_range']['mean']:.2f}")
 
    # Feature engineering
    feature_engineer = FeatureEngineer(config)
    df = feature_engineer.prepare_features(df, include_target=True)
 
    logger.info(f"Data shape after feature engineering: {df.shape}")
 
    # Plot class distribution
    plot_class_distribution(df)
 
    # Plot feature correlations
    plot_feature_correlations(df, top_n=10) 
    # Plot price with indicators
    plot_price_with_indicators(df, ticker=config.ticker)
 
    logger.info("Data exploration complete!")
 
 
if __name__ == "__main__":
    main()