"""
Data fetching module for stock price data.
"""
 
import logging
from typing import Optional
import pandas as pd
import yfinance as yf
from .config import Config
 
 
logger = logging.getLogger(__name__)
 
 
class DataFetcher:
    """Class for fetching stock data from Yahoo Finance."""
 
    def __init__(self, config: Optional[Config] = None):
        """Initialize DataFetcher.
 
        Args:
            config: Configuration object. If None, uses default Config.
        """
        self.config = config or Config()
        logger.info(f"Initialized DataFetcher for ticker: {self.config.ticker}")
 
    def fetch_data(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance.
 
        Args:
            ticker: Stock ticker symbol. If None, uses config value.
            start_date: Start date in YYYY-MM-DD format. If None, uses config value.
            end_date: End date in YYYY-MM-DD format. If None, uses config value.
 
        Returns:
            DataFrame with stock price data
 
        Raises:
            ValueError: If data fetching fails or returns empty DataFrame
        """
        ticker = ticker or self.config.ticker
        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date
 
        logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
 
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
 
            if df.empty:
                raise ValueError(
                    f"No data returned for ticker {ticker} "
                    f"between {start_date} and {end_date}"
                )
 
            # Flatten MultiIndex columns if present (happens with single ticker)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Reset index to make Date a regular column
            df.reset_index(inplace=True)
 
            logger.info(f"Successfully fetched {len(df)} rows of data")
            logger.debug(f"Columns: {df.columns.tolist()}")
 
            return df
 
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            raise ValueError(f"Failed to fetch data: {str(e)}") from e
 
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate that the dataframe has required columns.
 
        Args:
            df: DataFrame to validate
 
        Returns:
            True if valid
 
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
 
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
 
        logger.debug("Data validation passed")
        return True
 
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Get summary statistics of the data.
 
        Args:
            df: DataFrame to summarize
 
        Returns:
            Dictionary with summary statistics
        """
        
        # Convert to scalar values to handle MultiIndex DataFrames
        close_min = df['Close'].min()
        close_max = df['Close'].max()
        close_mean = df['Close'].mean()
 
        # Handle both Series and scalar values
        if hasattr(close_min, 'item'):
            close_min = close_min.item()
        if hasattr(close_max, 'item'):
            close_max = close_max.item()
        if hasattr(close_mean, 'item'):
            close_mean = close_mean.item()
 
        summary = {
            'rows': len(df),
            'columns': len(df.columns),
            'date_range': f"{df['Date'].min()} to {df['Date'].max()}",
            'missing_values': df.isnull().sum().to_dict(),
            'price_range': {
                'min': float(close_min),
                'max': float(close_max),
                'mean': float(close_mean),
            },
        }

        logger.info(f"Data summary: {summary['rows']} rows, {summary['date_range']}")
        return summary