"""
Configuration management for stock prediction project.
"""
 
from dataclasses import dataclass
from typing import List
 
 
@dataclass
class Config:
    """Configuration class for stock prediction parameters."""
 
    # Data parameters
    ticker: str = "AAPL"
    start_date: str = "2021-01-01"
    end_date: str = "2024-12-31"
 
    # Technical indicator parameters
    sma_window: int = 14
    ema_window: int = 14
    rsi_window: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_window: int = 20
    bollinger_std: int = 2
 
    # Model parameters
    test_size: float = 0.2
    random_state: int = 42
    time_series_splits: int = 5
 
    # Class imbalance handling
    use_class_weights: bool = True
 
    # Logistic Regression
    lr_max_iter: int = 2000  # Increased to prevent convergence warnings
    lr_solver: str = 'lbfgs'
 
    # Random Forest
    rf_n_estimators: int = 200  # Increased for better performance
    rf_max_depth: int = 10  # Increased depth
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
 
    # XGBoost
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_gamma: float = 0
 
    # GridSearch parameters
    grid_search_cv_splits: int = 5
    grid_search_scoring: str = "f1"
    grid_search_n_jobs: int = -1
 
    # Output parameters
    figure_size: tuple = (14, 6)
    plot_dpi: int = 100
 
    # Feature columns to drop
    columns_to_drop: List[str] = None
 
    def __post_init__(self):
        """Initialize default values that need runtime evaluation."""
        if self.columns_to_drop is None:
            self.columns_to_drop = ["Date", "Target", "Signal_Line", "Open", "High", "Low", "Volume", "Adj Close"]

    @classmethod
    def from_dict(cls, config_dict: dict) -> "Config":
        """Create Config instance from dictionary.
 
        Args:
            config_dict: Dictionary containing configuration parameters
 
        Returns:
            Config instance
        """
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
 
    def to_dict(self) -> dict:
        """Convert Config to dictionary.
 
        Returns:
            Dictionary representation of config
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
 
    def get_xgb_params(self) -> dict:
        """Get XGBoost parameters as dictionary.
 
        Returns:
            Dictionary of XGBoost parameters
        """
        return {
            'n_estimators': self.xgb_n_estimators,
            'max_depth': self.xgb_max_depth,
            'learning_rate': self.xgb_learning_rate,
            'subsample': self.xgb_subsample,
            'colsample_bytree': self.xgb_colsample_bytree,
            'gamma': self.xgb_gamma,
            'random_state': self.random_state,
        }
 
    def get_rf_params(self) -> dict:
        """Get Random Forest parameters as dictionary.
 
        Returns:
            Dictionary of Random Forest parameters
        """
        params = {
            'n_estimators': self.rf_n_estimators,
            'max_depth': self.rf_max_depth,
            'min_samples_split': self.rf_min_samples_split,
            'min_samples_leaf': self.rf_min_samples_leaf,
            'random_state': self.random_state,
        }
        if self.use_class_weights:
            params['class_weight'] = 'balanced'
        return params