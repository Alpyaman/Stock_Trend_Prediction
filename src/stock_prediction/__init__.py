"""
Stock Trend Prediction Package
 
A package for predicting stock price movements using technical indicators
and machine learning models.
"""
 
__version__ = "1.0.0"
__author__ = "Alpyaman"
 
from .config import Config
from .data_fetcher import DataFetcher
from .feature_engineering import FeatureEngineer
from .models import ModelTrainer
 
__all__ = [
    "Config",
    "DataFetcher",
    "FeatureEngineer",
    "ModelTrainer",
]