"""
Train XGBoost with Time Series Cross-Validation and GridSearch.
 
This script performs hyperparameter tuning using GridSearchCV with
TimeSeriesSplit for proper time series validation.
"""
 
import sys
from pathlib import Path
 
# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
 
from src.stock_prediction import Config, DataFetcher, FeatureEngineer, ModelTrainer
from src.stock_prediction.utils import setup_logging
 
 
def main():
    """Main execution function."""
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("Starting XGBoost training with Time Series CV")
 
    # Initialize configuration
    config = Config()
    logger.info(f"Training model for {config.ticker}")
 
    # Fetch data
    data_fetcher = DataFetcher(config)
    df = data_fetcher.fetch_data()
    data_fetcher.validate_data(df)
 
    # Feature engineering
    feature_engineer = FeatureEngineer(config)
    df = feature_engineer.prepare_features(df, include_target=True)
 
    logger.info(f"Data shape after feature engineering: {df.shape}")
 
    # Prepare features and target
    feature_cols = feature_engineer.get_feature_list(df, exclude_target=True)
    X = df[feature_cols]
    y = df['Target']
 
    # Initialize model trainer
    trainer = ModelTrainer(config)
 
    # Perform grid search with time series cross-validation
    logger.info("Performing GridSearchCV with TimeSeriesSplit")
    grid_search = trainer.grid_search_xgboost(X, y)
 
    print("\n✅ Best Parameters:", grid_search.best_params_)
    print(f"✅ Best CV Score: {grid_search.best_score_:.4f}")
 
    # Evaluate on holdout set (last 20%)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
 
    best_model = grid_search.best_estimator_
    logger.info("Training best model on full training set")
    best_model.fit(X_train, y_train)
 
    # Evaluate
    results = trainer.evaluate_model(
        best_model,
        X_test,
        y_test,
        scale_features=False,
        print_results=True
    )
 
    # Plot predictions
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    trainer.plot_predictions(
        df_test,
        results['predictions'],
        title=f"XGBoost (TimeSeriesCV) Buy/Sell Predictions for {config.ticker}",
        save_path="XGBoost_TimeSeriesCV_Buy_Sell_Predictions_For_AAPL.png"
    )
 
    # Plot feature importance
    logger.info("Plotting feature importance")
    trainer.plot_feature_importance(
        best_model,
        feature_cols,
        title="XGBoost Feature Importance (Best Model)",
        save_path="XGBoost_Feature_Importance_TimeSeriesCV.png"
    )
 
    logger.info("Training and evaluation complete!")
 
 
if __name__ == "__main__":
    main()