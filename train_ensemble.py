"""
Train ensemble model with multiple classifiers.
 
This script trains an ensemble voting classifier using Logistic Regression,
Random Forest, and XGBoost on stock price data with technical indicators.
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
    logger.info("Starting ensemble model training")
 
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
 
    # Split data (no shuffle for time series)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y, shuffle=False)
 
    # Create individual models
    logger.info("Creating individual models")
    lr = trainer.create_logistic_regression()
    rf = trainer.create_random_forest()
    xgb = trainer.create_xgboost()
 
    # Train individual models
    logger.info("Training Logistic Regression (with scaling)")
    lr_trained = trainer.train_model(lr, X_train, y_train, scale_features=True)
 
    logger.info("Training Random Forest")
    rf_trained = trainer.train_model(rf, X_train, y_train, scale_features=False)
 
    logger.info("Training XGBoost")
    xgb_trained = trainer.train_model(xgb, X_train, y_train, scale_features=False)
 
    # Create and train ensemble
    logger.info("Creating ensemble model")
    ensemble = trainer.create_ensemble(lr_trained, rf_trained, xgb_trained)
    ensemble_trained = trainer.train_model(ensemble, X_train, y_train, scale_features=False)
 
    # Evaluate ensemble
    results = trainer.evaluate_model(
        ensemble_trained,
        X_test,
        y_test,
        scale_features=False,
        print_results=True
    )
 
    # Plot predictions
    df_test = df.iloc[-len(y_test):].reset_index(drop=True)
    trainer.plot_predictions(
        df_test,
        results['predictions'],
        title=f"Ensemble Model Buy/Sell Predictions for {config.ticker}",
        save_path="Ensemble_Model_Buy_Sell_Predictions_For_AAPL.png"
    )
 
    # Plot feature importance for Random Forest
    logger.info("Plotting Random Forest feature importance")
    trainer.plot_feature_importance(
        rf_trained,
        feature_cols,
        title="Random Forest Feature Importance",
        save_path="rf_feature_importances.png"
    )
 
    # Plot feature importance for XGBoost
    logger.info("Plotting XGBoost feature importance")
    trainer.plot_feature_importance(
        xgb_trained,
        feature_cols,
        title="XGBoost Feature Importance",
        save_path="XGBoost_Feature_Importance.png"
    )
 
    logger.info("Training and evaluation complete!")
 
 
if __name__ == "__main__":
    main()