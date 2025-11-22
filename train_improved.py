"""
Train improved stock prediction model with advanced features.
 
This script incorporates all accuracy improvements:
- Advanced technical indicators (Stochastic, ADX, CCI, Williams %R, Ichimoku, MFI)
- Feature selection using mutual information
- Correlation-based feature filtering
- SMOTE for handling class imbalance
- Stacking classifier with meta-learner
- Walk-forward validation
- Expanded hyperparameter tuning
"""
 
import sys
from pathlib import Path
 
# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
 
from src.stock_prediction import Config, DataFetcher, FeatureEngineer, ModelTrainer
from src.stock_prediction.utils import setup_logging
 
 
def main():
    """Main execution function."""
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("=" * 80)
    logger.info("Starting IMPROVED stock prediction model training")
    logger.info("=" * 80)
 
    # Initialize configuration
    config = Config()
    logger.info(f"Training model for {config.ticker}")
 
    # ===========================
    # 1. FETCH AND PREPARE DATA
    # ===========================
    logger.info("\n[STEP 1] Fetching data...")
    data_fetcher = DataFetcher(config)
    df = data_fetcher.fetch_data()
    data_fetcher.validate_data(df)
 
    # ===========================
    # 2. FEATURE ENGINEERING
    # ===========================
    logger.info("\n[STEP 2] Engineering features (including advanced indicators)...")
    feature_engineer = FeatureEngineer(config)
    df = feature_engineer.prepare_features(df, include_target=True)
 
    logger.info(f"Total features after engineering: {df.shape[1]}")
    logger.info(f"Total samples: {df.shape[0]}")
 
    # ===========================
    # 3. FEATURE SELECTION
    # ===========================
    logger.info("\n[STEP 3] Performing feature selection...")
 
    # Prepare initial features and target
    feature_cols = feature_engineer.get_feature_list(df, exclude_target=True)
    X = df[feature_cols]
    y = df['Target']
 
    # Initialize model trainer
    trainer = ModelTrainer(config)
 
    # Remove highly correlated features
    X_reduced, remaining_features = trainer.remove_correlated_features(X, threshold=0.90)
    logger.info(f"Features after correlation filtering: {len(remaining_features)}")
 
    # Select top features using mutual information
    k_best = min(40, len(remaining_features))  # Select top 40 or all if less
    X_selected, selected_features = trainer.select_features_mutual_info(X_reduced, y, k=k_best)
    logger.info(f"Final selected features: {len(selected_features)}")
 
    # ===========================
    # 4. TRAIN-TEST SPLIT
    # ===========================
    logger.info("\n[STEP 4] Splitting data (time series split - no shuffle)...")
 
    # Time series split: last 20% for testing
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
 
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Testing samples: {len(X_test)}")
    logger.info(f"Class distribution in train: {y_train.value_counts().to_dict()}")
    logger.info(f"Class distribution in test: {y_test.value_counts().to_dict()}")
 
    # ===========================
    # 5. APPLY SMOTE
    # ===========================
    logger.info("\n[STEP 5] Applying SMOTE for class balancing...")
    X_train_balanced, y_train_balanced = trainer.apply_smote(X_train, y_train)
 
    # ===========================
    # 6. MODEL TRAINING
    # ===========================
    logger.info("\n[STEP 6] Training models...")
 
    # Train individual models
    logger.info("\n--- Training Random Forest ---")
    rf = trainer.create_random_forest()
    rf_trained = trainer.train_model(rf, X_train_balanced, y_train_balanced, scale_features=False)
 
    logger.info("\n--- Training XGBoost ---")
    xgb = trainer.create_xgboost()
    xgb_trained = trainer.train_model(xgb, X_train_balanced, y_train_balanced, scale_features=False)
 
    logger.info("\n--- Training Logistic Regression ---")
    lr = trainer.create_logistic_regression()
    lr_trained = trainer.train_model(lr, X_train_balanced, y_train_balanced, scale_features=True)
 
    # ===========================
    # 7. CREATE STACKING ENSEMBLE
    # ===========================
    logger.info("\n[STEP 7] Creating stacking classifier...")
 
    base_estimators = [
        ('rf', trainer.create_random_forest()),
        ('xgb', trainer.create_xgboost()),
    ]
 
    stacking_clf = trainer.create_stacking_classifier(
        base_estimators=base_estimators,
        final_estimator=None  # Uses LogisticRegression by default
    )
 
    logger.info("Training stacking classifier...")
    stacking_trained = trainer.train_model(
        stacking_clf,
        X_train_balanced,
        y_train_balanced,
        scale_features=False
    )
 
    # ===========================
    # 8. EVALUATION
    # ===========================
    logger.info("\n[STEP 8] Evaluating models on test set...")
 
    # Evaluate Random Forest
    logger.info("\n--- Random Forest Results ---")
    rf_results = trainer.evaluate_model(
        rf_trained,
        X_test,
        y_test,
        scale_features=False,
        print_results=True
    )
 
    # Evaluate XGBoost
    logger.info("\n--- XGBoost Results ---")
    xgb_results = trainer.evaluate_model(
        xgb_trained,
        X_test,
        y_test,
        scale_features=False,
        print_results=True
    )
 
    # Evaluate Logistic Regression
    logger.info("\n--- Logistic Regression Results ---")
    lr_results = trainer.evaluate_model(
        lr_trained,
        X_test,
        y_test,
        scale_features=True,
        print_results=True
    )
 
    # Evaluate Stacking Classifier
    logger.info("\n--- Stacking Classifier Results ---")
    stacking_results = trainer.evaluate_model(
        stacking_trained,
        X_test,
        y_test,
        scale_features=False,
        print_results=True
    )
 
    # ===========================
    # 9. WALK-FORWARD VALIDATION
    # ===========================
    logger.info("\n[STEP 9] Performing walk-forward validation on best model...")
 
    # Use stacking classifier for walk-forward validation
    wf_results = trainer.walk_forward_validation(
        X_selected,
        y,
        trainer.create_stacking_classifier(base_estimators=base_estimators),
        n_splits=5
    )
 
    logger.info("\nWalk-Forward Validation Results:")
    logger.info(f"Average F1 Score: {wf_results['avg_f1_score']:.4f}")
    logger.info(f"Average Accuracy: {wf_results['avg_accuracy']:.4f}")
 
    # ===========================
    # 10. VISUALIZATIONS
    # ===========================
    logger.info("\n[STEP 10] Generating visualizations...")
 
    # Plot predictions for stacking classifier
    df_test = df.iloc[split_idx:].reset_index(drop=True)
    trainer.plot_predictions(
        df_test,
        stacking_results['predictions'],
        title=f"Improved Stacking Model Buy/Sell Predictions for {config.ticker}",
        save_path="Improved_Stacking_Model_Predictions.png"
    )
 
    # Plot feature importance for Random Forest
    trainer.plot_feature_importance(
        rf_trained,
        selected_features,
        title="Random Forest Feature Importance (Improved Model)",
        save_path="Improved_RF_Feature_Importance.png"
    )
 
    # Plot feature importance for XGBoost
    trainer.plot_feature_importance(
        xgb_trained,
        selected_features,
        title="XGBoost Feature Importance (Improved Model)",
        save_path="Improved_XGBoost_Feature_Importance.png"
    )
 
    # ===========================
    # 11. SUMMARY
    # ===========================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total features engineered: {len(feature_cols)}")
    logger.info(f"Features after correlation filtering: {len(remaining_features)}")
    logger.info(f"Final features selected: {len(selected_features)}")
    logger.info(f"Training samples (after SMOTE): {len(X_train_balanced)}")
    logger.info(f"Testing samples: {len(X_test)}")
    logger.info("\nModel Performance on Test Set:")
    logger.info(f"  Random Forest F1: {rf_results['classification_report']['weighted avg']['f1-score']:.4f}")
    logger.info(f"  XGBoost F1: {xgb_results['classification_report']['weighted avg']['f1-score']:.4f}")
    logger.info(f"  Logistic Regression F1: {lr_results['classification_report']['weighted avg']['f1-score']:.4f}")
    logger.info(f"  Stacking Classifier F1: {stacking_results['classification_report']['weighted avg']['f1-score']:.4f}")
    logger.info("\nWalk-Forward Validation:")
    logger.info(f"  Average F1 Score: {wf_results['avg_f1_score']:.4f}")
    logger.info(f"  Average Accuracy: {wf_results['avg_accuracy']:.4f}")
    logger.info("=" * 80)
    logger.info("Training complete! All improvements applied successfully.")
    logger.info("=" * 80)
 
 
if __name__ == "__main__":
    main()