"""
Model training and evaluation utilities.
"""
 
import logging
from typing import Tuple, Optional, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
 
from .config import Config
 
 
logger = logging.getLogger(__name__)
 
 
class ModelTrainer:
    """Class for training and evaluating machine learning models."""
 
    def __init__(self, config: Optional[Config] = None):
        """Initialize ModelTrainer.
 
        Args:
            config: Configuration object. If None, uses default Config.
        """
        self.config = config or Config()
        self.scaler = StandardScaler()
        self.models = {}
        logger.info("Initialized ModelTrainer")
 
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: Optional[float] = None,
        shuffle: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and test sets.
 
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for test set
            shuffle: Whether to shuffle data before splitting
 
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or self.config.test_size
 
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            shuffle=shuffle,
            random_state=self.config.random_state
        )
 
        logger.info(f"Split data: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test
 
    def create_logistic_regression(self) -> LogisticRegression:
        """Create Logistic Regression model.
 
        Returns:
            Configured LogisticRegression model
        """
        params = {
            'max_iter': self.config.lr_max_iter,
            'solver': self.config.lr_solver,
            'random_state': self.config.random_state
        }
        if self.config.use_class_weights:
            params['class_weight'] = 'balanced'
 
        model = LogisticRegression(**params)
        logger.debug("Created Logistic Regression model with class weights")
        return model
 
    def create_random_forest(self) -> RandomForestClassifier:
        """Create Random Forest model.
 
        Returns:
            Configured RandomForestClassifier model
        """
        model = RandomForestClassifier(**self.config.get_rf_params())
        logger.debug("Created Random Forest model")
        return model
 
    def create_xgboost(self) -> XGBClassifier:
        """Create XGBoost model without deprecated parameters.
 
        Returns:
            Configured XGBClassifier model
        """
        params = self.config.get_xgb_params()
        # Remove deprecated parameters
        params.pop('use_label_encoder', None)
 
        model = XGBClassifier(**params)
        logger.debug("Created XGBoost model")
        return model
 
    def create_ensemble(
        self,
        lr: Optional[LogisticRegression] = None,
        rf: Optional[RandomForestClassifier] = None,
        xgb: Optional[XGBClassifier] = None
    ) -> VotingClassifier:
        """Create ensemble voting classifier.
 
        Args:
            lr: Trained Logistic Regression model
            rf: Trained Random Forest model
            xgb: Trained XGBoost model
 
        Returns:
            VotingClassifier ensemble model
        """
        lr = lr or self.create_logistic_regression()
        rf = rf or self.create_random_forest()
        xgb = xgb or self.create_xgboost()
 
        ensemble = VotingClassifier(
            estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
            voting='soft'
        )
 
        logger.debug("Created ensemble voting classifier")
        return ensemble
 
    def train_model(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        scale_features: bool = False
    ) -> Any:
        """Train a model.
 
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training labels
            scale_features: Whether to scale features
 
        Returns:
            Trained model
        """
        if scale_features:
            X_train = self.scaler.fit_transform(X_train)

        # Handle class imbalance for XGBoost

        if isinstance(model, XGBClassifier) and self.config.use_class_weights:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
            model.set_params(scale_pos_weight=scale_pos_weight)
            logger.info(f"Set XGBoost scale_pos_weight to {scale_pos_weight:.2f}")
 
        logger.info(f"Training {model.__class__.__name__}")
        model.fit(X_train, y_train)
        logger.info("Training complete")
 
        return model
 
    def evaluate_model(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        scale_features: bool = False,
        print_results: bool = True
    ) -> Dict[str, Any]:
        """Evaluate model performance.
 
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            scale_features: Whether to scale features
            print_results: Whether to print results
 
        Returns:
            Dictionary containing predictions and metrics
        """
        # Always ensure X_test is numeric only (drop datetime/object columns)
        if isinstance(X_test, pd.DataFrame):
            numeric_cols = X_test.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < len(X_test.columns):
                dropped_cols = set(X_test.columns) - set(numeric_cols)
                logger.warning(f"Dropping non-numeric columns before evaluation: {dropped_cols}")
                X_test = X_test[numeric_cols]
 
        if scale_features:
            X_test = self.scaler.transform(X_test)
 
        logger.info(f"Evaluating {model.__class__.__name__}")
        y_pred = model.predict(X_test)
 
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)
 
        if print_results:
            print(f"\n📊 Evaluation Results for {model.__class__.__name__}:")
            print("Confusion Matrix:")
            print(cm)
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, digits=3))
 
        return {
            'predictions': y_pred,
            'confusion_matrix': cm,
            'classification_report': cr
        }
 
    def grid_search_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict] = None
    ) -> GridSearchCV:
        """Perform grid search for XGBoost hyperparameters.
 
        Args:
            X: Feature matrix
            y: Target vector
            param_grid: Parameter grid for search
 
        Returns:
            Fitted GridSearchCV object
        """
        if param_grid is None:
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 6],
                "learning_rate": [0.05, 0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.6, 0.8],
                "gamma": [0, 0.1]
            }
 
        tscv = TimeSeriesSplit(n_splits=self.config.grid_search_cv_splits)
        xgb = self.create_xgboost()
 
        logger.info("Starting GridSearchCV for XGBoost")
        grid_search = GridSearchCV(
            estimator=xgb,
            param_grid=param_grid,
            scoring=self.config.grid_search_scoring,
            cv=tscv,
            verbose=1,
            n_jobs=self.config.grid_search_n_jobs
        )
 
        grid_search.fit(X, y)
 
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
 
        return grid_search
 
    def plot_predictions(
        self,
        df: pd.DataFrame,
        predictions: np.ndarray,
        title: str = "Buy/Sell Predictions",
        save_path: Optional[str] = None
    ) -> None:
        """Plot buy/sell predictions on price chart.
 
        Args:
            df: DataFrame with Date and Close columns
            predictions: Array of predictions (1=buy, 0=sell)
            title: Plot title
            save_path: Path to save figure (if None, displays plot)
        """
        df = df.copy()
        df['Predicted'] = predictions
 
        plt.figure(figsize=self.config.figure_size, dpi=self.config.plot_dpi)
        plt.plot(df['Date'], df['Close'], label='Close Price', linewidth=2)
 
        # Buy signals
        buy_signals = df[df['Predicted'] == 1]
        plt.scatter(
            buy_signals['Date'],
            buy_signals['Close'],
            color='green',
            label='Buy Signal (↑)',
            marker='^',
            s=100,
            alpha=0.7
        )
 
        # Sell signals
        sell_signals = df[df['Predicted'] == 0]
        plt.scatter(
            sell_signals['Date'],
            sell_signals['Close'],
            color='red',
            label='Sell Signal (↓)',
            marker='v',
            s=100,
            alpha=0.7
        )
 
        plt.title(f"📈 {title}")
        plt.xlabel("Date")
        plt.ylabel("Close Price ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
 
        if save_path:
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            logger.info(f"Saved plot to {save_path}")
        else:
            plt.show()
 
    def plot_feature_importance(
        self,
        model: Any,
        feature_names: list,
        title: str = "Feature Importance",
        save_path: Optional[str] = None,
        top_n: int = 20
    ) -> None:
        """Plot feature importance for tree-based models.
 
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            title: Plot title
            save_path: Path to save figure
            top_n: Number of top features to display
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return
 
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
 
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.tight_layout()
 
        if save_path:
            plt.savefig(save_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        else:
            plt.show()