"""
Machine learning models for financial prediction.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class MarketPredictor:
    """Machine learning models for market prediction."""
    
    def __init__(self):
        """Initialize the predictor."""
        # Try to import required libraries
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression, Ridge, Lasso
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            self.has_sklearn = True
        except ImportError:
            logger.warning("scikit-learn is not installed. Limited functionality available.")
            self.has_sklearn = False
    
    def _create_features(self, df: pd.DataFrame, target_col: str, n_lags: int = 5) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create lagged features from time series data.
        
        Args:
            df: DataFrame with time series data
            target_col: Target column name
            n_lags: Number of lag features to create
            
        Returns:
            Tuple of features DataFrame and target Series
        """
        # Create lagged features
        for i in range(1, n_lags + 1):
            df[f'{target_col}_lag_{i}'] = df[target_col].shift(i)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Extract features and target
        features = df.drop(columns=[target_col])
        target = df[target_col]
        
        return features, target
    
    def predict_price_movement(self, prices: pd.Series, additional_features: Optional[pd.DataFrame] = None, 
                               n_lags: int = 5, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Predict price movement using machine learning models.
        
        Args:
            prices: Series of prices
            additional_features: Optional DataFrame with additional features
            n_lags: Number of lag features to create
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with model performances and predictions
        """
        if not self.has_sklearn:
            logger.error("scikit-learn is required for price prediction")
            return {'error': 'scikit-learn not available'}
        
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from sklearn.model_selection import train_test_split
            
            # Create DataFrame from prices
            df = pd.DataFrame({'price': prices})
            
            # Add additional features if provided
            if additional_features is not None:
                # Ensure index alignment
                additional_features = additional_features.loc[df.index].dropna()
                df = pd.concat([df, additional_features], axis=1)
            
            # Create features
            features, target = self._create_features(df, 'price', n_lags)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, shuffle=False)
            
            # Define models
            models = {
                'linear_regression': Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', LinearRegression())
                ]),
                'random_forest': Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
                ]),
                'gradient_boosting': Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
                ])
            }
            
            # Train and evaluate models
            results = {}
            
            for name, model in models.items():
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Evaluate model
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                train_mae = mean_absolute_error(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                # Store results
                results[name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'model': model,
                    'test_predictions': test_pred,
                    'test_actual': y_test.values
                }
            
            # Find best model
            best_model = min(results.items(), key=lambda x: x[1]['test_rmse'])
            
            # Make future prediction
            future_prediction = None
            
            # Prepare features for the next prediction (latest data point)
            if len(features) > 0:
                latest_features = features.iloc[-1:].copy()
                
                # Update lag features for the next point
                for i in range(n_lags, 0, -1):
                    if i > 1:
                        latest_features[f'price_lag_{i}'] = latest_features[f'price_lag_{i-1}']
                    else:
                        latest_features[f'price_lag_{i}'] = target.iloc[-1]
                
                # Make prediction
                future_prediction = best_model[1]['model'].predict(latest_features)[0]
            
            return {
                'results': results,
                'best_model': best_model[0],
                'future_prediction': future_prediction
            }
            
        except Exception as e:
            logger.error(f"Error predicting price movement: {e}")
            return {'error': str(e)}
    
    def predict_volatility(self, returns: pd.Series, n_lags: int = 10, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Predict volatility using GARCH-like approach.
        
        Args:
            returns: Series of returns
            n_lags: Number of lag features to create
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with model performances and predictions
        """
        if not self.has_sklearn:
            logger.error("scikit-learn is required for volatility prediction")
            return {'error': 'scikit-learn not available'}
        
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            from sklearn.model_selection import train_test_split
            
            # Calculate realized volatility
            returns_array = returns.values
            volatility = pd.Series(index=returns.index[n_lags:], 
                                  data=[np.std(returns_array[i-n_lags:i]) for i in range(n_lags, len(returns))])
            
            # Create DataFrame from volatility
            df = pd.DataFrame({'volatility': volatility})
            
            # Create features
            features, target = self._create_features(df, 'volatility', n_lags)
            
            # Add squared returns as features (GARCH-like)
            for i in range(1, n_lags + 1):
                returns_shift = returns.shift(i).loc[features.index]
                features[f'returns_sq_{i}'] = returns_shift ** 2
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, shuffle=False)
            
            # Define models
            models = {
                'gradient_boosting': Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', GradientBoostingRegressor(n_estimators=100, random_state=42))
                ]),
                'random_forest': Pipeline([
                    ('scaler', StandardScaler()),
                    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
                ])
            }
            
            # Train and evaluate models
            results = {}
            
            for name, model in models.items():
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                # Evaluate model
                train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                train_mae = mean_absolute_error(y_train, train_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                
                # Store results
                results[name] = {
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'model': model,
                    'test_predictions': test_pred,
                    'test_actual': y_test.values
                }
            
            # Find best model
            best_model = min(results.items(), key=lambda x: x[1]['test_rmse'])
            
            # Make future prediction
            future_prediction = None
            
            # Prepare features for the next prediction (latest data point)
            if len(features) > 0:
                latest_features = features.iloc[-1:].copy()
                
                # Update lag features for the next point
                for i in range(n_lags, 0, -1):
                    if i > 1:
                        latest_features[f'volatility_lag_{i}'] = latest_features[f'volatility_lag_{i-1}']
                    else:
                        latest_features[f'volatility_lag_{i}'] = target.iloc[-1]
                
                # Make prediction
                future_prediction = best_model[1]['model'].predict(latest_features)[0]
            
            return {
                'results': results,
                'best_model': best_model[0],
                'future_prediction': future_prediction
            }
            
        except Exception as e:
            logger.error(f"Error predicting volatility: {e}")
            return {'error': str(e)}
