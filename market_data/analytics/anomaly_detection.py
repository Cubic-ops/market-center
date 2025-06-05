"""
Market anomaly detection algorithms.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class MarketAnomalyDetector:
    """Detect anomalies in financial market data."""
    
    @staticmethod
    def detect_price_anomalies(prices: Union[List[float], np.ndarray, pd.Series], window: int = 20, threshold: float = 3.0) -> np.ndarray:
        """
        Detect price anomalies using Z-score method.
        
        Args:
            prices: Array-like of prices
            window: Rolling window size
            threshold: Z-score threshold for anomaly
            
        Returns:
            Boolean array indicating anomalies
        """
        prices_array = np.array(prices)
        anomalies = np.zeros(len(prices_array), dtype=bool)
        
        # Calculate rolling mean and standard deviation
        for i in range(window, len(prices_array)):
            window_prices = prices_array[i-window:i]
            mean = np.mean(window_prices)
            std = np.std(window_prices)
            
            if std > 0:
                z_score = (prices_array[i] - mean) / std
                if abs(z_score) > threshold:
                    anomalies[i] = True
        
        return anomalies
    
    @staticmethod
    def detect_volume_anomalies(volumes: Union[List[float], np.ndarray, pd.Series], window: int = 20, threshold: float = 3.0) -> np.ndarray:
        """
        Detect volume anomalies using Z-score method.
        
        Args:
            volumes: Array-like of trading volumes
            window: Rolling window size
            threshold: Z-score threshold for anomaly
            
        Returns:
            Boolean array indicating anomalies
        """
        volumes_array = np.array(volumes)
        anomalies = np.zeros(len(volumes_array), dtype=bool)
        
        # Calculate rolling mean and standard deviation
        for i in range(window, len(volumes_array)):
            window_volumes = volumes_array[i-window:i]
            mean = np.mean(window_volumes)
            std = np.std(window_volumes)
            
            if std > 0:
                z_score = (volumes_array[i] - mean) / std
                if abs(z_score) > threshold:
                    anomalies[i] = True
        
        return anomalies
    
    @staticmethod
    def detect_volatility_regime_changes(returns: Union[List[float], np.ndarray, pd.Series], window: int = 20, threshold: float = 2.0) -> Dict[str, np.ndarray]:
        """
        Detect changes in volatility regime.
        
        Args:
            returns: Array-like of returns
            window: Rolling window size
            threshold: Threshold for regime change
            
        Returns:
            Dictionary with high and low volatility regime indicators
        """
        returns_array = np.array(returns)
        
        # Calculate rolling volatility
        volatility = np.zeros(len(returns_array))
        for i in range(window, len(returns_array)):
            volatility[i] = np.std(returns_array[i-window:i])
        
        # Calculate volatility of volatility
        vol_of_vol = np.zeros(len(volatility))
        for i in range(window, len(volatility)):
            vol_of_vol[i] = np.std(volatility[i-window:i])
        
        # Detect regime changes
        high_vol_regime = np.zeros(len(returns_array), dtype=bool)
        low_vol_regime = np.zeros(len(returns_array), dtype=bool)
        
        for i in range(window*2, len(returns_array)):
            if vol_of_vol[i] > 0:
                norm_vol = (volatility[i] - np.mean(volatility[i-window:i])) / vol_of_vol[i]
                if norm_vol > threshold:
                    high_vol_regime[i] = True
                elif norm_vol < -threshold:
                    low_vol_regime[i] = True
        
        return {
            'high_volatility': high_vol_regime,
            'low_volatility': low_vol_regime
        }
    
    @staticmethod
    def detect_momentum_anomalies(prices: Union[List[float], np.ndarray, pd.Series], short_window: int = 5, long_window: int = 20, threshold: float = 2.0) -> Dict[str, np.ndarray]:
        """
        Detect anomalies in price momentum.
        
        Args:
            prices: Array-like of prices
            short_window: Short moving average window
            long_window: Long moving average window
            threshold: Threshold for anomaly
            
        Returns:
            Dictionary with positive and negative momentum anomalies
        """
        prices_array = np.array(prices)
        
        # Calculate short and long moving averages
        short_ma = np.zeros(len(prices_array))
        long_ma = np.zeros(len(prices_array))
        
        for i in range(long_window, len(prices_array)):
            short_ma[i] = np.mean(prices_array[i-short_window:i])
            long_ma[i] = np.mean(prices_array[i-long_window:i])
        
        # Calculate momentum
        momentum = short_ma - long_ma
        
        # Calculate momentum z-score
        momentum_zscore = np.zeros(len(prices_array))
        for i in range(long_window*2, len(prices_array)):
            window_momentum = momentum[i-long_window:i]
            mean = np.mean(window_momentum)
            std = np.std(window_momentum)
            
            if std > 0:
                momentum_zscore[i] = (momentum[i] - mean) / std
        
        # Detect anomalies
        positive_momentum = np.zeros(len(prices_array), dtype=bool)
        negative_momentum = np.zeros(len(prices_array), dtype=bool)
        
        positive_momentum[momentum_zscore > threshold] = True
        negative_momentum[momentum_zscore < -threshold] = True
        
        return {
            'positive_momentum': positive_momentum,
            'negative_momentum': negative_momentum
        }
    
    @staticmethod
    def detect_correlation_breakdown(returns1: Union[List[float], np.ndarray, pd.Series], 
                                      returns2: Union[List[float], np.ndarray, pd.Series], 
                                      window: int = 60, 
                                      threshold: float = 0.5) -> np.ndarray:
        """
        Detect breakdowns in correlation between two return series.
        
        Args:
            returns1: First return series
            returns2: Second return series
            window: Rolling window size
            threshold: Correlation change threshold
            
        Returns:
            Boolean array indicating correlation breakdowns
        """
        returns1_array = np.array(returns1)
        returns2_array = np.array(returns2)
        
        # Ensure same length
        min_length = min(len(returns1_array), len(returns2_array))
        returns1_array = returns1_array[:min_length]
        returns2_array = returns2_array[:min_length]
        
        # Calculate rolling correlation
        rolling_corr = np.zeros(min_length)
        
        for i in range(window, min_length):
            window_returns1 = returns1_array[i-window:i]
            window_returns2 = returns2_array[i-window:i]
            
            corr, _ = stats.pearsonr(window_returns1, window_returns2)
            rolling_corr[i] = corr
        
        # Detect correlation changes
        corr_change = np.zeros(min_length)
        breakdowns = np.zeros(min_length, dtype=bool)
        
        for i in range(window*2, min_length):
            corr_change[i] = abs(rolling_corr[i] - np.mean(rolling_corr[i-window:i]))
            if corr_change[i] > threshold:
                breakdowns[i] = True
        
        return breakdowns
    
    @staticmethod
    def detect_clustering_anomalies(data: Union[pd.DataFrame, np.ndarray], window: int = 50, n_clusters: int = 3) -> np.ndarray:
        """
        Detect anomalies using clustering.
        
        Args:
            data: Feature matrix (samples x features)
            window: Rolling window size
            n_clusters: Number of clusters
            
        Returns:
            Boolean array indicating anomalies
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            logger.error("scikit-learn is required for clustering anomaly detection")
            return np.zeros(len(data), dtype=bool)
        
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = np.array(data)
        
        # Ensure 2D
        if data_array.ndim == 1:
            data_array = data_array.reshape(-1, 1)
        
        # Initialize anomalies array
        anomalies = np.zeros(len(data_array), dtype=bool)
        
        # For each window, train a clustering model and identify anomalies
        for i in range(window, len(data_array)):
            # Extract window data
            window_data = data_array[i-window:i]
            
            # Standardize data
            scaler = StandardScaler()
            window_data_scaled = scaler.fit_transform(window_data)
            
            # Fit clustering model
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            kmeans.fit(window_data_scaled)
            
            # Get cluster centers
            centers = kmeans.cluster_centers_
            
            # Get cluster labels
            labels = kmeans.labels_
            
            # Calculate distance to cluster center for each point
            distances = np.zeros(len(window_data))
            for j in range(len(window_data)):
                distances[j] = np.linalg.norm(window_data_scaled[j] - centers[labels[j]])
            
            # Calculate distance threshold (mean + 2*std)
            threshold = np.mean(distances) + 2 * np.std(distances)
            
            # New point (current observation)
            new_point = data_array[i].reshape(1, -1)
            new_point_scaled = scaler.transform(new_point)
            
            # Predict cluster for new point
            new_label = kmeans.predict(new_point_scaled)[0]
            
            # Calculate distance to assigned cluster center
            new_distance = np.linalg.norm(new_point_scaled - centers[new_label])
            
            # Mark as anomaly if distance exceeds threshold
            if new_distance > threshold:
                anomalies[i] = True
        
        return anomalies
