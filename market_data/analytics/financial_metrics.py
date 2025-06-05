"""
Advanced financial metrics and indicators.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class FinancialMetrics:
    """Calculate advanced financial metrics and indicators."""
    
    @staticmethod
    def calculate_returns(prices: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
        """
        Calculate returns from price series.
        
        Args:
            prices: Array-like of prices
            
        Returns:
            Array of returns
        """
        prices_array = np.array(prices)
        returns = np.diff(prices_array) / prices_array[:-1]
        return returns
    
    @staticmethod
    def calculate_log_returns(prices: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
        """
        Calculate logarithmic returns from price series.
        
        Args:
            prices: Array-like of prices
            
        Returns:
            Array of log returns
        """
        prices_array = np.array(prices)
        log_returns = np.diff(np.log(prices_array))
        return log_returns
    
    @staticmethod
    def calculate_volatility(returns: Union[List[float], np.ndarray, pd.Series], window: int = 20, annualize: bool = True) -> np.ndarray:
        """
        Calculate rolling volatility from returns.
        
        Args:
            returns: Array-like of returns
            window: Rolling window size
            annualize: Whether to annualize volatility
            
        Returns:
            Array of volatility values
        """
        returns_array = np.array(returns)
        
        # Calculate rolling standard deviation
        rolling_vol = np.array([np.std(returns_array[max(0, i-window+1):i+1]) for i in range(len(returns_array))])
        
        # Annualize if requested (assuming 252 trading days)
        if annualize:
            rolling_vol = rolling_vol * np.sqrt(252)
        
        return rolling_vol
    
    @staticmethod
    def calculate_sharpe_ratio(returns: Union[List[float], np.ndarray, pd.Series], risk_free_rate: float = 0.0, window: int = 252) -> np.ndarray:
        """
        Calculate rolling Sharpe ratio from returns.
        
        Args:
            returns: Array-like of returns
            risk_free_rate: Risk-free rate (annualized)
            window: Rolling window size
            
        Returns:
            Array of Sharpe ratio values
        """
        returns_array = np.array(returns)
        
        # Calculate daily risk-free rate
        daily_rf = risk_free_rate / 252
        
        # Calculate excess returns
        excess_returns = returns_array - daily_rf
        
        # Calculate rolling mean and standard deviation
        rolling_sharpe = np.array([
            (np.mean(excess_returns[max(0, i-window+1):i+1]) / np.std(excess_returns[max(0, i-window+1):i+1]) * np.sqrt(252))
            if np.std(excess_returns[max(0, i-window+1):i+1]) > 0 else 0
            for i in range(len(returns_array))
        ])
        
        return rolling_sharpe
    
    @staticmethod
    def calculate_drawdowns(prices: Union[List[float], np.ndarray, pd.Series]) -> Dict[str, np.ndarray]:
        """
        Calculate drawdowns from price series.
        
        Args:
            prices: Array-like of prices
            
        Returns:
            Dictionary with drawdown metrics
        """
        prices_array = np.array(prices)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(prices_array)
        
        # Calculate drawdown in percentage terms
        drawdown = (prices_array - running_max) / running_max
        
        # Calculate drawdown duration
        drawdown_start = np.zeros_like(drawdown, dtype=bool)
        drawdown_end = np.zeros_like(drawdown, dtype=bool)
        
        # Find start of drawdowns
        drawdown_start[0] = drawdown[0] < 0
        drawdown_start[1:] = (drawdown[:-1] == 0) & (drawdown[1:] < 0)
        
        # Find end of drawdowns
        drawdown_end[:-1] = (drawdown[:-1] < 0) & (drawdown[1:] == 0)
        drawdown_end[-1] = drawdown[-1] < 0
        
        # Find current drawdown duration
        duration = np.zeros_like(drawdown)
        current_duration = 0
        
        for i in range(len(drawdown)):
            if drawdown[i] < 0:
                current_duration += 1
            else:
                current_duration = 0
            
            duration[i] = current_duration
        
        return {
            'drawdown': drawdown,
            'duration': duration,
            'drawdown_start': drawdown_start,
            'drawdown_end': drawdown_end
        }
    
    @staticmethod
    def calculate_rsi(prices: Union[List[float], np.ndarray, pd.Series], window: int = 14) -> np.ndarray:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Array-like of prices
            window: RSI window size
            
        Returns:
            Array of RSI values
        """
        prices_array = np.array(prices)
        
        # Calculate price changes
        deltas = np.diff(prices_array)
        
        # Create seed values
        seed = deltas[:window+1]
        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window
        
        # Initialize RSI values
        rsi = np.zeros_like(prices_array)
        rsi[:window] = 100. - 100. / (1. + up / down if down != 0 else 1)
        
        # Calculate RSI values
        for i in range(window, len(prices_array)):
            delta = deltas[i-1]
            
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (window - 1) + upval) / window
            down = (down * (window - 1) + downval) / window
            
            rsi[i] = 100. - 100. / (1. + up / down if down != 0 else 1)
        
        return rsi
    
    @staticmethod
    def calculate_macd(prices: Union[List[float], np.ndarray, pd.Series], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Dict[str, np.ndarray]:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        Args:
            prices: Array-like of prices
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line period
            
        Returns:
            Dictionary with MACD, signal, and histogram values
        """
        prices_array = np.array(prices)
        n = len(prices_array)
        
        # Adjust periods if we don't have enough data
        if n < slow_period:
            # Use smaller periods if we have limited data
            slow_period = max(5, n // 3)
            fast_period = max(3, slow_period // 2)
            signal_period = max(2, fast_period // 2)
        
        # Calculate EMAs
        ema_fast = np.zeros_like(prices_array)
        ema_slow = np.zeros_like(prices_array)
        
        # Calculate initial values
        min_period = min(n, slow_period)
        ema_fast[min_period-1] = np.mean(prices_array[:min_period])
        ema_slow[min_period-1] = np.mean(prices_array[:min_period])
        
        # Calculate factor
        factor_fast = 2.0 / (fast_period + 1)
        factor_slow = 2.0 / (slow_period + 1)
        
        # Calculate EMAs
        for i in range(min_period, n):
            ema_fast[i] = prices_array[i] * factor_fast + ema_fast[i-1] * (1 - factor_fast)
            ema_slow[i] = prices_array[i] * factor_slow + ema_slow[i-1] * (1 - factor_slow)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = np.zeros_like(macd_line)
        min_signal_idx = min(min_period + signal_period - 1, n - 1)
        signal_line[min_signal_idx] = np.mean(macd_line[min_period:min_signal_idx+1])
        
        factor_signal = 2.0 / (signal_period + 1)
        
        for i in range(min_signal_idx + 1, n):
            signal_line[i] = macd_line[i] * factor_signal + signal_line[i-1] * (1 - factor_signal)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }  
    @staticmethod
    def calculate_bollinger_bands(prices: Union[List[float], np.ndarray, pd.Series], window: int = 20, num_std: float = 2.0) -> Dict[str, np.ndarray]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Array-like of prices
            window: Moving average window size
            num_std: Number of standard deviations for bands
            
        Returns:
            Dictionary with middle band, upper band, and lower band values
        """
        prices_array = np.array(prices)
        
        # Calculate middle band (SMA)
        middle_band = np.zeros_like(prices_array)
        
        for i in range(len(prices_array)):
            if i < window - 1:
                middle_band[i] = np.mean(prices_array[:i+1])
            else:
                middle_band[i] = np.mean(prices_array[i-window+1:i+1])
        
        # Calculate standard deviation
        std = np.zeros_like(prices_array)
        
        for i in range(len(prices_array)):
            if i < window - 1:
                std[i] = np.std(prices_array[:i+1])
            else:
                std[i] = np.std(prices_array[i-window+1:i+1])
        
        # Calculate upper and lower bands
        upper_band = middle_band + num_std * std
        lower_band = middle_band - num_std * std
        
        return {
            'middle': middle_band,
            'upper': upper_band,
            'lower': lower_band
        }
