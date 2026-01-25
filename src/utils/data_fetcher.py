"""
Data Fetcher Module for S&P 500 Time Series Data
Fetches historical data from Alpha Vantage API
"""

import pandas as pd
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta


class SP500DataFetcher:
    """Fetch and prepare S&P 500 data for GARCH modeling"""
    
    def __init__(self):
        """Initialize with Alpha Vantage API key from .env"""
        load_dotenv()
        self.api_key = os.getenv('ALPHAVANTAGE_API_KEY')
        
        if not self.api_key:
            raise ValueError("ALPHAVANTAGE_API_KEY not found in .env file!")
        
        self.ts = TimeSeries(key=self.api_key, output_format='pandas')
        self.symbol = 'SPY'  # S&P 500 ETF
    
    def fetch_data(self, outputsize='compact'):
        """
        Fetch S&P 500 historical data
        
        Parameters:
        -----------
        outputsize : str
            'compact' = last 100 data points (FREE)
            'full' = full historical data (PREMIUM ONLY)
        
        Returns:
        --------
        pd.DataFrame : Historical price data with columns [open, high, low, close, volume]
        """
        try:
            print(f"Fetching {outputsize} S&P 500 data from Alpha Vantage...")
            data, meta_data = self.ts.get_daily(symbol=self.symbol, outputsize=outputsize)
            
            # Rename columns for easier access
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Sort by date (oldest first)
            data = data.sort_index()
            
            print(f"[SUCCESS] Fetched {len(data)} days of data")
            print(f"  Date range: {data.index[0]} to {data.index[-1]}")
            
            return data
            
        except Exception as e:
            print(f"[ERROR] Error fetching data: {e}")
            raise
    
    def calculate_returns(self, data, return_type='log'):
        """
        Calculate returns from price data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data with 'close' column
        return_type : str
            'log' for log returns (recommended for GARCH)
            'simple' for simple returns
        
        Returns:
        --------
        pd.Series : Returns
        """
        if return_type == 'log':
            returns = np.log(data['close'] / data['close'].shift(1))
        elif return_type == 'simple':
            returns = data['close'].pct_change()
        else:
            raise ValueError("return_type must be 'log' or 'simple'")
        
        # Drop NaN (first value)
        returns = returns.dropna()
        
        print(f"[SUCCESS] Calculated {return_type} returns")
        print(f"  Mean return: {returns.mean():.6f}")
        print(f"  Std dev: {returns.std():.6f}")
        
        return returns
    
    def prepare_garch_data(self, start_date=None, end_date=None):
        """
        Fetch data and prepare returns for GARCH modeling
        
        Parameters:
        -----------
        start_date : str, optional
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date in 'YYYY-MM-DD' format
        
        Returns:
        --------
        tuple : (prices, returns)
            - prices: pd.DataFrame with OHLCV data
            - returns: pd.Series with log returns
        """
        # Fetch data (compact = last 100 days for free tier)
        prices = self.fetch_data(outputsize='compact')
        
        # Filter by date range if specified
        if start_date:
            prices = prices[prices.index >= start_date]
        if end_date:
            prices = prices[prices.index <= end_date]
        
        # Calculate returns
        returns = self.calculate_returns(prices, return_type='log')
        
        # Convert returns to percentage for easier interpretation
        returns_pct = returns * 100
        
        print(f"\n{'='*50}")
        print(f"Data prepared for GARCH modeling")
        print(f"{'='*50}")
        print(f"Total observations: {len(returns_pct)}")
        print(f"Date range: {returns_pct.index[0]} to {returns_pct.index[-1]}")
        print(f"Mean return: {returns_pct.mean():.4f}%")
        print(f"Volatility (std): {returns_pct.std():.4f}%")
        print(f"Min return: {returns_pct.min():.4f}%")
        print(f"Max return: {returns_pct.max():.4f}%")
        
        return prices, returns_pct
    
    def get_recent_data(self, days=100):
        """
        Get most recent N days of data
        
        Parameters:
        -----------
        days : int
            Number of recent days (max 100 for free tier)
        
        Returns:
        --------
        tuple : (prices, returns)
        """
        # Free tier only supports 'compact' (100 days)
        prices = self.fetch_data(outputsize='compact')
        
        # Limit to requested days if less than 100
        if days < len(prices):
            prices = prices.tail(days + 1)  # +1 for return calculation
        
        returns = self.calculate_returns(prices, return_type='log')
        returns_pct = returns * 100
        
        return prices, returns_pct


# Test function
if __name__ == "__main__":
    # Test the data fetcher
    fetcher = SP500DataFetcher()
    
    # Fetch recent data (free tier = 100 days max)
    prices, returns = fetcher.get_recent_data(days=100)
    
    print("\n" + "="*50)
    print("First 5 days of price data:")
    print(prices.head())
    
    print("\n" + "="*50)
    print("First 5 days of returns (%):")
    print(returns.head())
    
    print("\n" + "="*50)
    print("Last 5 days of returns (%):")
    print(returns.tail())