"""
Data Fetcher Utility
====================
Thin wrapper around yfinance for backward-compatibility.
The core models in src/models/garch.py call yfinance directly;
this module is kept for any standalone scripts that need it.
"""

import numpy as np
import pandas as pd
import yfinance as yf


class SP500DataFetcher:
    """Fetch market data via yfinance (free, no API key required)."""

    def __init__(self, symbol: str = "SPY"):
        self.symbol = symbol

    def fetch_data(self, start: str = "2015-01-01", end: str = "2026-01-01") -> pd.DataFrame:
        """
        Download daily OHLCV data for self.symbol.

        Returns
        -------
        pd.DataFrame with columns [open, high, low, close, volume], sorted ascending.
        """
        raw = yf.download(self.symbol, start=start, end=end,
                          auto_adjust=True, progress=False)
        raw.columns = [c.lower() for c in raw.columns]
        raw = raw.sort_index()
        print(f"[OK] Fetched {len(raw)} days  ({raw.index[0].date()} → {raw.index[-1].date()})")
        return raw

    def calculate_returns(self, data: pd.DataFrame,
                          return_type: str = "log") -> pd.Series:
        """Compute log or simple returns from a OHLCV DataFrame."""
        if return_type == "log":
            ret = np.log(data["close"] / data["close"].shift(1))
        elif return_type == "simple":
            ret = data["close"].pct_change()
        else:
            raise ValueError("return_type must be 'log' or 'simple'")
        return ret.dropna()

    def get_recent_data(self, days: int = 252,
                        start: str = "2015-01-01",
                        end:   str = "2026-01-01"):
        """
        Convenience wrapper: fetch data and return (prices, log_returns_pct).
        """
        prices  = self.fetch_data(start=start, end=end)
        if days < len(prices):
            prices = prices.tail(days + 1)
        returns_pct = self.calculate_returns(prices) * 100
        return prices, returns_pct


if __name__ == "__main__":
    fetcher = SP500DataFetcher("SPY")
    prices, returns = fetcher.get_recent_data(days=252)
    print(prices.tail())
    print(returns.tail())
