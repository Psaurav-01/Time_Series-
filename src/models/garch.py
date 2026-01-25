"""
GARCH Model Module for S&P 500 Volatility Forecasting
Implements GARCH(1,1) model for conditional volatility estimation
"""

import numpy as np
import pandas as pd
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')


class SP500GARCHModel:
    """GARCH(1,1) model for S&P 500 volatility forecasting"""
    
    def __init__(self, returns):
        """
        Initialize GARCH model with return data
        
        Parameters:
        -----------
        returns : pd.Series
            Time series of returns (in percentage)
        """
        self.returns = returns
        self.model = None
        self.fitted_model = None
        self.forecast = None
        
    def fit(self, p=1, q=1, mean='Zero', vol='GARCH', dist='normal'):
        """
        Fit GARCH model to returns
        
        Parameters:
        -----------
        p : int
            GARCH lag order (default 1)
        q : int
            ARCH lag order (default 1)
        mean : str
            Mean model specification ('Zero', 'Constant', 'AR')
        vol : str
            Volatility model ('GARCH', 'EGARCH', 'FIGARCH')
        dist : str
            Error distribution ('normal', 't', 'skewt')
        
        Returns:
        --------
        self : fitted model
        """
        print(f"\n{'='*60}")
        print(f"Fitting GARCH({p},{q}) Model to S&P 500 Returns")
        print(f"{'='*60}")
        print(f"Sample size: {len(self.returns)} observations")
        print(f"Mean model: {mean}")
        print(f"Volatility model: {vol}")
        print(f"Distribution: {dist}")
        
        # Create GARCH model
        self.model = arch_model(
            self.returns,
            mean=mean,
            vol=vol,
            p=p,
            q=q,
            dist=dist
        )
        
        # Fit the model
        print("\nFitting model...")
        self.fitted_model = self.model.fit(disp='off')
        
        print("[SUCCESS] Model fitted successfully!")
        
        return self
    
    def summary(self):
        """Display model summary statistics"""
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet! Call fit() first.")
        
        print("\n" + "="*60)
        print("GARCH Model Summary")
        print("="*60)
        print(self.fitted_model.summary())
        
        return self.fitted_model.summary()
    
    def get_parameters(self):
        """
        Get fitted GARCH parameters
        
        Returns:
        --------
        dict : Model parameters
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet! Call fit() first.")
        
        params = self.fitted_model.params
        
        param_dict = {
            'omega': params.get('omega', None),
            'alpha[1]': params.get('alpha[1]', None),
            'beta[1]': params.get('beta[1]', None)
        }
        
        print("\n" + "="*60)
        print("GARCH Model Parameters")
        print("="*60)
        for key, value in param_dict.items():
            if value is not None:
                print(f"{key:15s} = {value:.6f}")
        
        # Persistence
        if param_dict['alpha[1]'] and param_dict['beta[1]']:
            persistence = param_dict['alpha[1]'] + param_dict['beta[1]']
            print(f"\nPersistence (α + β) = {persistence:.6f}")
            
            if persistence < 1:
                print("[INFO] Model is stationary (persistence < 1)")
            else:
                print("[WARNING] Model may not be stationary (persistence >= 1)")
        
        return param_dict
    
    def get_conditional_volatility(self):
        """
        Extract conditional volatility from fitted model
        
        Returns:
        --------
        pd.Series : Conditional volatility (annualized %)
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet! Call fit() first.")
        
        # Get conditional volatility
        cond_vol = self.fitted_model.conditional_volatility
        
        print(f"\n[SUCCESS] Extracted conditional volatility")
        print(f"  Mean volatility: {cond_vol.mean():.4f}%")
        print(f"  Min volatility: {cond_vol.min():.4f}%")
        print(f"  Max volatility: {cond_vol.max():.4f}%")
        print(f"  Current volatility: {cond_vol.iloc[-1]:.4f}%")
        
        return cond_vol
    
    def forecast_volatility(self, horizon=5):
        """
        Forecast future volatility
        
        Parameters:
        -----------
        horizon : int
            Number of periods ahead to forecast
        
        Returns:
        --------
        pd.DataFrame : Volatility forecasts
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet! Call fit() first.")
        
        print(f"\n{'='*60}")
        print(f"Forecasting Volatility for Next {horizon} Days")
        print(f"{'='*60}")
        
        # Generate forecast
        self.forecast = self.fitted_model.forecast(horizon=horizon)
        
        # Extract variance forecast and convert to volatility (std dev)
        variance_forecast = self.forecast.variance.values[-1, :]
        volatility_forecast = np.sqrt(variance_forecast)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Day': range(1, horizon + 1),
            'Forecasted_Volatility_%': volatility_forecast
        })
        
        print("\nVolatility Forecasts:")
        print(forecast_df.to_string(index=False))
        
        print(f"\nAverage forecasted volatility: {volatility_forecast.mean():.4f}%")
        
        return forecast_df
    
    def calculate_var(self, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR)
        
        Parameters:
        -----------
        confidence_level : float
            Confidence level (default 0.95 for 95% VaR)
        
        Returns:
        --------
        dict : VaR metrics
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet! Call fit() first.")
        
        # Get current conditional volatility
        current_vol = self.fitted_model.conditional_volatility.iloc[-1]
        
        # Calculate VaR (assuming normal distribution)
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence_level)
        var = z_score * current_vol
        
        var_metrics = {
            'confidence_level': confidence_level,
            'current_volatility_%': current_vol,
            'VaR_%': var,
            'interpretation': f"{confidence_level*100:.0f}% confidence that loss will not exceed {abs(var):.2f}%"
        }
        
        print(f"\n{'='*60}")
        print(f"Value at Risk (VaR) - {confidence_level*100:.0f}% Confidence Level")
        print(f"{'='*60}")
        print(f"Current Volatility: {current_vol:.4f}%")
        print(f"VaR: {var:.4f}%")
        print(f"\n{var_metrics['interpretation']}")
        
        return var_metrics
    
    def model_diagnostics(self):
        """
        Perform model diagnostics
        
        Returns:
        --------
        dict : Diagnostic statistics
        """
        if self.fitted_model is None:
            raise ValueError("Model not fitted yet! Call fit() first.")
        
        # Get standardized residuals
        std_resid = self.fitted_model.std_resid
        
        # Calculate diagnostics
        diagnostics = {
            'AIC': self.fitted_model.aic,
            'BIC': self.fitted_model.bic,
            'Log-Likelihood': self.fitted_model.loglikelihood,
            'Num_Observations': self.fitted_model.nobs,
            'Mean_Std_Residuals': std_resid.mean(),
            'Std_Std_Residuals': std_resid.std()
        }
        
        print(f"\n{'='*60}")
        print(f"Model Diagnostics")
        print(f"{'='*60}")
        for key, value in diagnostics.items():
            print(f"{key:25s} = {value:.4f}")
        
        return diagnostics


# Test function
if __name__ == "__main__":
    # Import data fetcher
    import sys
    sys.path.append('.')
    from src.utils.data_fetcher import SP500DataFetcher
    
    print("Testing GARCH Model with S&P 500 Data")
    print("="*60)
    
    # Fetch data
    fetcher = SP500DataFetcher()
    prices, returns = fetcher.get_recent_data(days=100)
    
    # Initialize and fit GARCH model
    garch = SP500GARCHModel(returns)
    garch.fit(p=1, q=1, mean='Zero', vol='GARCH', dist='normal')
    
    # Display results
    garch.get_parameters()
    
    # Get conditional volatility
    cond_vol = garch.get_conditional_volatility()
    
    # Forecast volatility
    forecast = garch.forecast_volatility(horizon=5)
    
    # Calculate VaR
    var = garch.calculate_var(confidence_level=0.95)
    
    # Model diagnostics
    diagnostics = garch.model_diagnostics()
    
    print("\n" + "="*60)
    print("GARCH Model Testing Complete!")
    print("="*60)