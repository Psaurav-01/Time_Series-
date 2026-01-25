"""
S&P 500 GARCH Volatility Dashboard
Interactive web dashboard for visualizing GARCH model results
"""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from src.utils.data_fetcher import SP500DataFetcher
from src.models.garch import SP500GARCHModel


# Initialize Dash app
app = dash.Dash(__name__)
app.title = "S&P 500 GARCH Dashboard"

# Fetch data and fit model
print("Initializing dashboard...")
print("Fetching S&P 500 data...")
fetcher = SP500DataFetcher()
prices, returns = fetcher.get_recent_data(days=100)

print("Fitting GARCH model...")
garch_model = SP500GARCHModel(returns)
garch_model.fit(p=1, q=1, mean='Zero', vol='GARCH', dist='normal')

# Get model results
cond_vol = garch_model.get_conditional_volatility()
forecast_df = garch_model.forecast_volatility(horizon=10)
var_95 = garch_model.calculate_var(confidence_level=0.95)
params = garch_model.get_parameters()

# Calculate annualized volatility
annual_vol = cond_vol.iloc[-1] * np.sqrt(252)

print("Dashboard ready!")

# Dashboard Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("S&P 500 GARCH Volatility Dashboard", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 10}),
        html.P("Real-time volatility forecasting using GARCH(1,1) model",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 16}),
    ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),
    
    # Key Metrics Row
    html.Div([
        # Current Volatility
        html.Div([
            html.H4("Current Volatility", style={'color': '#34495e', 'marginBottom': 5}),
            html.H2(f"{cond_vol.iloc[-1]:.2f}%", style={'color': '#e74c3c', 'margin': 0}),
            html.P(f"Annualized: {annual_vol:.2f}%", style={'color': '#7f8c8d', 'fontSize': 14}),
        ], style={'flex': 1, 'padding': '20px', 'backgroundColor': '#fff', 'margin': '10px', 
                  'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'textAlign': 'center'}),
        
        # VaR 95%
        html.Div([
            html.H4("Value at Risk (95%)", style={'color': '#34495e', 'marginBottom': 5}),
            html.H2(f"{abs(var_95['VaR_%']):.2f}%", style={'color': '#e67e22', 'margin': 0}),
            html.P("Maximum 1-day loss", style={'color': '#7f8c8d', 'fontSize': 14}),
        ], style={'flex': 1, 'padding': '20px', 'backgroundColor': '#fff', 'margin': '10px',
                  'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'textAlign': 'center'}),
        
        # Persistence
        html.Div([
            html.H4("Persistence (α + β)", style={'color': '#34495e', 'marginBottom': 5}),
            html.H2(f"{params['alpha[1]'] + params['beta[1]']:.4f}", 
                   style={'color': '#3498db', 'margin': 0}),
            html.P("Volatility decay rate", style={'color': '#7f8c8d', 'fontSize': 14}),
        ], style={'flex': 1, 'padding': '20px', 'backgroundColor': '#fff', 'margin': '10px',
                  'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'textAlign': 'center'}),
        
        # Sample Size
        html.Div([
            html.H4("Data Points", style={'color': '#34495e', 'marginBottom': 5}),
            html.H2(f"{len(returns)}", style={'color': '#27ae60', 'margin': 0}),
            html.P("Trading days analyzed", style={'color': '#7f8c8d', 'fontSize': 14}),
        ], style={'flex': 1, 'padding': '20px', 'backgroundColor': '#fff', 'margin': '10px',
                  'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'textAlign': 'center'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-around'}),
    
    # Charts Row 1
    html.Div([
        # S&P 500 Price Chart
        html.Div([
            dcc.Graph(
                id='price-chart',
                figure={
                    'data': [
                        go.Scatter(
                            x=prices.index,
                            y=prices['close'],
                            mode='lines',
                            name='S&P 500 (SPY)',
                            line=dict(color='#3498db', width=2)
                        )
                    ],
                    'layout': go.Layout(
                        title='S&P 500 Price History',
                        xaxis={'title': 'Date'},
                        yaxis={'title': 'Price ($)'},
                        hovermode='x unified',
                        plot_bgcolor='#f8f9fa',
                        paper_bgcolor='#ffffff',
                    )
                }
            )
        ], style={'flex': 1, 'margin': '10px'}),
        
        # Returns Distribution
        html.Div([
            dcc.Graph(
                id='returns-dist',
                figure={
                    'data': [
                        go.Histogram(
                            x=returns,
                            nbinsx=30,
                            name='Returns',
                            marker=dict(color='#e74c3c', opacity=0.7)
                        )
                    ],
                    'layout': go.Layout(
                        title='Returns Distribution',
                        xaxis={'title': 'Daily Returns (%)'},
                        yaxis={'title': 'Frequency'},
                        plot_bgcolor='#f8f9fa',
                        paper_bgcolor='#ffffff',
                    )
                }
            )
        ], style={'flex': 1, 'margin': '10px'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),
    
    # Charts Row 2
    html.Div([
        # Conditional Volatility
        html.Div([
            dcc.Graph(
                id='volatility-chart',
                figure={
                    'data': [
                        go.Scatter(
                            x=cond_vol.index,
                            y=cond_vol,
                            mode='lines',
                            name='Conditional Volatility',
                            line=dict(color='#9b59b6', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(155, 89, 182, 0.2)'
                        )
                    ],
                    'layout': go.Layout(
                        title='Conditional Volatility (GARCH)',
                        xaxis={'title': 'Date'},
                        yaxis={'title': 'Volatility (%)'},
                        hovermode='x unified',
                        plot_bgcolor='#f8f9fa',
                        paper_bgcolor='#ffffff',
                    )
                }
            )
        ], style={'flex': 1, 'margin': '10px'}),
        
        # Volatility Forecast
        html.Div([
            dcc.Graph(
                id='forecast-chart',
                figure={
                    'data': [
                        go.Bar(
                            x=forecast_df['Day'],
                            y=forecast_df['Forecasted_Volatility_%'],
                            name='Forecast',
                            marker=dict(color='#1abc9c')
                        )
                    ],
                    'layout': go.Layout(
                        title='10-Day Volatility Forecast',
                        xaxis={'title': 'Days Ahead', 'dtick': 1},
                        yaxis={'title': 'Forecasted Volatility (%)'},
                        plot_bgcolor='#f8f9fa',
                        paper_bgcolor='#ffffff',
                    )
                }
            )
        ], style={'flex': 1, 'margin': '10px'}),
    ], style={'display': 'flex', 'flexWrap': 'wrap'}),
    
    # Model Parameters Section
    html.Div([
        html.H3("GARCH Model Parameters", style={'color': '#2c3e50', 'marginTop': 20}),
        html.Div([
            html.Div([
                html.P(f"ω (omega): {params['omega']:.6f}", style={'fontSize': 16, 'margin': 5}),
                html.P(f"α (alpha[1]): {params['alpha[1]']:.6f}", style={'fontSize': 16, 'margin': 5}),
                html.P(f"β (beta[1]): {params['beta[1]']:.6f}", style={'fontSize': 16, 'margin': 5}),
            ], style={'flex': 1}),
            html.Div([
                html.P(f"Mean Return: {returns.mean():.4f}%", style={'fontSize': 16, 'margin': 5}),
                html.P(f"Std Dev: {returns.std():.4f}%", style={'fontSize': 16, 'margin': 5}),
                html.P(f"Date Range: {returns.index[0].strftime('%Y-%m-%d')} to {returns.index[-1].strftime('%Y-%m-%d')}", 
                      style={'fontSize': 16, 'margin': 5}),
            ], style={'flex': 1}),
        ], style={'display': 'flex', 'backgroundColor': '#fff', 'padding': '20px', 
                 'borderRadius': '10px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}),
    ], style={'margin': '20px'}),
    
    # Footer
    html.Div([
        html.P("Built with Plotly Dash | Data from Alpha Vantage | GARCH modeling with ARCH library",
               style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': 12, 'marginTop': 30}),
    ]),
    
], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px', 'backgroundColor': '#f5f6fa'})


# Run the app
if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting S&P 500 GARCH Dashboard...")
    print("="*60)
    print("Open your browser and go to: http://127.0.0.1:8050/")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, port=8050)