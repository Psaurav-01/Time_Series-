# GARCH / DCC Volatility Dashboard

An interactive web dashboard for **univariate GARCH(1,1)** and **multivariate DCC-GARCH(1,1)** volatility modelling across user-selectable asset universes (11 S&P 500 sectors, stock/bond, NASDAQ vs SPX, gold/bitcoin vs SPX).

Deployed on **Render** via **Docker**.

---

## Features

| Module | What it does |
|---|---|
| **GARCH(1,1)** | Fits a constant-mean GARCH(1,1) per asset (Normal or Student-t). Reports conditional volatility, AIC/BIC, ω/α/β parameters. |
| **Goodness of fit** | Ljung-Box on z_t and z_t², Jarque-Bera normality test, Q-Q plot vs Normal. |
| **Realized variance** | Rolling 21-day squared-return proxy compared against GARCH conditional variance and the VIX daily variance proxy. |
| **DCC-GARCH(1,1)** | Two-step Engle (2002) DCC. Estimates scalar a and b by maximising the DCC quasi-log-likelihood. Produces time-varying correlation matrix R_t and covariance matrix H_t. |
| **Multivariate diagnostics** | Component-wise Ljung-Box, cross-product portmanteau, Mahalanobis distance Q-Q plot (vs χ²(N)). Optional Henze-Zirkler test (requires `pingouin`). |
| **Dashboard** | Plotly Dash with 5 tabs, asset-universe selector, date range, distribution choice. |

---

## Asset Universes

- **11 S&P 500 Sectors** — XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY, XLRE, XLC
- **Stock / Bond** — SPY + TLT
- **NASDAQ & SPX** — QQQ + SPY
- **Gold, Bitcoin & SPX** — GLD + BTC-USD + SPY

---

## Tech Stack

- Python 3.12
- [`arch`](https://arch.readthedocs.io/) — GARCH estimation
- [`yfinance`](https://pypi.org/project/yfinance/) — free market data (no API key)
- `numpy` / `pandas` / `scipy` / `statsmodels`
- [Plotly Dash](https://dash.plotly.com/) + `dash-bootstrap-components`
- **Docker** + **Render** for deployment

---

## Project Structure

```
.
├── Dockerfile
├── render.yaml
├── requirements.txt
├── main.py                    # entry point (local dev + gunicorn)
├── src/
│   ├── dashboard/
│   │   └── app.py             # Dash layout, callbacks, tab renderers
│   ├── models/
│   │   └── garch.py           # GARCH(1,1) + DCC-GARCH(1,1) engine
│   └── utils/
│       └── data_fetcher.py    # yfinance wrapper (backward-compat)
└── README.md
```

---

## Local Development

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
python main.py
# Open http://localhost:8050
```

---

## Docker (local)

```bash
docker build -t garch-dashboard .
docker run -p 8050:8050 garch-dashboard
```

---

## Deploy to Render

1. Push this repository to GitHub.
2. On [render.com](https://render.com), click **New → Web Service**.
3. Connect your GitHub repo.
4. Render detects `render.yaml` automatically — select **Docker** as the environment.
5. Click **Deploy**.

The `render.yaml` file pre-configures the service name, Docker build, and the free plan.

> **Note:** Model fitting on the free tier (512 MB RAM, shared CPU) takes roughly 1–2 minutes for 11 assets over 10 years. A progress spinner is shown while computing.

---

## Model Notes

### GARCH(1,1)
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

Stationarity requires α + β < 1.

### DCC(1,1) — Engle (2002)
Q_t = (1 - a - b)·Q̄ + a·z_{t-1}z'_{t-1} + b·Q_{t-1}

R_t = diag(Q_t)^{-½} Q_t diag(Q_t)^{-½}

H_t = D_t R_t D_t,   where D_t = diag(σ_{1,t}, …, σ_{N,t})

---

## Data

All prices are downloaded from Yahoo Finance via `yfinance`. No API key is required. Data covers `2016-01-01` to `2026-01-01` by default (adjustable in the dashboard sidebar).
