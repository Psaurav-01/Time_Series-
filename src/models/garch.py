
# PROJECT: Univariate GARCH(1,1) + DCC-GARCH(1,1)


import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import requests

from scipy.optimize import minimize
from scipy.stats import jarque_bera, probplot, chi2
from statsmodels.stats.diagnostic import acorr_ljungbox

from arch import arch_model

# Optional multivariate normality test
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False



# 1. CONFIG

START_DATE = "2016-01-01"
END_DATE   = "2026-01-01"

# Choose ONE universe to start.
# Example 1: SPY + bonds
# TICKERS = ["SPY", "TLT"]

# Example 2: SPY + Gold + Bitcoin
# TICKERS = ["SPY", "GLD", "BTC-USD"]

# Example 3: 11 S&P sectors
TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]

MARKET_PROXY = "SPY"      # used for VIX comparison if available
VIX_TICKER = "^VIX"
LAGS = 20
DIST = "normal"           # "normal" or "t" for univariate GARCH
PLOT_DIR_PREFIX = "plots_"  # just for naming titles, no saving to disk by default



# 2. DATA HELPERS

def _yf_session():
    """
    Return a requests.Session with browser-like headers.
    Cloud hosting providers (Render, AWS, GCP…) are often blocked by Yahoo
    Finance's CDN unless the request looks like it comes from a real browser.
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection":      "keep-alive",
    })
    return s


def download_prices(tickers, start, end):
    """
    Download adjusted close prices from Yahoo Finance.

    Uses per-ticker Ticker.history() with a browser-like session so requests
    succeed from cloud hosting environments (Render, AWS, GCP, etc.) where
    yf.download() batch requests are often blocked.

    Falls back to yf.download() batch call if per-ticker fetch fails.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    session = _yf_session()
    frames  = {}
    failed  = []

    # ── per-ticker fetch (most reliable on cloud) ─────────────────────────────
    for ticker in tickers:
        for attempt in range(3):
            try:
                t    = yf.Ticker(ticker, session=session)
                hist = t.history(start=start, end=end, auto_adjust=True)
                if not hist.empty:
                    close = hist["Close"].copy()
                    # strip timezone so all indices are tz-naive
                    if hasattr(close.index, "tz") and close.index.tz is not None:
                        close.index = close.index.tz_localize(None)
                    frames[ticker] = close
                    break
                elif attempt == 2:
                    failed.append(ticker)
            except Exception:
                if attempt < 2:
                    time.sleep(1 + attempt)   # brief back-off
                else:
                    failed.append(ticker)

    # ── batch fallback for any that failed above ──────────────────────────────
    if failed:
        try:
            batch = yf.download(
                failed, start=start, end=end,
                auto_adjust=True, progress=False, threads=False,
            )
            if not batch.empty:
                close_batch = batch["Close"] if "Close" in batch.columns else batch
                if isinstance(close_batch, pd.Series):
                    close_batch = close_batch.to_frame(name=failed[0])
                if hasattr(close_batch.index, "tz") and close_batch.index.tz is not None:
                    close_batch.index = close_batch.index.tz_localize(None)
                for col in close_batch.columns:
                    frames[col] = close_batch[col]
        except Exception:
            pass   # will surface as empty df below

    if not frames:
        raise RuntimeError(
            f"download_prices: could not fetch any data for {tickers} "
            f"({start} → {end}).  "
            "Check ticker symbols, date range, and network access."
        )

    data = pd.DataFrame(frames)
    data = data.sort_index().dropna(how="all")
    return data


def compute_log_returns(prices):
    """
    Compute daily log returns.
    """
    rets = np.log(prices / prices.shift(1)).dropna()
    return rets


def realized_variance_proxy(returns, window=21):
    """
    Simple realized variance proxy using rolling sum of squared daily returns.
    For true RV you would use intraday returns.
    """
    return returns.pow(2).rolling(window).sum()



# 3. UNIVARIATE GARCH

def fit_univariate_garch(series, dist="normal"):
    """
    Fit GARCH(1,1) with constant mean.
    Uses returns in percentage terms for arch package stability.
    """
    y = 100 * series.dropna()

    if dist.lower() == "normal":
        am = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
    elif dist.lower() == "t":
        am = arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
    else:
        raise ValueError("dist must be 'normal' or 't'")

    res = am.fit(disp="off")

    out = {
        "model": am,
        "result": res,
        "aic": res.aic,
        "bic": res.bic,
        "loglik": res.loglikelihood,
        "params": res.params,
        "conditional_vol": res.conditional_volatility / 100.0,  # back to decimal-return scale
        "resid": res.resid / 100.0,
        "std_resid": pd.Series(res.std_resid, index=series.dropna().index).replace([np.inf, -np.inf], np.nan).dropna()
    }
    return out


def univariate_diagnostics(name, std_resid, lags=20, qq_dist="norm"):
    """
    Return Ljung-Box tests and Jarque-Bera on standardized residuals.
    """
    z = std_resid.dropna()
    lb_z = acorr_ljungbox(z, lags=[lags], return_df=True)
    lb_z2 = acorr_ljungbox(z**2, lags=[lags], return_df=True)
    jb_stat, jb_p = jarque_bera(z)

    diag = {
        "asset": name,
        "lb_resid_stat": lb_z["lb_stat"].iloc[0],
        "lb_resid_pvalue": lb_z["lb_pvalue"].iloc[0],
        "lb_sqresid_stat": lb_z2["lb_stat"].iloc[0],
        "lb_sqresid_pvalue": lb_z2["lb_pvalue"].iloc[0],
        "jb_stat": jb_stat,
        "jb_pvalue": jb_p
    }
    return diag


def plot_univariate_diagnostics(name, std_resid, cond_var, actual_returns):
    """
    Plot:
    1) standardized residuals
    2) Q-Q plot
    3) conditional variance vs squared returns
    """
    z = std_resid.dropna()
    idx = z.index

    plt.figure(figsize=(10, 4))
    plt.plot(idx, z)
    plt.axhline(0, linestyle="--")
    plt.title(f"{name} - Standardized Residuals")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 6))
    probplot(z, dist="norm", plot=plt)
    plt.title(f"{name} - Q-Q Plot (Normal)")
    plt.tight_layout()
    plt.show()

    cond_var = pd.Series(cond_var**2, index=actual_returns.dropna().index).dropna()
    sq_ret = actual_returns.loc[cond_var.index] ** 2

    plt.figure(figsize=(10, 4))
    plt.plot(cond_var.index, cond_var, label="Conditional Variance")
    plt.plot(sq_ret.index, sq_ret, label="Squared Returns", alpha=0.6)
    plt.title(f"{name} - Conditional Variance vs Squared Returns")
    plt.legend()
    plt.tight_layout()
    plt.show()



# 4. DCC-GARCH

class DCCGARCH:
    """
    Two-step DCC(1,1) estimation.
    Step 1: use standardized residuals from univariate GARCHs
    Step 2: estimate DCC parameters (a, b) by maximizing DCC log-likelihood

    References/assumptions:
    - Q_t = (1-a-b)Qbar + a * e_{t-1}e_{t-1}' + b * Q_{t-1}
    - R_t = diag(Q_t)^(-1/2) Q_t diag(Q_t)^(-1/2)

    Inputs:
    E: T x N matrix of standardized residuals
    sigmas: T x N matrix of conditional std devs (decimal return scale)
    """

    def __init__(self):
        self.a_ = None
        self.b_ = None
        self.Qbar_ = None
        self.R_t_ = None
        self.H_t_ = None
        self.E_ = None
        self.sigmas_ = None
        self.loglik_ = None

    @staticmethod
    def _dcc_recursion(params, E, Qbar):
        a, b = params
        T, N = E.shape

        Q_t = np.zeros((T, N, N))
        R_t = np.zeros((T, N, N))

        Q_t[0] = Qbar.copy()

        for t in range(T):
            if t > 0:
                et_1 = E[t - 1].reshape(-1, 1)
                Q_t[t] = (1 - a - b) * Qbar + a * (et_1 @ et_1.T) + b * Q_t[t - 1]

            q_diag = np.sqrt(np.diag(Q_t[t]))
            inv_q_diag = np.diag(1.0 / q_diag)
            R_t[t] = inv_q_diag @ Q_t[t] @ inv_q_diag

        return Q_t, R_t

    @staticmethod
    def _negative_loglik(params, E, Qbar):
        a, b = params

        # constraints
        if a < 0 or b < 0 or (a + b) >= 0.999:
            return 1e12

        T, N = E.shape
        _, R_t = DCCGARCH._dcc_recursion(params, E, Qbar)

        nll = 0.0
        for t in range(T):
            Rt = R_t[t]
            et = E[t].reshape(-1, 1)

            try:
                sign, logdet = np.linalg.slogdet(Rt)
                if sign <= 0:
                    return 1e12
                invRt = np.linalg.inv(Rt)
            except np.linalg.LinAlgError:
                return 1e12

            term = logdet + float(et.T @ invRt @ et)
            nll += term

        # ignore constants
        return 0.5 * nll

    def fit(self, E, sigmas):
        """
        E: DataFrame or array, standardized residuals, shape (T, N)
        sigmas: DataFrame or array, conditional std devs, shape (T, N)
        """
        if isinstance(E, pd.DataFrame):
            idx = E.index
            cols = E.columns
            Evals = E.values
        else:
            idx = None
            cols = None
            Evals = np.asarray(E)

        sigmas_vals = sigmas.values if isinstance(sigmas, pd.DataFrame) else np.asarray(sigmas)

        self.E_ = Evals
        self.sigmas_ = sigmas_vals
        self.Qbar_ = np.cov(Evals.T)

        x0 = np.array([0.02, 0.95])
        bounds = [(1e-6, 0.5), (1e-6, 0.999)]
        cons = [{"type": "ineq", "fun": lambda x: 0.999 - x[0] - x[1]}]

        opt = minimize(
            self._negative_loglik,
            x0=x0,
            args=(Evals, self.Qbar_),
            method="SLSQP",
            bounds=bounds,
            constraints=cons
        )

        if not opt.success:
            raise RuntimeError(f"DCC optimization failed: {opt.message}")

        self.a_, self.b_ = opt.x
        self.loglik_ = -self._negative_loglik(opt.x, Evals, self.Qbar_)

        _, R_t = self._dcc_recursion(opt.x, Evals, self.Qbar_)
        self.R_t_ = R_t

        T, N = Evals.shape
        H_t = np.zeros((T, N, N))
        for t in range(T):
            D_t = np.diag(sigmas_vals[t])
            H_t[t] = D_t @ R_t[t] @ D_t

        self.H_t_ = H_t

        if idx is not None and cols is not None:
            self.index_ = idx
            self.columns_ = cols
        else:
            self.index_ = pd.RangeIndex(start=0, stop=Evals.shape[0], step=1)
            self.columns_ = [f"Asset_{i}" for i in range(Evals.shape[1])]

        return self

    def correlation_series(self, asset_i, asset_j):
        """
        Return time series of dynamic correlation between two assets.
        """
        i = self.columns_.index(asset_i) if isinstance(asset_i, str) else asset_i
        j = self.columns_.index(asset_j) if isinstance(asset_j, str) else asset_j

        vals = [self.R_t_[t, i, j] for t in range(len(self.index_))]
        return pd.Series(vals, index=self.index_, name=f"Corr({asset_i},{asset_j})")

    def covariance_series(self, asset_i, asset_j):
        i = self.columns_.index(asset_i) if isinstance(asset_i, str) else asset_i
        j = self.columns_.index(asset_j) if isinstance(asset_j, str) else asset_j

        vals = [self.H_t_[t, i, j] for t in range(len(self.index_))]
        return pd.Series(vals, index=self.index_, name=f"Cov({asset_i},{asset_j})")



# 5. MULTIVARIATE DIAGNOSTICS

def componentwise_ljungbox(E_df, lags=20):
    """
    Run Ljung-Box on each dimension's standardized residuals and squared residuals.
    """
    rows = []
    for col in E_df.columns:
        z = E_df[col].dropna()
        lb1 = acorr_ljungbox(z, lags=[lags], return_df=True)
        lb2 = acorr_ljungbox(z**2, lags=[lags], return_df=True)
        rows.append({
            "asset": col,
            "lb_resid_stat": lb1["lb_stat"].iloc[0],
            "lb_resid_pvalue": lb1["lb_pvalue"].iloc[0],
            "lb_sqresid_stat": lb2["lb_stat"].iloc[0],
            "lb_sqresid_pvalue": lb2["lb_pvalue"].iloc[0]
        })
    return pd.DataFrame(rows)


def cross_product_portmanteau(E_df, lags=10):
    """
    Simple multivariate diagnostic:
    construct vectorized centered cross-products e_i e_j and run Ljung-Box on each.
    """
    E = E_df.copy()
    cols = E.columns
    out = []

    for i in range(len(cols)):
        for j in range(i, len(cols)):
            s = (E.iloc[:, i] * E.iloc[:, j]).dropna()
            s = s - s.mean()
            lb = acorr_ljungbox(s, lags=[lags], return_df=True)
            out.append({
                "pair": f"{cols[i]} x {cols[j]}",
                "lb_stat": lb["lb_stat"].iloc[0],
                "lb_pvalue": lb["lb_pvalue"].iloc[0]
            })
    return pd.DataFrame(out)


def mahalanobis_distances(E_df):
    """
    Mahalanobis distances of standardized residual vectors.
    Under multivariate normality, squared Mahalanobis distances ~ Chi-square(df=N)
    """
    X = E_df.dropna().values
    mu = X.mean(axis=0)
    S = np.cov(X.T)
    S_inv = np.linalg.inv(S)

    d2 = []
    for x in X:
        dx = x - mu
        d2.append(dx.T @ S_inv @ dx)
    d2 = np.array(d2)
    return pd.Series(d2, index=E_df.dropna().index)


def plot_mahalanobis_qq(E_df):
    d2 = mahalanobis_distances(E_df)
    n = len(d2)
    p = E_df.shape[1]
    theo = chi2.ppf((np.arange(1, n + 1) - 0.5) / n, df=p)
    sample = np.sort(d2.values)

    plt.figure(figsize=(6, 6))
    plt.scatter(theo, sample, s=10)
    mn = min(theo.min(), sample.min())
    mx = max(theo.max(), sample.max())
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.title("Mahalanobis Distance Q-Q Plot")
    plt.xlabel(f"Theoretical Chi-square Quantiles (df={p})")
    plt.ylabel("Empirical Squared Mahalanobis Distances")
    plt.tight_layout()
    plt.show()


def multivariate_normality_test(E_df):
    """
    Henze-Zirkler test if pingouin is available.
    """
    if HAS_PINGOUIN:
        return pg.multivariate_normality(E_df.dropna(), alpha=0.05)
    return None



# 6. MAIN PIPELINE

def run_project(tickers, start, end, dist="normal", lags=20, market_proxy="SPY"):
    
    # Step A: Download data

    prices = download_prices(tickers, start, end)
    returns = compute_log_returns(prices)

    print("\nDownloaded price data shape:", prices.shape)
    print("Return matrix shape:", returns.shape)

    
    # Step B: Fit univariate GARCH

    uni_results = {}
    diag_rows = []

    common_index = returns.index.copy()
    for c in returns.columns:
        common_index = common_index.intersection(returns[c].dropna().index)

    returns_aligned = returns.loc[common_index].copy()

    sigma_df = pd.DataFrame(index=returns_aligned.index, columns=returns_aligned.columns, dtype=float)
    resid_df = pd.DataFrame(index=returns_aligned.index, columns=returns_aligned.columns, dtype=float)
    std_resid_df = pd.DataFrame(index=returns_aligned.index, columns=returns_aligned.columns, dtype=float)

    print("\n=== Univariate GARCH fits ===")
    for col in returns_aligned.columns:
        fit = fit_univariate_garch(returns_aligned[col], dist=dist)
        uni_results[col] = fit

        sigma_series = pd.Series(fit["conditional_vol"], index=returns_aligned.index)
        resid_series = pd.Series(fit["resid"], index=returns_aligned.index)
        z_series = pd.Series(fit["std_resid"], index=returns_aligned.index)

        sigma_df[col] = sigma_series
        resid_df[col] = resid_series
        std_resid_df[col] = z_series

        diag = univariate_diagnostics(col, z_series, lags=lags)
        diag["aic"] = fit["aic"]
        diag["bic"] = fit["bic"]
        diag_rows.append(diag)

        print(f"{col}: AIC={fit['aic']:.3f}, BIC={fit['bic']:.3f}, LL={fit['loglik']:.3f}")

    univariate_diag_table = pd.DataFrame(diag_rows).set_index("asset")
    print("\n=== Univariate diagnostics ===")
    print(univariate_diag_table.round(4))

    # Plot one asset example
    example_asset = returns_aligned.columns[0]
    plot_univariate_diagnostics(
        example_asset,
        std_resid_df[example_asset].dropna(),
        sigma_df[example_asset].dropna(),
        returns_aligned[example_asset].dropna()
    )


    # Step C: Variance comparison with RV and VIX

    print("\n=== Conditional variance vs realized variance proxy vs VIX ===")

    rv_proxy = realized_variance_proxy(returns_aligned, window=21)

    if market_proxy in returns_aligned.columns:
        market_cond_var = sigma_df[market_proxy] ** 2
        market_rv = rv_proxy[market_proxy]

        vix = download_prices([VIX_TICKER], start, end)
        if VIX_TICKER in vix.columns:
            vix_series = vix[VIX_TICKER].dropna()
        else:
            vix_series = vix.squeeze().dropna()

        compare_df = pd.concat(
            [
                market_cond_var.rename("cond_var"),
                market_rv.rename("rv_proxy"),
                vix_series.rename("VIX")
            ],
            axis=1
        ).dropna()

        # scale VIX to daily variance proxy:
        # annualized vol -> daily variance approx (VIX/100)^2 / 252
        compare_df["vix_var_proxy"] = (compare_df["VIX"] / 100.0) ** 2 / 252.0

        print(compare_df[["cond_var", "rv_proxy", "vix_var_proxy"]].corr().round(4))

        plt.figure(figsize=(10, 4))
        plt.plot(compare_df.index, compare_df["cond_var"], label="Conditional Variance")
        plt.plot(compare_df.index, compare_df["rv_proxy"], label="Realized Variance Proxy", alpha=0.8)
        plt.plot(compare_df.index, compare_df["vix_var_proxy"], label="VIX Variance Proxy", alpha=0.8)
        plt.title(f"{market_proxy} - Conditional Variance vs RV Proxy vs VIX Proxy")
        plt.legend()
        plt.tight_layout()
        plt.show()

    else:
        print(f"{market_proxy} not in selected tickers. Skipping VIX comparison.")

    
    # Step D: Build DCC input
    dcc_df = pd.concat([std_resid_df, sigma_df], axis=1).dropna()
    std_cols = returns_aligned.columns.tolist()
    sig_cols = returns_aligned.columns.tolist()

    E_df = std_resid_df.dropna()
    sigma_df_clean = sigma_df.loc[E_df.index].dropna()

    common_idx = E_df.index.intersection(sigma_df_clean.index)
    E_df = E_df.loc[common_idx]
    sigma_df_clean = sigma_df_clean.loc[common_idx]

    print("\nDCC input shape:", E_df.shape)


    # Step E: Fit DCC

    print("\n=== Fitting DCC(1,1) ===")
    dcc = DCCGARCH().fit(E_df, sigma_df_clean)
    print(f"DCC parameters: a={dcc.a_:.6f}, b={dcc.b_:.6f}")
    print(f"DCC log-likelihood (quasi): {dcc.loglik_:.6f}")

    # Plot one pair if possible
    if len(tickers) >= 2:
        a1, a2 = tickers[0], tickers[1]
        corr_series = dcc.correlation_series(a1, a2)
        plt.figure(figsize=(10, 4))
        plt.plot(corr_series.index, corr_series)
        plt.title(f"Dynamic Correlation: {a1} vs {a2}")
        plt.tight_layout()
        plt.show()


    # Step F: Multivariate diagnostics
    print("\n=== Multivariate diagnostics ===")

    comp_lb = componentwise_ljungbox(E_df, lags=lags)
    print("\nComponentwise Ljung-Box:")
    print(comp_lb.round(4))

    cross_lb = cross_product_portmanteau(E_df, lags=10)
    print("\nCross-product portmanteau (first 10 rows):")
    print(cross_lb.head(10).round(4))

    plot_mahalanobis_qq(E_df)

    mnorm = multivariate_normality_test(E_df)
    if mnorm is not None:
        print("\nHenze-Zirkler multivariate normality test:")
        print(mnorm)
    else:
        print("\nInstall 'pingouin' for a multivariate normality test:")
        print("pip install pingouin")


    # Return everything
    return {
        "prices": prices,
        "returns": returns_aligned,
        "univariate_results": uni_results,
        "univariate_diagnostics": univariate_diag_table,
        "sigma": sigma_df,
        "resid": resid_df,
        "std_resid": std_resid_df,
        "rv_proxy": rv_proxy,
        "dcc": dcc,
        "dcc_E": E_df,
        "dcc_sigma": sigma_df_clean,
        "componentwise_ljungbox": comp_lb,
        "cross_product_portmanteau": cross_lb
    }



# 7. RUN

if __name__ == "__main__":
    results = run_project(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
        dist=DIST,
        lags=LAGS,
        market_proxy=MARKET_PROXY
    )