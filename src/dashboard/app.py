"""
GARCH / DCC Volatility Dashboard
=================================
Interactive Dash application for GARCH(1,1) and DCC-GARCH(1,1) analysis
across multiple asset universes (11 S&P 500 sectors, stock/bond, etc.).

Architecture
------------
* All heavy computation happens inside the "Run Model" callback.
* Results are serialised to dcc.Store components (JSON) so other callbacks
  can read them without re-fitting the model.
* suppress_callback_exceptions=True allows tab-content dropdowns/charts
  (created dynamically) to be wired up before they exist in the layout.
"""

import io
import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import chi2, probplot, jarque_bera

import traceback

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

# ── Ensure project root is on sys.path so src.models.garch is importable ────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.models.garch import (
    download_prices,
    compute_log_returns,
    realized_variance_proxy,
    fit_univariate_garch,
    univariate_diagnostics,
    DCCGARCH,
    componentwise_ljungbox,
    cross_product_portmanteau,
    mahalanobis_distances,
    VIX_TICKER,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

UNIVERSES = {
    "11 S&P 500 Sectors": [
        "XLB", "XLE", "XLF", "XLI", "XLK",
        "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC",
    ],
    "Stock / Bond (SPY + TLT)":   ["SPY", "TLT"],
    "NASDAQ & SPX (QQQ + SPY)":   ["QQQ", "SPY"],
    "Gold, Bitcoin & SPX":        ["GLD", "BTC-USD", "SPY"],
}

SECTOR_NAMES = {
    "XLB": "Materials",      "XLE": "Energy",          "XLF": "Financials",
    "XLI": "Industrials",    "XLK": "Technology",      "XLP": "Consumer Staples",
    "XLU": "Utilities",      "XLV": "Health Care",     "XLY": "Consumer Discret.",
    "XLRE": "Real Estate",   "XLC": "Comm. Services",  "SPY": "S&P 500",
    "TLT": "20yr Treasuries","QQQ": "NASDAQ-100",      "GLD": "Gold",
    "BTC-USD": "Bitcoin",
}

# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _df_to_json(df: pd.DataFrame) -> str | None:
    if df is None:
        return None
    return df.to_json(date_format="iso", orient="split")


def _json_to_df(s: str | None) -> pd.DataFrame | None:
    if not s:
        return None
    df = pd.read_json(io.StringIO(s), orient="split")
    # Try to coerce the index to DatetimeIndex
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df


# ─────────────────────────────────────────────────────────────────────────────
# App initialisation
# ─────────────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="GARCH / DCC Dashboard",
)
server = app.server  # exposed for Gunicorn: `gunicorn main:server`

# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def _metric_card(title: str, value: str, subtitle: str, color: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="text-muted mb-1"),
            html.H3(value, style={"color": color}, className="mb-0 fw-bold"),
            html.Small(subtitle, className="text-muted"),
        ]),
        className="text-center shadow-sm",
    )


_sidebar = dbc.Card([
    html.H5("⚙️ Configuration", className="fw-bold"),
    html.Hr(className="my-2"),

    dbc.Label("Asset Universe", className="fw-semibold"),
    dcc.Dropdown(
        id="universe-select",
        options=[{"label": k, "value": k} for k in UNIVERSES],
        value="11 S&P 500 Sectors",
        clearable=False,
    ),
    html.Br(),

    dbc.Row([
        dbc.Col([dbc.Label("Start Date"), dbc.Input(id="start-date", value="2016-01-01", size="sm")], width=6),
        dbc.Col([dbc.Label("End Date"),   dbc.Input(id="end-date",   value="2026-01-01", size="sm")], width=6),
    ]),
    html.Br(),

    dbc.Label("GARCH Distribution", className="fw-semibold"),
    dbc.RadioItems(
        id="dist-select",
        options=[{"label": " Normal", "value": "normal"},
                 {"label": " Student-t", "value": "t"}],
        value="normal",
        inline=True,
    ),
    html.Br(),

    dbc.Label("Ljung-Box Lags", className="fw-semibold"),
    dbc.Input(id="lags-input", type="number", value=20, min=5, max=50, size="sm"),
    html.Br(),

    dbc.Button(
        "▶  Run Model",
        id="run-btn", color="primary", className="w-100 fw-bold",
    ),
    html.Div(id="run-status", className="mt-2 small text-muted"),
    html.Div(id="run-error",  className="mt-1"),   # shows red alert on failure

], body=True, className="sticky-top", style={"top": "20px"})


_tabs = dbc.Tabs([
    dbc.Tab(label="📈 GARCH Overview",    tab_id="tab-garch"),
    dbc.Tab(label="🔬 Diagnostics",       tab_id="tab-diag"),
    dbc.Tab(label="📊 DCC Correlations",  tab_id="tab-dcc"),
    dbc.Tab(label="📉 VIX / RV",          tab_id="tab-vix"),
    dbc.Tab(label="🧮 Multivariate",      tab_id="tab-multi"),
], id="main-tabs", active_tab="tab-garch")


app.layout = dbc.Container([

    # ── Header ────────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(html.H3("GARCH / DCC Volatility Dashboard",
                        className="fw-bold text-primary mb-0"), width=8),
        dbc.Col(html.P("GARCH(1,1) · DCC-GARCH(1,1) · S&P 500 & Multi-Asset",
                       className="text-muted text-end mt-2 mb-0"), width=4),
    ], className="py-3 border-bottom mb-3"),

    # ── dcc.Store: serialised model outputs ───────────────────────────────────
    dcc.Store(id="store-returns"),
    dcc.Store(id="store-sigma"),
    dcc.Store(id="store-std-resid"),
    dcc.Store(id="store-uni-diag"),
    dcc.Store(id="store-dcc-params"),
    dcc.Store(id="store-dcc-corr"),    # {"corr": {pair: [vals]}, "dates": [...]}
    dcc.Store(id="store-dcc-latest"),  # {"matrix": [[...]], "cols": [...]}
    dcc.Store(id="store-rv"),
    dcc.Store(id="store-vix"),         # {"dates": [...], "values": [...]}
    dcc.Store(id="store-comp-lb"),
    dcc.Store(id="store-cross-lb"),
    dcc.Store(id="store-tickers"),
    dcc.Store(id="store-error"),       # holds error text if run_model fails

    # ── Body ──────────────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(_sidebar, width=3),
        dbc.Col([
            _tabs,
            dcc.Loading(
                html.Div(id="tab-content", className="mt-3"),
                type="circle",
            ),
        ], width=9),
    ]),

], fluid=True)


# ─────────────────────────────────────────────────────────────────────────────
# Callback: Run Model
# ─────────────────────────────────────────────────────────────────────────────

_NO_DATA = (None,) * 12   # 12 dcc.Store outputs (error path)


@app.callback(
    Output("store-returns",    "data"),
    Output("store-sigma",      "data"),
    Output("store-std-resid",  "data"),
    Output("store-uni-diag",   "data"),
    Output("store-dcc-params", "data"),
    Output("store-dcc-corr",   "data"),
    Output("store-dcc-latest", "data"),
    Output("store-rv",         "data"),
    Output("store-vix",        "data"),
    Output("store-comp-lb",    "data"),
    Output("store-cross-lb",   "data"),
    Output("store-tickers",    "data"),
    Output("run-status",       "children"),
    Output("store-error",      "data"),
    Input("run-btn",           "n_clicks"),
    State("universe-select",   "value"),
    State("start-date",        "value"),
    State("end-date",          "value"),
    State("dist-select",       "value"),
    State("lags-input",        "value"),
    prevent_initial_call=True,
)
def run_model(n_clicks, universe, start, end, dist, lags):
    """Download data, fit GARCH(1,1) per asset, then DCC(1,1)."""
    try:
        tickers = UNIVERSES[universe]
        lags    = int(lags) if lags else 20

        # ── Step A: data ──────────────────────────────────────────────────────
        prices = download_prices(tickers, start, end)
        if prices is None or prices.empty:
            raise ValueError(
                f"No price data returned for {tickers} ({start} → {end}). "
                "Check tickers and date range."
            )

        returns = compute_log_returns(prices)
        # Ensure tz-naive, midnight-normalised index (defensive, mirrors garch.py)
        returns.index = pd.to_datetime(returns.index).normalize()
        # Remove duplicate dates (can arise from DST tz normalisation)
        returns = returns[~returns.index.duplicated(keep="last")]
        # Drop tickers whose entire column is NaN (failed download)
        returns = returns.dropna(how="all", axis=1)
        # Keep only rows where ALL remaining assets have a valid return
        returns = returns.dropna(how="any")

        if len(returns) < 100:
            raise ValueError(
                f"Only {len(returns)} common observations — need ≥ 100. "
                "Widen the date range."
            )

        # ── Step B: univariate GARCH ──────────────────────────────────────────
        sigma_df  = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        z_df      = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
        diag_rows = []

        for col in returns.columns:
            series      = returns[col].dropna()
            fit         = fit_univariate_garch(series, dist=dist)
            cond_vol_s  = pd.Series(fit["conditional_vol"], index=series.index)
            std_resid_s = fit["std_resid"]

            sigma_df[col] = cond_vol_s.reindex(returns.index)
            z_df[col]     = std_resid_s.reindex(returns.index)

            d            = univariate_diagnostics(col, std_resid_s, lags=lags)
            d["aic"]     = fit["aic"]
            d["bic"]     = fit["bic"]
            d["loglik"]  = fit["loglik"]
            p            = fit["params"]
            d["omega"]   = float(p.get("omega",    np.nan))
            d["alpha"]   = float(p.get("alpha[1]", np.nan))
            d["beta"]    = float(p.get("beta[1]",  np.nan))
            diag_rows.append(d)

        uni_diag = pd.DataFrame(diag_rows).set_index("asset")

        # ── Step C: DCC ───────────────────────────────────────────────────────
        E_df       = z_df.dropna()
        sig_cl     = sigma_df.loc[E_df.index].dropna()
        common_dcc = E_df.index.intersection(sig_cl.index)
        E_df, sig_cl = E_df.loc[common_dcc], sig_cl.loc[common_dcc]

        dcc_model     = DCCGARCH().fit(E_df, sig_cl)
        cols          = list(returns.columns)
        dcc_corr_dict = {
            f"{cols[i]}|{cols[j]}": dcc_model.R_t_[:, i, j].tolist()
            for i in range(len(cols))
            for j in range(i + 1, len(cols))
        }
        latest_matrix = dcc_model.R_t_[-1].tolist()
        dcc_dates     = [str(d) for d in E_df.index]

        # ── Step D: realized variance + VIX ──────────────────────────────────
        rv_proxy = realized_variance_proxy(returns, window=21)
        vix_json = None
        try:
            vix_prices = download_prices([VIX_TICKER], start, end)
            vix_s      = vix_prices.squeeze().dropna()
            vix_json   = json.dumps({
                "dates":  [str(d) for d in vix_s.index],
                "values": vix_s.tolist(),
            })
        except Exception:
            pass  # VIX is optional — skip silently

        # ── Step E: multivariate diagnostics ─────────────────────────────────
        comp_lb  = componentwise_ljungbox(E_df, lags=lags)
        cross_lb = cross_product_portmanteau(E_df, lags=10)

        status = (
            f"✅ {len(returns):,} obs · {len(tickers)} assets · "
            f"DCC a={dcc_model.a_:.4f}, b={dcc_model.b_:.4f}"
        )

        return (
            _df_to_json(returns),
            _df_to_json(sigma_df),
            _df_to_json(z_df),
            _df_to_json(uni_diag.reset_index()),
            json.dumps({"a": dcc_model.a_, "b": dcc_model.b_, "loglik": dcc_model.loglik_}),
            json.dumps({"corr": dcc_corr_dict, "dates": dcc_dates}),
            json.dumps({"matrix": latest_matrix, "cols": cols}),
            _df_to_json(rv_proxy),
            vix_json,
            _df_to_json(comp_lb),
            _df_to_json(cross_lb),
            json.dumps(tickers),
            status,
            None,   # clear any previous error
        )

    except Exception as exc:
        err_tb  = traceback.format_exc()
        err_msg = f"{type(exc).__name__}: {exc}"
        return (
            *_NO_DATA,
            "❌ Failed — see error card below",
            json.dumps({"msg": err_msg, "detail": err_tb}),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Callback: show error alert when run_model fails
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("run-error", "children"),
    Input("store-error", "data"),
)
def show_error(err_json):
    if not err_json:
        return None
    err = json.loads(err_json)
    return dbc.Alert([
        html.B("Error: "), err["msg"],
        html.Details([
            html.Summary("Full traceback"),
            html.Pre(err["detail"], style={"fontSize": "11px", "whiteSpace": "pre-wrap"}),
        ], className="mt-2"),
    ], color="danger", dismissable=True, className="small")


# ─────────────────────────────────────────────────────────────────────────────
# Callback: Tab router
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs",        "active_tab"),
    Input("store-returns",    "data"),
    Input("store-sigma",      "data"),
    Input("store-std-resid",  "data"),
    Input("store-uni-diag",   "data"),
    Input("store-dcc-params", "data"),
    Input("store-dcc-corr",   "data"),
    Input("store-dcc-latest", "data"),
    Input("store-rv",         "data"),
    Input("store-vix",        "data"),
    Input("store-comp-lb",    "data"),
    Input("store-cross-lb",   "data"),
    Input("store-tickers",    "data"),
)
def render_tab(
    active_tab, ret_j, sig_j, z_j, udiag_j, dcc_params_j,
    dcc_corr_j, dcc_latest_j, rv_j, vix_j,
    comp_lb_j, cross_lb_j, tickers_j,
):
    _placeholder = dbc.Alert(
        [
            html.B("Ready. "),
            "Press ▶ Run Model in the sidebar to fetch data and fit the models. ",
            html.Br(),
            html.Small(
                "Computation takes ~1–2 minutes for 11 assets over 10 years. "
                "The button disables while running.",
                className="text-muted",
            ),
        ],
        color="info",
    )

    if not ret_j:
        return _placeholder

    try:
        returns  = _json_to_df(ret_j)
        sigma    = _json_to_df(sig_j)
        z_df     = _json_to_df(z_j)
        uni_diag = _json_to_df(udiag_j)
        tickers  = json.loads(tickers_j)

        if active_tab == "tab-garch":
            return _tab_garch(returns, sigma, z_df, uni_diag, tickers)

        if active_tab == "tab-diag":
            return _tab_diagnostics(uni_diag)

        if active_tab == "tab-dcc":
            return _tab_dcc(
                json.loads(dcc_corr_j)   if dcc_corr_j   else None,
                json.loads(dcc_latest_j) if dcc_latest_j else None,
                json.loads(dcc_params_j) if dcc_params_j else None,
                tickers,
            )

        if active_tab == "tab-vix":
            return _tab_vix(sigma, _json_to_df(rv_j), vix_j, tickers)

        if active_tab == "tab-multi":
            return _tab_multi(z_df, _json_to_df(comp_lb_j), _json_to_df(cross_lb_j))

    except Exception as exc:
        err_tb = traceback.format_exc()
        return dbc.Alert([
            html.B(f"Tab rendering error ({active_tab}): "),
            str(exc),
            html.Details([
                html.Summary("Full traceback"),
                html.Pre(err_tb, style={"fontSize": "11px", "whiteSpace": "pre-wrap"}),
            ], className="mt-2"),
        ], color="danger", dismissable=True, className="small")

    return _placeholder


# ─────────────────────────────────────────────────────────────────────────────
# Tab: GARCH Overview
# ─────────────────────────────────────────────────────────────────────────────

def _tab_garch(returns, sigma, z_df, uni_diag, tickers):
    # All-asset annualised conditional vol
    ann_vol = sigma * np.sqrt(252) * 100
    fig_all = go.Figure()
    for col in tickers:
        if col in ann_vol.columns:
            s = ann_vol[col].dropna()
            fig_all.add_trace(go.Scatter(
                x=s.index, y=s.values, mode="lines",
                name=SECTOR_NAMES.get(col, col), opacity=0.85,
            ))
    fig_all.update_layout(
        title="Conditional Volatility — All Assets (Annualised %)",
        xaxis_title="Date", yaxis_title="Ann. Vol (%)",
        hovermode="x unified", template="plotly_white", height=370,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # AIC / BIC bar chart
    if uni_diag is not None and "asset" in uni_diag.columns:
        fig_ic = go.Figure([
            go.Bar(x=uni_diag["asset"], y=uni_diag["aic"], name="AIC", marker_color="#3498db"),
            go.Bar(x=uni_diag["asset"], y=uni_diag["bic"], name="BIC", marker_color="#e74c3c"),
        ])
        fig_ic.update_layout(
            title="AIC / BIC by Asset", barmode="group",
            template="plotly_white", height=280,
            xaxis_title="Asset", yaxis_title="Value",
        )
    else:
        fig_ic = go.Figure()

    # Parameter table
    param_cols = ["asset", "omega", "alpha", "beta", "loglik", "aic", "bic"]
    avail      = [c for c in param_cols if c in uni_diag.columns]
    param_tbl  = _datatable(uni_diag[avail].round(5), style_cond=[
        {"if": {"row_index": "odd"}, "backgroundColor": "#f5f6fa"}
    ])

    # Per-asset detail section
    asset_dd = dcc.Dropdown(
        id="garch-asset-pick",
        options=[{"label": SECTOR_NAMES.get(t, t), "value": t} for t in tickers],
        value=tickers[0], clearable=False,
        style={"width": "250px"},
    )

    return html.Div([
        dcc.Graph(figure=fig_all),
        dcc.Graph(figure=fig_ic),
        html.Hr(),
        dbc.Row([
            dbc.Col(html.H5("Per-Asset Detail"), width="auto"),
            dbc.Col(asset_dd, width="auto"),
        ], align="center", className="mb-3"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="garch-cond-vol"),   width=6),
            dbc.Col(dcc.Graph(id="garch-qq"),          width=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="garch-std-resid"),  width=6),
            dbc.Col(dcc.Graph(id="garch-cond-vs-sq"), width=6),
        ]),
        html.Hr(),
        html.H5("GARCH(1,1) Parameter Table"),
        param_tbl,
    ])


@app.callback(
    Output("garch-cond-vol",   "figure"),
    Output("garch-qq",          "figure"),
    Output("garch-std-resid",  "figure"),
    Output("garch-cond-vs-sq", "figure"),
    Input("garch-asset-pick",  "value"),
    Input("store-sigma",       "data"),
    Input("store-std-resid",   "data"),
    Input("store-returns",     "data"),
)
def _update_garch_asset(asset, sig_j, z_j, ret_j):
    _empty  = go.Figure()
    if not asset or not sig_j or not z_j or not ret_j:
        return _empty, _empty, _empty, _empty

    sigma   = _json_to_df(sig_j)
    z_df    = _json_to_df(z_j)
    returns = _json_to_df(ret_j)

    if sigma is None or z_df is None or returns is None:
        return _empty, _empty, _empty, _empty

    if asset not in sigma.columns:
        return _empty, _empty, _empty, _empty

    cond_vol = sigma[asset].dropna()
    z        = z_df[asset].dropna()
    ret      = returns[asset].dropna()

    # 1. Conditional volatility (annualised)
    fig1 = go.Figure(go.Scatter(
        x=cond_vol.index, y=(cond_vol * np.sqrt(252) * 100).values,
        mode="lines", fill="tozeroy",
        fillcolor="rgba(52,152,219,0.15)",
        line=dict(color="#3498db", width=1.5),
    ))
    fig1.update_layout(
        title=f"{asset} — Conditional Volatility (Ann. %)",
        xaxis_title="Date", yaxis_title="Ann. Vol (%)",
        template="plotly_white", height=300,
    )

    # 2. Q-Q plot vs Normal
    (osm, osr), (slope, intercept, _) = probplot(z.values, dist="norm")
    qq_x = np.linspace(osm.min(), osm.max(), 120)
    fig2 = go.Figure([
        go.Scatter(x=osm, y=osr, mode="markers",
                   marker=dict(color="#e74c3c", size=3, opacity=0.5), name="Sample"),
        go.Scatter(x=qq_x, y=slope * qq_x + intercept, mode="lines",
                   line=dict(color="#2c3e50", dash="dash"), name="Normal"),
    ])
    fig2.update_layout(
        title=f"{asset} — Q-Q Plot (Normal)",
        xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles",
        template="plotly_white", height=300,
    )

    # 3. Standardised residuals z_t with ±2 bands
    _, jb_p = jarque_bera(z.values)
    fig3 = go.Figure(go.Scatter(
        x=z.index, y=z.values, mode="lines",
        line=dict(color="#9b59b6", width=0.9),
    ))
    for y_val in [2, -2]:
        fig3.add_hline(y=y_val, line_dash="dot", line_color="red", opacity=0.5)
    fig3.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.4)
    fig3.update_layout(
        title=f"{asset} — Std. Residuals z_t  (JB p={jb_p:.4f})",
        xaxis_title="Date", yaxis_title="z_t",
        template="plotly_white", height=300,
    )

    # 4. Conditional variance vs squared returns
    cond_var   = (cond_vol ** 2).dropna()
    sq_ret     = (ret ** 2).dropna()
    common_idx = cond_var.index.intersection(sq_ret.index)
    fig4 = go.Figure([
        go.Scatter(x=common_idx, y=cond_var.loc[common_idx].values,
                   mode="lines", name="Cond. Variance (GARCH)",
                   line=dict(color="#1abc9c", width=1.5)),
        go.Scatter(x=common_idx, y=sq_ret.loc[common_idx].values,
                   mode="lines", name="Squared Returns",
                   line=dict(color="#f39c12", width=0.8), opacity=0.6),
    ])
    fig4.update_layout(
        title=f"{asset} — Cond. Variance vs Squared Returns",
        xaxis_title="Date", yaxis_title="Variance",
        template="plotly_white", height=300,
    )

    return fig1, fig2, fig3, fig4


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def _tab_diagnostics(uni_diag):
    if uni_diag is None:
        return dbc.Alert("No diagnostics data.", color="warning")

    diag_cols = ["asset", "lb_resid_stat", "lb_resid_pvalue",
                 "lb_sqresid_stat", "lb_sqresid_pvalue", "jb_stat", "jb_pvalue"]
    avail  = [c for c in diag_cols if c in uni_diag.columns]
    p_cols = ["lb_resid_pvalue", "lb_sqresid_pvalue", "jb_pvalue"]

    style_cond = []
    for col in p_cols:
        if col in avail:
            style_cond += [
                {"if": {"filter_query": f"{{{col}}} < 0.05", "column_id": col},
                 "backgroundColor": "#ffd6d6", "color": "#8b0000"},
                {"if": {"filter_query": f"{{{col}}} >= 0.05", "column_id": col},
                 "backgroundColor": "#d6f5d6", "color": "#006400"},
            ]

    tbl = _datatable(uni_diag[avail].round(4), style_cond=style_cond)

    note = dbc.Alert([
        html.B("Key: "),
        "🔴 red p < 0.05 → reject null hypothesis.  ",
        html.Br(),
        html.B("LB Resid: "), "no serial autocorrelation in z_t.  ",
        html.B("LB Sq Resid: "), "no remaining ARCH effects in z_t².  ",
        html.B("Jarque-Bera: "), "normality of standardised residuals.",
    ], color="light", className="small mt-3")

    return html.Div([html.H5("Ljung-Box & Normality Diagnostics"), tbl, note])


# ─────────────────────────────────────────────────────────────────────────────
# Tab: DCC Correlations
# ─────────────────────────────────────────────────────────────────────────────

def _tab_dcc(dcc_corr_data, dcc_latest_data, dcc_params, tickers):
    if dcc_latest_data is None:
        return dbc.Alert("No DCC data.", color="warning")

    matrix = np.array(dcc_latest_data["matrix"])
    cols   = dcc_latest_data["cols"]
    labels = [SECTOR_NAMES.get(c, c) for c in cols]

    # Latest DCC correlation heatmap
    fig_heat = go.Figure(go.Heatmap(
        z=np.round(matrix, 3), x=labels, y=labels,
        colorscale="RdBu_r", zmin=-1, zmax=1,
        text=np.round(matrix, 2),
        texttemplate="%{text}", textfont={"size": 10},
        hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
    ))
    fig_heat.update_layout(
        title="Latest Dynamic Conditional Correlation Matrix",
        template="plotly_white", height=520,
    )

    pairs      = list(dcc_corr_data["corr"].keys()) if dcc_corr_data else []
    pair_opts  = [{"label": p.replace("|", " vs "), "value": p} for p in pairs]

    params_txt = ""
    if dcc_params:
        params_txt = (
            f"DCC(1,1) parameters:  a = {dcc_params['a']:.5f},  "
            f"b = {dcc_params['b']:.5f},  "
            f"a+b = {dcc_params['a']+dcc_params['b']:.5f}  "
            f"(persistence)    |    Log-lik = {dcc_params['loglik']:.1f}"
        )

    return html.Div([
        dbc.Alert(params_txt, color="info", className="small py-2") if params_txt else html.Div(),
        dcc.Graph(figure=fig_heat),
        html.Hr(),
        html.H5("Dynamic Correlation Time Series"),
        dcc.Dropdown(
            id="dcc-pair-pick", options=pair_opts,
            value=pairs[0] if pairs else None, clearable=False,
            style={"width": "350px"},
        ),
        dcc.Graph(id="dcc-pair-chart"),
    ])


@app.callback(
    Output("dcc-pair-chart", "figure"),
    Input("dcc-pair-pick",  "value"),
    Input("store-dcc-corr", "data"),
)
def _update_dcc_pair(pair, dcc_corr_j):
    if not pair or not dcc_corr_j:
        return go.Figure()

    data   = json.loads(dcc_corr_j)
    corr   = data["corr"].get(pair, [])
    dates  = data["dates"]
    a1, a2 = pair.split("|")
    mean_c = float(np.mean(corr))

    fig = go.Figure(go.Scatter(
        x=dates, y=corr, mode="lines",
        line=dict(color="#e74c3c", width=1.4),
        name=f"ρ({a1}, {a2})",
    ))
    fig.add_hline(
        y=mean_c, line_dash="dash", line_color="#555",
        annotation_text=f"Mean = {mean_c:.3f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=f"Dynamic Correlation: {SECTOR_NAMES.get(a1,a1)} vs {SECTOR_NAMES.get(a2,a2)}",
        xaxis_title="Date", yaxis_title="Correlation",
        template="plotly_white", height=370,
        yaxis=dict(range=[-1, 1]),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Tab: VIX / Realized Variance
# ─────────────────────────────────────────────────────────────────────────────

def _tab_vix(sigma, rv, vix_j, tickers):
    # Use SPY if in the universe, else first ticker
    market = "SPY" if "SPY" in tickers else tickers[0]

    if market not in sigma.columns:
        return dbc.Alert(f"{market} not among fitted assets.", color="warning")

    cond_var  = (sigma[market] ** 2).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cond_var.index, y=cond_var.values,
        mode="lines", name="GARCH Cond. Variance",
        line=dict(color="#3498db", width=1.5),
    ))

    parts = {"GARCH Cond. Var": cond_var}

    if rv is not None and market in rv.columns:
        rv_s = rv[market].dropna()
        fig.add_trace(go.Scatter(
            x=rv_s.index, y=rv_s.values,
            mode="lines", name="RV Proxy (21-day rolling)",
            line=dict(color="#27ae60", width=1.2), opacity=0.75,
        ))
        parts["RV Proxy"] = rv_s

    if vix_j:
        vix_payload = json.loads(vix_j)
        vix_s       = pd.Series(
            vix_payload["values"],
            index=pd.to_datetime(vix_payload["dates"]),
        ).dropna()
        vix_var     = (vix_s / 100.0) ** 2 / 252.0
        fig.add_trace(go.Scatter(
            x=vix_var.index, y=vix_var.values,
            mode="lines", name="VIX Var. Proxy  ( (VIX/100)² / 252 )",
            line=dict(color="#e74c3c", width=1.2), opacity=0.75,
        ))
        parts["VIX Var. Proxy"] = vix_var

    fig.update_layout(
        title=f"{market} — GARCH Cond. Variance vs RV Proxy vs VIX Proxy",
        xaxis_title="Date", yaxis_title="Daily Variance",
        hovermode="x unified", template="plotly_white", height=420,
    )

    # Pairwise correlation table
    if len(parts) > 1:
        corr_df  = pd.concat(parts, axis=1).dropna().corr().round(4)
        corr_tbl = _datatable(corr_df.reset_index().rename(columns={"index": "Measure"}))
    else:
        corr_tbl = html.Div()

    return html.Div([
        dcc.Graph(figure=fig),
        html.H5("Pairwise Correlations"),
        corr_tbl,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Tab: Multivariate Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def _tab_multi(z_df, comp_lb, cross_lb):
    if z_df is None:
        return dbc.Alert("No data.", color="warning")

    # Mahalanobis Q-Q plot
    X = z_df.dropna().values
    p = X.shape[1]
    try:
        mu    = X.mean(axis=0)
        S_inv = np.linalg.inv(np.cov(X.T))
        d2    = np.array([float((x - mu) @ S_inv @ (x - mu)) for x in X])
        n     = len(d2)
        theo  = chi2.ppf((np.arange(1, n + 1) - 0.5) / n, df=p)
        samp  = np.sort(d2)
        mn, mx = min(theo.min(), samp.min()), max(theo.max(), samp.max())

        fig_maha = go.Figure([
            go.Scatter(x=theo, y=samp, mode="markers",
                       marker=dict(color="#9b59b6", size=2.5, opacity=0.45),
                       name="Empirical"),
            go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                       line=dict(color="gray", dash="dash"), name="45° line"),
        ])
        fig_maha.update_layout(
            title=f"Mahalanobis Distance Q-Q Plot  (df = {p})",
            xaxis_title=f"Theoretical χ²({p}) Quantiles",
            yaxis_title="Empirical Sq. Mahalanobis Distances",
            template="plotly_white", height=430,
        )
    except Exception as exc:
        fig_maha = go.Figure()
        fig_maha.add_annotation(
            text=f"Could not compute Mahalanobis distances: {exc}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )

    # Component-wise LB table
    comp_tbl = html.Div()
    if comp_lb is not None:
        p_style = [
            {"if": {"filter_query": f"{{{c}}} < 0.05", "column_id": c},
             "backgroundColor": "#ffd6d6"}
            for c in ["lb_resid_pvalue", "lb_sqresid_pvalue"]
            if c in comp_lb.columns
        ]
        comp_tbl = _datatable(comp_lb.round(4), style_cond=p_style)

    # Cross-product portmanteau (top 20 by LB stat)
    cross_tbl = html.Div()
    if cross_lb is not None:
        top = cross_lb.nlargest(20, "lb_stat")
        cross_tbl = _datatable(
            top.round(4),
            style_cond=[
                {"if": {"filter_query": "{lb_pvalue} < 0.05", "column_id": "lb_pvalue"},
                 "backgroundColor": "#ffd6d6"},
            ],
        )

    return html.Div([
        dcc.Graph(figure=fig_maha),
        html.Hr(),
        html.H5("Component-wise Ljung-Box on Standardised Residuals"),
        comp_tbl,
        html.Hr(),
        html.H5("Cross-Product Portmanteau — Top 20 Pairs by Statistic"),
        cross_tbl,
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Utility: reusable DataTable
# ─────────────────────────────────────────────────────────────────────────────

def _datatable(df: pd.DataFrame, style_cond: list | None = None) -> dash_table.DataTable:
    return dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df.columns],
        style_table={"overflowX": "auto"},
        style_cell={
            "fontSize": 12, "textAlign": "center",
            "padding": "4px 8px", "fontFamily": "monospace",
        },
        style_header={
            "fontWeight": "bold",
            "backgroundColor": "#2c3e50",
            "color": "white",
        },
        style_data_conditional=style_cond or [],
        page_size=20,
    )
