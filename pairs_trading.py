"""
High-Frequency Pairs Trading Algorithm
========================================
Cointegration + Ornstein-Uhlenbeck + Bollinger Bands
with full backtesting, slippage, and commission calculations.

Author: Generated for quantitative finance demonstration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import itertools
import yfinance as yf

from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from scipy.optimize import minimize
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Universe of tickers (SPY sector ETFs + liquid names)
    "tickers": [
        "XLK", "XLF", "XLE", "XLV", "XLI",   # Sector ETFs
        "XLY", "XLP", "XLU", "XLB", "XLRE",   # More sector ETFs
        "QQQ", "SPY", "IWM", "DIA",            # Broad market ETFs
        "EWA", "EWC",                           # Country ETFs (classic pair)
    ],
    "start_date": "2019-01-01",
    "end_date":   datetime.today().strftime("%Y-%m-%d"),

    # Cointegration
    "coint_pvalue_threshold": 0.05,

    # Ornstein-Uhlenbeck / Bollinger Bands
    "zscore_entry":  2.0,    # Enter when |z| exceeds this
    "zscore_exit":   0.5,    # Exit when |z| falls below this
    "bb_window":     30,     # Rolling window for mean/std of spread
    "ou_half_life_min": 1,   # Min half-life in days (filter too-fast reversion)
    "ou_half_life_max": 60,  # Max half-life in days (filter too-slow reversion)

    # Risk management
    "stop_loss_zscore": 4.0, # Hard stop if z-score exceeds this

    # Execution costs
    "slippage_bps":   5.0,   # Slippage in basis points per trade leg
    "commission_bps": 1.0,   # Commission in basis points per trade leg

    # Capital
    "initial_capital": 100_000,
    "position_size":   0.95, # Fraction of capital deployed per trade
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────

def fetch_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """Download adjusted closing prices for all tickers."""
    print(f"\n{'='*60}")
    print("  STEP 1: Fetching Price Data")
    print(f"{'='*60}")
    print(f"  Tickers : {tickers}")
    print(f"  Range   : {start} → {end}")

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Handle multi-level columns
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]

    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.9))
    prices = prices.ffill().dropna()

    print(f"  Loaded  : {prices.shape[0]} trading days × {prices.shape[1]} assets")
    return prices


# ─────────────────────────────────────────────────────────────────────────────
# 2. COINTEGRATION SCREENING
# ─────────────────────────────────────────────────────────────────────────────

def test_cointegration(series1: pd.Series, series2: pd.Series):
    """
    Engle-Granger cointegration test.
    Returns (p-value, hedge_ratio, spread).
    """
    y = series1.values
    x = add_constant(series2.values)
    result = OLS(y, x).fit()
    hedge_ratio = result.params[1]
    spread = series1 - hedge_ratio * series2

    _, pvalue, _ = coint(series1, series2)
    return pvalue, hedge_ratio, spread


def find_cointegrated_pairs(prices: pd.DataFrame, pvalue_threshold: float) -> list:
    """
    Scan all ticker pairs for cointegration.
    Returns sorted list of (ticker1, ticker2, pvalue, hedge_ratio).
    """
    print(f"\n{'='*60}")
    print("  STEP 2: Cointegration Screening")
    print(f"{'='*60}")

    tickers = list(prices.columns)
    pairs = list(itertools.combinations(tickers, 2))
    print(f"  Testing {len(pairs)} pairs for cointegration...")

    cointegrated = []
    for t1, t2 in pairs:
        try:
            pval, hedge, spread = test_cointegration(prices[t1], prices[t2])
            if pval < pvalue_threshold:
                cointegrated.append({
                    "ticker1":     t1,
                    "ticker2":     t2,
                    "pvalue":      round(pval, 4),
                    "hedge_ratio": round(hedge, 4),
                    "spread_mean": round(spread.mean(), 4),
                    "spread_std":  round(spread.std(), 4),
                })
        except Exception:
            continue

    cointegrated.sort(key=lambda x: x["pvalue"])

    print(f"\n  Found {len(cointegrated)} cointegrated pair(s) at p < {pvalue_threshold}:\n")
    print(f"  {'Pair':<18} {'p-value':<12} {'Hedge Ratio':<14}")
    print(f"  {'-'*44}")
    for p in cointegrated:
        pair_str = f"{p['ticker1']}/{p['ticker2']}"
        print(f"  {pair_str:<18} {p['pvalue']:<12} {p['hedge_ratio']:<14}")

    return cointegrated


# ─────────────────────────────────────────────────────────────────────────────
# 3. ORNSTEIN-UHLENBECK PARAMETER ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────

def estimate_ou_parameters(spread: pd.Series) -> dict:
    """
    Estimate Ornstein-Uhlenbeck parameters via MLE.
    dX = theta*(mu - X)*dt + sigma*dW

    Returns: theta (mean-reversion speed), mu (long-run mean),
             sigma (volatility), half_life (days).
    """
    spread = spread.dropna()
    n = len(spread)

    # OLS regression of dX on X_{t-1} (Euler-Maruyama discretization)
    x_lag = spread.shift(1).dropna()
    dx    = spread.diff().dropna()

    # Align
    x_lag = x_lag.iloc[:len(dx)]

    X = add_constant(x_lag.values)
    model = OLS(dx.values, X).fit()

    a = model.params[0]   # intercept
    b = model.params[1]   # slope (should be negative for mean-reversion)

    # OU parameters
    theta = -b                   # mean-reversion speed (per day)
    mu    = a / theta if theta != 0 else spread.mean()
    resid = dx.values - (a + b * x_lag.values)
    sigma = np.std(resid)

    half_life = np.log(2) / theta if theta > 0 else np.inf

    return {
        "theta":     theta,
        "mu":        mu,
        "sigma":     sigma,
        "half_life": half_life,
    }


def filter_pairs_by_ou(cointegrated_pairs: list, prices: pd.DataFrame,
                        min_hl: float, max_hl: float) -> list:
    """Filter pairs whose OU half-life falls in the desired range."""
    print(f"\n{'='*60}")
    print("  STEP 3: Ornstein-Uhlenbeck Parameter Estimation")
    print(f"{'='*60}")
    print(f"  Filtering for half-life in [{min_hl}, {max_hl}] days\n")

    valid = []
    for p in cointegrated_pairs:
        t1, t2 = p["ticker1"], p["ticker2"]
        spread = prices[t1] - p["hedge_ratio"] * prices[t2]
        ou = estimate_ou_parameters(spread)
        hl = ou["half_life"]

        status = "✓" if min_hl <= hl <= max_hl else "✗"
        print(f"  {status}  {t1}/{t2:<10}  θ={ou['theta']:.4f}  "
              f"half-life={hl:.1f}d  σ={ou['sigma']:.4f}")

        if min_hl <= hl <= max_hl:
            p["ou"] = ou
            valid.append(p)

    print(f"\n  {len(valid)} pair(s) passed OU filter.")
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# 4. BOLLINGER BAND SIGNAL GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_signals(spread: pd.Series, window: int,
                    entry_z: float, exit_z: float,
                    stop_loss_z: float) -> pd.DataFrame:
    """
    Generate long/short signals on the spread using Bollinger Bands (z-score).

    Signal convention:
        +1  → long spread  (spread is abnormally low)
        -1  → short spread (spread is abnormally high)
         0  → flat
    """
    rolling_mean = spread.rolling(window).mean()
    rolling_std  = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std

    signals = pd.DataFrame(index=spread.index)
    signals["spread"]       = spread
    signals["rolling_mean"] = rolling_mean
    signals["rolling_std"]  = rolling_std
    signals["zscore"]       = zscore
    signals["position"]     = 0

    position = 0
    for i in range(window, len(signals)):
        z = signals["zscore"].iloc[i]

        if position == 0:
            if z < -entry_z:
                position = 1   # Spread too low → go long
            elif z > entry_z:
                position = -1  # Spread too high → go short
        elif position == 1:
            if z > -exit_z or abs(z) > stop_loss_z:
                position = 0
        elif position == -1:
            if z < exit_z or abs(z) > stop_loss_z:
                position = 0

        signals.iloc[i, signals.columns.get_loc("position")] = position

    signals["trade"] = signals["position"].diff().fillna(0)  # Non-zero = trade event
    return signals


# ─────────────────────────────────────────────────────────────────────────────
# 5. BACKTESTING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def backtest_pair(prices: pd.DataFrame, pair_info: dict,
                  config: dict) -> dict:
    """
    Full backtest for a single pair with slippage and commissions.

    Position sizing: We hold (hedge_ratio) shares of t2 for every 1 share of t1.
    Each trade's entry/exit price is adjusted for slippage.
    Commission is deducted per leg.
    """
    t1, t2 = pair_info["ticker1"], pair_info["ticker2"]
    hedge   = pair_info["hedge_ratio"]
    capital = config["initial_capital"]
    pos_frac = config["position_size"]

    slippage_mult   = config["slippage_bps"]   / 10_000
    commission_mult = config["commission_bps"]  / 10_000

    p1 = prices[t1]
    p2 = prices[t2]
    spread = p1 - hedge * p2

    signals = compute_signals(
        spread,
        window       = config["bb_window"],
        entry_z      = config["zscore_entry"],
        exit_z       = config["zscore_exit"],
        stop_loss_z  = config["stop_loss_zscore"],
    )

    # ── Portfolio tracking ──
    portfolio = pd.DataFrame(index=signals.index)
    portfolio["price1"]   = p1
    portfolio["price2"]   = p2
    portfolio["position"] = signals["position"]
    portfolio["zscore"]   = signals["zscore"]

    cash       = capital
    shares1    = 0.0
    shares2    = 0.0
    equity     = []
    trades_log = []

    prev_pos = 0

    for i, (date, row) in enumerate(portfolio.iterrows()):
        pos = row["position"]
        px1 = row["price1"]
        px2 = row["price2"]

        if np.isnan(px1) or np.isnan(px2):
            equity.append(cash + shares1 * px1 + shares2 * px2
                          if not np.isnan(px1) else cash)
            prev_pos = pos
            continue

        if pos != prev_pos:
            # ── Close existing position ──
            if prev_pos != 0:
                # Sell with slippage (adverse)
                exit_px1 = px1 * (1 - prev_pos * slippage_mult)
                exit_px2 = px2 * (1 + prev_pos * slippage_mult)  # opposite leg

                proceeds = (shares1 * exit_px1 + shares2 * exit_px2)
                comm = abs(shares1) * px1 * commission_mult + \
                       abs(shares2) * px2 * commission_mult
                cash += proceeds - comm

                trades_log.append({
                    "date":   date,
                    "action": "close",
                    "pair":   f"{t1}/{t2}",
                    "pnl":    cash - capital,
                })
                shares1 = 0.0
                shares2 = 0.0

            # ── Open new position ──
            if pos != 0:
                # Notional per leg based on available capital
                notional = cash * pos_frac / 2

                # Shares (adjusted for slippage at entry)
                entry_px1 = px1 * (1 + pos * slippage_mult)
                entry_px2 = px2 * (1 - pos * slippage_mult)

                shares1 =  pos * notional / entry_px1
                shares2 = -pos * hedge * notional / entry_px2

                cost = shares1 * entry_px1 + abs(shares2) * entry_px2
                comm = abs(shares1) * px1 * commission_mult + \
                       abs(shares2) * px2 * commission_mult

                cash -= (cost + comm)

                trades_log.append({
                    "date":   date,
                    "action": "open",
                    "pair":   f"{t1}/{t2}",
                    "direction": "long spread" if pos == 1 else "short spread",
                })

        mark_to_market = shares1 * px1 + shares2 * px2
        equity.append(cash + mark_to_market)
        prev_pos = pos

    portfolio["equity"] = equity
    portfolio["returns"] = portfolio["equity"].pct_change()

    return {
        "pair":      f"{t1}/{t2}",
        "portfolio": portfolio,
        "signals":   signals,
        "trades":    pd.DataFrame(trades_log),
        "hedge":     hedge,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(portfolio: pd.DataFrame, initial_capital: float) -> dict:
    """Compute standard quant performance metrics."""
    eq  = portfolio["equity"].dropna()
    ret = portfolio["returns"].dropna()

    total_return  = (eq.iloc[-1] - initial_capital) / initial_capital * 100
    trading_days  = len(ret)
    annual_ret    = (1 + ret.mean()) ** 252 - 1

    sharpe  = (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() != 0 else 0

    # Max drawdown
    roll_max   = eq.cummax()
    drawdown   = (eq - roll_max) / roll_max
    max_dd     = drawdown.min() * 100

    # Calmar ratio
    calmar = (annual_ret * 100) / abs(max_dd) if max_dd != 0 else 0

    # Win rate (daily)
    win_rate = (ret > 0).mean() * 100

    return {
        "Total Return (%)":    round(total_return, 2),
        "Annual Return (%)":   round(annual_ret * 100, 2),
        "Sharpe Ratio":        round(sharpe, 3),
        "Max Drawdown (%)":    round(max_dd, 2),
        "Calmar Ratio":        round(calmar, 3),
        "Win Rate (%)":        round(win_rate, 2),
        "Trading Days":        trading_days,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG    = "#0d1117"
PANEL_BG   = "#161b22"
ACCENT     = "#58a6ff"
GREEN      = "#3fb950"
RED        = "#f85149"
YELLOW     = "#d29922"
TEXT_COLOR = "#c9d1d9"
GRID_COLOR = "#21262d"

def plot_results(result: dict, metrics: dict, config: dict):
    """Four-panel dashboard: prices, spread+BB, z-score+signals, equity curve."""
    pair     = result["pair"]
    t1, t2   = pair.split("/")
    portfolio = result["portfolio"]
    signals   = result["signals"]
    hedge     = result["hedge"]

    fig = plt.figure(figsize=(18, 14), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(4, 2, figure=fig,
                            hspace=0.45, wspace=0.3,
                            left=0.07, right=0.97,
                            top=0.90, bottom=0.06)

    def style_ax(ax, title=""):
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors=TEXT_COLOR, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.grid(color=GRID_COLOR, linewidth=0.5, alpha=0.7)
        if title:
            ax.set_title(title, color=TEXT_COLOR, fontsize=11,
                         fontweight="bold", pad=8)

    # ── Panel 1: Normalised prices ──
    ax1 = fig.add_subplot(gs[0, :])
    p1_norm = (portfolio["price1"] / portfolio["price1"].iloc[0]) * 100
    p2_norm = (portfolio["price2"] / portfolio["price2"].iloc[0]) * 100
    ax1.plot(p1_norm.index, p1_norm, color=ACCENT,  lw=1.2, label=t1)
    ax1.plot(p2_norm.index, p2_norm, color=YELLOW,  lw=1.2, label=t2)
    ax1.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9)
    style_ax(ax1, f"Normalised Prices — {t1} vs {t2}")

    # ── Panel 2: Spread + Bollinger Bands ──
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(signals.index, signals["spread"],       color=ACCENT, lw=1.0, label="Spread")
    ax2.plot(signals.index, signals["rolling_mean"], color=TEXT_COLOR, lw=0.8, ls="--", label="Mean")
    upper = signals["rolling_mean"] + config["zscore_entry"] * signals["rolling_std"]
    lower = signals["rolling_mean"] - config["zscore_entry"] * signals["rolling_std"]
    ax2.fill_between(signals.index, lower, upper,
                     color=ACCENT, alpha=0.07)
    ax2.plot(signals.index, upper, color=RED,   lw=0.7, ls="--", label=f"+{config['zscore_entry']}σ")
    ax2.plot(signals.index, lower, color=GREEN, lw=0.7, ls="--", label=f"-{config['zscore_entry']}σ")
    ax2.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=8, ncol=5)
    style_ax(ax2, "Spread with Bollinger Bands")

    # ── Panel 3: Z-score ──
    ax3 = fig.add_subplot(gs[2, :])
    zs = signals["zscore"]
    ax3.plot(zs.index, zs, color=ACCENT, lw=0.9, label="Z-score")
    ax3.axhline( config["zscore_entry"],  color=RED,   lw=0.8, ls="--")
    ax3.axhline(-config["zscore_entry"],  color=GREEN, lw=0.8, ls="--")
    ax3.axhline( config["zscore_exit"],   color=TEXT_COLOR, lw=0.6, ls=":")
    ax3.axhline(-config["zscore_exit"],   color=TEXT_COLOR, lw=0.6, ls=":")
    ax3.axhline(0, color=GRID_COLOR, lw=0.8)

    # Shade positions
    pos = signals["position"]
    ax3.fill_between(zs.index, zs, 0,
                     where=(pos == 1),  color=GREEN, alpha=0.15, label="Long spread")
    ax3.fill_between(zs.index, zs, 0,
                     where=(pos == -1), color=RED,   alpha=0.15, label="Short spread")
    ax3.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=8, ncol=4)
    style_ax(ax3, "Z-Score & Trade Signals")

    # ── Panel 4: Equity curve ──
    ax4 = fig.add_subplot(gs[3, 0])
    eq = portfolio["equity"].dropna()
    ax4.plot(eq.index, eq, color=GREEN, lw=1.2, label="Strategy")
    ax4.axhline(config["initial_capital"], color=TEXT_COLOR, lw=0.7, ls="--", label="Benchmark")
    ax4.fill_between(eq.index, config["initial_capital"], eq,
                     where=(eq >= config["initial_capital"]),
                     color=GREEN, alpha=0.15)
    ax4.fill_between(eq.index, config["initial_capital"], eq,
                     where=(eq < config["initial_capital"]),
                     color=RED, alpha=0.15)
    ax4.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax4.legend(facecolor=PANEL_BG, edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=8)
    style_ax(ax4, "Equity Curve")

    # ── Panel 5: Metrics table ──
    ax5 = fig.add_subplot(gs[3, 1])
    ax5.set_facecolor(PANEL_BG)
    ax5.axis("off")
    style_ax(ax5, "Performance Metrics")

    metric_colors = []
    rows = []
    for k, v in metrics.items():
        rows.append([k, str(v)])
        if "Return" in k or "Sharpe" in k or "Calmar" in k:
            metric_colors.append([TEXT_COLOR, GREEN if float(str(v)) > 0 else RED])
        elif "Drawdown" in k:
            metric_colors.append([TEXT_COLOR, RED if float(str(v)) < 0 else GREEN])
        else:
            metric_colors.append([TEXT_COLOR, TEXT_COLOR])

    tbl = ax5.table(
        cellText   = rows,
        colLabels  = ["Metric", "Value"],
        cellLoc    = "center",
        loc        = "center",
        bbox       = [0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(PANEL_BG if row % 2 == 0 else "#1c2128")
        cell.set_edgecolor(GRID_COLOR)
        if row == 0:
            cell.set_text_props(color=TEXT_COLOR, fontweight="bold")
        elif col == 1:
            val_str = rows[row - 1][1] if row <= len(rows) else ""
            try:
                val = float(val_str)
                if "Return" in rows[row - 1][0] or "Sharpe" in rows[row - 1][0]:
                    cell.set_text_props(color=GREEN if val >= 0 else RED)
                elif "Drawdown" in rows[row - 1][0]:
                    cell.set_text_props(color=RED)
                else:
                    cell.set_text_props(color=TEXT_COLOR)
            except ValueError:
                cell.set_text_props(color=TEXT_COLOR)
        else:
            cell.set_text_props(color=TEXT_COLOR)

    ou_info = pair_info_for(pair)
    hl_text = f"OU Half-Life: {ou_info['ou']['half_life']:.1f} days" if ou_info else ""

    fig.suptitle(
        f"Pairs Trading Strategy  ·  {pair}  ·  "
        f"Hedge Ratio: {result['hedge']:.4f}  ·  {hl_text}",
        color=TEXT_COLOR, fontsize=13, fontweight="bold", y=0.97,
    )

    filename = f"pairs_trading_{t1}_{t2}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight",
                facecolor=DARK_BG)
    print(f"\n  Chart saved → {filename}")
    plt.show()


_global_pairs_cache = []

def pair_info_for(pair_label: str):
    for p in _global_pairs_cache:
        if f"{p['ticker1']}/{p['ticker2']}" == pair_label:
            return p
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global _global_pairs_cache

    print("\n" + "═" * 60)
    print("   HIGH-FREQUENCY PAIRS TRADING ALGORITHM")
    print("   Cointegration + Ornstein-Uhlenbeck + Bollinger Bands")
    print("═" * 60)

    # 1. Fetch data
    prices = fetch_prices(
        CONFIG["tickers"],
        CONFIG["start_date"],
        CONFIG["end_date"],
    )

    # 2. Find cointegrated pairs
    cointegrated = find_cointegrated_pairs(prices, CONFIG["coint_pvalue_threshold"])
    if not cointegrated:
        print("\n  ⚠  No cointegrated pairs found. Try expanding the ticker universe.")
        return

    # 3. Filter by OU half-life
    valid_pairs = filter_pairs_by_ou(
        cointegrated, prices,
        CONFIG["ou_half_life_min"],
        CONFIG["ou_half_life_max"],
    )

    if not valid_pairs:
        print("\n  ⚠  No pairs passed the OU half-life filter.")
        print("     Proceeding with top cointegrated pair regardless...")
        best = cointegrated[0]
        spread = prices[best["ticker1"]] - best["hedge_ratio"] * prices[best["ticker2"]]
        best["ou"] = estimate_ou_parameters(spread)
        valid_pairs = [best]

    _global_pairs_cache = valid_pairs

    # 4. Backtest each valid pair
    print(f"\n{'='*60}")
    print("  STEP 4: Backtesting")
    print(f"{'='*60}")

    all_results = []
    for pair_info in valid_pairs:
        t1, t2 = pair_info["ticker1"], pair_info["ticker2"]
        print(f"\n  Backtesting {t1}/{t2}...")

        result  = backtest_pair(prices, pair_info, CONFIG)
        metrics = compute_metrics(result["portfolio"], CONFIG["initial_capital"])
        all_results.append((result, metrics, pair_info))

        print(f"\n  ── Performance: {t1}/{t2} ──")
        for k, v in metrics.items():
            print(f"     {k:<25} {v}")

        plot_results(result, metrics, CONFIG)

    # 5. Summary table
    print(f"\n{'='*60}")
    print("  STEP 5: Strategy Summary")
    print(f"{'='*60}")
    print(f"\n  {'Pair':<16} {'Total Ret%':>12} {'Sharpe':>10} {'MaxDD%':>10} {'Calmar':>10}")
    print(f"  {'-'*60}")
    for result, metrics, _ in all_results:
        p = result["pair"]
        tr = metrics["Total Return (%)"]
        sh = metrics["Sharpe Ratio"]
        dd = metrics["Max Drawdown (%)"]
        ca = metrics["Calmar Ratio"]
        print(f"  {p:<16} {tr:>12.2f} {sh:>10.3f} {dd:>10.2f} {ca:>10.3f}")

    print("\n  Done. Charts have been saved to the current directory.")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
