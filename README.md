# 📈 High-Frequency Pairs Trading Algorithm

> **Cointegration · Ornstein-Uhlenbeck · Bollinger Bands**  
> A production-grade quantitative trading strategy with full backtesting, slippage, and commission modelling.

---

## What This Does

This project implements a **statistical arbitrage pairs trading strategy** using three layers of rigour:

| Layer | Method | Purpose |
|---|---|---|
| Pair selection | **Engle-Granger Cointegration** | Identify pairs that share a long-run equilibrium |
| Mean-reversion timing | **Ornstein-Uhlenbeck process** | Estimate mean-reversion speed and filter tradeable pairs |
| Signal generation | **Bollinger Bands on the spread** | Define entry/exit z-score thresholds |
| Risk management | Hard stop-loss, position sizing | Protect against diverging pairs |
| Execution costs | Slippage + commission in bps | Realistic P&L simulation |

---

## Strategy Logic

```
For each pair (A, B) in universe:
  1. Test cointegration → reject if p-value > 0.05
  2. Fit OU model to spread → reject if half-life outside [1, 60] days
  3. Compute rolling z-score of spread (30-day window)
  4. ENTER long spread  when z-score < -2.0
     ENTER short spread when z-score > +2.0
  5. EXIT when |z-score| < 0.5
     HARD STOP when |z-score| > 4.0
```

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/pairs-trading.git
cd pairs-trading
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the strategy

```bash
python pairs_trading.py
```

The script will:
- Download 5 years of price data from Yahoo Finance
- Screen all pairs for cointegration
- Filter by OU half-life
- Run a full backtest on each valid pair
- Print performance metrics to the console
- Save a chart PNG for each pair in the current directory

---

## Configuration

All parameters live in the `CONFIG` dict at the top of `pairs_trading.py`:

```python
CONFIG = {
    "tickers": [...],           # Add/remove tickers here
    "start_date": "2019-01-01",
    "end_date":   "today",

    # Cointegration
    "coint_pvalue_threshold": 0.05,

    # OU filter
    "ou_half_life_min": 1,      # Min half-life in days
    "ou_half_life_max": 60,     # Max half-life in days

    # Signal
    "zscore_entry": 2.0,        # Enter when |z| > this
    "zscore_exit":  0.5,        # Exit when |z| < this
    "bb_window":    30,         # Rolling window (days)
    "stop_loss_zscore": 4.0,    # Hard stop

    # Execution costs
    "slippage_bps":   5.0,      # Per leg, in basis points
    "commission_bps": 1.0,      # Per leg, in basis points

    # Capital
    "initial_capital": 100_000,
    "position_size":   0.95,
}
```

---

## Output

### Console

```
════════════════════════════════════════════════════════════
   HIGH-FREQUENCY PAIRS TRADING ALGORITHM
   Cointegration + Ornstein-Uhlenbeck + Bollinger Bands
════════════════════════════════════════════════════════════

STEP 2: Cointegration Screening
  Testing 136 pairs for cointegration...
  Found 3 cointegrated pair(s) at p < 0.05:

  Pair               p-value      Hedge Ratio
  ─────────────────────────────────────────────
  EWA/EWC            0.0021       0.8734
  XLK/QQQ            0.0187       1.1203
  ...

── Performance: EWA/EWC ──
   Total Return (%)          18.43
   Annual Return (%)          3.41
   Sharpe Ratio               0.872
   Max Drawdown (%)          -6.21
   Calmar Ratio               0.549
   Win Rate (%)              53.20
```

### Charts

A 4-panel dark-theme dashboard is generated per pair:
1. Normalised price series
2. Spread with Bollinger Bands
3. Z-score with shaded trade regions
4. Equity curve + performance table

---

## Performance Metrics Explained

| Metric | Description |
|---|---|
| **Total Return** | Net P&L as % of starting capital |
| **Annual Return** | Geometrically annualised daily returns |
| **Sharpe Ratio** | Risk-adjusted return (annualised, rf=0) |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Calmar Ratio** | Annual return ÷ Max Drawdown |
| **Win Rate** | % of days with positive return |

---

## Key Concepts

### Cointegration (Engle-Granger)
Two price series are cointegrated if a linear combination is stationary. This means they share a long-run equilibrium and temporary deviations will revert — the core premise of pairs trading.

### Ornstein-Uhlenbeck Process
The spread is modelled as:
```
dX = θ(μ - X)dt + σdW
```
where θ is the **mean-reversion speed** and `ln(2)/θ` gives the **half-life** — how long it takes the spread to revert halfway to its mean. We trade only pairs with a half-life between 1 and 60 days.

### Bollinger Band Z-Score
```
z(t) = (spread(t) - μ_30d) / σ_30d
```
When |z| exceeds our threshold, the spread is statistically extreme and likely to revert.

---

## Dependencies

| Package | Purpose |
|---|---|
| `yfinance` | Historical OHLCV data from Yahoo Finance |
| `pandas` | Time series manipulation |
| `numpy` | Numerical computation |
| `statsmodels` | OLS regression, cointegration tests, ADF |
| `scipy` | OU parameter optimisation |
| `matplotlib` | Multi-panel strategy dashboard |

---

## Extending the Strategy

- **Add more tickers**: Edit `CONFIG["tickers"]` — any Yahoo Finance symbol works
- **Live trading**: Replace `yf.download` with a broker API (Alpaca, Interactive Brokers)
- **Kalman filter hedge ratio**: Replace static OLS hedge with a dynamic Kalman filter for non-stationary betas
- **Walk-forward optimisation**: Periodically re-estimate cointegration and OU parameters on rolling windows
- **Portfolio of pairs**: Combine multiple pairs with Kelly criterion position sizing

---

## Disclaimer

This code is for **educational and research purposes only**. Past backtested performance does not guarantee future results. Not financial advice.

---

## License

MIT
