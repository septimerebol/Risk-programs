# Stock Portfolio Risk Model

Interactive Python app that builds a 99% confidence risk model for a stock portfolio
using the last 500 daily returns from Yahoo Finance.

## Usage

```bash
pip install -r requirements.txt
python risk_portfolio.py              # options: --csv PREFIX, --chart FILE, --show
```

1. Enter the total portfolio value.
2. Enter ticker / dollar amount pairs. The app shows what's left to allocate, rejects
   unknown tickers and amounts over the remaining balance, and stops once the
   positions add up to the total.
3. It downloads 500 days of adjusted-close returns and prints:
   - **Position risk table** – weight, volatility, 99% 1-day VaR (parametric and
     historical), CVaR, and each position's contribution to portfolio VaR, plus the
     diversification benefit.
   - **99% confidence table** – for 1 day, 1 week, 2 weeks, 1 month and 1 quarter:
     expected return, the 99% return range (lower/upper bound and the matching
     portfolio values), and the 99% VaR.
   - **Bell curve chart** (`risk_bell_curve.png`) – histogram of the 500 daily
     portfolio returns with the fitted normal curve, the 99% range shaded, the
     worst-1% tail in red, and lines for the mean and the normal vs. historical VaR.
     Skew and excess kurtosis show how far real returns depart from the bell curve.
     Use `--show` to open it in a window.

Multi-day figures scale the daily mean by *t* and volatility by √*t*.

## Tests (both programs)

```bash
pip install pytest && pytest
```

# WACC Calculator

Pulls a company's financial profile from Yahoo Finance and calculates its weighted
average cost of capital, with a breakdown of every input and where it came from.

```bash
python wacc_calculator.py AAPL        # ticker is prompted for if omitted
python wacc_calculator.py MSFT --mrp 6 --debt-method synthetic --csv msft
```

WACC = E/V × Re + D/V × Rd × (1 − t), with the cost of equity from CAPM (Re = Rf + β × MRP).

| Input | Default source | Override |
|---|---|---|
| Risk-free rate | Live 10-year US Treasury yield (`^TNX`) | `--rf 4.3` |
| Beta | Yahoo Finance 5Y monthly beta (falls back to a regression against the S&P 500) | `--beta 1.1` |
| Market risk premium | 5% | `--mrp 6` |
| Equity (E) | Market capitalisation | – |
| Debt (D) | Total debt from the latest balance sheet (book value, incl. leases) | – |
| Cost of debt | Interest expense ÷ average total debt; if that's implausible (outside 0.5–20%), Rf + a synthetic-rating spread based on interest coverage | `--cost-of-debt 5`, `--debt-method implied\|synthetic` |
| Tax rate | Effective rate (tax provision ÷ pretax income); 21% if that is negative or over 50% | `--tax-rate 25` |

The report prints:
- **Inputs**: each rate with its source.
- **Capital structure & cost of capital**: equity and debt values, weights, costs, and
  each one's contribution to WACC.
- **Cross-checks**: regression beta, implied cost of debt, and interest coverage with
  the synthetic rating, so you can see when Yahoo's figures look off.
- **Sensitivity grid**: WACC across beta ±0.2 and market risk premium ±1%.

Rates can be given as `4.5`, `4.5%` or `0.045`. The risk-free rate and premium are
US-based; for non-US companies, pass `--rf` and `--mrp` for the relevant market. The
synthetic-rating spreads are approximate Damodaran figures (Jan 2025) in
`RATING_SPREADS`; update them as needed.
