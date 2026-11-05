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

## Bond Pricing Model

`bond_pricing.py` prices a fixed-rate bond and reports its risk measures.

```bash
python bond_pricing.py --maturity 2034-08-15 --coupon 4.25 --settle 2026-11-05 --price 98.5 \
    --curve 1:0.039,5:0.041,10:0.044 --cashflows
```

- Give either `--ytm` or a clean `--price` (per 100 face). The other is solved for.
- Options: `--freq` (1, 2, 4, 12), `--day-count` (30/360, ACT/ACT, ACT/365, ACT/360),
  `--face`, `--notional` (position size for $ value and DV01).
- Output: clean/dirty price and accrued interest, yield to maturity, current yield,
  Macaulay and modified duration, convexity, DV01, and a ±200bp price/yield table
  comparing full repricing with the duration and duration+convexity estimates.
- `--curve` takes continuously compounded zero rates (`tenor:rate` pairs, linearly
  interpolated). The bond is repriced off the curve, shown as rich or cheap against
  its market price, and the Z-spread is solved for.
- Coupon dates are generated backwards from maturity (month-end dates stay at month
  end). Yields use the street convention, compounded at the coupon frequency.

The functions (`Bond`, `dirty_price`, `yield_to_maturity`, `modified_duration`,
`convexity`, `dv01`, `ZeroCurve`, `z_spread`, ...) can also be imported and used directly.

## Tests

```bash
pip install pytest && pytest
```
