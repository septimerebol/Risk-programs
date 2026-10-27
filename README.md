# Stock Portfolio Risk Model

Interactive Python app that builds a 99% confidence risk model for a stock portfolio
using the last 500 daily returns from Yahoo Finance.

## Usage

```bash
pip install -r requirements.txt
python risk_portfolio.py              # optional: --csv myportfolio
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

Multi-day figures scale the daily mean by *t* and volatility by √*t*.

## Tests

```bash
pip install pytest && pytest
```
