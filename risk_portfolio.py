#!/usr/bin/env python3
"""Stock portfolio risk model.

Prompts for a total portfolio value, then for individual positions until the
positions add up to the total. Pulls the last 500 daily returns for each
position from Yahoo Finance and builds 99% confidence (Value-at-Risk) tables.
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

LOOKBACK_DAYS = 500
CONFIDENCE = 0.99
HORIZONS = {"1 Day": 1, "1 Week": 5, "2 Weeks": 10, "1 Month": 21, "1 Quarter": 63}
TOLERANCE = 0.005  # half a cent


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def parse_money(text):
    """Parse a dollar amount such as '$12,500.50' into a float."""
    value = float(text.replace("$", "").replace(",", "").strip())
    if value <= 0:
        raise ValueError("amount must be positive")
    return round(value, 2)


def ask_money(prompt):
    while True:
        try:
            return parse_money(input(prompt))
        except ValueError:
            print("  Please enter a positive dollar amount (e.g. 100000 or $100,000).")


def is_valid_ticker(ticker):
    """Return True if Yahoo Finance has recent price history for the ticker."""
    try:
        hist = yf.Ticker(ticker).history(period="5d")
    except Exception:
        return False
    return not hist.empty


def collect_positions(total, validate=is_valid_ticker):
    """Ask for positions until their amounts sum to the total portfolio value."""
    positions = {}
    while True:
        allocated = sum(positions.values())
        remaining = round(total - allocated, 2)
        if remaining <= TOLERANCE:
            return positions

        print(f"\nAllocated: ${allocated:,.2f}   Remaining: ${remaining:,.2f}")
        ticker = input("Ticker symbol: ").strip().upper()
        if not ticker:
            continue
        if ticker not in positions and not validate(ticker):
            print(f"  Could not find price data for '{ticker}' on Yahoo Finance.")
            continue

        amount = ask_money(f"Dollar amount in {ticker} (max ${remaining:,.2f}): ")
        if amount > remaining + TOLERANCE:
            print(f"  ${amount:,.2f} exceeds the remaining ${remaining:,.2f}. Try again.")
            continue
        positions[ticker] = round(positions.get(ticker, 0.0) + amount, 2)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_returns(tickers, days=LOOKBACK_DAYS):
    """Download adjusted closes and return the last `days` daily simple returns."""
    # ~252 trading days per year; pad the calendar window so we get enough rows.
    calendar_days = int(days * 365 / 252) + 60
    prices = yf.download(
        tickers,
        period=f"{calendar_days}d",
        auto_adjust=True,
        progress=False,
    )["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices = prices[tickers].dropna()

    returns = prices.pct_change().dropna().tail(days)
    if len(returns) < days:
        print(f"\nWarning: only {len(returns)} days of overlapping history available "
              f"(requested {days}).")
    return returns


# ---------------------------------------------------------------------------
# Risk model
# ---------------------------------------------------------------------------

def position_table(returns, positions, confidence=CONFIDENCE):
    """Per-position and portfolio 1-day risk statistics at the given confidence."""
    tickers = list(positions)
    values = pd.Series(positions)[tickers]
    total = values.sum()
    weights = values / total
    z = norm.ppf(confidence)
    alpha = 1 - confidence

    port_returns = returns[tickers] @ weights
    cov = returns[tickers].cov()
    port_vol = np.sqrt(weights @ cov @ weights)
    # Component VaR: each position's share of diversified portfolio VaR.
    marginal = (cov @ weights) / port_vol
    component_pct = weights * marginal / port_vol

    rows = []
    for t in tickers:
        r = returns[t]
        rows.append(_risk_row(t, values[t], weights[t], r, z, alpha,
                              contribution=component_pct[t]))
    rows.append(_risk_row("PORTFOLIO", total, 1.0, port_returns, z, alpha,
                          contribution=1.0))

    table = pd.DataFrame(rows).set_index("Ticker")
    undiversified = table.loc[tickers, "Param VaR $"].sum()
    return table, undiversified


def _risk_row(name, value, weight, r, z, alpha, contribution):
    mu, sigma = r.mean(), r.std()
    hist_cutoff = r.quantile(alpha)
    param_var = -(mu - z * sigma)
    hist_var = -hist_cutoff
    cvar = -r[r <= hist_cutoff].mean()
    return {
        "Ticker": name,
        "Value $": value,
        "Weight": weight,
        "Mean Daily": mu,
        "Daily Vol": sigma,
        "Annual Vol": sigma * np.sqrt(252),
        "Param VaR %": param_var,
        "Param VaR $": param_var * value,
        "Hist VaR %": hist_var,
        "Hist VaR $": hist_var * value,
        "CVaR $": cvar * value,
        "VaR Contrib": contribution,
    }


def confidence_table(port_returns, total, confidence=CONFIDENCE, horizons=HORIZONS):
    """Two-sided confidence interval and one-sided VaR of portfolio returns by horizon."""
    mu, sigma = port_returns.mean(), port_returns.std()
    z_two = norm.ppf(1 - (1 - confidence) / 2)  # 2.576 for 99%
    z_one = norm.ppf(confidence)                # 2.326 for 99%

    rows = []
    for label, days in horizons.items():
        m, s = mu * days, sigma * np.sqrt(days)
        low, high = m - z_two * s, m + z_two * s
        var = -(m - z_one * s)
        rows.append({
            "Horizon": label,
            "Days": days,
            "Expected Return": m,
            "Volatility": s,
            "Lower Bound": low,
            "Upper Bound": high,
            "Lower Value $": total * (1 + low),
            "Upper Value $": total * (1 + high),
            "VaR %": var,
            "VaR $": var * total,
        })
    return pd.DataFrame(rows).set_index("Horizon")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(df, pct_cols, money_cols):
    out = df.copy().astype(object)
    for c in pct_cols:
        out[c] = df[c].map(lambda v: f"{v:.2%}")
    for c in money_cols:
        out[c] = df[c].map(lambda v: f"${v:,.0f}")
    return out


def print_report(returns, positions, confidence=CONFIDENCE):
    pct = f"{confidence:.0%}"
    table, undiversified = position_table(returns, positions, confidence)
    total = table.loc["PORTFOLIO", "Value $"]
    weights = pd.Series(positions) / total
    port_returns = returns[list(positions)] @ weights[list(positions)]
    horizon = confidence_table(port_returns, total, confidence)

    print(f"\nData: {len(returns)} daily returns, "
          f"{returns.index[0].date()} to {returns.index[-1].date()}")

    print(f"\n=== {pct} 1-Day Position Risk ===")
    print(_fmt(table,
               ["Weight", "Mean Daily", "Daily Vol", "Annual Vol",
                "Param VaR %", "Hist VaR %", "VaR Contrib"],
               ["Value $", "Param VaR $", "Hist VaR $", "CVaR $"]).to_string())
    diversified = table.loc["PORTFOLIO", "Param VaR $"]
    print(f"\nSum of individual VaRs:  ${undiversified:,.0f}")
    print(f"Diversified VaR:         ${diversified:,.0f}")
    print(f"Diversification benefit: ${undiversified - diversified:,.0f}")

    print(f"\n=== {pct} Confidence Table: Portfolio Returns by Horizon ===")
    print(_fmt(horizon,
               ["Expected Return", "Volatility", "Lower Bound", "Upper Bound", "VaR %"],
               ["Lower Value $", "Upper Value $", "VaR $"]).to_string())

    print(f"\nInterpretation: with {pct} confidence the portfolio's 1-day loss "
          f"should not exceed ${horizon.loc['1 Day', 'VaR $']:,.0f} "
          f"({horizon.loc['1 Day', 'VaR %']:.2%}), and its 1-day return should fall "
          f"between {horizon.loc['1 Day', 'Lower Bound']:.2%} and "
          f"{horizon.loc['1 Day', 'Upper Bound']:.2%}.")
    print("Param = variance-covariance (normal) model; Hist = historical simulation; "
          "CVaR = average loss beyond historical VaR.")
    return table, horizon


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", metavar="PREFIX",
                        help="also save tables to PREFIX_positions.csv and PREFIX_confidence.csv")
    args = parser.parse_args(argv)

    print("Stock Portfolio Risk Model")
    print("--------------------------")
    total = ask_money("Total portfolio value ($): ")
    positions = collect_positions(total)

    print("\nPositions:")
    for t, v in positions.items():
        print(f"  {t:<8} ${v:>14,.2f}  ({v / total:.2%})")

    print(f"\nDownloading {LOOKBACK_DAYS} days of returns from Yahoo Finance...")
    returns = fetch_returns(list(positions))
    if returns.empty:
        sys.exit("No return data downloaded; check your tickers and connection.")

    table, horizon = print_report(returns, positions)
    if args.csv:
        table.to_csv(f"{args.csv}_positions.csv")
        horizon.to_csv(f"{args.csv}_confidence.csv")
        print(f"\nSaved {args.csv}_positions.csv and {args.csv}_confidence.csv")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
