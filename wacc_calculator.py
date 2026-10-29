#!/usr/bin/env python3
"""WACC calculator.

Pulls a company's financial profile from Yahoo Finance (market cap, beta,
debt, interest expense, taxes) and the 10-year Treasury yield, then builds
the weighted average cost of capital with a full breakdown of every input.

    WACC = E/V * Re + D/V * Rd * (1 - t)
    Re   = Rf + beta * MRP          (CAPM)
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

MARKET_RISK_PREMIUM = 0.05   # long-run US equity risk premium
STATUTORY_TAX_RATE = 0.21    # US federal rate, used when the effective rate is unusable
DEFAULT_RISK_FREE = 0.0425   # used only if the 10-year yield can't be downloaded
MARKET_INDEX = "^GSPC"
TREASURY_10Y = "^TNX"        # quoted in percent, e.g. 4.25
BETA_YEARS = 5

# Interest coverage (EBIT / interest) -> synthetic rating and default spread over
# the risk-free rate, for large non-financial firms. Approximate Damodaran
# figures (Jan 2025); update them from pages.stern.nyu.edu/~adamodar as needed.
RATING_SPREADS = [
    (8.50, "AAA", 0.0059),
    (6.50, "AA", 0.0070),
    (5.50, "A+", 0.0092),
    (4.25, "A", 0.0107),
    (3.00, "A-", 0.0121),
    (2.50, "BBB", 0.0147),
    (2.25, "BB+", 0.0183),
    (2.00, "BB", 0.0221),
    (1.75, "B+", 0.0275),
    (1.50, "B", 0.0321),
    (1.25, "B-", 0.0396),
    (0.80, "CCC", 0.0529),
    (0.65, "CC", 0.0851),
    (0.20, "C", 0.1160),
    (-np.inf, "D", 0.1714),
]

# An implied cost of debt outside this band usually means bad data (e.g. debt
# raised or repaid mid-year), so fall back to the synthetic rating.
PLAUSIBLE_DEBT_COST = (0.005, 0.20)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _latest(frame, *rows, column=0):
    """First available value among `rows` in a yfinance statement, or None."""
    if frame is None or frame.empty or column >= frame.shape[1]:
        return None
    for row in rows:
        if row in frame.index:
            value = frame.loc[row].iloc[column]
            if pd.notna(value):
                return float(value)
    return None


def fetch_profile(ticker):
    """Download the financial-profile fields WACC needs for one ticker."""
    t = yf.Ticker(ticker)
    info = t.info or {}
    income = t.income_stmt
    balance = t.balance_sheet

    market_cap = info.get("marketCap")
    if not market_cap:
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        shares = info.get("sharesOutstanding")
        market_cap = price * shares if price and shares else None

    interest = _latest(income, "Interest Expense", "Interest Expense Non Operating")
    return {
        "ticker": ticker,
        "name": info.get("longName") or info.get("shortName") or ticker,
        "currency": info.get("financialCurrency") or info.get("currency") or "USD",
        "market_cap": market_cap,
        "beta": info.get("beta"),
        "total_debt": _latest(balance, "Total Debt"),
        "prior_debt": _latest(balance, "Total Debt", column=1),
        "interest_expense": abs(interest) if interest is not None else None,
        "ebit": _latest(income, "EBIT", "Operating Income"),
        "tax_provision": _latest(income, "Tax Provision"),
        "pretax_income": _latest(income, "Pretax Income"),
        "fiscal_year_end": (balance.columns[0].date()
                            if balance is not None and not balance.empty else None),
    }


def fetch_risk_free():
    """Latest 10-year US Treasury yield as a decimal, or None if unavailable."""
    try:
        close = yf.Ticker(TREASURY_10Y).history(period="5d")["Close"].dropna()
    except Exception:
        return None
    return float(close.iloc[-1]) / 100 if not close.empty else None


def fetch_monthly_returns(ticker, years=BETA_YEARS):
    """Monthly returns of the ticker and the market index over `years` years."""
    prices = yf.download([ticker, MARKET_INDEX], period=f"{years}y", interval="1mo",
                         auto_adjust=True, progress=False)["Close"]
    returns = prices[[ticker, MARKET_INDEX]].pct_change().dropna()
    return returns[ticker], returns[MARKET_INDEX]


# ---------------------------------------------------------------------------
# Estimates
# ---------------------------------------------------------------------------

def regression_beta(stock_returns, market_returns):
    """OLS beta of stock returns on market returns."""
    aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
    s, m = aligned.iloc[:, 0], aligned.iloc[:, 1]
    return float(np.cov(s, m)[0, 1] / m.var())


def synthetic_rating(ebit, interest_expense):
    """Rating and default spread implied by interest coverage."""
    if not interest_expense:
        coverage = np.inf
    else:
        coverage = (ebit or 0.0) / interest_expense
    for floor, rating, spread in RATING_SPREADS:
        if coverage > floor:
            return rating, spread, coverage
    return RATING_SPREADS[-1][1], RATING_SPREADS[-1][2], coverage


def effective_tax_rate(tax_provision, pretax_income):
    """Effective tax rate, or None when it's negative, above 50% or not computable."""
    if tax_provision is None or not pretax_income or pretax_income <= 0:
        return None
    rate = tax_provision / pretax_income
    return rate if 0 <= rate <= 0.5 else None


def implied_debt_cost(interest_expense, total_debt, prior_debt=None):
    """Interest expense divided by average total debt."""
    if not interest_expense or not total_debt:
        return None
    avg_debt = (total_debt + prior_debt) / 2 if prior_debt else total_debt
    return interest_expense / avg_debt


def build_inputs(profile, risk_free, mrp=MARKET_RISK_PREMIUM, beta=None,
                 regression=None, tax_rate=None, cost_of_debt=None,
                 debt_method="auto"):
    """Choose each WACC input and record where it came from.

    `regression` is the regression beta (or None); it's used when `beta` isn't
    given and Yahoo has no beta, and is otherwise shown for comparison.
    """
    if not profile.get("market_cap"):
        raise ValueError(f"No market capitalisation available for {profile['ticker']}.")

    inputs = {"risk_free": (risk_free, "10Y Treasury yield"),
              "mrp": (mrp, "Market risk premium")}

    if beta is not None:
        inputs["beta"] = (beta, "User override")
    elif profile.get("beta") is not None:
        inputs["beta"] = (profile["beta"], f"Yahoo Finance ({BETA_YEARS}Y monthly)")
    elif regression is not None:
        inputs["beta"] = (regression, f"Regression vs S&P 500 ({BETA_YEARS}Y monthly)")
    else:
        raise ValueError(f"No beta available for {profile['ticker']}; pass --beta.")

    if tax_rate is not None:
        inputs["tax_rate"] = (tax_rate, "User override")
    else:
        eff = effective_tax_rate(profile.get("tax_provision"), profile.get("pretax_income"))
        inputs["tax_rate"] = ((eff, "Effective (tax provision / pretax income)")
                              if eff is not None else
                              (STATUTORY_TAX_RATE, "US statutory (effective rate unusable)"))

    rating, spread, coverage = synthetic_rating(profile.get("ebit"),
                                                profile.get("interest_expense"))
    implied = implied_debt_cost(profile.get("interest_expense"), profile.get("total_debt"),
                                profile.get("prior_debt"))
    synthetic = risk_free + spread
    plausible = implied is not None and PLAUSIBLE_DEBT_COST[0] <= implied <= PLAUSIBLE_DEBT_COST[1]
    if cost_of_debt is not None:
        inputs["cost_of_debt"] = (cost_of_debt, "User override")
    elif debt_method == "implied" or (debt_method == "auto" and plausible):
        if implied is None:
            raise ValueError("Implied cost of debt unavailable (no interest expense or debt).")
        inputs["cost_of_debt"] = (implied, "Implied (interest expense / avg total debt)")
    else:
        inputs["cost_of_debt"] = (synthetic, f"Synthetic rating {rating}: Rf + {spread:.2%}")

    inputs["equity"] = (profile["market_cap"], "Market capitalisation")
    inputs["debt"] = (profile.get("total_debt") or 0.0, "Total debt (book, incl. leases)")
    checks = {"regression_beta": regression, "implied_cost_of_debt": implied,
              "synthetic_cost_of_debt": synthetic, "rating": rating,
              "interest_coverage": coverage}
    return inputs, checks


# ---------------------------------------------------------------------------
# WACC
# ---------------------------------------------------------------------------

def compute_wacc(risk_free, beta, mrp, cost_of_debt, tax_rate, equity, debt):
    """WACC and its components as a dict."""
    cost_of_equity = risk_free + beta * mrp
    after_tax_debt = cost_of_debt * (1 - tax_rate)
    total = equity + debt
    we, wd = equity / total, debt / total
    return {
        "cost_of_equity": cost_of_equity,
        "after_tax_cost_of_debt": after_tax_debt,
        "equity_weight": we,
        "debt_weight": wd,
        "equity_contribution": we * cost_of_equity,
        "debt_contribution": wd * after_tax_debt,
        "wacc": we * cost_of_equity + wd * after_tax_debt,
        "debt_to_equity": debt / equity,
    }


def _values(inputs):
    return {k: v for k, (v, _) in inputs.items()}


def breakdown_tables(inputs):
    """Inputs table, capital-structure / cost table, and the WACC result."""
    v = _values(inputs)
    result = compute_wacc(**v)

    inputs_table = pd.DataFrame([
        ("Risk-free rate (Rf)", f"{v['risk_free']:.2%}", inputs["risk_free"][1]),
        ("Beta", f"{v['beta']:.2f}", inputs["beta"][1]),
        ("Market risk premium", f"{v['mrp']:.2%}", inputs["mrp"][1]),
        ("Cost of equity (Re)", f"{result['cost_of_equity']:.2%}", "Rf + beta x MRP"),
        ("Pre-tax cost of debt (Rd)", f"{v['cost_of_debt']:.2%}", inputs["cost_of_debt"][1]),
        ("Tax rate", f"{v['tax_rate']:.2%}", inputs["tax_rate"][1]),
        ("After-tax cost of debt", f"{result['after_tax_cost_of_debt']:.2%}", "Rd x (1 - tax)"),
    ], columns=["Input", "Value", "Source"]).set_index("Input")

    structure = pd.DataFrame({
        "Value": [v["equity"], v["debt"], v["equity"] + v["debt"]],
        "Weight": [result["equity_weight"], result["debt_weight"], 1.0],
        "Cost": [result["cost_of_equity"], result["after_tax_cost_of_debt"], result["wacc"]],
        "Contribution": [result["equity_contribution"], result["debt_contribution"],
                         result["wacc"]],
    }, index=pd.Index(["Equity", "Debt", "WACC"], name="Component"))
    return inputs_table, structure, result


def sensitivity_table(inputs, beta_steps=(-0.2, -0.1, 0, 0.1, 0.2),
                      mrp_steps=(-0.01, -0.005, 0, 0.005, 0.01)):
    """WACC across a grid of beta (rows) and market risk premium (columns)."""
    v = _values(inputs)
    grid = {}
    for dm in mrp_steps:
        mrp = v["mrp"] + dm
        grid[f"MRP {mrp:.1%}"] = {
            f"Beta {v['beta'] + db:.2f}": compute_wacc(**{**v, "beta": v["beta"] + db,
                                                          "mrp": mrp})["wacc"]
            for db in beta_steps
        }
    return pd.DataFrame(grid)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _money(x, currency):
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(x) >= div:
            return f"{x / div:,.2f}{suffix} {currency}"
    return f"{x:,.0f} {currency}"


def print_report(profile, inputs, checks):
    inputs_table, structure, result = breakdown_tables(inputs)
    currency = profile["currency"]

    print(f"\n{profile['name']} ({profile['ticker']})")
    if profile.get("fiscal_year_end"):
        print(f"Latest fiscal year end: {profile['fiscal_year_end']}   Currency: {currency}")

    print("\n=== Inputs ===")
    print(inputs_table.to_string())

    print("\n=== Capital Structure & Cost of Capital ===")
    out = structure.astype(object)
    out["Value"] = structure["Value"].map(lambda x: _money(x, currency))
    for c in ("Weight", "Cost", "Contribution"):
        out[c] = structure[c].map(lambda x: f"{x:.2%}")
    print(out.to_string())
    print(f"\nDebt / Equity: {result['debt_to_equity']:.2f}")

    print("\n=== Cross-checks ===")
    if checks["regression_beta"] is not None:
        print(f"Regression beta vs S&P 500:    {checks['regression_beta']:.2f}")
    if checks["implied_cost_of_debt"] is not None:
        print(f"Implied cost of debt:          {checks['implied_cost_of_debt']:.2%}")
    coverage = checks["interest_coverage"]
    coverage_text = "n/a" if np.isinf(coverage) else f"{coverage:.1f}x"
    print(f"Interest coverage (EBIT/int.): {coverage_text}  -> synthetic rating "
          f"{checks['rating']}, cost of debt {checks['synthetic_cost_of_debt']:.2%}")

    print("\n=== WACC Sensitivity (beta x market risk premium) ===")
    print(sensitivity_table(inputs).map(lambda x: f"{x:.2%}").to_string())

    print(f"\nWACC = {result['equity_weight']:.1%} x {result['cost_of_equity']:.2%}"
          f" + {result['debt_weight']:.1%} x {result['after_tax_cost_of_debt']:.2%}"
          f" = {result['wacc']:.2%}")
    return inputs_table, structure, result


def parse_rate(text):
    """Parse '4.5', '4.5%' or '0.045' into a decimal rate."""
    value = float(str(text).replace("%", "").strip())
    return value / 100 if value >= 1 or str(text).strip().endswith("%") else value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ticker", nargs="?", help="ticker symbol, e.g. AAPL (prompted if omitted)")
    parser.add_argument("--rf", type=parse_rate, metavar="RATE",
                        help="risk-free rate (default: live 10Y Treasury yield)")
    parser.add_argument("--mrp", type=parse_rate, default=MARKET_RISK_PREMIUM, metavar="RATE",
                        help="market risk premium (default: %(default).1%%)")
    parser.add_argument("--beta", type=float, help="override beta")
    parser.add_argument("--tax-rate", type=parse_rate, metavar="RATE",
                        help="override tax rate (default: effective rate)")
    parser.add_argument("--cost-of-debt", type=parse_rate, metavar="RATE",
                        help="override pre-tax cost of debt")
    parser.add_argument("--debt-method", choices=["auto", "implied", "synthetic"],
                        default="auto",
                        help="how to estimate cost of debt (default: %(default)s)")
    parser.add_argument("--csv", metavar="PREFIX",
                        help="also save PREFIX_inputs.csv and PREFIX_wacc.csv")
    args = parser.parse_args(argv)

    ticker = (args.ticker or input("Ticker symbol: ")).strip().upper()
    if not ticker:
        sys.exit("No ticker given.")

    print(f"Downloading {ticker} financial profile from Yahoo Finance...")
    try:
        profile = fetch_profile(ticker)
    except Exception as exc:
        sys.exit(f"Could not download data for {ticker}: {exc}")

    risk_free = args.rf
    if risk_free is None:
        risk_free = fetch_risk_free()
        if risk_free is None:
            print(f"Could not download the 10Y yield; using {DEFAULT_RISK_FREE:.2%}.")
            risk_free = DEFAULT_RISK_FREE

    try:
        regression = regression_beta(*fetch_monthly_returns(ticker))
    except Exception:
        regression = None

    try:
        inputs, checks = build_inputs(profile, risk_free, mrp=args.mrp, beta=args.beta,
                                      regression=regression, tax_rate=args.tax_rate,
                                      cost_of_debt=args.cost_of_debt,
                                      debt_method=args.debt_method)
    except ValueError as exc:
        sys.exit(str(exc))
    if args.rf is not None:
        inputs["risk_free"] = (args.rf, "User override")
    if args.mrp != MARKET_RISK_PREMIUM:
        inputs["mrp"] = (args.mrp, "User override")

    inputs_table, structure, _ = print_report(profile, inputs, checks)
    if args.csv:
        inputs_table.to_csv(f"{args.csv}_inputs.csv")
        structure.to_csv(f"{args.csv}_wacc.csv")
        print(f"\nSaved {args.csv}_inputs.csv and {args.csv}_wacc.csv")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
