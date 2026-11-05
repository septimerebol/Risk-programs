#!/usr/bin/env python3
"""Fixed-rate bond pricing model.

Prices a fixed-coupon bond from a yield (or solves the yield from a price),
splits the price into clean price and accrued interest, and reports the
standard risk measures: Macaulay and modified duration, convexity, DV01 and a
price/yield sensitivity table. It can also price the bond off a zero-coupon
curve and solve for the Z-spread over that curve.

Conventions
-----------
* Prices are quoted per 100 of face value unless stated otherwise.
* Yields use the street convention: compounded at the coupon frequency, with
  the first (partial) period discounted by the fraction of a period remaining.
* Zero-curve rates are continuously compounded and linearly interpolated
  (flat beyond the first and last tenor).
"""

import argparse
import calendar
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

DAY_COUNTS = ("30/360", "ACT/ACT", "ACT/365", "ACT/360")
FREQUENCIES = (1, 2, 4, 12)
DEFAULT_SHIFTS_BP = (-200, -100, -50, -25, 0, 25, 50, 100, 200)


# ---------------------------------------------------------------------------
# Dates and day counts
# ---------------------------------------------------------------------------

def _is_month_end(d):
    return d.day == calendar.monthrange(d.year, d.month)[1]


def add_months(d, months, month_end=False):
    """Shift a date by whole months, clamping to the month's last day."""
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    last = calendar.monthrange(year, month)[1]
    return date(year, month, last if month_end else min(d.day, last))


def days_30_360(start, end):
    """US (NASD) 30/360 day count between two dates."""
    d1, d2 = start.day, end.day
    if d1 == 31 or (start.month == 2 and _is_month_end(start)):
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return 360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)


def accrual_fraction(prev_cpn, settle, next_cpn, freq, day_count):
    """Fraction of the current coupon period that has elapsed at settlement."""
    if day_count == "30/360":
        return days_30_360(prev_cpn, settle) / days_30_360(prev_cpn, next_cpn)
    if day_count == "ACT/ACT":
        return (settle - prev_cpn).days / (next_cpn - prev_cpn).days
    if day_count == "ACT/365":
        return (settle - prev_cpn).days * freq / 365
    if day_count == "ACT/360":
        return (settle - prev_cpn).days * freq / 360
    raise ValueError(f"unknown day count {day_count!r}; use one of {DAY_COUNTS}")


# ---------------------------------------------------------------------------
# Bond
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Bond:
    """A fixed-rate bullet bond.

    coupon_rate is annual (0.05 = 5%); face is the redemption amount.
    """

    maturity: date
    coupon_rate: float
    face: float = 100.0
    freq: int = 2
    day_count: str = "30/360"

    def __post_init__(self):
        if self.freq not in FREQUENCIES:
            raise ValueError(f"frequency must be one of {FREQUENCIES}")
        if self.day_count not in DAY_COUNTS:
            raise ValueError(f"day count must be one of {DAY_COUNTS}")
        if self.coupon_rate < 0 or self.face <= 0:
            raise ValueError("coupon must be >= 0 and face > 0")

    @property
    def coupon(self):
        """Coupon amount paid each period."""
        return self.face * self.coupon_rate / self.freq

    def schedule(self, settle):
        """Previous coupon date and all coupon dates after settlement.

        Dates are generated backwards from maturity so any odd period falls at
        the start. A coupon paid on the settlement date belongs to the seller.
        """
        if settle >= self.maturity:
            raise ValueError("settlement must be before maturity")
        step = 12 // self.freq
        eom = _is_month_end(self.maturity)
        dates, k = [], 0
        while True:
            d = add_months(self.maturity, -step * k, eom)
            if d <= settle:
                return d, dates[::-1]
            dates.append(d)
            k += 1

    def cash_flows(self, settle):
        """Table of future cash flows with their timing.

        Columns: date, amount, periods (coupon periods from settlement) and
        years (periods / freq).
        """
        prev_cpn, dates = self.schedule(settle)
        accrued = accrual_fraction(prev_cpn, settle, dates[0], self.freq, self.day_count)
        w = 1.0 - accrued  # fraction of a period until the next coupon
        periods = w + np.arange(len(dates))
        amounts = np.full(len(dates), self.coupon)
        amounts[-1] += self.face
        return pd.DataFrame({"date": dates, "amount": amounts,
                             "periods": periods, "years": periods / self.freq})

    def accrued_interest(self, settle):
        prev_cpn, dates = self.schedule(settle)
        return self.coupon * accrual_fraction(prev_cpn, settle, dates[0],
                                              self.freq, self.day_count)


# ---------------------------------------------------------------------------
# Yield-based pricing and risk
# ---------------------------------------------------------------------------

def _per100(bond, value):
    return value * 100.0 / bond.face


def dirty_price(bond, ytm, settle):
    """Full (invoice) price per 100 face at an annual yield."""
    cf = bond.cash_flows(settle)
    df = (1 + ytm / bond.freq) ** -cf["periods"].to_numpy()
    return _per100(bond, float(cf["amount"].to_numpy() @ df))


def clean_price(bond, ytm, settle):
    """Quoted price per 100 face: dirty price less accrued interest."""
    return dirty_price(bond, ytm, settle) - _per100(bond, bond.accrued_interest(settle))


def yield_to_maturity(bond, price, settle, clean=True, tol=1e-12, max_iter=100):
    """Solve for the yield that reproduces a price per 100 face.

    Uses Newton's method and falls back to bisection if Newton strays.
    """
    target = price + (_per100(bond, bond.accrued_interest(settle)) if clean else 0.0)
    if target <= 0:
        raise ValueError("price must be positive")
    f = lambda y: dirty_price(bond, y, settle) - target  # noqa: E731

    y = bond.coupon_rate if bond.coupon_rate > 0 else 0.05
    for _ in range(max_iter):
        diff = f(y)
        if abs(diff) < tol:
            return y
        slope = -modified_duration(bond, y, settle) * dirty_price(bond, y, settle)
        step = diff / slope
        y_next = y - step
        if not np.isfinite(y_next) or y_next <= -bond.freq:
            break
        y = y_next
        if abs(step) < tol:
            return y

    # Bisection fallback: price falls monotonically as yield rises.
    lo, hi = -0.99 * bond.freq, 10.0
    if f(lo) * f(hi) > 0:
        raise ValueError("could not bracket the yield")
    for _ in range(500):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def macaulay_duration(bond, ytm, settle):
    """PV-weighted average time to cash flows, in years."""
    cf = bond.cash_flows(settle)
    pv = cf["amount"] * (1 + ytm / bond.freq) ** -cf["periods"]
    return float((pv * cf["years"]).sum() / pv.sum())


def modified_duration(bond, ytm, settle):
    """Percentage price change for a 1.00 (100%) change in yield."""
    return macaulay_duration(bond, ytm, settle) / (1 + ytm / bond.freq)


def convexity(bond, ytm, settle):
    """Second-order price sensitivity to yield, in years squared."""
    cf = bond.cash_flows(settle)
    n, f = cf["periods"], bond.freq
    pv = cf["amount"] * (1 + ytm / f) ** -n
    return float((pv * n * (n + 1)).sum() / pv.sum() / (f ** 2 * (1 + ytm / f) ** 2))


def dv01(bond, ytm, settle, notional=None):
    """Dollar value of a 1bp fall in yield (positive for a long position).

    Per 100 face by default, or for a given notional face amount.
    """
    value = modified_duration(bond, ytm, settle) * dirty_price(bond, ytm, settle) * 1e-4
    return value if notional is None else value * notional / 100.0


def analytics(bond, settle, ytm=None, price=None):
    """All yield-based measures for a bond, from either a yield or a clean price."""
    if (ytm is None) == (price is None):
        raise ValueError("give exactly one of ytm or price")
    if ytm is None:
        ytm = yield_to_maturity(bond, price, settle)
    prev_cpn, dates = bond.schedule(settle)
    dirty = dirty_price(bond, ytm, settle)
    accrued = _per100(bond, bond.accrued_interest(settle))
    coupon = bond.coupon_rate * 100
    return {
        "Settlement": settle,
        "Maturity": bond.maturity,
        "Previous coupon": prev_cpn,
        "Next coupon": dates[0],
        "Coupons remaining": len(dates),
        "Yield to maturity": ytm,
        "Current yield": coupon / (dirty - accrued) if coupon else 0.0,
        "Clean price": dirty - accrued,
        "Accrued interest": accrued,
        "Dirty price": dirty,
        "Macaulay duration": macaulay_duration(bond, ytm, settle),
        "Modified duration": modified_duration(bond, ytm, settle),
        "Convexity": convexity(bond, ytm, settle),
        "DV01 (per 100)": dv01(bond, ytm, settle),
    }


def sensitivity_table(bond, ytm, settle, shifts_bp=DEFAULT_SHIFTS_BP):
    """Full repricing vs. duration and duration+convexity approximations."""
    p0 = dirty_price(bond, ytm, settle)
    md = modified_duration(bond, ytm, settle)
    cx = convexity(bond, ytm, settle)
    accrued = _per100(bond, bond.accrued_interest(settle))
    rows = []
    for bp in shifts_bp:
        dy = bp / 1e4
        p = dirty_price(bond, ytm + dy, settle)
        rows.append({
            "Shift (bp)": bp,
            "Yield": ytm + dy,
            "Clean Price": p - accrued,
            "Price Chg %": p / p0 - 1,
            "Duration Est %": -md * dy + 0.0,  # avoid -0.0
            "Dur+Conv Est %": -md * dy + 0.5 * cx * dy ** 2,
        })
    return pd.DataFrame(rows).set_index("Shift (bp)")


# ---------------------------------------------------------------------------
# Curve-based pricing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ZeroCurve:
    """Continuously compounded zero rates at tenors in years."""

    tenors: tuple
    rates: tuple

    def __post_init__(self):
        if len(self.tenors) != len(self.rates) or not self.tenors:
            raise ValueError("tenors and rates must be non-empty and the same length")
        if any(np.diff(self.tenors) <= 0):
            raise ValueError("tenors must be strictly increasing")

    def rate(self, t):
        return np.interp(t, self.tenors, self.rates)

    def discount(self, t, spread=0.0):
        t = np.asarray(t, dtype=float)
        return np.exp(-(self.rate(t) + spread) * t)

    @classmethod
    def parse(cls, text):
        """Parse '0.5:0.04,1:0.042,5:0.045' (tenor:rate pairs)."""
        pairs = sorted(tuple(map(float, p.split(":"))) for p in text.split(",") if p.strip())
        return cls(tuple(t for t, _ in pairs), tuple(r for _, r in pairs))


def curve_dirty_price(bond, curve, settle, spread=0.0):
    """Dirty price per 100 face discounting each cash flow off the zero curve."""
    cf = bond.cash_flows(settle)
    df = curve.discount(cf["years"].to_numpy(), spread)
    return _per100(bond, float(cf["amount"].to_numpy() @ df))


def z_spread(bond, curve, price, settle, clean=True, tol=1e-12):
    """Constant spread over the zero curve that reproduces a market price."""
    target = price + (_per100(bond, bond.accrued_interest(settle)) if clean else 0.0)
    f = lambda s: curve_dirty_price(bond, curve, settle, s) - target  # noqa: E731
    lo, hi = -0.5, 0.5
    while f(hi) > 0 and hi < 10:
        hi *= 2
    if f(lo) * f(hi) > 0:
        raise ValueError("could not bracket the Z-spread")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Reporting / CLI
# ---------------------------------------------------------------------------

def _parse_date(text):
    return date.fromisoformat(text)


def _parse_rate(text):
    """Accept '5', '5%' or '0.05' as five percent."""
    text = text.strip()
    value = float(text.rstrip("%"))
    return value / 100 if text.endswith("%") or abs(value) >= 1 else value


def format_analytics(stats, notional):
    pct = {"Yield to maturity", "Current yield"}
    lines = []
    for k, v in stats.items():
        if k in pct:
            s = f"{v:.4%}"
        elif isinstance(v, float):
            s = f"{v:,.4f}"
        else:
            s = str(v)
        lines.append(f"  {k:<20} {s:>14}")
    scale = notional / 100.0
    lines.append(f"  {'Position value':<20} {stats['Dirty price'] * scale:>14,.2f}")
    lines.append(f"  {'Position DV01':<20} {stats['DV01 (per 100)'] * scale:>14,.2f}")
    return "\n".join(lines)


def format_table(df):
    out = df.copy()
    for col in out.columns:
        if col == "Clean Price":
            out[col] = out[col].map("{:,.4f}".format)
        else:
            out[col] = out[col].map("{:.3%}".format)
    return out.to_string()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--maturity", type=_parse_date, required=True, help="YYYY-MM-DD")
    parser.add_argument("--coupon", type=_parse_rate, required=True,
                        help="annual coupon rate: 4.5, 4.5%% or 0.045 (use %% below 1%%)")
    parser.add_argument("--settle", type=_parse_date, default=date.today(),
                        help="settlement date YYYY-MM-DD (default: today)")
    parser.add_argument("--face", type=float, default=100.0, help="face value per bond")
    parser.add_argument("--notional", type=float, default=1_000_000,
                        help="position face amount for $ value and DV01")
    parser.add_argument("--freq", type=int, default=2, choices=FREQUENCIES)
    parser.add_argument("--day-count", default="30/360", choices=DAY_COUNTS)
    quote = parser.add_mutually_exclusive_group(required=True)
    quote.add_argument("--ytm", type=_parse_rate, help="yield to maturity")
    quote.add_argument("--price", type=float, help="clean price per 100 face")
    parser.add_argument("--curve", type=ZeroCurve.parse,
                        help="zero curve as tenor:rate pairs, e.g. 1:0.04,5:0.042,10:0.045")
    parser.add_argument("--cashflows", action="store_true", help="print the cash-flow schedule")
    args = parser.parse_args(argv)

    bond = Bond(args.maturity, args.coupon, args.face, args.freq, args.day_count)
    stats = analytics(bond, args.settle, ytm=args.ytm, price=args.price)
    ytm = stats["Yield to maturity"]

    print(f"\n{bond.coupon_rate:.3%} bond maturing {bond.maturity} "
          f"({bond.freq}x/yr, {bond.day_count})")
    print("\nPricing and risk")
    print(format_analytics(stats, args.notional))

    if args.curve is not None:
        curve_px = curve_dirty_price(bond, args.curve, args.settle) - stats["Accrued interest"]
        zs = z_spread(bond, args.curve, stats["Clean price"], args.settle)
        print("\nZero-curve valuation")
        print(f"  {'Curve clean price':<20} {curve_px:>14,.4f}")
        print(f"  {'Rich(+)/cheap(-)':<20} {stats['Clean price'] - curve_px:>14,.4f}")
        print(f"  {'Z-spread (bp)':<20} {zs * 1e4:>14,.2f}")

    print("\nPrice / yield sensitivity")
    print(format_table(sensitivity_table(bond, ytm, args.settle)))

    if args.cashflows:
        cf = bond.cash_flows(args.settle)
        cf["pv"] = cf["amount"] * (1 + ytm / bond.freq) ** -cf["periods"]
        print("\nCash flows (per bond)")
        print(cf.to_string(index=False, float_format="{:,.4f}".format))
    print()


if __name__ == "__main__":
    main()
