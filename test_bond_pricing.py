from datetime import date

import numpy as np
import pytest

import bond_pricing as bp

SETTLE = date(2026, 3, 15)


def bond(**kw):
    args = dict(maturity=date(2036, 3, 15), coupon_rate=0.05)
    args.update(kw)
    return bp.Bond(**args)


def test_par_bond_on_coupon_date():
    b = bond()
    assert b.accrued_interest(SETTLE) == 0
    assert bp.dirty_price(b, 0.05, SETTLE) == pytest.approx(100.0)
    assert len(b.cash_flows(SETTLE)) == 20


def test_matches_closed_form_annuity():
    b = bond(coupon_rate=0.06)
    y, n = 0.07, 20
    i = y / 2
    expected = 3 * (1 - (1 + i) ** -n) / i + 100 * (1 + i) ** -n
    assert bp.clean_price(b, y, SETTLE) == pytest.approx(expected)


def test_zero_coupon():
    b = bond(coupon_rate=0.0, freq=1)
    assert bp.dirty_price(b, 0.04, SETTLE) == pytest.approx(100 / 1.04 ** 10)
    assert bp.macaulay_duration(b, 0.04, SETTLE) == pytest.approx(10.0)


def test_accrued_interest_30_360():
    b = bond()
    # 15 Mar -> 15 Jun is 90 of 180 days: half a 2.50 coupon.
    assert b.accrued_interest(date(2026, 6, 15)) == pytest.approx(1.25)
    # Clean price is continuous across settlement dates at a constant yield.
    s = date(2026, 6, 15)
    assert bp.dirty_price(b, 0.05, s) - b.accrued_interest(s) == pytest.approx(100.0, abs=0.05)


def test_month_end_schedule():
    b = bond(maturity=date(2030, 2, 28), freq=4)
    _, dates = b.schedule(date(2029, 1, 1))
    assert dates == [date(2029, 2, 28), date(2029, 5, 31), date(2029, 8, 31),
                     date(2029, 11, 30), date(2030, 2, 28)]


@pytest.mark.parametrize("dc", bp.DAY_COUNTS)
@pytest.mark.parametrize("price", [80.0, 100.0, 117.5])
def test_yield_round_trip(dc, price):
    b = bond(day_count=dc)
    s = date(2026, 7, 9)
    y = bp.yield_to_maturity(b, price, s)
    assert bp.clean_price(b, y, s) == pytest.approx(price, abs=1e-9)


def test_duration_and_convexity_match_finite_differences():
    b = bond()
    s, y, h = date(2027, 1, 20), 0.045, 1e-5
    p0, up, dn = (bp.dirty_price(b, y + d, s) for d in (0, h, -h))
    assert bp.modified_duration(b, y, s) == pytest.approx((dn - up) / (2 * h * p0), rel=1e-6)
    assert bp.convexity(b, y, s) == pytest.approx((up + dn - 2 * p0) / (h * h * p0), rel=1e-4)
    assert bp.dv01(b, y, s) == pytest.approx((dn - up) / 2 * 1e-4 / h, rel=1e-6)


def test_sensitivity_table_convexity_improves_estimate():
    b = bond()
    t = bp.sensitivity_table(b, 0.05, SETTLE)
    assert t.loc[0, "Price Chg %"] == 0
    for shift in (-200, 200):
        row = t.loc[shift]
        assert abs(row["Dur+Conv Est %"] - row["Price Chg %"]) < abs(row["Duration Est %"] - row["Price Chg %"])


def test_flat_curve_matches_continuous_yield_and_z_spread():
    b = bond()
    curve = bp.ZeroCurve((1.0, 30.0), (0.04, 0.04))
    y = 2 * (np.exp(0.04 / 2) - 1)  # semiannual equivalent of 4% continuous
    assert bp.curve_dirty_price(b, curve, SETTLE) == pytest.approx(bp.dirty_price(b, y, SETTLE))
    target = bp.curve_dirty_price(b, curve, SETTLE, spread=0.0125)
    assert bp.z_spread(b, curve, target, SETTLE) == pytest.approx(0.0125, abs=1e-9)


def test_parse_inputs():
    assert bp._parse_rate("5") == pytest.approx(0.05)
    assert bp._parse_rate("0.5%") == pytest.approx(0.005)
    assert bp._parse_rate("0.045") == pytest.approx(0.045)
    c = bp.ZeroCurve.parse("5:0.045, 1:0.04")
    assert c.tenors == (1.0, 5.0) and c.rate(3.0) == pytest.approx(0.0425)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        bond(freq=3)
    with pytest.raises(ValueError):
        bond().schedule(date(2036, 3, 15))
    with pytest.raises(ValueError):
        bp.analytics(bond(), SETTLE)


def test_cli_runs(capsys):
    bp.main(["--maturity", "2034-08-15", "--coupon", "4.25", "--settle", "2026-11-05",
             "--price", "98.5", "--curve", "1:0.039,5:0.041,10:0.044", "--cashflows"])
    out = capsys.readouterr().out
    assert "Modified duration" in out and "Z-spread" in out
