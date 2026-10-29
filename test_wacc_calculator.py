import numpy as np
import pandas as pd
import pytest

import wacc_calculator as wc


def profile(**overrides):
    base = {
        "ticker": "TEST",
        "name": "Test Corp",
        "currency": "USD",
        "market_cap": 800e9,
        "beta": 1.2,
        "total_debt": 200e9,
        "prior_debt": 180e9,
        "interest_expense": 9.5e9,
        "ebit": 95e9,
        "tax_provision": 15e9,
        "pretax_income": 100e9,
        "fiscal_year_end": None,
    }
    return {**base, **overrides}


def test_compute_wacc_by_hand():
    r = wc.compute_wacc(risk_free=0.04, beta=1.2, mrp=0.055, cost_of_debt=0.05,
                        tax_rate=0.15, equity=800, debt=200)
    assert r["cost_of_equity"] == pytest.approx(0.106)
    assert r["after_tax_cost_of_debt"] == pytest.approx(0.0425)
    assert r["wacc"] == pytest.approx(0.8 * 0.106 + 0.2 * 0.0425)
    assert r["equity_contribution"] + r["debt_contribution"] == pytest.approx(r["wacc"])


def test_build_inputs_uses_profile_data():
    inputs, checks = wc.build_inputs(profile(), risk_free=0.04)
    assert inputs["beta"][0] == 1.2
    assert inputs["tax_rate"][0] == pytest.approx(0.15)
    assert inputs["cost_of_debt"][0] == pytest.approx(9.5e9 / 190e9)  # avg debt
    assert "Implied" in inputs["cost_of_debt"][1]
    assert checks["rating"] == "AAA"  # coverage 10x


def test_implausible_implied_rate_falls_back_to_synthetic():
    # Interest of 30% of debt is bad data -> synthetic rating instead.
    inputs, checks = wc.build_inputs(profile(interest_expense=57e9, ebit=114e9),
                                     risk_free=0.04)
    assert checks["rating"] == "B+"  # coverage exactly 2.0x
    assert "Synthetic" in inputs["cost_of_debt"][1]
    assert inputs["cost_of_debt"][0] == pytest.approx(checks["synthetic_cost_of_debt"])


def test_synthetic_rating_bands():
    assert wc.synthetic_rating(100, 10)[0] == "AAA"
    assert wc.synthetic_rating(35, 10)[0] == "A-"
    assert wc.synthetic_rating(-5, 10)[0] == "D"
    assert wc.synthetic_rating(50, 0)[0] == "AAA"  # no interest expense


def test_tax_and_beta_fallbacks():
    p = profile(beta=None, tax_provision=5e9, pretax_income=-10e9)
    inputs, _ = wc.build_inputs(p, risk_free=0.04, regression=0.9)
    assert inputs["beta"] == (0.9, "Regression vs S&P 500 (5Y monthly)")
    assert inputs["tax_rate"][0] == wc.STATUTORY_TAX_RATE
    with pytest.raises(ValueError):
        wc.build_inputs(profile(beta=None), risk_free=0.04)


def test_overrides_win():
    inputs, _ = wc.build_inputs(profile(), risk_free=0.04, beta=1.5, tax_rate=0.25,
                                cost_of_debt=0.07)
    assert inputs["beta"][0] == 1.5
    assert inputs["tax_rate"][0] == 0.25
    assert inputs["cost_of_debt"][0] == 0.07


def test_no_debt_company():
    inputs, _ = wc.build_inputs(profile(total_debt=None, prior_debt=None,
                                        interest_expense=None), risk_free=0.04)
    _, structure, result = wc.breakdown_tables(inputs)
    assert result["debt_weight"] == 0
    assert result["wacc"] == pytest.approx(result["cost_of_equity"])


def test_regression_beta():
    rng = np.random.default_rng(0)
    market = pd.Series(rng.normal(0.01, 0.04, 60))
    stock = 1.3 * market + rng.normal(0, 0.001, 60)
    assert wc.regression_beta(stock, market) == pytest.approx(1.3, abs=0.02)


def test_sensitivity_centre_matches_wacc():
    inputs, _ = wc.build_inputs(profile(), risk_free=0.04)
    _, _, result = wc.breakdown_tables(inputs)
    grid = wc.sensitivity_table(inputs)
    assert grid.iloc[2, 2] == pytest.approx(result["wacc"])
    assert (grid.diff().dropna() > 0).all().all()  # higher beta -> higher WACC


def test_parse_rate():
    assert wc.parse_rate("4.5") == 0.045
    assert wc.parse_rate("4.5%") == 0.045
    assert wc.parse_rate("0.045") == 0.045
    assert wc.parse_rate("1") == 0.01


def test_latest_reads_yfinance_statement():
    cols = pd.to_datetime(["2025-09-30", "2024-09-30"])
    frame = pd.DataFrame({cols[0]: [100.0, np.nan], cols[1]: [90.0, 5.0]},
                         index=["Total Debt", "EBIT"])
    assert wc._latest(frame, "Total Debt") == 100.0
    assert wc._latest(frame, "Total Debt", column=1) == 90.0
    assert wc._latest(frame, "EBIT", "Operating Income") is None
