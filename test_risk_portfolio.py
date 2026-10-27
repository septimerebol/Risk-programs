import builtins

import numpy as np
import pandas as pd
import pytest

import risk_portfolio as rp


def synthetic_returns(n=500, seed=0):
    rng = np.random.default_rng(seed)
    cov = np.array([[0.0004, 0.0002], [0.0002, 0.0009]])
    data = rng.multivariate_normal([0.0005, 0.0008], cov, size=n)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(data, index=idx, columns=["AAA", "BBB"])


def feed(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr(builtins, "input", lambda _="": next(it))


def test_parse_money():
    assert rp.parse_money("$100,000.50") == 100000.50
    with pytest.raises(ValueError):
        rp.parse_money("-5")


def test_collect_positions_until_total(monkeypatch):
    # Over-allocation and an unknown ticker are rejected; loop ends at total.
    feed(monkeypatch, ["aaa", "60000", "zzz", "bbb", "50000", "bbb", "40000"])
    positions = rp.collect_positions(100000, validate=lambda t: t != "ZZZ")
    assert positions == {"AAA": 60000, "BBB": 40000}


def test_position_table():
    returns = synthetic_returns()
    table, undiversified = rp.position_table(returns, {"AAA": 60000, "BBB": 40000})
    port = table.loc["PORTFOLIO"]
    assert port["Value $"] == 100000
    assert table.loc[["AAA", "BBB"], "VaR Contrib"].sum() == pytest.approx(1.0)
    assert 0 < port["Param VaR $"] < undiversified


def test_confidence_table_scales_with_horizon():
    r = synthetic_returns()["AAA"]
    t = rp.confidence_table(r, 100000)
    sigma = r.std()
    assert t.loc["1 Day", "Volatility"] == pytest.approx(sigma)
    assert t.loc["1 Month", "Volatility"] == pytest.approx(sigma * np.sqrt(21))
    one_day = t.loc["1 Day"]
    assert one_day["Upper Bound"] - one_day["Lower Bound"] == pytest.approx(2 * 2.5758 * sigma, rel=1e-3)
    assert (t["VaR $"].diff().dropna() > 0).all()


def test_plot_bell_curve(tmp_path):
    r = synthetic_returns() @ pd.Series({"AAA": 0.6, "BBB": 0.4})
    out = rp.plot_bell_curve(r, 100000, path=tmp_path / "curve.png")
    assert out.exists() and out.stat().st_size > 0
