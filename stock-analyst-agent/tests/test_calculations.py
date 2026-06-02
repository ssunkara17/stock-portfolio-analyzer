import pandas as pd
import pytest


def test_calculate_portfolio_summary_basic():
    from app import calculate_portfolio_summary
    holdings = [
        {"ticker": "AAPL", "shares": 10, "price": 150.0, "prev_close": 148.0},
        {"ticker": "MSFT", "shares": 5, "price": 300.0, "prev_close": 295.0},
    ]
    result = calculate_portfolio_summary(holdings)
    assert result["total_value"] == pytest.approx(3000.0)
    assert result["day_gain"] == pytest.approx(45.0)
    assert result["day_pct"] == pytest.approx(45.0 / 2955.0 * 100, rel=1e-3)


def test_calculate_portfolio_summary_empty():
    from app import calculate_portfolio_summary
    result = calculate_portfolio_summary([])
    assert result["total_value"] == 0.0
    assert result["day_gain"] == 0.0
    assert result["day_pct"] == 0.0


def test_calculate_returns_basic():
    from app import calculate_returns
    closes = list(range(100, 125))
    hist = pd.DataFrame({"Close": [float(c) for c in closes]})
    result = calculate_returns(hist)
    current = 124.0
    assert result["1d"] == pytest.approx((current - 123.0) / 123.0 * 100)
    assert result["1w"] == pytest.approx((current - 119.0) / 119.0 * 100)
    assert result["1m"] == pytest.approx((current - 103.0) / 103.0 * 100)


def test_calculate_returns_empty():
    from app import calculate_returns
    result = calculate_returns(pd.DataFrame())
    assert result == {"1d": 0.0, "1w": 0.0, "1m": 0.0}


def test_calculate_returns_single_row():
    from app import calculate_returns
    hist = pd.DataFrame({"Close": [150.0]})
    result = calculate_returns(hist)
    assert result == {"1d": 0.0, "1w": 0.0, "1m": 0.0}


def test_compute_moving_averages_columns():
    from app import compute_moving_averages
    closes = [float(i) for i in range(1, 201)]
    hist = pd.DataFrame({"Close": closes})
    result = compute_moving_averages(hist)
    assert "MA50" in result.columns
    assert "MA200" in result.columns
    assert not pd.isna(result["MA50"].iloc[-1])
    assert not pd.isna(result["MA200"].iloc[-1])


def test_compute_moving_averages_does_not_mutate():
    from app import compute_moving_averages
    closes = [float(i) for i in range(1, 201)]
    hist = pd.DataFrame({"Close": closes})
    compute_moving_averages(hist)
    assert "MA50" not in hist.columns


def test_build_ai_prompt_contains_required_fields():
    from app import build_ai_prompt
    prompt = build_ai_prompt("AAPL", 150.0, 1.5, 28.5, 120.0, 180.0, 145.0, 130.0)
    assert "AAPL" in prompt
    assert "150.00" in prompt
    assert "+1.5" in prompt
    assert "28.5" in prompt
    assert "3 bullets" in prompt
