# Stock Analyst Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-file Streamlit app that tracks a US stock portfolio with live yfinance data and AI-powered insights via Groq LLM.

**Architecture:** Pure business-logic functions live at the top of `app.py` (importable and testable without Streamlit). Streamlit UI is isolated in render functions called from `main()`. External API calls use `@st.cache_data(ttl=300)` for 5-minute caching. All files live under `stock-analyst-agent/`.

**Tech Stack:** Python 3.14, Streamlit, yfinance, Groq SDK, pandas, plotly, python-dotenv, pytest, uv (package manager)

---

## Token Estimates

### Claude Code Implementation Tokens (estimated)

| Task | Input Tokens |
|------|-------------|
| Task 1 – Setup | ~600 |
| Task 2 – Write tests | ~1,200 |
| Task 3 – Pure functions | ~900 |
| Task 4 – Portfolio dashboard | ~1,500 |
| Task 5 – Stock analysis + charts | ~1,200 |
| Task 6 – AI insights | ~800 |
| Task 7 – Watchlist + performance | ~900 |
| Task 8 – Documentation | ~500 |
| Overhead (re-reads, fixes) | ~700 |
| **Total** | **~8,300** |

### Groq LLM Tokens Per Analysis Session

| Component | Tokens |
|-----------|--------|
| System prompt | ~15 |
| User prompt per stock | ~60 |
| Output per stock | ~100 |
| **Per 5-stock session** | **~875** |
| Monthly (2 sessions/day × 30 days) | ~52,500 |

---

## File Structure

| File | Purpose |
|------|---------|
| `stock-analyst-agent/app.py` | Main Streamlit app — max 400 lines |
| `stock-analyst-agent/pyproject.toml` | Updated with all dependencies |
| `stock-analyst-agent/requirements.txt` | Pip-compatible dependency list |
| `stock-analyst-agent/.env.example` | API key template |
| `stock-analyst-agent/.gitignore` | Excludes .env, .venv, __pycache__ |
| `stock-analyst-agent/tests/__init__.py` | Makes tests a package |
| `stock-analyst-agent/tests/conftest.py` | Mocks streamlit so pure functions are importable |
| `stock-analyst-agent/tests/test_calculations.py` | Tests for portfolio calcs and returns |
| `stock-analyst-agent/tests/test_watchlist.py` | Tests for watchlist CRUD |
| `stock-analyst-agent/token_estimate.md` | Token usage estimates |
| `stock-analyst-agent/README.md` | Setup and usage (max 300 words) |

---

## Task 1: Project Setup — Dependencies & Config Files

**Files:**
- Modify: `stock-analyst-agent/pyproject.toml`
- Create: `stock-analyst-agent/requirements.txt`
- Create: `stock-analyst-agent/.env.example`
- Create: `stock-analyst-agent/.gitignore`

- [ ] **Step 1: Update pyproject.toml with all dependencies**

Replace the entire contents of `stock-analyst-agent/pyproject.toml` with:

```toml
[project]
name = "stock-analyst-agent"
version = "0.1.0"
description = "AI-powered personal stock portfolio analyst"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "streamlit>=1.35.0",
    "yfinance>=0.2.40",
    "groq>=0.9.0",
    "pandas>=2.2.0",
    "plotly>=5.22.0",
    "python-dotenv>=1.0.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Install dependencies with uv**

Run from `stock-analyst-agent/` directory:

```powershell
cd stock-analyst-agent
uv sync
uv add --dev pytest
```

Expected output: packages installed into `.venv`, no errors.

- [ ] **Step 3: Create requirements.txt**

Run from `stock-analyst-agent/` directory:

```powershell
uv export --no-hashes --output-file requirements.txt
```

Expected: `requirements.txt` created with pinned versions.

- [ ] **Step 4: Create .env.example**

Create `stock-analyst-agent/.env.example`:

```
GROQ_API_KEY=your_groq_api_key_here
```

- [ ] **Step 5: Create .gitignore**

Create `stock-analyst-agent/.gitignore`:

```
.env
.venv/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
watchlist.json
```

- [ ] **Step 6: Commit**

```powershell
git add stock-analyst-agent/pyproject.toml stock-analyst-agent/requirements.txt stock-analyst-agent/.env.example stock-analyst-agent/.gitignore
git commit -m "chore: add dependencies and config files"
```

---

## Task 2: Write Tests for Pure Functions (TDD — Red Phase)

**Files:**
- Create: `stock-analyst-agent/tests/__init__.py`
- Create: `stock-analyst-agent/tests/conftest.py`
- Create: `stock-analyst-agent/tests/test_calculations.py`
- Create: `stock-analyst-agent/tests/test_watchlist.py`

- [ ] **Step 1: Create tests package**

Create `stock-analyst-agent/tests/__init__.py` as an empty file.

- [ ] **Step 2: Create conftest.py to mock Streamlit**

Create `stock-analyst-agent/tests/conftest.py`:

```python
import sys
from unittest.mock import MagicMock

mock_st = MagicMock()
mock_st.cache_data = lambda **kwargs: (lambda f: f)
sys.modules["streamlit"] = mock_st
```

This runs before any test file is imported. It replaces `streamlit` in `sys.modules` so that `import streamlit as st` in `app.py` gets the mock, and `@st.cache_data(ttl=300)` becomes a no-op pass-through decorator.

- [ ] **Step 3: Write tests for calculation functions**

Create `stock-analyst-agent/tests/test_calculations.py`:

```python
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
    assert result["day_gain"] == pytest.approx(45.0)  # (150-148)*10 + (300-295)*5
    assert result["day_pct"] > 0


def test_calculate_portfolio_summary_empty():
    from app import calculate_portfolio_summary
    result = calculate_portfolio_summary([])
    assert result["total_value"] == 0.0
    assert result["day_gain"] == 0.0
    assert result["day_pct"] == 0.0


def test_calculate_returns_basic():
    from app import calculate_returns
    closes = list(range(100, 125))  # 25 values: 100..124
    hist = pd.DataFrame({"Close": [float(c) for c in closes]})
    result = calculate_returns(hist)
    current = 124.0
    assert result["1d"] == pytest.approx((current - 123.0) / 123.0 * 100)
    assert result["1w"] == pytest.approx((current - 119.0) / 119.0 * 100)


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
```

- [ ] **Step 4: Write tests for watchlist functions**

Create `stock-analyst-agent/tests/test_watchlist.py`:

```python
import pytest


def test_load_watchlist_returns_empty_when_no_file(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "WATCHLIST_FILE", tmp_path / "watchlist.json")
    result = app.load_watchlist()
    assert result == []


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "WATCHLIST_FILE", tmp_path / "watchlist.json")
    app.save_watchlist(["AAPL", "MSFT"])
    result = app.load_watchlist()
    assert "AAPL" in result
    assert "MSFT" in result
    assert len(result) == 2


def test_save_watchlist_deduplicates_and_uppercases(tmp_path, monkeypatch):
    import app
    monkeypatch.setattr(app, "WATCHLIST_FILE", tmp_path / "watchlist.json")
    app.save_watchlist(["aapl", "AAPL", "msft"])
    result = app.load_watchlist()
    assert len(result) == 2
    assert "AAPL" in result
    assert "MSFT" in result


def test_load_watchlist_handles_corrupt_json(tmp_path, monkeypatch):
    import app
    wf = tmp_path / "watchlist.json"
    wf.write_text("not valid json")
    monkeypatch.setattr(app, "WATCHLIST_FILE", wf)
    result = app.load_watchlist()
    assert result == []
```

- [ ] **Step 5: Run tests — expect ALL to fail (app.py has no pure functions yet)**

Run from `stock-analyst-agent/`:

```powershell
uv run pytest tests/ -v
```

Expected: `ImportError` or `AttributeError` — `app` module only has `main()` placeholder. This confirms the tests are wired correctly.

- [ ] **Step 6: Commit the tests**

```powershell
git add stock-analyst-agent/tests/
git commit -m "test: add failing tests for portfolio calculations and watchlist"
```

---

## Task 3: Implement Pure Functions (TDD — Green Phase)

**Files:**
- Create: `stock-analyst-agent/app.py` (pure functions + cached API stubs; NO Streamlit UI yet)

- [ ] **Step 1: Write app.py with all pure functions and cached stubs**

Create `stock-analyst-agent/app.py`:

```python
import json
import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

WATCHLIST_FILE = Path("watchlist.json")
GROQ_MODEL = "llama3-8b-8192"
CACHE_TTL = 300


# ── Pure functions (no Streamlit calls — importable in tests) ──────────────────

def calculate_portfolio_summary(holdings: list[dict]) -> dict:
    if not holdings:
        return {"total_value": 0.0, "day_gain": 0.0, "day_pct": 0.0}
    total_value = sum(h["price"] * h["shares"] for h in holdings)
    total_prev = sum(h["prev_close"] * h["shares"] for h in holdings)
    day_gain = total_value - total_prev
    day_pct = (day_gain / total_prev * 100) if total_prev else 0.0
    return {"total_value": total_value, "day_gain": day_gain, "day_pct": day_pct}


def calculate_returns(hist: pd.DataFrame) -> dict:
    if hist is None or hist.empty or len(hist) < 2:
        return {"1d": 0.0, "1w": 0.0, "1m": 0.0}
    closes = hist["Close"]
    current = float(closes.iloc[-1])

    def ret(days: int) -> float:
        idx = max(0, len(closes) - 1 - days)
        past = float(closes.iloc[idx])
        return (current - past) / past * 100 if past else 0.0

    return {"1d": ret(1), "1w": ret(5), "1m": ret(21)}


def compute_moving_averages(hist: pd.DataFrame) -> pd.DataFrame:
    df = hist.copy()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    return df


def load_watchlist() -> list[str]:
    if WATCHLIST_FILE.exists():
        try:
            return json.loads(WATCHLIST_FILE.read_text())
        except json.JSONDecodeError:
            return []
    return []


def save_watchlist(tickers: list[str]) -> None:
    WATCHLIST_FILE.write_text(json.dumps(sorted(set(t.upper() for t in tickers))))


def build_ai_prompt(
    ticker: str,
    price: float,
    change_pct: float,
    pe: float,
    week52_low: float,
    week52_high: float,
    ma50: float,
    ma200: float,
) -> str:
    return (
        f"Stock:{ticker} ${price:.2f}({change_pct:+.1f}%) "
        f"PE:{pe:.1f} 52w:${week52_low:.0f}-${week52_high:.0f} "
        f"MA50:${ma50:.0f} MA200:${ma200:.0f}. "
        "3 bullets: 1)performance 2)trend 3)buy/hold/sell+reason"
    )


# ── Cached API functions ───────────────────────────────────────────────────────

@st.cache_data(ttl=CACHE_TTL)
def get_stock_data(ticker: str) -> dict | None:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        if hist.empty:
            return None
        return {"info": info, "hist": hist}
    except Exception:
        return None


@st.cache_data(ttl=CACHE_TTL)
def get_ai_insight(
    ticker: str,
    price: float,
    change_pct: float,
    pe: float,
    week52_low: float,
    week52_high: float,
    ma50: float,
    ma200: float,
) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return "• GROQ_API_KEY not set in .env"
    try:
        client = Groq(api_key=api_key)
        prompt = build_ai_prompt(ticker, price, change_pct, pe, week52_low, week52_high, ma50, ma200)
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "Stock analyst. Respond with 3 bullet points only."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"• AI error: {e}"


def main() -> None:
    pass  # UI added in Tasks 4–7


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run tests — expect ALL to pass**

Run from `stock-analyst-agent/`:

```powershell
uv run pytest tests/ -v
```

Expected output (all green):
```
tests/test_calculations.py::test_calculate_portfolio_summary_basic PASSED
tests/test_calculations.py::test_calculate_portfolio_summary_empty PASSED
tests/test_calculations.py::test_calculate_returns_basic PASSED
tests/test_calculations.py::test_calculate_returns_empty PASSED
tests/test_calculations.py::test_calculate_returns_single_row PASSED
tests/test_calculations.py::test_compute_moving_averages_columns PASSED
tests/test_calculations.py::test_compute_moving_averages_does_not_mutate PASSED
tests/test_calculations.py::test_build_ai_prompt_contains_required_fields PASSED
tests/test_watchlist.py::test_load_watchlist_returns_empty_when_no_file PASSED
tests/test_watchlist.py::test_save_and_load_roundtrip PASSED
tests/test_watchlist.py::test_save_watchlist_deduplicates_and_uppercases PASSED
tests/test_watchlist.py::test_load_watchlist_handles_corrupt_json PASSED
12 passed in ...
```

If any test fails, fix only the function under test — do not change tests.

- [ ] **Step 3: Commit**

```powershell
git add stock-analyst-agent/app.py
git commit -m "feat: implement pure functions (portfolio calc, returns, watchlist, prompt builder)"
```

---

## Task 4: Portfolio Dashboard UI

**Files:**
- Modify: `stock-analyst-agent/app.py` — add `render_portfolio_section()` and wire into `main()`

- [ ] **Step 1: Add the portfolio section render function**

In `app.py`, replace the `def main() -> None:` block and everything after it with the following (keep all existing code above `main()` unchanged):

```python
# ── Streamlit UI ───────────────────────────────────────────────────────────────

def render_portfolio_section(tickers_shares: dict[str, float]) -> list[dict]:
    holdings = []
    rows = []
    for ticker, shares in tickers_shares.items():
        data = get_stock_data(ticker)
        if not data:
            st.warning(f"Could not fetch data for {ticker}")
            continue
        info = data["info"]
        hist = data["hist"]
        price = float(hist["Close"].iloc[-1])
        hist_len = len(hist)
        prev_close = float(
            info.get("previousClose") or (hist["Close"].iloc[-2] if hist_len > 1 else price)
        )
        change_pct = (price - prev_close) / prev_close * 100 if prev_close else 0.0
        mkt_cap = info.get("marketCap") or 0
        volume = info.get("volume") or 0
        holdings.append({"ticker": ticker, "shares": shares, "price": price, "prev_close": prev_close})
        rows.append({
            "Ticker": ticker,
            "Shares": shares,
            "Price": f"${price:.2f}",
            "Day %": f"{change_pct:+.2f}%",
            "Value": f"${price * shares:,.2f}",
            "Mkt Cap": f"${mkt_cap / 1e9:.1f}B" if mkt_cap else "N/A",
            "Volume": f"{volume:,}" if volume else "N/A",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        summary = calculate_portfolio_summary(holdings)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Value", f"${summary['total_value']:,.2f}")
        col2.metric(
            "Day Gain/Loss",
            f"${summary['day_gain']:,.2f}",
            f"{summary['day_pct']:+.2f}%",
        )
        col3.metric("Positions", len(holdings))
    else:
        st.info("Enter valid tickers in the sidebar to see your portfolio.")
    return holdings


def main() -> None:
    st.set_page_config(page_title="Stock Analyst Agent", page_icon="📈", layout="wide")
    st.title("📈 Stock Analyst Agent")

    st.sidebar.header("My Portfolio")
    raw_input = st.sidebar.text_area(
        "Tickers & Shares (TICKER SHARES, one per line)",
        value="AAPL 10\nMSFT 5\nGOOGL 3",
        height=150,
    )

    tickers_shares: dict[str, float] = {}
    for line in raw_input.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                tickers_shares[parts[0].upper()] = float(parts[1])
            except ValueError:
                pass

    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Analysis", "AI Insights", "Watchlist"])

    with tab1:
        st.subheader("Portfolio Overview")
        holdings = render_portfolio_section(tickers_shares)

    with tab2:
        st.subheader("Stock Analysis")
        st.info("Coming in next task.")
        holdings = []

    with tab3:
        st.subheader("AI Insights")
        st.info("Coming in next task.")

    with tab4:
        st.subheader("Watchlist")
        st.info("Coming in next task.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify tests still pass**

```powershell
uv run pytest tests/ -v
```

Expected: 12 passed (no regressions).

- [ ] **Step 3: Smoke-test the app in the browser**

```powershell
uv run streamlit run app.py
```

Open `http://localhost:8501`. Verify:
- Sidebar shows a text area with AAPL/MSFT/GOOGL defaults
- "Portfolio" tab loads a table with price, day %, value columns
- Metrics row shows Total Value, Day Gain/Loss, Positions
- Other tabs show "Coming in next task" placeholders
- No Python errors in terminal

Stop the server with Ctrl+C.

- [ ] **Step 4: Commit**

```powershell
git add stock-analyst-agent/app.py
git commit -m "feat: add portfolio dashboard with live price table and summary metrics"
```

---

## Task 5: Stock Analysis + Interactive Charts

**Files:**
- Modify: `stock-analyst-agent/app.py` — add `render_stock_analysis()`, wire into `main()` tab2

- [ ] **Step 1: Add the stock analysis render function**

Add the following function to `app.py` immediately before `def main()`:

```python
def render_stock_analysis(ticker: str) -> None:
    data = get_stock_data(ticker)
    if not data:
        st.error(f"No data available for {ticker}")
        return
    info = data["info"]
    hist = compute_moving_averages(data["hist"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], name="Price", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA50"], name="MA50", line=dict(color="orange", dash="dash")))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MA200"], name="MA200", line=dict(color="red", dash="dash")))
    fig.update_layout(
        title=f"{ticker} — 1-Year Price + Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    pe = info.get("trailingPE")
    div_yield = info.get("dividendYield")
    w52_low = info.get("fiftyTwoWeekLow", 0)
    w52_high = info.get("fiftyTwoWeekHigh", 0)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("P/E Ratio", f"{pe:.1f}" if pe else "N/A")
    col2.metric("Div Yield", f"{div_yield * 100:.2f}%" if div_yield else "0.00%")
    col3.metric("52w Low", f"${w52_low:.2f}")
    col4.metric("52w High", f"${w52_high:.2f}")
```

- [ ] **Step 2: Wire into main() tab2**

In `main()`, replace:
```python
    with tab2:
        st.subheader("Stock Analysis")
        st.info("Coming in next task.")
        holdings = []
```

With:
```python
    with tab2:
        st.subheader("Stock Analysis")
        ticker_list = list(tickers_shares.keys())
        if ticker_list:
            selected = st.selectbox("Select stock to analyze", ticker_list)
            render_stock_analysis(selected)
        else:
            st.info("Add holdings in the sidebar first.")
        holdings = []
```

Also remove the line `holdings = []` that was left in tab2 — it's unused there (holdings comes from tab1's render call, but since Streamlit runs top-to-bottom each rerender, just remove it).

The final `main()` should have tab1 assign holdings and tabs 2/3/4 not need to reassign it. Update `main()` so that tab1 renders first and holdings is used only in tab1 and passed to performance in a later task.

- [ ] **Step 3: Verify tests still pass**

```powershell
uv run pytest tests/ -v
```

Expected: 12 passed.

- [ ] **Step 4: Smoke-test in browser**

```powershell
uv run streamlit run app.py
```

Open `http://localhost:8501`. Verify:
- "Analysis" tab shows a dropdown of tickers
- Selecting a ticker renders a Plotly line chart with Price, MA50, MA200 lines
- Below the chart: P/E, Div Yield, 52w Low, 52w High metrics
- No errors in terminal

Stop with Ctrl+C.

- [ ] **Step 5: Commit**

```powershell
git add stock-analyst-agent/app.py
git commit -m "feat: add 1-year stock chart with MA50/MA200 and fundamental metrics"
```

---

## Task 6: AI-Powered Insights

**Files:**
- Modify: `stock-analyst-agent/app.py` — add `render_ai_insights()`, wire into `main()` tab3

- [ ] **Step 1: Add the AI insights render function**

Add the following function immediately before `def main()`:

```python
def render_ai_insights(ticker: str) -> None:
    data = get_stock_data(ticker)
    if not data:
        st.error(f"No data for {ticker}")
        return
    info = data["info"]
    hist = compute_moving_averages(data["hist"])

    price = float(hist["Close"].iloc[-1])
    hist_len = len(hist["Close"])
    prev_close = float(info.get("previousClose") or (hist["Close"].iloc[-2] if hist_len > 1 else price))
    change_pct = (price - prev_close) / prev_close * 100 if prev_close else 0.0
    pe = float(info.get("trailingPE") or 0)
    w52_low = float(info.get("fiftyTwoWeekLow") or 0)
    w52_high = float(info.get("fiftyTwoWeekHigh") or 0)
    ma50_val = hist["MA50"].iloc[-1]
    ma200_val = hist["MA200"].iloc[-1]
    ma50 = float(ma50_val) if not pd.isna(ma50_val) else 0.0
    ma200 = float(ma200_val) if not pd.isna(ma200_val) else 0.0

    st.caption(f"**{ticker}** — ${price:.2f} ({change_pct:+.2f}% today) — cached 5 min")

    with st.spinner("Asking Groq..."):
        insight = get_ai_insight(ticker, price, change_pct, pe, w52_low, w52_high, ma50, ma200)
    st.markdown(insight)

    with st.expander("Prompt sent to Groq (~75 tokens)"):
        st.code(build_ai_prompt(ticker, price, change_pct, pe, w52_low, w52_high, ma50, ma200))
```

- [ ] **Step 2: Wire into main() tab3**

Replace:
```python
    with tab3:
        st.subheader("AI Insights")
        st.info("Coming in next task.")
```

With:
```python
    with tab3:
        st.subheader("AI Insights")
        ticker_list = list(tickers_shares.keys())
        if ticker_list:
            selected_ai = st.selectbox("Select stock for AI analysis", ticker_list, key="ai_sel")
            if st.button("Get AI Analysis", type="primary"):
                render_ai_insights(selected_ai)
            st.caption("Results cached for 5 minutes per ticker.")
        else:
            st.info("Add holdings in the sidebar first.")
```

- [ ] **Step 3: Verify tests still pass**

```powershell
uv run pytest tests/ -v
```

Expected: 12 passed.

- [ ] **Step 4: Smoke-test in browser**

```powershell
uv run streamlit run app.py
```

Open `http://localhost:8501`. Verify:
- "AI Insights" tab shows a ticker dropdown and "Get AI Analysis" button
- Clicking the button shows a spinner, then 3 bullet points from Groq
- The expander shows the exact prompt sent
- Clicking again immediately returns cached result (no spinner delay)
- If `.env` is missing `GROQ_API_KEY`, shows `• GROQ_API_KEY not set in .env`

Stop with Ctrl+C.

- [ ] **Step 5: Commit**

```powershell
git add stock-analyst-agent/app.py
git commit -m "feat: add AI-powered stock insights via Groq with 5-minute cache"
```

---

## Task 7: Watchlist + Performance Metrics

**Files:**
- Modify: `stock-analyst-agent/app.py` — add `render_watchlist_section()`, `render_performance_metrics()`, wire into `main()`

- [ ] **Step 1: Add watchlist render function**

Add immediately before `def main()`:

```python
def render_watchlist_section() -> None:
    watchlist = load_watchlist()
    col1, col2 = st.columns([4, 1])
    new_ticker = col1.text_input("Add ticker to watchlist", placeholder="e.g. TSLA", label_visibility="collapsed")
    if col2.button("Add", use_container_width=True) and new_ticker.strip():
        watchlist.append(new_ticker.strip().upper())
        save_watchlist(watchlist)
        st.rerun()

    if not watchlist:
        st.caption("Watchlist is empty. Add tickers above.")
        return

    for i, t in enumerate(watchlist):
        col_t, col_p, col_r = st.columns([3, 3, 1])
        col_t.write(f"**{t}**")
        data = get_stock_data(t)
        if data:
            price = float(data["hist"]["Close"].iloc[-1])
            col_p.write(f"${price:.2f}")
        else:
            col_p.write("N/A")
        if col_r.button("✕", key=f"rm_{i}"):
            watchlist.pop(i)
            save_watchlist(watchlist)
            st.rerun()
```

- [ ] **Step 2: Add performance metrics render function**

Add immediately after `render_watchlist_section()`:

```python
def render_performance_metrics(holdings: list[dict]) -> None:
    if not holdings:
        st.info("Add holdings in the sidebar to see performance metrics.")
        return

    total_val = sum(h["price"] * h["shares"] for h in holdings)
    rows = []
    alloc_labels, alloc_values = [], []

    for h in holdings:
        data = get_stock_data(h["ticker"])
        if not data:
            continue
        returns = calculate_returns(data["hist"])
        val = h["price"] * h["shares"]
        pct_alloc = val / total_val * 100 if total_val else 0.0
        alloc_labels.append(h["ticker"])
        alloc_values.append(val)
        rows.append({
            "Ticker": h["ticker"],
            "Value": f"${val:,.2f}",
            "Allocation": f"{pct_alloc:.1f}%",
            "1D %": f"{returns['1d']:+.2f}%",
            "1W %": f"{returns['1w']:+.2f}%",
            "1M %": f"{returns['1m']:+.2f}%",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if alloc_labels:
        fig = go.Figure(go.Pie(labels=alloc_labels, values=alloc_values, hole=0.4))
        fig.update_layout(
            title="Portfolio Allocation",
            height=350,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 3: Update main() to use holdings across tabs and wire new sections**

Replace the entire `main()` function with:

```python
def main() -> None:
    st.set_page_config(page_title="Stock Analyst Agent", page_icon="📈", layout="wide")
    st.title("📈 Stock Analyst Agent")

    st.sidebar.header("My Portfolio")
    raw_input = st.sidebar.text_area(
        "Tickers & Shares (TICKER SHARES, one per line)",
        value="AAPL 10\nMSFT 5\nGOOGL 3",
        height=150,
    )

    tickers_shares: dict[str, float] = {}
    for line in raw_input.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) == 2:
            try:
                tickers_shares[parts[0].upper()] = float(parts[1])
            except ValueError:
                pass

    tab1, tab2, tab3, tab4 = st.tabs(["Portfolio", "Analysis", "AI Insights", "Watchlist"])

    with tab1:
        st.subheader("Portfolio Overview")
        holdings = render_portfolio_section(tickers_shares)
        st.divider()
        st.subheader("Performance Metrics")
        render_performance_metrics(holdings)

    with tab2:
        st.subheader("Stock Analysis")
        ticker_list = list(tickers_shares.keys())
        if ticker_list:
            selected = st.selectbox("Select stock to analyze", ticker_list)
            render_stock_analysis(selected)
        else:
            st.info("Add holdings in the sidebar first.")

    with tab3:
        st.subheader("AI Insights")
        ticker_list = list(tickers_shares.keys())
        if ticker_list:
            selected_ai = st.selectbox("Select stock for AI analysis", ticker_list, key="ai_sel")
            if st.button("Get AI Analysis", type="primary"):
                render_ai_insights(selected_ai)
            st.caption("Results cached for 5 minutes per ticker.")
        else:
            st.info("Add holdings in the sidebar first.")

    with tab4:
        st.subheader("Watchlist")
        render_watchlist_section()
```

- [ ] **Step 4: Verify tests still pass**

```powershell
uv run pytest tests/ -v
```

Expected: 12 passed.

- [ ] **Step 5: Verify line count stays under 400**

```powershell
(Get-Content stock-analyst-agent/app.py | Measure-Object -Line).Lines
```

Expected: number ≤ 400.

- [ ] **Step 6: Smoke-test all tabs in browser**

```powershell
uv run streamlit run app.py
```

Open `http://localhost:8501`. Verify each tab:

**Portfolio tab:**
- Table with Ticker, Shares, Price, Day %, Value, Mkt Cap, Volume
- Three metrics: Total Value, Day Gain/Loss, Positions
- Performance table with 1D/1W/1M returns
- Donut chart showing allocation by ticker

**Analysis tab:**
- Ticker dropdown; selecting renders 1-year Plotly chart with MA lines
- Four metrics below chart: P/E, Div Yield, 52w Low, 52w High

**AI Insights tab:**
- Ticker dropdown and "Get AI Analysis" button
- Clicking shows bullet-point Groq response
- Expandable prompt preview

**Watchlist tab:**
- Text input + Add button
- Each watchlist item shows ticker, current price, remove button
- Clicking remove immediately removes the ticker (page reruns)
- Watchlist persists after browser refresh (saved to `watchlist.json`)

Stop with Ctrl+C.

- [ ] **Step 7: Commit**

```powershell
git add stock-analyst-agent/app.py
git commit -m "feat: add watchlist with persistence and portfolio performance metrics with allocation chart"
```

---

## Task 8: Documentation — README + Token Estimate

**Files:**
- Modify: `stock-analyst-agent/README.md`
- Create: `stock-analyst-agent/token_estimate.md`

- [ ] **Step 1: Write README.md**

Replace `stock-analyst-agent/README.md` with:

```markdown
# Stock Analyst Agent

AI-powered personal stock portfolio tracker built with Streamlit and Groq LLM.

## Setup

**Prerequisites:** Python 3.14+, [uv](https://docs.astral.sh/uv/)

```bash
cd stock-analyst-agent
uv sync
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

## Run

```bash
uv run streamlit run app.py
```

Opens at `http://localhost:8501`.

## Features

- **Portfolio tab** — Enter tickers + share counts in the sidebar. See live prices, day % change, market cap, volume, total portfolio value, and 1D/1W/1M returns with an allocation donut chart.
- **Analysis tab** — 1-year interactive price chart with 50-day and 200-day moving averages. Shows P/E ratio, dividend yield, 52-week high/low.
- **AI Insights tab** — Select a stock and click "Get AI Analysis" for a Groq-powered 3-bullet summary: performance, trend, and buy/hold/sell recommendation. Results cached 5 minutes.
- **Watchlist tab** — Save tickers to a local `watchlist.json` file. Shows current price for each watchlist item.

## Run Tests

```bash
uv run pytest tests/ -v
```

## Configuration

| Variable | Description |
|----------|-------------|
| `GROQ_API_KEY` | Required. Groq API key from console.groq.com |

All API responses (yfinance + Groq) are cached for 5 minutes to minimize redundant calls.
```

- [ ] **Step 2: Write token_estimate.md**

Create `stock-analyst-agent/token_estimate.md`:

```markdown
# Token Usage Estimates

## Claude Code — Implementation Tokens

| Task | Input Tokens (est.) |
|------|-------------------|
| Task 1 — Setup | ~600 |
| Task 2 — Write tests | ~1,200 |
| Task 3 — Pure functions | ~900 |
| Task 4 — Portfolio dashboard | ~1,500 |
| Task 5 — Stock analysis + charts | ~1,200 |
| Task 6 — AI insights | ~800 |
| Task 7 — Watchlist + performance | ~900 |
| Task 8 — Documentation | ~500 |
| Overhead (re-reads, fixes) | ~700 |
| **Total** | **~8,300** |

---

## Groq LLM — Per Analysis Session

### Prompt Template (actual)

**System prompt (~15 tokens):**
```
Stock analyst. Respond with 3 bullet points only.
```

**User prompt (~60 tokens, example):**
```
Stock:AAPL $189.50(+1.2%) PE:28.5 52w:$164-$199 MA50:$185 MA200:$178.
3 bullets: 1)performance 2)trend 3)buy/hold/sell+reason
```

**Max output tokens:** 150 (configured via `max_tokens=150`)

### Per-Stock Cost

| Component | Tokens |
|-----------|--------|
| System prompt | ~15 |
| User prompt | ~60 |
| Output | ~100 |
| **Total per stock** | **~175** |

### Session Cost (5 stocks)

| Metric | Value |
|--------|-------|
| Input per session | ~375 tokens |
| Output per session | ~500 tokens |
| **Total per session** | **~875 tokens** |

### Monthly Estimate

| Usage | Tokens |
|-------|--------|
| 2 sessions/day × 30 days × 5 stocks | ~52,500 |
| Groq free tier limit | 14,400 req/day (Llama 3 8B) |
| **Status** | Well within free tier |

---

## yfinance API

yfinance calls are cached for 5 minutes (`@st.cache_data(ttl=300)`).
No token cost — uses Yahoo Finance unofficial API (free).

## Token Minimization Decisions

1. System prompt kept to 1 sentence (~15 tokens vs typical 50–100)
2. User prompt uses compact format (no full sentences, just key–value pairs)
3. `max_tokens=150` caps output at ~3 short bullets
4. `temperature=0.3` reduces output verbosity
5. Results cached per ticker per 5 minutes — repeat clicks cost 0 tokens
```

- [ ] **Step 3: Verify README word count is under 300**

```powershell
(Get-Content stock-analyst-agent/README.md) -join " " -split "\s+" | Where-Object { $_ -ne "" } | Measure-Object | Select-Object -ExpandProperty Count
```

Expected: number ≤ 300.

- [ ] **Step 4: Final test run**

```powershell
uv run pytest tests/ -v
```

Expected: 12 passed.

- [ ] **Step 5: Final line count check**

```powershell
(Get-Content stock-analyst-agent/app.py | Measure-Object -Line).Lines
```

Expected: ≤ 400.

- [ ] **Step 6: Commit**

```powershell
git add stock-analyst-agent/README.md stock-analyst-agent/token_estimate.md
git commit -m "docs: add README and token usage estimates"
```

---

## Self-Review Checklist

### Spec Coverage

| Requirement | Task |
|-------------|------|
| Ticker + shares input | Task 4 |
| Current price, day %, mkt cap, volume | Task 4 |
| Portfolio total value, gain/loss, % change | Task 4 |
| 1-year historical chart (Plotly) | Task 5 |
| 50-day and 200-day moving averages | Task 5 |
| P/E ratio, dividend yield, 52w high/low | Task 5 |
| AI daily summary per stock | Task 6 |
| AI trend analysis | Task 6 |
| AI buy/hold/sell recommendation | Task 6 |
| System prompt < 100 tokens | Task 6 (15 tokens) |
| User prompt < 200 tokens per stock | Task 6 (~60 tokens) |
| Cache AI insights 5 minutes | Task 6 |
| Watchlist save/load (JSON) | Task 7 |
| Portfolio returns 1D/1W/1M | Task 7 |
| Portfolio allocation % | Task 7 |
| uv for package management | Task 1 |
| .env for GROQ_API_KEY | Task 1 |
| .env in .gitignore | Task 1 |
| 5-minute API caching | Tasks 3, 6 |
| app.py ≤ 400 lines | Task 7 step 5 |
| requirements.txt | Task 1 |
| .env.example | Task 1 |
| .gitignore | Task 1 |
| README.md ≤ 300 words | Task 8 |
| token_estimate.md | Task 8 |

All requirements covered. No gaps found.
