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

WATCHLIST_FILE = Path(__file__).parent / "watchlist.json"
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


# ── Streamlit UI ───────────────────────────────────────────────────────────────

def render_portfolio_section(tickers_shares: dict[str, float]) -> list[dict]:
    holdings = []
    rows = []
    for ticker, shares in tickers_shares.items():
        data = get_stock_data(ticker)
        if not data:
            st.warning(f"Could not fetch data for {ticker}. Check ticker symbol.")
            continue
        info = data["info"]
        hist = data["hist"]
        price = float(hist["Close"].iloc[-1])
        hist_len = len(hist)
        prev_close_raw = info.get("previousClose")
        prev_close = float(
            prev_close_raw if prev_close_raw is not None
            else (hist["Close"].iloc[-2] if hist_len > 1 else price)
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


def render_ai_insights(ticker: str) -> None:
    data = get_stock_data(ticker)
    if not data:
        st.error(f"No data for {ticker}")
        return
    info = data["info"]
    hist = compute_moving_averages(data["hist"])

    price = float(hist["Close"].iloc[-1])
    hist_len = len(hist["Close"])
    prev_close_raw = info.get("previousClose")
    prev_close = float(
        prev_close_raw if prev_close_raw is not None
        else (hist["Close"].iloc[-2] if hist_len > 1 else price)
    )
    change_pct = (price - prev_close) / prev_close * 100 if prev_close else 0.0
    pe = float(info.get("trailingPE") or 0)
    w52_low = float(info.get("fiftyTwoWeekLow") or 0)
    w52_high = float(info.get("fiftyTwoWeekHigh") or 0)
    ma50_val = hist["MA50"].iloc[-1]
    ma200_val = hist["MA200"].iloc[-1]
    ma50 = float(ma50_val) if not pd.isna(ma50_val) else 0.0
    ma200 = float(ma200_val) if not pd.isna(ma200_val) else 0.0

    st.caption(f"**{ticker}** — ${price:.2f} ({change_pct:+.2f}% today) — results cached 5 min")

    with st.spinner("Asking Groq..."):
        insight = get_ai_insight(ticker, price, change_pct, pe, w52_low, w52_high, ma50, ma200)
    st.markdown(insight)

    with st.expander("Prompt sent to Groq (~75 tokens)"):
        st.code(build_ai_prompt(ticker, price, change_pct, pe, w52_low, w52_high, ma50, ma200))


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
        st.info("Coming soon.")


if __name__ == "__main__":
    main()
