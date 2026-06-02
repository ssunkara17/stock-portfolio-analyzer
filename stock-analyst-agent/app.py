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
    except Exception as e:
        st.warning(f"Failed to fetch data for {ticker}: {e}")
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
    pass  # UI added in Tasks 4-7


if __name__ == "__main__":
    main()
