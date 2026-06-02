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
