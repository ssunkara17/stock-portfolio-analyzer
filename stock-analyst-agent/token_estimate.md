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
