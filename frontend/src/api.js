const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

async function get(path) {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`API ${res.status}: ${path}`)
  return res.json()
}

// ── FastAPI routes ──
export const getPrediction  = (ticker) => get(`/predict/${ticker}`)
export const getAllPredictions = (refresh = false) => get(`/predict/all${refresh ? '?refresh=true' : ''}`)
export const getMarketData  = (ticker) => get(`/market/${ticker}`)
export const getNews        = (ticker) => get(`/news/${ticker}`)
export const getDailyReport = ()       => get('/report/daily')
export const getGlobalReport = ()      => get('/report/global')
export const getHealth      = ()       => get('/health')

// ── Yahoo Finance — OHLC (5m, 1 day) ──
const ohlcCache = {}

async function _fetchOHLC(sym) {
  const res  = await fetch(`${BASE}/market/proxy/chart?symbol=${sym}&interval=5m&range_val=1d`)
  if (!res.ok) return null          // 404 / 500 → return null, don't throw
  const json = await res.json()
  const result = json?.chart?.result?.[0]
  if (!result) return null
  const ts = result.timestamp
  const q  = result.indicators.quote[0]
  const data = ts
    .map((t, i) => ({ t: t * 1000, o: q.open[i], h: q.high[i], l: q.low[i], c: q.close[i], v: q.volume[i] }))
    .filter(d => d.o && d.c)
  return data.length ? data : null
}

export async function getOHLC(sym) {
  if (ohlcCache[sym]) return ohlcCache[sym]

  let data = await _fetchOHLC(sym)

  // Fallback: try BSE (.BO) if NS fails (e.g. TATAMOTORS.NS after demerger)
  if (!data && sym.endsWith('.NS')) {
    const bseSym = sym.replace('.NS', '.BO')
    data = await _fetchOHLC(bseSym)
  }

  if (data) ohlcCache[sym] = data
  return data ?? []
}

// ── Yahoo Finance — macro indices ──
export async function getMacroIndex(sym) {
  const res  = await fetch(`${BASE}/market/proxy/chart?symbol=${encodeURIComponent(sym)}&interval=1d&range_val=2d`)
  if (!res.ok) throw new Error(`Macro fetch failed: ${sym}`)
  const json = await res.json()
  const result = json?.chart?.result?.[0]
  if (!result) throw new Error(`No data for ${sym}`)
  const q      = result.indicators.quote[0]
  const closes = q.close.filter(Boolean)
  if (closes.length < 2) throw new Error(`Insufficient data for ${sym}`)
  const prev   = closes[closes.length - 2]
  const curr   = closes[closes.length - 1]
  const chgPct = (((curr - prev) / prev) * 100).toFixed(2)
  return { value: curr, chgPct, isUp: curr >= prev }
}