import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { STOCKS } from '../constants'
import { useOHLC, useMacro, useAllPredictions } from '../hooks'
import OHLCChart from '../components/OHLCChart'
import ChartToggle from '../components/ChartToggle'

export default function Detail() {
  const { ticker } = useParams()
  const navigate = useNavigate()
  const stockIndex = STOCKS.findIndex(s => s.t === ticker)
  const stock = stockIndex !== -1 ? STOCKS[stockIndex] : STOCKS[0]

  const [mode, setMode] = useState('line')
  const { data, loading: ohlcLoading, last } = useOHLC(stock.sym)
  const { predictions, loading: predLoading, error } = useAllPredictions()
  const macro = useMacro()
  
  const [isMobile, setIsMobile] = useState(window.innerWidth < 1000)
  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 1000)
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  const prediction = predictions ? predictions[stock.id] : null
  const isThinking = predLoading || !prediction

  const isBull = prediction?.prediction === 'Bullish'
  const isBear = prediction?.prediction === 'Bearish'

  if (isThinking && !error) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', alignItems: 'center', justifyContent: 'center', background: '#fff', fontFamily: 'Inter' }}>
        <div style={{ fontSize: '14px', fontWeight: '900', letterSpacing: '3px', color: '#000', marginBottom: '12px' }}>MARKETDRIVE AI</div>
        <div style={{ fontSize: '12px', color: '#888', letterSpacing: '0.5px' }}>Agents are gathering intelligence for {stock.t}...</div>
      </div>
    )
  }

  if (error && !prediction) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', alignItems: 'center', justifyContent: 'center', background: '#fff', color: 'red' }}>
        <div style={{ fontSize: '16px', fontWeight: '700' }}>Pipeline Error: {error}</div>
        <button onClick={() => navigate('/')} style={{ marginTop: '24px', padding: '12px 24px', cursor: 'pointer', background: '#000', color: '#fff', border: 'none', fontWeight: '700' }}>GO BACK</button>
      </div>
    )
  }

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: isMobile ? 'auto' : '100vh',
      overflowX: 'hidden',
      overflowY: isMobile ? 'auto' : 'hidden',
      background: 'white',
      fontFamily: "'Inter', sans-serif",
      color: '#000'
    }}>

      {/* ── Header ── */}
      <div style={{
        flexShrink: 0,
        padding: isMobile ? '16px 20px' : '24px 48px',
        display: 'flex',
        flexDirection: isMobile ? 'column' : 'row',
        gap: isMobile ? '16px' : '0',
        justifyContent: 'space-between',
        alignItems: isMobile ? 'flex-start' : 'center',
        borderBottom: '1.5px solid #000',
        background: 'white'
      }}>
        {/* Left */}
        <div style={{ display: 'flex', alignItems: 'center', gap: isMobile ? '12px' : '24px' }}>
          <button onClick={() => navigate('/')} style={{
            background: 'none', border: '1.5px solid #000', padding: '8px 18px',
            cursor: 'pointer', fontWeight: '900', fontSize: '12px', color: '#000',
            textTransform: 'uppercase', letterSpacing: '0.5px'
          }}>← BACK</button>
          <div style={{ fontSize: isMobile ? '22px' : '28px', fontWeight: '900', letterSpacing: '-0.5px' }}>{stock.t}</div>
          <span style={{
            fontSize: '11px', fontWeight: '900', padding: '6px 14px',
            textTransform: 'uppercase', letterSpacing: '1.5px',
            background: isBull ? '#000' : isBear ? '#333' : '#f0f0f0',
            color: (isBull || isBear) ? '#fff' : '#333'
          }}>{prediction.prediction}</span>
        </div>

        {/* Right Ticker Navigation */}
        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
          {STOCKS.map(s => (
            <button key={s.t} onClick={() => navigate(`/detail/${s.t}`)} style={{
              padding: '8px 18px', fontSize: '12px', fontWeight: s.t === stock.t ? '900' : '600',
              background: s.t === stock.t ? '#000' : 'white',
              color: s.t === stock.t ? '#fff' : '#666',
              border: '1.5px solid ' + (s.t === stock.t ? '#000' : '#eee'),
              cursor: 'pointer', transition: 'all 0.2s ease'
            }}>{s.t}</button>
          ))}
        </div>
      </div>

      {/* ── Body ── */}
      <div style={{ flex: 1, display: 'grid', gridTemplateColumns: isMobile ? '1fr' : '1fr 420px', minHeight: 0, overflow: 'hidden' }}>

        {/* Left Panel — scrollable */}
        <div style={{ overflowY: isMobile ? 'visible' : 'auto', padding: isMobile ? '24px 20px' : '40px 48px', borderRight: isMobile ? 'none' : '1.5px solid #f5f5f5' }}>

          {/* Expected Price + Parameters */}
          <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr', gap: '24px', marginBottom: '40px' }}>
            <div style={{ background: '#fafafa', padding: '24px', border: '1px solid #f0f0f0' }}>
              <div style={{ fontSize: '11px', fontWeight: '800', textTransform: 'uppercase', color: '#666', letterSpacing: '1.5px', marginBottom: '12px' }}>Target Projection</div>
              <div style={{ fontSize: isMobile ? '28px' : '32px', fontWeight: '900', letterSpacing: '-1.5px', marginBottom: '4px' }}>
                ₹{prediction.price_range_low} – {prediction.price_range_high}
              </div>
              <div style={{ fontSize: '12px', color: '#666', fontWeight: '500' }}>AI Volatility Corridor (ATR-Based)</div>
            </div>
            <div style={{ background: '#fafafa', padding: '24px', border: '1px solid #f0f0f0' }}>
              <div style={{ fontSize: '11px', fontWeight: '800', textTransform: 'uppercase', color: '#666', letterSpacing: '1.5px', marginBottom: '16px' }}>Technical Calibration</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                {[
                  { label: 'Back Acc', val: `${prediction.model_accuracy}%` },
                  { label: 'Model Conf', val: `${prediction.confidence}%` },
                  { label: 'Risk Factor', val: prediction.risk },
                ].map(m => (
                  <div key={m.label}>
                    <div style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase', letterSpacing: '1px', marginBottom: '4px' }}>{m.label}</div>
                    <div style={{ fontSize: '16px', fontWeight: '800' }}>{m.val}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Expert Analyst Reasoning */}
          <div style={{ marginBottom: '40px', padding: isMobile ? '24px' : '32px', border: '3.5px solid #000', position: 'relative' }}>
            <div style={{ position: 'absolute', top: '-12px', left: '24px', background: '#fff', padding: '0 12px', fontSize: '12px', fontWeight: '900', letterSpacing: '2px' }}>EXPERT ANALYST REASONING</div>
            <p style={{ fontSize: isMobile ? '14px' : '15px', lineHeight: '1.8', color: '#000', fontWeight: '500' }}>{prediction.explanation}</p>
          </div>

          {/* SHAP Interpretability */}
          <div style={{ marginBottom: '40px' }}>
            <div style={{ fontSize: '11px', fontWeight: '800', textTransform: 'uppercase', color: '#666', letterSpacing: '1.5px', marginBottom: '16px' }}>AI Model Interpretability (SHAP)</div>
            {prediction.top_features?.map(s => (
              <div key={s.feature} style={{ marginBottom: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '6px' }}>
                  <span style={{ color: '#000', fontWeight: '600' }}>{s.feature}</span>
                  <span style={{ fontWeight: '900', color: s.direction === 'supporting' ? '#000' : '#888' }}>
                    {s.shap_value > 0 ? '+' : ''}{s.shap_value.toFixed(3)}
                  </span>
                </div>
                <div style={{ height: '4px', background: '#f5f5f5' }}>
                  <div style={{ 
                    height: '100%', 
                    background: s.direction === 'supporting' ? '#000' : '#bbb', 
                    width: `${Math.min(100, Math.abs(s.shap_value) * 100)}%` 
                  }} />
                </div>
              </div>
            ))}
          </div>

          {/* Institutional News Feed */}
          <div style={{ marginBottom: '40px' }}>
            <div style={{ fontSize: '11px', fontWeight: '800', textTransform: 'uppercase', color: '#666', letterSpacing: '1.5px', marginBottom: '16px' }}>Institutional News Intelligence</div>
            {prediction.key_signals?.articles?.length > 0 ? (
              prediction.key_signals.articles.map((n, i) => (
                <div key={i} style={{ display: 'flex', gap: '16px', marginBottom: '16px', paddingBottom: '16px', borderBottom: '1px solid #f5f5f5' }}>
                  <span style={{ fontSize: '10px', padding: '4px 10px', background: '#000', color: '#fff', fontWeight: '800', flexShrink: 0, textTransform: 'uppercase', height: 'fit-content' }}>{n.dominant_event}</span>
                  <div style={{ display: 'flex', flexDirection: 'column' }}>
                    <span style={{ fontSize: '14px', color: '#000', fontWeight: '800', lineHeight: '1.4' }}>{n.headline}</span>
                    <span style={{ fontSize: '13px', color: '#555', marginTop: '6px', lineHeight: '1.6' }}>{n.summary}</span>
                  </div>
                </div>
              ))
            ) : (
              <div style={{ fontSize: '13px', color: '#666', fontStyle: 'italic' }}>No high-impact news detected today.</div>
            )}
            
            <div style={{ marginTop: '20px', padding: '16px', background: '#000', color: '#fff' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#666', marginBottom: '10px', textTransform: 'uppercase' }}>
                <span>Sentiment Velocity</span>
                <span style={{ fontWeight: '900', color: '#fff' }}>{prediction.key_signals?.sentiment_score?.toFixed(2)} · {prediction.key_signals?.dominant_event}</span>
              </div>
              <div style={{ height: '4px', background: '#333' }}>
                <div style={{ 
                  width: `${Math.round(((prediction.key_signals?.sentiment_score || 0 + 1) / 2) * 100)}%`, 
                  height: '100%', 
                  background: '#fff' 
                }} />
              </div>
            </div>
          </div>

          {/* Quantitative Indicators Grid */}
          <div>
            <div style={{ fontSize: '11px', fontWeight: '800', textTransform: 'uppercase', color: '#666', letterSpacing: '1.5px', marginBottom: '16px' }}>Quantitative Indicators</div>
            <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr 1fr' : 'repeat(3, 1fr)', gap: '12px' }}>
              {[
                { name: 'RSI', val: prediction.key_signals?.rsi?.toFixed(1) },
                { name: 'MACD', val: prediction.key_signals?.macd?.toFixed(2) },
                { name: 'EMA 20', val: prediction.key_signals?.ema_20?.toFixed(1) },
                { name: 'EMA 50', val: prediction.key_signals?.ema_50?.toFixed(1) },
                { name: 'Vol Ratio', val: prediction.key_signals?.volume_ratio?.toFixed(1) },
                { name: 'Nifty Trend', val: prediction.key_signals?.nifty_trend },
              ].map(ind => (
                <div key={ind.name} style={{ background: '#fafafa', padding: '16px', border: '1px solid #f0f0f0' }}>
                  <div style={{ fontSize: '10px', color: '#666', textTransform: 'uppercase', marginBottom: '6px' }}>{ind.name}</div>
                  <div style={{ fontSize: '18px', fontWeight: '800' }}>{ind.val}</div>
                </div>
              ))}
            </div>
          </div>

        </div>

        {/* Right Panel — Fixed Chart & Macros */}
        <div style={{ display: 'flex', flexDirection: 'column', overflowY: 'auto', padding: '32px', borderLeft: isMobile ? 'none' : '1.5px solid #f5f5f5' }}>

          {/* Live Chart Section */}
          <div style={{ height: '400px', flexShrink: 0, background: '#fafafa', padding: '24px', marginBottom: '24px', border: '1px solid #f0f0f0', display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div style={{ fontSize: '11px', fontWeight: '800', textTransform: 'uppercase', color: '#666' }}>Live OHLC · {stock.t}</div>
              <ChartToggle mode={mode} onChange={setMode} />
            </div>
            <div style={{ flex: 1, position: 'relative' }}>
              {ohlcLoading ? <div style={{ fontSize: '13px', color: '#666' }}>Loading...</div> : <OHLCChart data={data} mode={mode} color="#000" />}
            </div>
            {last && (
              <div style={{ display: 'flex', gap: '12px', marginTop: '16px' }}>
                 {[['O', last.o], ['H', last.h], ['L', last.l], ['C', last.c]].map(([k, v]) => (
                   <span key={k} style={{ fontSize: '12px', color: '#999', fontWeight: '600' }}>
                     {k} <strong style={{ color: '#000' }}>{v?.toFixed(1)}</strong>
                   </span>
                 ))}
              </div>
            )}
          </div>

          {/* Risk Disclaimer */}
          <div style={{ padding: '24px', background: '#000', color: '#fff', marginBottom: '24px' }}>
            <div style={{ fontSize: '12px', fontWeight: '900', letterSpacing: '1.5px', marginBottom: '12px' }}>⚠ RISK DISCLOSURE</div>
            <p style={{ fontSize: '12px', lineHeight: '1.7', color: '#666' }}>
              Multi-Agent AI Research system. Probabilistic predictions for informational use only. Trading involve substantial risk.
            </p>
          </div>

          {/* Macro Overlays */}
          <div style={{ marginTop: 'auto', borderTop: '2px solid #000', paddingTop: '24px' }}>
            <div style={{ fontSize: '11px', fontWeight: '800', textTransform: 'uppercase', color: '#666', letterSpacing: '1.5px', marginBottom: '16px' }}>Market Breadth (Macros)</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', minHeight: '120px' }}>
               {macro.map(m => (
                 <div key={m.label} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', opacity: m.value ? 1 : 0.6 }}>
                   <span style={{ fontSize: '12px', fontWeight: '800', textTransform: 'uppercase', color: '#000' }}>{m.label}</span>
                   <div style={{ textAlign: 'right' }}>
                      <div style={{ fontSize: '16px', fontWeight: '900', color: '#000' }}>
                        {m.value ? m.value.toLocaleString() : 'Loading...'}
                      </div>
                      {m.chgPct && (
                        <div style={{ fontSize: '12px', color: m.isUp ? '#000' : '#666', fontWeight: '900' }}>
                          {m.isUp ? '↑' : '↓'} {m.chgPct}%
                        </div>
                      )}
                   </div>
                 </div>
               ))}
            </div>
          </div>

        </div>
      </div>
    </div>
  )
}