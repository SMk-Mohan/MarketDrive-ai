import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useOHLC } from '../hooks'
import OHLCChart from './OHLCChart'

export default function StockRow({ stock, prediction, loading: isGlobalLoading, index = 0 }) {
  const navigate = useNavigate()
  const { data: ohlc, loading: isChartLoading, last } = useOHLC(stock.sym)
  const [isMobile, setIsMobile] = useState(window.innerWidth < 900)

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 900)
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  // Use the REAL agent data if it exists, otherwise show a "Processing" state
  const isThinking = isGlobalLoading || !prediction
  
  const predSignal = prediction?.prediction || '...'
  const isBull     = predSignal === 'Bullish'
  const isBear     = predSignal === 'Bearish'

  const headerBg    = isThinking ? '#f5f5f5' : isBull ? '#000' : isBear ? '#3a3a3a' : '#888'
  const headerTextColor = isThinking ? '#aaa' : '#fff'
  const outerBorder = isThinking ? '1px dashed #ddd' : isBull ? '1.5px solid #000' : isBear ? '1.5px solid #555' : '1.5px solid #aaa'

  const isFlipped = index % 2 !== 0 && !isMobile // No flip on mobile

  const hoverOn = e => {
    if (isThinking || isMobile) return
    e.currentTarget.style.boxShadow = '0 6px 28px rgba(0,0,0,0.13)'
    e.currentTarget.style.transform = 'translateY(-2px)'
  }
  const hoverOff = e => {
    e.currentTarget.style.boxShadow = 'none'
    e.currentTarget.style.transform = 'translateY(0)'
  }

  /* ── Info rectangle ── */
  const InfoBox = (
    <div
      style={{
        border: outerBorder,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        transition: 'box-shadow 0.18s ease, transform 0.15s ease',
        opacity: isThinking ? 0.7 : 1,
        minHeight: isMobile ? 'auto' : '44vh'
      }}
      onMouseEnter={hoverOn}
      onMouseLeave={hoverOff}
    >
      {/* Header */}
      <div style={{
        background: headerBg,
        padding: isMobile ? '12px 16px' : '10px 16px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        flexShrink: 0,
      }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '10px' }}>
          <span style={{ fontSize: isMobile ? '16px' : '18px', fontWeight: '900', color: headerTextColor, letterSpacing: '-0.3px' }}>
            {stock.t}
          </span>
          <span style={{ fontSize: '10px', color: isThinking ? '#ccc' : 'rgba(255,255,255,0.5)', fontWeight: '500' }}>
            {stock.n}
          </span>
        </div>
        <span style={{
          fontSize: '10px', fontWeight: '800', letterSpacing: '1.2px', textTransform: 'uppercase',
          color: isThinking ? '#ccc' : 'rgba(255,255,255,0.75)', 
          border: `1px solid ${isThinking ? '#ddd' : 'rgba(255,255,255,0.3)'}`, 
          padding: '4px 10px',
        }}>
          {isThinking ? 'THINKING' : predSignal}
        </span>
      </div>

      {/* Body */}
      <div style={{ padding: isMobile ? '16px' : '18px 20px', display: 'flex', flexDirection: 'column', gap: '16px', flex: 1 }}>

        {/* Price row */}
        <div style={{ display: 'flex', gap: isMobile ? '12px' : '24px', alignItems: 'flex-end', flexWrap: isMobile ? 'wrap' : 'nowrap' }}>
          <div>
            <div style={{ fontSize: '10px', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '1px', color: '#666', marginBottom: '4px' }}>Range</div>
            <div style={{ fontSize: isMobile ? '20px' : '26px', fontWeight: '900', letterSpacing: '-1px', lineHeight: 1 }}>
              {isThinking ? '...' : `₹${prediction.price_range_low} – ${prediction.price_range_high}`}
            </div>
          </div>
          {last && (
            <div>
              <div style={{ fontSize: '10px', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '1px', color: '#666', marginBottom: '4px' }}>Now</div>
              <div style={{ fontSize: isMobile ? '20px' : '26px', fontWeight: '700', color: '#333', letterSpacing: '-0.4px', lineHeight: 1 }}>₹{last.c.toFixed(1)}</div>
            </div>
          )}
          <div style={{ marginLeft: isMobile ? '0' : 'auto', display: 'flex', gap: '8px', width: isMobile ? '100%' : 'auto' }}>
            <div style={{ flex: 1, border: '1px solid #f0f0f0', padding: '6px 10px', textAlign: 'center', background: '#fafafa' }}>
              <div style={{ fontSize: '9px', fontWeight: '800', textTransform: 'uppercase', color: '#666' }}>Conf</div>
              <div style={{ fontSize: '14px', fontWeight: '900' }}>{isThinking ? '...' : prediction.confidence}%</div>
            </div>
            <div style={{ flex: 1, border: '1px solid #f0f0f0', padding: '6px 10px', textAlign: 'center', background: '#fafafa' }}>
              <div style={{ fontSize: '9px', fontWeight: '800', textTransform: 'uppercase', color: '#666' }}>Acc</div>
              <div style={{ fontSize: '14px', fontWeight: '900' }}>{isThinking ? '...' : prediction.model_accuracy}%</div>
            </div>
          </div>
        </div>

        {!isMobile && <div style={{ height: '1.5px', background: '#f5f5f5' }} />}

        {/* Dynamic Signals */}
        <div style={{ display: 'flex', gap: '15px', flexDirection: isMobile ? 'column' : 'row' }}>
           <div style={{ flex: 1 }}>
              <div style={{ fontSize: '10px', fontWeight: '800', textTransform: 'uppercase', color: '#666', marginBottom: '6px' }}>Signals</div>
              <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                 {[
                   { l: 'RSI', v: prediction?.key_signals?.rsi?.toFixed(0) },
                   { l: 'MACD', v: prediction?.key_signals?.macd?.toFixed(1) },
                   { l: 'SENT', v: prediction?.key_signals?.sentiment_score?.toFixed(1) }
                 ].map(s => (
                   <div key={s.l} style={{ fontSize: '11px', background: '#000', color: '#fff', padding: '4px 10px', fontWeight: '800' }}>
                     <span style={{ color: '#aaa', marginRight: '4px' }}>{s.l}</span>{isThinking ? '...' : s.v}
                   </div>
                 ))}
              </div>
           </div>
        </div>

        <div style={{ height: '1.5px', background: '#f5f5f5' }} />

        {/* Reasoning */}
        <div>
          <div style={{ fontSize: '10px', fontWeight: '900', textTransform: 'uppercase', color: '#000', marginBottom: '6px' }}>
            Reasoning
          </div>
          <div style={{ fontSize: '13px', color: '#000', lineHeight: 1.6, fontWeight: '500' }}>
            {isThinking ? 'Agent analyzing...' : prediction.explanation}
          </div>
        </div>
      </div>
    </div>
  )

  /* ── Chart rectangle ── */
  const ChartBox = (
    <div
      style={{
        border: outerBorder,
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        background: '#fff',
        opacity: isThinking ? 0.7 : 1,
        height: isMobile ? '200px' : 'auto'
      }}
    >
      <div style={{
        background: headerBg,
        padding: '8px 14px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
      }}>
        <span style={{ fontSize: '8px', fontWeight: '800', textTransform: 'uppercase', color: isThinking ? '#ccc' : 'rgba(255,255,255,0.7)' }}>
          Real-Time Chart
        </span>
        {last && (
          <span style={{ fontSize: '11px', fontWeight: '900', color: headerTextColor }}>
            ₹{last.c.toFixed(1)}
          </span>
        )}
      </div>

      <div style={{ flex: 1, padding: '10px', background: '#fafafa', minHeight: 0 }}>
        {isChartLoading
          ? <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', fontSize: '10px', color: '#ccc' }}>...</div>
          : <OHLCChart data={ohlc} mode="line" color="#000" />
        }
      </div>
    </div>
  )

  return (
    <div
      onClick={() => !isThinking && navigate(`/detail/${stock.t}`)}
      style={{
        display: 'grid',
        gridTemplateColumns: isMobile ? '1fr' : (isFlipped ? '240px 1fr' : '1fr 240px'),
        gap: isMobile ? '0' : '12px',
        cursor: isThinking ? 'wait' : 'pointer',
        marginBottom: isMobile ? '32px' : '0'
      }}
    >
      {isMobile ? (
        <React.Fragment>{InfoBox}{ChartBox}</React.Fragment>
      ) : (
        isFlipped ? <React.Fragment>{ChartBox}{InfoBox}</React.Fragment> : <React.Fragment>{InfoBox}{ChartBox}</React.Fragment>
      )}
    </div>
  )
}