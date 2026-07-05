import { useEffect, useRef } from 'react'

function Gauge({ pct, color, label, size = 96 }) {
  const ref = useRef(null)

  useEffect(() => {
    const svg = ref.current
    if (!svg) return
    const R = size * 0.38, cx = size / 2, cy = size * 0.6
    const sx = cx - R, ex = cx + R
    const fa = Math.PI + (pct / 100) * Math.PI
    const fx = cx + R * Math.cos(fa), fy = cy + R * Math.sin(fa)
    const la = pct > 50 ? 1 : 0
    const nx = cx + (R - 8) * Math.cos(fa), ny = cy + (R - 8) * Math.sin(fa)
    const ticks = Array.from({ length: 9 }, (_, i) => {
      const a = Math.PI + (i / 8) * Math.PI
      const r1 = R - 5, r2 = R - 1
      return `<line x1="${(cx + r1*Math.cos(a)).toFixed(1)}" y1="${(cy + r1*Math.sin(a)).toFixed(1)}" x2="${(cx + r2*Math.cos(a)).toFixed(1)}" y2="${(cy + r2*Math.sin(a)).toFixed(1)}" stroke="#ffffff18" stroke-width="1"/>`
    }).join('')
    svg.setAttribute('viewBox', `0 0 ${size} ${Math.round(size * 0.68)}`)
    svg.setAttribute('width', size)
    svg.setAttribute('height', Math.round(size * 0.68))
    svg.innerHTML = `
      ${ticks}
      <path d="M${sx},${cy} A${R},${R} 0 1,1 ${ex},${cy}" fill="none" stroke="#ffffff10" stroke-width="5" stroke-linecap="round"/>
      <path d="M${sx},${cy} A${R},${R} 0 ${la},1 ${fx.toFixed(1)},${fy.toFixed(1)}" fill="none" stroke="${color}" stroke-width="5" stroke-linecap="round"/>
      <line x1="${cx}" y1="${cy}" x2="${nx.toFixed(1)}" y2="${ny.toFixed(1)}" stroke="${color}" stroke-width="1.5" stroke-linecap="round"/>
      <circle cx="${cx}" cy="${cy}" r="3" fill="${color}"/>
      <text x="${cx}" y="${(cy-5).toFixed(1)}" text-anchor="middle" font-size="12" font-weight="600" fill="#f0f0f5">${Math.round(pct)}%</text>
      <text x="${cx}" y="${(cy+10).toFixed(1)}" text-anchor="middle" font-size="8" fill="#5a5a70">${label}</text>
    `
  }, [pct, color, size, label])

  return <svg ref={ref} />
}

export default function GaugeCard({ conf, risk, vol, sigColor, riskColor }) {
  const gauges = [
    { pct: conf, color: sigColor,  label: 'CONF' },
    { pct: risk, color: riskColor, label: 'RISK' },
    { pct: vol,  color: '#818cf8', label: 'VOL'  },
  ]
  return (
    <div className="section-card">
      <div className="section-head">
        <span className="section-title">Speedometers</span>
      </div>
      <div className="section-body">
        <div style={{ display:'flex', justifyContent:'space-around', gap:8 }}>
          {gauges.map(g => (
            <div key={g.label} style={{ textAlign:'center' }}>
              <Gauge pct={g.pct} color={g.color} label={g.label} />
              <div style={{ fontSize:9, color:'var(--text3)', textTransform:'uppercase', letterSpacing:'.5px', marginTop:3 }}>
                {g.label === 'CONF' ? 'Confidence' : g.label === 'RISK' ? 'Risk' : 'Volatility'}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}