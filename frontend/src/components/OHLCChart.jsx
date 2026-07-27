import { useEffect, useRef } from 'react'

function drawLine(ctx, data, W, H) {
  const p = { l: 6, r: 6, t: 8, b: 8 }
  const cw = W - p.l - p.r, ch = H - p.t - p.b
  const prices = data.flatMap(d => [d.h, d.l])
  const min = Math.min(...prices), max = Math.max(...prices), rng = max - min || 1
  const y = v => p.t + ch * (1 - (v - min) / rng)
  const n = data.length

  ctx.clearRect(0, 0, W, H)
  ctx.beginPath()
  data.forEach((d, i) => {
    const x = p.l + (i / (n - 1)) * cw
    i === 0 ? ctx.moveTo(x, y(d.c)) : ctx.lineTo(x, y(d.c))
  })
  ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5; ctx.stroke()

  // Subtle fill
  ctx.lineTo(p.l + cw, H); ctx.lineTo(p.l, H); ctx.closePath()
  ctx.fillStyle = 'rgba(0,0,0,0.04)'; ctx.fill()
}

function drawCandle(ctx, data, W, H) {
  const p = { l: 6, r: 6, t: 8, b: 8 }
  const cw = W - p.l - p.r, ch = H - p.t - p.b
  const prices = data.flatMap(d => [d.h, d.l])
  const min = Math.min(...prices), max = Math.max(...prices), rng = max - min || 1
  const y = v => p.t + ch * (1 - (v - min) / rng)
  const n = data.length
  const bw = Math.max(1, Math.floor(cw / n) - 2)

  ctx.clearRect(0, 0, W, H)
  data.forEach((d, i) => {
    const x = p.l + (i / (n - 1)) * cw
    const isUp = d.c >= d.o
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1
    ctx.beginPath(); ctx.moveTo(x, y(d.h)); ctx.lineTo(x, y(d.l)); ctx.stroke()
    if (isUp) {
      ctx.fillStyle = '#000'
    } else {
      ctx.fillStyle = '#fff'
    }
    const rectY = Math.min(y(d.o), y(d.c))
    const rectH = Math.abs(y(d.c) - y(d.o)) || 1
    ctx.fillRect(x - bw / 2, rectY, bw, rectH)
    if (!isUp) ctx.strokeRect(x - bw / 2, rectY, bw, rectH)
  })
}

export default function OHLCChart({ data, mode, color = '#000' }) {
  const containerRef = useRef(null)
  const canvasRef = useRef(null)

  useEffect(() => {
    const container = containerRef.current
    const canvas = canvasRef.current
    if (!container || !canvas || !data?.length) return

    let rafId = null

    const render = () => {
      // Read parent layout size BEFORE touching canvas resolution
      const W = container.clientWidth
      const H = container.clientHeight
      if (W === 0 || H === 0) return

      // Only update resolution if changed (absolutely positioned canvas won't trigger parent resize)
      if (canvas.width !== W || canvas.height !== H) {
        canvas.width = W
        canvas.height = H
      }
      const ctx = canvas.getContext('2d')
      mode === 'line' ? drawLine(ctx, data, W, H) : drawCandle(ctx, data, W, H)
    }

    render()

    const ro = new ResizeObserver(() => {
      if (rafId) cancelAnimationFrame(rafId)
      rafId = requestAnimationFrame(render)
    })
    ro.observe(container)

    return () => {
      ro.disconnect()
      if (rafId) cancelAnimationFrame(rafId)
    }
  }, [data, mode])

  if (!data?.length) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', fontSize: 11, color: '#ccc' }}>
      No data
    </div>
  )

  return (
    <div ref={containerRef} style={{ position: 'relative', width: '100%', height: '100%', minHeight: 0, overflow: 'hidden' }}>
      <canvas ref={canvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', display: 'block' }} />
    </div>
  )
}