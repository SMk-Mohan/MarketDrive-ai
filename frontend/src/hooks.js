import { useState, useEffect, useRef } from 'react'
import { getOHLC, getMacroIndex, getAllPredictions } from './api'

// ── useOHLC ──
export function useOHLC(sym) {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)

  useEffect(() => {
    if (!sym) return
    setLoading(true)
    setError(null)
    getOHLC(sym)
      .then(d  => { setData(d);        setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [sym])

  const last   = data?.[data.length - 1] ?? null
  const first  = data?.[0] ?? null
  const chgPct = last && first ? (((last.c - first.o) / first.o) * 100).toFixed(2) : null
  const isUp   = last && first ? last.c >= first.o : null

  return { data, loading, error, last, chgPct, isUp }
}

// ── useAllPredictions ──
export function useAllPredictions() {
  const [predictions, setPredictions] = useState(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    let mounted = true
    let timer = null

    const fetchAll = (isManual = false) => {
      if (isManual) setLoading(true)
      getAllPredictions()
        .then(res => {
          if (!mounted) return
          setPredictions(res.data)
          setRefreshing(res.is_refreshing)
          setLoading(false)

          // If background agents are still running, poll again in 5s
          if (res.is_refreshing) {
            timer = setTimeout(() => fetchAll(false), 5000)
          }
        })
        .catch(err => {
          if (!mounted) return
          setError(err.message)
          setLoading(false)
        })
    }

    fetchAll(true)

    return () => {
      mounted = false
      if (timer) clearTimeout(timer)
    }
  }, [])

  return { predictions, loading, refreshing, error }
}

// ── useMacro ──
const INDICES = [
  { label:'NIFTY 50',  sym:'^NSEI'    },
  { label:'SENSEX',    sym:'^BSESN'   },
  { label:'BANKNIFTY', sym:'^NSEBANK' },
]

export function useMacro() {
  const [macro, setMacro] = useState(
    INDICES.map(i => ({ ...i, value:null, chgPct:null, isUp:null }))
  )

  useEffect(() => {
    INDICES.forEach(async (idx, i) => {
      try {
        const result = await getMacroIndex(idx.sym)
        setMacro(prev => {
          const next = [...prev]
          next[i] = { ...next[i], ...result }
          return next
        })
      } catch (_) {}
    })
  }, [])

  return macro
}

// ── useScrollAnimation ──
export function useScrollAnimation(itemCount, delay = 80) {
  const refs = useRef([])

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            const i = parseInt(entry.target.dataset.idx, 10)
            setTimeout(() => entry.target.classList.add('visible'), i * delay)
            observer.unobserve(entry.target)
          }
        })
      },
      { threshold: 0.1 }
    )
    refs.current.forEach(el => { if (el) observer.observe(el) })
    return () => observer.disconnect()
  }, [itemCount, delay])

  const setRef = (el, i) => {
    if (el) { el.dataset.idx = i; refs.current[i] = el }
  }

  return { setRef }
}