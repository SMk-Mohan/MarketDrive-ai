import { useNavigate } from 'react-router-dom'
import { STOCKS } from '../constants'

export default function DetailNav({ activeTicker }) {
  const navigate = useNavigate()
  return (
    <div style={{ display:'flex', alignItems:'center', borderBottom:'1px solid var(--border)', background:'var(--bg2)', position:'sticky', top:0, zIndex:50, flexShrink:0 }}>
      <div
        onClick={() => navigate('/')}
        style={{ display:'flex', alignItems:'center', gap:5, padding:'10px 16px', borderRight:'1px solid var(--border)', cursor:'pointer', fontSize:12, color:'var(--text2)', whiteSpace:'nowrap', flexShrink:0, transition:'color .15s' }}
        onMouseEnter={e => e.currentTarget.style.color='var(--text)'}
        onMouseLeave={e => e.currentTarget.style.color='var(--text2)'}
      >
        <svg width="13" height="13" viewBox="0 0 13 13" fill="none">
          <path d="M8.5 2L3.5 6.5L8.5 11" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        Dashboard
      </div>
      <div style={{ display:'flex', flex:1, overflowX:'auto' }}>
        {STOCKS.map(s => (
          <div
            key={s.t}
            onClick={() => navigate(`/detail/${s.t}`)}
            style={{
              padding:'10px 16px', fontSize:11, fontWeight:500,
              cursor:'pointer', whiteSpace:'nowrap',
              borderRight:'1px solid var(--border)',
              borderBottom: s.t === activeTicker ? '2px solid var(--blue)' : '2px solid transparent',
              color: s.t === activeTicker ? 'var(--text)' : 'var(--text3)',
              transition:'all .15s',
            }}
          >
            {s.t}
          </div>
        ))}
      </div>
    </div>
  )
}