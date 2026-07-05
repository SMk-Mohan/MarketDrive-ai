import { useMacro } from '../hooks'

export default function MacroBar() {
  const macro = useMacro()
  return (
    <div style={{ display:'flex', borderBottom:'1px solid var(--border)' }}>
      {macro.map((m, i) => (
        <div key={m.label} style={{ flex:1, padding:'10px 16px', borderRight: i < macro.length-1 ? '1px solid var(--border)' : 'none', display:'flex', alignItems:'center', gap:10 }}>
          <div style={{ fontSize:10, color:'var(--text3)', textTransform:'uppercase', letterSpacing:'.6px', minWidth:70 }}>{m.label}</div>
          <div style={{ fontSize:14, fontWeight:600, color:'var(--text)' }}>
            {m.value ? m.value.toLocaleString('en-IN', { maximumFractionDigits:0 }) : '—'}
          </div>
          {m.chgPct && (
            <div style={{ fontSize:11, color: m.isUp ? 'var(--green)' : 'var(--red)' }}>
              {m.isUp ? '↑ +' : '↓ '}{m.chgPct}%
            </div>
          )}
        </div>
      ))}
    </div>
  )
}