export default function SHAPCard({ shap }) {
  return (
    <div className="section-card">
      <div className="section-head">
        <span className="section-title">SHAP feature importance</span>
        <span style={{ fontSize:9, color:'var(--text3)' }}>what drove this prediction</span>
      </div>
      <div className="section-body">
        {shap.map((f, i) => {
          const color = f.p ? '#22c55e' : '#ef4444'
          return (
            <div key={i} style={{ display:'flex', alignItems:'center', gap:8, marginBottom: i < shap.length-1 ? 8 : 0 }}>
              <span style={{ flex:'0 0 100px', fontSize:10, color:'var(--text3)', textAlign:'right' }}>
                {f.f}
              </span>
              <div style={{ flex:1, height:7, background:'var(--bg3)', borderRadius:3, overflow:'hidden' }}>
                <div style={{ width:`${Math.round(f.v * 260)}%`, height:'100%', background:color, borderRadius:3, transition:'width .6s ease' }}/>
              </div>
              <span style={{ flex:'0 0 30px', fontSize:10, fontWeight:600, textAlign:'right', color }}>
                {f.v.toFixed(2)}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}