export default function WhyCard({ why, ev, sigColor }) {
  return (
    <div className="section-card">
      <div className="section-head">
        <span className="section-title">Why this prediction</span>
      </div>
      <div className="section-body">
        <p style={{ fontSize:12, color:'var(--text2)', lineHeight:1.7, marginBottom:10 }}>
          {why}
        </p>
        <div>
          {ev.map((e, i) => (
            <div key={i} style={{ display:'flex', alignItems:'flex-start', gap:7, padding:'5px 0', borderBottom: i < ev.length-1 ? '1px solid var(--border)' : 'none' }}>
              <span style={{ width:5, height:5, borderRadius:'50%', background:sigColor, flexShrink:0, marginTop:4, display:'inline-block' }}/>
              <span style={{ fontSize:12, color:'var(--text2)', lineHeight:1.5 }}>{e}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}