import { sigClass } from '../constants'

export default function PastPredictions({ past }) {
  return (
    <div className="section-card">
      <div className="section-head">
        <span className="section-title">Past predictions</span>
        <span style={{ fontSize:9, color:'var(--text3)' }}>+ why it failed</span>
      </div>
      <div className="section-body">
        <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:8 }}>
          {past.map((p, i) => (
            <div key={i} style={{ background:'var(--bg3)', border:'1px solid var(--border)', borderRadius:7, padding:10, textAlign:'center' }}>
              <div style={{ fontSize:9, color:'var(--text3)', marginBottom:5 }}>{p.d}</div>
              <span className={`sig-pill ${sigClass(p.s)}`}>{p.s}</span>
              <div style={{ fontSize:10, fontWeight:500, margin:'4px 0 6px', color: p.failed ? 'var(--red)' : 'var(--green)' }}>
                {p.outcome}
              </div>
              {p.failed ? (
                <div style={{ background:'var(--bg)', border:'1px solid #ef444422', borderRadius:6, padding:'7px 8px', textAlign:'left' }}>
                  <div style={{ fontSize:9, color:'var(--red)', textTransform:'uppercase', letterSpacing:'.5px', marginBottom:4 }}>
                    Why it failed
                  </div>
                  <div style={{ fontSize:10, color:'var(--text2)', lineHeight:1.5 }}>{p.failReason}</div>
                </div>
              ) : (
                <div style={{ fontSize:10, color:'var(--text3)' }}>Prediction accurate</div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}