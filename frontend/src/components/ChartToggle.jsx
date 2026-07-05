export default function ChartToggle({ mode, onChange }) {
  return (
    <div style={{ display: 'flex', border: '2px solid black', borderRadius: '4px', overflow: 'hidden' }}>
      {['line', 'candle'].map(m => (
        <button
          key={m}
          onClick={() => onChange(m)}
          style={{
            padding: '4px 8px',
            fontSize: '9px',
            fontWeight: 'bold',
            textTransform: 'uppercase',
            border: 'none',
            outline: 'none',
            cursor: 'pointer',
            background: mode === m ? 'black' : 'white',
            color: mode === m ? 'white' : 'black',
            transition: 'none'
          }}
        >
          {m}
        </button>
      ))}
    </div>
  )
}