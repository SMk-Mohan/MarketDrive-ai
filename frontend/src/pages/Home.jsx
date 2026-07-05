import Hero from '../components/Hero'
import StockRow from '../components/StockRow'
import { STOCKS } from '../constants'
import { useAllPredictions } from '../hooks'

export default function Home() {
  const { predictions, loading, error } = useAllPredictions()

  return (
    <div style={{ background: 'white', minHeight: '100vh', fontFamily: "'Inter', sans-serif" }}>
      <Hero />

      {/* Section header */}
      <div id="stocks-section" style={{ padding: '60px 5% 24px' }}>
        <div style={{ fontSize: '14px', fontWeight: '800', letterSpacing: '2.5px', textTransform: 'uppercase', color: '#888', marginBottom: '8px' }}>
          Analyst Intelligence Feed
        </div>
        <div style={{ fontSize: '12px', color: '#bbb', letterSpacing: '0.5px', fontWeight: '500' }}>
          {loading ? 'Agents are processing live market data...' : 'Live multi-agent consensus for NSE top equities'}
        </div>
      </div>

      {/* Stock Cards Grid */}
      <div style={{
        padding: '0 5% 100px',
        display: 'flex',
        flexDirection: 'column',
        gap: '32px',
      }}>
        {STOCKS.map((stock, i) => {
          const predData = predictions ? predictions[stock.id] : null
          return (
            <StockRow
              key={stock.t}
              stock={stock}
              prediction={predData}
              loading={loading}
              index={i}
            />
          )
        })}

        {error && (
          <div style={{ padding: '20px', color: 'red', textAlign: 'center', fontSize: '12px' }}>
            Agent Pipeline Error: {error}
          </div>
        )}
      </div>
    </div>
  )
}