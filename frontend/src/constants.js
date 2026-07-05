// constants.js
export const STOCKS = [
  { id: 'infosys',  t: 'INFY',     n: 'Infosys Ltd',      sym: 'INFY.NS'     },
  { id: 'vodafone', t: 'IDEA',     n: 'Vodafone Idea',    sym: 'IDEA.NS'     },
  { id: 'tata',     t: 'TMPV',     n: 'Tata Motors PV',   sym: 'TMCV.NS'     },
  { id: 'adani',    t: 'ADANIENT', n: 'Adani Enterprises', sym: 'ADANIENT.NS' },
  { id: 'yesbank',  t: 'YESBANK',  n: 'Yes Bank Ltd',     sym: 'YESBANK.NS'  },
]

export const sigClass = (s) => s === 'Bullish' ? 'bull' : s === 'Bearish' ? 'bear' : 'neut'
export const sigColor = (s) => s === 'Bullish' ? '#22c55e' : s === 'Bearish' ? '#ef4444' : '#f59e0b'
export const riskColor = (r) => r === 'Low' ? '#22c55e' : r === 'Medium' ? '#f59e0b' : '#ef4444'