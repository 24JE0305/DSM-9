import { Routes, Route } from 'react-router-dom'
import Home from './pages/Home'
import Screener from './pages/Screener'
import Backtest from './pages/Backtest'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/screener" element={<Screener />} />
      <Route path="/backtest" element={<Backtest />} />
    </Routes>
  )
}

export default App
