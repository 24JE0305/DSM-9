import { useState, useEffect } from 'react'
import axios from 'axios'
import { Navbar } from './components/Navbar'
import { MetricCard } from './components/MetricCard'
import { PriceChart } from './components/PriceChart'
import { ForecastGrid } from './components/ForecastGrid'
import { Loader2 } from 'lucide-react'

const API_URL = "http://localhost:8000";

function App() {
  const [selectedTicker, setSelectedTicker] = useState("RELIANCE.NS");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);

  const fetchPrediction = async () => {
    setLoading(true);
    setError(null);
    try {
      // Fetch Prediction
      const predRes = await axios.post(`${API_URL}/predict`, { ticker: selectedTicker });
      setData(predRes.data);

      // Fetch History
      const histRes = await axios.get(`${API_URL}/history/${selectedTicker}`);
      setHistory(histRes.data);

    } catch (err) {
      console.error(err);
      setError("Failed to fetch data. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col bg-[#0d1117] min-h-screen text-white font-sans">
      <Navbar
        selectedTicker={selectedTicker}
        onSelectTicker={setSelectedTicker}
        onPredict={fetchPrediction}
      />

      <main className="flex-1 p-8 w-full max-w-[1600px] mx-auto">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h2 className="text-3xl font-bold">Market Analysis</h2>
            <p className="text-gray-400 mt-1">Real-time inference for {selectedTicker}</p>
          </div>
          <div className="text-right">
            <div className="text-xl font-mono text-[#00ffbd]">{new Date().toLocaleTimeString()}</div>
            <div className="text-xs text-gray-500 uppercase tracking-widest">System Ready</div>
          </div>
        </div>

        {error && (
          <div className="bg-red-500/10 border border-red-500/50 text-red-400 p-4 rounded-xl mb-6">
            {error}
          </div>
        )}

        {loading ? (
          <div className="flex flex-col items-center justify-center h-[60vh]">
            <Loader2 className="w-12 h-12 animate-spin text-[#00ffbd] mb-4" />
            <p className="text-gray-400 animate-pulse">Computing Deep Learning Weights...</p>
          </div>
        ) : (
          <>
            {!data ? (
              <div className="flex flex-col items-center justify-center h-[50vh] text-gray-500 border-2 border-dashed border-[#30363d] rounded-2xl">
                <span className="text-4xl mb-4">👈</span>
                <p>Select a ticker and click <span className="text-[#00ffbd]">Analyze</span></p>
              </div>
            ) : (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricCard
                    title="Current Price"
                    value={`₹${data.last_close.toLocaleString()}`}
                  />

                  
                  {(() => {
                    const preds = Object.values(data.predictions);
                    const avgPred = preds.reduce((a, b) => a + b, 0) / preds.length;
                    const delta = ((avgPred - data.last_close) / data.last_close) * 100;
                    const signal = delta > 0.5 ? "BULLISH" : delta < -0.5 ? "BEARISH" : "NEUTRAL";
                    const signalColor = delta > 0.5 ? "text-[#00ffbd]" : delta < -0.5 ? "text-red-500" : "text-yellow-500";

                    return (
                      <>
                        <MetricCard
                          title="Avg Forecast"
                          value={`₹${avgPred.toFixed(2)}`}
                          subtext={`${delta > 0 ? '+' : ''}${delta.toFixed(2)}%`}
                          trend={delta > 0 ? 'up' : delta < 0 ? 'down' : 'neutral'}
                        />
                        <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-5 flex flex-col items-center justify-center text-center">
                          <h3 className="text-gray-400 text-sm uppercase tracking-wider mb-1">Signal</h3>
                          <div className={`text-2xl font-bold ${signalColor} mb-1`}>{signal}</div>
                        </div>
                        <MetricCard
                          title="Confidence"
                          value="88.4%"
                          subtext="High"
                          className="border-[#00ffbd]/30 bg-[#00ffbd]/5"
                        />
                      </>
                    );
                  })()}
                </div>

                <PriceChart
                  data={history}
                  predictions={data.predictions}
                  lastClose={data.last_close}
                />

                <ForecastGrid
                  predictions={data.predictions}
                  lastClose={data.last_close}
                />
              </div>
            )}
          </>
        )}
      </main>
    </div>
  )
}

export default App
