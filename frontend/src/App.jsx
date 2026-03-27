import { useState, useEffect } from 'react'
import axios from 'axios'
import { Navbar } from './components/Navbar'
import { Hero } from './components/Hero'
import { About } from './components/About'
import { Feedback } from './components/Feedback'
import { Footer } from './components/Footer'
import { MetricCard } from './components/MetricCard'
import { PriceChart } from './components/PriceChart'
import { ForecastGrid } from './components/ForecastGrid'
import { AnalystGauge } from './components/AnalystGauge'
import { Loader2, Search, Activity } from 'lucide-react'

const API_URL = "https://s8world7-backend-back.hf.space";

function App() {
  const [selectedTicker, setSelectedTicker] = useState("RELIANCE.NS");
  const [tickers, setTickers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);

  // Fetch available tickers on mount
  useEffect(() => {
    async function fetchTickers() {
      try {
        const res = await axios.get(`${API_URL}/tickers`);
        if (res.data.tickers) {
          setTickers(res.data.tickers);
        } else if (Array.isArray(res.data)) {
          setTickers(res.data);
        } else {
          setTickers(["ITC.NS", "RELIANCE.NS"]);
        }
      } catch (e) {
        console.error("Failed to fetch tickers", e);
        setTickers(["ITC.NS", "RELIANCE.NS"]);
      }
    }
    fetchTickers();
  }, []);

  const fetchPrediction = async () => {
    setLoading(true);
    setError(null);
    try {
      const histRes = await axios.get(`${API_URL}/history/${selectedTicker}`);
      setHistory(histRes.data);

      const historyData = histRes.data;
      const lastClose = historyData.length > 0 ? historyData[historyData.length - 1].Close : 100;

      const predRes = await axios.get(`${API_URL}/predict_v3/${selectedTicker}`);
      const rawPredictions = predRes.data.predictions;

      // Transform v3 returns to expected price formats
      const mappedPredictions = {};
      Object.entries(rawPredictions).forEach(([period, details]) => {
        const daysLabel = period.replace('_days', 'D'); // e.g. "15_days" -> "15D"
        mappedPredictions[daysLabel] = lastClose * (1 + details.expected_return);
      });

      setData({
        last_close: lastClose,
        predictions: mappedPredictions,
        model_info: predRes.data.model_info,
        risk_metrics: predRes.data.risk_metrics,
      });

    } catch (err) {
      console.error(err);
      setError("Failed to fetch data. Ensure backend is running and Model 3 has data for this ticker.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col bg-[#05060a] min-h-screen text-white font-sans relative">
      {/* Universal Fading Background Visual Effects */}
      <div
        className="fixed top-0 left-0 w-full h-screen pointer-events-none overflow-hidden z-0"
      >
        <div className="absolute top-[10%] left-1/4 w-[600px] h-[600px] bg-purple-600 opacity-10 rounded-full blur-[120px] mix-blend-screen"></div>
        <div className="absolute top-[40%] right-1/4 w-[600px] h-[600px] bg-cyan-600 opacity-10 rounded-full blur-[120px] mix-blend-screen"></div>
        <div className="absolute bottom-[-10%] left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-pink-600 opacity-10 rounded-full blur-[150px] mix-blend-screen"></div>
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4IiBoZWlnaHQ9IjgiPgo8cmVjdCB3aWR0aD0iOCIgaGVpZ2h0PSI4IiBmaWxsPSIjZmZmIiBmaWxsLW9wYWNpdHk9IjAuMDMiLz4KPHBhdGggZD0iTTAgMEw4IDhaTTAgOEw4IDBaIiBzdHJva2U9IiNmZmYiIHN0cm9rZS1vcGFjaXR5PSIwLjA1IiBzdHJva2Utd2lkdGg9IjEiLz4KPC9zdmc+')] opacity-20"></div>
      </div>

      <div className="absolute top-0 left-0 right-0 z-50 pointer-events-none">
        {/* Pointer events none on the wrapper so you can still click the background */}
        <div className="pointer-events-auto">
          <Navbar />
        </div>
      </div>

      <Hero />

      <section id="dashboard" className="pt-24 pb-16">
        <main className="w-full max-w-[1600px] mx-auto px-8">

          {/* Dashboard Header */}
          <div className="flex flex-col items-center justify-center mb-16 text-center pt-8 relative z-10">
            <h2 className="text-4xl md:text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-pink-500 via-purple-400 to-cyan-400 uppercase tracking-wider drop-shadow-[0_0_15px_rgba(236,72,153,0.3)] pb-2">
              Predictive Analytics Terminal
            </h2>
            <div className="flex flex-col md:flex-row items-center gap-4 mt-2">
              <p className="text-gray-400 text-lg">Powered by our V3 Architecture: Transformer, BiLSTM, XGBoost & Learned Fusion</p>
              <button
                onClick={(e) => { e.preventDefault(); document.getElementById('about-section')?.scrollIntoView({ behavior: 'smooth' }); }}
                className="text-xs bg-[#2a2a4a] hover:bg-[#3a3a5a] text-white px-3 py-1.5 rounded-md border border-purple-500/30 transition-colors"
              >
                Inside the Algorithm
              </button>
            </div>
            {data && data.model_info && data.model_info.last_retrained && (
              <span className="mt-4 px-4 py-1.5 bg-[#111326] border border-[#2a2a4a] rounded-full text-xs text-purple-400 font-semibold tracking-wider">
                LAST RETRAINED: {data.model_info.last_retrained.split('T')[0]}
              </span>
            )}
          </div>

          {error && (
            <div className="bg-red-500/10 border border-red-500/50 text-red-400 p-4 rounded-xl mb-6">
              {error}
            </div>
          )}

          {loading ? (
            <div className="flex flex-col items-center justify-center h-[60vh] relative z-10">
              <Loader2 className="w-12 h-12 animate-spin text-purple-500 mb-4 drop-shadow-[0_0_10px_rgba(168,85,247,0.8)]" />
              <p className="text-gray-300 font-medium animate-pulse">Aggregating market data and executing hybrid inference for {selectedTicker}...</p>
            </div>
          ) : (
            <>
              {!data ? (
                <div className='flex flex-col items-center justify-center'>
                  <div className="flex flex-col items-center justify-center h-[30vh] w-[50vw] text-gray-400 border-2 border-dashed border-[#2a2a4a] bg-[#111326]/40 backdrop-blur-sm rounded-2xl mb-10 relative z-10 shadow-lg">
                    <Activity className="w-12 h-12 text-gray-500 mb-4 animate-pulse" />
                    <p className="font-medium text-center px-4">Select an NSE asset below and initialize the engine to generate an AI-driven forecast.</p>
                  </div>
                </div>

              ) : (
                <div className="space-y-12 animate-in fade-in slide-in-from-bottom-4 duration-500 mb-16 relative z-10">

                  <PriceChart
                    data={history}
                    predictions={data.predictions}
                    lastClose={data.last_close}
                  />

                  {/* Horizontally Centered Metric Row */}
                  <div className="flex flex-col md:flex-row justify-center items-stretch gap-6 w-full">
                    <MetricCard
                      title="Current Price"
                      value={`₹${data.last_close.toLocaleString()}`}
                    />

                    {(() => {
                      const preds = Object.values(data.predictions);
                      const avgPred = preds.reduce((a, b) => a + b, 0) / preds.length;
                      const delta = ((avgPred - data.last_close) / data.last_close) * 100;

                      let computedScore = ((delta + 20) / 40) * 100;
                      if (computedScore < 0) computedScore = 5;
                      if (computedScore > 100) computedScore = 95;

                      return (
                        <>
                          {/* Centered Gauge */}
                          <div className="md:px-12">
                            <AnalystGauge score={computedScore} analystsCount={33} />
                          </div>

                          <MetricCard
                            title="Expected Price"
                            value={`₹${avgPred.toFixed(2)}`}
                            subtext={`${delta > 0 ? '+' : ''}${delta.toFixed(2)}%`}
                            trend={delta > 0 ? 'up' : delta < 0 ? 'down' : 'neutral'}
                          />

                          {data.risk_metrics && (
                            <MetricCard
                              title="Risk & Model Agreement"
                              value={(data.risk_metrics.model_agreement * 100).toFixed(0) + "%"}
                              subtext={`Volatility Level: ${data.risk_metrics.volatility_level}`}
                              trend="neutral"
                            />
                          )}
                        </>
                      );
                    })()}
                  </div>



                  <ForecastGrid
                    predictions={data.predictions}
                    lastClose={data.last_close}
                  />
                </div>
              )}

              {/* Ticker Selector & Analyze Button moved below dashboard components */}
              <div className="flex justify-center w-full pb-12 relative z-20">
                <div className="flex flex-wrap justify-center items-center gap-4 bg-[#111326]/80 backdrop-blur-xl border border-[#2a2a4a] p-4 rounded-2xl shadow-[0_0_30px_rgba(0,0,0,0.5)]">
                  <div className="relative">
                    <select
                      value={selectedTicker}
                      onChange={(e) => setSelectedTicker(e.target.value)}
                      className="bg-[#0a0a16] border border-[#2a2a4a] text-white rounded-lg pl-5 pr-12 py-3.5 text-sm font-semibold appearance-none focus:outline-none focus:border-purple-500 transition-colors w-48 md:w-64"
                    >
                      {tickers.map(t => (
                        <option key={t} value={t}>{t}</option>
                      ))}
                    </select>
                    <div className="absolute right-4 top-4 pointer-events-none text-gray-400 text-xs">▼</div>
                  </div>

                  <button
                    onClick={fetchPrediction}
                    className="bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-500 hover:brightness-110 text-white font-bold py-3.5 px-8 rounded-lg transition-all shadow-[0_0_20px_rgba(168,85,247,0.4)] hover:shadow-[0_0_25px_rgba(168,85,247,0.6)] hover:-translate-y-0.5 active:scale-95 text-sm flex items-center gap-2"
                  >
                    <Search className="w-5 h-5" />
                    Execute Forecast
                  </button>
                </div>
              </div>
            </>
          )}
        </main>
      </section>

      <div id="about-section">
        <About />
      </div>

      <div id="feedback-section">
        <Feedback />
      </div>

      <Footer />
    </div>
  )
}

export default App
