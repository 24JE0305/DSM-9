import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Play, Settings2, BarChart2, ShieldAlert, History as HistoryIcon, Calendar, Activity } from 'lucide-react';

import { Navbar } from '../components/Navbar';
import { Footer } from '../components/Footer';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';

const API_URL = "http://localhost:8000";

function Backtest() {
  const [isRunning, setIsRunning] = useState(false);
  const [ticker, setTicker] = useState('RELIANCE.NS');
  const [horizon, setHorizon] = useState(90);
  const [step, setStep] = useState(30);

  const [tickers, setTickers] = useState(["RELIANCE.NS"]);

  useEffect(() => {
    async function fetchTickers() {
      try {
        const res = await axios.get(`${API_URL}/tickers`);
        if (res.data.tickers) {
          setTickers(res.data.tickers);
        } else if (Array.isArray(res.data)) {
          setTickers(res.data);
        }
      } catch (e) {
        console.error("Failed to fetch tickers", e);
      }
    }
    fetchTickers();
  }, []);

  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const handleRun = async () => {
    setIsRunning(true);
    setError(null);
    try {
      const response = await axios.get(`${API_URL}/backtest_v3/${ticker}`, {
        params: { horizon, step }
      });
      setResults(response.data);
    } catch (err) {
      console.error(err);
      setError("Failed to run backtest. Ensure backend is running and model data exists for this ticker.");
    } finally {
      setIsRunning(false);
    }
  };

  // Prepare chart data
  const chartData = results?.equity_curve ? results.equity_curve.map((val, idx) => ({
    step: idx,
    equity: val
  })) : [];

  return (
    <div className="flex flex-col bg-[#05060a] min-h-screen text-white font-sans relative">
      <div className="fixed top-0 left-0 w-full h-screen pointer-events-none overflow-hidden z-0">
        <div className="absolute top-[30%] right-1/3 w-[500px] h-[500px] bg-pink-600 opacity-10 rounded-full blur-[120px] mix-blend-screen"></div>
        <div className="absolute bottom-[20%] left-1/4 w-[600px] h-[600px] bg-cyan-600 opacity-10 rounded-full blur-[120px] mix-blend-screen"></div>
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4IiBoZWlnaHQ9IjgiPgo8cmVjdCB3aWR0aD0iOCIgaGVpZ2h0PSI4IiBmaWxsPSIjZmZmIiBmaWxsLW9wYWNpdHk9IjAuMDMiLz4KPHBhdGggZD0iTTAgMEw4IDhaTTAgOEw4IDBaIiBzdHJva2U9IiNmZmYiIHN0cm9rZS1vcGFjaXR5PSIwLjA1IiBzdHJva2Utd2lkdGg9IjEiLz4KPC9zdmc+')] opacity-20"></div>
      </div>

      <div className="relative z-50">
        <Navbar />
      </div>

      <main className="flex-1 w-full max-w-[1600px] mx-auto px-8 pt-12 pb-24 relative z-10">

        <div className="mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-cyan-400 text-xs font-semibold tracking-wider mb-4">
            <HistoryIcon className="w-3.5 h-3.5" />
            HISTORICAL VALIDATION
          </div>
          <h1 className="text-4xl md:text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 uppercase tracking-wider drop-shadow-[0_0_15px_rgba(56,189,248,0.3)] pb-2">
            Strategy Backtest
          </h1>
          <p className="text-gray-400 text-lg mt-2 max-w-2xl">
            Rigorously test your trading logic across years of tick data. Mitigate risk by validating strategies before deploying capital.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">

          {/* Controls Panel */}
          <div className="lg:col-span-1 space-y-6">
            <div className="bg-[#111326]/60 backdrop-blur-xl border border-[#2a2a4a] rounded-3xl p-6 shadow-xl relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/5 rounded-full blur-2xl -translate-y-1/2 translate-x-1/2"></div>

              <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
                <Settings2 className="w-5 h-5 text-cyan-400" />
                Parameters
              </h3>

              {error && (
                <div className="mb-4 p-3 bg-red-500/10 border border-red-500/40 text-red-400 text-sm rounded-xl">
                  {error}
                </div>
              )}

              <div className="space-y-5">
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">Ticker Symbol</label>
                  <div className="relative">
                    <select
                      value={ticker}
                      onChange={(e) => setTicker(e.target.value)}
                      className="w-full bg-[#0a0a16] border border-[#2a2a4a] rounded-xl p-3 text-sm text-gray-200 focus:outline-none focus:border-cyan-500 appearance-none uppercase cursor-pointer"
                    >
                      {tickers.map(t => (
                        <option key={t} value={t}>{t}</option>
                      ))}
                    </select>
                    <div className="absolute right-4 top-[50%] -translate-y-[50%] pointer-events-none text-gray-400 text-xs">▼</div>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-2">Horizon (Days)</label>
                    <select
                      value={horizon}
                      onChange={(e) => setHorizon(Number(e.target.value))}
                      className="w-full bg-[#0a0a16] border border-[#2a2a4a] rounded-xl p-3 text-sm text-gray-200 focus:outline-none focus:border-cyan-500 appearance-none"
                    >
                      <option value={90}>90 Days</option>
                      <option value={365}>365 Days</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-400 mb-2">Step (Days)</label>
                    <input
                      type="number"
                      value={step}
                      onChange={(e) => setStep(Number(e.target.value))}
                      className="w-full bg-[#0a0a16] border border-[#2a2a4a] rounded-xl p-3 text-sm text-gray-200 focus:outline-none focus:border-cyan-500"
                    />
                  </div>
                </div>

                <div className="pt-4">
                  <button
                    onClick={handleRun}
                    disabled={isRunning}
                    className="w-full bg-gradient-to-r from-cyan-600 to-blue-600 hover:brightness-110 text-white font-bold py-3.5 rounded-xl transition-all shadow-[0_0_20px_rgba(56,189,248,0.3)] hover:shadow-[0_0_25px_rgba(56,189,248,0.5)] flex items-center justify-center gap-2 disabled:opacity-70"
                  >
                    {isRunning ? (
                      <span className="animate-pulse flex items-center gap-2">
                        <Activity className="w-5 h-5 animate-spin" />
                        Simulating Engine...
                      </span>
                    ) : (
                      <>
                        <Play className="w-5 h-5 fill-current" />
                        Run Validation
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Results Panel */}
          <div className="lg:col-span-2 space-y-6">

            {/* Quick Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-[#111326]/40 backdrop-blur-md border border-[#2a2a4a] rounded-2xl p-5 hover:border-cyan-500/30 transition-colors">
                <p className="text-gray-400 text-sm mb-1 text-center">Net Profit</p>
                <p className={`text-2xl font-bold text-center ${results?.metrics?.total_return_pct > 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {results ? `${results.metrics.total_return_pct > 0 ? '+' : ''}${results.metrics.total_return_pct}%` : '---'}
                </p>
              </div>
              <div className="bg-[#111326]/40 backdrop-blur-md border border-[#2a2a4a] rounded-2xl p-5 hover:border-cyan-500/30 transition-colors">
                <p className="text-gray-400 text-sm mb-1 text-center">Max Drawdown</p>
                <p className="text-2xl font-bold text-red-400 text-center">
                  {results ? `${results.metrics.max_drawdown_pct}%` : '---'}
                </p>
              </div>
              <div className="bg-[#111326]/40 backdrop-blur-md border border-[#2a2a4a] rounded-2xl p-5 hover:border-cyan-500/30 transition-colors">
                <p className="text-gray-400 text-sm mb-1 text-center">Win Rate</p>
                <p className="text-2xl font-bold text-white text-center">
                  {results ? `${results.metrics.win_rate_pct}%` : '---'}
                </p>
              </div>
              <div className="bg-[#111326]/40 backdrop-blur-md border border-[#2a2a4a] rounded-2xl p-5 hover:border-cyan-500/30 transition-colors">
                <p className="text-gray-400 text-sm mb-1 text-center">Sharpe Ratio</p>
                <p className="text-2xl font-bold text-white text-center">
                  {results ? results.metrics.sharpe_ratio : '---'}
                </p>
              </div>
            </div>

            {/* Chart Area */}
            <div className="bg-[#111326]/40 backdrop-blur-md border border-[#2a2a4a] rounded-3xl p-6 h-[400px] flex flex-col shadow-xl">
              <h3 className="text-lg font-bold mb-4 flex items-center gap-2 text-gray-200">
                <BarChart2 className="w-5 h-5 text-purple-400" />
                Equity Curve vs Buy & Hold
              </h3>

              <div className="flex-1 w-full h-full min-h-0 bg-[#0a0a16]/50 rounded-xl border border-[#2a2a4a] p-4 relative">
                {results ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" vertical={false} />
                      <XAxis
                        dataKey="step"
                        stroke="#6b7280"
                        tick={{ fill: '#6b7280', fontSize: 12 }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <YAxis
                        domain={['auto', 'auto']}
                        stroke="#6b7280"
                        tick={{ fill: '#6b7280', fontSize: 12 }}
                        tickFormatter={(val) => val.toFixed(2)}
                        tickLine={false}
                        axisLine={false}
                        width={40}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: '#111326', borderColor: '#2a2a4a', color: '#fff', borderRadius: '8px' }}
                        itemStyle={{ color: '#22d3ee' }}
                        formatter={(val) => [val.toFixed(4), "Equity Multiple"]}
                        labelFormatter={(label) => `Trade Step: ${label}`}
                      />
                      <Line
                        type="monotone"
                        dataKey="equity"
                        stroke="#22d3ee"
                        strokeWidth={3}
                        dot={false}
                        activeDot={{ r: 6, fill: "#c084fc", stroke: "#c084fc" }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="flex items-center justify-center h-full text-center">
                    <div>
                      <Activity className="w-10 h-10 text-gray-600 mx-auto mb-3" />
                      <p className="text-gray-500 text-sm">Equity curve will render here after simulation.</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Warning Area */}
            <div className="bg-orange-500/10 border border-orange-500/30 rounded-2xl p-5 flex items-start gap-4">
              <ShieldAlert className="w-6 h-6 text-orange-400 shrink-0 mt-0.5" />
              <div>
                <h4 className="text-orange-400 font-semibold mb-1">Live Engine Disclaimer</h4>
                <p className="text-sm text-gray-400 leading-relaxed">
                  Backtesting uses walk-forward optimization across {horizon}-day loops using historical Model 3 features.
                  Slippage & commissions are not accounted for. {results && `Buy & Hold total return for the same period was ${results.benchmark.total_return_pct}%`}
                </p>
              </div>
            </div>

          </div>

        </div>

        {/* Strategy Explanation Section */}
        <div className="mt-12 bg-[#111326]/40 backdrop-blur-md border border-[#2a2a4a] rounded-3xl p-8 shadow-xl">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-2xl font-bold mb-4 flex items-center gap-3 text-cyan-400">
              <Settings2 className="w-6 h-6" />
              How Our AI Trades: The "Triple-Exit" State Machine Strategy
            </h2>
            <p className="text-gray-300 mb-6 leading-relaxed">
              Most investors ride the market roller coaster, blindly holding stocks through massive crashes. Our AI takes a different approach.
            </p>
            <p className="text-gray-300 mb-8 leading-relaxed">
              Powered by our advanced <strong>Model 3.1 Architecture (Transformer + BiLSTM-Attention + XGBoost)</strong>, our trading engine doesn't just guess where the market is going. It operates as a highly disciplined State Machine, meaning it only risks your capital when strict mathematical conditions are met, and it actively manages open trades to protect your downside.
            </p>

            <h3 className="text-xl font-semibold mb-6 text-white border-b border-[#2a2a4a] pb-2">Here is exactly how our AI makes its decisions:</h3>

            <div className="space-y-8">
              {/* Point 1 */}
              <div className="flex gap-4">
                <div className="shrink-0 mt-1">
                  <div className="w-8 h-8 rounded-full bg-cyan-500/20 flex items-center justify-center text-cyan-400 font-bold border border-cyan-500/30 shadow-[0_0_10px_rgba(34,211,238,0.2)]">1</div>
                </div>
                <div>
                  <h4 className="text-lg font-bold text-gray-200 mb-2">The Engine: 90-Day Predictive Vision</h4>
                  <p className="text-gray-400 leading-relaxed">
                    Our AI ignores the daily noise and volatility of the market. Instead, it continuously analyzes historical data to generate a high-probability forecast of how a stock will perform over the next 90 days. Every decision is based on this rolling 3-month window.
                  </p>
                </div>
              </div>

              {/* Point 2 */}
              <div className="flex gap-4">
                <div className="shrink-0 mt-1">
                  <div className="w-8 h-8 rounded-full bg-purple-500/20 flex items-center justify-center text-purple-400 font-bold border border-purple-500/30 shadow-[0_0_10px_rgba(168,85,247,0.2)]">2</div>
                </div>
                <div>
                  <h4 className="text-lg font-bold text-gray-200 mb-2">The Entry Rule: High-Conviction Buying</h4>
                  <p className="text-gray-400 leading-relaxed">
                    We do not invest simply because a stock "might" go up. To risk capital and trigger a BUY signal, the AI must have absolute conviction. It strictly requires a predicted gain of at least +3.0% over the next 90 days. If the market looks flat, uncertain, or bearish, the system safely sits in CASH.
                  </p>
                </div>
              </div>

              {/* Point 3 */}
              <div className="flex gap-4">
                <div className="shrink-0 mt-1">
                  <div className="w-8 h-8 rounded-full bg-pink-500/20 flex items-center justify-center text-pink-400 font-bold border border-pink-500/30 shadow-[0_0_10px_rgba(236,72,153,0.2)]">3</div>
                </div>
                <div>
                  <h4 className="text-lg font-bold text-gray-200 mb-3">Risk Management: The "Triple-Exit" System</h4>
                  <p className="text-gray-400 leading-relaxed mb-4">
                    The moment the AI enters a trade, its primary focus shifts to protecting your money. It constantly monitors the open position against three strict exit rules. If any of these three triggers are hit, the AI automatically executes a SELL:
                  </p>
                  
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-2">
                    <div className="bg-[#0a0a16]/80 border border-green-500/30 rounded-xl p-5 hover:border-green-500/60 transition-colors">
                      <h5 className="font-bold text-green-400 text-sm mb-2 flex items-center gap-2">Take-Profit</h5>
                      <p className="text-xs text-gray-400 leading-relaxed"><strong className="text-gray-300">Locking in Gains:</strong> If a stock surges and hits a +30% profit from our entry price, the AI automatically sells. We lock in the gains rather than getting greedy and risking a sudden reversal.</p>
                    </div>
                    <div className="bg-[#0a0a16]/80 border border-red-500/30 rounded-xl p-5 hover:border-red-500/60 transition-colors">
                      <h5 className="font-bold text-red-400 text-sm mb-2 flex items-center gap-2">Stop-Loss</h5>
                      <p className="text-xs text-gray-400 leading-relaxed"><strong className="text-gray-300">Cutting Losses Early:</strong> If a trade goes against us and the stock drops -8%, the AI executes a hard sell. This strict cut-off prevents a bad trade from turning into a catastrophic portfolio loss.</p>
                    </div>
                    <div className="bg-[#0a0a16]/80 border border-orange-500/30 rounded-xl p-5 hover:border-orange-500/60 transition-colors">
                      <h5 className="font-bold text-orange-400 text-sm mb-2 flex items-center gap-2">The Bearish Flip</h5>
                      <p className="text-xs text-gray-400 leading-relaxed"><strong className="text-gray-300">Capital Protection:</strong> If we hold a stock for our target 90-day horizon and the AI's continuously updated forecast suddenly turns negative, it sells immediately to protect your current position.</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Point 4 */}
              <div className="flex gap-4">
                <div className="shrink-0 mt-1">
                  <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center text-blue-400 font-bold border border-blue-500/30 shadow-[0_0_10px_rgba(59,130,246,0.2)]">4</div>
                </div>
                <div>
                  <h4 className="text-lg font-bold text-gray-200 mb-2">The Cool-Off Period</h4>
                  <p className="text-gray-400 leading-relaxed">
                    Our AI does not revenge-trade. If it executes a sell order, it enforces a strict "cool-off" period. It pulls your money back to safety, waits for the market to develop, and demands a brand new, high-conviction setup before re-entering the market.
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-10 p-5 rounded-2xl bg-gradient-to-r from-purple-500/10 to-cyan-500/10 border border-purple-500/30 shadow-inner">
              <p className="text-gray-300 font-medium leading-relaxed">
                <strong className="text-white">The Bottom Line:</strong> Our strategy is designed to capture high-probability upswings while strictly capping your risk on the downside. We buy with conviction, protect with discipline, and sit in cash when the market is uncertain.
              </p>
            </div>
          </div>
        </div>

      </main>

      <Footer />
    </div>
  );
}

export default Backtest;
