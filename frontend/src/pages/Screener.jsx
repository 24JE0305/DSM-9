import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Search, Filter, ArrowUpRight, TrendingUp, TrendingDown, Activity, SlidersHorizontal, ChevronDown } from 'lucide-react';
import { Navbar } from '../components/Navbar';
import { Footer } from '../components/Footer';


const API_URL = "https://s8world7-backend-back.hf.space";

function Screener() {
  const [activePreset, setActivePreset] = useState('momentum');
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);

  // Filters State
  const [horizon, setHorizon] = useState(90);
  const [signal, setSignal] = useState('');
  const [confidence, setConfidence] = useState('');

  // Sorting State
  const [sortConfig, setSortConfig] = useState({ key: 'expected_return', direction: 'desc' });

  const presets = [
    { id: 'momentum', name: 'High Momentum', params: { signal: 'Bullish,Strong Bullish', confidence: 'High' } },
    { id: 'safe', name: 'Low Risk', params: { signal: 'Bullish,Neutral', confidence: 'High' } },
    { id: 'all', name: 'All Assets', params: { signal: '', confidence: '' } },
  ];

  const fetchScreener = async (paramsOverride = null) => {
    setIsLoading(true);
    setError(null);
    try {
      const queryParams = paramsOverride || { horizon, signal, confidence };
      // Clean up empty params
      const cleanParams = Object.fromEntries(Object.entries(queryParams).filter(([_, v]) => v !== ''));

      const response = await axios.get(`${API_URL}/screener`, { params: cleanParams });
      setResults(response.data.results || []);
      setSummary(response.data.summary || null);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch screener data. Ensure backend is running and model data exists.");
    } finally {
      setIsLoading(false);
    }
  };

  const applyPreset = (preset) => {
    setActivePreset(preset.id);
    setSignal(preset.params.signal);
    setConfidence(preset.params.confidence);
    fetchScreener({ horizon, ...preset.params });
  };

  const handleSort = (key) => {
    let direction = 'desc';
    if (sortConfig.key === key && sortConfig.direction === 'desc') {
      direction = 'asc';
    }
    setSortConfig({ key, direction });
  };

  const sortedResults = React.useMemo(() => {
    let sortableItems = [...results];
    sortableItems.sort((a, b) => {
      let aValue = a[sortConfig.key];
      let bValue = b[sortConfig.key];
      if (aValue < bValue) {
        return sortConfig.direction === 'asc' ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortConfig.direction === 'asc' ? 1 : -1;
      }
      return 0;
    });
    return sortableItems;
  }, [results, sortConfig]);

  const SortIcon = ({ columnKey }) => {
    if (sortConfig.key !== columnKey) return <ChevronDown className="w-3 h-3 opacity-30 inline-block ml-1" />;
    return <ChevronDown className={`w-3 h-3 inline-block ml-1 transition-transform ${sortConfig.direction === 'asc' ? 'rotate-180' : ''}`} />;
  };

  // Initial load
  useEffect(() => {
    applyPreset(presets[0]);
    // eslint-disable-next-line
  }, []);

  return (
    <div className="flex flex-col bg-[#05060a] min-h-screen text-white font-sans relative">
      <div className="fixed top-0 left-0 w-full h-screen pointer-events-none overflow-hidden z-0">
        <div className="absolute top-[20%] left-1/3 w-[500px] h-[500px] bg-purple-600 opacity-10 rounded-full blur-[120px] mix-blend-screen"></div>
        <div className="absolute bottom-[10%] right-1/4 w-[600px] h-[600px] bg-cyan-600 opacity-10 rounded-full blur-[120px] mix-blend-screen"></div>
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4IiBoZWlnaHQ9IjgiPgo8cmVjdCB3aWR0aD0iOCIgaGVpZ2h0PSI4IiBmaWxsPSIjZmZmIiBmaWxsLW9wYWNpdHk9IjAuMDMiLz4KPHBhdGggZD0iTTAgMEw4IDhaTTAgOEw4IDBaIiBzdHJva2U9IiNmZmYiIHN0cm9rZS1vcGFjaXR5PSIwLjA1IiBzdHJva2Utd2lkdGg9IjEiLz4KPC9zdmc+')] opacity-20"></div>
      </div>

      <div className="relative z-50">
        <Navbar />
      </div>

      <main className="flex-1 w-full max-w-[1600px] mx-auto px-8 pt-12 pb-24 relative z-10">

        <div className="mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-500/10 border border-purple-500/20 text-purple-400 text-xs font-semibold tracking-wider mb-4">
            <Activity className="w-3.5 h-3.5" />
            AI-POWERED DISCOVERY
          </div>
          <h1 className="text-4xl md:text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-pink-500 via-purple-400 to-cyan-400 uppercase tracking-wider drop-shadow-[0_0_15px_rgba(236,72,153,0.3)] pb-2">
            Market Screener
          </h1>
          <p className="text-gray-400 text-lg mt-2 max-w-2xl">
            Execute complex screening logic instantly. Uncover high-potential setups across NSE assets using Model 3.0 aggregate inference.
          </p>
        </div>

        {error && (
          <div className="mb-8 p-4 bg-red-500/10 border border-red-500/40 text-red-400 text-sm rounded-xl">
            {error}
          </div>
        )}

        {/* Filter Controls */}
        <div className="flex flex-col xl:flex-row gap-6 mb-8">

          <div className="flex-1 bg-[#111326]/60 backdrop-blur-xl border border-[#2a2a4a] p-3 rounded-2xl flex flex-wrap items-center gap-4 shadow-lg">
            
            <div className="flex items-center gap-2 min-w-[150px]">
              <span className="text-sm text-gray-400">Horizon:</span>
              <select
                value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
                className="bg-[#0a0a16] border border-[#2a2a4a] text-sm text-gray-200 py-2 px-3 rounded-lg focus:outline-none focus:border-purple-500 flex-1"
              >
                <option value={90}>90 Days</option>
                <option value={365}>365 Days</option>
              </select>
            </div>

            <div className="flex items-center gap-2 min-w-[200px]">
              <span className="text-sm text-gray-400">Signal:</span>
              <select
                value={signal}
                onChange={(e) => setSignal(e.target.value)}
                className="bg-[#0a0a16] border border-[#2a2a4a] text-sm text-gray-200 py-2 px-3 rounded-lg focus:outline-none focus:border-purple-500 flex-1"
              >
                <option value="">Any</option>
                <option value="Strong Bullish,Bullish">Bullish</option>
                <option value="Strong Bearish,Bearish">Bearish</option>
                <option value="Neutral">Neutral</option>
              </select>
            </div>

            <div className="flex items-center gap-2 min-w-[200px]">
              <span className="text-sm text-gray-400">Confidence:</span>
              <select
                value={confidence}
                onChange={(e) => setConfidence(e.target.value)}
                className="bg-[#0a0a16] border border-[#2a2a4a] text-sm text-gray-200 py-2 px-3 rounded-lg focus:outline-none focus:border-purple-500 flex-1"
              >
                <option value="">Any</option>
                <option value="High">High Only</option>
                <option value="High,Moderate">High & Moderate</option>
              </select>
            </div>


            <button
              onClick={() => fetchScreener()}
              disabled={isLoading}
              className="ml-auto bg-gradient-to-r from-purple-600 to-pink-500 hover:brightness-110 text-white font-medium py-2.5 px-6 rounded-xl transition-all shadow-[0_0_15px_rgba(168,85,247,0.4)] disabled:opacity-50 flex items-center gap-2"
            >
              {isLoading ? <Activity className="w-4 h-4 animate-spin" /> : <Filter className="w-4 h-4" />}
              Apply Filters
            </button>
          </div>

        </div>

        {/* Presets Grid */}
        <div className="flex flex-wrap gap-3 mb-10">
          {presets.map(preset => (
            <button
              key={preset.id}
              onClick={() => applyPreset(preset)}
              className={`px-5 py-2.5 rounded-xl text-sm font-medium transition-all ${activePreset === preset.id
                  ? 'bg-purple-500/20 border border-purple-500 text-purple-300 shadow-[0_0_10px_rgba(168,85,247,0.2)]'
                  : 'bg-[#111326]/50 border border-[#2a2a4a] text-gray-400 hover:text-white hover:border-gray-500'
                }`}
            >
              {preset.name}
            </button>
          ))}
        </div>

        {/* Results Table */}
        <div className="bg-[#111326]/40 backdrop-blur-md border border-[#2a2a4a] rounded-3xl overflow-hidden shadow-2xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-[#1a1c36]/50 border-b border-[#2a2a4a]">
                  <th className="py-5 px-6 font-semibold text-gray-400 text-sm uppercase tracking-wider cursor-pointer hover:text-white transition-colors" onClick={() => handleSort('symbol')}>Asset <SortIcon columnKey="symbol" /></th>
                  <th className="py-5 px-6 font-semibold text-gray-400 text-sm uppercase tracking-wider cursor-pointer hover:text-white transition-colors" onClick={() => handleSort('signal_bias')}>Signal Bias <SortIcon columnKey="signal_bias" /></th>
                  <th className="py-5 px-6 font-semibold text-gray-400 text-sm uppercase tracking-wider cursor-pointer hover:text-white transition-colors" onClick={() => handleSort('expected_return')}>Expected Return <SortIcon columnKey="expected_return" /></th>
                  <th className="py-5 px-6 font-semibold text-gray-400 text-sm uppercase tracking-wider cursor-pointer hover:text-white transition-colors" onClick={() => handleSort('confidence_score')}>Confidence <SortIcon columnKey="confidence_score" /></th>
                  <th className="py-5 px-6 font-semibold text-gray-400 text-sm uppercase tracking-wider text-right cursor-pointer hover:text-white transition-colors" onClick={() => handleSort('model_agreement')}>Agreement <SortIcon columnKey="model_agreement" /></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#2a2a4a]/50">
                {sortedResults.length > 0 ? sortedResults.map((row, idx) => (
                  <tr key={idx} className="hover:bg-[#1a1c36]/30 transition-colors group">
                    <td className="py-4 px-6">
                      <div className="flex flex-col">
                        <span className="font-bold text-white tracking-wide">{row.symbol}</span>
                      </div>
                    </td>
                    <td className="py-4 px-6 font-medium">
                      <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold ${row.signal_bias.includes('Bullish') ? 'bg-green-500/10 text-green-400 border border-green-500/20' :
                          row.signal_bias.includes('Bearish') ? 'bg-red-500/10 text-red-400 border border-red-500/20' :
                            'bg-gray-500/10 text-gray-400 border border-gray-500/20'
                        }`}>
                        {row.signal_bias.includes('Bullish') ? <TrendingUp className="w-3 h-3" /> :
                          row.signal_bias.includes('Bearish') ? <TrendingDown className="w-3 h-3" /> : null}
                        {row.signal_bias}
                      </span>
                    </td>
                    <td className="py-4 px-6 font-bold text-white">
                      {row.expected_return > 0 ? '+' : ''}{row.expected_return}%
                    </td>
                    <td className="py-4 px-6">
                      <div className="flex items-center gap-3">
                        <div className="w-20 h-1.5 bg-[#0a0a16] rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${row.confidence_score * 100 > 80 ? 'bg-cyan-400' : row.confidence_score * 100 > 60 ? 'bg-purple-400' : 'bg-pink-400'}`}
                            style={{ width: `${row.confidence_score * 100}%` }}
                          ></div>
                        </div>
                        <span className="text-sm font-semibold text-gray-300">{(row.confidence_score * 100).toFixed(0)}%</span>
                      </div>
                    </td>

                    <td className="py-4 px-6 text-right font-medium text-cyan-400">
                      {(row.model_agreement * 100).toFixed(0)}%
                    </td>
                  </tr>
                )) : (
                  <tr>
                    <td colSpan={6} className="py-12 text-center text-gray-500">
                      {isLoading ? 'Scanning universe...' : 'No assets match the current criteria.'}
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <div className="p-4 border-t border-[#2a2a4a] text-center text-sm text-gray-500 bg-[#111326]/50">
            {summary && !isLoading ? `Found ${summary.bullish_count} Bullish, ${summary.bearish_count} Bearish matching criteria.` : 'Ready to scan.'}
          </div>
        </div>

      </main>

      <Footer />
    </div>
  );
}

export default Screener;
