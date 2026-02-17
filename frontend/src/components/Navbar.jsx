import React, { useState, useEffect } from 'react';
import { Zap, Home, Info, Search } from 'lucide-react';
import axios from 'axios';

const API_URL = "http://localhost:8000";

export function Navbar({ selectedTicker, onSelectTicker, onPredict }) {
  const [tickers, setTickers] = useState([]);

  useEffect(() => {
    async function fetchTickers() {
      try {
        const res = await axios.get(`${API_URL}/tickers`);
        if (res.data.tickers) {
          setTickers(res.data.tickers);
        } else if (Array.isArray(res.data)) {
          setTickers(res.data);
        } else {
          console.warn("Unexpected ticker format", res.data);
          setTickers(["ITC.NS", "RELIANCE.NS"]);
        }
      } catch (e) {
        console.error("Failed to fetch tickers", e);
        setTickers(["ITC.NS", "RELIANCE.NS"]);
      }
    }
    fetchTickers();
  }, []);

  return (
    <nav className="bg-[#161b22] border-b border-[#30363d] px-6 py-4 flex items-center justify-between sticky top-0 z-50">
      {/* Left: Logo and Nav Links */}
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-2">
          <Zap className="w-6 h-6 text-[#00ffbd]" />
          <div>
            <h1 className="text-xl font-bold text-white leading-none">DSM-9</h1>
            <span className="text-xs text-gray-400">Pro Terminal</span>
          </div>
        </div>

        <div className="flex items-center gap-6">
          <button className="flex items-center gap-2 text-gray-300 hover:text-white transition-colors text-sm font-medium">
            <Home className="w-4 h-4" />
            Home
          </button>
          <button className="flex items-center gap-2 text-gray-300 hover:text-white transition-colors text-sm font-medium">
            <Info className="w-4 h-4" />
            About Us
          </button>
        </div>
      </div>

      {/* Right: Ticker Control */}
      <div className="flex items-center gap-4">
        <div className="relative">
          <select
            value={selectedTicker}
            onChange={(e) => onSelectTicker(e.target.value)}
            className="bg-[#0d1117] border border-[#30363d] text-white rounded-lg pl-4 pr-10 py-2 text-sm appearance-none focus:outline-none focus:border-[#00ffbd] transition-colors w-48"
          >
            {tickers.map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
          <div className="absolute right-3 top-2.5 pointer-events-none text-gray-500 text-xs">
            ▼
          </div>
        </div>

        <button
          onClick={onPredict}
          className="bg-[#00ffbd] hover:bg-[#00e6aa] text-black font-bold py-2 px-6 rounded-lg transition-all shadow-[0_0_10px_rgba(0,255,189,0.2)] active:scale-95 text-sm flex items-center gap-2"
        >
          <Search className="w-4 h-4" />
          Analyze
        </button>
      </div>
    </nav>
  );
}
