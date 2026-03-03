import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

const data = [
  { name: 'Highly Bearish', value: 1, color: '#f43f5e' },
  { name: 'Bearish', value: 1, color: '#d29922' },
  { name: 'Neutral', value: 1, color: '#6b7280' },
  { name: 'Bullish', value: 1, color: '#3fb950' },
  { name: 'Highly Bullish', value: 1, color: '#0ea5e9' },
];

export function AnalystGauge({ score = 50, analystsCount = 33 }) {
  // Map score (0-100) to rotation angle (-90 to 90 degrees) for the needle
  // Score 0 = Strong Sell (-90deg), Score 100 = Strong Buy (+90deg), Score 50 = Neutral (0deg)
  const rotation = (score / 100) * 180 - 90;

  let currentRating = "Neutral";
  if (score < 20) currentRating = "Highly Bearish";
  else if (score < 40) currentRating = "Bearish";
  else if (score < 60) currentRating = "Neutral";
  else if (score < 80) currentRating = "Bullish";
  else currentRating = "Highly Bullish";

  return (
    <div className="bg-[#111326]/60 backdrop-blur-md border border-[#2a2a4a] shadow-lg rounded-xl p-5 flex flex-col items-center relative overflow-hidden h-full">
      <h3 className="text-gray-300 font-bold mb-1 w-full text-left">Signal Bias</h3>
      <p className="text-[10px] text-gray-500 mb-6 w-full text-left leading-snug">
        Aggregated directional probability derived from structural prediction targets. Not financial advice.
      </p>

      {/* Gauge Container */}
      <div className="relative w-full max-w-[220px] h-[120px]">
        {/* Recharts Pie for the semi-circle */}
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="100%"
              startAngle={180}
              endAngle={0}
              innerRadius={65}
              outerRadius={80}
              paddingAngle={2}
              dataKey="value"
              stroke="none"
              isAnimationActive={true}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>

        <div className="absolute inset-0 pointer-events-none">
          <span className="absolute bottom-[10px] left-[-20px] text-[10px] text-gray-400 font-medium text-center">Highly<br />Bearish</span>
          <span className="absolute top-[35px] left-[15px] text-[10px] text-gray-400 font-medium">Bearish</span>
          <span className="absolute top-[-15px] left-1/2 -translate-x-1/2 text-[10px] text-gray-400 font-bold">Neutral</span>
          <span className="absolute top-[35px] right-[15px] text-[10px] text-gray-400 font-medium">Bullish</span>
          <span className="absolute bottom-[10px] right-[-20px] text-[10px] text-gray-400 font-medium text-center">Highly<br />Bullish</span>
        </div>

        {/* The Needle Container */}
        <div
          className="absolute bottom-0 left-1/2 flex items-end justify-center pointer-events-none transition-transform duration-1000 ease-in-out origin-bottom"
          style={{
            transform: `translateX(-50%) rotate(${rotation}deg)`,
            height: '80px',
            width: '4px'
          }}
        >
          {/* Needle stick */}
          <div className="w-1 h-full bg-white rounded-t-full shadow-lg"></div>
          {/* Needle Base Dot */}
          <div className="absolute bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 w-4 h-4 bg-white rounded-full border-4 border-[#0a0a16]"></div>
        </div>
      </div>

      <div className="mt-4 text-center">
        <div className="text-xl font-bold text-white">{currentRating}</div>
        <div className="text-sm text-cyan-400 mt-1 font-semibold">Score: {score.toFixed(1)}/100</div>
      </div>
    </div>
  );
}
