import React from 'react';
import { ResponsiveContainer, ComposedChart, XAxis, YAxis, Tooltip, CartesianGrid, Bar, Line } from 'recharts';

export function PriceChart({ data, predictions, lastClose }) {
  if (!data || data.length === 0) return null;

  const chartData = data.map(d => ({
    ...d,
    dateStr: new Date(d.Date).toLocaleDateString()
  }));

  const avgPred = predictions
    ? Object.values(predictions).reduce((a, b) => a + b, 0) / Object.values(predictions).length
    : null;

  return (
    <div className="mt-8 bg-[#161b22] border border-[#30363d] rounded-xl p-4">
      <h3 className="text-lg font-bold mb-4 text-gray-300">Market Data & Prediction</h3>
      <div className="h-[400px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData}>
            <defs>
              <linearGradient id="colorClose" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00ffbd" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#00ffbd" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#30363d" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="dateStr"
              stroke="#8b949e"
              tick={{ fill: '#8b949e', fontSize: 12 }}
              tickLine={false}
              minTickGap={30}
            />
            <YAxis
              domain={['auto', 'auto']}
              stroke="#8b949e"
              tick={{ fill: '#8b949e', fontSize: 12 }}
              tickLine={false}
              tickFormatter={(val) => `₹${val}`}
            />
            <Tooltip
              contentStyle={{ backgroundColor: '#0d1117', borderColor: '#30363d', color: '#fff' }}
              itemStyle={{ color: '#00ffbd' }}
              labelStyle={{ color: '#8b949e' }}
            />
            <Line
              type="monotone"
              dataKey="Close"
              stroke="#00ffbd"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 6, fill: "#fff" }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      {avgPred && (
        <div className="mt-2 text-center text-sm text-[#00ffbd]">
          Target Average: ₹{avgPred.toFixed(2)} (dashed line in spirit)
        </div>
      )}
    </div>
  );
}
