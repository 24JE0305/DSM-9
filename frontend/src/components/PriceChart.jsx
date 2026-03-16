import React, { useState, useMemo } from 'react';
import { ResponsiveContainer, ComposedChart, XAxis, YAxis, Tooltip, CartesianGrid, Area, Line, ReferenceLine, ReferenceDot, Label } from 'recharts';

// Custom Label for the current price, min, max, exp on the right axis
const CustomRightLabel = ({ viewBox, value, bg, color, text, yOffset = 0 }) => {
  const { x, y } = viewBox;
  return (
    <g>
      <rect x={x} y={y - 12 + yOffset} width={65} height={24} fill={bg} rx={4} />
      <text x={x + 32} y={y + 4 + yOffset} fill={color} fontSize={10} textAnchor="middle" fontWeight="bold">
        {text}
      </text>
    </g>
  );
};

export function PriceChart({ data, predictions, lastClose }) {
  const [horizon, setHorizon] = useState('Overall');

  const chartData = useMemo(() => {
    if (!data || data.length === 0) return [];

    let filteredData = data;
    if (horizon === '90D') {
      filteredData = data.slice(Math.max(data.length - 90, 0));
    } else if (horizon === '365D') {
      filteredData = data.slice(Math.max(data.length - 365, 0));
    }

    const baseData = filteredData.map((d, i) => {
      const isLast = i === filteredData.length - 1; // Check against filteredData length
      const dateObj = new Date(d.Date);
      return {
        ...d,
        timestamp: dateObj.getTime(),
        dateStr: dateObj.toLocaleDateString(),
        // Range connects the last point linearly to the future bounds
        range: isLast ? [lastClose, lastClose] : undefined,
        Avg: isLast ? lastClose : undefined,
      };
    });

    if (!predictions) return baseData;

    // Use 365D target values for the "Overall" plotting cone
    const activeHorizon = horizon === 'Overall' ? '365D' : horizon;

    if (!predictions[activeHorizon]) return baseData;

    const avgPred = predictions[activeHorizon];
    const maxPred = avgPred * 1.15; // 15% upside spread roughly
    const minPred = avgPred * 0.85; // 15% downside spread roughly

    const lastPoint = baseData[baseData.length - 1];
    const targetDate = new Date(lastPoint.timestamp);

    // Parse horizon span
    let days = parseInt(activeHorizon.replace('D', ''));
    if (isNaN(days)) days = 365; // fallback
    targetDate.setDate(targetDate.getDate() + days);

    const futureEnd = {
      timestamp: targetDate.getTime(),
      dateStr: targetDate.toLocaleDateString(),
      range: [minPred, maxPred],
      Avg: avgPred,
    };

    return [...baseData, futureEnd];
  }, [data, predictions, horizon, lastClose]);

  if (!data || data.length === 0) return null;

  const currentPred = predictions ? (horizon === 'Overall' ? predictions['365D'] : predictions[horizon]) : null;

  return (
    <div className="mt-8 bg-[#111326]/60 backdrop-blur-md border border-[#2a2a4a] rounded-xl p-4 max-w-6xl mx-auto shadow-2xl">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-cyan-400 ml-4">Live Market & Predictive Cone</h3>
        {predictions && (
          <div className="flex space-x-2 bg-[#0a0a16] p-1.5 rounded-lg border border-[#2a2a4a] shadow-inner">
            {['Overall', '90D', '365D'].map(h => (
              (h === 'Overall' || predictions[h]) && (
                <button
                  key={h}
                  onClick={() => setHorizon(h)}
                  className={`px-4 py-1.5 text-xs tracking-wider uppercase font-bold rounded-md transition-all ${horizon === h ? 'bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-500/50 text-cyan-400 shadow-[0_0_10px_rgba(168,85,247,0.3)]' : 'text-gray-400 hover:text-white hover:bg-[#ffffff0a] border border-transparent'}`}
                >
                  {h}
                </button>
              )
            ))}
          </div>
        )}
      </div>

      <div className="h-[400px] w-full relative">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartData} margin={{ top: 20, right: 70, bottom: 20, left: 20 }}>
            <defs>
              <linearGradient id="colorCone" x1="0" y1="0" x2="1" y2="0">
                {currentPred && currentPred < lastClose ? (
                  // Bearish: Red/Pink Gradient
                  <>
                    <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.15} />
                    <stop offset="100%" stopColor="#be123c" stopOpacity={0.4} />
                  </>
                ) : (
                  // Bullish/Neutral: Default Purple/Pink Gradient
                  <>
                    <stop offset="0%" stopColor="#a855f7" stopOpacity={0.15} />
                    <stop offset="100%" stopColor="#ec4899" stopOpacity={0.4} />
                  </>
                )}
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#2a2a4a" strokeDasharray="3 3" vertical={false} />
            <XAxis
              type="number"
              domain={['dataMin', 'dataMax']}
              dataKey="timestamp"
              stroke="#6b7280"
              tick={{ fill: '#6b7280', fontSize: 11, fontWeight: 'bold' }}
              tickLine={false}
              tickFormatter={(val) => new Date(val).toLocaleDateString([], { month: 'short', year: '2-digit' })}
              minTickGap={30}
            />
            <YAxis
              domain={['auto', 'auto']}
              stroke="#6b7280"
              tick={{ fill: '#6b7280', fontSize: 11, fontWeight: 'bold' }}
              tickLine={false}
              tickFormatter={(val) => `₹${val}`}
            />
            <Tooltip
              labelFormatter={(val) => new Date(val).toLocaleDateString()}
              formatter={(value, name, props) => {
                // Ignore formatting for range data
                if (name === "range") return [null, null];
                return [`₹${Number(value).toFixed(2)}`, name === 'Avg' ? 'Exp. Price' : name];
              }}
              contentStyle={{ backgroundColor: '#0a0a16', borderColor: '#2a2a4a', color: '#fff', borderRadius: '12px', boxShadow: '0 0 20px rgba(168,85,247,0.2)' }}
              itemStyle={{ color: '#22d3ee', fontWeight: 'bold' }}
              labelStyle={{ color: '#9ca3af', marginBottom: '4px' }}
            />

            {/* Cone Area */}
            <Area
              type="linear"
              dataKey="range"
              fill="url(#colorCone)"
              stroke="none"
              isAnimationActive={true}
            />

            {/* Average Target Line inside cone */}
            <Line
              type="linear"
              dataKey="Avg"
              stroke="#22d3ee"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              isAnimationActive={true}
            />

            {/* Historical Close */}
            <Line
              type="monotone"
              dataKey="Close"
              stroke="#a855f7"
              strokeWidth={3}
              dot={false}
              activeDot={{ r: 6, fill: "#fff", stroke: "#ec4899", strokeWidth: 2 }}
            />

            {/* Target Bound Lines & Rectangular Badges */}
            {currentPred && (() => {
              const maxPred = currentPred * 1.15;
              const minPred = currentPred * 0.85;

              // Collision Detection: If 'Cur' is too close to Max, Exp, or Min (within 5%), shift its label vertically.
              let curYOffset = 0;
              const pushAmount = 26; // pixels to shift label so it perfectly clears the 24px box

              if (Math.abs(lastClose - currentPred) / lastClose < 0.05) {
                curYOffset = lastClose >= currentPred ? -pushAmount : pushAmount;
              } else if (Math.abs(lastClose - maxPred) / lastClose < 0.05) {
                curYOffset = lastClose >= maxPred ? -pushAmount : pushAmount;
              } else if (Math.abs(lastClose - minPred) / lastClose < 0.05) {
                curYOffset = lastClose >= minPred ? -pushAmount : pushAmount;
              }

              return (
                <>
                  <ReferenceLine y={maxPred} stroke="#0ea5e9" strokeDasharray="3 3" strokeOpacity={0.4}>
                    <Label position="right" content={<CustomRightLabel bg="#0ea5e9" color="#fff" text={`Max ₹${maxPred.toFixed(2)}`} />} />
                  </ReferenceLine>

                  {/* Expected Price */}
                  <ReferenceLine y={currentPred} stroke="#ec4899" strokeDasharray="3 3" strokeOpacity={0.5}>
                    <Label position="right" content={<CustomRightLabel bg="#ec4899" color="#fff" text={`Exp ₹${currentPred.toFixed(2)}`} />} />
                  </ReferenceLine>

                  <ReferenceLine y={minPred} stroke="#f43f5e" strokeDasharray="3 3" strokeOpacity={0.4}>
                    <Label position="right" content={<CustomRightLabel bg="#f43f5e" color="#fff" text={`Min ₹${minPred.toFixed(2)}`} />} />
                  </ReferenceLine>

                  <ReferenceLine y={lastClose} stroke="#6b7280" strokeDasharray="3 3" strokeOpacity={0.5}>
                    <Label position="right" content={<CustomRightLabel bg="#374151" color="#fff" text={`Cur ₹${lastClose.toFixed(2)}`} yOffset={curYOffset} />} />
                  </ReferenceLine>

                  {/* Dot for Current Price at the last historical timestamp */}
                  {(() => {
                    // Find the timestamp for the current price (last historical data point)
                    const lastPoint = data[data.length - 1];
                    if (!lastPoint) return null;
                    const dateObj = new Date(lastPoint.Date);
                    return (
                      <ReferenceDot
                        x={dateObj.getTime()}
                        y={lastClose}
                        r={6}
                        fill="#fff"
                        stroke="#ec4899"
                        strokeWidth={2}
                        isFront={true}
                      />
                    );
                  })()}
                </>
              );
            })()}

          </ComposedChart>
        </ResponsiveContainer>

        {/* CENTER FORECAST LABEL */}
        {predictions && horizon !== 'Overall' && (
          <div className="absolute bottom-8 inset-x-0 flex justify-center pointer-events-none">
            <span className="bg-[#0a0a16]/80 backdrop-blur-sm text-cyan-400 text-xs font-bold px-5 py-2 rounded-full border border-cyan-500/30 shadow-[0_0_15px_rgba(34,211,238,0.2)] uppercase tracking-widest">
              {horizon === '365D' ? '1Y' : horizon} FORECAST
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
