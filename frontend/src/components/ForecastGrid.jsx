import React from 'react';
import { MetricCard } from './MetricCard';

export function ForecastGrid({ predictions, lastClose }) {
  if (!predictions) return null;

  return (
    <div className="mt-8 max-w-6xl mx-auto">
      <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
        <span className="text-purple-400"></span> Forecast Horizon
      </h2>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {Object.entries(predictions).map(([period, price]) => {
          const diff = price - lastClose;
          const percent = (diff / lastClose) * 100;

          return (
            <MetricCard
              key={period}
              title={`⏳ ${period}`}
              value={`₹${price.toLocaleString()}`}
              subtext={`${diff > 0 ? '+' : ''}${diff.toFixed(2)} (${percent.toFixed(2)}%)`}
              trend={diff > 0 ? 'up' : diff < 0 ? 'down' : 'neutral'}
              className="bg-[#0a0a16]/60"
            />
          );
        })}
      </div>
    </div>
  );
}
