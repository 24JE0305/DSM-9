import React from 'react';
import { cn } from '../lib/utils';

export function MetricCard({ title, value, subtext, trend, className }) {
  return (
    <div className={cn("bg-[#161b22] border border-[#30363d] rounded-xl p-5 flex flex-col items-center justify-center text-center", className)}>
      <h3 className="text-gray-400 text-sm uppercase tracking-wider mb-1">{title}</h3>
      <div className="text-2xl font-bold text-white mb-1">{value}</div>
      {(subtext || trend) && (
        <div className={cn("text-sm",
          trend === 'up' ? "text-green-400" :
            trend === 'down' ? "text-red-400" : "text-gray-400"
        )}>
          {subtext}
        </div>
      )}
    </div>
  );
}
