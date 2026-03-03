import React from 'react';
import { cn } from '../lib/utils';

export function MetricCard({ title, value, subtext, trend, className }) {
  return (
    <div className={cn("bg-[#111326]/60 backdrop-blur-md border border-[#2a2a4a] rounded-xl p-5 flex flex-col items-center justify-center text-center shadow-lg transition-transform hover:scale-[1.02]", className)}>
      <h3 className="text-purple-300/80 font-semibold text-xs tracking-widest uppercase mb-1">{title}</h3>
      <div className="text-2xl font-black text-white mb-1 drop-shadow-md">{value}</div>
      {(subtext || trend) && (
        <div className={cn("text-xs font-bold pt-1",
          trend === 'up' ? "text-cyan-400 drop-shadow-[0_0_5px_rgba(34,211,238,0.5)]" :
            trend === 'down' ? "text-pink-500 drop-shadow-[0_0_5px_rgba(236,72,153,0.5)]" : "text-gray-400"
        )}>
          {subtext}
        </div>
      )}
    </div>
  );
}
