import React, { useEffect, useRef, useState } from 'react';
import { Database, Zap, Shield, TrendingUp } from 'lucide-react';
import { TypewriterText } from './TypewriterText';

function FadeInView({ children, direction = 'left', delay = 0 }) {
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setIsVisible(true);
        observer.unobserve(entry.target);
      }
    }, { threshold: 0.2 });

    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  const slideClass = direction === 'left' ? '-translate-x-16' : 'translate-x-16';

  return (
    <div
      ref={ref}
      className={`transition-all duration-1000 transform flex w-full max-w-4xl mx-auto ${isVisible ? 'opacity-100 translate-x-0' : `opacity-0 ${slideClass}`}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      <div className={`flex w-full items-center gap-8 text-left justify-start`}>
        {children}
      </div>
    </div>
  );
}


export function About() {
  const features = [
    {
      icon: <Database className="w-12 h-12 text-cyan-400 drop-shadow-[0_0_8px_rgba(34,211,238,0.5)] flex-shrink-0" />,
      title: 'Vast Data Pipelines',
      desc: 'Ingesting millions of financial data points daily to maintain an unparalleled understanding of market history'
    },
    {
      icon: <Zap className="w-12 h-12 text-purple-400 drop-shadow-[0_0_8px_rgba(168,85,247,0.5)] flex-shrink-0" />,
      title: 'Real-time Inference',
      desc: 'Executing complex deep learning models in milliseconds to give you leading-edge predictions'
    },
    {
      icon: <Shield className="w-12 h-12 text-pink-400 drop-shadow-[0_0_8px_rgba(236,72,153,0.5)] flex-shrink-0" />,
      title: 'Confidence Scoring',
      desc: 'Dynamically evaluated analyst sentiment built securely into every forecast'
    },
    {
      icon: <TrendingUp className="w-12 h-12 text-blue-400 drop-shadow-[0_0_8px_rgba(59,130,246,0.5)] flex-shrink-0" />,
      title: 'Min/Max Cones',
      desc: 'Visualizing actionable standard deviation boundaries directly on your asset timeline'
    }
  ];

  return (
    <section className="py-24 relative overflow-hidden">
      <div className="max-w-6xl mx-auto px-6 lg:px-8">
        <div className="mx-auto max-w-3xl text-center mb-24">
          <h2 className="text-base/7 font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-cyan-400 uppercase tracking-widest drop-shadow-sm">The Engine</h2>
          <p className="mt-2 text-4xl font-semibold tracking-tight text-white sm:text-5xl drop-shadow-[0_0_10px_rgba(255,255,255,0.2)]">
            What is DSM-9?
          </p>
          <p className="mt-6 text-xl/8 text-gray-400 min-h-[96px]">
            <TypewriterText text="DSM-9 is an advanced artificial intelligence matrix designed to untangle complex macroeconomic threads. By evaluating structural patterns in both historical and active market streams, it serves as an uncompromised copilot for your financial strategy." />
          </p>
        </div>

        <div className="flex flex-col gap-y-16 w-full">
          {features.map((feature, idx) => {
            const direction = idx % 2 === 0 ? 'left' : 'right';

            return (
              <FadeInView key={feature.title} direction={direction}>
                <div className="rounded-2xl bg-[#111326]/50 p-6 ring-1 ring-[#2a2a4a] shadow-[0_10px_30px_rgba(0,0,0,0.5)] flex-shrink-0 backdrop-blur-sm">
                  {feature.icon}
                </div>
                <div className="flex-1">
                  <h3 className="text-3xl font-bold tracking-tight text-white drop-shadow-md">
                    {feature.title}
                  </h3>
                  <p className="mt-4 text-xl/8 text-gray-400">
                    {feature.desc}
                  </p>
                </div>
              </FadeInView>
            );
          })}
        </div>
      </div>

      {/* Decorative gradient blur */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-3xl h-[1px] bg-gradient-to-r from-transparent via-purple-500/50 to-transparent shadow-[0_0_15px_rgba(168,85,247,0.5)]"></div>
    </section>
  );
}
