import React, { useEffect, useRef, useState } from 'react';
import { Database, Zap, Shield, TrendingUp } from 'lucide-react';
import { TypewriterText } from './TypewriterText';

function FadeInUp({ children, delay = 0 }) {
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

  return (
    <div
      ref={ref}
      className={`transition-all duration-1000 transform w-full ${isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-16'}`}
      style={{ transitionDelay: `${delay}ms` }}
    >
      {children}
    </div>
  );
}

const FlipCard = ({ icon, title, desc, imageUrl }) => {
  return (
    <div className="group w-full h-[320px] [perspective:1000px]">
      <div className="relative w-full h-full transition-transform duration-700 [transform-style:preserve-3d] group-hover:[transform:rotateY(180deg)]">
        {/* Front Side */}
        <div className="absolute inset-0 bg-[#111326]/60 backdrop-blur-md ring-1 ring-[#2a2a4a] group-hover:ring-purple-500/50 shadow-[0_10px_30px_rgba(0,0,0,0.5)] group-hover:shadow-[0_0_30px_rgba(168,85,247,0.3)] rounded-2xl flex flex-col items-center justify-center text-center overflow-hidden [backface-visibility:hidden]">

          {/* Background Image with Overlay */}
          {imageUrl && (
            <>
              <div
                className="absolute inset-0 bg-cover bg-center bg-no-repeat opacity-60 group-hover:opacity-80 transition-opacity duration-500"
                style={{ backgroundImage: `url(${imageUrl})` }}
              />
              <div className="absolute inset-0 bg-gradient-to-t from-[#0a0a16] via-[#111326]/80 to-transparent"></div>
            </>
          )}

          {/* Text Content aligned to the bottom */}
          <div className="relative z-10 p-8 flex flex-col items-center justify-end w-full h-full pb-10">
            
            <h3 className="text-2xl font-bold tracking-tight text-white drop-shadow-md text-center">
              {title}
            </h3>
            <p className="text-gray-300 mt-2 text-sm font-medium opacity-90 flex items-center gap-1 drop-shadow-sm group-hover:text-white transition-colors">
              <span className="animate-pulse">Hover to see details</span>
            </p>
          </div>
        </div>

        {/* Back Side */}
        <div className="absolute inset-0 bg-gradient-to-br from-[#1a1c3a] to-[#0a0a16] backdrop-blur-xl ring-1 ring-purple-500/60 shadow-[0_0_40px_rgba(168,85,247,0.2)] rounded-2xl p-8 flex flex-col items-center justify-center text-center [transform:rotateY(180deg)] [backface-visibility:hidden]">
          <h3 className="text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-cyan-400 mb-4 tracking-wider uppercase text-sm">{title}</h3>
          <p className="text-lg text-gray-300 leading-relaxed font-medium">
            {desc}
          </p>
        </div>
      </div>
    </div>
  );
};


export function About() {
  const features = [
    {
      icon: <Database className="w-12 h-12 text-cyan-400 drop-shadow-[0_0_8px_rgba(34,211,238,0.5)] flex-shrink-0" />,
      title: 'Hybrid Machine Learning',
      desc: 'Fusing neural networks with advanced tree-based algorithms for superior predictive accuracy and reduced overfitting.',
      imageUrl: 'https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=800&auto=format&fit=crop' // Placeholder: Abstract tech network
    },
    {
      icon: <Zap className="w-12 h-12 text-purple-400 drop-shadow-[0_0_8px_rgba(168,85,247,0.5)] flex-shrink-0" />,
      title: 'Historical & Active Data',
      desc: 'Processing years of historical price action alongside current market momentum to calculate robust trajectories.',
      imageUrl: 'https://images.unsplash.com/photo-1642543492481-44e81e3914a7?q=80&w=800&auto=format&fit=crop' // Placeholder: Stock charts on screens
    },
    {
      icon: <Shield className="w-12 h-12 text-pink-400 drop-shadow-[0_0_8px_rgba(236,72,153,0.5)] flex-shrink-0" />,
      title: 'Dynamic Risk Scoring',
      desc: 'Evaluating model agreement and localized asset volatility to keep you informed of potential market turbulence.',
      imageUrl: 'https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=800&auto=format&fit=crop' // Placeholder: Data analytics visualization
    },
    {
      icon: <TrendingUp className="w-12 h-12 text-blue-400 drop-shadow-[0_0_8px_rgba(59,130,246,0.5)] flex-shrink-0" />,
      title: 'Multi-Horizon Targets',
      desc: 'Projecting expected price targets across short, medium, and long-term investment windows.',
      imageUrl: 'https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=800&auto=format&fit=crop' // Placeholder: Rising abstract graph
    }
  ];

  return (
    <section className="py-24 relative overflow-hidden">
      <div className="max-w-6xl mx-auto px-6 lg:px-8">
        <div className="mx-auto max-w-3xl text-center mb-24">
          <h2 className="text-base/7 font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-cyan-400 uppercase tracking-widest drop-shadow-sm">The Architecture</h2>
          <p className="mt-2 text-4xl font-semibold tracking-tight text-white sm:text-5xl drop-shadow-[0_0_10px_rgba(255,255,255,0.2)]">
            What powers DSM-9?
          </p>

          <div className="mt-12 text-left bg-[#111326]/50 p-6 sm:p-8 rounded-2xl border border-[#2a2a4a] shadow-xl">
            <h3 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
              <Zap className="w-6 h-6 text-purple-400" />
              The Vision Behind DSM-9
            </h3>
            <div className="text-gray-300 space-y-4 text-lg leading-relaxed">
              <p>
                For decades, institutional hedge funds and Wall Street quantitative firms have possessed a massive advantage: predictive AI. Meanwhile, everyday investors have been forced to rely on delayed news, emotion, and outdated technical indicators.
              </p>
              <p>
                DSM-9 was built to level the playing field. We engineered a system that ingests millions of data points and recognizes hidden market patterns that the human eye simply cannot see. We don't believe in "get-rich-quick" schemes or crystal balls. We believe in data, probability, and rigorous risk management. DSM-9 is your ultimate quantitative copilot, designed to remove human emotion from trading and replace it with pure, calculated foresight.
              </p>
            </div>
          </div>

          <p className="mt-12 text-xl/8 text-gray-400 min-h-[96px]">
            <TypewriterText text="DSM-9 is a state-of-the-art predictive engine built specifically for the Indian equity market. By fusing the sequential pattern recognition of Deep Learning (LSTM) with the decision-tree precision of Gradient Boosting (XGBoost), it provides data-driven foresight into asset price movements to serve as your quantitative copilot." />
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-5xl mx-auto mt-16">
          {features.map((feature, idx) => (
            <FadeInUp key={feature.title} delay={idx * 150}>
              <FlipCard
                icon={feature.icon}
                title={feature.title}
                desc={feature.desc}
                imageUrl={feature.imageUrl}
              />
            </FadeInUp>
          ))}
        </div>
      </div>

      {/* Decorative gradient blur */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-3xl h-[1px] bg-gradient-to-r from-transparent via-purple-500/50 to-transparent shadow-[0_0_15px_rgba(168,85,247,0.5)]"></div>
    </section>
  );
}
