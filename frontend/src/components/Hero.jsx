import React from 'react';
import { ArrowDown, Play, Users, Target, Activity, Cpu, LineChart, Shield, Globe } from 'lucide-react';
import { motion } from 'framer-motion';
import { TypewriterText } from './TypewriterText';

const FeatureCard = ({ icon, title, desc }) => (
  <div className="flex items-center gap-4 bg-[#111326]/60 backdrop-blur-md border border-[#2a2a4a] p-4 rounded-xl hover:bg-[#1a1c3a]/80 transition-colors shadow-lg">
    <div className="flex-shrink-0 bg-[#0a0a16] p-2 rounded-lg border border-[#1a1a3a] shadow-[0_0_15px_rgba(168,85,247,0.3)] group-hover:shadow-[0_0_20px_rgba(168,85,247,0.6)]">
      {icon}
    </div>
    <div className="flex flex-col">
      <span className="text-white font-bold text-sm tracking-wide">{title}</span>
      <span className="text-gray-400 text-xs mt-0.5">{desc}</span>
    </div>
  </div>
);

export function Hero() {
  const scrollToDashboard = (e) => {
    e.preventDefault();
    const element = document.getElementById('dashboard');
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <section className="relative min-h-[100vh] flex items-center justify-start bg-transparent pt-24 overflow-hidden">

      {/* Background Image with Overlay */}
      <div
        className="absolute inset-0 z-0 opacity-30 bg-cover bg-center bg-no-repeat transition-opacity duration-1000 mix-blend-screen"
        style={{ backgroundImage: "url('/hero-bg.png')" }}
      />
      <div className="absolute inset-0 z-0 bg-gradient-to-b from-[#090b14]/40 via-transparent to-[#05060a]"></div>

      <div className="relative z-10 px-6 md:px-12 xl:px-32 w-full max-w-7xl flex flex-col items-start text-left mt-10 md:mt-20">

        {/* Main Title Area */}
        <div className="flex flex-col mb-5 -mt-10">
          <motion.h1
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="text-5xl md:text-7xl lg:text-8xl font-black tracking-tight leading-[1.05] m-0 p-0 text-white drop-shadow-md"
          >
            PRICE
          </motion.h1>
          <motion.h1
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: "easeOut", delay: 0.2 }}
            className="text-5xl md:text-7xl lg:text-8xl font-black tracking-tight leading-[1.05] m-0 p-0 text-transparent bg-clip-text bg-gradient-to-r from-pink-500 via-purple-400 to-cyan-400 drop-shadow-md pb-1"
          >
            PREDICTION
          </motion.h1>
          <motion.h1
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: "easeOut", delay: 0.4 }}
            className="text-5xl md:text-7xl lg:text-8xl font-black tracking-tight leading-[1.05] m-0 p-0 text-white drop-shadow-md"
          >
            MODEL
          </motion.h1>
        </div>

        {/* Subtitles */}
        <div className="flex items-center gap-3 text-gray-300 font-semibold tracking-widest text-xs md:text-sm mt-4 animate-fade-in-up"
         style={{ animationDelay: '0.1s' }}>
          <motion.span className="w-8 h-[1px] bg-gray-500 hidden md:block"
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          ></motion.span>
          <motion.span 
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, ease: "easeOut",  delay: 0.4 }}>Predict</motion.span>
           <motion.span 
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, ease: "easeOut", delay: 0.8 }}>Analyze</motion.span>  <motion.span
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.8, ease: "easeOut", delay: 1.2 }}>Maximize Profits</motion.span> 
        </div>

        <p className="mt-8 text-gray-300 text-base md:text-lg max-w-md font-medium leading-relaxed animate-fade-in-up border-l-2 border-purple-500/50 pl-4 min-h-[60px]" style={{ animationDelay: '0.2s' }}>
          <TypewriterText text="AI-Powered Market Forecasting for " speed={30} delay={800} />
          <br className="hidden md:block" />
          <span className="text-white font-bold">
            <TypewriterText text="Smarter Investment Decisions" speed={30} delay={1800} />
          </span>
        </p>

        {/* Action Buttons */}
        <div className="flex items-center gap-6 mt-10 mb-20 animate-fade-in-up" style={{ animationDelay: '0.3s' }}>
          <button
            onClick={scrollToDashboard}
            className="group relative flex items-center justify-center space-x-2 bg-gradient-to-r from-pink-500 via-purple-500 to-cyan-500 text-white px-8 py-3 rounded-full font-bold text-base transition-transform hover:scale-105 shadow-[0_0_20px_rgba(236,72,153,0.5)]"
          >
            <span>Get Started</span>
            <span className="ml-1 group-hover:translate-x-1 transition-transform">→</span>
          </button>
        </div>



        {/* Features Row */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 w-full mt-4 pb-20 animate-fade-in-up mt-10" style={{ animationDelay: '0.5s' }}>
          <FeatureCard
            icon={<Cpu className="w-6 h-6 text-purple-400" />}
            title="AI & Machine Learning"
            desc="Advanced Algorithms"
          />
          <FeatureCard
            icon={<LineChart className="w-6 h-6 text-cyan-400" />}
            title="Real-Time Data"
            desc="Live Market Trends"
          />
          <FeatureCard
            icon={<Shield className="w-6 h-6 text-blue-400" />}
            title="Risk Analysis"
            desc="Smart Insights"
          />
          <FeatureCard
            icon={<Globe className="w-6 h-6 text-pink-400" />}
            title="Global Markets"
            desc="Crypto • Stocks • Forex"
          />
        </div>
      </div>
    </section>
  );
}
