import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

export function Navbar() {
  const [activeTab, setActiveTab] = useState('Home');
  const location = useLocation();
  const navigate = useNavigate();

  useEffect(() => {
    if (location.pathname === '/screener') {
      setActiveTab('Screener');
    } else if (location.pathname === '/backtest') {
      setActiveTab('Backtest');
    } else if (location.pathname === '/') {
      setActiveTab(prev => {
        if (prev === 'Screener' || prev === 'Backtest') {
          return 'Home';
        }
        return prev;
      });
    }
  }, [location.pathname]);

  const isHome = location.pathname === '/';

  const scrollToTop = (e) => {
    e.preventDefault();
    setActiveTab('Home');
    if (!isHome) {
      navigate('/');
      setTimeout(() => window.scrollTo({ top: 0, behavior: 'smooth' }), 100);
    } else {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const scrollToAbout = (e) => {
    e.preventDefault();
    setActiveTab('About');
    if (!isHome) {
      navigate('/');
      setTimeout(() => document.getElementById('about-section')?.scrollIntoView({ behavior: 'smooth' }), 100);
    } else {
      const el = document.getElementById('about-section');
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const scrollToFeedback = (e) => {
    e.preventDefault();
    setActiveTab('Feedback');
    if (!isHome) {
      navigate('/');
      setTimeout(() => document.getElementById('feedback-section')?.scrollIntoView({ behavior: 'smooth' }), 100);
    } else {
      const el = document.getElementById('feedback-section');
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const scrollToDashboard = (e) => {
    e.preventDefault();
    if (!isHome) {
      navigate('/');
      setTimeout(() => document.getElementById('dashboard')?.scrollIntoView({ behavior: 'smooth' }), 100);
    } else {
      const el = document.getElementById('dashboard');
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const navLinks = [
    { name: 'Home', action: scrollToTop },
    { name: 'About', action: scrollToAbout },
    { name: 'Feedback', action: scrollToFeedback },
    { name: 'Screener', path: '/screener' },
    { name: 'Backtest', path: '/backtest' },
  ];

  const handleNavClick = (e, link) => {
    if (link.action) {
      link.action(e);
    } else if (link.path) {
      e.preventDefault();
      navigate(link.path);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  return (
    <div className="w-full px-6 py-6 max-w-7xl mx-auto">
      <nav className="flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-2 cursor-pointer" onClick={scrollToTop}>
          <img src="/logo.png" alt="DSM-9 Logo" className="h-12 rounded-full" onError={(e) => { e.target.style.display = 'none'; }} />
        </div>

        {/* Nav Links */}
        <div className="hidden md:flex items-center gap-8">
          {navLinks.map((link) => (
            <span
              key={link.name}
              onClick={(e) => handleNavClick(e, link)}
              className={`relative font-medium cursor-pointer pb-1 transition-colors duration-300 ${activeTab === link.name ? 'text-white' : 'text-gray-300 hover:text-white'
                }`}
            >
              {link.name}
              {/* Animated Underline */}
              <span
                className={`absolute left-0 bottom-0 h-[2px] bg-gradient-to-r from-purple-500 to-pink-500 rounded-full shadow-[0_0_8px_rgba(236,72,153,0.8)] transition-all duration-300 ease-out ${activeTab === link.name ? 'w-full opacity-100' : 'w-0 opacity-0'
                  }`}
              ></span>
            </span>
          ))}
        </div>

        {/* Action Buttons */}
        <div className="hidden md:flex items-center gap-4">

          <button
            onClick={scrollToDashboard}
            className="px-6 py-2 rounded-lg bg-gradient-to-r from-purple-600 via-pink-500 to-blue-500 text-white font-medium hover:opacity-90 transition-opacity shadow-[0_0_15px_rgba(236,72,153,0.5)]"
          >
            Access Terminal
          </button>
        </div>
      </nav>
    </div>
  );
}
