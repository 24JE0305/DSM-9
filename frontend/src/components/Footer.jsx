import React from 'react';
import { Github, Twitter, Linkedin } from 'lucide-react';

export function Footer() {
  return (
    <footer className="bg-[#05060a] border-t border-[#2a2a4a] pt-12 pb-8 relative z-10">
      <div className="max-w-7xl mx-auto px-6 lg:px-8 flex flex-col items-center justify-between space-y-8 md:flex-row md:space-y-0">

        {/* Logo & Info */}
        <div className="flex flex-col items-center md:items-start space-y-2">
          <span className="text-2xl font-black italic tracking-wider text-transparent bg-clip-text bg-gradient-to-r from-white to-gray-400">
            DSM-9
          </span>
          <p className="text-sm text-gray-500 max-w-sm text-center md:text-left">
            Empowering the modern investor with real-time deep learning analytics.
          </p>
        </div>

        {/* Social Links */}
        <div className="flex space-x-6 text-gray-400">
          <a href="#" className="hover:text-white transition-colors">
            <span className="sr-only">GitHub</span>
            <Github className="w-5 h-5" />
          </a>
          <a href="#" className="hover:text-white transition-colors">
            <span className="sr-only">Twitter</span>
            <Twitter className="w-5 h-5" />
          </a>
          <a href="#" className="hover:text-white transition-colors">
            <span className="sr-only">LinkedIn</span>
            <Linkedin className="w-5 h-5" />
          </a>
        </div>

      </div>

      <div className="max-w-7xl mx-auto px-6 lg:px-8 mt-12 pt-8 border-t border-[#2a2a4a]/50 flex flex-col items-center justify-between text-xs text-gray-500 md:flex-row gap-4">
        <p>&copy; {new Date().getFullYear()} DSM-9 Predictive Intelligence. All rights reserved.</p>
        <div className="flex space-x-6">
          <a href="#" className="hover:text-gray-400">Privacy Policy</a>
          <a href="#" className="hover:text-gray-400">Terms of Service</a>
          <a href="#" className="hover:text-gray-400">Disclaimer</a>
        </div>
      </div>
    </footer>
  );
}
