import React, { useState } from 'react';
import { Send, MessageSquare } from 'lucide-react';

export function Feedback() {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    feedback: ''
  });
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    // Simulate API call for feedback submission
    console.log("Feedback submitted:", formData);
    setIsSubmitted(true);
    setTimeout(() => {
      setIsSubmitted(false);
      setFormData({ name: '', email: '', feedback: '' });
    }, 3000);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  return (
    <section className="py-24 relative overflow-hidden bg-[#05060a]">
      {/* Background gradients for Feedback section to match the site theme */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full max-w-3xl h-[1px] bg-gradient-to-r from-transparent via-cyan-500/30 to-transparent shadow-[0_0_15px_rgba(34,211,238,0.3)]"></div>

      <div className="max-w-4xl mx-auto px-6 lg:px-8 relative z-10">
        <div className="text-center mb-16">
          <h2 className="text-base/7 font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-400 uppercase tracking-widest drop-shadow-sm flex items-center justify-center gap-2">
            <MessageSquare className="w-5 h-5 text-cyan-400" />
            We Value Your Input
          </h2>
          <p className="mt-2 text-4xl font-semibold tracking-tight text-white sm:text-5xl drop-shadow-[0_0_10px_rgba(255,255,255,0.2)]">
            Send Us Your Feedback
          </p>
          <p className="mt-4 text-lg text-gray-400 max-w-2xl mx-auto">
            Help us improve DSM-9. Let us know how the predictive engine is working for you or what features you'd like to see next.
          </p>
        </div>

        <div className="bg-[#111326]/50 backdrop-blur-xl border border-[#2a2a4a] rounded-2xl p-8 sm:p-12 shadow-[0_10px_40px_rgba(0,0,0,0.5)] max-w-2xl mx-auto relative overflow-hidden group">
          {/* Subtle hover glow effect */}
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>

          {isSubmitted ? (
            <div className="text-center py-16 animate-in fade-in zoom-in duration-500">
              <div className="w-16 h-16 bg-gradient-to-br from-green-400 to-cyan-500 rounded-full flex items-center justify-center mx-auto mb-6 shadow-[0_0_20px_rgba(34,211,238,0.4)]">
                <Send className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-2xl font-bold text-white mb-2">Message Sent!</h3>
              <p className="text-gray-400">Thank you for helping us improve DSM-9.</p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-6 relative z-10">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <label htmlFor="name" className="text-sm font-medium text-gray-300 ml-1">Full Name</label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    required
                    value={formData.name}
                    onChange={handleChange}
                    className="w-full bg-[#0a0a16] border border-[#2a2a4a] rounded-xl px-4 py-3.5 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all font-medium"
                    placeholder="John Doe"
                  />
                </div>
                <div className="space-y-2">
                  <label htmlFor="email" className="text-sm font-medium text-gray-300 ml-1">Email (Gmail preferred)</label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    required
                    value={formData.email}
                    onChange={handleChange}
                    className="w-full bg-[#0a0a16] border border-[#2a2a4a] rounded-xl px-4 py-3.5 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all font-medium"
                    placeholder="john@gmail.com"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <label htmlFor="feedback" className="text-sm font-medium text-gray-300 ml-1">Your Feedback</label>
                <textarea
                  id="feedback"
                  name="feedback"
                  required
                  rows={4}
                  value={formData.feedback}
                  onChange={handleChange}
                  className="w-full bg-[#0a0a16] border border-[#2a2a4a] rounded-xl px-4 py-4 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500 transition-all resize-none font-medium"
                  placeholder="Tell us what you think..."
                />
              </div>

              <button
                type="submit"
                className="w-full bg-gradient-to-r from-cyan-500 to-purple-500 hover:brightness-110 text-white font-bold py-4 px-8 rounded-xl transition-all shadow-[0_0_20px_rgba(34,211,238,0.3)] hover:shadow-[0_0_30px_rgba(34,211,238,0.5)] active:scale-95 flex items-center justify-center gap-2 mt-4"
              >
                <Send className="w-5 h-5" />
                Submit Feedback
              </button>
            </form>
          )}
        </div>
      </div>
    </section>
  );
}
