import React, { useState, useEffect, useRef } from 'react';

export function TypewriterText({ text, speed = 25, delay = 0 }) {
  const [displayedText, setDisplayedText] = useState("");
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setIsVisible(true);
        observer.unobserve(entry.target);
      }
    }, { threshold: 0.5 });

    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (isVisible) {
      let i = 0;
      let interval;

      const timeout = setTimeout(() => {
        interval = setInterval(() => {
          setDisplayedText(text.slice(0, i));
          i++;
          if (i > text.length) clearInterval(interval);
        }, speed);
      }, delay);

      return () => {
        clearTimeout(timeout);
        clearInterval(interval);
      };
    }
  }, [isVisible, text, speed, delay]);

  return <span ref={ref}>{displayedText}</span>;
}
