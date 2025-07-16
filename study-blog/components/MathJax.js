import Head from 'next/head';
import Script from 'next/script';
import { useEffect } from 'react';

export default function MathJax() {
  return (
    <>
      <Head>
        <link 
          rel="stylesheet" 
          href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css" 
          integrity="sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntxDyxJRuWUlZcP6wlbOQJE5l0C0hOUWbX" 
          crossOrigin="anonymous" 
        />
      </Head>
      
      <Script 
        id="katex-script" 
        src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"
        integrity="sha384-cpW21h6RZv/phavutF+AuVYrr+dA8xD9zs6FwLpaCct6O9ctzYFfFr4dgmgccOTx" 
        crossOrigin="anonymous"
        strategy="beforeInteractive"
      />
      
      <Script 
        id="katex-auto-render"
        src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
        integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" 
        crossOrigin="anonymous"
        strategy="afterInteractive"
        onLoad={() => {
          console.log('KaTeX auto-render 已加载');
          if (typeof window !== 'undefined' && window.renderMathInElement) {
            window.renderMathInElement(document.body, {
              delimiters: [
                {left: "$$", right: "$$", display: true},
                {left: "$", right: "$", display: false},
                {left: "\\(", right: "\\)", display: false},
                {left: "\\[", right: "\\]", display: true}
              ],
              throwOnError: false
            });
          }
        }}
      />
    </>
  );
} 