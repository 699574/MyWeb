import { Html, Head, Main, NextScript } from 'next/document';

export default function Document() {
  return (
    <Html>
      <Head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              MathJax = {
                loader: {load: ['[tex]/ams']},
                tex: {
                  inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                  displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                  processEscapes: true,
                  processEnvironments: true,
                  packages: {'[+]': ['ams']}
                },
                options: {
                  ignoreHtmlClass: 'no-mathjax',
                  processHtmlClass: 'mathjax'
                },
                startup: {
                  ready: () => {
                    MathJax.startup.defaultReady();
                    MathJax.startup.promise.then(() => {
                      console.log('MathJax initial typesetting complete');
                    });
                  }
                }
              };
            `,
          }}
        />
        <script
          id="MathJax-script"
          async
          src="https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js"
        />
        <link
          rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/katex@0.16.0/dist/katex.min.css"
          integrity="sha384-Xi8rHCmBmhbuyyhbI88391ZKP2dmfnOl4rT9ZfRI7mLTdk1wblIUnrIq35nqwEvC"
          crossOrigin="anonymous"
        />
      </Head>
      <body className="mathjax">
        <Main />
        <NextScript />
      </body>
    </Html>
  );
} 