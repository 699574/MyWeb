import { useEffect } from 'react';
import Layout from '../components/Layout';
import dynamic from 'next/dynamic';
import Script from 'next/script';
// import { InlineMath, DisplayMath } from '../components/Formula';
// import MathJax from '../components/MathJax';
const InlineMath = dynamic(() => import('../components/Formula').then(mod => mod.InlineMath), { ssr: false });
const DisplayMath = dynamic(() => import('../components/Formula').then(mod => mod.DisplayMath), { ssr: false });
const MathJax = dynamic(() => import('../components/MathJax'), { ssr: false });

export default function TestMath() {
  return (
    <Layout title="数学公式测试">
      <MathJax />
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-6">数学公式渲染测试</h1>
        
        <section className="mb-8">
          <h2 className="text-2xl font-bold mb-4">组件方式渲染</h2>
          <h3 className="text-xl font-bold mb-2">行内公式</h3>
          <p className="mb-4">
            这是一个行内公式：<InlineMath formula="E = mc^2" />，它应该和文本在同一行。
          </p>
          <p className="mb-4">
            这是另一个行内公式：<InlineMath formula="\alpha + \beta = \gamma" />，它不应该导致换行。
          </p>
          <p className="mb-4">
            这是一个包含下标的行内公式：<InlineMath formula="x_{i} + y_{i} = z_{i}" />。
          </p>
          
          <h3 className="text-xl font-bold mb-2 mt-4">块级公式</h3>
          <p className="mb-2">以下是一个块级公式：</p>
          <DisplayMath formula="\sum_{i=1}^{n} i = \frac{n(n+1)}{2}" />
          
          <p className="mb-2">以下是另一个块级公式：</p>
          <DisplayMath formula="f(x) = \int_{-\infty}^{\infty} \hat{f}(\xi) e^{2\pi i \xi x} d\xi" />
        </section>
        
        <section className="mb-8">
          <h2 className="text-2xl font-bold mb-4">原生Markdown语法测试</h2>
          <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
            <p className="mb-4">
              这是使用单美元符号的行内公式：$E = mc^2$，它应该和文本在同一行。
            </p>
            <p className="mb-4">
              这是另一个行内公式：$\alpha + \beta = \gamma$，它不应该导致换行。
            </p>
            <p className="mb-4">
              这是使用双美元符号的块级公式：
            </p>
            <p className="mb-4 text-center">
              $$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$
            </p>
            <p className="mb-4">
              以下是另一个块级公式：
            </p>
            <p className="mb-4 text-center">
              {'$$f(x) = \\int_{-\\infty}^{\\infty} \\hat{f}(\\xi) e^{2\\pi i \\xi x} d\\xi$$'}
            </p>
          </div>
        </section>
        
        <section className="mb-8">
          <h2 className="text-2xl font-bold mb-4">RL1.md中的公式测试</h2>
          <div className="bg-gray-100 dark:bg-gray-800 p-4 rounded-lg">
            <p className="mb-4">
              状态 $s_t$ 满足马尔可夫性当且仅当：给定当前状态和行动 $a_t$，未来的状态 $s_{t+1}$ 与过去的完整历史 $h_t$ 无关。即
            </p>
            <p className="mb-4 text-center">
              {'$$p(s_{t+1} | s_t, a_t) = p(s_{t+1} | h_t, a_t)$$'}
            </p>
            <p className="mb-4">
              状态价值函数 $V(s)$ 对于马尔可夫奖励过程，是从状态 $s$ 开始的预期回报。
            </p>
            <p className="mb-4 text-center">
              {'$$V(s) = E[G_t | S_t = s] = E[r_t + \\gamma r_{t+1} + \\gamma^2 r_{t+2} + \\dots + \\gamma^{H-1} r_{t+H-1} | S_t = s]$$'}
            </p>
          </div>
        </section>
      </div>
    </Layout>
  );
} 