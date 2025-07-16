import { useEffect, useRef } from 'react';
import katex from 'katex';

// 处理行内公式
export function InlineMath({ formula }) {
  const ref = useRef(null);

  useEffect(() => {
    if (ref.current) {
      try {
        katex.render(formula, ref.current, {
          throwOnError: false,
          displayMode: false
        });
      } catch (err) {
        console.error('KaTeX处理行内公式失败:', err);
      }
    }
  }, [formula]);

  return <span ref={ref} className="math math-inline"></span>;
}

// 处理块级公式
export function DisplayMath({ formula }) {
  const ref = useRef(null);

  useEffect(() => {
    if (ref.current) {
      try {
        katex.render(formula, ref.current, {
          throwOnError: false,
          displayMode: true
        });
      } catch (err) {
        console.error('KaTeX处理块级公式失败:', err);
      }
    }
  }, [formula]);

  return <div ref={ref} className="math math-display"></div>;
}

// 默认导出一个通用的公式组件
export default function Formula({ tex, display = false }) {
  return display 
    ? <DisplayMath formula={tex} /> 
    : <InlineMath formula={tex} />;
} 