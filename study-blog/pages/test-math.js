import dynamic from 'next/dynamic';

const TestMathContent = dynamic(() => import('../components/TestMathContent'), { ssr: false });

export default function TestMath() {
  return <TestMathContent />;
} 