import Head from 'next/head';
import Navbar from './Navbar';
import MathJax from './MathJax';

export default function Layout({ children, title = '学习笔记' }) {
  return (
    <div className="flex flex-col min-h-screen">
      <Head>
        <title>{title}</title>
        <meta name="description" content="我的学习笔记和心得体会" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <MathJax />
      
      <Navbar />

      <main className="container mx-auto px-4 py-8 flex-grow">
        {children}
      </main>

      <footer className="bg-gray-100 dark:bg-gray-800 py-6">
        <div className="container mx-auto px-4 text-center">
          <p>&copy; {new Date().getFullYear()} 我的学习笔记. 基于 Next.js 和 Tailwind CSS 构建.</p>
        </div>
      </footer>
    </div>
  );
} 