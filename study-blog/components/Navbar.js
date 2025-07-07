import Link from 'next/link';
import { useRouter } from 'next/router';

export default function Navbar() {
  const router = useRouter();
  
  const isActive = (path) => {
    return router.pathname === path ? 'text-blue-600 dark:text-blue-400' : '';
  };

  return (
    <nav className="bg-white dark:bg-gray-800 shadow-md">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center h-16">
          <div className="flex-shrink-0">
            <Link href="/" className="text-xl font-bold">
              学习笔记
            </Link>
          </div>
          
          <div className="hidden md:block">
            <div className="ml-10 flex items-baseline space-x-4">
              <Link href="/" className={`px-3 py-2 rounded-md text-sm font-medium hover:text-blue-600 dark:hover:text-blue-400 ${isActive('/')}`}>
                首页
              </Link>
              
              <Link href="/tags" className={`px-3 py-2 rounded-md text-sm font-medium hover:text-blue-600 dark:hover:text-blue-400 ${isActive('/tags')}`}>
                标签
              </Link>
              
              <Link href="/about" className={`px-3 py-2 rounded-md text-sm font-medium hover:text-blue-600 dark:hover:text-blue-400 ${isActive('/about')}`}>
                关于
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
} 