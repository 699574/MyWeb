import Link from 'next/link';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';

export default function PostCard({ post }) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden mb-6 transition-transform hover:scale-[1.01]">
      <div className="p-6">
        <Link href={`/posts/${post.slug}`} className="block">
          <h2 className="text-xl font-semibold mb-2 hover:text-blue-600 dark:hover:text-blue-400">
            {post.title}
          </h2>
        </Link>
        
        <div className="text-sm text-gray-500 dark:text-gray-400 mb-4">
          <time dateTime={post.date}>
            {format(new Date(post.date), 'yyyy年MM月dd日', { locale: zhCN })}
          </time>
          {post.tags && (
            <span className="ml-4">
              {post.tags.map((tag) => (
                <Link 
                  href={`/tags/${tag}`}
                  key={tag}
                  className="inline-block bg-gray-100 dark:bg-gray-700 rounded-full px-2 py-1 text-xs mr-2 hover:bg-gray-200 dark:hover:bg-gray-600"
                >
                  #{tag}
                </Link>
              ))}
            </span>
          )}
        </div>
        
        <Link href={`/posts/${post.slug}`} className="block">
          {post.excerpt && (
            <p className="text-gray-600 dark:text-gray-300">
              {post.excerpt}
            </p>
          )}
        </Link>
      </div>
    </div>
  );
} 