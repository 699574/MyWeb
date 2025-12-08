import Link from 'next/link';

export default function LeetCodeCard({ problem }) {
  const getDifficultyColor = (difficulty) => {
    switch ((difficulty || '').toLowerCase()) {
      case 'easy':
        return 'text-green-600 bg-green-100 dark:bg-green-900 dark:text-green-400';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900 dark:text-yellow-400';
      case 'hard':
        return 'text-red-600 bg-red-100 dark:bg-red-900 dark:text-red-400';
      default:
        return 'text-gray-600 bg-gray-100 dark:bg-gray-700 dark:text-gray-400';
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6 transition-shadow">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h2 className="text-xl font-bold mb-2">
            <Link href={`/leetcode/${problem.slug}`} className="hover:text-blue-600 dark:hover:text-blue-400">
              {problem.title || problem.slug}
            </Link>
          </h2>
          {problem.number && (
            <p className="text-gray-600 dark:text-gray-400 text-sm">#{problem.number}</p>
          )}
        </div>
        {problem.difficulty && (
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getDifficultyColor(problem.difficulty)}`}>
            {problem.difficulty}
          </span>
        )}
      </div>
      {problem.excerpt && (
        <p className="text-gray-600 dark:text-gray-400 mb-4 line-clamp-3">{problem.excerpt}</p>
      )}
      {problem.tags && problem.tags.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-4">
          {problem.tags.map((tag) => (
            <Link
              key={tag}
              href={`/leetcode?tag=${encodeURIComponent(tag)}`}
              className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs rounded-full hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors"
            >
              {tag}
            </Link>
          ))}
        </div>
      )}
      <div className="flex justify-between items-center text-sm text-gray-500 dark:text-gray-400">
        {problem.date && <span>{new Date(problem.date).toLocaleDateString('zh-CN')}</span>}
        {problem.solution && (
          <span className="text-green-600 dark:text-green-400">已解决</span>
        )}
      </div>
    </div>
  );
} 