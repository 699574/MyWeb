import Layout from '../../components/Layout';
import { getAllLeetCodeSlugs, getLeetCodeData } from '../../lib/api';

export default function LeetCodeDetail({ problem }) {
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
    <Layout title={`${problem.title} - LeetCode 题解`}>
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <a href="/leetcode" className="text-blue-600 dark:text-blue-400 hover:underline mb-4 block">
            ← 返回题目列表
          </a>
          <h1 className="text-3xl font-bold mb-2">{problem.title}</h1>
          {problem.number && (
            <p className="text-gray-600 dark:text-gray-400 text-lg mb-4">#{problem.number}</p>
          )}
          {problem.difficulty && (
            <span className={`px-3 py-1 rounded-full text-sm font-medium ${getDifficultyColor(problem.difficulty)}`}>
              {problem.difficulty}
            </span>
          )}
        </div>
        {problem.tags && problem.tags.length > 0 && (
          <div className="mb-6 flex flex-wrap gap-2">
            {problem.tags.map((tag) => (
              <a
                key={tag}
                href={`/leetcode?tag=${encodeURIComponent(tag)}`}
                className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 text-xs rounded-full hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors"
              >
                {tag}
              </a>
            ))}
          </div>
        )}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          <div
            className="prose dark:prose-invert max-w-none"
            dangerouslySetInnerHTML={{ __html: problem.contentHtml }}
          />
        </div>
        {problem.date && (
          <div className="mt-6 text-sm text-gray-500 dark:text-gray-400">
            解决时间: {typeof problem.date === 'string' ? problem.date : new Date(problem.date).toLocaleDateString('zh-CN')}
          </div>
        )}
      </div>
    </Layout>
  );
}

export async function getStaticPaths() {
  const paths = getAllLeetCodeSlugs();
  return {
    paths,
    fallback: false,
  };
}

export async function getStaticProps({ params }) {
  const problem = await getLeetCodeData(params.slug);
  // 确保date为字符串，避免Next.js序列化错误
  if (problem.date instanceof Date) {
    problem.date = problem.date.toISOString().slice(0, 10);
  }
  return {
    props: {
      problem,
    },
  };
} 