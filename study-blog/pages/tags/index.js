import Link from 'next/link';
import Layout from '../../components/Layout';
import { getAllTags } from '../../lib/api';

export default function Tags({ tags }) {
  return (
    <Layout title="标签 | 学习笔记">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">标签</h1>
        
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
          {Object.keys(tags).length > 0 ? (
            <div className="flex flex-wrap gap-3">
              {Object.keys(tags).map((tag) => (
                <Link
                  key={tag}
                  href={`/tags/${tag}`}
                  className="inline-flex items-center bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-full px-4 py-2"
                >
                  <span className="text-sm font-medium">{tag}</span>
                  <span className="ml-2 text-xs bg-gray-200 dark:bg-gray-600 rounded-full w-6 h-6 flex items-center justify-center">
                    {tags[tag]}
                  </span>
                </Link>
              ))}
            </div>
          ) : (
            <p>暂无标签</p>
          )}
        </div>
      </div>
    </Layout>
  );
}

export async function getStaticProps() {
  const tags = getAllTags();
  return {
    props: {
      tags,
    },
  };
} 