import { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import Layout from '../components/Layout';
import LeetCodeCard from '../components/LeetCodeCard';
import { getSortedLeetCodeData, getAllLeetCodeTags } from '../lib/api';

export default function LeetCode({ allProblems, allTags }) {
  const router = useRouter();
  const [filteredProblems, setFilteredProblems] = useState(allProblems);
  const [currentTag, setCurrentTag] = useState('');

  useEffect(() => {
    const tag = router.query.tag || '';
    setCurrentTag(tag);
    
    if (tag) {
      const filtered = allProblems.filter(problem => 
        problem.tags && problem.tags.includes(tag)
      );
      setFilteredProblems(filtered);
    } else {
      setFilteredProblems(allProblems);
    }
  }, [router.query.tag, allProblems]);

  const handleTagClick = (tag) => {
    if (tag === currentTag) {
      router.push('/leetcode');
    } else {
      router.push(`/leetcode?tag=${encodeURIComponent(tag)}`);
    }
  };

  return (
    <Layout title="LeetCode 题解">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">LeetCode 题解</h1>
          <p className="text-gray-600 dark:text-gray-400">收集我的 LeetCode 题目与代码，支持标签检索。</p>
        </div>
        <div className="mb-6 flex flex-wrap gap-2">
          <button
            className={`px-3 py-1 rounded-full text-sm font-medium border ${!currentTag ? 'bg-blue-600 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200'}`}
            onClick={() => handleTagClick('')}
          >
            全部
          </button>
          {Object.keys(allTags).map((tag) => (
            <button
              key={tag}
              className={`px-3 py-1 rounded-full text-sm font-medium border ${currentTag === tag ? 'bg-blue-600 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200'}`}
              onClick={() => handleTagClick(tag)}
            >
              {tag} <span className="ml-1 text-xs">({allTags[tag]})</span>
            </button>
          ))}
        </div>
        <div>
          {filteredProblems.length > 0 ? (
            filteredProblems.map((problem) => (
              <LeetCodeCard key={problem.slug} problem={problem} />
            ))
          ) : (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <p>暂无题目，快去刷题吧！</p>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}

export async function getStaticProps() {
  const allProblems = getSortedLeetCodeData();
  const allTags = getAllLeetCodeTags();
  
  return {
    props: {
      allProblems,
      allTags,
    },
  };
} 