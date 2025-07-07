import Layout from '../components/Layout';
import PostCard from '../components/PostCard';
import { getSortedPostsData } from '../lib/api';

export default function Home({ allPostsData }) {
  return (
    <Layout title="首页 | 学习笔记">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">我的学习笔记</h1>
          <p className="text-gray-600 dark:text-gray-400">
            记录我的学习历程、心得体会和感悟
          </p>
        </div>

        <div className="mb-8">
          {allPostsData.length > 0 ? (
            allPostsData.map((post) => (
              <PostCard key={post.slug} post={post} />
            ))
          ) : (
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
              <p>暂无文章，开始写作吧！</p>
            </div>
          )}
        </div>
      </div>
    </Layout>
  );
}

export async function getStaticProps() {
  const allPostsData = getSortedPostsData();
  return {
    props: {
      allPostsData,
    },
  };
} 