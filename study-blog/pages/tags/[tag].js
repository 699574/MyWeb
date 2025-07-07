import Link from 'next/link';
import Layout from '../../components/Layout';
import PostCard from '../../components/PostCard';
import { getAllTags, getPostsByTag } from '../../lib/api';

export default function Tag({ tag, posts }) {
  return (
    <Layout title={`${tag} | 学习笔记`}>
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center mb-8">
          <h1 className="text-3xl font-bold">#{tag}</h1>
          <span className="ml-4 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded-full px-3 py-1 text-sm">
            {posts.length} 篇文章
          </span>
        </div>
        
        <div className="mb-8">
          {posts.map((post) => (
            <PostCard key={post.slug} post={post} />
          ))}
        </div>
        
        <div>
          <Link 
            href="/tags"
            className="inline-flex items-center text-blue-600 dark:text-blue-400 hover:underline"
          >
            ← 返回所有标签
          </Link>
        </div>
      </div>
    </Layout>
  );
}

export async function getStaticPaths() {
  const tags = getAllTags();
  
  return {
    paths: Object.keys(tags).map((tag) => ({
      params: { tag },
    })),
    fallback: false,
  };
}

export async function getStaticProps({ params }) {
  const { tag } = params;
  const posts = getPostsByTag(tag);
  
  return {
    props: {
      tag,
      posts,
    },
  };
} 