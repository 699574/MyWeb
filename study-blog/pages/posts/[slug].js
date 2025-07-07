import { useRouter } from 'next/router';
import Link from 'next/link';
import { format } from 'date-fns';
import { zhCN } from 'date-fns/locale';
import Layout from '../../components/Layout';
import { getAllPostSlugs, getPostData } from '../../lib/api';

export default function Post({ postData }) {
  const router = useRouter();

  if (router.isFallback) {
    return <div>加载中...</div>;
  }

  return (
    <Layout title={`${postData.title} | 学习笔记`}>
      <article className="max-w-3xl mx-auto bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
        <div className="p-6">
          <h1 className="text-3xl font-bold mb-4">{postData.title}</h1>
          
          <div className="flex items-center text-sm text-gray-500 dark:text-gray-400 mb-6">
            <time dateTime={postData.date}>
              {format(new Date(postData.date), 'yyyy年MM月dd日', { locale: zhCN })}
            </time>
            
            {postData.tags && (
              <div className="ml-4">
                {postData.tags.map((tag) => (
                  <Link 
                    href={`/tags/${tag}`}
                    key={tag}
                    className="inline-block bg-gray-100 dark:bg-gray-700 rounded-full px-2 py-1 text-xs mr-2 hover:bg-gray-200 dark:hover:bg-gray-600"
                  >
                    #{tag}
                  </Link>
                ))}
              </div>
            )}
          </div>
          
          <div 
            className="prose dark:prose-invert max-w-none"
            dangerouslySetInnerHTML={{ __html: postData.contentHtml }} 
          />
        </div>
      </article>
      
      <div className="max-w-3xl mx-auto mt-6">
        <Link 
          href="/"
          className="inline-flex items-center text-blue-600 dark:text-blue-400 hover:underline"
        >
          ← 返回首页
        </Link>
      </div>
    </Layout>
  );
}

export async function getStaticPaths() {
  const paths = getAllPostSlugs();
  return {
    paths,
    fallback: false,
  };
}

export async function getStaticProps({ params }) {
  const postData = await getPostData(params.slug);
  return {
    props: {
      postData,
    },
  };
} 