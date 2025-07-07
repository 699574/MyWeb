import Layout from '../components/Layout';

export default function About() {
  return (
    <Layout title="关于 | 学习笔记">
      <div className="max-w-3xl mx-auto bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden">
        <div className="p-6">
          <h1 className="text-3xl font-bold mb-6">关于我</h1>
          
          <div className="prose dark:prose-invert max-w-none">
            <p>
              这是我的个人学习笔记网站，用于记录我在学习过程中的心得体会、感悟和总结。
              通过这个网站，我希望能够：
            </p>
            
            <ul>
              <li>记录学习过程中的重要知识点</li>
              <li>分享学习经验和方法</li>
              <li>记录自己的成长历程</li>
              <li>形成自己的知识体系</li>
            </ul>
            
            <h2>网站技术栈</h2>
            <p>本网站使用以下技术构建：</p>
            <ul>
              <li>内容格式：Markdown (.md)</li>
              <li>前端框架：Next.js</li>
              <li>CSS 框架：Tailwind CSS</li>
              <li>部署平台：GitHub Pages</li>
            </ul>
            
            <h2>联系我</h2>
            <p>
              如果你有任何问题、建议或者想法，欢迎通过以下方式联系我：
            </p>
            <ul>
              <li>Email: your.email@example.com</li>
              <li>GitHub: <a href="https://github.com/yourusername" target="_blank" rel="noopener noreferrer">github.com/yourusername</a></li>
            </ul>
          </div>
        </div>
      </div>
    </Layout>
  );
} 