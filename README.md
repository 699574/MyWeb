# 学习笔记网站

这是一个基于Next.js和Tailwind CSS构建的个人学习笔记网站，部署在GitHub Pages上。

## 部署到GitHub Pages的步骤

1. 确保你的仓库已经推送到GitHub

2. 在GitHub仓库页面，点击"Settings"标签

3. 在左侧菜单中，点击"Pages"

4. 在"Build and deployment"部分：
   - Source: 选择"GitHub Actions"
   - 这将使用仓库中的`.github/workflows/deploy.yml`文件进行部署

5. 点击"Save"按钮

6. 等待GitHub Actions工作流完成构建和部署

7. 部署完成后，你可以通过以下URL访问你的网站：
   `https://[你的GitHub用户名].github.io/MyWeb/`

## 本地开发

1. 克隆仓库
   ```bash
   git clone https://github.com/[你的GitHub用户名]/MyWeb.git
   cd MyWeb
   ```

2. 安装依赖
   ```bash
   cd study-blog
   npm install
   ```

3. 启动开发服务器
   ```bash
   npm run dev
   ```

4. 在浏览器中打开 http://localhost:3000 查看网站

## 添加新文章

1. 在`study-blog/content/posts`目录下创建新的Markdown文件
2. 添加文章元数据（Front Matter）
   ```markdown
   ---
   title: '文章标题'
   date: '2023-07-10'
   tags: ['标签1', '标签2']
   ---

   文章内容...
   ```

3. 编写文章内容
4. 提交并推送到GitHub，GitHub Actions将自动部署更新后的网站 