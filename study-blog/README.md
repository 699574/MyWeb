# 学习笔记网站

这是一个基于 Next.js 和 Tailwind CSS 构建的个人学习笔记网站，部署在 GitHub Pages 上。

## 功能特点

- 使用 Markdown 编写文章
- 支持文章分类和标签
- 响应式设计，适配各种设备
- 支持暗色模式
- 自动生成文章摘要

## 技术栈

- **内容格式**: Markdown (.md)
- **前端框架**: Next.js
- **CSS 框架**: Tailwind CSS
- **部署平台**: GitHub Pages

## 快速开始

### 本地开发

1. 克隆仓库
   ```bash
   git clone https://github.com/yourusername/study-blog.git
   cd study-blog
   ```

2. 安装依赖
   ```bash
   npm install
   ```

3. 启动开发服务器
   ```bash
   npm run dev
   ```

4. 在浏览器中打开 http://localhost:3000 查看网站

### 添加新文章

1. 在 `content/posts` 目录下创建新的 Markdown 文件
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

### 构建和部署

网站会在推送到 main 分支时自动部署到 GitHub Pages。

如果需要手动构建：

```bash
npm run build
```

构建后的静态文件将生成在 `out` 目录中。

## 自定义

- 修改 `components/Layout.js` 更改网站布局
- 修改 `styles/globals.css` 自定义样式
- 修改 `next.config.js` 调整 Next.js 配置

## 许可证

MIT 