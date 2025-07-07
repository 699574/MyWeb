import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { remark } from 'remark';
import html from 'remark-html';

const postsDirectory = path.join(process.cwd(), 'content', 'posts');

export function getSortedPostsData() {
  // 获取posts目录下的所有文件名
  const fileNames = fs.readdirSync(postsDirectory);
  const allPostsData = fileNames.map((fileName) => {
    // 移除文件名中的".md"以获取id
    const slug = fileName.replace(/\.md$/, '');

    // 将markdown文件读取为字符串
    const fullPath = path.join(postsDirectory, fileName);
    const fileContents = fs.readFileSync(fullPath, 'utf8');

    // 使用gray-matter解析文章元数据
    const matterResult = matter(fileContents);

    // 创建摘要
    const excerpt = matterResult.content
      .trim()
      .split('\n')
      .slice(0, 3)
      .join(' ')
      .substring(0, 150) + '...';

    // 将数据与id组合
    return {
      slug,
      excerpt,
      ...matterResult.data,
    };
  });

  // 按日期排序
  return allPostsData.sort((a, b) => {
    if (a.date < b.date) {
      return 1;
    } else {
      return -1;
    }
  });
}

export function getAllPostSlugs() {
  const fileNames = fs.readdirSync(postsDirectory);
  return fileNames.map((fileName) => {
    return {
      params: {
        slug: fileName.replace(/\.md$/, ''),
      },
    };
  });
}

export async function getPostData(slug) {
  const fullPath = path.join(postsDirectory, `${slug}.md`);
  const fileContents = fs.readFileSync(fullPath, 'utf8');

  // 使用gray-matter解析文章元数据
  const matterResult = matter(fileContents);

  // 使用remark将markdown转换为HTML字符串
  const processedContent = await remark()
    .use(html)
    .process(matterResult.content);
  const contentHtml = processedContent.toString();

  // 将数据与id和contentHtml组合
  return {
    slug,
    contentHtml,
    ...matterResult.data,
  };
}

export function getAllTags() {
  const posts = getSortedPostsData();
  const tags = {};

  posts.forEach((post) => {
    if (post.tags && Array.isArray(post.tags)) {
      post.tags.forEach((tag) => {
        if (!tags[tag]) {
          tags[tag] = 1;
        } else {
          tags[tag]++;
        }
      });
    }
  });

  return tags;
}

export function getPostsByTag(tag) {
  const posts = getSortedPostsData();
  return posts.filter((post) => post.tags && post.tags.includes(tag));
} 