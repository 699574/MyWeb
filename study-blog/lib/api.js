import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { remark } from 'remark';
import html from 'remark-html';
import gfm from 'remark-gfm';

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

// 处理LaTeX公式，确保其中的特殊字符不被Markdown解析器处理
function processLatexFormulas(content) {
  // 存储所有公式的映射
  const formulas = [];
  let counter = 0;
  
  // 替换块级公式 ($$...$$)
  let processedContent = content.replace(/\$\$([\s\S]*?)\$\$/g, (match, formula) => {
    const placeholder = `LATEX_BLOCK_${counter}`;
    formulas.push({ placeholder, formula: match });
    counter++;
    return placeholder;
  });
  
  // 替换行内公式 ($...$)
  processedContent = processedContent.replace(/\$([^\$\n]+?)\$/g, (match, formula) => {
    const placeholder = `LATEX_INLINE_${counter}`;
    formulas.push({ placeholder, formula: match });
    counter++;
    return placeholder;
  });
  
  return { processedContent, formulas };
}

// 恢复LaTeX公式
function restoreLatexFormulas(html, formulas) {
  let restoredHtml = html;
  
  formulas.forEach(({ placeholder, formula }) => {
    // 对于块级公式，确保它们在自己的div中
    if (placeholder.includes('BLOCK')) {
      // 处理被p标签包裹的情况
      restoredHtml = restoredHtml.replace(
        new RegExp(`<p>\\s*${placeholder}\\s*<\/p>`, 'g'),
        `<div class="math math-display">${formula}</div>`
      );
      // 处理可能没有被p标签包裹的情况
      restoredHtml = restoredHtml.replace(
        new RegExp(`\\b${placeholder}\\b`, 'g'),
        `<div class="math math-display">${formula}</div>`
      );
    } else {
      // 对于行内公式，简单替换回原始公式
      restoredHtml = restoredHtml.replace(
        new RegExp(`\\b${placeholder}\\b`, 'g'),
        `<span class="math math-inline">${formula}</span>`
      );
    }
  });
  
  return restoredHtml;
}

export async function getPostData(slug) {
  const fullPath = path.join(postsDirectory, `${slug}.md`);
  const fileContents = fs.readFileSync(fullPath, 'utf8');

  // 使用gray-matter解析文章元数据
  const matterResult = matter(fileContents);

  try {
    // 预处理LaTeX公式，防止它们被Markdown解析器修改
    const { processedContent, formulas } = processLatexFormulas(matterResult.content);
    
    // 使用remark将markdown转换为HTML字符串
    const processedResult = await remark()
      .use(gfm)  // 使用GitHub风格Markdown支持，包括表格
      .use(html, { 
        sanitize: false,  // 不进行严格的HTML过滤
      })
      .process(processedContent);
      
    // 获取处理后的HTML
    let contentHtml = processedResult.toString();
    
    // 恢复LaTeX公式
    contentHtml = restoreLatexFormulas(contentHtml, formulas);

    // 将数据与id和contentHtml组合
    return {
      slug,
      contentHtml,
      ...matterResult.data,
    };
  } catch (error) {
    console.error('处理Markdown时出错:', error);
    // 返回原始内容，避免完全失败
    return {
      slug,
      contentHtml: `<pre>${matterResult.content}</pre>`,
      ...matterResult.data,
    };
  }
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

// LeetCode相关函数
const leetcodeDirectory = path.join(process.cwd(), 'content', 'leetcode');

export function getSortedLeetCodeData() {
  // 检查leetcode目录是否存在
  if (!fs.existsSync(leetcodeDirectory)) {
    return [];
  }

  // 获取leetcode目录下的所有文件名
  const fileNames = fs.readdirSync(leetcodeDirectory);
  const allLeetCodeData = fileNames.map((fileName) => {
    // 移除文件名中的".md"以获取id
    const slug = fileName.replace(/\.md$/, '');

    // 将markdown文件读取为字符串
    const fullPath = path.join(leetcodeDirectory, fileName);
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

    // 处理date字段，确保为字符串
    let date = matterResult.data.date;
    if (date instanceof Date) {
      date = date.toISOString().slice(0, 10);
    } else if (typeof date === 'number') {
      date = new Date(date).toISOString().slice(0, 10);
    } else if (typeof date !== 'string') {
      date = '';
    }

    // 将数据与id组合
    return {
      slug,
      excerpt,
      ...matterResult.data,
      date,
    };
  });

  // 按日期排序
  return allLeetCodeData.sort((a, b) => {
    if (a.date < b.date) {
      return 1;
    } else {
      return -1;
    }
  });
}

export function getAllLeetCodeSlugs() {
  if (!fs.existsSync(leetcodeDirectory)) {
    return [];
  }

  const fileNames = fs.readdirSync(leetcodeDirectory);
  return fileNames.map((fileName) => {
    return {
      params: {
        slug: fileName.replace(/\.md$/, ''),
      },
    };
  });
}

export async function getLeetCodeData(slug) {
  const fullPath = path.join(leetcodeDirectory, `${slug}.md`);
  const fileContents = fs.readFileSync(fullPath, 'utf8');
  // 使用gray-matter解析文章元数据
  const matterResult = matter(fileContents);

  try {
    // 预处理LaTeX公式，防止它们被Markdown解析器修改
    const { processedContent, formulas } = processLatexFormulas(matterResult.content);
    
    // 使用remark将markdown转换为HTML字符串
    const processedResult = await remark()
      .use(gfm)  // 使用GitHub风格Markdown支持，包括表格
      .use(html, { 
        sanitize: false,  // 不进行严格的HTML过滤
      })
      .process(processedContent);
      
    // 获取处理后的HTML
    let contentHtml = processedResult.toString();
    
    // 恢复LaTeX公式
    contentHtml = restoreLatexFormulas(contentHtml, formulas);

    // 将数据与id和contentHtml组合
    return {
      slug,
      contentHtml,
      ...matterResult.data,
    };
  } catch (error) {
    console.error('处理Markdown时出错:', error);
    // 返回原始内容，避免完全失败
    return {
      slug,
      contentHtml: `<pre>${matterResult.content}</pre>`,
      ...matterResult.data,
    };
  }
}

export function getAllLeetCodeTags() {
  const leetcodeProblems = getSortedLeetCodeData();
  const tags = {};

  leetcodeProblems.forEach((problem) => {
    if (problem.tags && Array.isArray(problem.tags)) {
      problem.tags.forEach((tag) => {
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

export function getLeetCodeByTag(tag) {
  const leetcodeProblems = getSortedLeetCodeData();
  return leetcodeProblems.filter((problem) => problem.tags && problem.tags.includes(tag));
} 