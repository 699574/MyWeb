/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'export',
  // 如果你的GitHub Pages仓库不是在根目录，需要设置basePath
  // basePath: '/study-blog',
  images: {
    unoptimized: true
  }
};

module.exports = nextConfig; 