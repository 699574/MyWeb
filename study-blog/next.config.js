/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'export',
  // 根据环境变量来设置basePath，仅在生产环境下使用/MyWeb前缀
  basePath: process.env.NODE_ENV === 'production' ? '/MyWeb' : '',
  images: {
    unoptimized: true
  }
};

module.exports = nextConfig; 