/** @type {import('next').NextConfig} */
const nextConfig = {
  serverExternalPackages: ["pdf-parse", "pdfjs-dist"],
  experimental: {
    outputFileTracingIncludes: {
      '/api/**/*': ['./node_modules/pdf-parse/**/*']
    }
  }
};

export default nextConfig;
