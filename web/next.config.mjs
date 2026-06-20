/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Keep pdfjs-dist out of the webpack bundle for /api/upload so it resolves
  // its worker file from node_modules at runtime instead of a bundled chunk.
  experimental: {
    serverComponentsExternalPackages: ["pdfjs-dist"],
  },
};

export default nextConfig;
