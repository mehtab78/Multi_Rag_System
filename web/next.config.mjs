/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Keep pdfjs-dist out of the webpack bundle for /api/upload so it resolves
  // its worker file from node_modules at runtime instead of a bundled chunk.
  experimental: {
    serverComponentsExternalPackages: ["pdfjs-dist"],
    // pdfjs-dist requires its worker file dynamically (not a static import),
    // so Next's serverless file tracer misses it unless told explicitly —
    // without this the Vercel function bundle lacks pdf.worker.mjs.
    outputFileTracingIncludes: {
      "/api/upload": ["./node_modules/pdfjs-dist/legacy/build/pdf.worker.mjs"],
    },
  },
};

export default nextConfig;
