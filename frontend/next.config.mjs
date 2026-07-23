// This URL is consumed by the Next.js server-side rewrite. Keep it server-only:
// a *.railway.internal address cannot be reached from a visitor's browser.
const backendUrl = process.env.BACKEND_URL || "http://127.0.0.1:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/api/:path*`
      }
    ];
  }
};

export default nextConfig;
