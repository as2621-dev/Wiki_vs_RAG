/** @type {import('next').NextConfig} */
// Static export: `next build` emits a fully static site under `out/` — no runtime backend.
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
};

export default nextConfig;
