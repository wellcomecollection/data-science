/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};

module.exports = nextConfig && {
  images: {
    domains: ["iiif.wellcomecollection.org"],
    unoptimized: true
  },
};
