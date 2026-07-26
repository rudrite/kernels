import { defineConfig } from 'astro/config'

export default defineConfig({
  site: 'https://kernels.rudrite.com',
  trailingSlash: 'never',
  build: { format: 'file' },
})
