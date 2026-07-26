import { defineConfig } from 'astro/config'
import react from '@astrojs/react'

export default defineConfig({
  site: 'https://kernels.rudrite.com',
  trailingSlash: 'never',
  build: { format: 'file' },
  integrations: [react()],
})
