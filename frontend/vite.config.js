import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3007,
    proxy: {
      '/ocr': {
        target: 'http://localhost:8007',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://localhost:8007',
        changeOrigin: true,
      },
      '/models': {
        target: 'http://localhost:8007',
        changeOrigin: true,
      },
    },
  },
})

