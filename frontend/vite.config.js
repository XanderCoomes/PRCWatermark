import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
export default {
    server: {
      proxy: {
        '/api': {
          target: 'http://localhost:8000', // backend port
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ''),
        },
      },
    },
  }
  