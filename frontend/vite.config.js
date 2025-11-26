import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,      // This exposes the app to Docker
    strictPort: true,
    port: 5173,
    watch: {
      usePolling: true // This fixes "changes not showing up" on Macs
    }
  }
})