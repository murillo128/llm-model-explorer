import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [react()],
  // Worker-only ELK imports escape the initial scan. Optimize both entries up
  // front so cold development pages do not reload during graph interaction.
  optimizeDeps: { include: ['elkjs/lib/elk-api.js', 'elkjs/lib/elk.bundled.js'] },
  build: {
    rollupOptions: { input: { app: 'index.html', renderer: 'renderer-demo.html' } },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    restoreMocks: true,
  },
});
