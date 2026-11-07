import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:5000",
      "/socket.io": {
        target: "ws://127.0.0.1:5000",
        ws: true,
      },
    },
  },
  test: {
    environment: "jsdom",
  },
});
