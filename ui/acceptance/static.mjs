// Production assets on an independent origin. No development server or API proxy.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { resolve, extname } from 'node:path';
const root = resolve('dist');
const port = Number(process.env.LMEX_STATIC_PORT ?? 4175);
const backend = `http://127.0.0.1:${Number(process.env.LMEX_BACKEND_PORT ?? 8765)}`;
createServer(async (request, response) => {
  const path = new URL(request.url, 'http://localhost').pathname;
  if (path === '/runtime-config.json') {
    response.setHeader('Content-Type', 'application/json');
    response.end(JSON.stringify({ backend_base_url: backend }));
    return;
  }
  const file = resolve(root, `.${path === '/' ? '/index.html' : path}`);
  if (!file.startsWith(root + '/')) { response.writeHead(403).end(); return; }
  try {
    const bytes = await readFile(file);
    response.setHeader('Content-Type', { '.js': 'text/javascript', '.css': 'text/css', '.html': 'text/html' }[extname(file)] ?? 'application/octet-stream');
    response.end(bytes);
  } catch { response.writeHead(404).end(); }
}).listen(port, '127.0.0.1');
