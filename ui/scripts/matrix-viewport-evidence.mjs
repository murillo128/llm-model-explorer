// Run against each checkout's dev server with identical browser/render settings.
// MATRIX_EVIDENCE_URL=http://127.0.0.1:4317 node scripts/matrix-viewport-evidence.mjs before /tmp/evidence
import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';

const output = process.argv[3] ?? '/tmp/matrix-viewport-evidence';
const phase = process.argv[2] ?? 'after';
const baseURL = process.env.MATRIX_EVIDENCE_URL ?? 'http://127.0.0.1:4317';
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: process.env.MATRIX_EVIDENCE_HEADED !== '1',
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const measurements = [];
try {
  for (const [width, scale, dpr, shape, suffix] of [
    [1178, 1, 1, [6144, 1024], '1178'],
    [1440, 2, 1, [6144, 1024], '1440'],
    [1178, null, 2, [4, 128], '1178-short-dpr2'],
  ]) {
    const page = await browser.newPage({ viewport: { width, height: 900 }, deviceScaleFactor: dpr });
    await page.goto(`${baseURL}/tests/matrix-explorer.html`);
    await page.waitForSelector('.matrix-scroll');
    await page.addStyleTag({ content: '#workspace { width: 100%; height: 800px; }' });
    await page.evaluate(shape => {
      const f = window.matrixFixture;
      Object.assign(f.sources.tall.descriptor, { shape, numel: shape[0] * shape[1] });
      f.render('tall');
    }, shape);
    await page.waitForSelector('.row-distributions canvas');
    await page.evaluate(({ scale, shape }) => {
      const f = window.matrixFixture, u = f.subscriptions.at(-1), [rows, columns] = shape;
      const high = rows === 4 ? Math.fround(.124) : 1;
      const values = Float32Array.from({ length: rows * columns }, (_, i) => ((i * 17) % 257 - 128) / 128 * high);
      const rowCounts = new Uint32Array(rows * 100), columnCounts = new Uint32Array(columns * 100);
      for (let i = 0; i < values.length; i++) {
        const bin = Math.min(99, Math.floor((values[i] + high) / (2 * high) * 100));
        rowCounts[Math.floor(i / columns) * 100 + bin]++;
        columnCounts[bin * columns + i % columns]++;
      }
      u.values(values, 0); u.transfer({ anchors: [-high, high], slope: 12 });
      u.distribution('rows', rowCounts, 0); u.distribution('columns', columnCounts, 0);
      u.distributionDomain({ minimum: -high, maximum: high });
      if (scale !== null) f.viewports.at(-1).zoomAt(scale, 0, 0);
    }, { scale, shape });
    await page.mouse.move(2, 2);
    await page.waitForTimeout(250);
    measurements.push(await page.evaluate(suffix => {
      const v = window.matrixFixture.viewports.at(-1);
      const rect = selector => document.querySelector(selector).getBoundingClientRect().toJSON();
      return { suffix, view: v.renderer.view, client: [v.host.clientWidth, v.host.clientHeight],
        gutter: [v.host.offsetWidth - v.host.clientWidth, v.host.offsetHeight - v.host.clientHeight],
        viewport: rect('.matrix-scroll'), matrix: rect('.matrix-scroll canvas'), rowTrack: rect('.row-distributions'), columnTrack: rect('.column-distributions'),
        rowData: rect('.row-distributions canvas'), columnData: rect('.column-distributions canvas'),
        low: rect('.distribution-scale-rows .distribution-low'), high: rect('.distribution-scale-rows .distribution-high') };
    }, suffix));
    await page.screenshot({ path: `${output}/${phase}-${suffix}.png` });
    await page.close();
  }
} finally { await browser.close(); }
await writeFile(`${output}/${phase}-geometry.json`, JSON.stringify({ browser: browser.version(), phase, measurements }, null, 2));
