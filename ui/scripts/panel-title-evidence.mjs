// PANEL_EVIDENCE_URL=http://127.0.0.1:4378 node scripts/panel-title-evidence.mjs before /tmp/panel-evidence
import { chromium } from 'playwright';
import { mkdir, writeFile } from 'node:fs/promises';

const phase = process.argv[2] ?? 'after';
const output = process.argv[3] ?? '/tmp/panel-title-evidence';
const baseURL = process.env.PANEL_EVIDENCE_URL ?? 'http://127.0.0.1:4378';
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const measurements = [];
try {
  for (const width of [1178, 1440]) {
    const page = await browser.newPage({ viewport: { width, height: 900 }, deviceScaleFactor: 1 });
    await page.goto(`${baseURL}/tests/tensor-explorer.html`);
    await page.waitForFunction(() => !!window.explorerFixture);
    await page.evaluate(() => {
      window.explorerFixture.tensors.find(t => t.id === 'reference').path =
        ['model', 'layers', '123', 'feed_forward', 'projection_with_a_long_logical_name', 'weight'];
    });
    await page.getByRole('combobox').selectOption('lab/alpha');
    const leaf = page.getByRole('button', { name: /^model.layers.123.feed_forward/, includeHidden: true });
    await leaf.evaluate(node => {
      for (let parent = node.parentElement; parent; parent = parent.parentElement) {
        if (parent instanceof HTMLDetailsElement) parent.open = true;
      }
    });
    await leaf.click();
    await page.waitForFunction(() => window.explorerFixture.requests.length === 3);
    await page.evaluate(() => {
      const f = window.explorerFixture, [rows, columns] = f.tensors.find(t => t.id === 'reference').shape;
      const values = Array.from({ length: rows * columns }, (_, i) => ((i * 17) % 257 - 128) / 64);
      const rowCounts = Array(rows * 100).fill(0), columnCounts = Array(columns * 100).fill(0);
      for (let i = 0; i < values.length; i++) {
        const bin = Math.min(99, Math.floor((values[i] + 2) / 4 * 100));
        rowCounts[Math.floor(i / columns) * 100 + bin]++;
        columnCounts[bin * columns + i % columns]++;
      }
      f.emit(0, 1, f.metadata(0)); f.data(0, values); f.end(0);
      f.emit(1, 1, f.metadata(1)); f.end(1);
      f.emit(2, 1, f.metadata(2)); f.data(2, [...rowCounts, ...columnCounts]); f.end(2);
    });
    await page.waitForFunction(() => !document.querySelector('[data-result]'));
    await page.getByRole('button', { name: 'Collapse inventory' }).click();
    await page.getByRole('button', { name: 'Expand inventory' }).hover();
    await page.getByRole('tooltip').waitFor({ state: 'visible' });
    measurements.push(await page.evaluate(width => {
      const rect = selector => document.querySelector(selector)?.getBoundingClientRect().toJSON();
      const tip = document.querySelector('[role="tooltip"]'), bounds = tip.getBoundingClientRect();
      return { width, header: rect('.matrix-panel-header'), body: rect('.viewer-panel-body'), matrix: rect('.matrix-surfaces'),
        tooltip: bounds.toJSON(), tooltipUncovered: [0.1, 0.5, 0.9].every(fraction =>
          tip.contains(document.elementFromPoint(bounds.left + bounds.width * fraction, bounds.top + bounds.height / 2))) };
    }, width));
    await page.screenshot({ path: `${output}/${phase}-${width}-tooltip.png` });
    await page.getByRole('button', { name: 'Tensor information', exact: true }).click();
    await page.getByRole('button', { name: 'Close tensor information' }).waitFor();
    await page.screenshot({ path: `${output}/${phase}-${width}-information.png` });
    await page.close();
  }
} finally { await browser.close(); }
await writeFile(`${output}/${phase}-geometry.json`, JSON.stringify({ browser: browser.version(), measurements }, null, 2));
