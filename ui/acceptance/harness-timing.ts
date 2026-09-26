import type { TestInfo } from '@playwright/test';

/** One compact attachment per test; independent of application stream timestamps. */
export class HarnessTiming {
  fixtureGenerationMs: number | null = null;
  spawned = 0;
  ready = 0;
  bodyStarted = 0;
  teardownStarted = 0;

  async attach(info: TestInfo, backendLog: string, probe: unknown) {
    const fixtureLine = backendLog.split('\n').find(line => line.startsWith('LMEX_HARNESS '));
    const fixture = fixtureLine ? JSON.parse(fixtureLine.slice('LMEX_HARNESS '.length)) as { fixtureGenerationMs: number } : null;
    await info.attach('harness-timing', { contentType: 'application/json', body: JSON.stringify({
      fixtureGenerationMs: this.fixtureGenerationMs ?? fixture?.fixtureGenerationMs ?? null,
      backendSpawnToReadyMs: this.ready && this.spawned ? this.ready - this.spawned : null,
      browserSetupMs: this.bodyStarted && this.ready ? this.bodyStarted - this.ready : null,
      testBodyMs: this.teardownStarted && this.bodyStarted ? this.teardownStarted - this.bodyStarted : null,
      teardownMs: this.teardownStarted ? performance.now() - this.teardownStarted : null,
      probe,
    }) });
  }
}
