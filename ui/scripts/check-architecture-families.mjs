/** Registered producers cross the actual default projection/action boundary. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { registerHooks } from 'node:module';

// Node 24 strips TypeScript. Resolve this frontend's extensionless local imports
// without npm, a browser, transpilation, or substituted production functions.
const hooks = registerHooks({ resolve(specifier, context, nextResolve) {
  return nextResolve(/^\.{1,2}\//.test(specifier) && !/\.[a-z]+$/.test(specifier)
    ? `${specifier}.ts` : specifier, context);
} });
try {
  const { projectGraph } = await import('../src/architecture-explorer/projection.ts');
  const { GraphViews } = await import('../src/architecture-explorer/graph.ts');
  const { toggleComponent } = await import('../src/architecture-explorer/component-actions.ts');
  const { projectionOptions } = await import('../src/architecture-explorer/scope-navigation.ts');
  const { expandCompactGraph } = await import('../src/api/compact-architecture.ts');
  // Expected paths and counts are fixed by the fixtures/source, not opt-in discovery.
  for (const [family, generation, minimumIndexed, expectedRanges] of [
    ['llama', true, 1], ['qwen3', true, 1], ['qwen35', true, 2],
    ['deepseek', true, 1, [[1,26]]], ['glm', true, 1, [[1,46]]], ['kimi', true, 2],
    ['clm', false, 2], ['kev', false, 1], ['visual', false, 2], ['lora', true, 1],
    ['qwen-lora', true, 1], ['qwen-gptq-lora', true, 1], ['owned-linear', false, 0],
    ['owned-shared', false, 1], ['owned-generation', true, 0],
  ]) {
    const graph = expandCompactGraph(JSON.parse(readFileSync(`${process.argv[2]}/${family}.json`, 'utf8')));
    const view = new GraphViews().get(family, graph);
    assert.equal(graph.nodes.some((n) => n.operation === 'generation_prepare_inputs'), generation, family);
    // Open real containers while keeping every repetition closed; preserve default filters.
    const layerRoots = new Set(graph.repetitions.flatMap((r) => r.instances.map((i) => i.node_id)));
    const parents = new Map(graph.nodes.map((n) => [n.id, n.parent_id]));
    view.update({ expanded: graph.nodes.filter((n) => n.kind === 'group' && !layerRoots.has(n.id) &&
      !layerRoots.has(parents.get(n.id))).map((n) => n.id) });
    let projection = projectGraph(graph, projectionOptions(view));
    for (let pass = 0; pass < 3; pass++) {
      for (const node of projection.nodes.filter((n) => n.presentation === 'repetition' && !n.expanded))
        toggleComponent(view, graph, node, 2);
      projection = projectGraph(graph, projectionOptions(view));
    }
    const returns = projection.edges.filter((e) => e.relationship?.owner === 'repetition' && e.relationship.kind === 'return');
    assert.ok(returns.length >= minimumIndexed, `${family}: expected ${minimumIndexed} indexed ranges, got ${returns.length}`);
    if (['qwen35', 'kimi', 'kev'].includes(family)) assert.ok(graph.repetitions.some((r) => r.bodies?.length), `${family}: missing periodic adoption`);
    if (family === 'qwen35') {
      const rep = graph.repetitions.find((r) => r.instances.length === 24);
      assert.equal(rep.bodies.length, 1);
      assert.deepEqual([rep.bodies[0].width, rep.bodies[0].count], [4, 6]);
      assert.deepEqual(rep.bodies[0].ranges, [{ start: 0, count: 3 }, { start: 3, count: 1 }]);
      assert.equal(projection.nodes.filter((n) => n.label === 'Hybrid block[j]').length, 1);
      assert.equal(projection.nodes.filter((n) => n.label === 'Linear layer[4j+k]').length, 1);
      assert.deepEqual(returns.map((e) => e.paths.length), [5, 12]);
    }
    if (family === 'kimi') {
      const range = projection.nodes.find((n) => n.nestedRepetition && n.instances?.length === 12);
      assert.ok(range, 'Kimi: missing inner KDA ×2 range');
      const mask = projection.edges.find((e) => e.target.node_id === range.id && e.target.port_id === 'padding_mask');
      const layers = graph.repetitions.find((r) => r.bodies?.length).instances;
      const consumers = [1, 2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22];
      assert.deepEqual(mask.paths.map((p) => p.at(-1).target.node_id), consumers.map((index) => layers.find((i) => i.index === index).node_id));
      assert.deepEqual(mask.relationship.instances.map((i) => i.index), consumers);
    }
    const ranges = returns.map((e) => [e.relationship.instances[0].index, e.relationship.instances.at(-1).index]);
    if (expectedRanges) assert.deepEqual(ranges, expectedRanges, family);
    if (!minimumIndexed) assert.equal(returns.length, 0, family);
    if (generation) assert.ok(!projection.nodes.some((n) => n.presentation === 'model'), family);
    if (generation) assert.ok(projection.edges.some((e) => e.relationship?.owner === 'generation' && e.relationship.kind === 'return'), family);
    console.log(`PASS ${family}: generation=${generation}, indexed=${JSON.stringify(ranges)}`);
  }
} finally { hooks.deregister(); }
