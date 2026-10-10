// @vitest-environment node
import { expectTypeOf, it } from 'vitest';
import type { components } from './generated/types';

type ModelState = components['schemas']['ModelState'];

// Native `npm run typecheck` checks these consumer contracts as well as Vitest.
it('accepts both wire states and narrows revision nullability by status', () => {
  const states: ModelState[] = [
    {
      epoch: '01234567-89ab-cdef-0123-456789abcdef',
      sequence: 0,
      model_id: 'org/model',
      status: 'present',
      model_revision: 'snapshot_1',
    },
    {
      epoch: '01234567-89ab-cdef-0123-456789abcdef',
      sequence: 1,
      model_id: 'org/model',
      status: 'unavailable',
      model_revision: null,
    },
  ];

  expectTypeOf<ModelState['status']>().toEqualTypeOf<'present' | 'unavailable'>();
  for (const state of states) {
    if (state.status === 'present') {
      expectTypeOf(state.model_revision).toEqualTypeOf<string>();
    } else {
      expectTypeOf(state.status).toEqualTypeOf<'unavailable'>();
      expectTypeOf(state.model_revision).toEqualTypeOf<null>();
    }
  }
});
