import { expect, it } from 'vitest';
import { annotate, utf16Boundaries } from './annotations';
import type { Token, Tokenization } from './annotations';

const token = (index: number, start?: number, end?: number, special = false): Token => ({ index, id: 100 + index, token: `native-${index}`, decoded: '�', special, ...(start === undefined ? {} : { start, end: end! }) });
it('maps code points without normalizing accents, combining marks or surrogate pairs', () => {
  expect(utf16Boundaries(' A😀e\u0301中\t\n')).toEqual([0, 1, 2, 4, 5, 6, 7, 8, 9]);
});
it('unions transitively overlapping source spans, keeping all IDs in sequence order', () => {
  const result: Tokenization = { text: 'A😀e\u0301<special>', add_special_tokens: true,
    tokens: [token(0, undefined, undefined, true), token(1, 0, 1), token(2, 1, 2), token(3, 1, 3), token(4, 2, 4), token(5, 4, 13, true), token(6, undefined, undefined, true)] };
  const { groups, unmapped } = annotate(result);
  expect(groups.map(({ start, end, tokens }) => [result.text.slice(start, end), tokens.map((token) => token.id)])).toEqual([
    ['A', [101]], ['😀e\u0301', [102, 103, 104]], ['<special>', [105]],
  ]);
  expect(unmapped.map(({ anchor, token }) => [anchor, token.id])).toEqual([[0, 100], [14, 106]]);
});
it('never fabricates spans for missing or empty offsets, even if decoded text matches input', () => {
  const { groups, unmapped } = annotate({ text: 'abc', add_special_tokens: false, tokens: [token(0), token(1, 0, 0)] });
  expect(groups).toEqual([]); expect(unmapped.map(({ anchor }) => anchor)).toEqual([0, 0]);
});
