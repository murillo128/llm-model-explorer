import { expect, it } from 'vitest';
import { annotate, selectedTokenRows, utf16Boundaries } from './annotations';
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

it('selects exact sequence positions by strict overlap, never union annotations or repeated IDs', () => {
  const result: Tokenization = { text: 'Hi 中😀\nHi <s>', add_special_tokens: true, tokens: [
    token(0, undefined, undefined, true), token(1, 0, 2), token(2, 2, 3), token(3, 3, 4),
    token(4, 4, 5), token(5, 4, 5), token(6), token(7, 5, 6),
    { ...token(8, 6, 8), id: 101 }, token(9, 8, 9), token(10, 9, 12, true), token(11, 12, 12, true),
  ] };
  expect(selectedTokenRows(result, [{ from: 1, to: 2 }])).toEqual([1]);
  expect(selectedTokenRows(result, [{ from: 2, to: 1 }])).toEqual([1]);
  expect(selectedTokenRows(result, [{ from: 3, to: 7 }])).toEqual([3, 4, 5, 7]);
  expect(selectedTokenRows(result, [{ from: 5, to: 6 }])).toEqual([4, 5]); // Half a surrogate pair still intersects.
  expect(selectedTokenRows(result, [{ from: 2, to: 3 }, { from: 10, to: 12 }, { from: 2, to: 3 }])).toEqual([2, 10]);
  expect(selectedTokenRows(result, [{ from: 0, to: result.text.length }])).toEqual([1, 2, 3, 4, 5, 7, 8, 9, 10]);
  expect(selectedTokenRows(result, [{ from: 4, to: 4 }])).toEqual([]);
  expect(selectedTokenRows({ ...result, tokens: [token(0, -1, 1), token(1, 1, 99), token(2, 0.5, 2)] }, [{ from: 0, to: 99 }])).toEqual([]);
});

it('intersects each token span individually even when annotation unions overlap transitively', () => {
  const result: Tokenization = { text: 'abcde', add_special_tokens: false,
    tokens: [token(0, 0, 2), token(1, 1, 4), token(2, 3, 5)] };
  expect(selectedTokenRows(result, [{ from: 0, to: 1 }])).toEqual([0]);
  expect(selectedTokenRows(result, [{ from: 4, to: 5 }])).toEqual([2]);
});
