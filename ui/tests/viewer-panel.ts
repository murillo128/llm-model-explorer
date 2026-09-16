import { expect } from '@playwright/test';
import type { Locator } from '@playwright/test';

/** Shared geometry oracle for populated, empty and unavailable scientific cards. */
export async function integratedCard(panel: Locator) {
  await expect(panel).toHaveCount(1);
  const composition = await panel.evaluate(card => {
    const title = card.querySelector('.matrix-explorer-header')!;
    const header = title.querySelector('.matrix-panel-header')!;
    const body = card.querySelector('.viewer-panel-body')!;
    const bounds = card.getBoundingClientRect(), row = header.getBoundingClientRect();
    const content = body.getBoundingClientRect();
    const outerStyle = getComputedStyle(card), headerStyle = getComputedStyle(header), bodyStyle = getComputedStyle(body);
    const borders = (style: CSSStyleDeclaration) => [style.borderTopWidth, style.borderRightWidth, style.borderBottomWidth, style.borderLeftWidth];
    const radii = (style: CSSStyleDeclaration) => [style.borderTopLeftRadius, style.borderTopRightRadius, style.borderBottomRightRadius, style.borderBottomLeftRadius];
    const matrix = body.querySelector('.matrix-surfaces')?.getBoundingClientRect();
    return {
      ownsRegions: title.parentElement === card && body.parentElement === card && title.nextElementSibling === body,
      card: { border: borders(outerStyle), radius: radii(outerStyle), shadow: outerStyle.boxShadow, background: outerStyle.backgroundColor,
        overflow: [outerStyle.overflowX, outerStyle.overflowY] },
      header: { border: borders(headerStyle), radius: radii(headerStyle), background: headerStyle.backgroundColor,
        height: row.height, left: row.left - bounds.left, top: row.top - bounds.top, right: bounds.right - row.right },
      body: { border: borders(bodyStyle), radius: radii(bodyStyle), shadow: bodyStyle.boxShadow,
        left: content.left - bounds.left, right: bounds.right - content.right, bottom: bounds.bottom - content.bottom,
        gap: content.top - row.bottom,
        padding: [bodyStyle.paddingTop, bodyStyle.paddingRight, bodyStyle.paddingBottom, bodyStyle.paddingLeft] },
      padding: matchMedia('(max-width: 760px)').matches ? 8 : 12,
      matrixInset: matrix ? [matrix.left - content.left, matrix.top - content.top] : null,
    };
  });
  expect(composition.ownsRegions).toBe(true);
  expect(composition.card.border).toEqual(['1px', '1px', '1px', '1px']);
  expect(composition.card.radius).toEqual(['9px', '9px', '9px', '9px']);
  expect(composition.card.shadow).not.toBe('none');
  expect(composition.card.background).not.toBe('rgba(0, 0, 0, 0)');
  expect(composition.card.overflow).toEqual(['visible', 'visible']);
  expect(composition.header).toEqual({
    border: ['0px', '0px', '1px', '0px'], radius: ['8px', '8px', '0px', '0px'],
    background: 'rgb(251, 250, 247)', height: 40, left: 1, top: 1, right: 1,
  });
  expect(composition.body).toEqual({
    border: ['0px', '0px', '0px', '0px'], radius: ['0px', '0px', '0px', '0px'], shadow: 'none',
    left: 1, right: 1, bottom: 1, gap: 0, padding: Array<string>(4).fill(`${composition.padding}px`),
  });
  if (composition.matrixInset) expect(composition.matrixInset).toEqual([composition.padding, composition.padding]);
}

/** Visibility alone does not prove an overflowing popover actually paints. */
export async function unclippedPopover(dialog: Locator) {
  await expect(dialog).toBeVisible();
  expect(await dialog.evaluate(node => {
    const rect = node.getBoundingClientRect();
    return [rect.left + 8, rect.right - 8].every(x =>
      [rect.top + 8, rect.bottom - 8].every(y => node.contains(document.elementFromPoint(x, y))));
  })).toBe(true);
}
