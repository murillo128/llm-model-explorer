import { useEffect, useMemo, useState } from 'react';
import type { KeyboardEvent } from 'react';
import type { TensorDescriptor } from '../app/session-controller';
import { readInventoryPreference, writeInventoryPreference } from './inventory-preferences';

const expansionKey = 'lmex.tensor-inventory.branches';
function initialExpansion(): Record<string, boolean> {
  const saved = readInventoryPreference(expansionKey);
  return saved && typeof saved === 'object' && !Array.isArray(saved)
    ? Object.fromEntries(Object.entries(saved).filter(([, value]) => typeof value === 'boolean')) : {};
}

interface Branch { children: Map<string, Branch>; tensors: TensorDescriptor[] }
function buildTree(tensors: TensorDescriptor[]) {
  const root: Branch = { children: new Map(), tensors: [] };
  for (const tensor of tensors) {
    let branch = root;
    // The last path segment is the leaf. Do not split names or infer architecture.
    for (const segment of tensor.path.slice(0, -1)) {
      let child = branch.children.get(segment);
      if (!child) { child = { children: new Map(), tensors: [] }; branch.children.set(segment, child); }
      branch = child;
    }
    branch.tensors.push(tensor);
  }
  return root;
}

export function TensorTree({ tensors, selectedId, onSelect }: {
  tensors: TensorDescriptor[]; selectedId: string | undefined; onSelect: (tensor: TensorDescriptor) => void;
}) {
  const tree = useMemo(() => buildTree(tensors), [tensors]);
  const [expanded, setExpanded] = useState(initialExpansion);
  useEffect(() => writeInventoryPreference(expansionKey, expanded), [expanded]);
  function toggle(details: HTMLDetailsElement) {
    const key = details.dataset.branchKey!;
    const open = !details.open;
    setExpanded((old) => ({ ...old, [key]: open }));
  }
  function navigate(event: KeyboardEvent<HTMLDivElement>) {
    const target = event.target as HTMLElement;
    if (!target.matches('summary, button')) return;
    const rows = [...event.currentTarget.querySelectorAll<HTMLElement>('summary, button')].filter((row) => {
      let parent = row.parentElement;
      while (parent && parent !== event.currentTarget) {
        if (parent instanceof HTMLDetailsElement && !parent.open && parent.firstElementChild !== row) return false;
        parent = parent.parentElement;
      }
      return true;
    });
    const index = rows.indexOf(target);
    const details = target.matches('summary') ? target.parentElement as HTMLDetailsElement : null;
    let next: HTMLElement | undefined;
    switch (event.key) {
      case 'ArrowDown': next = rows[index + 1]; break;
      case 'ArrowUp': next = rows[index - 1]; break;
      case 'Home': next = rows[0]; break;
      case 'End': next = rows.at(-1); break;
      case 'ArrowRight':
        if (details && !details.open) toggle(details);
        else if (details) next = rows[index + 1];
        break;
      case 'ArrowLeft':
        if (details?.open) toggle(details);
        else next = (details?.parentElement ?? target.parentElement)?.closest('details')?.querySelector('summary') ?? undefined;
        break;
      case 'Enter':
      case ' ':
        if (!details) return;
        toggle(details);
        break;
      default: return;
    }
    event.preventDefault();
    next?.focus();
  }
  function renderBranch(branch: Branch, path: string[] = []) {
    const depth = path.length;
    const paddingInlineStart = 4 + Math.min(depth, 7) * 8;
    return <ul className="tensor-tree">
      {[...branch.children].map(([segment, child]) => {
        const childPath = [...path, segment];
        const key = JSON.stringify(childPath);
        const open = expanded[key] ?? depth === 0;
        return <li key={`branch:${segment}`}>
          <details open={open} data-branch-key={key}><summary style={{ paddingInlineStart }} title={childPath.join(' › ')}
            onClick={(event) => { event.preventDefault(); toggle(event.currentTarget.parentElement as HTMLDetailsElement); }}>
            <span>{segment || '(unnamed segment)'}</span>
          </summary>{renderBranch(child, childPath)}</details>
        </li>;
      })}
      {branch.tensors.map((tensor) => <li key={`tensor:${tensor.id}`}>
        <button className="tensor-choice" type="button" aria-pressed={selectedId === tensor.id}
          style={{ paddingInlineStart }} title={tensor.path.join(' › ') || tensor.name}
          aria-label={`${tensor.path.join('.') || tensor.name} [${tensor.shape.join(', ')}] · ${tensor.storage_dtype}${tensor.rank !== 1 && tensor.rank !== 2 ? ` · Rank ${tensor.rank}, descriptor only` : ''}`}
          onClick={() => onSelect(tensor)}>
          <svg className="tensor-leaf-icon" aria-hidden="true" viewBox="0 0 16 16"><rect x="2" y="2" width="12" height="12" rx="1" /><path d="M6 2v12M10 2v12M2 6h12M2 10h12" /></svg>
          <span className="tensor-leaf-name">{tensor.path.at(-1) || tensor.name}</span>
          <span className="tensor-leaf-metadata metadata">[{tensor.shape.join(' × ')}] · {tensor.storage_dtype}</span>
        </button>
      </li>)}
    </ul>;
  }
  return <div className="tensor-navigator" onKeyDown={navigate}>{renderBranch(tree)}</div>;
}
