import { useMemo } from 'react';
import type { TensorDescriptor } from '../app/session-controller';

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
  function renderBranch(branch: Branch) {
    return <ul className="tensor-tree">
      {[...branch.children].map(([segment, child]) => <li key={`branch:${segment}`}>
        <details open><summary>{segment || '(unnamed segment)'}</summary>{renderBranch(child)}</details>
      </li>)}
      {branch.tensors.map((tensor) => <li key={`tensor:${tensor.id}`}>
        <button className="tensor-choice" type="button" aria-pressed={selectedId === tensor.id}
          onClick={() => onSelect(tensor)}>
          <span>{tensor.name}</span>
          <span className="metadata">[{tensor.shape.join(', ')}] · {tensor.storage_dtype}</span>
          {tensor.rank !== 1 && tensor.rank !== 2 && <span className="rank-note">Rank {tensor.rank} · descriptor only</span>}
        </button>
      </li>)}
    </ul>;
  }
  return renderBranch(tree);
}
