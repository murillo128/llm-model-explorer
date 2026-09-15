import Ajv2020 from 'ajv/dist/2020';
import schemas from './generated/schemas.json';
import type { components } from './generated/types';
import { requireProtocol } from './errors';

type Schemas = components['schemas'];
export type Metadata = Schemas['StreamMetadata'];
export type Progress = Schemas['StreamProgress'];
export const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const ajv = new Ajv2020({ strict: false, strictNumbers: true });
ajv.addFormat('uuid', uuidPattern);
ajv.addSchema(schemas, 'contract');

export function safeSize(value: number): number {
  requireProtocol(Number.isSafeInteger(value) && value >= 0, 'Unsafe size or count');
  return value;
}

export function product(shape: number[]): number {
  // A zero dimension makes the mathematical product zero, regardless of ordering.
  shape.forEach(safeSize);
  return shape.includes(0) ? 0 : shape.reduce((a, b) => safeSize(a * b), 1);
}

function crossFields(name: string, value: unknown): void {
  if (name === 'StreamMetadata') {
    crossFields(({ tensor: 'TensorMetadata', input_embeddings: 'InputEmbeddingsMetadata', input_embeddings_statistics: 'InputEmbeddingsStatisticsMetadata', input_embeddings_distributions: 'InputEmbeddingsDistributionsMetadata', tensor_statistics: 'TensorStatisticsMetadata', tensor_distributions: 'TensorDistributionsMetadata' })[(value as Metadata).kind], value);
  } else if (name === 'TensorMetadata' || name === 'InputEmbeddingsMetadata') {
    const m = value as Schemas['TensorMetadata'] | Schemas['InputEmbeddingsMetadata'];
    requireProtocol(m.byte_length === safeSize(product(m.shape) * 4), 'Tensor byte length differs from shape');
    if (m.kind === 'input_embeddings') {
      requireProtocol(m.shape[0] === m.token_ids.length, 'Embedding row count differs from token IDs');
    }
  } else if (name === 'TensorDescriptor') {
    const m = value as Schemas['TensorDescriptor'];
    requireProtocol(m.rank === m.shape.length && m.numel === product(m.shape), 'Invalid tensor descriptor dimensions');
  } else if (name === 'TensorInventory') {
    for (const tensor of (value as Schemas['TensorInventory']).tensors) crossFields('TensorDescriptor', tensor);
  } else if (name === 'TensorStatisticsMetadata' || name === 'InputEmbeddingsStatisticsMetadata') {
    const m = value as Schemas['TensorStatisticsMetadata'] | Schemas['InputEmbeddingsStatisticsMetadata'];
    if (m.kind === 'input_embeddings_statistics') {
      requireProtocol(m.shape[0] === m.token_ids.length && m.count === product(m.shape), 'Invalid embedding statistics geometry');
    }
    requireProtocol(m.count === safeSize(m.finite_count + m.non_finite_count), 'Invalid statistics counts');
    if (m.finite_count > 0) {
      const ordered = [m.minimum, m.percentiles.p01, m.percentiles.p05, m.percentiles.p50, m.percentiles.p95, m.percentiles.p99, m.maximum] as number[];
      requireProtocol(ordered.every((v, i) => i === 0 || v >= ordered[i - 1]!), 'Invalid statistics range or percentile ordering');
      requireProtocol(m.mean! >= m.minimum! && m.mean! <= m.maximum!, 'Invalid statistics mean');
    }
  } else if (name === 'DistributionSection') {
    const m = value as Schemas['DistributionSection'];
    requireProtocol(m.byte_length === safeSize(product(m.shape) * 4), 'Invalid distribution section length');
    safeSize(m.offset + m.byte_length);
  } else if (name === 'TensorDistributionsMetadata' || name === 'InputEmbeddingsDistributionsMetadata') {
    const m = value as Schemas['TensorDistributionsMetadata'] | Schemas['InputEmbeddingsDistributionsMetadata'];
    if (m.kind === 'input_embeddings_distributions') {
      requireProtocol(m.rows === m.token_ids.length, 'Invalid embedding distribution row count');
    }
    const [row, column] = m.sections;
    requireProtocol(row && column, 'Missing distribution sections');
    for (const section of m.sections) crossFields('DistributionSection', section);
    requireProtocol(row.shape[0] === m.rows && row.shape[1] === 100 && column.shape[0] === 100 && column.shape[1] === m.columns && row.offset === 0 && column.offset === row.byte_length && m.byte_length === safeSize(row.byte_length + column.byte_length), 'Invalid distribution layout');
    product([m.rows, m.columns]);
    requireProtocol(m.domain_minimum === null ? m.domain_maximum === null : m.domain_maximum !== null && m.domain_minimum <= m.domain_maximum, 'Invalid distribution domain');
    requireProtocol(m.rows !== 0 && m.columns !== 0 || m.domain_minimum === null, 'Empty source requires a null domain');
  } else if (name === 'StreamProgress') {
    const m = value as Progress;
    requireProtocol(m.total === undefined || m.completed <= m.total, 'Progress exceeds total');
  } else if (name === 'Token') {
    const m = value as Schemas['Token'];
    requireProtocol(m.start === undefined || m.end !== undefined && m.start <= m.end, 'Invalid token span');
  } else if (name === 'TokenizeResponse') {
    const m = value as Schemas['TokenizeResponse'];
    const length = Array.from(m.text).length;
    m.tokens.forEach((token, index) => {
      crossFields('Token', token);
      requireProtocol(token.index === index && (token.end === undefined || token.end <= length), 'Invalid token index or source span');
    });
  }
}

export function validateSchema<K extends keyof Schemas>(name: K, value: unknown): Schemas[K] {
  const validate = ajv.getSchema(`contract#/components/schemas/${name}`)!;
  requireProtocol(validate(value), `Invalid ${name} schema`);
  crossFields(name, value);
  return value as Schemas[K];
}

export function validateResponse(name: keyof typeof schemas.$defs, value: unknown): void {
  requireProtocol(ajv.getSchema(`contract#/$defs/${name}`)!(value), `Invalid ${name} response`);
  if (name === 'listTensors') crossFields('TensorInventory', value);
  if (name === 'tokenize') crossFields('TokenizeResponse', value);
}
