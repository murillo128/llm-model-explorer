import type { components } from './generated/types';

export type ApiError = components['schemas']['Error'];
export type FailureKind = 'http' | 'protocol' | 'backend' | 'cancelled' | 'transport';

/** The discriminant is stable; backend messages are never used for program logic. */
export class ApiFailure extends Error {
  constructor(
    readonly kind: FailureKind,
    message: string,
    readonly status?: number,
    readonly detail?: ApiError,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = 'ApiFailure';
  }
}

export function requireProtocol(condition: unknown, message: string): asserts condition {
  if (!condition) throw new ApiFailure('protocol', message);
}
