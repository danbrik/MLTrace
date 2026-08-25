import type { RedundancyResult } from '../types';

export const INCOMPLETE_CORRELATION_MESSAGE = 'Excluded from clustering because the correlation matrix is incomplete.';

export function correlationMatrixView(
  result: Pick<RedundancyResult, 'variables' | 'leaf_order' | 'spearman' | 'common_n'>,
  order: 'original' | 'clustered',
  values: 'signed' | 'absolute',
) {
  const names = order === 'clustered' ? result.leaf_order : result.variables;
  const indexes = names.map((name) => result.variables.indexOf(name));
  return {
    names,
    values: indexes.map((row) => indexes.map((column) => {
      const value = result.spearman[row][column];
      return value == null ? null : values === 'absolute' ? Math.abs(value) : value;
    })),
    commonN: indexes.map((row) => indexes.map((column) => result.common_n[row][column])),
  };
}

export function filteredPairRows<T extends { absolute_rho: number }>(pairs: T[], threshold: number): T[] {
  return pairs.filter((item) => item.absolute_rho >= threshold);
}
